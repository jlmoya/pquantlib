"""Bond — abstract base class for bond instruments.

# C++ parity: ql/instruments/bond.hpp + ql/instruments/bond.cpp (v1.42.1).

A Bond owns a sorted ``Leg`` of coupon cashflows + redemption flows. The
default discount-based valuation (clean / dirty / accrued / yield) is
delegated to ``CashFlows`` aggregator functions plus a pricing engine
that fills ``settlement_value``.

Two construction styles match the C++ API:

- ``Bond(settlement_days, calendar, issue_date, coupons)`` — the modern
  amortising-bond constructor. Pass coupons WITHOUT redemptions; the
  base class builds and appends them via ``_add_redemptions_to_cashflows``.
- ``Bond.with_face_amount(settlement_days, calendar, face_amount,
  maturity_date, issue_date, cashflows)`` — legacy non-amortising
  constructor where the *last* cashflow in ``cashflows`` is the
  redemption (mirrors the C++ second constructor at bond.cpp:63-102).

Subclass responsibilities mirror C++: derived bonds populate
``self._cashflows`` (and optionally ``self._maturity_date``) in their
own ``__init__`` before delegating to the base, then call either
``_add_redemptions_to_cashflows`` (variable-notional case) or
``_set_single_redemption`` (single-redemption case) to fill the
notional schedule + redemption flows.

# C++ parity divergence — Settings.evaluationDate:
# The C++ ``Bond`` registers itself as observer of
# ``Settings::instance().evaluationDate()``. In pquantlib we register
# with the ``ObservableSettings`` singleton instead (same observability
# semantics; the settings field is ``evaluation_date``). Callers wanting
# the C++ "today" default for settlement_date must set
# ``ObservableSettings().evaluation_date`` explicitly.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum

from pquantlib import qassert
from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.cash_flows import CashFlows
from pquantlib.cashflows.coupon import Coupon
from pquantlib.cashflows.simple_cash_flow import AmortizingPayment, Redemption
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.instruments.instrument import Instrument, InstrumentResults
from pquantlib.interest_rate import InterestRate
from pquantlib.math.solvers1d.brent import Brent
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.pricing_engine import (
    PricingEngineArguments,
    PricingEngineResults,
)
from pquantlib.time.calendar import Calendar
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.time_unit import TimeUnit

_NULL_DATE: Date = Date()


class BondPriceType(IntEnum):
    """C++ parity: nested enum ``Bond::Price::Type`` (bond.hpp:64)."""

    Dirty = 0
    Clean = 1


class BondPrice:
    """Bond price + type carrier.

    # C++ parity: ``Bond::Price`` nested class (bond.hpp:62-76). The C++
    # class uses ``Null<Real>()`` for "no amount given"; the Python port
    # uses ``None`` and exposes ``is_valid`` accordingly.
    """

    def __init__(
        self,
        amount: float | None = None,
        price_type: BondPriceType = BondPriceType.Clean,
    ) -> None:
        self._amount: float | None = amount
        self._type: BondPriceType = price_type

    def amount(self) -> float:
        qassert.require(self._amount is not None, "no amount given")
        assert self._amount is not None
        return self._amount

    def type(self) -> BondPriceType:
        return self._type

    def is_valid(self) -> bool:
        return self._amount is not None


class BondArguments(PricingEngineArguments):
    """Engine-arguments carrier for Bond.

    # C++ parity: ``Bond::arguments`` (bond.hpp:295-301).
    """

    def __init__(self) -> None:
        self.settlement_date: Date = _NULL_DATE
        self.cashflows: list[CashFlow] = []
        self.calendar: Calendar | None = None

    def validate(self) -> None:
        qassert.require(self.settlement_date != _NULL_DATE, "no settlement date provided")
        qassert.require(len(self.cashflows) > 0, "no cash flow provided")
        # The C++ ``Bond::arguments::validate`` (bond.cpp:399-404) also
        # checks ``cf != nullptr``. Python's typed list of CashFlow cannot
        # contain a null sentinel (None is statically excluded), so we
        # skip the per-element None check — pyright would flag it as
        # ``reportUnnecessaryComparison``.


class BondResults(InstrumentResults):
    """Engine-results carrier for Bond.

    # C++ parity: ``Bond::results`` (bond.hpp:303-310).
    """

    def __init__(self) -> None:
        super().__init__()
        self.settlement_value: float | None = None

    def reset(self) -> None:
        super().reset()
        self.settlement_value = None


# Default IRR / yield-solver settings (mirror C++ Bond::yield defaults).
_DEFAULT_YIELD_ACCURACY: float = 1.0e-8
_DEFAULT_YIELD_MAX_ITER: int = 100
_DEFAULT_YIELD_GUESS: float = 0.05


class Bond(Instrument):
    """Abstract base bond.

    Concrete bonds populate ``self._cashflows`` and either
    ``self._maturity_date`` (legacy face-amount ctor) or rely on the
    coupon-derived ``calculate_notionals_from_cashflows`` path.

    # C++ parity divergence — subclassing:
    # the C++ ``Bond`` is *concrete* (provides both constructors) but
    # always overridden by subclasses (FixedRateBond / ZeroCouponBond /
    # etc.). The Python port keeps it instantiable via the
    # ``with_face_amount`` factory but otherwise treats it as a base
    # class — concrete bonds subclass and use the protected helpers.
    """

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        settlement_days: int,
        calendar: Calendar,
        issue_date: Date | None = None,
        coupons: Sequence[CashFlow] | None = None,
    ) -> None:
        """Modern (amortising) constructor.

        # C++ parity: bond.cpp:38-61. ``coupons`` must NOT contain
        # redemptions — these are appended by
        # ``_add_redemptions_to_cashflows``.
        """
        super().__init__()
        self._settlement_days: int = settlement_days
        self._calendar: Calendar = calendar
        self._issue_date: Date = issue_date if issue_date is not None else _NULL_DATE
        self._notional_schedule: list[Date] = []
        self._notionals: list[float] = []
        self._cashflows: list[CashFlow] = list(coupons) if coupons else []
        self._redemptions: list[CashFlow] = []
        self._maturity_date: Date = _NULL_DATE
        self._settlement_value: float | None = None

        if self._cashflows:
            self._cashflows.sort(key=lambda cf: cf.date())
            if self._issue_date != _NULL_DATE:
                qassert.require(
                    self._issue_date < self._cashflows[0].date(),
                    f"issue date ({self._issue_date}) must be earlier than first "
                    f"payment date ({self._cashflows[0].date()})",
                )
            self._maturity_date = self._cashflows[-1].date()
            self._add_redemptions_to_cashflows()

        ObservableSettings().register_with(self)
        for cf in self._cashflows:
            cf.register_with(self)

    @classmethod
    def with_face_amount(
        cls,
        settlement_days: int,
        calendar: Calendar,
        face_amount: float,
        maturity_date: Date,
        issue_date: Date | None = None,
        cashflows: Sequence[CashFlow] | None = None,
    ) -> Bond:
        """Legacy non-amortising bond constructor.

        # C++ parity: bond.cpp:63-102. The last cashflow in ``cashflows``
        # must be the redemption; preceding ones are sorted in-place.
        """
        # We're a concrete-enough base for the legacy constructor path —
        # build the bare instance via __new__ to bypass the abstract
        # ``is_expired`` check at the Instrument level (the factory's
        # callers always know what they're constructing).
        bond = cls.__new__(cls)
        Instrument.__init__(bond)
        bond._settlement_days = settlement_days
        bond._calendar = calendar
        bond._issue_date = issue_date if issue_date is not None else _NULL_DATE
        bond._notional_schedule = []
        bond._notionals = []
        bond._cashflows = list(cashflows) if cashflows else []
        bond._redemptions = []
        bond._maturity_date = maturity_date
        bond._settlement_value = None

        if bond._cashflows:
            # Sort everything except the redemption (last element).
            head = sorted(bond._cashflows[:-1], key=lambda cf: cf.date())
            bond._cashflows = [*head, bond._cashflows[-1]]

            if bond._maturity_date == _NULL_DATE:
                bond._maturity_date = CashFlows.maturity_date(bond._cashflows)

            if bond._issue_date != _NULL_DATE:
                qassert.require(
                    bond._issue_date < bond._cashflows[0].date(),
                    f"issue date ({bond._issue_date}) must be earlier than first "
                    f"payment date ({bond._cashflows[0].date()})",
                )

            bond._notionals = [face_amount, 0.0]
            bond._notional_schedule = [_NULL_DATE, bond._maturity_date]
            bond._redemptions.append(bond._cashflows[-1])

        ObservableSettings().register_with(bond)
        for cf in bond._cashflows:
            cf.register_with(bond)
        return bond

    # ------------------------------------------------------------------
    # Instrument interface
    # ------------------------------------------------------------------

    def is_expired(self) -> bool:
        """All cashflows already paid at evaluation date.

        # C++ parity: bond.cpp:104-111. C++ passes ``nullopt`` for
        # include_settlement_date_flows so CashFlows::isExpired falls
        # back to the Settings default.
        """
        eval_date = ObservableSettings().evaluation_date_or_today()
        return CashFlows.is_expired(self._cashflows, None, eval_date)

    def deep_update(self) -> None:
        """Cascade ``update`` to every cashflow.

        # C++ parity: bond.cpp:356-361. Python observers don't have a
        # separate ``deep_update`` hook on every Observable, so we emulate
        # by calling ``update`` on each cashflow that supports it.
        """
        for cf in self._cashflows:
            update = getattr(cf, "update", None)
            if callable(update):
                update()
        self.update()

    # ------------------------------------------------------------------
    # Inspectors
    # ------------------------------------------------------------------

    def settlement_days(self) -> int:
        return self._settlement_days

    def calendar(self) -> Calendar:
        return self._calendar

    def notionals(self) -> list[float]:
        return list(self._notionals)

    def notional(self, d: Date | None = None) -> float:
        """Notional outstanding at date ``d`` (defaults to settlement_date).

        # C++ parity: bond.cpp:113-139.
        """
        d_eff = d if d is not None and d != _NULL_DATE else self.settlement_date()
        if not self._notional_schedule:
            return 0.0
        if d_eff > self._notional_schedule[-1]:
            return 0.0
        # Lower-bound search starting at index 1 (mirrors C++).
        # Find the first schedule entry >= d_eff (skipping index 0,
        # which is the sentinel null Date).
        idx: int = 0
        for i in range(1, len(self._notional_schedule)):
            if self._notional_schedule[i] >= d_eff:
                idx = i
                break
        else:
            return 0.0
        if d_eff < self._notional_schedule[idx]:
            return self._notionals[idx - 1]
        # d_eff == schedule[idx] (redemption date): bond already paid.
        return self._notionals[idx]

    def cashflows(self) -> list[CashFlow]:
        return list(self._cashflows)

    def redemptions(self) -> list[CashFlow]:
        return list(self._redemptions)

    def redemption(self) -> CashFlow:
        qassert.require(
            len(self._redemptions) == 1, "multiple redemption cash flows given"
        )
        return self._redemptions[-1]

    def start_date(self) -> Date:
        # C++ parity: bond.cpp:147-149.
        return CashFlows.start_date(self._cashflows)

    def maturity_date(self) -> Date:
        # C++ parity: bond.cpp:151-156.
        if self._maturity_date != _NULL_DATE:
            return self._maturity_date
        return CashFlows.maturity_date(self._cashflows)

    def issue_date(self) -> Date:
        return self._issue_date

    def is_tradable(self, d: Date | None = None) -> bool:
        # C++ parity: bondfunctions.cpp:39-45. Tradable iff notional != 0.
        d_eff = d if d is not None and d != _NULL_DATE else self.settlement_date()
        return self.notional(d_eff) != 0.0

    def settlement_date(self, d: Date | None = None) -> Date:
        """Settlement date for ``d`` (defaults to evaluation_date).

        # C++ parity: bond.cpp:162-173.
        """
        d_eff = (
            d if d is not None and d != _NULL_DATE
            else ObservableSettings().evaluation_date_or_today()
        )
        settle = self._calendar.advance(d_eff, self._settlement_days, TimeUnit.Days)
        if self._issue_date == _NULL_DATE:
            return settle
        return max(settle, self._issue_date)

    # ------------------------------------------------------------------
    # Calculations
    # ------------------------------------------------------------------

    def clean_price(self) -> float:
        """Theoretical clean price from the engine settlement value.

        # C++ parity: bond.cpp:175-177.
        """
        return self.dirty_price() - self.accrued_amount(self.settlement_date())

    def dirty_price(self) -> float:
        """Theoretical dirty price.

        # C++ parity: bond.cpp:179-185.
        """
        current = self.notional(self.settlement_date())
        if current == 0.0:
            return 0.0
        return self.settlement_value() * 100.0 / current

    def settlement_value(self) -> float:
        """Engine-supplied settlement value.

        # C++ parity: bond.cpp:187-192. Triggers calculate() lazily.
        """
        self.calculate()
        qassert.require(self._settlement_value is not None, "settlement value not provided")
        assert self._settlement_value is not None
        return self._settlement_value

    def settlement_value_from_clean(self, clean_price: float) -> float:
        """Settlement value derived from an externally supplied clean price.

        # C++ parity: bond.cpp:194-197. The C++ overload-on-arg-type is
        # collapsed into an explicit second method (mypy-friendly).
        """
        dirty_price = clean_price + self.accrued_amount(self.settlement_date())
        return dirty_price / 100.0 * self.notional(self.settlement_date())

    def accrued_amount(self, d: Date | None = None) -> float:
        """Accrued amount in price-quote units (per 100 notional).

        # C++ parity: bond.cpp:256-262 + bondfunctions.cpp:224-235.
        """
        d_eff = d if d is not None and d != _NULL_DATE else self.settlement_date()
        if self.notional(d_eff) == 0.0:
            return 0.0
        if not self.is_tradable(d_eff):
            return 0.0
        return (
            CashFlows.accrued_amount(self._cashflows, False, d_eff)
            * 100.0
            / self.notional(d_eff)
        )

    def next_coupon_rate(self, d: Date | None = None) -> float:
        # C++ parity: bond.cpp:264-266.
        d_eff = d if d is not None and d != _NULL_DATE else self.settlement_date()
        return CashFlows.next_coupon_rate(self._cashflows, False, d_eff)

    def previous_coupon_rate(self, d: Date | None = None) -> float:
        # C++ parity: bond.cpp:268-270.
        d_eff = d if d is not None and d != _NULL_DATE else self.settlement_date()
        return CashFlows.previous_coupon_rate(self._cashflows, False, d_eff)

    def next_cash_flow_date(self, d: Date | None = None) -> Date:
        # C++ parity: bond.cpp:272-274.
        d_eff = d if d is not None and d != _NULL_DATE else self.settlement_date()
        return CashFlows.next_cash_flow_date(self._cashflows, False, d_eff)

    def previous_cash_flow_date(self, d: Date | None = None) -> Date:
        # C++ parity: bond.cpp:276-278.
        d_eff = d if d is not None and d != _NULL_DATE else self.settlement_date()
        return CashFlows.previous_cash_flow_date(self._cashflows, False, d_eff)

    # ------------------------------------------------------------------
    # Yield <-> price round-trip
    # ------------------------------------------------------------------

    def clean_price_from_yield(
        self,
        yield_rate: float,
        day_counter: DayCounter,
        compounding: Compounding,
        frequency: Frequency,
        settlement_date: Date | None = None,
    ) -> float:
        """Clean price from a yield + day-counter convention.

        # C++ parity: bond.cpp:218-224 + bondfunctions.cpp:312-320.
        """
        settle = (
            settlement_date
            if settlement_date is not None and settlement_date != _NULL_DATE
            else self.settlement_date()
        )
        return self.dirty_price_from_yield(
            yield_rate, day_counter, compounding, frequency, settle
        ) - self.accrued_amount(settle)

    def dirty_price_from_yield(
        self,
        yield_rate: float,
        day_counter: DayCounter,
        compounding: Compounding,
        frequency: Frequency,
        settlement_date: Date | None = None,
    ) -> float:
        """Dirty price from a yield + day-counter convention.

        # C++ parity: bond.cpp:226-237 + bondfunctions.cpp:322-336.
        """
        settle = (
            settlement_date
            if settlement_date is not None and settlement_date != _NULL_DATE
            else self.settlement_date()
        )
        current = self.notional(settle)
        if current == 0.0:
            return 0.0
        ir = InterestRate(yield_rate, day_counter, compounding, frequency)
        npv = CashFlows.npv_yield(self._cashflows, ir, False, settle, settle)
        return npv * 100.0 / current

    def yield_rate(
        self,
        day_counter: DayCounter,
        compounding: Compounding,
        frequency: Frequency,
        accuracy: float = _DEFAULT_YIELD_ACCURACY,
        max_evaluations: int = _DEFAULT_YIELD_MAX_ITER,
        guess: float = _DEFAULT_YIELD_GUESS,
        price_type: BondPriceType = BondPriceType.Clean,
    ) -> float:
        """Yield from the theoretical price (engine-driven).

        # C++ parity: bond.cpp:199-216. The C++ method is called ``yield``;
        # Python's ``yield`` is reserved so we rename to ``yield_rate``.
        """
        settle = self.settlement_date()
        current = self.notional(settle)
        if current == 0.0:
            return 0.0
        price_amount = (
            self.clean_price() if price_type == BondPriceType.Clean else self.dirty_price()
        )
        price = BondPrice(price_amount, price_type)
        return self.yield_from_price(
            price, day_counter, compounding, frequency, settle,
            accuracy, max_evaluations, guess,
        )

    def yield_from_price(
        self,
        price: BondPrice,
        day_counter: DayCounter,
        compounding: Compounding,
        frequency: Frequency,
        settlement_date: Date | None = None,
        accuracy: float = _DEFAULT_YIELD_ACCURACY,
        max_evaluations: int = _DEFAULT_YIELD_MAX_ITER,
        guess: float = _DEFAULT_YIELD_GUESS,
    ) -> float:
        """Solve for the yield that reproduces the supplied ``price``.

        # C++ parity: bond.cpp:239-254 + bondfunctions.cpp:373-387.
        # The C++ uses NewtonSafe; we use Brent (already ported) since
        # we don't have NewtonSafe yet and Brent is robust enough for
        # the LOOSE-tier yield-solver tolerance documented in the spec.
        """
        settle = (
            settlement_date
            if settlement_date is not None and settlement_date != _NULL_DATE
            else self.settlement_date()
        )
        current = self.notional(settle)
        if current == 0.0:
            return 0.0
        # Convert price to a target NPV.
        dirty_price = price.amount() + (
            self.accrued_amount(settle) if price.type() == BondPriceType.Clean else 0.0
        )
        target_npv = dirty_price / 100.0 * current

        def objective(y: float) -> float:
            ir = InterestRate(y, day_counter, compounding, frequency)
            return CashFlows.npv_yield(self._cashflows, ir, False, settle, settle) - target_npv

        solver = Brent()
        solver.set_max_evaluations(max_evaluations)
        return solver.solve(objective, accuracy, guess, accuracy)

    # ------------------------------------------------------------------
    # Engine plumbing
    # ------------------------------------------------------------------

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        """Copy bond data into the engine's argument carrier.

        # C++ parity: bond.cpp:285-292.
        """
        qassert.require(isinstance(args, BondArguments), "wrong argument type")
        assert isinstance(args, BondArguments)
        args.settlement_date = self.settlement_date()
        args.cashflows = list(self._cashflows)
        args.calendar = self._calendar

    def fetch_results(self, results: PricingEngineResults) -> None:
        """Pull NPV + settlement_value from engine results.

        # C++ parity: bond.cpp:294-302.
        """
        super().fetch_results(results)
        qassert.require(isinstance(results, BondResults), "wrong result type")
        assert isinstance(results, BondResults)
        self._settlement_value = results.settlement_value

    def setup_expired(self) -> None:
        """Clear NPV + settlement_value when expired.

        # C++ parity: bond.cpp:280-283.
        """
        super().setup_expired()
        self._settlement_value = 0.0

    # ------------------------------------------------------------------
    # Protected helpers for subclasses
    # ------------------------------------------------------------------

    def _calculate_notionals_from_cashflows(self) -> None:
        """Walk coupons to derive ``_notionals`` + ``_notional_schedule``.

        # C++ parity: bond.cpp:363-396.
        """
        self._notional_schedule = []
        self._notionals = []
        last_payment_date: Date = _NULL_DATE
        self._notional_schedule.append(_NULL_DATE)
        for cf in self._cashflows:
            if not isinstance(cf, Coupon):
                continue
            notional = cf.nominal()
            if not self._notionals:
                self._notionals.append(notional)
                last_payment_date = cf.date()
            elif not _close(notional, self._notionals[-1]):
                self._notionals.append(notional)
                self._notional_schedule.append(last_payment_date)
                last_payment_date = cf.date()
            else:
                last_payment_date = cf.date()
        qassert.require(len(self._notionals) > 0, "no coupons provided")
        self._notionals.append(0.0)
        self._notional_schedule.append(last_payment_date)

    def _add_redemptions_to_cashflows(
        self, redemptions: Sequence[float] | None = None,
    ) -> None:
        """Build redemption payments from the existing coupons.

        # C++ parity: bond.cpp:304-329.
        """
        self._calculate_notionals_from_cashflows()
        self._redemptions = []
        red_list = list(redemptions) if redemptions else []
        for i in range(1, len(self._notional_schedule)):
            if i < len(red_list):
                r = red_list[i]
            elif red_list:
                r = red_list[-1]
            else:
                r = 100.0
            amount = (r / 100.0) * (self._notionals[i - 1] - self._notionals[i])
            payment: CashFlow
            if i < len(self._notional_schedule) - 1:
                payment = AmortizingPayment(amount, self._notional_schedule[i])
            else:
                payment = Redemption(amount, self._notional_schedule[i])
            self._cashflows.append(payment)
            self._redemptions.append(payment)
        # Stable-sort so redemptions follow same-date coupons.
        self._cashflows.sort(key=lambda cf: cf.date())

    def _set_single_redemption(
        self, notional: float, redemption: float, date: Date,
    ) -> None:
        """Build a single-redemption schedule with no coupons.

        # C++ parity: bond.cpp:331-338.
        """
        redemption_cf = Redemption(notional * redemption / 100.0, date)
        self._set_single_redemption_cf(notional, redemption_cf)

    def _set_single_redemption_cf(
        self, notional: float, redemption_cf: CashFlow,
    ) -> None:
        """Set the single-redemption schedule from a pre-built cashflow.

        # C++ parity: bond.cpp:340-354.
        """
        self._notionals = [notional, 0.0]
        self._notional_schedule = [_NULL_DATE, redemption_cf.date()]
        self._redemptions = [redemption_cf]
        self._cashflows.append(redemption_cf)


def _close(a: float, b: float, n: int = 42) -> bool:
    """C++ parity: comparison helper close(x, y) used by Bond.

    The C++ ``close`` defaults to checking within 42 ulps (Boost's
    standard). For notional-equality we want a tight check; matching
    that scale via abs+rel.
    """
    del n  # signature parity placeholder; tolerance is fixed.
    diff = abs(a - b)
    if diff == 0.0:
        return True
    return diff <= max(abs(a), abs(b)) * 1e-14


__all__ = [
    "Bond",
    "BondArguments",
    "BondPrice",
    "BondPriceType",
    "BondResults",
]

"""Bootstrap helpers for default-probability term structures.

# C++ parity: ql/termstructures/credit/defaultprobabilityhelpers.{hpp,cpp}
   (v1.42.1).

Provides:

- ``CdsHelper`` base — wraps a CDS instrument + a discount curve. Builds
  the schedule + the CDS at construction; ``implied_quote`` re-prices via
  the configured engine and reads ``fair_spread()`` / ``fair_upfront()``.
- ``SpreadCdsHelper`` — running-spread-quoted CDS helper.
- ``UpfrontCdsHelper`` — upfront-quoted CDS helper.

The helpers subclass ``BootstrapHelper[DefaultProbabilityTermStructure]``;
``set_term_structure`` records the curve being bootstrapped and (re)builds
the pricing engine bound to it. ``implied_quote()`` re-prices and returns
either ``fair_spread()`` (Spread variant) or ``fair_upfront()`` (Upfront
variant).

# C++ parity divergence: the C++ helpers also wire into Settings-driven
   ``RelativeDateBootstrapHelper`` machinery to re-initialize dates when
   the global evaluation date moves. The Python port supports a manual
   ``initialize_dates(evaluation_date)`` plus auto-registration with
   ``ObservableSettings`` for parity with the existing
   ``DepositRateHelper``-style relative-date helpers.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.instruments.claim import Claim
from pquantlib.instruments.credit_default_swap import (
    CreditDefaultSwap,
    PricingModel,
    ProtectionSide,
)
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.credit.midpoint_cds_engine import MidPointCdsEngine
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.bootstrap_helper import RelativeDateBootstrapHelper
from pquantlib.termstructures.credit.default_probability_term_structure import (
    DefaultProbabilityTermStructure,
)
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit


class CdsHelper(RelativeDateBootstrapHelper[DefaultProbabilityTermStructure]):
    """Abstract base for CDS bootstrap helpers.

    Concrete subclasses implement ``implied_quote()`` (reading fair_spread
    for SpreadCdsHelper or fair_upfront for UpfrontCdsHelper) and supply
    upfront semantics via ``_reset_engine``.

    # C++ parity: defaultprobabilityhelpers.hpp:48-128.
    """

    def __init__(
        self,
        quote: Quote | float,
        tenor: Period,
        settlement_days: int,
        calendar: Calendar,
        frequency: Frequency,
        payment_convention: BusinessDayConvention,
        rule: DateGeneration,
        day_counter: DayCounter,
        recovery_rate: float,
        discount_curve: YieldTermStructure,
        settles_accrual: bool = True,
        pays_at_default_time: bool = True,
        start_date: Date | None = None,
        last_period_day_counter: DayCounter | None = None,
        rebates_accrual: bool = True,
        model: PricingModel = PricingModel.Midpoint,
    ) -> None:
        super().__init__(quote)
        self._tenor: Period = tenor
        self._settlement_days: int = settlement_days
        self._calendar: Calendar = calendar
        self._frequency: Frequency = frequency
        self._payment_convention: BusinessDayConvention = payment_convention
        self._rule: DateGeneration = rule
        self._day_counter: DayCounter = day_counter
        self._recovery_rate: float = recovery_rate
        self._discount_curve: YieldTermStructure = discount_curve
        self._settles_accrual: bool = settles_accrual
        self._pays_at_default_time: bool = pays_at_default_time
        self._last_period_day_counter: DayCounter | None = last_period_day_counter
        self._rebates_accrual: bool = rebates_accrual
        self._model: PricingModel = model
        self._explicit_start_date: Date | None = start_date

        # Filled by initialize_dates / set_term_structure.
        self._schedule: Schedule | None = None
        self._swap: CreditDefaultSwap | None = None
        self._protection_start: Date | None = None
        self._start_date: Date | None = None

        # Eagerly initialize against the current evaluation date.
        self._initialize_dates()
        discount_curve.register_with(self)

    def _initialize_dates(self) -> None:
        """C++ parity: defaultprobabilityhelpers.cpp:75-108.

        Builds the schedule + the CDS instrument from the current
        evaluation date. Called both at construction and on Settings-
        driven evaluation-date moves.
        """
        evaluation_date = ObservableSettings().evaluation_date_or_today()
        protection_start: Date = evaluation_date + self._settlement_days
        self._protection_start = protection_start
        explicit_start = self._explicit_start_date
        has_explicit_start = explicit_start is not None and explicit_start != Date()
        start: Date = (
            explicit_start if has_explicit_start and explicit_start is not None
            else protection_start
        )
        # The CDS-specific rules (CDS / CDS2015 / OldCDS) anchor to CDS IMM
        # dates via ``cds_maturity``; that's deferred (see L8-B carve-out).
        # For non-CDS rules, adjust the start date.
        if self._rule not in (
            DateGeneration.OldCDS,
            DateGeneration.CDS,
            DateGeneration.CDS2015,
        ):
            start = self._calendar.adjust(start, self._payment_convention)
            ref: Date = (
                explicit_start if has_explicit_start and explicit_start is not None
                else protection_start
            )
        else:
            # CDS / CDS2015 / OldCDS: cdsMaturity is deferred — fall back
            # to ref + tenor.
            ref = (
                explicit_start if has_explicit_start and explicit_start is not None
                else evaluation_date
            )
        end: Date = ref + self._tenor
        self._schedule = Schedule.from_rule(
            effective_date=start,
            termination_date=end,
            tenor=Period.from_frequency(self._frequency),
            calendar=self._calendar,
            convention=self._payment_convention,
            termination_date_convention=BusinessDayConvention.Unadjusted,
            rule=self._rule,
            end_of_month=False,
        )
        self._earliest_date = self._schedule.date(0)
        self._latest_date = self._calendar.adjust(
            self._schedule.dates[-1], self._payment_convention,
        )
        if self._model == PricingModel.ISDA:
            self._latest_date = self._latest_date + 1
        self._start_date = start

    def set_term_structure(self, ts: DefaultProbabilityTermStructure) -> None:
        """C++ parity: defaultprobabilityhelpers.cpp:60-68.

        Records the curve being bootstrapped and rebuilds the engine.
        """
        super().set_term_structure(ts)
        self._reset_engine(ts)

    def _reset_engine(self, ts: DefaultProbabilityTermStructure) -> None:
        """Rebuild the CDS instrument + bound pricing engine.

        # C++ parity: SpreadCdsHelper::resetEngine
        # (defaultprobabilityhelpers.cpp:137-157). UpfrontCdsHelper has its
        # own override taking the upfront cash settlement date into account.
        """
        assert self._schedule is not None
        assert self._protection_start is not None
        # 100 notional / 1% running spread are conventional; the actual
        # quote is read via ``implied_quote()`` and isn't dependent on
        # them.
        self._swap = CreditDefaultSwap(
            ProtectionSide.Buyer, 100.0, 0.01, self._schedule,
            self._payment_convention, self._day_counter,
            self._settles_accrual, self._pays_at_default_time,
            self._protection_start, claim=self._make_claim(),
            last_period_day_counter=self._last_period_day_counter,
            rebates_accrual=self._rebates_accrual,
            trade_date=ObservableSettings().evaluation_date_or_today(),
        )
        # Midpoint engine for both Midpoint + ISDA models in this stage
        # (ISDA engine deferred — see L8-B carve-out).
        self._swap.set_pricing_engine(
            MidPointCdsEngine(ts, self._recovery_rate, self._discount_curve),
        )

    def _make_claim(self) -> Claim | None:
        """Subclass hook for non-default claims (deferred — returns None)."""
        return None

    def swap(self) -> CreditDefaultSwap | None:
        return self._swap


class SpreadCdsHelper(CdsHelper):
    """Spread-quoted CDS bootstrap helper.

    # C++ parity: defaultprobabilityhelpers.cpp:110-130.
    """

    def implied_quote(self) -> float:
        qassert.require(
            self._swap is not None,
            "SpreadCdsHelper: set_term_structure must be called before implied_quote",
        )
        assert self._swap is not None
        return self._swap.fair_spread()


class UpfrontCdsHelper(CdsHelper):
    """Upfront-quoted CDS bootstrap helper.

    # C++ parity: defaultprobabilityhelpers.cpp:159-225.

    The upfront is quoted in fractional units (e.g. 0.05 == 5% upfront on
    notional). The helper stores the running spread that accompanies the
    upfront quote and re-prices the CDS with a midpoint engine.
    """

    def __init__(
        self,
        upfront: Quote | float,
        running_spread: float,
        tenor: Period,
        settlement_days: int,
        calendar: Calendar,
        frequency: Frequency,
        payment_convention: BusinessDayConvention,
        rule: DateGeneration,
        day_counter: DayCounter,
        recovery_rate: float,
        discount_curve: YieldTermStructure,
        upfront_settlement_days: int = 3,
        settles_accrual: bool = True,
        pays_at_default_time: bool = True,
        start_date: Date | None = None,
        last_period_day_counter: DayCounter | None = None,
        rebates_accrual: bool = True,
        model: PricingModel = PricingModel.Midpoint,
    ) -> None:
        super().__init__(
            upfront, tenor, settlement_days, calendar, frequency,
            payment_convention, rule, day_counter, recovery_rate,
            discount_curve, settles_accrual, pays_at_default_time,
            start_date, last_period_day_counter, rebates_accrual, model,
        )
        self._upfront_settlement_days: int = upfront_settlement_days
        self._running_spread: float = running_spread
        # Rebuild the swap with explicit upfront semantics.
        self._upfront_date = self._compute_upfront_date()

    def _compute_upfront_date(self) -> Date:
        return self._calendar.advance(
            ObservableSettings().evaluation_date_or_today(),
            self._upfront_settlement_days,
            TimeUnit.Days,
            self._payment_convention,
        )

    def _reset_engine(self, ts: DefaultProbabilityTermStructure) -> None:
        """Rebuild a CDS with explicit upfront flow.

        # C++ parity: UpfrontCdsHelper::resetEngine
        # (defaultprobabilityhelpers.cpp:195-217).
        """
        assert self._schedule is not None
        assert self._protection_start is not None
        upfront_date = self._compute_upfront_date()
        self._swap = CreditDefaultSwap.with_upfront(
            ProtectionSide.Buyer, 100.0, 0.01, self._running_spread,
            self._schedule, self._payment_convention, self._day_counter,
            self._settles_accrual, self._pays_at_default_time,
            self._protection_start, upfront_date=upfront_date,
            last_period_day_counter=self._last_period_day_counter,
            rebates_accrual=self._rebates_accrual,
            trade_date=ObservableSettings().evaluation_date_or_today(),
        )
        self._swap.set_pricing_engine(
            MidPointCdsEngine(ts, self._recovery_rate, self._discount_curve),
        )

    def implied_quote(self) -> float:
        qassert.require(
            self._swap is not None,
            "UpfrontCdsHelper: set_term_structure must be called before implied_quote",
        )
        assert self._swap is not None
        return self._swap.fair_upfront()


__all__ = ["CdsHelper", "SpreadCdsHelper", "UpfrontCdsHelper"]

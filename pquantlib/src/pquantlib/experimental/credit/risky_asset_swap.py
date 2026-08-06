"""RiskyAssetSwap — credit-risky asset-swap instrument.

# C++ parity: ql/experimental/credit/riskyassetswap.{hpp,cpp} (v1.42.1).

A risky asset swap is an asset swap where the underlying bond is
subject to default risk. The instrument pays the spread + the
recovered value on default. The pricing decomposes the NPV into:

* Risky bond price (coupon stream weighted by survival probabilities)
* Recovery contribution from the Euler-integral over the default density
* Fixed-coupon discount (asset-swap PV)
* Spread leg (paid as long as no default)

This port keeps the C++ structure 1-1 — eager mutable result fields
populated by ``perform_calculations``.

``AssetSwapHelper`` is the matching bootstrap helper: it rebuilds a
100-notional, 1%-coupon ``RiskyAssetSwap`` off the current evaluation date
and reports its fair spread as the implied quote.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.instruments.instrument import Instrument
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.bootstrap_helper import BootstrapHelper
from pquantlib.termstructures.credit.default_probability_term_structure import (
    DefaultProbabilityTermStructure,
)
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit


class RiskyAssetSwap(Instrument):
    """Risky asset-swap instrument.

    # C++ parity: ``RiskyAssetSwap`` class.

    Construction binds the instrument to two schedules (fixed + float
    legs), two day-counters, a spread, a recovery rate, and the two
    term-structure curves. If ``coupon`` is omitted, the par coupon is
    derived from the curve at calculation time (matches C++ behavior).
    """

    def __init__(
        self,
        fixed_payer: bool,
        nominal: float,
        fixed_schedule: Schedule,
        float_schedule: Schedule,
        fixed_day_counter: DayCounter,
        float_day_counter: DayCounter,
        spread: float,
        recovery_rate: float,
        yield_ts: YieldTermStructure,
        default_ts: DefaultProbabilityTermStructure,
        coupon: float | None = None,
    ) -> None:
        super().__init__()
        self._fixed_payer: bool = fixed_payer
        self._nominal: float = nominal
        self._fixed_schedule: Schedule = fixed_schedule
        self._float_schedule: Schedule = float_schedule
        self._fixed_day_counter: DayCounter = fixed_day_counter
        self._float_day_counter: DayCounter = float_day_counter
        self._spread: float = spread
        self._recovery_rate: float = recovery_rate
        self._yield_ts: YieldTermStructure = yield_ts
        self._default_ts: DefaultProbabilityTermStructure = default_ts
        self._coupon: float | None = coupon

        # Mutable result cache (mirrors C++ mutable fields).
        self._fixed_annuity: float | None = None
        self._float_annuity_v: float | None = None
        self._par_coupon: float | None = None
        self._recovery_value: float | None = None
        self._risky_bond_price: float | None = None

        yield_ts.register_with(self)
        default_ts.register_with(self)

    # ---- Instrument interface --------------------------------------

    def is_expired(self) -> bool:
        """The last fixed-leg date is in the past.

        # C++ parity: riskyassetswap.cpp:47-50.
        """
        today = ObservableSettings().evaluation_date_or_today()
        last = self._fixed_schedule.dates[-1]
        return last <= today

    def setup_expired(self) -> None:
        super().setup_expired()

    def perform_calculations(self) -> None:
        """Run the closed-form pricing.

        # C++ parity: riskyassetswap.cpp:58-79.
        """
        # Order matters (C++ does these in sequence).
        self._float_annuity_v = self._compute_float_annuity()
        self._fixed_annuity = self._compute_fixed_annuity()
        self._par_coupon = self._compute_par_coupon()

        if self._coupon is None:
            self._coupon = self._par_coupon

        self._recovery_value = self._compute_recovery_value()
        self._risky_bond_price = self._compute_risky_bond_price()

        first = self._fixed_schedule.dates[0]
        last = self._fixed_schedule.dates[-1]
        # Local rebinds for the type-checker after the asserts above.
        rbp = self._risky_bond_price
        cpn = self._coupon
        fa = self._fixed_annuity
        fla = self._float_annuity_v
        assert rbp is not None
        assert cpn is not None
        assert fa is not None
        assert fla is not None
        npv = (
            rbp
            - cpn * fa
            + self._yield_ts.discount(first)
            - self._yield_ts.discount(last)
            + self._spread * fla
        )
        npv *= self._nominal
        if not self._fixed_payer:
            npv *= -1.0

        self._npv = npv

    # Use this hook directly because RiskyAssetSwap has no engine.
    def _perform_calculations(self) -> None:
        qassert.require(
            self._engine is None,
            "RiskyAssetSwap has its own perform_calculations; "
            "set_pricing_engine is not supported on this instrument.",
        )
        self.perform_calculations()

    # ---- closed-form helpers -------------------------------------

    def _compute_float_annuity(self) -> float:
        """C++ parity: riskyassetswap.cpp:82-90."""
        annuity = 0.0
        for i in range(1, len(self._float_schedule)):
            dcf = self._float_day_counter.year_fraction(
                self._float_schedule.date(i - 1), self._float_schedule.date(i)
            )
            annuity += dcf * self._yield_ts.discount(self._float_schedule.date(i))
        return annuity

    def _compute_fixed_annuity(self) -> float:
        """C++ parity: riskyassetswap.cpp:93-101.

        # C++ parity note: the C++ code iterates ``float_schedule`` here
        # (not ``fixed_schedule`` — this matches the original behaviour
        # at riskyassetswap.cpp:95). The Python port preserves this
        # apparently-cross-schedule iteration verbatim.
        """
        annuity = 0.0
        for i in range(1, len(self._float_schedule)):
            dcf = self._fixed_day_counter.year_fraction(
                self._float_schedule.date(i - 1), self._float_schedule.date(i)
            )
            annuity += dcf * self._yield_ts.discount(self._float_schedule.date(i))
        return annuity

    def _compute_par_coupon(self) -> float:
        """C++ parity: riskyassetswap.cpp:104-108."""
        assert self._fixed_annuity is not None
        return (
            self._yield_ts.discount(self._fixed_schedule.dates[0])
            - self._yield_ts.discount(self._fixed_schedule.dates[-1])
        ) / self._fixed_annuity

    def _compute_recovery_value(self) -> float:
        """C++ parity: riskyassetswap.cpp:111-138.

        Euler integral on a daily grid (C++ ``stepSize = Days``) over
        each fixed-leg period, accumulating
        ``discount(d) * default_density(d) * dcf(d0, d)``.
        """
        recovery = 0.0
        ref = self._default_ts.reference_date()
        null_cal = NullCalendar()
        dc = self._default_ts.day_counter()

        for i in range(1, len(self._fixed_schedule)):
            prev = self._fixed_schedule.date(i - 1)
            end = self._fixed_schedule.date(i)
            d = prev if prev >= ref else ref
            d0 = d
            while d < end:
                disc = self._yield_ts.discount(d)
                dd = self._default_ts.default_density(d, True)
                dcf = dc.year_fraction(d0, d)
                recovery += disc * dd * dcf
                d0 = d
                d = null_cal.advance(d0, 1, TimeUnit.Days, BusinessDayConvention.Unadjusted)
        return recovery * self._recovery_rate

    def _compute_risky_bond_price(self) -> float:
        """C++ parity: riskyassetswap.cpp:141-156."""
        assert self._coupon is not None
        assert self._recovery_value is not None
        value = 0.0
        for i in range(1, len(self._fixed_schedule)):
            prev = self._fixed_schedule.date(i - 1)
            cur = self._fixed_schedule.date(i)
            dcf = self._fixed_day_counter.year_fraction(prev, cur)
            value += (
                dcf
                * self._yield_ts.discount(cur)
                * self._default_ts.survival_probability(cur, True)
            )
        value *= self._coupon

        last = self._fixed_schedule.dates[-1]
        value += (
            self._yield_ts.discount(last)
            * self._default_ts.survival_probability(last, True)
        )
        return value + self._recovery_value

    # ---- inspectors -----------------------------------------------

    def nominal(self) -> float:
        return self._nominal

    def spread(self) -> float:
        return self._spread

    def fixed_payer(self) -> bool:
        return self._fixed_payer

    def float_annuity(self) -> float:
        """Return cached float annuity (calculates if needed).

        # C++ parity: ``RiskyAssetSwap::floatAnnuity()`` calls the
        # private compute method; Python returns the cached value
        # after ``calculate`` runs lazily on first access.
        """
        self.calculate()
        assert self._float_annuity_v is not None
        return self._float_annuity_v

    # ---- fair-spread -----------------------------------------------

    def fair_spread(self) -> float:
        """Fair par spread that zeros the contract NPV.

        # C++ parity: riskyassetswap.cpp:159-178.
        """
        self.calculate()
        assert self._coupon is not None
        assert self._fixed_annuity is not None
        assert self._recovery_value is not None

        value = 0.0
        for i in range(1, len(self._fixed_schedule)):
            prev = self._fixed_schedule.date(i - 1)
            cur = self._fixed_schedule.date(i)
            dcf = self._fixed_day_counter.year_fraction(prev, cur)
            value += (
                dcf
                * self._yield_ts.discount(cur)
                * self._default_ts.default_probability(cur, True)
            )
        value *= self._coupon

        last = self._fixed_schedule.dates[-1]
        value += (
            self._yield_ts.discount(last)
            * self._default_ts.default_probability(last, True)
        )

        initial_disc = self._yield_ts.discount(self._fixed_schedule.dates[0])
        return (1.0 - initial_disc + value - self._recovery_value) / self._fixed_annuity


class AssetSwapHelper(BootstrapHelper[DefaultProbabilityTermStructure]):
    """Risky-asset-swap helper for default-probability-curve bootstrap.

    # C++ parity: ``class AssetSwapHelper : public DefaultProbabilityHelper``
    # (riskyassetswap.hpp:89-127, riskyassetswap.cpp:181-260).
    # ``DefaultProbabilityHelper`` is the C++ typedef for
    # ``BootstrapHelper<DefaultProbabilityTermStructure>``, which is what this
    # class derives from here. Note C++ does NOT derive from
    # ``RelativeDateBootstrapHelper``; it registers with the evaluation date
    # itself and hand-rolls the ``update()`` guard (riskyassetswap.cpp:225-230),
    # which is reproduced below rather than delegated to the relative-date base.

    The helper rebuilds a 100-notional, 1%-coupon :class:`RiskyAssetSwap` off
    the current evaluation date and reports its fair spread as the implied
    quote.
    """

    def __init__(
        self,
        spread: Quote | float,
        tenor: Period,
        settlement_days: int,
        calendar: Calendar,
        fixed_period: Period,
        fixed_convention: BusinessDayConvention,
        fixed_day_count: DayCounter,
        float_period: Period,
        float_convention: BusinessDayConvention,
        float_day_count: DayCounter,
        recovery_rate: float,
        yield_ts: YieldTermStructure,
        integration_step_size: Period | None = None,
    ) -> None:
        # C++ parity: riskyassetswap.cpp:181-205.
        super().__init__(spread)
        self._tenor: Period = tenor
        self._settlement_days: int = settlement_days
        self._calendar: Calendar = calendar
        self._fixed_convention: BusinessDayConvention = fixed_convention
        self._fixed_period: Period = fixed_period
        self._fixed_day_count: DayCounter = fixed_day_count
        self._float_convention: BusinessDayConvention = float_convention
        self._float_period: Period = float_period
        self._float_day_count: DayCounter = float_day_count
        self._recovery_rate: float = recovery_rate
        self._yield_ts: YieldTermStructure = yield_ts
        # C++ parity note: ``integrationStepSize_`` is stored by the C++ ctor
        # and then never read — ``RiskyAssetSwap``'s own ctor takes no such
        # argument (riskyassetswap.hpp:44-55) and ``initializeDates`` does not
        # pass it (riskyassetswap.cpp:247-258). Kept as a field for signature
        # parity; reproducing the dead store rather than dropping the
        # parameter keeps the constructor callable exactly as in C++.
        self._integration_step_size: Period = (
            Period() if integration_step_size is None else integration_step_size
        )

        self._evaluation_date: Date = Date()
        self._asw: RiskyAssetSwap | None = None
        self._probability: DefaultProbabilityTermStructure | None = None

        self._initialize_dates()

        ObservableSettings().register_with(self)
        yield_ts.register_with(self)

    def implied_quote(self) -> float:
        """Fair spread of the rebuilt risky asset swap.

        # C++ parity: ``AssetSwapHelper::impliedQuote``
        # (riskyassetswap.cpp:207-213).
        """
        qassert.require(self._probability is not None, "default term structure not set")
        assert self._asw is not None
        # C++ parity: "we didn't register as observers - force calculation".
        self._asw.recalculate()
        return self._asw.fair_spread()

    def set_term_structure(self, ts: DefaultProbabilityTermStructure) -> None:
        """# C++ parity: ``AssetSwapHelper::setTermStructure``
        (riskyassetswap.cpp:215-223).

        C++ links a ``RelinkableHandle`` that the already-built swap holds; the
        Python port has no handle indirection, so the swap is (re)built by
        ``_initialize_dates`` once the curve is known. C++ calls
        ``initializeDates()`` here too, so the rebuild happens at the same
        point in both.
        """
        super().set_term_structure(ts)
        self._probability = ts
        self._initialize_dates()

    def update(self) -> None:
        """# C++ parity: ``AssetSwapHelper::update``
        (riskyassetswap.cpp:225-230).
        """
        if self._evaluation_date != ObservableSettings().evaluation_date_or_today():
            self._initialize_dates()
        super().update()

    def _initialize_dates(self) -> None:
        """# C++ parity: ``AssetSwapHelper::initializeDates``
        (riskyassetswap.cpp:232-259).
        """
        self._evaluation_date = ObservableSettings().evaluation_date_or_today()

        self._earliest_date = self._calendar.advance(
            self._evaluation_date,
            self._settlement_days,
            TimeUnit.Days,
        )
        maturity = self._earliest_date + self._tenor
        self._latest_date = self._calendar.adjust(maturity, self._fixed_convention)

        if self._probability is None:
            # Nothing to price against yet; C++ gets here too (the ctor runs
            # before setTermStructure) but its Handle absorbs the emptiness.
            self._asw = None
            return

        fixed_schedule = Schedule.from_rule(
            effective_date=self._earliest_date,
            termination_date=maturity,
            tenor=self._fixed_period,
            calendar=self._calendar,
            convention=self._fixed_convention,
            termination_date_convention=self._fixed_convention,
            rule=DateGeneration.Forward,
            end_of_month=False,
        )
        float_schedule = Schedule.from_rule(
            effective_date=self._earliest_date,
            termination_date=maturity,
            tenor=self._float_period,
            calendar=self._calendar,
            convention=self._float_convention,
            termination_date_convention=self._float_convention,
            rule=DateGeneration.Forward,
            end_of_month=False,
        )

        self._asw = RiskyAssetSwap(
            True,
            100.0,
            fixed_schedule,
            float_schedule,
            self._fixed_day_count,
            self._float_day_count,
            0.01,
            self._recovery_rate,
            self._yield_ts,
            self._probability,
        )


__all__ = ["AssetSwapHelper", "RiskyAssetSwap"]

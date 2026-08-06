"""MakeOIS — fluent builder for overnight-indexed swaps.

# C++ parity: ql/instruments/makeois.{hpp,cpp} (v1.43).

The C++ class is a chained builder ending in an ``operator
OvernightIndexedSwap()`` conversion. PQuantLib ports it as a class with chained
``with_*()`` setters and a terminal :meth:`MakeOIS.build`; :func:`make_ois` is a
keyword-argument façade over it, so the resolution logic exists exactly once.

Python divergences from C++:

- ``Settings::instance().evaluationDate()`` is
  :meth:`~pquantlib.patterns.observable_settings.ObservableSettings.evaluation_date_or_today`;
  :meth:`MakeOIS.with_evaluation_date` pins it locally without touching global
  state (see :mod:`pquantlib.instruments.make_vanilla_swap`).
- C++ picks the default settlement days by ``dynamic_pointer_cast`` to ``Sonia``
  (0) or ``Corra`` (1), falling back to 2. PQuantLib has no ``Corra`` index yet,
  so only the ``Sonia`` branch exists; everything else takes the 2-day default,
  exactly as C++ does for a non-Sonia non-Corra index.
- ``with_averaging_method`` / ``with_lookback_days`` / ``with_lockout_days`` /
  ``with_observation_shift`` are part of the C++ surface but
  :class:`~pquantlib.instruments.overnight_indexed_swap.OvernightIndexedSwap`
  cannot yet carry them to its leg. Rather than accept and silently drop them,
  :meth:`MakeOIS.build` rejects any non-default value with an explicit message.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.rate_averaging import RateAveraging
from pquantlib.indexes.ibor.sonia import Sonia
from pquantlib.instruments.overnight_indexed_swap import OvernightIndexedSwap
from pquantlib.instruments.swap import SwapType
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.swap.discounting_swap_engine import DiscountingSwapEngine
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule, allows_end_of_month
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.indexes.overnight_index import OvernightIndex
    from pquantlib.pricingengines.pricing_engine import PricingEngine
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol
    from pquantlib.time.calendar import Calendar

_ZERO_PERIOD: Period = Period(0, TimeUnit.Days)
_NULL_DATE: Date = Date()

_SONIA_SETTLEMENT_DAYS: int = 0
_DEFAULT_SETTLEMENT_DAYS: int = 2


def _default_settlement_days(index: OvernightIndex) -> int:
    """Per-index spot-date convention.

    # C++ parity: makeois.cpp:58-70 — ``dynamic_pointer_cast<Sonia>`` → 0,
    ``dynamic_pointer_cast<Corra>`` → 1, otherwise 2. There is no ``Corra``
    index in this port yet, and ``OvernightIndex.clone`` returns a plain
    ``OvernightIndex`` in C++ too, so a cloned Sonia takes the 2-day default on
    both sides.
    """
    if isinstance(index, Sonia):
        return _SONIA_SETTLEMENT_DAYS
    return _DEFAULT_SETTLEMENT_DAYS


class MakeOIS:
    """Fluent builder for :class:`~pquantlib.instruments.overnight_indexed_swap.OvernightIndexedSwap`.

    # C++ parity: ``MakeOIS`` (makeois.hpp:39-137).
    """

    def __init__(
        self,
        swap_tenor: Period,
        overnight_index: OvernightIndex,
        fixed_rate: float | None = None,
        forward_start: Period = _ZERO_PERIOD,
    ) -> None:
        # C++ parity: makeois.cpp:32-40.
        self._swap_tenor: Period | None = swap_tenor
        self._overnight_index: OvernightIndex = overnight_index
        self._fixed_rate: float | None = fixed_rate
        self._forward_start: Period = forward_start

        self._settlement_days: int | None = None
        self._effective_date: Date | None = None
        self._termination_date: Date | None = None
        self._fixed_calendar: Calendar = overnight_index.fixing_calendar()
        self._overnight_calendar: Calendar = overnight_index.fixing_calendar()

        self._fixed_payment_frequency: Frequency = Frequency.Annual
        self._overnight_payment_frequency: Frequency = Frequency.Annual
        self._payment_calendar: Calendar | None = None
        self._payment_adjustment: BusinessDayConvention = BusinessDayConvention.Following
        self._payment_lag: int = 0

        self._fixed_convention: BusinessDayConvention = BusinessDayConvention.ModifiedFollowing
        self._fixed_termination_date_convention: BusinessDayConvention = (
            BusinessDayConvention.ModifiedFollowing
        )
        self._overnight_convention: BusinessDayConvention = (
            BusinessDayConvention.ModifiedFollowing
        )
        self._overnight_termination_date_convention: BusinessDayConvention = (
            BusinessDayConvention.ModifiedFollowing
        )
        self._fixed_rule: DateGeneration = DateGeneration.Backward
        self._overnight_rule: DateGeneration = DateGeneration.Backward
        self._fixed_end_of_month: bool = False
        self._overnight_end_of_month: bool = False
        self._is_default_eom: bool = True
        self._maturity_end_of_month: bool | None = None

        self._type: SwapType = SwapType.Payer
        self._nominal: float = 1.0

        self._overnight_spread: float = 0.0
        self._fixed_day_count: DayCounter = overnight_index.day_counter()

        self._engine: PricingEngine | None = None

        self._telescopic_value_dates: bool = False
        self._averaging_method: RateAveraging = RateAveraging.Compound
        self._lookback_days: int | None = None
        self._lockout_days: int = 0
        self._apply_observation_shift: bool = False

        # PQuantLib addition — see the module docstring.
        self._evaluation_date: Date | None = None

    # --- chained setters ---------------------------------------------------

    def receive_fixed(self, flag: bool = True) -> MakeOIS:
        self._type = SwapType.Receiver if flag else SwapType.Payer
        return self

    def with_type(self, swap_type: SwapType) -> MakeOIS:
        self._type = swap_type
        return self

    def with_nominal(self, n: float) -> MakeOIS:
        self._nominal = n
        return self

    def with_settlement_days(self, settlement_days: int) -> MakeOIS:
        self._settlement_days = settlement_days
        return self

    def with_effective_date(self, effective_date: Date) -> MakeOIS:
        self._effective_date = None if effective_date == _NULL_DATE else effective_date
        return self

    def with_termination_date(self, termination_date: Date) -> MakeOIS:
        """Set the end date; a non-null one clears the swap tenor.

        # C++ parity: makeois.cpp:199-205.
        """
        if termination_date == _NULL_DATE:
            self._termination_date = None
        else:
            self._termination_date = termination_date
            self._swap_tenor = None
        return self

    def with_rule(self, rule: DateGeneration) -> MakeOIS:
        return self.with_fixed_leg_rule(rule).with_overnight_leg_rule(rule)

    def with_fixed_leg_rule(self, rule: DateGeneration) -> MakeOIS:
        self._fixed_rule = rule
        return self

    def with_overnight_leg_rule(self, rule: DateGeneration) -> MakeOIS:
        self._overnight_rule = rule
        return self

    def with_payment_frequency(self, f: Frequency) -> MakeOIS:
        return self.with_fixed_leg_payment_frequency(f).with_overnight_leg_payment_frequency(f)

    def with_fixed_leg_payment_frequency(self, f: Frequency) -> MakeOIS:
        self._fixed_payment_frequency = f
        return self

    def with_overnight_leg_payment_frequency(self, f: Frequency) -> MakeOIS:
        self._overnight_payment_frequency = f
        return self

    def with_payment_adjustment(self, convention: BusinessDayConvention) -> MakeOIS:
        self._payment_adjustment = convention
        return self

    def with_payment_lag(self, lag: int) -> MakeOIS:
        self._payment_lag = lag
        return self

    def with_payment_calendar(self, calendar: Calendar) -> MakeOIS:
        self._payment_calendar = calendar
        return self

    def with_calendar(self, calendar: Calendar) -> MakeOIS:
        return self.with_fixed_leg_calendar(calendar).with_overnight_leg_calendar(calendar)

    def with_fixed_leg_calendar(self, calendar: Calendar) -> MakeOIS:
        self._fixed_calendar = calendar
        return self

    def with_overnight_leg_calendar(self, calendar: Calendar) -> MakeOIS:
        self._overnight_calendar = calendar
        return self

    def with_convention(self, bdc: BusinessDayConvention) -> MakeOIS:
        return self.with_fixed_leg_convention(bdc).with_overnight_leg_convention(bdc)

    def with_fixed_leg_convention(self, bdc: BusinessDayConvention) -> MakeOIS:
        self._fixed_convention = bdc
        return self

    def with_overnight_leg_convention(self, bdc: BusinessDayConvention) -> MakeOIS:
        self._overnight_convention = bdc
        return self

    def with_termination_date_convention(self, bdc: BusinessDayConvention) -> MakeOIS:
        self.with_fixed_leg_termination_date_convention(bdc)
        return self.with_overnight_leg_termination_date_convention(bdc)

    def with_fixed_leg_termination_date_convention(self, bdc: BusinessDayConvention) -> MakeOIS:
        self._fixed_termination_date_convention = bdc
        return self

    def with_overnight_leg_termination_date_convention(
        self, bdc: BusinessDayConvention
    ) -> MakeOIS:
        self._overnight_termination_date_convention = bdc
        return self

    def with_end_of_month(self, flag: bool = True) -> MakeOIS:
        return self.with_fixed_leg_end_of_month(flag).with_overnight_leg_end_of_month(flag)

    def with_fixed_leg_end_of_month(self, flag: bool = True) -> MakeOIS:
        self._fixed_end_of_month = flag
        self._is_default_eom = False
        return self

    def with_overnight_leg_end_of_month(self, flag: bool = True) -> MakeOIS:
        self._overnight_end_of_month = flag
        self._is_default_eom = False
        return self

    def with_maturity_end_of_month(self, flag: bool = True) -> MakeOIS:
        self._maturity_end_of_month = flag
        self._is_default_eom = False
        return self

    def with_fixed_leg_day_count(self, day_counter: DayCounter) -> MakeOIS:
        self._fixed_day_count = day_counter
        return self

    def with_overnight_leg_spread(self, spread: float) -> MakeOIS:
        self._overnight_spread = spread
        return self

    def with_discounting_term_structure(
        self, discounting_term_structure: YieldTermStructureProtocol
    ) -> MakeOIS:
        self._engine = DiscountingSwapEngine(
            discounting_term_structure, include_settlement_date_flows=False
        )
        return self

    def with_telescopic_value_dates(self, telescopic_value_dates: bool) -> MakeOIS:
        self._telescopic_value_dates = telescopic_value_dates
        return self

    def with_averaging_method(self, averaging_method: RateAveraging) -> MakeOIS:
        self._averaging_method = averaging_method
        return self

    def with_lookback_days(self, lookback_days: int) -> MakeOIS:
        self._lookback_days = lookback_days
        return self

    def with_lockout_days(self, lockout_days: int) -> MakeOIS:
        self._lockout_days = lockout_days
        return self

    def with_observation_shift(self, apply_observation_shift: bool = True) -> MakeOIS:
        self._apply_observation_shift = apply_observation_shift
        return self

    def with_pricing_engine(self, engine: PricingEngine) -> MakeOIS:
        self._engine = engine
        return self

    def with_evaluation_date(self, d: Date) -> MakeOIS:
        """Pin the reference date used for the spot-date calculation.

        PQuantLib addition — see :meth:`~pquantlib.instruments.make_vanilla_swap.MakeVanillaSwap.with_evaluation_date`.
        """
        self._evaluation_date = d
        return self

    # --- build -------------------------------------------------------------

    def _start_date(self) -> Date:
        """Resolve the swap's start date.

        # C++ parity: makeois.cpp:52-83. Note the asymmetry with
        ``MakeVanillaSwap``: a zero-length forward start is still adjusted
        ``Following`` here.
        """
        if self._effective_date is not None:
            return self._effective_date
        settlement_days = (
            self._settlement_days
            if self._settlement_days is not None
            else _default_settlement_days(self._overnight_index)
        )
        ref_date = (
            self._evaluation_date
            if self._evaluation_date is not None
            else ObservableSettings().evaluation_date_or_today()
        )
        ref_date = self._overnight_calendar.adjust(ref_date)
        spot_date = self._overnight_calendar.advance(ref_date, settlement_days, TimeUnit.Days)
        start_date = spot_date + self._forward_start
        if self._forward_start.length < 0:
            return self._overnight_calendar.adjust(start_date, BusinessDayConvention.Preceding)
        return self._overnight_calendar.adjust(start_date, BusinessDayConvention.Following)

    def _end_of_month_flags(self, start_date: Date) -> tuple[bool, bool, bool]:
        """(fixed, overnight, maturity) end-of-month flags.

        # C++ parity: makeois.cpp:85-94. With no end-of-month setter called at
        all, all three follow "is the start date the calendar end of month?".
        """
        if self._is_default_eom:
            is_eom = self._overnight_calendar.is_end_of_month(start_date)
            return is_eom, is_eom, is_eom
        maturity_end_of_month = (
            self._maturity_end_of_month
            if self._maturity_end_of_month is not None
            else self._overnight_end_of_month
        )
        return self._fixed_end_of_month, self._overnight_end_of_month, maturity_end_of_month

    def _end_date(self, start_date: Date, maturity_end_of_month: bool) -> Date:
        """# C++ parity: makeois.cpp:96-102."""
        if self._termination_date is not None:
            return self._termination_date
        assert self._swap_tenor is not None
        end_date = start_date + self._swap_tenor
        if (
            maturity_end_of_month
            and allows_end_of_month(self._swap_tenor)
            and self._overnight_calendar.is_end_of_month(start_date)
        ):
            end_date = self._overnight_calendar.end_of_month(end_date)
        return end_date

    @staticmethod
    def _resolve_frequency_and_rule(
        frequency: Frequency, rule: DateGeneration
    ) -> tuple[Frequency, DateGeneration]:
        """``Once`` and ``Zero`` imply each other.

        # C++ parity: makeois.cpp:104-118 — either one alone forces both.
        """
        if frequency == Frequency.Once or rule == DateGeneration.Zero:
            return Frequency.Once, DateGeneration.Zero
        return frequency, rule

    def _check_unsupported(self) -> None:
        """Reject the C++ setters this port cannot yet carry to the instrument.

        ``OvernightIndexedSwap`` builds its leg through ``overnight_leg``, which
        exposes neither the averaging method nor the lookback / lockout /
        observation-shift modifiers. Accepting them silently would reproduce
        exactly the class of defect this module is tested against.
        """
        qassert.require(
            self._averaging_method == RateAveraging.Compound,
            "MakeOIS: only RateAveraging.Compound is supported; "
            "OvernightIndexedSwap cannot carry an averaging method to its leg",
        )
        qassert.require(
            self._lookback_days is None,
            "MakeOIS: lookback days are not supported; "
            "OvernightIndexedSwap cannot carry them to its leg",
        )
        qassert.require(
            self._lockout_days == 0,
            "MakeOIS: lockout days are not supported; "
            "OvernightIndexedSwap cannot carry them to its leg",
        )
        qassert.require(
            not self._apply_observation_shift,
            "MakeOIS: the observation shift is not supported; "
            "OvernightIndexedSwap cannot carry it to its leg",
        )

    def _make_swap(
        self, fixed_schedule: Schedule, overnight_schedule: Schedule, fixed_rate: float
    ) -> OvernightIndexedSwap:
        return OvernightIndexedSwap(
            self._type,
            self._nominal,
            fixed_schedule,
            fixed_rate,
            self._fixed_day_count,
            self._overnight_index,
            spread=self._overnight_spread,
            payment_lag=self._payment_lag,
            payment_adjustment=self._payment_adjustment,
            payment_calendar=self._payment_calendar,
            telescopic_value_dates=self._telescopic_value_dates,
            overnight_schedule=overnight_schedule,
        )

    def build(self) -> OvernightIndexedSwap:
        """Construct the swap.

        # C++ parity: ``MakeOIS::operator ext::shared_ptr<OvernightIndexedSwap>()``
        (makeois.cpp:47-183).
        """
        qassert.require(
            self._effective_date is None or self._settlement_days is None,
            "cannot set both an explicit effective date and settlement days; "
            "use one or the other",
        )
        self._check_unsupported()

        start_date = self._start_date()
        fixed_eom, overnight_eom, maturity_eom = self._end_of_month_flags(start_date)
        end_date = self._end_date(start_date, maturity_eom)

        fixed_frequency, fixed_rule = self._resolve_frequency_and_rule(
            self._fixed_payment_frequency, self._fixed_rule
        )
        overnight_frequency, overnight_rule = self._resolve_frequency_and_rule(
            self._overnight_payment_frequency, self._overnight_rule
        )

        fixed_schedule = Schedule.from_rule(
            start_date,
            end_date,
            Period.from_frequency(fixed_frequency),
            self._fixed_calendar,
            self._fixed_convention,
            self._fixed_termination_date_convention,
            fixed_rule,
            fixed_eom,
        )
        overnight_schedule = Schedule.from_rule(
            start_date,
            end_date,
            Period.from_frequency(overnight_frequency),
            self._overnight_calendar,
            self._overnight_convention,
            self._overnight_termination_date_convention,
            overnight_rule,
            overnight_eom,
        )

        used_fixed_rate = self._fixed_rate
        if used_fixed_rate is None:
            temp = self._make_swap(fixed_schedule, overnight_schedule, 0.0)
            if self._engine is None:
                discount = self._overnight_index.forecast_term_structure()
                qassert.require(
                    discount is not None,
                    f"null term structure set to this instance of {self._overnight_index.name()}",
                )
                assert discount is not None
                temp.set_pricing_engine(
                    DiscountingSwapEngine(discount, include_settlement_date_flows=False)
                )
            else:
                temp.set_pricing_engine(self._engine)
            used_fixed_rate = temp.fair_rate()

        ois = self._make_swap(fixed_schedule, overnight_schedule, used_fixed_rate)
        if self._engine is None:
            discount = self._overnight_index.forecast_term_structure()
            if discount is not None:
                ois.set_pricing_engine(
                    DiscountingSwapEngine(discount, include_settlement_date_flows=False)
                )
        else:
            ois.set_pricing_engine(self._engine)
        return ois

    def __call__(self) -> OvernightIndexedSwap:
        """Alias for :meth:`build`, mirroring C++ ``operator OvernightIndexedSwap()``."""
        return self.build()


def make_ois(
    swap_tenor: Period,
    overnight_index: OvernightIndex,
    fixed_rate: float | None = None,
    forward_start: Period = _ZERO_PERIOD,
    *,
    nominal: float = 1.0,
    swap_type: SwapType = SwapType.Payer,
    settlement_days: int | None = None,
    effective_date: Date | None = None,
    termination_date: Date | None = None,
    payment_frequency: Frequency | None = None,
    fixed_leg_payment_frequency: Frequency | None = None,
    overnight_leg_payment_frequency: Frequency | None = None,
    payment_adjustment: BusinessDayConvention = BusinessDayConvention.Following,
    payment_lag: int = 0,
    payment_calendar: Calendar | None = None,
    calendar: Calendar | None = None,
    fixed_leg_calendar: Calendar | None = None,
    overnight_leg_calendar: Calendar | None = None,
    fixed_convention: BusinessDayConvention = BusinessDayConvention.ModifiedFollowing,
    overnight_convention: BusinessDayConvention = BusinessDayConvention.ModifiedFollowing,
    fixed_termination_convention: BusinessDayConvention = (
        BusinessDayConvention.ModifiedFollowing
    ),
    overnight_termination_convention: BusinessDayConvention = (
        BusinessDayConvention.ModifiedFollowing
    ),
    fixed_rule: DateGeneration = DateGeneration.Backward,
    overnight_rule: DateGeneration = DateGeneration.Backward,
    end_of_month: bool | None = None,
    fixed_leg_end_of_month: bool | None = None,
    overnight_leg_end_of_month: bool | None = None,
    maturity_end_of_month: bool | None = None,
    fixed_leg_day_count: DayCounter | None = None,
    overnight_leg_spread: float = 0.0,
    discount_curve: YieldTermStructureProtocol | None = None,
    pricing_engine: PricingEngine | None = None,
    telescopic_value_dates: bool = False,
    averaging_method: RateAveraging = RateAveraging.Compound,
    evaluation_date: Date | None = None,
) -> OvernightIndexedSwap:
    """Keyword-argument façade over :class:`MakeOIS`.

    Every keyword maps to the identically-named chained setter, so the class is
    the single implementation. Leaving all four end-of-month keywords at
    ``None`` keeps the C++ "default end-of-month" behaviour, in which the flags
    follow whether the start date is the calendar end of month.
    """
    builder = MakeOIS(swap_tenor, overnight_index, fixed_rate, forward_start)
    builder.with_type(swap_type).with_nominal(nominal)
    builder.with_payment_adjustment(payment_adjustment).with_payment_lag(payment_lag)
    builder.with_fixed_leg_convention(fixed_convention)
    builder.with_overnight_leg_convention(overnight_convention)
    builder.with_fixed_leg_termination_date_convention(fixed_termination_convention)
    builder.with_overnight_leg_termination_date_convention(overnight_termination_convention)
    builder.with_fixed_leg_rule(fixed_rule).with_overnight_leg_rule(overnight_rule)
    builder.with_overnight_leg_spread(overnight_leg_spread)
    builder.with_telescopic_value_dates(telescopic_value_dates)
    builder.with_averaging_method(averaging_method)
    if settlement_days is not None:
        builder.with_settlement_days(settlement_days)
    if effective_date is not None:
        builder.with_effective_date(effective_date)
    if termination_date is not None:
        builder.with_termination_date(termination_date)
    if payment_frequency is not None:
        builder.with_payment_frequency(payment_frequency)
    if fixed_leg_payment_frequency is not None:
        builder.with_fixed_leg_payment_frequency(fixed_leg_payment_frequency)
    if overnight_leg_payment_frequency is not None:
        builder.with_overnight_leg_payment_frequency(overnight_leg_payment_frequency)
    if payment_calendar is not None:
        builder.with_payment_calendar(payment_calendar)
    if calendar is not None:
        builder.with_calendar(calendar)
    if fixed_leg_calendar is not None:
        builder.with_fixed_leg_calendar(fixed_leg_calendar)
    if overnight_leg_calendar is not None:
        builder.with_overnight_leg_calendar(overnight_leg_calendar)
    if end_of_month is not None:
        builder.with_end_of_month(end_of_month)
    if fixed_leg_end_of_month is not None:
        builder.with_fixed_leg_end_of_month(fixed_leg_end_of_month)
    if overnight_leg_end_of_month is not None:
        builder.with_overnight_leg_end_of_month(overnight_leg_end_of_month)
    if maturity_end_of_month is not None:
        builder.with_maturity_end_of_month(maturity_end_of_month)
    if fixed_leg_day_count is not None:
        builder.with_fixed_leg_day_count(fixed_leg_day_count)
    if discount_curve is not None:
        builder.with_discounting_term_structure(discount_curve)
    if pricing_engine is not None:
        builder.with_pricing_engine(pricing_engine)
    if evaluation_date is not None:
        builder.with_evaluation_date(evaluation_date)
    return builder.build()


__all__ = ["MakeOIS", "make_ois"]

"""Cross-currency basis-swap rate helpers.

# C++ parity: ql/experimental/termstructures/crosscurrencyratehelpers.{hpp,cpp}
# (v1.42.1).

Helpers for bootstrapping a discount curve from quoted cross-currency
(xccy) swaps. Following the C++ conventions for a pair like EUR-USD:
EUR is the *base* currency, USD the *quote* currency; the base/quote
legs and the basis/collateral/notional flags are defined relative to
those roles (see N. Moreni & A. Pallavicini, 2015, "FX Modelling in
Collateralized Markets").

Hierarchy:

* :class:`CrossCurrencySwapRateHelperBase` — shared date setup +
  notional-exchange dates.
* :class:`CrossCurrencyBasisSwapRateHelperBase` — two-floating-leg base.
* :class:`ConstNotionalCrossCurrencyBasisSwapRateHelper` — constant
  notionals on both legs.
* :class:`MtMCrossCurrencyBasisSwapRateHelper` — one leg's notional
  resets to the prevailing FX each period (marked-to-market).
* :class:`ConstNotionalCrossCurrencySwapRateHelper` — fixed-vs-floating
  par xccy swap (the FX spot cancels at par).

Each ``implied_quote`` reproduces the C++ basis/par-rate solve in terms
of the per-leg NPV + BPS (the NPV includes the start/maturity notional
exchanges). NPV/BPS of a leg are computed via ``CashFlows.npv_curve`` +
``CashFlows.bps`` (the C++ ``CashFlows::npvbps`` single-pass split).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, final

from pquantlib import qassert
from pquantlib.cashflows.cash_flows import CashFlows
from pquantlib.cashflows.coupon import Coupon
from pquantlib.cashflows.coupon_pricer import IborCouponPricer, set_coupon_pricer
from pquantlib.cashflows.fixed_rate_leg import fixed_rate_leg
from pquantlib.cashflows.ibor_leg import ibor_leg
from pquantlib.cashflows.overnight_leg import overnight_leg
from pquantlib.indexes.overnight_index import OvernightIndex
from pquantlib.termstructures.bootstrap_helper import (
    PillarChoice,
    RelativeDateBootstrapHelper,
)
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period
from pquantlib.time.schedule import MakeSchedule
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from pquantlib.cashflows.cash_flow import CashFlow
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.indexes.ibor_index import IborIndex
    from pquantlib.quotes.quote import Quote
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol
    from pquantlib.time.calendar import Calendar
    from pquantlib.time.date import Date

_SAMPLE_FIXED_RATE = 0.01
_BASIS_POINT = 1.0e-4


def _eval_date() -> Date:
    from pquantlib.patterns.observable_settings import (  # noqa: PLC0415
        ObservableSettings,
    )

    return ObservableSettings().evaluation_date_or_today()


def _leg_schedule(
    evaluation_date: Date,
    tenor: Period,
    frequency: Period,
    fixing_days: int,
    calendar: Calendar,
    convention: BusinessDayConvention,
    end_of_month: bool,
) -> object:
    # C++ parity: crosscurrencyratehelpers.cpp:36-57 (legSchedule).
    qassert.require(
        tenor >= frequency,
        "XCCY instrument tenor should not be smaller than coupon frequency.",
    )
    reference_date = calendar.adjust(evaluation_date, BusinessDayConvention.Following)
    earliest = calendar.advance(
        reference_date, fixing_days, TimeUnit.Days, convention
    )
    maturity = earliest + tenor
    return (
        MakeSchedule()
        .from_date(earliest)
        .to(maturity)
        .with_tenor(frequency)
        .with_calendar(calendar)
        .with_convention(convention)
        .with_end_of_month(end_of_month)
        .backwards()
        .build()
    )


def _build_floating_leg(
    evaluation_date: Date,
    tenor: Period,
    fixing_days: int,
    calendar: Calendar,
    convention: BusinessDayConvention,
    end_of_month: bool,
    idx: IborIndex,
    payment_frequency: Frequency,
) -> list[CashFlow]:
    # C++ parity: crosscurrencyratehelpers.cpp:59-86 (buildFloatingLeg).
    is_overnight = isinstance(idx, OvernightIndex)
    if payment_frequency == Frequency.NoFrequency:
        qassert.require(
            not is_overnight, "Require payment frequency for overnight indices."
        )
        freq_period = idx.tenor()
    else:
        freq_period = Period.from_frequency(payment_frequency)

    sch = _leg_schedule(
        evaluation_date, tenor, freq_period, fixing_days, calendar, convention, end_of_month
    )
    if is_overnight:
        assert isinstance(idx, OvernightIndex)
        return overnight_leg(sch, idx, [1.0])  # type: ignore[arg-type]
    leg = ibor_leg(sch, idx, [1.0])  # type: ignore[arg-type]
    set_coupon_pricer(leg, IborCouponPricer())
    return leg


def _build_fixed_leg(
    evaluation_date: Date,
    tenor: Period,
    fixing_days: int,
    calendar: Calendar,
    convention: BusinessDayConvention,
    end_of_month: bool,
    payment_frequency: Frequency,
    day_count: DayCounter,
) -> list[CashFlow]:
    # C++ parity: crosscurrencyratehelpers.cpp:88-106 (buildFixedLeg).
    freq_period = Period.from_frequency(payment_frequency)
    sch = _leg_schedule(
        evaluation_date, tenor, freq_period, fixing_days, calendar, convention, end_of_month
    )
    return fixed_rate_leg(sch, [1.0], [_SAMPLE_FIXED_RATE], day_count)  # type: ignore[arg-type]


def _npvbps_const_notional_leg(
    leg: list[CashFlow],
    initial_exchange_date: Date,
    final_exchange_date: Date,
    discount_handle: YieldTermStructureProtocol,
) -> tuple[float, float]:
    # C++ parity: crosscurrencyratehelpers.cpp:108-123 (npvbpsConstNotionalLeg).
    ref = discount_handle.reference_date()
    npv = CashFlows.npv_curve(leg, discount_handle, True, ref, ref)
    bps = CashFlows.bps(leg, discount_handle, True, ref, ref)
    # Include the notional exchange at start and maturity.
    npv += -1.0 * discount_handle.discount(initial_exchange_date)
    npv += discount_handle.discount(final_exchange_date)
    bps /= _BASIS_POINT
    return npv, bps


def _npvbps_resetting_leg(
    leg: list[CashFlow],
    payment_lag: int,
    payment_calendar: Calendar,
    convention: BusinessDayConvention,
    discount_handle: YieldTermStructureProtocol,
    foreign_handle: YieldTermStructureProtocol,
) -> tuple[float, float]:
    # C++ parity: crosscurrencyratehelpers.cpp:125-211 (ResettingLegCalculator
    # + npvbpsResettingLeg). The notional on the resetting leg is reset each
    # period to the implied forward FX (ratio of foreign/domestic discounts).
    npv = 0.0
    bps = 0.0
    for cf in leg:
        if not isinstance(cf, Coupon):
            continue
        start = cf.accrual_start_date()
        end = cf.accrual_end_date()
        accrual = cf.accrual_period()
        notional_adj = foreign_handle.discount(start) / discount_handle.discount(start)
        adjusted_notional = cf.nominal() * notional_adj

        if payment_lag == 0:
            discount_start = discount_handle.discount(start)
            discount_end = discount_handle.discount(end)
        else:
            payment_start = payment_calendar.advance(
                start, payment_lag, TimeUnit.Days, convention
            )
            payment_end = payment_calendar.advance(
                end, payment_lag, TimeUnit.Days, convention
            )
            discount_start = discount_handle.discount(payment_start)
            discount_end = discount_handle.discount(payment_end)

        npv_redeemed = adjusted_notional * discount_end * (1.0 + cf.rate() * accrual)
        npv_borrowed = -adjusted_notional * discount_start
        npv += npv_redeemed + npv_borrowed
        bps += adjusted_notional * discount_end * accrual
    return npv, bps


class CrossCurrencySwapRateHelperBase(
    RelativeDateBootstrapHelper["YieldTermStructureProtocol"]
):
    """Shared base for cross-currency swap rate helpers.

    # C++ parity: ``class CrossCurrencySwapRateHelperBase :
    # public RelativeDateRateHelper`` in crosscurrencyratehelpers.hpp:32-61.
    """

    def __init__(
        self,
        quote: Quote | float,
        tenor: Period,
        fixing_days: int,
        calendar: Calendar,
        convention: BusinessDayConvention,
        end_of_month: bool,
        collateral_curve: YieldTermStructureProtocol,
        payment_lag: int,
    ) -> None:
        super().__init__(quote)
        self._tenor: Period = tenor
        self._fixing_days: int = fixing_days
        self._calendar: Calendar = calendar
        self._convention: BusinessDayConvention = convention
        self._end_of_month: bool = end_of_month
        self._payment_lag: int = payment_lag
        self._collateral_handle: YieldTermStructureProtocol = collateral_curve
        self._initial_notional_exchange_date: Date | None = None
        self._final_notional_exchange_date: Date | None = None

    def _initialize_dates_from_legs(
        self, first_leg: list[CashFlow], second_leg: list[CashFlow]
    ) -> None:
        # C++ parity: crosscurrencyratehelpers.cpp:241-262.
        self._earliest_date = min(
            CashFlows.start_date(first_leg), CashFlows.start_date(second_leg)
        )
        self._maturity_date = max(
            CashFlows.maturity_date(first_leg), CashFlows.maturity_date(second_leg)
        )
        if self._payment_lag == 0:
            self._initial_notional_exchange_date = self._earliest_date
            self._final_notional_exchange_date = self._maturity_date
        else:
            self._initial_notional_exchange_date = self._calendar.advance(
                self._earliest_date, self._payment_lag, TimeUnit.Days, self._convention
            )
            self._final_notional_exchange_date = self._calendar.advance(
                self._maturity_date, self._payment_lag, TimeUnit.Days, self._convention
            )
        last_payment_date = max(first_leg[-1].date(), second_leg[-1].date())
        self._latest_relevant_date = self._latest_date = max(
            self._maturity_date, last_payment_date
        )
        self._pillar_date = self._latest_relevant_date
        self._pillar_choice = PillarChoice.LastRelevantDate


class CrossCurrencyBasisSwapRateHelperBase(CrossCurrencySwapRateHelperBase):
    """Base for cross-currency *basis* swap rate helpers (two floating legs).

    # C++ parity: ``class CrossCurrencyBasisSwapRateHelperBase :
    # public CrossCurrencySwapRateHelperBase`` in
    # crosscurrencyratehelpers.hpp:65-93.
    """

    def __init__(
        self,
        basis: Quote | float,
        tenor: Period,
        fixing_days: int,
        calendar: Calendar,
        convention: BusinessDayConvention,
        end_of_month: bool,
        base_currency_index: IborIndex,
        quote_currency_index: IborIndex,
        collateral_curve: YieldTermStructureProtocol,
        is_fx_base_currency_collateral_currency: bool,
        is_basis_on_fx_base_currency_leg: bool,
        payment_frequency: Frequency = Frequency.NoFrequency,
        payment_lag: int = 0,
    ) -> None:
        super().__init__(
            basis, tenor, fixing_days, calendar, convention, end_of_month,
            collateral_curve, payment_lag,
        )
        self._base_ccy_idx: IborIndex = base_currency_index
        self._quote_ccy_idx: IborIndex = quote_currency_index
        self._is_fx_base_ccy_collateral_ccy: bool = is_fx_base_currency_collateral_currency
        self._is_basis_on_fx_base_ccy_leg: bool = is_basis_on_fx_base_currency_leg
        self._payment_frequency: Frequency = payment_frequency
        self._base_ccy_ibor_leg: list[CashFlow] = []
        self._quote_ccy_ibor_leg: list[CashFlow] = []
        self._initialize_dates()

    def _initialize_dates(self) -> None:
        # C++ parity: crosscurrencyratehelpers.cpp:292-300.
        today = _eval_date()
        self._base_ccy_ibor_leg = _build_floating_leg(
            today, self._tenor, self._fixing_days, self._calendar, self._convention,
            self._end_of_month, self._base_ccy_idx, self._payment_frequency,
        )
        self._quote_ccy_ibor_leg = _build_floating_leg(
            today, self._tenor, self._fixing_days, self._calendar, self._convention,
            self._end_of_month, self._quote_ccy_idx, self._payment_frequency,
        )
        self._initialize_dates_from_legs(
            self._base_ccy_ibor_leg, self._quote_ccy_ibor_leg
        )

    def _base_ccy_leg_discount_handle(self) -> YieldTermStructureProtocol:
        # C++ parity: crosscurrencyratehelpers.cpp:302-305.
        assert self._term_structure is not None
        return (
            self._collateral_handle
            if self._is_fx_base_ccy_collateral_ccy
            else self._term_structure
        )

    def _quote_ccy_leg_discount_handle(self) -> YieldTermStructureProtocol:
        # C++ parity: crosscurrencyratehelpers.cpp:307-310.
        assert self._term_structure is not None
        return (
            self._term_structure
            if self._is_fx_base_ccy_collateral_ccy
            else self._collateral_handle
        )


@final
class ConstNotionalCrossCurrencyBasisSwapRateHelper(CrossCurrencyBasisSwapRateHelperBase):
    """Constant-notional cross-currency basis swap rate helper.

    # C++ parity: ``class ConstNotionalCrossCurrencyBasisSwapRateHelper``
    # in crosscurrencyratehelpers.hpp:119-143.
    """

    def implied_quote(self) -> float:
        # C++ parity: crosscurrencyratehelpers.cpp:340-352.
        qassert.require(self._term_structure is not None, "term structure not set")
        assert self._initial_notional_exchange_date is not None
        assert self._final_notional_exchange_date is not None

        npv_base, bps_base = _npvbps_const_notional_leg(
            self._base_ccy_ibor_leg,
            self._initial_notional_exchange_date,
            self._final_notional_exchange_date,
            self._base_ccy_leg_discount_handle(),
        )
        npv_quote, bps_quote = _npvbps_const_notional_leg(
            self._quote_ccy_ibor_leg,
            self._initial_notional_exchange_date,
            self._final_notional_exchange_date,
            self._quote_ccy_leg_discount_handle(),
        )
        bps = -bps_base if self._is_basis_on_fx_base_ccy_leg else bps_quote
        qassert.require(abs(bps) > 0.0, "null BPS")
        return -(npv_quote - npv_base) / bps


@final
class MtMCrossCurrencyBasisSwapRateHelper(CrossCurrencyBasisSwapRateHelperBase):
    """Mark-to-market (resetting-notional) xccy basis swap rate helper.

    # C++ parity: ``class MtMCrossCurrencyBasisSwapRateHelper`` in
    # crosscurrencyratehelpers.hpp:158-184.
    """

    def __init__(
        self,
        basis: Quote | float,
        tenor: Period,
        fixing_days: int,
        calendar: Calendar,
        convention: BusinessDayConvention,
        end_of_month: bool,
        base_currency_index: IborIndex,
        quote_currency_index: IborIndex,
        collateral_curve: YieldTermStructureProtocol,
        is_fx_base_currency_collateral_currency: bool,
        is_basis_on_fx_base_currency_leg: bool,
        is_fx_base_currency_leg_resettable: bool,
        payment_frequency: Frequency = Frequency.NoFrequency,
        payment_lag: int = 0,
    ) -> None:
        self._is_fx_base_ccy_leg_resettable: bool = is_fx_base_currency_leg_resettable
        super().__init__(
            basis, tenor, fixing_days, calendar, convention, end_of_month,
            base_currency_index, quote_currency_index, collateral_curve,
            is_fx_base_currency_collateral_currency, is_basis_on_fx_base_currency_leg,
            payment_frequency, payment_lag,
        )

    def implied_quote(self) -> float:
        # C++ parity: crosscurrencyratehelpers.cpp:393-416.
        qassert.require(self._term_structure is not None, "term structure not set")
        assert self._initial_notional_exchange_date is not None
        assert self._final_notional_exchange_date is not None

        if self._is_fx_base_ccy_leg_resettable:
            npv_base, bps_base = _npvbps_resetting_leg(
                self._base_ccy_ibor_leg, self._payment_lag, self._calendar,
                self._convention, self._base_ccy_leg_discount_handle(),
                self._quote_ccy_leg_discount_handle(),
            )
            npv_quote, bps_quote = _npvbps_const_notional_leg(
                self._quote_ccy_ibor_leg, self._initial_notional_exchange_date,
                self._final_notional_exchange_date, self._quote_ccy_leg_discount_handle(),
            )
        else:
            npv_base, bps_base = _npvbps_const_notional_leg(
                self._base_ccy_ibor_leg, self._initial_notional_exchange_date,
                self._final_notional_exchange_date, self._base_ccy_leg_discount_handle(),
            )
            npv_quote, bps_quote = _npvbps_resetting_leg(
                self._quote_ccy_ibor_leg, self._payment_lag, self._calendar,
                self._convention, self._quote_ccy_leg_discount_handle(),
                self._base_ccy_leg_discount_handle(),
            )

        bps = -bps_base if self._is_basis_on_fx_base_ccy_leg else bps_quote
        qassert.require(abs(bps) > 0.0, "null BPS")
        return -(npv_quote - npv_base) / bps


@final
class ConstNotionalCrossCurrencySwapRateHelper(CrossCurrencySwapRateHelperBase):
    """Fixed-vs-floating par cross-currency swap rate helper.

    # C++ parity: ``class ConstNotionalCrossCurrencySwapRateHelper`` in
    # crosscurrencyratehelpers.hpp:196-227.
    """

    def __init__(
        self,
        fixed_rate: Quote | float,
        tenor: Period,
        fixing_days: int,
        calendar: Calendar,
        convention: BusinessDayConvention,
        end_of_month: bool,
        fixed_frequency: Frequency,
        fixed_day_count: DayCounter,
        float_index: IborIndex,
        collateral_curve: YieldTermStructureProtocol,
        collateral_on_fixed_leg: bool,
        payment_lag: int = 0,
    ) -> None:
        super().__init__(
            fixed_rate, tenor, fixing_days, calendar, convention, end_of_month,
            collateral_curve, payment_lag,
        )
        self._fixed_frequency: Frequency = fixed_frequency
        self._fixed_day_count: DayCounter = fixed_day_count
        self._float_index: IborIndex = float_index
        self._collateral_on_fixed_leg: bool = collateral_on_fixed_leg
        self._fixed_leg: list[CashFlow] = []
        self._float_leg: list[CashFlow] = []
        self._initialize_dates()

    def _initialize_dates(self) -> None:
        # C++ parity: crosscurrencyratehelpers.cpp:453-461.
        today = _eval_date()
        self._fixed_leg = _build_fixed_leg(
            today, self._tenor, self._fixing_days, self._calendar, self._convention,
            self._end_of_month, self._fixed_frequency, self._fixed_day_count,
        )
        self._float_leg = _build_floating_leg(
            today, self._tenor, self._fixing_days, self._float_index.fixing_calendar(),
            self._float_index.business_day_convention(), self._end_of_month,
            self._float_index, self._float_index.tenor().frequency(),
        )
        self._initialize_dates_from_legs(self._fixed_leg, self._float_leg)

    def _fixed_leg_discount_handle(self) -> YieldTermStructureProtocol:
        # C++ parity: crosscurrencyratehelpers.cpp:463-466.
        assert self._term_structure is not None
        return (
            self._collateral_handle
            if self._collateral_on_fixed_leg
            else self._term_structure
        )

    def _floating_leg_discount_handle(self) -> YieldTermStructureProtocol:
        # C++ parity: crosscurrencyratehelpers.cpp:468-471.
        assert self._term_structure is not None
        return (
            self._term_structure
            if self._collateral_on_fixed_leg
            else self._collateral_handle
        )

    def implied_quote(self) -> float:
        # C++ parity: crosscurrencyratehelpers.cpp:473-486.
        qassert.require(self._term_structure is not None, "term structure not set")
        assert self._initial_notional_exchange_date is not None
        assert self._final_notional_exchange_date is not None

        fixed_npv, fixed_bps = _npvbps_const_notional_leg(
            self._fixed_leg, self._initial_notional_exchange_date,
            self._final_notional_exchange_date, self._fixed_leg_discount_handle(),
        )
        float_npv, _float_bps = _npvbps_const_notional_leg(
            self._float_leg, self._initial_notional_exchange_date,
            self._final_notional_exchange_date, self._floating_leg_discount_handle(),
        )
        qassert.require(abs(fixed_bps) > 0.0, "null fixed-leg BPS")
        return _SAMPLE_FIXED_RATE + (float_npv - fixed_npv) / fixed_bps


__all__ = [
    "ConstNotionalCrossCurrencyBasisSwapRateHelper",
    "ConstNotionalCrossCurrencySwapRateHelper",
    "CrossCurrencyBasisSwapRateHelperBase",
    "CrossCurrencySwapRateHelperBase",
    "MtMCrossCurrencyBasisSwapRateHelper",
]

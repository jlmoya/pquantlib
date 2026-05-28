"""ZeroCouponInflationSwap — single-payment inflation-indexed swap.

# C++ parity: ql/instruments/zerocouponinflationswap.{hpp,cpp} (v1.42.1).

A ZCIIS exchanges two single cashflows at maturity:

* **Fixed leg** — pays ``N * ((1 + K)^T - 1)`` where ``K`` is the quoted
  fixed rate and ``T`` is the year-fraction from start to maturity.
* **Inflation leg** — pays ``N * (I(T) / I(0) - 1)`` where ``I(T)`` is the
  inflation-index observation at the maturity-lag date and ``I(0)`` is
  the base-date observation.

In this swap, ``Type`` (Payer / Receiver) refers to the *inflation* leg.

Cashflow seam: the inflation leg cashflow is a scaffold
``_ZeroInflationCashFlow`` built in the constructor. L7-C will land a
production ``ZeroInflationCashFlow``; this scaffold preserves the
discounted-NPV contract until then.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.simple_cash_flow import SimpleCashFlow
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.indexes.inflation.cpi import (
    InterpolationType,
    effective_interpolation_type,
)
from pquantlib.indexes.inflation.inflation_index import (
    ZeroInflationIndex,
    inflation_period,
)
from pquantlib.instruments.swap import Swap, SwapType
from pquantlib.pricingengines.pricing_engine import PricingEngineArguments
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.period import Period


class _ZeroInflationCashFlow(CashFlow):
    """Single cashflow paying ``nominal * (I(T) / I(0) - 1)`` at maturity.

    # C++ parity: ql/cashflows/zeroinflationcashflow.{hpp,cpp} (v1.42.1).

    L7-D scaffold: replaced by L7-C's production ``ZeroInflationCashFlow``
    at merge time. The contract is unchanged; tests can construct either.
    """

    def __init__(
        self,
        *,
        nominal: float,
        index: ZeroInflationIndex,
        observation_interpolation: InterpolationType,
        start_date: Date,
        end_date: Date,
        observation_lag: Period,
        payment_date: Date,
        growth_only: bool = True,
    ) -> None:
        super().__init__()
        self._nominal: float = nominal
        self._index: ZeroInflationIndex = index
        self._observation_interpolation: InterpolationType = observation_interpolation
        self._start_date: Date = start_date
        self._end_date: Date = end_date
        self._observation_lag: Period = observation_lag
        self._payment_date: Date = payment_date
        self._growth_only: bool = growth_only

        # # C++ parity: base/obs dates are observation_lag earlier than
        # # the contract start/end. The InterpolationType controls whether
        # # the actual fixing date is the period-start or the linearly
        # # interpolated date — for the AsIndex-→Flat default we bucket
        # # to the period start.
        self._base_date: Date = self._compute_observation_date(start_date)
        self._fixing_date: Date = self._compute_observation_date(end_date)

    def _compute_observation_date(self, contract_date: Date) -> Date:
        obs = contract_date - self._observation_lag
        if effective_interpolation_type(self._observation_interpolation) == InterpolationType.Linear:
            # Linear: the fixing date IS the lagged date (no bucketing).
            return obs
        start, _ = inflation_period(obs, self._index.frequency())
        return start

    def base_date(self) -> Date:
        return self._base_date

    def fixing_date(self) -> Date:
        return self._fixing_date

    def date(self) -> Date:
        return self._payment_date

    def notional(self) -> float:
        return self._nominal

    def amount(self) -> float:
        # # C++ parity: ZeroInflationCashFlow::amount() reads
        # # I_T = index.fixing(fixingDate), I_0 = index.fixing(baseDate),
        # # and returns N * ((I_T / I_0 - 1) if growthOnly else I_T / I_0).
        i_t = self._index.fixing(self._fixing_date)
        i_0 = self._index.fixing(self._base_date)
        ratio = i_t / i_0
        if self._growth_only:
            return self._nominal * (ratio - 1.0)
        return self._nominal * ratio


class ZeroCouponInflationSwapArguments(PricingEngineArguments):
    """Engine arguments carrier for ZCIIS.

    # C++ parity: ``ZeroCouponInflationSwap::arguments`` (zerocouponinflationswap.hpp:146-149).
    Carries the fixed rate on top of the inherited Swap::arguments.
    """

    def __init__(self) -> None:
        self.fixed_rate: float = math.nan
        # Inherit-on-demand: Swap.setup_arguments writes legs + payer
        # before the inflation-specific data via super().setup_arguments.
        self.legs: list[list[CashFlow]] = []
        self.payer: list[float] = []


class ZeroCouponInflationSwap(Swap):
    """Zero-coupon inflation-indexed swap.

    # C++ parity: ``ZeroCouponInflationSwap`` (zerocouponinflationswap.{hpp,cpp}).

    Constructor builds the two single-cashflow legs internally and
    populates the Swap-base ``_legs`` / ``_payer`` lists.
    """

    def __init__(
        self,
        *,
        type_: SwapType,
        nominal: float,
        start_date: Date,
        maturity: Date,
        fix_calendar: Calendar,
        fix_convention: BusinessDayConvention,
        day_counter: DayCounter,
        fixed_rate: float,
        index: ZeroInflationIndex,
        observation_lag: Period,
        observation_interpolation: InterpolationType = InterpolationType.AsIndex,
        adjust_inflation_obs_dates: bool = False,
        inf_calendar: Calendar | None = None,
        inf_convention: BusinessDayConvention | None = None,
    ) -> None:
        # # C++ parity: ZeroCouponInflationSwap::ZeroCouponInflationSwap
        # # (zerocouponinflationswap.cpp:35-115).
        super().__init__(n_legs=2)

        self._type: SwapType = type_
        self._nominal: float = nominal
        self._start_date: Date = start_date
        self._maturity_date: Date = maturity
        self._fix_calendar: Calendar = fix_calendar
        self._fix_convention: BusinessDayConvention = fix_convention
        self._day_counter: DayCounter = day_counter
        self._fixed_rate: float = fixed_rate
        self._index: ZeroInflationIndex = index
        self._observation_lag: Period = observation_lag
        self._observation_interpolation: InterpolationType = observation_interpolation
        self._adjust_inf_obs_dates: bool = adjust_inflation_obs_dates
        self._inf_calendar: Calendar = inf_calendar if inf_calendar is not None else fix_calendar
        self._inf_convention: BusinessDayConvention = (
            inf_convention if inf_convention is not None else fix_convention
        )

        # # C++ consistency check: when effective interpolation is Linear,
        # # the lag minus the index period must be at least the index lag;
        # # otherwise, just lag >= index availability lag.
        if effective_interpolation_type(observation_interpolation) == InterpolationType.Linear:
            # period subtraction not yet supported; conservative check.
            qassert.require(
                index.availability_lag().length <= observation_lag.length,
                "interpolated ZCIIS observation lag is shorter than index availability "
                f"lag (obsLag={observation_lag}, availabilityLag={index.availability_lag()})",
            )
        else:
            qassert.require(
                index.availability_lag().length <= observation_lag.length,
                "index tries to observe inflation fixings that do not yet exist: "
                f"availability lag {index.availability_lag()} > obs lag {observation_lag}",
            )

        inf_pay_date = self._inf_calendar.adjust(maturity, self._inf_convention)
        fixed_pay_date = self._fix_calendar.adjust(maturity, self._fix_convention)

        # # C++ parity: the inflation leg is a single ZeroInflationCashFlow
        # # with growthOnly=true (swaps exchange growth, not notionals).
        inflation_cf = _ZeroInflationCashFlow(
            nominal=nominal,
            index=index,
            observation_interpolation=observation_interpolation,
            start_date=start_date,
            end_date=maturity,
            observation_lag=observation_lag,
            payment_date=inf_pay_date,
            growth_only=True,
        )
        self._base_date: Date = inflation_cf.base_date()
        self._obs_date: Date = inflation_cf.fixing_date()

        # # Fixed leg: a SimpleCashFlow paying N * ((1+K)^T - 1) at the
        # # fixed-leg payment date.
        t = day_counter.year_fraction(start_date, maturity)
        fixed_amount = nominal * (math.pow(1.0 + fixed_rate, t) - 1.0)
        fixed_cf = SimpleCashFlow(fixed_amount, fixed_pay_date)

        self._legs = [[fixed_cf], [inflation_cf]]
        # # C++ parity: payer multipliers (zerocouponinflationswap.cpp:103-114).
        if type_ == SwapType.Payer:
            self._payer = [+1.0, -1.0]
        else:
            self._payer = [-1.0, +1.0]

        # Register the inflation cashflow (the fixed amount is static; the
        # inflation cashflow's amount depends on the index forecast curve).
        inflation_cf.register_with(self)

    # ---- inspectors --------------------------------------------------

    def type(self) -> SwapType:
        """``Payer``/``Receiver`` refers to the inflation leg."""
        return self._type

    def nominal(self) -> float:
        return self._nominal

    def start_date(self) -> Date:
        return self._start_date

    def maturity_date(self) -> Date:
        return self._maturity_date

    def fixed_calendar(self) -> Calendar:
        return self._fix_calendar

    def fixed_convention(self) -> BusinessDayConvention:
        return self._fix_convention

    def day_counter(self) -> DayCounter:
        return self._day_counter

    def fixed_rate(self) -> float:
        return self._fixed_rate

    def inflation_index(self) -> ZeroInflationIndex:
        return self._index

    def observation_lag(self) -> Period:
        return self._observation_lag

    def observation_interpolation(self) -> InterpolationType:
        return self._observation_interpolation

    def adjust_observation_dates(self) -> bool:
        return self._adjust_inf_obs_dates

    def inflation_calendar(self) -> Calendar:
        return self._inf_calendar

    def inflation_convention(self) -> BusinessDayConvention:
        return self._inf_convention

    def fixed_leg(self) -> list[CashFlow]:
        return self._legs[0]

    def inflation_leg(self) -> list[CashFlow]:
        return self._legs[1]

    # ---- results -----------------------------------------------------

    def fixed_leg_npv(self) -> float:
        self.calculate()
        qassert.require(self._leg_npv[0] is not None, "result not available")
        assert self._leg_npv[0] is not None
        return self._leg_npv[0]

    def inflation_leg_npv(self) -> float:
        self.calculate()
        qassert.require(self._leg_npv[1] is not None, "result not available")
        assert self._leg_npv[1] is not None
        return self._leg_npv[1]

    def fair_rate(self) -> float:
        """``K`` such that NPV is zero for this instrument given current curves.

        # C++ parity: ``ZeroCouponInflationSwap::fairRate``
        # (zerocouponinflationswap.cpp:118-138). Calls calculate() and then
        # derives from the inflation cashflow's growth amount:
        #   growth = inflation_cf.amount / nominal + 1.0
        #   fair_rate = growth^(1/T) - 1
        """
        self.calculate()
        inflation_cf = self._legs[1][0]
        growth = inflation_cf.amount() / self._nominal + 1.0
        t = self._day_counter.year_fraction(self._start_date, self._maturity_date)
        return math.pow(growth, 1.0 / t) - 1.0


__all__ = [
    "ZeroCouponInflationSwap",
    "ZeroCouponInflationSwapArguments",
]

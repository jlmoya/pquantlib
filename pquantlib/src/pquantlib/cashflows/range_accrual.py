"""Range-accrual coupons, their BGM smile pricer, and the leg builder.

# C++ parity: ql/cashflows/rangeaccrual.hpp + .cpp (v1.43).

A range-accrual coupon pays ``gearing * L + spread`` scaled by the fraction of
the observation dates on which the index fixing landed inside the
``[lowerTrigger, upperTrigger]`` band. ``RangeAccrualPricerByBgm`` values that
fraction as an average of digital ranges under the Brace-Gatarek-Musiela
lognormal-forward dynamics, with an optional smile treatment.

Ported here:

- :class:`RangeAccrualFloatersCoupon` — the coupon.
- :class:`RangeAccrualPricer` — the abstract pricer base (no ``swaplet_price``,
  exactly as in C++ where it is left pure-virtual).
- :class:`RangeAccrualPricerByBgm` — the concrete BGM pricer.
- :class:`RangeAccrualLeg` — the chained builder; C++'s ``operator Leg()``
  becomes :meth:`RangeAccrualLeg.build`.

# C++ parity divergences, all deliberate:
#
# - The two ``[[deprecated]]`` (v1.40) members are NOT ported: the coupon
#   constructor overload taking ``ext::shared_ptr<Schedule>`` and
#   ``observationsSchedule()``. Both are pure forwarders to the non-deprecated
#   ``Schedule`` / ``observationSchedule()`` forms that are ported here.
#
# - ``RangeAccrualLeg::operator Leg()`` in v1.43 (rangeaccrual.cpp:619-694)
#   opens with ``Leg leg(n);`` — n default-constructed, i.e. NULL,
#   ``shared_ptr<CashFlow>`` — and then ``push_back``s the n coupons, so the
#   returned Leg has 2n entries whose first n are null and crash anything that
#   dereferences them (``CashFlows::npv`` included). :meth:`RangeAccrualLeg.build`
#   returns the n coupons only. The probe pins ``raw_leg_size`` /
#   ``null_prefix`` so the upstream defect stays documented rather than
#   accidentally "fixed" in silence.
#
# - ``accept(AcyclicVisitor&)`` is not ported — PQuantLib ports no coupon
#   visitors (consistent with FloatingRateCoupon / CmsCoupon).
#
# - ``RangeAccrualPricer::observationTimeLags_`` is declared in C++ but never
#   written or read; it has no Python counterpart.
#
# - C++ leaves ``paymentDayCounter_`` default-constructed, which builds a leg
#   whose every coupon throws "no day counter implementation provided" the
#   first time it is used. :meth:`RangeAccrualLeg.build` requires the day
#   counter up front instead: same contract (the leg is unusable without one),
#   reported at the point where it can still be fixed. Falling back to the
#   index day counter is specifically avoided — that would silently produce
#   different accruals from C++.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows import cash_flow_vectors as cfv
from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.coupon_pricer import FloatingRateCouponPricer
from pquantlib.cashflows.fixed_rate_coupon import FixedRateCoupon
from pquantlib.cashflows.floating_rate_coupon import FloatingRateCoupon
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.math.distributions.normal_distribution import NormalDistribution
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.black_formula import black_formula
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pquantlib.termstructures.protocols import YieldTermStructureProtocol
    from pquantlib.termstructures.volatility.smile_section import SmileSection


# =======================================================================
#                       RangeAccrualFloatersCoupon
# =======================================================================


class RangeAccrualFloatersCoupon(FloatingRateCoupon):
    """Floating coupon accruing only while the index sits inside a band.

    # C++ parity: rangeaccrual.cpp:38-99.

    The constructor argument order mirrors C++ exactly. ``observation_schedule``
    must start and end on the coupon's accrual dates; its interior dates are the
    observation dates (the endpoints are dropped, as in C++).
    """

    def __init__(
        self,
        payment_date: Date,
        nominal: float,
        index: IborIndex,
        start_date: Date,
        end_date: Date,
        fixing_days: int,
        day_counter: DayCounter,
        gearing: float,
        spread: float,
        ref_period_start: Date | None,
        ref_period_end: Date | None,
        observation_schedule: Schedule,
        lower_trigger: float,
        upper_trigger: float,
    ) -> None:
        super().__init__(
            payment_date,
            nominal,
            start_date,
            end_date,
            fixing_days,
            index,
            gearing,
            spread,
            ref_period_start,
            ref_period_end,
            day_counter,
        )
        self._ibor_index: IborIndex = index
        self._observation_schedule: Schedule = observation_schedule
        self._lower_trigger: float = lower_trigger
        self._upper_trigger: float = upper_trigger

        qassert.require(lower_trigger < upper_trigger, "lowerTrigger_>=upperTrigger")
        qassert.require(
            observation_schedule.start_date == start_date, "incompatible start date"
        )
        qassert.require(observation_schedule.end_date == end_date, "incompatible end date")

        # C++ drops the schedule's end date then its start date; what is left
        # are the interior observation dates.
        self._observation_dates: tuple[Date, ...] = tuple(observation_schedule.dates[1:-1])
        self._observations_no: int = len(self._observation_dates)

        rate_curve = index.forecast_term_structure()
        qassert.require(
            rate_curve is not None,
            f"null term structure set to this instance of {index.name()}",
        )
        assert rate_curve is not None
        reference_date = rate_curve.reference_date()

        self._start_time: float = day_counter.year_fraction(reference_date, start_date)
        self._end_time: float = day_counter.year_fraction(reference_date, end_date)
        self._observation_times: tuple[float, ...] = tuple(
            day_counter.year_fraction(reference_date, d) for d in self._observation_dates
        )

    # --- inspectors ----------------------------------------------------

    def ibor_index(self) -> IborIndex:
        """The underlying index, narrowed to the C++ ``IborIndex`` type."""
        return self._ibor_index

    def start_time(self) -> float:
        """``S`` — year fraction from the curve reference date to accrual start."""
        return self._start_time

    def end_time(self) -> float:
        """``T`` — year fraction from the curve reference date to accrual end."""
        return self._end_time

    def lower_trigger(self) -> float:
        return self._lower_trigger

    def upper_trigger(self) -> float:
        return self._upper_trigger

    def observations_no(self) -> int:
        return self._observations_no

    def observation_dates(self) -> tuple[Date, ...]:
        return self._observation_dates

    def observation_times(self) -> tuple[float, ...]:
        return self._observation_times

    def observation_schedule(self) -> Schedule:
        return self._observation_schedule

    # --- pricing -------------------------------------------------------

    def price_without_optionality(
        self, discounting_curve: YieldTermStructureProtocol
    ) -> float:
        """Value of the coupon ignoring the range condition.

        # C++ parity: rangeaccrual.cpp:101-105.
        """
        return (
            self.accrual_period()
            * (self._gearing * self.index_fixing() + self._spread)
            * self.nominal()
            * discounting_curve.discount(self.date())
        )


# =======================================================================
#                            RangeAccrualPricer
# =======================================================================


class RangeAccrualPricer(FloatingRateCouponPricer):
    """Abstract base for range-accrual pricers.

    # C++ parity: rangeaccrual.cpp:111-166.

    Abstract for the same reason as C++: ``swaplet_price`` is left to the
    concrete pricer, and every cap/floor entry point fails.
    """

    def __init__(self) -> None:
        super().__init__()
        self._coupon: RangeAccrualFloatersCoupon | None = None
        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self._accrual_factor: float = 0.0
        self._observation_times: tuple[float, ...] = ()
        self._initial_values: tuple[float, ...] = ()
        self._observations_no: int = 0
        self._lower_trigger: float = 0.0
        self._upper_trigger: float = 0.0
        self._discount: float = 0.0
        self._gearing: float = 1.0
        self._spread: float = 0.0
        self._spread_leg_value: float = 0.0

    def initialize(self, coupon: FloatingRateCoupon) -> None:
        """Cache the per-coupon state the price methods need.

        # C++ parity: rangeaccrual.cpp:111-148.
        """
        qassert.require(
            isinstance(coupon, RangeAccrualFloatersCoupon), "range-accrual coupon required"
        )
        assert isinstance(coupon, RangeAccrualFloatersCoupon)
        self._coupon = coupon
        self._gearing = coupon.gearing()
        self._spread = coupon.spread()

        payment_date = coupon.date()

        index = coupon.ibor_index()
        rate_curve = index.forecast_term_structure()
        qassert.require(
            rate_curve is not None,
            f"null term structure set to this instance of {index.name()}",
        )
        assert rate_curve is not None
        self._discount = rate_curve.discount(payment_date)
        self._accrual_factor = coupon.accrual_period()
        self._spread_leg_value = self._spread * self._accrual_factor * self._discount

        self._start_time = coupon.start_time()
        self._end_time = coupon.end_time()
        self._observation_times = coupon.observation_times()
        self._lower_trigger = coupon.lower_trigger()
        self._upper_trigger = coupon.upper_trigger()
        self._observations_no = coupon.observations_no()

        observation_dates = coupon.observation_schedule().dates
        qassert.require(
            len(observation_dates) == self._observations_no + 2,
            "incompatible size of initialValues vector",
        )
        calendar = index.fixing_calendar()
        self._initial_values = tuple(
            index.fixing(calendar.advance(d, -coupon.fixing_days(), TimeUnit.Days))
            for d in observation_dates
        )

    def swaplet_rate(self) -> float:
        """``swaplet_price / (accrual_factor * discount)``.

        # C++ parity: rangeaccrual.cpp:150-152.
        """
        return self.swaplet_price() / (self._accrual_factor * self._discount)

    def caplet_price(self, effective_cap: float) -> float:
        del effective_cap
        qassert.fail("RangeAccrualPricer::capletPrice not implemented")

    def caplet_rate(self, effective_cap: float) -> float:
        del effective_cap
        qassert.fail("RangeAccrualPricer::capletRate not implemented")

    def floorlet_price(self, effective_floor: float) -> float:
        del effective_floor
        qassert.fail("RangeAccrualPricer::floorletPrice not implemented")

    def floorlet_rate(self, effective_floor: float) -> float:
        del effective_floor
        qassert.fail("RangeAccrualPricer::floorletRate not implemented")


# =======================================================================
#                         RangeAccrualPricerByBgm
# =======================================================================


class RangeAccrualPricerByBgm(RangeAccrualPricer):
    """Range-accrual pricer under Brace-Gatarek-Musiela forward dynamics.

    # C++ parity: rangeaccrual.cpp:171-524.

    ``with_smile`` selects the smile-aware digital price;
    ``by_call_spread`` then chooses between a call-spread replication and an
    analytic price plus a first-order smile correction. The two flags select
    genuinely different code paths, and ``by_call_spread`` is inert when
    ``with_smile`` is False — exactly as in C++.
    """

    #: Finite-difference step used to differentiate the smile and to width the
    #: call spread. C++ ``eps_`` (rangeaccrual.hpp:222).
    _EPS: float = 1.0e-8

    def __init__(
        self,
        correlation: float,
        smiles_on_expiry: SmileSection,
        smiles_on_payment: SmileSection,
        with_smile: bool,
        by_call_spread: bool,
    ) -> None:
        super().__init__()
        self._correlation: float = correlation
        self._with_smile: bool = with_smile
        self._by_call_spread: bool = by_call_spread
        self._smiles_on_expiry: SmileSection = smiles_on_expiry
        self._smiles_on_payment: SmileSection = smiles_on_payment

    # --- price ---------------------------------------------------------

    def swaplet_price(self) -> float:
        """Average digital-range value over the observation dates.

        # C++ parity: rangeaccrual.cpp:180-191.
        """
        result = 0.0
        deflator = self._discount * self._initial_values[0]
        for i in range(self._observations_no):
            result += self._digital_range_price(
                self._lower_trigger,
                self._upper_trigger,
                self._initial_values[i + 1],
                self._observation_times[i],
                deflator,
            )
        return (
            self._gearing * (result * self._accrual_factor / self._observations_no)
            + self._spread_leg_value
        )

    # --- drifts and volatilities ---------------------------------------

    def _drifts_over_period(
        self, u: float, lambda_s: float, lambda_t: float, correlation: float
    ) -> tuple[float, float]:
        """``(driftBeforeFixing, driftAfterFixing)``.

        # C++ parity: rangeaccrual.cpp:193-215 (returns a 2-vector).
        """
        p = (u - self._start_time) / self._accrual_factor
        q = (self._end_time - u) / self._accrual_factor
        l0t = self._initial_values[-1]
        lam = self._lambda(u, lambda_s, lambda_t)

        drift_before_fixing = (
            p
            * self._accrual_factor
            * l0t
            / (1.0 + l0t * self._accrual_factor)
            * (p * lambda_t * lambda_t + q * lambda_s * lambda_t * correlation)
            + q * lambda_s * lambda_s
            + p * lambda_s * lambda_t * correlation
            - 0.5 * lam * lam
        )
        drift_after_fixing = (
            p * self._accrual_factor * l0t / (1.0 + l0t * self._accrual_factor) - 0.5
        ) * lambda_t * lambda_t
        return (drift_before_fixing, drift_after_fixing)

    def _lambdas_over_period(
        self, u: float, lambda_s: float, lambda_t: float
    ) -> tuple[float, float]:
        """``(lambdaBeforeFixing, lambdaAfterFixing)``.

        # C++ parity: rangeaccrual.cpp:217-231.
        """
        p = (u - self._start_time) / self._accrual_factor
        q = (self._end_time - u) / self._accrual_factor
        return (q * lambda_s + p * lambda_t, lambda_t)

    def _drift(
        self, u: float, lambda_s: float, lambda_t: float, correlation: float
    ) -> float:
        """# C++ parity: rangeaccrual.cpp:232-252."""
        p = (u - self._start_time) / self._accrual_factor
        q = (self._end_time - u) / self._accrual_factor
        l0t = self._initial_values[-1]

        drift_before_fixing = (
            p
            * self._accrual_factor
            * l0t
            / (1.0 + l0t * self._accrual_factor)
            * (p * lambda_t * lambda_t + q * lambda_s * lambda_t * correlation)
            + q * lambda_s * lambda_s
            + p * lambda_s * lambda_t * correlation
        )
        drift_after_fixing = (
            p * self._accrual_factor * l0t / (1.0 + l0t * self._accrual_factor) - 0.5
        ) * lambda_t * lambda_t

        return drift_before_fixing if self._start_time > 0 else drift_after_fixing

    def _lambda(self, u: float, lambda_s: float, lambda_t: float) -> float:
        """# C++ parity: rangeaccrual.cpp:254-266."""
        p = (u - self._start_time) / self._accrual_factor
        q = (self._end_time - u) / self._accrual_factor
        return q * lambda_s + p * lambda_t if self._start_time > 0 else lambda_t

    def _der_drift_der_lambda_s(
        self, u: float, lambda_s: float, lambda_t: float, correlation: float
    ) -> float:
        """# C++ parity: rangeaccrual.cpp:269-288."""
        p = (u - self._start_time) / self._accrual_factor
        q = (self._end_time - u) / self._accrual_factor
        l0t = self._initial_values[-1]

        drift_before_fixing = (
            p
            * self._accrual_factor
            * l0t
            / (1.0 + l0t * self._accrual_factor)
            * (q * lambda_t * correlation)
            + 2 * q * lambda_s
            + p * lambda_t * correlation
        )
        drift_after_fixing = 0.0

        return drift_before_fixing if self._start_time > 0 else drift_after_fixing

    def _der_lambda_der_lambda_s(self, u: float) -> float:
        """# C++ parity: rangeaccrual.cpp:290-298."""
        if self._start_time > 0:
            return (self._end_time - u) / self._accrual_factor
        return 0.0

    def _der_drift_der_lambda_t(
        self, u: float, lambda_s: float, lambda_t: float, correlation: float
    ) -> float:
        """# C++ parity: rangeaccrual.cpp:300-319."""
        p = (u - self._start_time) / self._accrual_factor
        q = (self._end_time - u) / self._accrual_factor
        l0t = self._initial_values[-1]

        drift_before_fixing = (
            p
            * self._accrual_factor
            * l0t
            / (1.0 + l0t * self._accrual_factor)
            * (2 * p * lambda_t + q * lambda_s * correlation)
            + p * lambda_s * correlation
        )
        drift_after_fixing = (
            (p * self._accrual_factor * l0t / (1.0 + l0t * self._accrual_factor) - 0.5)
            * 2
            * lambda_t
        )

        return drift_before_fixing if self._start_time > 0 else drift_after_fixing

    def _der_lambda_der_lambda_t(self, u: float) -> float:
        """# C++ parity: rangeaccrual.cpp:321-329."""
        if self._start_time > 0:
            return (u - self._start_time) / self._accrual_factor
        return 0.0

    # --- digitals ------------------------------------------------------

    def _digital_range_price(
        self,
        lower_trigger: float,
        upper_trigger: float,
        initial_value: float,
        expiry: float,
        deflator: float,
    ) -> float:
        """# C++ parity: rangeaccrual.cpp:331-344."""
        lower_price = self._digital_price(lower_trigger, initial_value, expiry, deflator)
        upper_price = self._digital_price(upper_trigger, initial_value, expiry, deflator)
        result = lower_price - upper_price
        qassert.require(
            result >= 0.0,
            f"RangeAccrualPricerByBgm::digitalRangePrice:\n digitalPrice({upper_trigger}): "
            f"{upper_price} >  digitalPrice({lower_trigger}): {lower_price}",
        )
        return result

    def _digital_price(
        self, strike: float, initial_value: float, expiry: float, deflator: float
    ) -> float:
        """# C++ parity: rangeaccrual.cpp:345-357."""
        result = deflator
        if strike > self._EPS / 2:
            if self._with_smile:
                result = self._digital_price_with_smile(strike, initial_value, expiry, deflator)
            else:
                result = self._digital_price_without_smile(
                    strike, initial_value, expiry, deflator
                )
        return result

    def _digital_price_without_smile(
        self, strike: float, initial_value: float, expiry: float, deflator: float
    ) -> float:
        """Analytic lognormal digital under the BGM drift.

        # C++ parity: rangeaccrual.cpp:359-391.
        """
        lambda_s = self._smiles_on_expiry.volatility(strike)
        lambda_t = self._smiles_on_payment.volatility(strike)

        lambda_u = self._lambdas_over_period(expiry, lambda_s, lambda_t)
        variance = (
            self._start_time * lambda_u[0] * lambda_u[0]
            + (expiry - self._start_time) * lambda_u[1] * lambda_u[1]
        )

        lambda_s_atm = self._smiles_on_expiry.volatility(initial_value)
        lambda_t_atm = self._smiles_on_payment.volatility(initial_value)
        # drift of the lognormal (Libor) process — "a_U()" in the paper
        mu_u = self._drifts_over_period(expiry, lambda_s_atm, lambda_t_atm, self._correlation)
        adjustment = self._start_time * mu_u[0] + (expiry - self._start_time) * mu_u[1]

        d2 = (math.log(initial_value / strike) + adjustment - 0.5 * variance) / math.sqrt(
            variance
        )

        phi = CumulativeNormalDistribution()
        result = deflator * phi(d2)

        qassert.require(
            result > 0.0,
            f"RangeAccrualPricerByBgm::digitalPriceWithoutSmile: result< 0. Result:{result}",
        )
        qassert.require(
            result / deflator <= 1.0,
            "RangeAccrualPricerByBgm::digitalPriceWithoutSmile: result/deflator > 1. "
            f"Ratio: {result / deflator} result: {result} deflator: {deflator}",
        )
        return result

    def _digital_price_with_smile(
        self, strike: float, initial_value: float, expiry: float, deflator: float
    ) -> float:
        """Smile-aware digital: call-spread replication or analytic + correction.

        # C++ parity: rangeaccrual.cpp:393-450.
        """
        if self._by_call_spread:
            # Previous strike
            previous_strike = strike - self._EPS / 2
            lambda_s = self._smiles_on_expiry.volatility(previous_strike)
            lambda_t = self._smiles_on_payment.volatility(previous_strike)

            lambda_u = self._lambdas_over_period(expiry, lambda_s, lambda_t)
            previous_variance = (
                max(self._start_time, 0.0) * lambda_u[0] * lambda_u[0]
                + min(expiry - self._start_time, expiry) * lambda_u[1] * lambda_u[1]
            )

            lambda_s_atm = self._smiles_on_expiry.volatility(initial_value)
            lambda_t_atm = self._smiles_on_payment.volatility(initial_value)
            mu_u = self._drifts_over_period(
                expiry, lambda_s_atm, lambda_t_atm, self._correlation
            )
            # C++ recomputes ``muU`` verbatim for the "next" leg from the same
            # ATM inputs, so both adjustments are bit-identical; computed once.
            adjustment = math.exp(
                max(self._start_time, 0.0) * mu_u[0]
                + min(expiry - self._start_time, expiry) * mu_u[1]
            )
            previous_forward = initial_value * adjustment

            # Next strike
            next_strike = strike + self._EPS / 2
            lambda_s = self._smiles_on_expiry.volatility(next_strike)
            lambda_t = self._smiles_on_payment.volatility(next_strike)

            lambda_u = self._lambdas_over_period(expiry, lambda_s, lambda_t)
            next_variance = (
                max(self._start_time, 0.0) * lambda_u[0] * lambda_u[0]
                + min(expiry - self._start_time, expiry) * lambda_u[1] * lambda_u[1]
            )
            next_forward = initial_value * adjustment

            result = self._call_spread_price(
                previous_forward,
                next_forward,
                previous_strike,
                next_strike,
                deflator,
                previous_variance,
                next_variance,
            )
        else:
            result = self._digital_price_without_smile(
                strike, initial_value, expiry, deflator
            ) + self._smile_correction(strike, initial_value, expiry, deflator)

        qassert.require(
            result > -math.pow(self._EPS, 0.5),
            f"RangeAccrualPricerByBgm::digitalPriceWithSmile: result< 0 Result:{result}",
        )
        qassert.require(
            result / deflator <= 1.0 + math.pow(self._EPS, 0.2),
            "RangeAccrualPricerByBgm::digitalPriceWithSmile: result/deflator > 1. "
            f"Ratio: {result / deflator} result: {result} deflator: {deflator}",
        )
        return result

    def _smile_correction(
        self, strike: float, forward: float, expiry: float, deflator: float
    ) -> float:
        """First-order smile correction to the analytic digital.

        # C++ parity: rangeaccrual.cpp:452-503. The ``derDriftDerK`` term is
        # commented out upstream; it stays commented out here.
        """
        previous_strike = strike - self._EPS / 2
        next_strike = strike + self._EPS / 2

        der_smile_s = (
            self._smiles_on_expiry.volatility(next_strike)
            - self._smiles_on_expiry.volatility(previous_strike)
        ) / self._EPS
        der_smile_t = (
            self._smiles_on_payment.volatility(next_strike)
            - self._smiles_on_payment.volatility(previous_strike)
        ) / self._EPS

        lambda_s = self._smiles_on_expiry.volatility(strike)
        lambda_t = self._smiles_on_payment.volatility(strike)

        der_lambda_der_k = (
            self._der_lambda_der_lambda_s(expiry) * der_smile_s
            + self._der_lambda_der_lambda_t(expiry) * der_smile_t
        )

        lambda_s_atm = self._smiles_on_expiry.volatility(forward)
        lambda_t_atm = self._smiles_on_payment.volatility(forward)
        lambdas_over_period_u = self._lambdas_over_period(expiry, lambda_s, lambda_t)
        mu_u = self._drifts_over_period(expiry, lambda_s_atm, lambda_t_atm, self._correlation)

        variance = (
            max(self._start_time, 0.0) * lambdas_over_period_u[0] * lambdas_over_period_u[0]
            + min(expiry - self._start_time, expiry)
            * lambdas_over_period_u[1]
            * lambdas_over_period_u[1]
        )

        forward_adjustment = math.exp(
            max(self._start_time, 0.0) * mu_u[0]
            + min(expiry - self._start_time, expiry) * mu_u[1]
        )
        forward_adjusted = forward * forward_adjustment

        d1 = (math.log(forward_adjusted / strike) + 0.5 * variance) / math.sqrt(variance)

        sqrt_of_time_to_expiry = (
            max(self._start_time, 0.0) * lambdas_over_period_u[0]
            + min(expiry - self._start_time, expiry) * lambdas_over_period_u[1]
        ) * (1.0 / math.sqrt(variance))

        psi = NormalDistribution()
        result = -forward_adjusted * psi(d1) * sqrt_of_time_to_expiry * der_lambda_der_k
        result *= deflator

        qassert.require(
            abs(result / deflator) <= 1.0 + math.pow(self._EPS, 0.2),
            "RangeAccrualPricerByBgm::smileCorrection: abs(result/deflator) > 1. "
            f"Ratio: {result / deflator} result: {result} deflator: {deflator}",
        )
        return result

    def _call_spread_price(
        self,
        previous_forward: float,
        next_forward: float,
        previous_strike: float,
        next_strike: float,
        deflator: float,
        previous_variance: float,
        next_variance: float,
    ) -> float:
        """Digital replicated by a tight Black call spread.

        # C++ parity: rangeaccrual.cpp:505-524.
        """
        next_call = black_formula(
            OptionType.Call, next_strike, next_forward, math.sqrt(next_variance), deflator
        )
        previous_call = black_formula(
            OptionType.Call,
            previous_strike,
            previous_forward,
            math.sqrt(previous_variance),
            deflator,
        )
        qassert.require(
            next_call < previous_call,
            "RangeAccrualPricerByBgm::callSpreadPrice: nextCall > previousCall"
            f"\n nextCall: strike :{next_strike}; variance: {next_variance} "
            f"adjusted initial value {next_forward}"
            f"\n previousCall: strike :{previous_strike}; variance: {previous_variance} "
            f"adjusted initial value {previous_forward}",
        )
        return (previous_call - next_call) / (next_strike - previous_strike)


# =======================================================================
#                             RangeAccrualLeg
# =======================================================================


class RangeAccrualLeg:
    """Chained builder for a sequence of range-accrual coupons.

    # C++ parity: rangeaccrual.cpp:526-694 (``operator Leg()`` -> :meth:`build`).

    Each C++ scalar/vector ``withXxx`` overload pair collapses to one Python
    setter accepting either shape, matching :class:`~pquantlib.cashflows.ibor_coupon.IborLeg`.
    """

    def __init__(self, schedule: Schedule, index: IborIndex) -> None:
        self._schedule: Schedule = schedule
        self._index: IborIndex = index
        self._notionals: list[float] = []
        self._payment_day_counter: DayCounter | None = None
        self._payment_adjustment: BusinessDayConvention = BusinessDayConvention.Following
        self._fixing_days: list[int] = []
        self._gearings: list[float] = []
        self._spreads: list[float] = []
        self._lower_triggers: list[float] = []
        self._upper_triggers: list[float] = []
        self._observation_tenor: Period = Period()
        self._observation_convention: BusinessDayConvention = (
            BusinessDayConvention.ModifiedFollowing
        )

    # --- setters -------------------------------------------------------

    def with_notionals(self, notionals: float | Sequence[float]) -> RangeAccrualLeg:
        """# C++ parity: ``withNotionals`` (rangeaccrual.cpp:529-538)."""
        self._notionals = cfv.as_float_list(notionals)
        return self

    def with_payment_day_counter(self, day_counter: DayCounter) -> RangeAccrualLeg:
        """# C++ parity: ``withPaymentDayCounter`` (rangeaccrual.cpp:540-544)."""
        self._payment_day_counter = day_counter
        return self

    def with_payment_adjustment(
        self, convention: BusinessDayConvention
    ) -> RangeAccrualLeg:
        """# C++ parity: ``withPaymentAdjustment`` (rangeaccrual.cpp:546-550)."""
        self._payment_adjustment = convention
        return self

    def with_fixing_days(self, fixing_days: int | Sequence[int]) -> RangeAccrualLeg:
        """# C++ parity: ``withFixingDays`` (rangeaccrual.cpp:552-561)."""
        self._fixing_days = cfv.as_int_list(fixing_days)
        return self

    def with_gearings(self, gearings: float | Sequence[float]) -> RangeAccrualLeg:
        """# C++ parity: ``withGearings`` (rangeaccrual.cpp:563-572)."""
        self._gearings = cfv.as_float_list(gearings)
        return self

    def with_spreads(self, spreads: float | Sequence[float]) -> RangeAccrualLeg:
        """# C++ parity: ``withSpreads`` (rangeaccrual.cpp:574-583)."""
        self._spreads = cfv.as_float_list(spreads)
        return self

    def with_lower_triggers(self, triggers: float | Sequence[float]) -> RangeAccrualLeg:
        """# C++ parity: ``withLowerTriggers`` (rangeaccrual.cpp:585-594)."""
        self._lower_triggers = cfv.as_float_list(triggers)
        return self

    def with_upper_triggers(self, triggers: float | Sequence[float]) -> RangeAccrualLeg:
        """# C++ parity: ``withUpperTriggers`` (rangeaccrual.cpp:596-605)."""
        self._upper_triggers = cfv.as_float_list(triggers)
        return self

    def with_observation_tenor(self, tenor: Period) -> RangeAccrualLeg:
        """# C++ parity: ``withObservationTenor`` (rangeaccrual.cpp:607-611)."""
        self._observation_tenor = tenor
        return self

    def with_observation_convention(
        self, convention: BusinessDayConvention
    ) -> RangeAccrualLeg:
        """# C++ parity: ``withObservationConvention`` (rangeaccrual.cpp:613-617)."""
        self._observation_convention = convention
        return self

    # --- operator Leg() ------------------------------------------------

    def build(self) -> list[CashFlow]:
        """Build the leg.

        # C++ parity: ``RangeAccrualLeg::operator Leg()`` (rangeaccrual.cpp:619-694).

        Returns exactly one cash flow per schedule period: a
        :class:`RangeAccrualFloatersCoupon`, or — when the period's gearing is
        zero — a :class:`~pquantlib.cashflows.fixed_rate_coupon.FixedRateCoupon`
        paying the spread. See the module docstring for the leading-null-entries
        defect in the C++ original, which is not reproduced.
        """
        qassert.require(len(self._notionals) > 0, "no notional given")
        # C++ leaves this default-constructed and fails later, on first use.
        qassert.require(
            self._payment_day_counter is not None,
            "no payment day counter given",
        )
        assert self._payment_day_counter is not None
        day_counter = self._payment_day_counter

        n = len(self._schedule) - 1
        qassert.require(
            len(self._notionals) <= n,
            f"too many nominals ({len(self._notionals)}), only {n} required",
        )
        qassert.require(
            len(self._fixing_days) <= n,
            f"too many fixingDays ({len(self._fixing_days)}), only {n} required",
        )
        qassert.require(
            len(self._gearings) <= n,
            f"too many gearings ({len(self._gearings)}), only {n} required",
        )
        qassert.require(
            len(self._spreads) <= n,
            f"too many spreads ({len(self._spreads)}), only {n} required",
        )
        qassert.require(
            len(self._lower_triggers) <= n,
            f"too many lowerTriggers ({len(self._lower_triggers)}), only {n} required",
        )
        qassert.require(
            len(self._upper_triggers) <= n,
            f"too many upperTriggers ({len(self._upper_triggers)}), only {n} required",
        )

        # C++ comment: "the following is not always correct".
        calendar = self._schedule.calendar

        leg: list[CashFlow] = []
        for i in range(n):
            start = self._schedule.date(i)
            end = self._schedule.date(i + 1)
            ref_start, ref_end = start, end
            payment_date = calendar.adjust(end, self._payment_adjustment)
            if i == 0 and self._schedule.has_is_regular() and not self._schedule.is_regular_at(1):
                bdc = self._schedule.business_day_convention
                ref_start = calendar.adjust(end - self._schedule.tenor, bdc)
            if (
                i == n - 1
                and self._schedule.has_is_regular()
                and not self._schedule.is_regular_at(i + 1)
            ):
                bdc = self._schedule.business_day_convention
                ref_end = calendar.adjust(start + self._schedule.tenor, bdc)

            if cfv.get(self._gearings, i, 1.0) == 0.0:
                # fixed coupon
                leg.append(
                    FixedRateCoupon.from_rate(
                        payment_date,
                        cfv.get(self._notionals, i, math.nan),
                        cfv.get(self._spreads, i, 0.0),
                        day_counter,
                        start,
                        end,
                        ref_start,
                        ref_end,
                    )
                )
            else:
                # floating coupon
                observation_schedule = Schedule.from_rule(
                    start,
                    end,
                    self._observation_tenor,
                    calendar,
                    self._observation_convention,
                    self._observation_convention,
                    DateGeneration.Forward,
                    False,
                )
                leg.append(
                    RangeAccrualFloatersCoupon(
                        payment_date,
                        cfv.get(self._notionals, i, math.nan),
                        self._index,
                        start,
                        end,
                        cfv.get(self._fixing_days, i, 2),
                        day_counter,
                        cfv.get(self._gearings, i, 1.0),
                        cfv.get(self._spreads, i, 0.0),
                        ref_start,
                        ref_end,
                        observation_schedule,
                        # C++ passes Null<Rate>() when the triggers were never
                        # set; the coupon's lower < upper check then fails,
                        # which NaN reproduces.
                        cfv.get(self._lower_triggers, i, math.nan),
                        cfv.get(self._upper_triggers, i, math.nan),
                    )
                )
        return leg


__all__ = [
    "RangeAccrualFloatersCoupon",
    "RangeAccrualLeg",
    "RangeAccrualPricer",
    "RangeAccrualPricerByBgm",
]

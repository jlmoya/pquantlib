"""Conundrum / Hagan CMS-coupon replication pricers.

# C++ parity: ql/cashflows/conundrumpricer.hpp + .cpp (v1.42.1, 099987f0).

Hagan's "Conundrums..." static-replication CMS-coupon pricing. The module
ports:

* :class:`GFunction` (abstract) + the three concrete G-functions
  (:class:`GFunctionStandard`, :class:`GFunctionExactYield`,
  :class:`GFunctionWithShifts`) and the :class:`GFunctionFactory` that selects
  one by yield-curve model.
* :class:`VanillaOptionPricer` (abstract) + :class:`MarketQuotedOptionPricer`
  — the swaption price as a function of strike, used by the replication.
* :class:`HaganPricer` (abstract) + :class:`AnalyticHaganPricer` (closed form,
  Hagan 3.4c/3.5b/3.5c) + :class:`NumericHaganPricer` (numeric integration of
  the static replication via :class:`ConundrumIntegrand`).

Math-symbol variable names (G, q, n, x, Rs, AA, B, ...) follow the C++
source verbatim — flagged ``# noqa`` + ``# C++ parity:`` where they trip the
naming linter.

# C++ parity divergences:
# - GFunctionWithShifts.calibrationOfShift uses a Newton solve over an
#   ObjectiveFunction exposing ``derivative(x)``; ported faithfully using
#   PQuantLib's ``Newton`` solver (which reads ``f.derivative`` via the same
#   protocol).
# - NumericHaganPricer.integrate ports the C++ adaptive/non-adaptive
#   GaussKronrod cascade onto PQuantLib's GaussKronrodAdaptive +
#   scipy-backed semi-infinite handling; the LOOSE tolerance covers the
#   quadrature-rule difference (the canonical cms.cpp test only asserts
#   Numeric ≈ Analytic to 2e-4).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.cms_coupon import CmsCoupon
from pquantlib.cashflows.cms_coupon_pricer import CmsCouponPricer, MeanRevertingPricer
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.math.integrals.kronrod import GaussKronrodAdaptive
from pquantlib.math.solvers1d.newton import Newton
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.black_formula import (
    bachelier_black_formula,
    black_formula,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.time.period import Period

if TYPE_CHECKING:
    from collections.abc import Callable

    from pquantlib.cashflows.floating_rate_coupon import FloatingRateCoupon
    from pquantlib.quotes.quote import Quote
    from pquantlib.termstructures.volatility.swaption.swaption_volatility_structure import (
        SwaptionVolatilityStructure,
    )
    from pquantlib.time.date import Date


_CND = CumulativeNormalDistribution()


def _cumulative_normal(x: float) -> float:
    return _CND(x)


def _fixed_leg_accruals(swap: object) -> list[float]:
    """Accrual periods of a swap's fixed-leg coupons.

    C++ parity: ``dynamic_pointer_cast<Coupon>(fixedLeg[i])->accrualPeriod()``.
    """
    from pquantlib.cashflows.coupon import Coupon  # noqa: PLC0415

    accruals: list[float] = []
    for cf in swap.fixed_leg():  # type: ignore[attr-defined]
        qassert.require(isinstance(cf, Coupon), "fixed leg cash flow is not a Coupon")
        assert isinstance(cf, Coupon)
        accruals.append(cf.accrual_period())
    return accruals


# =====================================================================
#  VanillaOptionPricer
# =====================================================================


class VanillaOptionPricer(ABC):
    """Swaption price as a function of strike (replication kernel).

    C++ parity: conundrumpricer.hpp ``VanillaOptionPricer``.
    """

    @abstractmethod
    def __call__(
        self, strike: float, option_type: OptionType, deflator: float
    ) -> float: ...


class MarketQuotedOptionPricer(VanillaOptionPricer):
    """Black / Bachelier swaption price off a market swaption-vol smile.

    C++ parity: conundrumpricer.cpp ``MarketQuotedOptionPricer``.
    """

    def __init__(
        self,
        forward_value: float,
        expiry_date: Date,
        swap_tenor: Period | float,
        volatility_structure: SwaptionVolatilityStructure,
    ) -> None:
        self._forward_value = forward_value
        self._volatility_structure = volatility_structure
        self._smile = volatility_structure.smile_section(expiry_date, swap_tenor)
        qassert.require(
            volatility_structure.volatility_type() == VolatilityType.Normal
            or (
                volatility_structure.volatility_type() == VolatilityType.ShiftedLognormal
                and math.isclose(
                    volatility_structure.shift(expiry_date, swap_tenor), 0.0, abs_tol=1e-15
                )
            ),
            "VanillaOptionPricer: a normal or a zero-shift lognormal volatility is required",
        )

    def __call__(
        self, strike: float, option_type: OptionType, deflator: float
    ) -> float:
        variance = self._smile.variance(strike)
        if self._volatility_structure.volatility_type() == VolatilityType.ShiftedLognormal:
            return deflator * black_formula(
                option_type, strike, self._forward_value, math.sqrt(variance)
            )
        return deflator * bachelier_black_formula(
            option_type, strike, self._forward_value, math.sqrt(variance)
        )


# =====================================================================
#  GFunction hierarchy
# =====================================================================


class GFunction(ABC):
    """The G(x) function in the Hagan convexity-adjustment replication.

    C++ parity: conundrumpricer.hpp ``GFunction``.
    """

    @abstractmethod
    def __call__(self, x: float) -> float: ...

    @abstractmethod
    def first_derivative(self, x: float) -> float: ...

    @abstractmethod
    def second_derivative(self, x: float) -> float: ...


class GFunctionStandard(GFunction):
    """Standard yield-curve-model G-function.

    C++ parity: conundrumpricer.cpp ``GFunctionFactory::GFunctionStandard``.
    """

    def __init__(self, q: int, delta: float, swap_length: int) -> None:
        # C++ parity: q = periods/year, delta = pay-date fraction, swapLength.
        self._q = q
        self._delta = delta
        self._swap_length = swap_length

    def __call__(self, x: float) -> float:
        # C++ parity: conundrumpricer.cpp:588-592.
        q = self._q
        delta = self._delta
        n = float(self._swap_length) * q
        return (
            x
            / math.pow(1.0 + x / q, delta)
            * 1.0
            / (1.0 - 1.0 / math.pow(1.0 + x / q, n))
        )

    def first_derivative(self, x: float) -> float:
        # C++ parity: conundrumpricer.cpp:594-606.
        q = self._q
        delta = self._delta
        n = float(self._swap_length) * q
        a = 1.0 + x / q
        aa = a - delta / q * x
        b = math.pow(a, n - delta - 1.0) / (math.pow(a, n) - 1.0)

        sec_num = n * x * math.pow(a, n - 1.0)
        sec_den = q * math.pow(a, delta) * (math.pow(a, n) - 1.0) * (math.pow(a, n) - 1.0)
        sec = sec_num / sec_den

        return aa * b - sec

    def second_derivative(self, x: float) -> float:
        # C++ parity: conundrumpricer.cpp:608-629.
        q = self._q
        delta = self._delta
        n = float(self._swap_length) * q
        a = 1.0 + x / q
        aa = a - delta / q * x
        a1 = (1.0 - delta) / q
        b = math.pow(a, n - delta - 1.0) / (math.pow(a, n) - 1.0)
        num = (1.0 + delta - n) * math.pow(a, n - delta - 2.0) - (1.0 + delta) * math.pow(
            a, 2.0 * n - delta - 2.0
        )
        den = (math.pow(a, n) - 1.0) * (math.pow(a, n) - 1.0)
        b1 = 1.0 / q * num / den

        c = x / math.pow(a, delta)
        c1 = (math.pow(a, delta) - delta / q * x * math.pow(a, delta - 1.0)) / math.pow(
            a, 2 * delta
        )

        d = math.pow(a, n - 1.0) / ((math.pow(a, n) - 1.0) * (math.pow(a, n) - 1.0))
        d1 = (
            (n - 1.0) * math.pow(a, n - 2.0) * (math.pow(a, n) - 1.0)
            - 2 * n * math.pow(a, 2 * (n - 1.0))
        ) / (q * (math.pow(a, n) - 1.0) * (math.pow(a, n) - 1.0) * (math.pow(a, n) - 1.0))

        return a1 * b + aa * b1 - n / q * (c1 * d + c * d1)


class GFunctionExactYield(GFunction):
    """Exact-yield yield-curve-model G-function.

    C++ parity: conundrumpricer.cpp ``GFunctionFactory::GFunctionExactYield``.
    """

    def __init__(self, coupon: CmsCoupon) -> None:
        # C++ parity: conundrumpricer.cpp:640-670.
        swap_index = coupon.swap_index()
        swap = swap_index.underlying_swap(coupon.fixing_date())
        schedule = swap.fixed_schedule()
        rate_curve = swap_index.forwarding_term_structure()
        qassert.require(rate_curve is not None, "no forwarding term structure set")
        assert rate_curve is not None
        dc = swap_index.day_counter()

        swap_start_time = dc.year_fraction(rate_curve.reference_date(), schedule.start_date)
        swap_first_payment_time = dc.year_fraction(
            rate_curve.reference_date(), schedule.date(1)
        )
        payment_time = dc.year_fraction(rate_curve.reference_date(), coupon.date())

        self._delta = (payment_time - swap_start_time) / (
            swap_first_payment_time - swap_start_time
        )

        self._accruals: list[float] = _fixed_leg_accruals(swap)

    def __call__(self, x: float) -> float:
        # C++ parity: conundrumpricer.cpp:672-678.
        product = 1.0
        for accrual in self._accruals:
            product *= 1.0 / (1.0 + accrual * x)
        return x * math.pow(1.0 + self._accruals[0] * x, -self._delta) * (1.0 / (1.0 - product))

    def first_derivative(self, x: float) -> float:
        # C++ parity: conundrumpricer.cpp:680-699.
        delta = self._delta
        c = -1.0
        der_c = 0.0
        b: list[float] = []
        for accrual in self._accruals:
            temp = 1.0 / (1.0 + accrual * x)
            b.append(temp)
            c *= temp
            der_c += accrual * temp
        c += 1.0
        c = 1.0 / c
        der_c *= c - c * c

        return (
            -delta * self._accruals[0] * math.pow(b[0], delta + 1.0) * x * c
            + math.pow(b[0], delta) * c
            + math.pow(b[0], delta) * x * der_c
        )

    def second_derivative(self, x: float) -> float:
        # C++ parity: conundrumpricer.cpp:701-724.
        delta = self._delta
        c = -1.0
        s = 0.0
        sum_of_square = 0.0
        b: list[float] = []
        for accrual in self._accruals:
            temp = 1.0 / (1.0 + accrual * x)
            b.append(temp)
            c *= temp
            s += accrual * temp
            sum_of_square += math.pow(accrual * temp, 2.0)
        c += 1.0
        c = 1.0 / c
        der_c = s * (c - c * c)

        a0 = self._accruals[0]
        return (
            -delta * a0 * math.pow(b[0], delta + 1.0) * c + math.pow(b[0], delta) * der_c
        ) * (-delta * a0 * b[0] * x + 1.0 + x * (1.0 - c) * s) + math.pow(b[0], delta) * c * (
            delta * math.pow(a0 * b[0], 2.0) * x
            - delta * a0 * b[0]
            - x * der_c * s
            + (1.0 - c) * s
            - x * (1.0 - c) * sum_of_square
        )


class _GFunctionWithShiftsObjective:
    """Newton objective: f(x) = 0 calibrates the parallel/non-parallel shift.

    C++ parity: conundrumpricer.cpp ``GFunctionWithShifts::ObjectiveFunction``.

    The C++ inner class holds a (friend) reference to its owning
    ``GFunctionWithShifts`` and reads its member vectors; the Python port
    instead receives those vectors via :meth:`bind` once the owner has built
    them (avoids reaching into owner privates).
    """

    def __init__(self, rs: float) -> None:
        self._rs = rs
        self._derivative = 0.0
        self._accruals: list[float] = []
        self._swap_payment_discounts: list[float] = []
        self._shaped_swap_payment_times: list[float] = []
        self._discount_at_start = 0.0

    def bind(
        self,
        accruals: list[float],
        swap_payment_discounts: list[float],
        shaped_swap_payment_times: list[float],
        discount_at_start: float,
    ) -> None:
        self._accruals = accruals
        self._swap_payment_discounts = swap_payment_discounts
        self._shaped_swap_payment_times = shaped_swap_payment_times
        self._discount_at_start = discount_at_start

    def __call__(self, x: float) -> float:
        # C++ parity: conundrumpricer.cpp:902-919.
        result = 0.0
        self._derivative = 0.0
        for i in range(len(self._accruals)):
            temp = (
                self._accruals[i]
                * self._swap_payment_discounts[i]
                * math.exp(-self._shaped_swap_payment_times[i] * x)
            )
            result += temp
            self._derivative -= self._shaped_swap_payment_times[i] * temp
        result *= self._rs
        self._derivative *= self._rs
        temp = self._swap_payment_discounts[-1] * math.exp(
            -self._shaped_swap_payment_times[-1] * x
        )
        result += temp - self._discount_at_start
        self._derivative -= self._shaped_swap_payment_times[-1] * temp
        return result

    def derivative(self, x: float) -> float:
        del x
        return self._derivative

    def set_swap_rate_value(self, x: float) -> None:
        self._rs = x


class GFunctionWithShifts(GFunction):
    """Parallel- / non-parallel-shift yield-curve-model G-function.

    C++ parity: conundrumpricer.cpp ``GFunctionFactory::GFunctionWithShifts``.
    """

    def __init__(self, coupon: CmsCoupon, mean_reversion: Quote) -> None:
        # C++ parity: conundrumpricer.cpp:736-776.
        self._mean_reversion = mean_reversion
        self._calibrated_shift = 0.03
        self._tmp_rs = 10000000.0
        self._accuracy = 1.0e-14

        swap_index = coupon.swap_index()
        swap = swap_index.underlying_swap(coupon.fixing_date())
        self._swap_rate_value = swap.fair_rate()

        self._objective = _GFunctionWithShiftsObjective(self._swap_rate_value)

        schedule = swap.fixed_schedule()
        rate_curve = swap_index.forwarding_term_structure()
        qassert.require(rate_curve is not None, "no forwarding term structure set")
        assert rate_curve is not None
        dc = swap_index.day_counter()

        self._swap_start_time = dc.year_fraction(
            rate_curve.reference_date(), schedule.start_date
        )
        self._discount_at_start = rate_curve.discount(schedule.start_date)

        payment_time = dc.year_fraction(rate_curve.reference_date(), coupon.date())
        self._shaped_payment_time = self._shape_of_shift(payment_time)

        from pquantlib.cashflows.coupon import Coupon  # noqa: PLC0415

        self._accruals: list[float] = []
        self._shaped_swap_payment_times: list[float] = []
        self._swap_payment_discounts: list[float] = []
        for cpn in swap.fixed_leg():
            qassert.require(isinstance(cpn, Coupon), "fixed leg cash flow is not a Coupon")
            assert isinstance(cpn, Coupon)
            self._accruals.append(cpn.accrual_period())
            payment_date = cpn.date()
            swap_payment_time = dc.year_fraction(rate_curve.reference_date(), payment_date)
            self._shaped_swap_payment_times.append(self._shape_of_shift(swap_payment_time))
            self._swap_payment_discounts.append(rate_curve.discount(payment_date))
        self._discount_ratio = self._swap_payment_discounts[-1] / self._discount_at_start
        self._objective.bind(
            self._accruals,
            self._swap_payment_discounts,
            self._shaped_swap_payment_times,
            self._discount_at_start,
        )

    def _shape_of_shift(self, s: float) -> float:
        # C++ parity: conundrumpricer.cpp:929-938.
        x = s - self._swap_start_time
        mean_reversion = self._mean_reversion.value()
        if mean_reversion > 0:
            return (1.0 - math.exp(-mean_reversion * x)) / mean_reversion
        return x

    def _function_z(self, x: float) -> float:
        # C++ parity: conundrumpricer.cpp:783-786.
        return math.exp(-self._shaped_payment_time * x) / (
            1.0 - self._discount_ratio * math.exp(-self._shaped_swap_payment_times[-1] * x)
        )

    def _der_rs_der_x(self, x: float) -> float:
        # C++ parity: conundrumpricer.cpp:788-806.
        sqrt_denominator = 0.0
        der_sqrt_denominator = 0.0
        for i in range(len(self._accruals)):
            e = math.exp(-self._shaped_swap_payment_times[i] * x)
            sqrt_denominator += self._accruals[i] * self._swap_payment_discounts[i] * e
            der_sqrt_denominator -= (
                self._shaped_swap_payment_times[i]
                * self._accruals[i]
                * self._swap_payment_discounts[i]
                * e
            )
        denominator = sqrt_denominator * sqrt_denominator
        last_t = self._shaped_swap_payment_times[-1]
        last_d = self._swap_payment_discounts[-1]
        numerator = last_t * last_d * math.exp(-last_t * x) * sqrt_denominator
        numerator -= (
            self._discount_at_start - last_d * math.exp(-last_t * x)
        ) * der_sqrt_denominator
        qassert.require(denominator != 0, "GFunctionWithShifts::derRs_derX: denominator == 0")
        return numerator / denominator

    def _der2_rs_der_x2(self, x: float) -> float:
        # C++ parity: conundrumpricer.cpp:808-847.
        den_of_r = 0.0
        der_den_of_r = 0.0
        der2_den_of_r = 0.0
        for i in range(len(self._accruals)):
            t = self._shaped_swap_payment_times[i]
            ad = self._accruals[i] * self._swap_payment_discounts[i]
            e = math.exp(-t * x)
            den_of_r += ad * e
            der_den_of_r -= t * ad * e
            der2_den_of_r += t * t * ad * e

        denominator = math.pow(den_of_r, 4)
        last_t = self._shaped_swap_payment_times[-1]
        last_d = self._swap_payment_discounts[-1]
        e_last = math.exp(-last_t * x)

        num_of_der_r = last_t * last_d * e_last * den_of_r
        num_of_der_r -= (self._discount_at_start - last_d * e_last) * der_den_of_r
        den_of_der_r = math.pow(den_of_r, 2)

        der_num_of_der_r = -last_t * last_t * last_d * e_last * den_of_r
        der_num_of_der_r += last_t * last_d * e_last * der_den_of_r
        der_num_of_der_r -= (last_t * last_d * e_last) * der_den_of_r
        der_num_of_der_r -= (self._discount_at_start - last_d * e_last) * der2_den_of_r

        der_den_of_der_r = 2 * den_of_r * der_den_of_r
        numerator = der_num_of_der_r * den_of_der_r - num_of_der_r * der_den_of_der_r
        qassert.require(
            denominator != 0, "GFunctionWithShifts::der2Rs_derX2: denominator == 0"
        )
        return numerator / denominator

    def _der_z_der_x(self, x: float) -> float:
        # C++ parity: conundrumpricer.cpp:849-859.
        last_t = self._shaped_swap_payment_times[-1]
        sqrt_denominator = 1.0 - self._discount_ratio * math.exp(-last_t * x)
        denominator = sqrt_denominator * sqrt_denominator
        qassert.require(denominator != 0, "GFunctionWithShifts::derZ_derX: denominator == 0")
        spt = self._shaped_payment_time
        numerator = -spt * math.exp(-spt * x) * sqrt_denominator
        numerator -= last_t * math.exp(-spt * x) * (1.0 - sqrt_denominator)
        return numerator / denominator

    def _der2_z_der_x2(self, x: float) -> float:
        # C++ parity: conundrumpricer.cpp:861-883.
        last_t = self._shaped_swap_payment_times[-1]
        dr = self._discount_ratio
        spt = self._shaped_payment_time
        den_of_z = 1.0 - dr * math.exp(-last_t * x)
        der_den_of_z = last_t * dr * math.exp(-last_t * x)
        denominator = math.pow(den_of_z, 4)
        qassert.require(denominator != 0, "GFunctionWithShifts::der2Z_derX2: denominator == 0")

        num_of_der_z = -spt * math.exp(-spt * x) * den_of_z
        num_of_der_z -= last_t * math.exp(-spt * x) * (1.0 - den_of_z)
        den_of_der_z = math.pow(den_of_z, 2)
        der_num_of_der_z = -spt * math.exp(-spt * x) * (
            -spt + (spt * dr - last_t * dr) * math.exp(-last_t * x)
        ) - last_t * math.exp(-spt * x) * (spt * dr - last_t * dr) * math.exp(-last_t * x)
        der_den_of_der_z = 2 * den_of_z * der_den_of_z
        numerator = der_num_of_der_z * den_of_der_z - num_of_der_z * der_den_of_der_z
        return numerator / denominator

    def __call__(self, x: float) -> float:
        # C++ parity: conundrumpricer.cpp:778-781 (x == Rs, the swap rate).
        rs = x
        calibrated_shift = self._calibration_of_shift(rs)
        return rs * self._function_z(calibrated_shift)

    def first_derivative(self, x: float) -> float:
        # C++ parity: conundrumpricer.cpp:885-890 (x == Rs).
        rs = x
        calibrated_shift = self._calibration_of_shift(rs)
        return self._function_z(calibrated_shift) + rs * self._der_z_der_x(
            calibrated_shift
        ) / self._der_rs_der_x(calibrated_shift)

    def second_derivative(self, x: float) -> float:
        # C++ parity: conundrumpricer.cpp:892-900 (x == Rs).
        rs = x
        calibrated_shift = self._calibration_of_shift(rs)
        der_rs = self._der_rs_der_x(calibrated_shift)
        return (
            2.0 * self._der_z_der_x(calibrated_shift) / der_rs
            + rs * self._der2_z_der_x2(calibrated_shift) / math.pow(der_rs, 2.0)
            - rs
            * self._der_z_der_x(calibrated_shift)
            * self._der2_rs_der_x2(calibrated_shift)
            / math.pow(der_rs, 3.0)
        )

    def _calibration_of_shift(self, rs: float) -> float:
        # C++ parity: conundrumpricer.cpp:940-980.
        if rs != self._tmp_rs:
            n = 0.0
            d = 0.0
            for i in range(len(self._accruals)):
                ad = self._accruals[i] * self._swap_payment_discounts[i]
                n += ad
                d += ad * self._shaped_swap_payment_times[i]
            n *= rs
            d *= rs
            n += (
                self._accruals[-1] * self._swap_payment_discounts[-1]
                - self._discount_at_start
            )
            d += (
                self._accruals[-1]
                * self._swap_payment_discounts[-1]
                * self._shaped_swap_payment_times[-1]
            )
            initial_guess = n / d

            self._objective.set_swap_rate_value(rs)
            solver = Newton()
            solver.set_max_evaluations(1000)

            lower = -20.0
            upper = 20.0
            try:
                self._calibrated_shift = solver.solve(
                    self._objective,
                    self._accuracy,
                    max(min(initial_guess, upper * 0.99), lower * 0.99),
                    lower,
                    upper,
                )
            except Exception as exc:
                qassert.fail(
                    f"meanReversion: {self._mean_reversion.value()}, "
                    f"swapRateValue: {self._swap_rate_value}, "
                    f"swapStartTime: {self._swap_start_time}, "
                    f"shapedPaymentTime: {self._shaped_payment_time}\n"
                    f" error message: {exc}"
                )
            self._tmp_rs = rs
        return self._calibrated_shift


class YieldCurveModel:
    """Enumerated yield-curve models for ``GFunctionFactory``.

    C++ parity: conundrumpricer.hpp ``GFunctionFactory::YieldCurveModel``.
    """

    Standard = 0
    ExactYield = 1
    ParallelShifts = 2
    NonParallelShifts = 3


class GFunctionFactory:
    """Factory selecting a G-function by yield-curve model.

    C++ parity: conundrumpricer.hpp/.cpp ``GFunctionFactory``.
    """

    @staticmethod
    def new_g_function_standard(q: int, delta: float, swap_length: int) -> GFunction:
        return GFunctionStandard(q, delta, swap_length)

    @staticmethod
    def new_g_function_exact_yield(coupon: CmsCoupon) -> GFunction:
        return GFunctionExactYield(coupon)

    @staticmethod
    def new_g_function_with_shifts(coupon: CmsCoupon, mean_reversion: Quote) -> GFunction:
        return GFunctionWithShifts(coupon, mean_reversion)


# =====================================================================
#  HaganPricer hierarchy
# =====================================================================

_MAX_REAL = 1.7976931348623157e308  # QL_MAX_REAL


class HaganPricer(CmsCouponPricer, MeanRevertingPricer):
    """Hagan CMS-coupon pricer base (convexity via static replication).

    C++ parity: conundrumpricer.hpp/.cpp ``HaganPricer``.
    """

    def __init__(
        self,
        swaption_vol: SwaptionVolatilityStructure | None,
        model_of_yield_curve: int,
        mean_reversion: Quote,
    ) -> None:
        super().__init__(swaption_vol)
        self._model_of_yield_curve = model_of_yield_curve
        self._mean_reversion_quote = mean_reversion
        # populated by initialize()
        self._coupon: CmsCoupon | None = None
        self._rate_curve: object = None
        self._g_function: GFunction | None = None
        self._payment_date: Date | None = None
        self._fixing_date: Date | None = None
        self._swap_rate_value = 0.0
        self._discount = 1.0
        self._annuity = 0.0
        self._gearing = 1.0
        self._spread = 0.0
        self._spread_leg_value = 0.0
        self._cutoff_for_caplet = 2.0
        self._cutoff_for_floorlet = 0.0
        self._swap_tenor: Period | None = None
        self._vanilla_option_pricer: VanillaOptionPricer | None = None

    # --- MeanRevertingPricer -------------------------------------------

    def mean_reversion(self) -> float:
        return self._mean_reversion_quote.value()

    def set_mean_reversion(self, mean_reversion: Quote) -> None:
        self._mean_reversion_quote = mean_reversion
        self.update()

    # --- abstract optionlet hook ---------------------------------------

    @abstractmethod
    def _optionlet_price(self, option_type: OptionType, strike: float) -> float: ...

    @abstractmethod
    def swaplet_price(self) -> float: ...

    # --- initialization ------------------------------------------------

    def initialize(self, coupon: FloatingRateCoupon) -> None:
        # C++ parity: conundrumpricer.cpp:86-150.
        qassert.require(isinstance(coupon, CmsCoupon), "CMS coupon needed")
        assert isinstance(coupon, CmsCoupon)
        self._coupon = coupon
        self._gearing = coupon.gearing()
        self._spread = coupon.spread()
        accrual_period = coupon.accrual_period()
        qassert.require(accrual_period != 0.0, "null accrual period")

        self._fixing_date = coupon.fixing_date()
        self._payment_date = coupon.date()
        swap_index = coupon.swap_index()
        disc_ts = swap_index.discounting_term_structure()
        self._rate_curve = (
            disc_ts if disc_ts is not None else swap_index.forwarding_term_structure()
        )

        today = ObservableSettings().evaluation_date_or_today()

        rate_curve = self._rate_curve
        assert rate_curve is not None
        if self._payment_date > today:
            self._discount = rate_curve.discount(self._payment_date)
        else:
            self._discount = 1.0

        self._spread_leg_value = self._spread * accrual_period * self._discount

        if self._fixing_date > today:
            self._swap_tenor = swap_index.tenor()
            swap = swap_index.underlying_swap(self._fixing_date)
            self._swap_rate_value = swap.fair_rate()

            bp = 1.0e-4
            self._annuity = abs(swap.fixed_leg_bps() / bp)

            q = int(swap_index.fixed_leg_tenor().frequency())
            schedule = swap.fixed_schedule()
            dc = swap_index.day_counter()
            start_time = dc.year_fraction(rate_curve.reference_date(), swap.start_date())
            swap_first_payment_time = dc.year_fraction(
                rate_curve.reference_date(), schedule.date(1)
            )
            payment_time = dc.year_fraction(rate_curve.reference_date(), self._payment_date)
            delta = (payment_time - start_time) / (swap_first_payment_time - start_time)

            model = self._model_of_yield_curve
            if model == YieldCurveModel.Standard:
                self._g_function = GFunctionFactory.new_g_function_standard(
                    q, delta, self._swap_tenor.length
                )
            elif model == YieldCurveModel.ExactYield:
                self._g_function = GFunctionFactory.new_g_function_exact_yield(coupon)
            elif model == YieldCurveModel.ParallelShifts:
                self._g_function = GFunctionFactory.new_g_function_with_shifts(
                    coupon, SimpleQuote(0.0)
                )
            elif model == YieldCurveModel.NonParallelShifts:
                self._g_function = GFunctionFactory.new_g_function_with_shifts(
                    coupon, self._mean_reversion_quote
                )
            else:
                qassert.fail("unknown/illegal gFunction type")

            vol = self.swaption_volatility()
            assert vol is not None
            self._vanilla_option_pricer = MarketQuotedOptionPricer(
                self._swap_rate_value, self._fixing_date, self._swap_tenor, vol
            )

    # --- CouponPricer interface ----------------------------------------

    def swaplet_rate(self) -> float:
        # C++ parity: conundrumpricer.cpp:154-156.
        assert self._coupon is not None
        return self.swaplet_price() / (self._coupon.accrual_period() * self._discount)

    def caplet_price(self, effective_cap: float) -> float:
        # C++ parity: conundrumpricer.cpp:158-186.
        assert self._coupon is not None
        assert self._fixing_date is not None
        today = ObservableSettings().evaluation_date_or_today()
        if self._fixing_date <= today:
            rs = max(
                self._coupon.swap_index().fixing(self._fixing_date) - effective_cap, 0.0
            )
            return (self._gearing * rs) * (self._coupon.accrual_period() * self._discount)
        caplet_price = 0.0
        vol = self.swaption_volatility()
        assert vol is not None
        if vol.volatility_type() == VolatilityType.ShiftedLognormal:
            cutoff_near_zero = 1e-10
            if effective_cap < self._cutoff_for_caplet:
                eff_strike = max(effective_cap, cutoff_near_zero)
                caplet_price = self._optionlet_price(OptionType.Call, eff_strike)
        else:
            caplet_price = self._optionlet_price(OptionType.Call, effective_cap)
        return self._gearing * caplet_price

    def caplet_rate(self, effective_cap: float) -> float:
        assert self._coupon is not None
        return self.caplet_price(effective_cap) / (
            self._coupon.accrual_period() * self._discount
        )

    def floorlet_price(self, effective_floor: float) -> float:
        # C++ parity: conundrumpricer.cpp:192-220.
        assert self._coupon is not None
        assert self._fixing_date is not None
        today = ObservableSettings().evaluation_date_or_today()
        if self._fixing_date <= today:
            rs = max(
                effective_floor - self._coupon.swap_index().fixing(self._fixing_date), 0.0
            )
            return (self._gearing * rs) * (self._coupon.accrual_period() * self._discount)
        floorlet_price = 0.0
        vol = self.swaption_volatility()
        assert vol is not None
        if vol.volatility_type() == VolatilityType.ShiftedLognormal:
            cutoff_near_zero = 1e-10
            if effective_floor > self._cutoff_for_floorlet:
                eff_strike = max(effective_floor, cutoff_near_zero)
                floorlet_price = self._optionlet_price(OptionType.Put, eff_strike)
        else:
            floorlet_price = self._optionlet_price(OptionType.Put, effective_floor)
        return self._gearing * floorlet_price

    def floorlet_rate(self, effective_floor: float) -> float:
        assert self._coupon is not None
        return self.floorlet_price(effective_floor) / (
            self._coupon.accrual_period() * self._discount
        )


# =====================================================================
#  ConundrumIntegrand
# =====================================================================


class ConundrumIntegrand:
    """Integrand for the numeric Hagan static replication.

    C++ parity: conundrumpricer.hpp/.cpp
    ``NumericHaganPricer::ConundrumIntegrand``.
    """

    def __init__(
        self,
        vanilla_option_pricer: VanillaOptionPricer,
        rate_curve: object,
        g_function: GFunction,
        fixing_date: Date,
        payment_date: Date,
        annuity: float,
        forward_value: float,
        strike: float,
        option_type: OptionType,
    ) -> None:
        del rate_curve  # C++ ignores the rateCurve arg in the ctor body.
        self._vanilla_option_pricer = vanilla_option_pricer
        self._forward_value = forward_value
        self._annuity = annuity
        self._fixing_date = fixing_date
        self._payment_date = payment_date
        self._strike = strike
        self._option_type = option_type
        self._g_function = g_function

    def set_strike(self, strike: float) -> None:
        self._strike = strike

    def strike(self) -> float:
        return self._strike

    def annuity(self) -> float:
        return self._annuity

    def fixing_date(self) -> Date:
        return self._fixing_date

    def function_f(self, x: float) -> float:
        # C++ parity: conundrumpricer.cpp:472-476.
        gx = self._g_function(x)
        gr = self._g_function(self._forward_value)
        return (x - self._strike) * (gx / gr - 1.0)

    def first_derivative_of_f(self, x: float) -> float:
        # C++ parity: conundrumpricer.cpp:478-483.
        gx = self._g_function(x)
        gr = self._g_function(self._forward_value)
        g1 = self._g_function.first_derivative(x)
        return (gx / gr - 1.0) + g1 / gr * (x - self._strike)

    def second_derivative_of_f(self, x: float) -> float:
        # C++ parity: conundrumpricer.cpp:485-490.
        gr = self._g_function(self._forward_value)
        g1 = self._g_function.first_derivative(x)
        g2 = self._g_function.second_derivative(x)
        return 2.0 * g1 / gr + (x - self._strike) * g2 / gr

    def __call__(self, x: float) -> float:
        # C++ parity: conundrumpricer.cpp:492-495.
        option = self._vanilla_option_pricer(x, self._option_type, self._annuity)
        return option * self.second_derivative_of_f(x)


# =====================================================================
#  NumericHaganPricer
# =====================================================================


class _VariableChange:
    """Change of variable x -> a + (b-a)*t^k for semi-infinite integration.

    C++ parity: conundrumpricer.cpp anonymous-namespace ``VariableChange``.
    """

    def __init__(self, f: Callable[[float], float], a: float, b: float, k: int) -> None:
        self._a = a
        self._width = b - a
        self._f = f
        self._k = k

    def value(self, x: float) -> float:
        temp = self._width
        for _ in range(1, self._k):
            temp *= x
        new_var = self._a + x * temp
        return self._f(new_var) * self._k * temp


class NumericHaganPricer(HaganPricer):
    """Numeric-integration Hagan CMS-coupon pricer.

    C++ parity: conundrumpricer.hpp/.cpp ``NumericHaganPricer``.

    # C++ parity divergence: the C++ adaptive/non-adaptive GaussKronrod
    # cascade (GaussKronrodNonAdaptive for semi-infinite, falling back to
    # GaussKronrodAdaptive) is simplified to PQuantLib's GaussKronrodAdaptive
    # (the non-adaptive variant is a deferred carve-out). The semi-infinite
    # upper boundary is estimated as in C++ and integrated with the
    # x -> a + (b-a)*t^3 change of variable. The LOOSE tolerance + the
    # cms.cpp Numeric ≈ Analytic 2e-4 check cover the quadrature difference.
    """

    def __init__(
        self,
        swaption_vol: SwaptionVolatilityStructure | None,
        model_of_yield_curve: int,
        mean_reversion: Quote,
        lower_limit: float = 0.0,
        upper_limit: float = 1.0,
        precision: float = 1.0e-6,
        hard_upper_limit: float = _MAX_REAL,
    ) -> None:
        super().__init__(swaption_vol, model_of_yield_curve, mean_reversion)
        self._lower_limit = lower_limit
        self._upper_limit = upper_limit
        self._precision = precision
        self._hard_upper_limit = hard_upper_limit
        self._required_std_deviations = 8.0
        self._std_deviations_for_upper_limit = 0.0
        self._std_deviations_for_lower_limit = 0.0

    def upper_limit(self) -> float:
        return self._upper_limit

    def lower_limit(self) -> float:
        return self._lower_limit

    def std_deviations(self) -> float:
        return self._std_deviations_for_upper_limit

    def _integrate(self, a: float, b: float, integrand: ConundrumIntegrand) -> float:
        # C++ parity: conundrumpricer.cpp:281-343.
        result = 0.0
        if a > 0:
            upper_boundary = 2 * a
            while integrand(upper_boundary) > self._precision:
                upper_boundary *= 2.0
            if b > a:
                upper_boundary = min(upper_boundary, b)

            upper_boundary = max(a, min(upper_boundary, self._hard_upper_limit))
            if upper_boundary > 2 * a:
                k = 3
                variable_change = _VariableChange(integrand, a, upper_boundary, k)
                integrator = GaussKronrodAdaptive(self._precision, 100000)
                result = integrator(variable_change.value, 0.0, 1.0)
            else:
                integrator = GaussKronrodAdaptive(self._precision, 100000)
                result = integrator(integrand, a, upper_boundary)
        else:
            b = max(a, min(b, self._hard_upper_limit))
            integrator = GaussKronrodAdaptive(self._precision, 100000)
            result = integrator(integrand, a, b)
        return result

    def _optionlet_price(self, option_type: OptionType, strike: float) -> float:
        # C++ parity: conundrumpricer.cpp:346-378.
        assert self._g_function is not None
        assert self._vanilla_option_pricer is not None
        assert self._fixing_date is not None
        assert self._payment_date is not None
        integrand = ConundrumIntegrand(
            self._vanilla_option_pricer,
            self._rate_curve,
            self._g_function,
            self._fixing_date,
            self._payment_date,
            self._annuity,
            self._swap_rate_value,
            strike,
            option_type,
        )
        self._std_deviations_for_upper_limit = self._required_std_deviations
        self._std_deviations_for_lower_limit = self._required_std_deviations
        if option_type == OptionType.Call:
            self._upper_limit = self._reset_upper_limit(self._std_deviations_for_upper_limit)
            integral_value = self._integrate(strike, self._upper_limit, integrand)
        else:
            self._lower_limit = self._reset_lower_limit(self._std_deviations_for_lower_limit)
            a = min(strike, self._lower_limit)
            b = strike
            integral_value = self._integrate(a, b, integrand)

        d_f_dk = integrand.first_derivative_of_f(strike)
        swaption_price = self._vanilla_option_pricer(strike, option_type, self._annuity)

        assert self._coupon is not None
        # Hagan, Conundrums..., 2.17a, 2.18a
        return (
            self._coupon.accrual_period()
            * (self._discount / self._annuity)
            * ((1 + d_f_dk) * swaption_price + int(option_type) * integral_value)
        )

    def swaplet_price(self) -> float:
        # C++ parity: conundrumpricer.cpp:380-395.
        assert self._coupon is not None
        assert self._fixing_date is not None
        today = ObservableSettings().evaluation_date_or_today()
        if self._fixing_date <= today:
            rs = self._coupon.swap_index().fixing(self._fixing_date)
            return (self._gearing * rs + self._spread) * (
                self._coupon.accrual_period() * self._discount
            )
        atm_caplet_price = self._optionlet_price(OptionType.Call, self._swap_rate_value)
        atm_floorlet_price = self._optionlet_price(OptionType.Put, self._swap_rate_value)
        return (
            self._gearing
            * (
                self._coupon.accrual_period() * self._discount * self._swap_rate_value
                + atm_caplet_price
                - atm_floorlet_price
            )
            + self._spread_leg_value
        )

    def _reset_upper_limit(self, std_deviations_for_upper_limit: float) -> float:
        # C++ parity: conundrumpricer.cpp:411-422.
        vol = self.swaption_volatility()
        assert vol is not None
        assert self._fixing_date is not None
        assert self._swap_tenor is not None
        variance = vol.black_variance(
            self._fixing_date, self._swap_tenor, self._swap_rate_value
        )
        if vol.volatility_type() == VolatilityType.ShiftedLognormal:
            return self._swap_rate_value * math.exp(
                std_deviations_for_upper_limit * math.sqrt(variance)
            )
        return self._swap_rate_value + std_deviations_for_upper_limit * math.sqrt(variance)

    def _reset_lower_limit(self, std_deviations_for_upper_limit: float) -> float:
        # C++ parity: conundrumpricer.cpp:425-435.
        vol = self.swaption_volatility()
        assert vol is not None
        assert self._fixing_date is not None
        assert self._swap_tenor is not None
        variance = vol.black_variance(
            self._fixing_date, self._swap_tenor, self._swap_rate_value
        )
        if vol.volatility_type() == VolatilityType.ShiftedLognormal:
            return self._lower_limit
        return self._swap_rate_value - std_deviations_for_upper_limit * math.sqrt(variance)


# =====================================================================
#  AnalyticHaganPricer
# =====================================================================


class AnalyticHaganPricer(HaganPricer):
    """Closed-form Hagan CMS-coupon pricer.

    C++ parity: conundrumpricer.hpp/.cpp ``AnalyticHaganPricer``.
    """

    def _optionlet_price(self, option_type: OptionType, strike: float) -> float:
        # Hagan, 3.5b, 3.5c — C++ parity: conundrumpricer.cpp:511-552.
        vol = self.swaption_volatility()
        assert vol is not None
        assert self._g_function is not None
        assert self._vanilla_option_pricer is not None
        assert self._fixing_date is not None
        assert self._swap_tenor is not None
        assert self._coupon is not None
        variance = vol.black_variance(self._fixing_date, self._swap_tenor, self._swap_rate_value)
        first_derivative_of_g = self._g_function.first_derivative(self._swap_rate_value)

        ck = self._vanilla_option_pricer(strike, option_type, self._annuity)
        price = (self._discount / self._annuity) * ck

        sign = int(option_type)
        if vol.volatility_type() == VolatilityType.ShiftedLognormal:
            sqrt_sigma2_t = math.sqrt(variance)
            ln_r_over_k = math.log(self._swap_rate_value / strike)
            d32 = (ln_r_over_k + 1.5 * variance) / sqrt_sigma2_t
            d12 = (ln_r_over_k + 0.5 * variance) / sqrt_sigma2_t
            dminus12 = (ln_r_over_k - 0.5 * variance) / sqrt_sigma2_t

            n32 = _cumulative_normal(sign * d32)
            n12 = _cumulative_normal(sign * d12)
            nminus12 = _cumulative_normal(sign * dminus12)

            price += (
                sign
                * first_derivative_of_g
                * self._annuity
                * self._swap_rate_value
                * (
                    self._swap_rate_value * math.exp(variance) * n32
                    - (self._swap_rate_value + strike) * n12
                    + strike * nminus12
                )
            )
        else:
            sqrt_sigma2_t = math.sqrt(variance)
            d = (self._swap_rate_value - strike) / sqrt_sigma2_t
            n = _cumulative_normal(sign * d)
            price += sign * first_derivative_of_g * self._annuity * variance * n

        price *= self._coupon.accrual_period()
        return price

    def swaplet_price(self) -> float:
        # Hagan 3.4c — C++ parity: conundrumpricer.cpp:555-580.
        assert self._coupon is not None
        assert self._fixing_date is not None
        today = ObservableSettings().evaluation_date_or_today()
        if self._fixing_date <= today:
            rs = self._coupon.swap_index().fixing(self._fixing_date)
            return (self._gearing * rs + self._spread) * (
                self._coupon.accrual_period() * self._discount
            )
        vol = self.swaption_volatility()
        assert vol is not None
        assert self._g_function is not None
        assert self._swap_tenor is not None
        assert self._fixing_date is not None
        variance = vol.black_variance(self._fixing_date, self._swap_tenor, self._swap_rate_value)
        first_derivative_of_g = self._g_function.first_derivative(self._swap_rate_value)
        price = self._discount * self._swap_rate_value
        if vol.volatility_type() == VolatilityType.ShiftedLognormal:
            price += (
                first_derivative_of_g
                * self._annuity
                * self._swap_rate_value
                * self._swap_rate_value
                * (math.exp(variance) - 1.0)
            )
        else:
            price += first_derivative_of_g * self._annuity * variance
        return (self._gearing * price + self._spread * self._discount) * (
            self._coupon.accrual_period()
        )

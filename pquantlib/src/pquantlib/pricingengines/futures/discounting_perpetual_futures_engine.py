"""DiscountingPerpetualFuturesEngine — discounting engine for perpetual futures.

# C++ parity: ql/pricingengines/futures/discountingperpetualfuturesengine.{hpp,cpp}
(v1.43).

The engine evaluates the perpetual-futures price factor

    factor = sum_i  prod_{j<=i} 1/(1+fr_j) * (fr_i - i^diff_i) * P^for(t_i)/P^dom(t_i)

for discrete funding, or the corresponding integral for continuous funding, and
returns ``spot * factor`` for a ``Linear`` payoff / ``spot / factor`` for an
``Inverse`` one.  Beyond ``max_t`` every rate is extrapolated flat and the tail
is summed in closed form (discountingperpetualfuturesengine.cpp:187-206 for the
discrete geometric tail, 232-243 for the continuous one).

Python port notes:

- The C++ nested ``DiscountingPerpetualFuturesEngine::InterpolationType``
  becomes the module-level ``FundingInterpolationType`` IntEnum, mirroring how
  ``RateAveraging`` translates the same nested-enum idiom.
- ``Handle<YieldTermStructure>`` / ``Handle<Quote>`` become plain objects, so
  the C++ "handle is empty" guards have no Python counterpart — the arguments
  are non-optional instead.
- The C++ ``productIRDiff(i)`` lambda recomputes ``1/(1+fr_0)/.../(1+fr_i)``
  from scratch on every call (O(n^2)).  The port accumulates it incrementally,
  which performs the *same* left-to-right division chain and is therefore
  bit-identical, in O(n).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import IntEnum

import numpy as np

from pquantlib import qassert
from pquantlib.daycounters.year_fraction_to_date import year_fraction_to_date
from pquantlib.instruments.perpetual_futures import (
    PerpetualFuturesEngine,
    PerpetualFuturesFundingType,
    PerpetualFuturesPayoffType,
)
from pquantlib.math.integrals.trapezoid import TrapezoidIntegral
from pquantlib.math.interpolations.backward_flat import BackwardFlatInterpolation
from pquantlib.math.interpolations.cubic_interpolation import CubicNaturalSpline
from pquantlib.math.interpolations.interpolation import Interpolation
from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.time_unit import TimeUnit

_HOURS_PER_DAY = 24.0
_MINUTES_PER_HOUR = 60.0
_SECONDS_PER_MINUTE = 60.0
_MILLIS_PER_SECOND = 1000.0
_MICROS_PER_MILLI = 1000.0
_MONTHS_PER_YEAR = 12.0
_DEFAULT_MAX_T = 60.0


class FundingInterpolationType(IntEnum):
    """# C++ parity: ``DiscountingPerpetualFuturesEngine::InterpolationType``
    (discountingperpetualfuturesengine.hpp:41)."""

    PiecewiseConstant = 0
    Linear = 1
    CubicSpline = 2


class DiscountingPerpetualFuturesEngine(PerpetualFuturesEngine):
    """Discounting engine for perpetual futures.

    # C++ parity: ``DiscountingPerpetualFuturesEngine``
    (discountingperpetualfuturesengine.hpp:38-68, .cpp:30-267).
    """

    def __init__(
        self,
        domestic_discount_curve: YieldTermStructure,
        foreign_discount_curve: YieldTermStructure,
        asset_spot: Quote,
        funding_times: Sequence[float],
        funding_rates: Sequence[float],
        interest_rate_diffs: Sequence[float],
        funding_interp_type: FundingInterpolationType = (FundingInterpolationType.PiecewiseConstant),
        max_t: float = _DEFAULT_MAX_T,
    ) -> None:
        # # C++ parity: ctor (discountingperpetualfuturesengine.cpp:30-56).
        super().__init__()
        self._domestic_discount_curve: YieldTermStructure = domestic_discount_curve
        self._foreign_discount_curve: YieldTermStructure = foreign_discount_curve
        self._asset_spot: Quote = asset_spot
        self._funding_times: list[float] = list(funding_times)
        self._funding_rates: list[float] = list(funding_rates)
        self._interest_rate_diffs: list[float] = list(interest_rate_diffs)
        self._funding_interp_type: FundingInterpolationType = funding_interp_type
        self._max_t: float = max_t

        domestic_discount_curve.register_with(self)
        foreign_discount_curve.register_with(self)
        asset_spot.register_with(self)

        qassert.require(len(self._funding_times) > 0, "fundingTimes is empty")
        qassert.require(len(self._funding_rates) > 0, "fundingRates is empty")
        qassert.require(len(self._interest_rate_diffs) > 0, "interestRateDiffs is empty")
        qassert.require(
            len(self._funding_times) == len(self._funding_rates),
            "fundingTimes and fundingRates must have the same size.",
        )
        qassert.require(
            len(self._funding_times) == len(self._interest_rate_diffs),
            "fundingTimes and interestRateDiffs must have the same size.",
        )

    # --- inspectors ----------------------------------------------------

    def domestic_discount_curve(self) -> YieldTermStructure:
        return self._domestic_discount_curve

    def foreign_discount_curve(self) -> YieldTermStructure:
        return self._foreign_discount_curve

    def asset_spot(self) -> Quote:
        return self._asset_spot

    def funding_times(self) -> list[float]:
        return self._funding_times

    def funding_rates(self) -> list[float]:
        return self._funding_rates

    def interest_rate_diffs(self) -> list[float]:
        return self._interest_rate_diffs

    # --- helpers -------------------------------------------------------

    def _select_interpolation(self, times: Sequence[float], values: Sequence[float]) -> Interpolation:
        """# C++ parity: ``selectInterpolation``
        (discountingperpetualfuturesengine.cpp:246-265)."""
        xs = np.asarray(times, dtype=np.float64)
        ys = np.asarray(values, dtype=np.float64)
        if self._funding_interp_type == FundingInterpolationType.Linear:
            return LinearInterpolation(xs, ys)
        if self._funding_interp_type == FundingInterpolationType.PiecewiseConstant:
            return BackwardFlatInterpolation(xs, ys)
        if self._funding_interp_type == FundingInterpolationType.CubicSpline:
            return CubicNaturalSpline(xs, ys)
        qassert.fail("Unknown interpolation type")

    def _time_grid(self, funding_frequency_length: int, units: TimeUnit) -> list[float]:
        """Funding-date time grid out to ``max_t``.

        # C++ parity: the ``while (tGrid < maxT_)`` loop
        (discountingperpetualfuturesengine.cpp:100-143).
        """
        args = self._arguments
        ref_date = ObservableSettings().evaluation_date_or_today()
        dc = args.dc
        cal = args.cal
        length = float(funding_frequency_length)
        grid: list[float] = []
        t_grid = 0.0
        while t_grid < self._max_t:
            grid.append(t_grid)
            date = year_fraction_to_date(dc, ref_date, t_grid)
            days_in_year = float(
                dc.day_count(
                    Date.from_ymd(1, Month.January, date.year()),
                    Date.from_ymd(1, Month.January, date.year() + 1),
                )
            )
            if units == TimeUnit.Years:
                t_grid += length
            elif units == TimeUnit.Months:
                t_grid += (1.0 / _MONTHS_PER_YEAR) * length
            elif units in (TimeUnit.Weeks, TimeUnit.Days):
                t_grid = dc.year_fraction(ref_date, cal.advance_period(date, args.funding_frequency))
            elif units == TimeUnit.Hours:
                t_grid += (1.0 / days_in_year / _HOURS_PER_DAY) * length
            elif units == TimeUnit.Minutes:
                t_grid += (1.0 / days_in_year / _HOURS_PER_DAY / _MINUTES_PER_HOUR) * length
            elif units == TimeUnit.Seconds:
                t_grid += (
                    1.0 / days_in_year / _HOURS_PER_DAY / _MINUTES_PER_HOUR / _SECONDS_PER_MINUTE
                ) * length
            elif units == TimeUnit.Milliseconds:
                t_grid += (
                    1.0
                    / days_in_year
                    / _HOURS_PER_DAY
                    / _MINUTES_PER_HOUR
                    / _SECONDS_PER_MINUTE
                    / _MILLIS_PER_SECOND
                ) * length
            elif units == TimeUnit.Microseconds:
                t_grid += (
                    1.0
                    / days_in_year
                    / _HOURS_PER_DAY
                    / _MINUTES_PER_HOUR
                    / _SECONDS_PER_MINUTE
                    / _MILLIS_PER_SECOND
                    / _MICROS_PER_MILLI
                ) * length
            else:
                qassert.fail("Unknown unit in fundingFrequency")
        return grid

    def _discrete_factor(
        self,
        eff_dom_curve: YieldTermStructure,
        eff_for_curve: YieldTermStructure,
        funding_rate_interp: Interpolation,
        interest_rate_diff_interp: Interpolation,
    ) -> float:
        """# C++ parity: the discrete-time branch
        (discountingperpetualfuturesengine.cpp:98-206)."""
        args = self._arguments
        freq = args.funding_frequency
        time_grid = self._time_grid(freq.length, freq.units)

        funding_rate_grid = [funding_rate_interp(t) for t in time_grid]
        interest_rate_diff_grid = [interest_rate_diff_interp(t) for t in time_grid]

        n = len(time_grid)
        if args.funding_type == PerpetualFuturesFundingType.FundingWithCurrentSpot:
            ratio = 1.0
            for i in range(n - 1):
                time = time_grid[i]
                next_time = time_grid[i + 1]
                ratio = (
                    eff_for_curve.discount(next_time)
                    / eff_for_curve.discount(time)
                    / eff_dom_curve.discount(next_time)
                    * eff_dom_curve.discount(time)
                )
                funding_rate_grid[i] *= ratio
                interest_rate_diff_grid[i] *= ratio
            # C++ leaves the loop with i == n-1 and applies the last ratio again.
            funding_rate_grid[n - 1] *= ratio
            interest_rate_diff_grid[n - 1] *= ratio

        # productIRDiff(i) accumulated left-to-right — bit-identical to the
        # C++ lambda's fresh ``ret = 1.; for j<=i: ret /= 1+fr_j``.
        product_ir_diff: list[float] = []
        running = 1.0
        for rate in funding_rate_grid:
            running /= 1.0 + rate
            product_ir_diff.append(running)

        total = 0.0
        for i in range(n - 1):
            time = time_grid[i]
            total += (
                product_ir_diff[i]
                * (funding_rate_grid[i] - interest_rate_diff_grid[i])
                * eff_for_curve.discount(time)
                / eff_dom_curve.discount(time)
            )

        i_last = n - 1
        time_last = time_grid[i_last]
        product_last = product_ir_diff[i_last]
        funding_rate_last = funding_rate_grid[i_last]
        interest_rate_diff_last = interest_rate_diff_grid[i_last]

        dom_rate_last = eff_dom_curve.forward_rate(
            time_last, time_last, Compounding.Continuous, Frequency.NoFrequency
        ).rate()
        for_rate_last = eff_for_curve.forward_rate(
            time_last, time_last, Compounding.Continuous, Frequency.NoFrequency
        ).rate()

        # For t > max_t every rate is flat-extrapolated, so the tail is a
        # geometric series with ratio ``1/(1+fr) * exp(-dt (r_for - r_dom))``.
        last_term = (
            product_last
            * (funding_rate_last - interest_rate_diff_last)
            * eff_for_curve.discount(time_last)
            / eff_dom_curve.discount(time_last)
        )
        time_step = (time_grid[-1] - time_grid[0]) / (n - 1)
        ratio = 1.0 / (1.0 + funding_rate_last) * math.exp(-time_step * (for_rate_last - dom_rate_last))
        return total + last_term / (1.0 - ratio)

    def _continuous_factor(
        self,
        eff_dom_curve: YieldTermStructure,
        eff_for_curve: YieldTermStructure,
        funding_rate_interp: Interpolation,
        interest_rate_diff_interp: Interpolation,
    ) -> float:
        """# C++ parity: the continuous-time branch
        (discountingperpetualfuturesengine.cpp:208-243)."""
        integrator = TrapezoidIntegral(1.0e-6, 30)
        funding_rate_x_max = funding_rate_interp.x_max

        def exp_ir_diff(s: float) -> float:
            if s < funding_rate_x_max:
                return math.exp(-integrator(funding_rate_interp, 0.0, s))
            return math.exp(
                -integrator(funding_rate_interp, 0.0, funding_rate_x_max)
                - funding_rate_interp(funding_rate_x_max) * (s - funding_rate_x_max)
            )

        def time_integrand(s: float) -> float:
            return (
                (funding_rate_interp(s) - interest_rate_diff_interp(s))
                * exp_ir_diff(s)
                * eff_for_curve.discount(s)
                / eff_dom_curve.discount(s)
            )

        factor = integrator(time_integrand, 0.0, self._max_t)

        # For t > max_t assume flat extrapolation on all rates.
        funding_rate_last = funding_rate_interp(self._max_t)
        interest_rate_diff_last = interest_rate_diff_interp(self._max_t)
        exp_ir_diff_last = exp_ir_diff(self._max_t)
        dom_rate_last = eff_dom_curve.forward_rate(
            self._max_t, self._max_t, Compounding.Continuous, Frequency.NoFrequency
        ).rate()
        for_rate_last = eff_for_curve.forward_rate(
            self._max_t, self._max_t, Compounding.Continuous, Frequency.NoFrequency
        ).rate()
        ratio = funding_rate_last + for_rate_last - dom_rate_last
        factor += (
            (funding_rate_last - interest_rate_diff_last)
            * exp_ir_diff_last
            * eff_for_curve.discount(self._max_t)
            / eff_dom_curve.discount(self._max_t)
            / ratio
        )
        return factor

    # --- engine --------------------------------------------------------

    def calculate(self) -> None:
        """# C++ parity: ``DiscountingPerpetualFuturesEngine::calculate``
        (discountingperpetualfuturesengine.cpp:58-244)."""
        args = self._arguments
        results = self._results

        results.value = 0.0
        results.error_estimate = None

        qassert.require(
            args.payoff_type in (PerpetualFuturesPayoffType.Linear, PerpetualFuturesPayoffType.Inverse),
            "Only Linear and Inverse payoffs are supported in DiscountingPerpetualFuturesEngine",
        )

        # Linear <-> Inverse: swap the domestic and foreign curves, and invert
        # the futures price (handled at the very end).
        is_linear = args.payoff_type == PerpetualFuturesPayoffType.Linear
        eff_dom_curve = self._domestic_discount_curve if is_linear else self._foreign_discount_curve
        eff_for_curve = self._foreign_discount_curve if is_linear else self._domestic_discount_curve

        funding_rate_interp = self._select_interpolation(self._funding_times, self._funding_rates)
        funding_rate_interp.enable_extrapolation()
        qassert.require(
            funding_rate_interp(funding_rate_interp.x_max) > 0,
            "fundingRate at max time is negative. Because the last funding rate is "
            "flatly extrapolated, integral diverges.",
        )
        interest_rate_diff_interp = self._select_interpolation(self._funding_times, self._interest_rate_diffs)
        interest_rate_diff_interp.enable_extrapolation()

        if args.funding_frequency.length > 0:
            factor = self._discrete_factor(
                eff_dom_curve, eff_for_curve, funding_rate_interp, interest_rate_diff_interp
            )
        else:
            factor = self._continuous_factor(
                eff_dom_curve, eff_for_curve, funding_rate_interp, interest_rate_diff_interp
            )

        if is_linear:
            results.value = self._asset_spot.value() * factor
        else:
            results.value = self._asset_spot.value() / factor


__all__ = ["DiscountingPerpetualFuturesEngine", "FundingInterpolationType"]

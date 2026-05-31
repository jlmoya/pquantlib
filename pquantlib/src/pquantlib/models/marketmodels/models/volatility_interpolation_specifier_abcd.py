"""VolatilityInterpolationSpecifierabcd — abcd-form vol interpolation specifier.

# C++ parity:
# ql/models/marketmodels/models/volatilityinterpolationspecifierabcd.{hpp,cpp}
# (v1.42.1).

Decides the volatility structure for synthetic shorter-tenor rates interleaved
between the original "big" rates, by interpolating the abcd ``(a, b, c, d)``
parameters of the big-rate ``PiecewiseConstantAbcdVariance`` curves:

- **before the offset**: the parameters of the first big rate;
- **between big rates ``j`` and ``j+1``**: the arithmetic mean of their
  (scaled) ``(a, b, c, d)``;
- **after the last big rate**: the parameters of the last big rate, with the
  very last synthetic rate rescaled so its total volatility matches the
  supplied terminal caplet vol.

The big-rate parameters can be pre-scaled per rate (``set_scaling_factors``):
``a``, ``b``, ``d`` are scaled, ``c`` is left unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.models.marketmodels.models.piecewise_constant_abcd_variance import (
    PiecewiseConstantAbcdVariance,
)
from pquantlib.models.marketmodels.models.piecewise_constant_variance import (
    PiecewiseConstantVariance,
)
from pquantlib.models.marketmodels.models.volatility_interpolation_specifier import (
    VolatilityInterpolationSpecifier,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class VolatilityInterpolationSpecifierabcd(VolatilityInterpolationSpecifier):
    """Abcd-form synthetic-rate volatility interpolation specifier.

    # C++ parity: volatilityinterpolationspecifierabcd.hpp/.cpp
    VolatilityInterpolationSpecifierabcd.
    """

    def __init__(
        self,
        period: int,
        offset: int,
        original_variances: Sequence[PiecewiseConstantAbcdVariance],
        times_for_small_rates: list[float],
        last_caplet_vol: float = 0.0,
    ) -> None:
        self._period = period
        self._offset = offset
        self._no_big_rates = len(original_variances)
        self._no_small_rates = len(times_for_small_rates) - 1
        self._times_for_small_rates = list(times_for_small_rates)
        # C++ keeps two copies of the original abcd variances: the unscaled set
        # and the scaled set (recomputed in recompute()).
        self._original_abcd_variances: list[PiecewiseConstantAbcdVariance] = list(
            original_variances
        )
        self._original_abcd_variances_scaled: list[PiecewiseConstantAbcdVariance] = list(
            original_variances
        )
        self._last_caplet_vol = last_caplet_vol
        self._scaling_factors: list[float] = [1.0] * self._no_big_rates

        self._interpolated_variances: list[PiecewiseConstantVariance] = [
            self._original_abcd_variances[0]
        ] * self._no_small_rates
        # client-facing original variances (typed as the base class)
        self._original_variances: list[PiecewiseConstantVariance] = list(
            original_variances
        )

        qassert.require(
            (self._no_small_rates - offset) // period == self._no_big_rates,
            "size mismatch in VolatilityInterpolationSpecifierabcd",
        )
        for i in range(self._no_big_rates):
            big_rate_times = original_variances[i].rate_times()
            for j in range(len(big_rate_times)):
                qassert.require(
                    big_rate_times[j] == times_for_small_rates[offset + j * period],
                    "rate times in variances passed in don't match small times in "
                    "VolatilityInterpolationSpecifierabcd",
                )

        if self._last_caplet_vol == 0.0:
            self._last_caplet_vol = original_variances[
                self._no_big_rates - 1
            ].total_volatility(self._no_big_rates - 1)

        self._recompute()

    def set_scaling_factors(self, scales: Sequence[float]) -> None:
        """Set per-big-rate scaling factors and recompute.

        # C++ parity: VolatilityInterpolationSpecifierabcd::setScalingFactors.
        """
        qassert.require(
            len(self._scaling_factors) == len(scales),
            "inappropriate number of scales passed in to "
            "VolatilityInterpolationSpecifierabcd::setScalingFactors ",
        )
        self._scaling_factors = list(scales)
        self._recompute()

    def set_last_caplet_vol(self, vol: float) -> None:
        """Set the terminal caplet vol target and recompute.

        # C++ parity: VolatilityInterpolationSpecifierabcd::setLastCapletVol.
        """
        self._last_caplet_vol = vol
        self._recompute()

    def interpolated_variances(self) -> list[PiecewiseConstantVariance]:
        """The interpolated per-small-rate variances."""
        return self._interpolated_variances

    def original_variances(self) -> list[PiecewiseConstantVariance]:
        """The original per-big-rate variances."""
        return self._original_variances

    def get_period(self) -> int:
        """The interleaving period."""
        return self._period

    def get_offset(self) -> int:
        """The leading small-rate offset."""
        return self._offset

    def get_no_big_rates(self) -> int:
        """Number of big rates."""
        return self._no_big_rates

    def get_no_small_rates(self) -> int:
        """Number of small rates."""
        return self._no_small_rates

    def _recompute(self) -> None:
        """Rebuild the interpolated small-rate variances.

        # C++ parity: VolatilityInterpolationSpecifierabcd::recompute.
        """
        # scale the big-rate abcd params (a, b, d scaled; c unchanged)
        for i in range(self._no_big_rates):
            a, b, c, d = self._original_abcd_variances[i].get_abcd()
            a *= self._scaling_factors[i]
            b *= self._scaling_factors[i]
            d *= self._scaling_factors[i]
            self._original_abcd_variances_scaled[i] = PiecewiseConstantAbcdVariance(
                a, b, c, d, i, self._original_abcd_variances[i].rate_times()
            )

        # three regions: before offset, between big rates, after last big rate
        # before offset -> params of the first (scaled) big rate
        a0, b0, c0, d0 = self._original_abcd_variances_scaled[0].get_abcd()
        for i in range(self._offset):
            self._interpolated_variances[i] = PiecewiseConstantAbcdVariance(
                a0, b0, c0, d0, i, self._times_for_small_rates
            )

        # in between rates -> mean of neighbouring (scaled) big-rate params
        for j in range(self._no_big_rates - 1):
            aj0, bj0, cj0, dj0 = self._original_abcd_variances_scaled[j].get_abcd()
            aj1, bj1, cj1, dj1 = self._original_abcd_variances_scaled[j + 1].get_abcd()
            a = 0.5 * (aj0 + aj1)
            b = 0.5 * (bj0 + bj1)
            c = 0.5 * (cj0 + cj1)
            d = 0.5 * (dj0 + dj1)
            for i in range(self._period):
                idx = i + j * self._period + self._offset
                self._interpolated_variances[idx] = PiecewiseConstantAbcdVariance(
                    a, b, c, d, i + j * self._period, self._times_for_small_rates
                )

        # after last big rate -> params of the last (scaled) big rate
        a, b, c, d = self._original_abcd_variances_scaled[
            self._no_big_rates - 1
        ].get_abcd()
        start = self._offset + (self._no_big_rates - 1) * self._period
        for i in range(start, self._no_small_rates):
            self._interpolated_variances[i] = PiecewiseConstantAbcdVariance(
                a, b, c, d, i, self._times_for_small_rates
            )

        # the very last rate is special: rescale to match the caplet vol
        vol = self._interpolated_variances[self._no_small_rates - 1].total_volatility(
            self._no_small_rates - 1
        )
        scale = self._last_caplet_vol / vol
        a *= scale
        b *= scale
        d *= scale
        self._interpolated_variances[self._no_small_rates - 1] = (
            PiecewiseConstantAbcdVariance(
                a, b, c, d, self._no_small_rates - 1, self._times_for_small_rates
            )
        )

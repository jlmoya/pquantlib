"""VolatilityInterpolationSpecifier — abstract synthetic-rate vol interpolator.

# C++ parity:
# ql/models/marketmodels/models/volatilityinterpolationspecifier.hpp (v1.42.1).

Abstract base specifying how to decide the volatility structure for the
additional synthetic (shorter-tenor) rates interleaved between the "big" rates
during caplet-coterminal calibration. Subclasses hold an interpolated set of
``PiecewiseConstantVariance`` for the small rates derived from the original
big-rate variances.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pquantlib.models.marketmodels.models.piecewise_constant_variance import (
        PiecewiseConstantVariance,
    )


class VolatilityInterpolationSpecifier(ABC):
    """Abstract synthetic-rate volatility interpolation specifier.

    # C++ parity: volatilityinterpolationspecifier.hpp
    VolatilityInterpolationSpecifier.
    """

    @abstractmethod
    def set_scaling_factors(self, scales: Sequence[float]) -> None:
        """Set per-big-rate scaling factors and recompute the interpolation."""

    @abstractmethod
    def set_last_caplet_vol(self, vol: float) -> None:
        """Set the terminal caplet vol target and recompute."""

    @abstractmethod
    def interpolated_variances(self) -> list[PiecewiseConstantVariance]:
        """The interpolated per-small-rate variances."""

    @abstractmethod
    def original_variances(self) -> list[PiecewiseConstantVariance]:
        """The original per-big-rate variances."""

    @abstractmethod
    def get_period(self) -> int:
        """The interleaving period (small rates per big rate)."""

    @abstractmethod
    def get_offset(self) -> int:
        """The leading small-rate offset before the first big rate."""

    @abstractmethod
    def get_no_big_rates(self) -> int:
        """Number of big (original) rates."""

    @abstractmethod
    def get_no_small_rates(self) -> int:
        """Number of small (interpolated) rates."""

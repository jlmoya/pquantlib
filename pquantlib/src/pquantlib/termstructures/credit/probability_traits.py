"""Bootstrap traits for default-probability curves.

# C++ parity: ql/termstructures/credit/probabilitytraits.hpp (v1.42.1).

The C++ defines three template traits structs — ``SurvivalProbability``,
``HazardRate``, ``DefaultDensity`` — used to parameterize the
``PiecewiseDefaultCurve<Traits, Interpolator, Bootstrap>`` template. The
traits provide static methods to:

- pick the initial pillar value (``initialValue``),
- guess the next pillar (``guess``),
- bound the search (``minValueAfter`` / ``maxValueAfter``),
- update the curve data after a successful solve (``updateGuess``).

In Python we expose these as classmethods on three classes
(``SurvivalProbabilityTrait`` / ``HazardRateTrait`` / ``DefaultDensityTrait``).
Bootstrap consumers receive a trait class and call the classmethods
without instantiation — matching the C++ all-static API.

# C++ parity divergence: full ``PiecewiseDefaultCurve`` bootstrap is
   deferred (see Phase 8 carve-out). The traits land here as scaffolding
   so cluster L8-B can complete its abstract surface without needing the
   piecewise iterator — which depends on the L2-B PiecewiseYieldCurve
   carry-over still pending.
"""

from __future__ import annotations

import math
import sys
from typing import Final

# C++ parity: ql/termstructures/credit/probabilitytraits.hpp anon namespace.
_AVG_HAZARD_RATE: Final[float] = 0.01
_MAX_HAZARD_RATE: Final[float] = 1.0
_QL_EPSILON: Final[float] = sys.float_info.epsilon


class SurvivalProbabilityTrait:
    """Survival-probability bootstrap trait.

    # C++ parity: ``struct SurvivalProbability`` in
    # probabilitytraits.hpp:44-110.
    """

    max_iterations: int = 50

    @staticmethod
    def initial_value() -> float:
        """Value at the reference date."""
        return 1.0

    @staticmethod
    def guess(i: int, data: list[float], dates_times: list[float], valid_data: bool) -> float:
        """Initial guess for the i-th pillar.

        # C++ parity: ``SurvivalProbability::guess`` (lines 63-78).
        """
        del dates_times  # only used by the extrapolate branch (deferred).
        if valid_data:
            return data[i]
        if i == 1:
            return 1.0 / (1.0 + _AVG_HAZARD_RATE * 0.25)
        # Linear extrapolation in the simple case.
        return data[i - 1]

    @staticmethod
    def min_value_after(
        i: int, data: list[float], times: list[float], valid_data: bool,
    ) -> float:
        """Lower bound for the search at pillar ``i``.

        # C++ parity: ``SurvivalProbability::minValueAfter`` (lines 81-91).
        """
        if valid_data:
            return data[-1] / 2.0
        dt = times[i] - times[i - 1]
        return data[i - 1] * math.exp(-_MAX_HAZARD_RATE * dt)

    @staticmethod
    def max_value_after(
        i: int, data: list[float], times: list[float], valid_data: bool,
    ) -> float:
        """Upper bound: survival probability cannot increase.

        # C++ parity: ``SurvivalProbability::maxValueAfter`` (lines 93-100).
        """
        del times, valid_data
        return data[i - 1]

    @staticmethod
    def update_guess(data: list[float], p: float, i: int) -> None:
        """C++ parity: ``SurvivalProbability::updateGuess``."""
        data[i] = p


class HazardRateTrait:
    """Hazard-rate bootstrap trait.

    # C++ parity: ``struct HazardRate`` in probabilitytraits.hpp:115-188.
    """

    max_iterations: int = 30

    @staticmethod
    def initial_value() -> float:
        # C++ uses ``avgHazardRate`` (a dummy seed; not actually a probability).
        return _AVG_HAZARD_RATE

    @staticmethod
    def guess(i: int, data: list[float], dates_times: list[float], valid_data: bool) -> float:
        del dates_times
        if valid_data:
            return data[i]
        if i == 1:
            return _AVG_HAZARD_RATE
        return data[i - 1]

    @staticmethod
    def min_value_after(
        i: int, data: list[float], times: list[float], valid_data: bool,
    ) -> float:
        del i, times
        if valid_data:
            return min(data) / 2.0
        return _QL_EPSILON

    @staticmethod
    def max_value_after(
        i: int, data: list[float], times: list[float], valid_data: bool,
    ) -> float:
        del i, times
        if valid_data:
            return max(data) * 2.0
        return _MAX_HAZARD_RATE

    @staticmethod
    def update_guess(data: list[float], rate: float, i: int) -> None:
        data[i] = rate
        if i == 1:
            data[0] = rate


class DefaultDensityTrait:
    """Default-density bootstrap trait.

    # C++ parity: ``struct DefaultDensity`` in probabilitytraits.hpp:192-265.
    """

    max_iterations: int = 30

    @staticmethod
    def initial_value() -> float:
        return _AVG_HAZARD_RATE

    @staticmethod
    def guess(i: int, data: list[float], dates_times: list[float], valid_data: bool) -> float:
        del dates_times
        if valid_data:
            return data[i]
        if i == 1:
            return _AVG_HAZARD_RATE
        return data[i - 1]

    @staticmethod
    def min_value_after(
        i: int, data: list[float], times: list[float], valid_data: bool,
    ) -> float:
        del i, times
        if valid_data:
            return min(data) / 2.0
        return _QL_EPSILON

    @staticmethod
    def max_value_after(
        i: int, data: list[float], times: list[float], valid_data: bool,
    ) -> float:
        del i, times
        if valid_data:
            return max(data) * 2.0
        return _MAX_HAZARD_RATE

    @staticmethod
    def update_guess(data: list[float], density: float, i: int) -> None:
        data[i] = density
        if i == 1:
            data[0] = density


# Convenience aliases matching C++ struct names.
SurvivalProbability = SurvivalProbabilityTrait
HazardRate = HazardRateTrait
DefaultDensity = DefaultDensityTrait


__all__ = [
    "DefaultDensity",
    "DefaultDensityTrait",
    "HazardRate",
    "HazardRateTrait",
    "SurvivalProbability",
    "SurvivalProbabilityTrait",
]

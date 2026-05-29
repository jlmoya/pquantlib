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

Python ports these as ``@staticmethod`` members on three classes
(:class:`SurvivalProbabilityTrait` / :class:`HazardRateTrait` /
:class:`DefaultDensityTrait`) so both class-bound usage
(``SurvivalProbabilityTrait.initial_value()``) and instance-bound usage
(``SurvivalProbabilityTrait().initial_value()``) work — matching how the
C++ ``struct`` traits are consumed without instantiation.

:class:`PiecewiseDefaultCurve` and :class:`IterativeBootstrap` accept
either a class OR an instance for ``traits=``; both paths route to the
same staticmethods. The Python ``initial_value`` etc accept an optional
``ts`` argument (ignored) for protocol parity with the yield traits
(which depend on the curve's reference_date / times).

The 4-arg form ``guess(i, data, times, valid_data)`` is preserved from
L8-B for the credit unit tests; the bootstrap calls the 3-arg form
``guess(i, data, valid_data)`` (with ``times`` defaulting to None).
``min/max_value_after`` follow the same pattern.
"""

from __future__ import annotations

import math
import sys
from typing import Any, Final

# C++ parity: ql/termstructures/credit/probabilitytraits.hpp anon namespace.
_AVG_HAZARD_RATE: Final[float] = 0.01
_MAX_HAZARD_RATE: Final[float] = 1.0
_QL_EPSILON: Final[float] = sys.float_info.epsilon


class SurvivalProbabilityTrait:
    """Survival-probability bootstrap trait.

    # C++ parity: ``struct SurvivalProbability`` in
    # probabilitytraits.hpp:44-110.

    Bootstrap state is ``S(t_i)`` — monotonically decreasing in i.
    """

    max_iterations_value: int = 50

    @staticmethod
    def initial_date(ts: Any) -> Any:
        # C++ parity: ``SurvivalProbability::initialDate``.
        return ts.reference_date()

    @staticmethod
    def initial_value(ts: Any = None) -> float:
        """Value at the reference date.

        # C++ parity: ``SurvivalProbability::initialValue`` = 1.0.
        """
        del ts
        return 1.0

    @staticmethod
    def guess(
        i: int, data: list[float], times_or_valid: Any = None,
        valid_data: bool = False,
    ) -> float:
        """Initial guess for the i-th pillar.

        # C++ parity: ``SurvivalProbability::guess`` (lines 63-78).

        Two calling conventions supported:
        - 3-arg ``guess(i, data, valid_data)`` — bootstrap protocol form.
        - 4-arg ``guess(i, data, times, valid_data)`` — L8-B unit-test
          form (times unused for SurvivalProbability).
        """
        if isinstance(times_or_valid, bool):
            valid_data = times_or_valid
        if valid_data:
            return data[i]
        if i == 1:
            return 1.0 / (1.0 + _AVG_HAZARD_RATE * 0.25)
        # Flat extrapolation.
        return data[i - 1]

    @staticmethod
    def min_value_after(
        i: int, data: list[float], times_or_valid: Any = None,
        valid_data: bool = False,
    ) -> float:
        """Lower bound for the search at pillar ``i``.

        # C++ parity: ``SurvivalProbability::minValueAfter`` (lines 81-91).

        Two calling conventions: 3-arg ``(i, data, valid_data)`` and
        4-arg ``(i, data, times, valid_data)``.
        """
        times: list[float] | None = None
        if isinstance(times_or_valid, bool):
            valid_data = times_or_valid
        elif times_or_valid is not None:
            times = times_or_valid
        if valid_data:
            return data[-1] / 2.0
        dt = times[i] - times[i - 1] if times is not None else 1.0
        return data[i - 1] * math.exp(-_MAX_HAZARD_RATE * dt)

    @staticmethod
    def max_value_after(
        i: int, data: list[float], times_or_valid: Any = None,
        valid_data: bool = False,
    ) -> float:
        """Upper bound: survival probability cannot increase.

        # C++ parity: ``SurvivalProbability::maxValueAfter`` (lines 93-100).
        """
        del times_or_valid, valid_data
        return data[i - 1]

    @staticmethod
    def update_guess(data: list[float], p: float, i: int) -> None:
        """C++ parity: ``SurvivalProbability::updateGuess``."""
        data[i] = p

    @staticmethod
    def max_iterations() -> int:
        return SurvivalProbabilityTrait.max_iterations_value


class HazardRateTrait:
    """Hazard-rate bootstrap trait.

    # C++ parity: ``struct HazardRate`` in probabilitytraits.hpp:115-188.
    """

    max_iterations_value: int = 30

    @staticmethod
    def initial_date(ts: Any) -> Any:
        # C++ parity: ``HazardRate::initialDate``.
        return ts.reference_date()

    @staticmethod
    def initial_value(ts: Any = None) -> float:
        # C++ uses ``avgHazardRate`` (a dummy seed; not actually a probability).
        del ts
        return _AVG_HAZARD_RATE

    @staticmethod
    def guess(
        i: int, data: list[float], times_or_valid: Any = None,
        valid_data: bool = False,
    ) -> float:
        if isinstance(times_or_valid, bool):
            valid_data = times_or_valid
        if valid_data:
            return data[i]
        if i == 1:
            return _AVG_HAZARD_RATE
        return data[i - 1]

    @staticmethod
    def min_value_after(
        i: int, data: list[float], times_or_valid: Any = None,
        valid_data: bool = False,
    ) -> float:
        del i
        if isinstance(times_or_valid, bool):
            valid_data = times_or_valid
        if valid_data:
            return min(data) / 2.0
        return _QL_EPSILON

    @staticmethod
    def max_value_after(
        i: int, data: list[float], times_or_valid: Any = None,
        valid_data: bool = False,
    ) -> float:
        del i
        if isinstance(times_or_valid, bool):
            valid_data = times_or_valid
        if valid_data:
            return max(data) * 2.0
        return _MAX_HAZARD_RATE

    @staticmethod
    def update_guess(data: list[float], rate: float, i: int) -> None:
        data[i] = rate
        if i == 1:
            data[0] = rate

    @staticmethod
    def max_iterations() -> int:
        return HazardRateTrait.max_iterations_value


class DefaultDensityTrait:
    """Default-density bootstrap trait.

    # C++ parity: ``struct DefaultDensity`` in probabilitytraits.hpp:192-265.
    """

    max_iterations_value: int = 30

    @staticmethod
    def initial_date(ts: Any) -> Any:
        # C++ parity: ``DefaultDensity::initialDate``.
        return ts.reference_date()

    @staticmethod
    def initial_value(ts: Any = None) -> float:
        del ts
        return _AVG_HAZARD_RATE

    @staticmethod
    def guess(
        i: int, data: list[float], times_or_valid: Any = None,
        valid_data: bool = False,
    ) -> float:
        if isinstance(times_or_valid, bool):
            valid_data = times_or_valid
        if valid_data:
            return data[i]
        if i == 1:
            return _AVG_HAZARD_RATE
        return data[i - 1]

    @staticmethod
    def min_value_after(
        i: int, data: list[float], times_or_valid: Any = None,
        valid_data: bool = False,
    ) -> float:
        del i
        if isinstance(times_or_valid, bool):
            valid_data = times_or_valid
        if valid_data:
            return min(data) / 2.0
        return _QL_EPSILON

    @staticmethod
    def max_value_after(
        i: int, data: list[float], times_or_valid: Any = None,
        valid_data: bool = False,
    ) -> float:
        del i
        if isinstance(times_or_valid, bool):
            valid_data = times_or_valid
        if valid_data:
            return max(data) * 2.0
        return _MAX_HAZARD_RATE

    @staticmethod
    def update_guess(data: list[float], density: float, i: int) -> None:
        data[i] = density
        if i == 1:
            data[0] = density

    @staticmethod
    def max_iterations() -> int:
        return DefaultDensityTrait.max_iterations_value


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

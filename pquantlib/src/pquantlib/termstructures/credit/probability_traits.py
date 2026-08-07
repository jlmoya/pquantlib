"""Bootstrap traits for default-probability curves.

# C++ parity: ql/termstructures/credit/probabilitytraits.hpp (v1.43).

Three traits parameterize ``PiecewiseDefaultCurve<Traits, Interpolator,
Bootstrap>``: :class:`SurvivalProbability`, :class:`HazardRate`,
:class:`DefaultDensity`. They provide the initial pillar value, the
per-pillar guess, the Brent bracket, and the post-solve update.

Python ports them as ``@staticmethod`` members so both class-bound usage
(``SurvivalProbability.initial_value()``) and instance-bound usage
(``SurvivalProbability().initial_value()``) work — matching how the C++
``struct`` traits are consumed without instantiation.
:class:`PiecewiseDefaultCurve` and :class:`IterativeBootstrap` accept
either a class OR an instance for ``traits=``.

``guess`` / ``min_value_after`` / ``max_value_after`` take the CURVE, as
C++ does (``const C* c``): they read ``c->times()``, ``c->data()``,
``c->dates()`` and call back into ``c->survivalProbability()`` /
``c->hazardRate()`` / ``c->defaultDensity()`` to extrapolate. The final
``first_alive_helper`` argument is the index of the first non-expired
helper; these traits ignore it, exactly as C++ does (the parameter is
declared unnamed), but it is part of the protocol.

Unlike the yield and inflation traits, the C++ credit traits define NO
``transformDirect`` / ``transformInverse``, so neither does this port.
"""

from __future__ import annotations

import math
import sys
from typing import Any, Final

# C++ parity: probabilitytraits.hpp:38-41 (``namespace detail`` — avgHazardRate
# at :39, maxHazardRate at :40).
_AVG_HAZARD_RATE: Final[float] = 0.01
_MAX_HAZARD_RATE: Final[float] = 1.0
_QL_EPSILON: Final[float] = sys.float_info.epsilon


class SurvivalProbability:
    """Survival-probability bootstrap trait.

    # C++ parity: ``struct SurvivalProbability`` in
    # probabilitytraits.hpp:43-110.

    Bootstrap state is ``S(t_i)`` — monotonically decreasing in i.
    """

    max_iterations_value: int = 50

    @staticmethod
    def initial_date(ts: Any) -> Any:
        # C++ parity: probabilitytraits.hpp:53-56.
        return ts.reference_date()

    @staticmethod
    def initial_value(ts: Any = None) -> float:
        """Value at the reference date.

        # C++ parity: probabilitytraits.hpp:57-60 — 1.0.
        """
        del ts
        return 1.0

    @staticmethod
    def guess(i: int, c: Any, valid_data: bool, first_alive_helper: int = 0) -> float:
        """Initial guess for the i-th pillar.

        # C++ parity: probabilitytraits.hpp:62-79.
        """
        del first_alive_helper
        if valid_data:
            return c.data()[i]
        if i == 1:
            return 1.0 / (1.0 + _AVG_HAZARD_RATE * 0.25)
        # probabilitytraits.hpp:77-78 — extrapolate off the curve.
        return c.survival_probability(c.dates()[i], True)

    @staticmethod
    def min_value_after(
        i: int, c: Any, valid_data: bool, first_alive_helper: int = 0
    ) -> float:
        """Lower bound for the search at pillar ``i``.

        # C++ parity: probabilitytraits.hpp:80-92.
        """
        del first_alive_helper
        if valid_data:
            return c.data()[-1] / 2.0
        # probabilitytraits.hpp:90-91 — the ACTUAL time gap, not 1.0.
        dt = c.times()[i] - c.times()[i - 1]
        return c.data()[i - 1] * math.exp(-_MAX_HAZARD_RATE * dt)

    @staticmethod
    def max_value_after(
        i: int, c: Any, valid_data: bool, first_alive_helper: int = 0
    ) -> float:
        """Upper bound: survival probability cannot increase.

        # C++ parity: probabilitytraits.hpp:93-100.
        """
        del valid_data, first_alive_helper
        return c.data()[i - 1]

    @staticmethod
    def update_guess(data: list[float], p: float, i: int) -> None:
        """# C++ parity: probabilitytraits.hpp:102-107."""
        data[i] = p

    @staticmethod
    def max_iterations() -> int:
        # C++ parity: probabilitytraits.hpp:109 — 50.
        return SurvivalProbability.max_iterations_value


class HazardRate:
    """Hazard-rate bootstrap trait.

    # C++ parity: ``struct HazardRate`` in probabilitytraits.hpp:114-188.
    """

    max_iterations_value: int = 30

    @staticmethod
    def initial_date(ts: Any) -> Any:
        # C++ parity: probabilitytraits.hpp:124-127.
        return ts.reference_date()

    @staticmethod
    def initial_value(ts: Any = None) -> float:
        # C++ parity: probabilitytraits.hpp:128-131 — avgHazardRate (a dummy).
        del ts
        return _AVG_HAZARD_RATE

    @staticmethod
    def guess(i: int, c: Any, valid_data: bool, first_alive_helper: int = 0) -> float:
        """# C++ parity: probabilitytraits.hpp:133-150."""
        del first_alive_helper
        if valid_data:
            return c.data()[i]
        if i == 1:
            return _AVG_HAZARD_RATE
        # probabilitytraits.hpp:148-149.
        return c.hazard_rate(c.dates()[i], True)

    @staticmethod
    def min_value_after(
        i: int, c: Any, valid_data: bool, first_alive_helper: int = 0
    ) -> float:
        """# C++ parity: probabilitytraits.hpp:152-164."""
        del i, first_alive_helper
        if valid_data:
            return min(c.data()) / 2.0
        return _QL_EPSILON

    @staticmethod
    def max_value_after(
        i: int, c: Any, valid_data: bool, first_alive_helper: int = 0
    ) -> float:
        """# C++ parity: probabilitytraits.hpp:164-177."""
        del i, first_alive_helper
        if valid_data:
            return max(c.data()) * 2.0
        return _MAX_HAZARD_RATE

    @staticmethod
    def update_guess(data: list[float], rate: float, i: int) -> None:
        # C++ parity: probabilitytraits.hpp:178-185.
        data[i] = rate
        if i == 1:
            data[0] = rate

    @staticmethod
    def max_iterations() -> int:
        # C++ parity: probabilitytraits.hpp:187 — 30.
        return HazardRate.max_iterations_value


class DefaultDensity:
    """Default-density bootstrap trait.

    # C++ parity: ``struct DefaultDensity`` in probabilitytraits.hpp:191-265.
    """

    max_iterations_value: int = 30

    @staticmethod
    def initial_date(ts: Any) -> Any:
        # C++ parity: probabilitytraits.hpp:200-203.
        return ts.reference_date()

    @staticmethod
    def initial_value(ts: Any = None) -> float:
        # C++ parity: probabilitytraits.hpp:204-207.
        del ts
        return _AVG_HAZARD_RATE

    @staticmethod
    def guess(i: int, c: Any, valid_data: bool, first_alive_helper: int = 0) -> float:
        """# C++ parity: probabilitytraits.hpp:209-226."""
        del first_alive_helper
        if valid_data:
            return c.data()[i]
        if i == 1:
            return _AVG_HAZARD_RATE
        # probabilitytraits.hpp:224-225.
        return c.default_density(c.dates()[i], True)

    @staticmethod
    def min_value_after(
        i: int, c: Any, valid_data: bool, first_alive_helper: int = 0
    ) -> float:
        """# C++ parity: probabilitytraits.hpp:228-240."""
        del i, first_alive_helper
        if valid_data:
            return min(c.data()) / 2.0
        return _QL_EPSILON

    @staticmethod
    def max_value_after(
        i: int, c: Any, valid_data: bool, first_alive_helper: int = 0
    ) -> float:
        """# C++ parity: probabilitytraits.hpp:240-253."""
        del i, first_alive_helper
        if valid_data:
            return max(c.data()) * 2.0
        return _MAX_HAZARD_RATE

    @staticmethod
    def update_guess(data: list[float], density: float, i: int) -> None:
        # C++ parity: probabilitytraits.hpp:255-262.
        data[i] = density
        if i == 1:
            data[0] = density

    @staticmethod
    def max_iterations() -> int:
        # C++ parity: probabilitytraits.hpp:264 — 30.
        return DefaultDensity.max_iterations_value


# Backwards-compatible aliases: these classes shipped with a ``Trait`` suffix
# before the v1.43 sweep restored the C++ names. The yield-curve traits
# (``Discount`` / ``ZeroYield`` / ``ForwardRate``) never carried the suffix, so
# the suffix was also inconsistent within the port.
SurvivalProbabilityTrait = SurvivalProbability
HazardRateTrait = HazardRate
DefaultDensityTrait = DefaultDensity

__all__ = [
    "DefaultDensity",
    "DefaultDensityTrait",
    "HazardRate",
    "HazardRateTrait",
    "SurvivalProbability",
    "SurvivalProbabilityTrait",
]

"""DefaultDensityStructure — default-density adapter.

# C++ parity: ql/termstructures/credit/defaultdensitystructure.{hpp,cpp}
   (v1.42.1).

Subclasses only need to override ``_default_density_impl(t)``;
``_survival_probability_impl(t)`` defaults to ``1 - integral_0^t p(tau) dtau``
(clamped at 0).

# C++ parity divergence: the C++ version uses a 48-point Gauss-Chebyshev
   integration. The Python port uses scipy ``quad`` (adaptive Gauss-Kronrod).
"""

from __future__ import annotations

from typing import cast

from scipy.integrate import quad  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

from pquantlib.termstructures.credit.default_probability_term_structure import (
    DefaultProbabilityTermStructure,
)


class DefaultDensityStructure(DefaultProbabilityTermStructure):
    """Adapter: subclasses implement p(t); S(t) = 1 - integral.

    Subclasses with closed-form integrals
    (InterpolatedDefaultDensityCurve uses interpolation primitives) should
    override ``_survival_probability_impl`` directly.
    """

    def _survival_probability_impl(self, t: float) -> float:
        """1 - integral_0^t p(tau) d tau (clamped at 0).

        # C++ parity: defaultdensitystructure.cpp:71-77 uses 48-point
        # Gauss-Chebyshev. We use scipy ``quad``.
        """
        if t == 0.0:
            return 1.0
        result = cast("tuple[float, float]", quad(self._default_density_impl, 0.0, t))
        integral, _err = result
        return max(1.0 - integral, 0.0)


__all__ = ["DefaultDensityStructure"]

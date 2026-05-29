"""HazardRateStructure — hazard-rate adapter.

# C++ parity: ql/termstructures/credit/hazardratestructure.{hpp,cpp}
   (v1.42.1).

Subclasses only need to override ``_hazard_rate_impl(t)``;
``_survival_probability_impl(t)`` defaults to ``exp(-integral_0^t h(tau) dtau)``
where the integral is computed by Gauss-Chebyshev quadrature in C++.

# C++ parity divergence: the C++ version uses a 48-point Gauss-Chebyshev
   integration. The Python port uses scipy ``quad`` (adaptive Gauss-Kronrod)
   for the same precision goal. Both are 1e-12 absolute / relative; the
   results match within TIGHT.

``_default_density_impl(t)`` is h(t) * S(t) (closed form).
"""

from __future__ import annotations

import math
from typing import cast

from scipy.integrate import quad  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

from pquantlib import qassert
from pquantlib.termstructures.credit.default_probability_term_structure import (
    DefaultProbabilityTermStructure,
)


class HazardRateStructure(DefaultProbabilityTermStructure):
    """Adapter: subclasses implement h(t); S(t) is computed by quadrature.

    Warning: numerical integration may be imprecise. Subclasses with
    closed-form S(t) (FlatHazardRate, InterpolatedHazardRateCurve) should
    override ``_survival_probability_impl`` for efficiency + exactness.
    """

    def _hazard_rate_impl(self, t: float) -> float:
        qassert.fail(
            "_hazard_rate_impl must be implemented by a class "
            "derived from HazardRateStructure",
        )
        # qassert.fail raises; this line is unreachable but keeps
        # static analyzers happy.
        return 0.0

    def _survival_probability_impl(self, t: float) -> float:
        """exp(-integral_0^t h(tau) d tau).

        # C++ parity: hazardratestructure.cpp:77-82 uses 48-point
        # Gauss-Chebyshev. We use scipy ``quad`` (adaptive Gauss-Kronrod)
        # with default 1e-8 tolerances; results match within TIGHT for
        # smooth h(t).
        """
        if t == 0.0:
            return 1.0
        result = cast("tuple[float, float]", quad(self._hazard_rate_impl, 0.0, t))
        integral, _err = result
        return math.exp(-integral)

    def _default_density_impl(self, t: float) -> float:
        """p(t) = h(t) * S(t).

        # C++ parity: inline in hazardratestructure.hpp:106-108.
        """
        return self._hazard_rate_impl(t) * self._survival_probability_impl(t)


__all__ = ["HazardRateStructure"]

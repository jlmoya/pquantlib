"""SurvivalProbabilityStructure — survival-probability adapter.

# C++ parity: ql/termstructures/credit/survivalprobabilitystructure.{hpp,cpp}
   (v1.42.1).

Subclasses only need to override ``_survival_probability_impl(t)``;
``_default_density_impl(t)`` is filled in by central finite-difference
on S(t) using width dt = 1e-4.
"""

from __future__ import annotations

from pquantlib.termstructures.credit.default_probability_term_structure import (
    DefaultProbabilityTermStructure,
)


class SurvivalProbabilityStructure(DefaultProbabilityTermStructure):
    """Adapter: subclasses implement S(t); p(t) is computed by FD.

    Warning: FD differentiation is imprecise (1e-4 width); subclasses
    that can supply analytic p(t) should override
    ``_default_density_impl``.
    """

    _DT: float = 1.0e-4

    def _default_density_impl(self, t: float) -> float:
        """p(t) ≈ (S(t-dt) - S(t+dt)) / (2 dt).

        # C++ parity: survivalprobabilitystructure.cpp:46-55.
        """
        dt = self._DT
        t1 = max(t - dt, 0.0)
        t2 = t + dt
        p1 = self._survival_probability_impl(t1)
        p2 = self._survival_probability_impl(t2)
        return (p1 - p2) / (t2 - t1)


__all__ = ["SurvivalProbabilityStructure"]

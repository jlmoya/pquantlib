"""MultidimIntegrator — tensor-product multi-dimensional integration.

# C++ parity: ql/experimental/math/multidimintegrator.{hpp,cpp} @ v1.42.1 (099987f0).

Integrates a scalar function of a vector domain using a collection of arbitrary
1-D integrators, one per dimension. The C++ class uses template recursion over
the dimensions (to avoid a runtime depth test); the Python port expresses the
same cross-section recursion directly: the outermost integrator integrates, over
its axis, the result of recursively integrating the remaining axes with the
current outer coordinate fixed.

This generalises the two-dimensional ``TwoDimensionalIntegral`` to an arbitrary
number of dimensions (the C++ ``maxDimensions_`` cap of 15 is enforced).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from pquantlib import qassert
from pquantlib.math.integrals.integrator import Integrator

_MAX_DIMENSIONS = 15


class MultidimIntegral:
    """Tensor-product integrator over a box, one 1-D integrator per axis."""

    __slots__ = ("_integrators",)

    def __init__(self, integrators: Sequence[Integrator]) -> None:
        qassert.require(
            len(integrators) <= _MAX_DIMENSIONS,
            "Too many dimensions in integration.",
        )
        self._integrators = list(integrators)

    def __call__(
        self,
        f: Callable[[Sequence[float]], float],
        a: Sequence[float],
        b: Sequence[float],
    ) -> float:
        """Integrate ``f`` over the box ``[a, b]`` (per-axis limits)."""
        qassert.require(
            len(a) == len(b) and len(b) == len(self._integrators),
            "Incompatible integration problem dimensions",
        )
        n = len(self._integrators)
        var_buffer = [0.0] * n

        def integrate(axis: int) -> float:
            # integrate over `axis`, recursing into lower axes (axis-1 .. 0).
            def cross_section(z: float) -> float:
                var_buffer[axis] = z
                if axis == 0:
                    return f(var_buffer)
                return integrate(axis - 1)

            return self._integrators[axis](cross_section, a[axis], b[axis])

        return integrate(n - 1)

"""Two-dimensional integration by nesting two 1-D integrators.

# C++ parity: ql/math/integrals/twodimensionalintegral.hpp (v1.43).

``int_{a_x}^{b_x} int_{a_y}^{b_y} f(x, y) dy dx`` is evaluated as an outer
integration in ``x`` whose integrand is itself an inner integration in ``y``.

Note that nesting two *adaptive* rules at the same tolerance does not
terminate in general: the inner rule's own error floors the outer rule's
refinement increment. Pair an adaptive outer rule with a fixed-node inner one
(or use fixed-node rules on both axes).
"""

from __future__ import annotations

from collections.abc import Callable

from pquantlib.math.integrals.integrator import Integrator


class TwoDimensionalIntegral:
    """Nest ``integrator_x`` over ``integrator_y``.

    # C++ parity: twodimensionalintegral.hpp:39-62.
    """

    __slots__ = ("_integrator_x", "_integrator_y")

    def __init__(self, integrator_x: Integrator, integrator_y: Integrator) -> None:
        self._integrator_x: Integrator = integrator_x
        self._integrator_y: Integrator = integrator_y

    def __call__(
        self,
        f: Callable[[float, float], float],
        a: tuple[float, float],
        b: tuple[float, float],
    ) -> float:
        # C++ parity: twodimensionalintegral.hpp:47-52.
        return self._integrator_x(
            lambda x: self._g(f, x, a[1], b[1]),
            a[0],
            b[0],
        )

    def _g(
        self, f: Callable[[float, float], float], x: float, a: float, b: float
    ) -> float:
        # C++ parity: twodimensionalintegral.hpp:55-58.
        return self._integrator_y(lambda y: f(x, y), a, b)


__all__ = ["TwoDimensionalIntegral"]

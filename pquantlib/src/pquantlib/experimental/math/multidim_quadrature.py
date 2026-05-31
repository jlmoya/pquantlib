"""MultidimGaussianQuadrature — tensor-product Gauss-Hermite quadrature.

# C++ parity: ql/experimental/math/multidimquadrature.{hpp,cpp} @ v1.42.1 (099987f0).
# class GaussianQuadMultidimIntegrator.

Integrates a scalar function of a vector domain over R^dim using a Gauss-Hermite
quadrature along each axis. The C++ class uses template recursion over the
dimensions; the Python port expresses the same nested cross-section recursion
directly.

The 1-D building block is QuantLib's ``GaussHermiteIntegration(order, mu)`` —
the generalised Gauss-Hermite rule with weight ``w(x) = |x|^(2 mu) exp(-x^2)``.
Crucially, QuantLib *divides the quadrature weights by* ``w(x_i)``
(``w_i = mu_0 * ev[0,i]^2 / w(x_i)``), so the rule approximates the plain
Lebesgue integral ``∫ f(x) dx`` (the Gaussian decay is assumed to be carried by
the integrand). This convention is reproduced here so the results match C++.

The nodes/weights are obtained from the symmetric tridiagonal Jacobi matrix of
the Hermite three-term recurrence (``alpha_i = 0``, ``beta_i`` per parity) via
the Golub-Welsch eigen-decomposition (``numpy.linalg.eigh``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import exp, gamma

import numpy as np

from pquantlib import qassert

_MAX_DIMENSIONS = 15


def _gauss_hermite_nodes_weights(
    order: int, mu: float
) -> tuple[list[float], list[float]]:
    """Nodes/weights of QuantLib's GaussHermiteIntegration(order, mu)."""
    # Hermite recurrence: alpha_i = 0; beta_i = i/2 (+ mu if i odd).
    alpha = np.zeros(order)
    beta = np.array(
        [(i / 2.0 + mu) if (i % 2) != 0 else (i / 2.0) for i in range(order)]
    )
    off = np.sqrt(beta[1:])
    jac = np.diag(alpha) + np.diag(off, 1) + np.diag(off, -1)
    evals, evecs = np.linalg.eigh(jac)
    mu_0 = gamma(mu + 0.5)
    nodes: list[float] = []
    weights: list[float] = []
    for i in range(order):
        x = float(evals[i])
        w_x = abs(x) ** (2.0 * mu) * exp(-x * x)
        nodes.append(x)
        weights.append(mu_0 * float(evecs[0, i]) ** 2 / w_x)
    return nodes, weights


class MultidimGaussianQuadrature:
    """Tensor-product Gauss-Hermite quadrature over ``R^dimension``."""

    __slots__ = ("_dimension", "_nodes", "_order", "_weights")

    def __init__(self, dimension: int, quad_order: int, mu: float = 0.0) -> None:
        qassert.require(
            dimension <= _MAX_DIMENSIONS,
            "Too many dimensions in integration.",
        )
        self._dimension = dimension
        self._order = quad_order
        self._nodes, self._weights = _gauss_hermite_nodes_weights(quad_order, mu)

    def order(self) -> int:
        """Quadrature order."""
        return self._order

    def __call__(self, f: Callable[[Sequence[float]], float]) -> float:
        """Integrate scalar ``f`` over ``R^dimension``."""
        n = self._dimension
        var_buffer = [0.0] * n
        nodes = self._nodes
        weights = self._weights
        order = self._order

        def integrate(axis: int) -> float:
            total = 0.0
            for k in range(order):
                var_buffer[axis] = nodes[k]
                inner = f(var_buffer) if axis == 0 else integrate(axis - 1)
                total += weights[k] * inner
            return total

        return integrate(n - 1)

"""Optimization cost function abstract class.

# C++ parity: ql/math/optimization/costfunction.hpp (v1.42.1).

C++'s ``CostFunction`` is an abstract class with one pure-virtual
``values(x)`` (vector residuals for nonlinear least squares) and a
default ``value(x) = sqrt(mean(values(x)^2))`` (i.e. the RMS of the
residuals). The Python port keeps the same shape: subclasses must
override ``values`` and may optionally override ``value`` and
``gradient`` for analytic derivatives.

``valueAndGradient`` is ported: it is the *only* entry point L-BFGS-B
uses to reach the objective, and subclasses override it to compute
value and gradient in one pass when the two share work.

The remaining higher-order methods (``jacobian``, ``valuesAndJacobian``,
``ParametersTransformation``, the templated ``SimpleCostFunction``) are
still deferred — they are only needed by Levenberg-Marquardt, carved
out of L1-D.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class CostFunction(ABC):
    """Cost-function abstract base.

    # C++ parity: ``class CostFunction`` in
    # ql/math/optimization/costfunction.hpp:34-97 (v1.42.1).

    Subclasses must override ``values`` (the vector of residuals);
    ``value`` defaults to ``sqrt(mean(values(x)^2))`` matching the
    C++ default implementation.
    """

    @abstractmethod
    def values(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return the vector of residuals at ``x``."""
        ...

    def value(self, x: npt.NDArray[np.float64]) -> float:
        """Return ``sqrt(mean(values(x)^2))``.

        # C++ parity: costfunction.hpp:38-43 — default ``value`` impl.
        """
        v = self.values(x)
        return float(np.sqrt(np.sum(v * v) / v.size))

    def gradient(self, grad: npt.NDArray[np.float64], x: npt.NDArray[np.float64]) -> None:
        """Central-difference gradient of ``value`` at ``x``, into ``grad``.

        # C++ parity: costfunction.hpp:49-60 — default ``gradient`` impl.

        ``grad`` is mutated in place to match the C++ in-out parameter
        style (the caller pre-allocates the gradient buffer).
        """
        eps = self.finite_difference_epsilon()
        xx = x.astype(np.float64, copy=True)
        for i in range(x.size):
            xx[i] += eps
            fp = self.value(xx)
            xx[i] -= 2.0 * eps
            fm = self.value(xx)
            grad[i] = 0.5 * (fp - fm) / eps
            xx[i] = x[i]

    def value_and_gradient(self, grad: npt.NDArray[np.float64], x: npt.NDArray[np.float64]) -> float:
        """Store the gradient at ``x`` into ``grad`` and return ``value(x)``.

        # C++ parity: costfunction.hpp:64-68 — default
        # ``valueAndGradient`` impl, ``gradient(grad, x); return value(x);``.

        Subclasses that can compute both in one pass override this. Note
        that the default routes through ``self.gradient`` / ``self.value``
        directly, so a subclass supplying only ``value`` gets the
        central-difference gradient here — and those inner evaluations
        never reach ``Problem``, so they do not move its counters.
        """
        self.gradient(grad, x)
        return self.value(x)

    def finite_difference_epsilon(self) -> float:
        """Step size for the central-difference gradient (default 1e-8).

        # C++ parity: costfunction.hpp:96 — ``finiteDifferenceEpsilon``.
        """
        return 1e-8

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

``jacobian`` and ``valuesAndJacobian`` are ported (central differences
with ``finiteDifferenceEpsilon()``, matching costfunction.hpp:72-93),
along with ``SimpleCostFunction`` — which C++ needs as a template only
because it stores the functor without type erasure — and the
``ParametersTransformation`` interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

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

    def jacobian(self, jac: npt.NDArray[np.float64], x: npt.NDArray[np.float64]) -> None:
        """Central-difference Jacobian of ``values`` at ``x``, into ``jac``.

        # C++ parity: costfunction.hpp:72-85 — default ``jacobian`` impl.

        ``jac`` is mutated in place and must be shaped
        ``(len(values(x)), len(x))``; ``jac[j][i]`` is d values_j / d x_i.
        Note the C++ default is a CENTRAL difference (order 2), unlike the
        forward difference MINPACK's ``fdjac2`` computes inside
        ``LevenbergMarquardt`` — that asymmetry is deliberate upstream and
        is documented in levenbergmarquardt.hpp:40-45.
        """
        eps = self.finite_difference_epsilon()
        xx = x.astype(np.float64, copy=True)
        for i in range(x.size):
            xx[i] += eps
            fp = self.values(xx)
            xx[i] -= 2.0 * eps
            fm = self.values(xx)
            for j in range(fp.size):
                jac[j][i] = 0.5 * (fp[j] - fm[j]) / eps
            xx[i] = x[i]

    def values_and_jacobian(
        self, jac: npt.NDArray[np.float64], x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Store the Jacobian at ``x`` into ``jac`` and return ``values(x)``.

        # C++ parity: costfunction.hpp:89-93 — default
        # ``valuesAndJacobian`` impl, ``jacobian(jac, x); return values(x);``.
        """
        self.jacobian(jac, x)
        return self.values(x)

    def finite_difference_epsilon(self) -> float:
        """Step size for the central-difference gradient (default 1e-8).

        # C++ parity: costfunction.hpp:96 — ``finiteDifferenceEpsilon``.
        """
        return 1e-8


class SimpleCostFunction(CostFunction):
    """Cost function built from a plain ``values`` callable.

    # C++ parity: ``template <class ValuesFn> class SimpleCostFunction``
    # in ql/math/optimization/costfunction.hpp:99-107 (v1.43).

    C++ needs the template because it stores the functor by value with
    no type erasure; Python stores the callable directly. Everything
    else — ``value``, ``gradient``, ``jacobian`` — comes from the
    ``CostFunction`` defaults, exactly as in C++.
    """

    __slots__ = ("_values_fn",)

    def __init__(
        self,
        values_fn: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    ) -> None:
        self._values_fn: Callable[
            [npt.NDArray[np.float64]], npt.NDArray[np.float64]
        ] = values_fn

    def values(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # C++ parity: costfunction.hpp:104.
        return self._values_fn(x)


class ParametersTransformation(ABC):
    """Bijection between an optimizer's search space and a model's parameters.

    # C++ parity: ``class ParametersTransformation`` in
    # ql/math/optimization/costfunction.hpp:109-114 (v1.43).

    ``direct`` maps unconstrained search coordinates onto the model
    parameters; ``inverse`` maps back. Both are pure-virtual in C++ with
    no default implementation.
    """

    @abstractmethod
    def direct(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Map search coordinates ``x`` onto model parameters."""
        ...

    @abstractmethod
    def inverse(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Map model parameters ``x`` back onto search coordinates."""
        ...

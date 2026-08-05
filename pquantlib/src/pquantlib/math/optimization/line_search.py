"""Line-search abstract base.

# C++ parity: ql/math/optimization/linesearch.{hpp,cpp} (v1.43).

A ``LineSearch`` is the inner one-dimensional minimizer that
``LineSearchBasedMethod`` (and hence ``SteepestDescent``,
``ConjugateGradient`` and ``BFGS``) drives along a search direction. It
owns four pieces of mutable state that the outer loop reads back after
every call:

- ``last_x`` — the accepted point ``x + t*d``;
- ``last_function_value`` — ``f`` there;
- ``last_gradient`` — ``grad f`` there;
- ``last_gradient_norm2`` — ``|grad f|^2`` there;

plus ``succeed``, the flag the outer loop uses to decide whether to keep
iterating, and ``search_direction``, which the outer loop WRITES before
each call and the subclass reads.

Note the constructor's ``eps`` argument: C++ declares
``explicit LineSearch(Real = 0.0) {}`` — the parameter is unnamed and
therefore discarded. ``ArmijoLineSearch`` and ``GoldsteinLineSearch``
both forward their own ``eps`` into it, so their ``eps`` is dead too.
The Python port keeps the parameter for API parity and documents that it
does nothing rather than silently inventing a use for it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from pquantlib import qassert

if TYPE_CHECKING:
    from pquantlib.math.optimization.constraint import Constraint
    from pquantlib.math.optimization.end_criteria import EndCriteria, Type
    from pquantlib.math.optimization.problem import Problem


def dot_product(v1: npt.NDArray[np.float64], v2: npt.NDArray[np.float64]) -> float:
    """Sequential inner product of two vectors.

    # C++ parity: ``DotProduct`` in ql/math/array.hpp (v1.43), which is
    # ``std::inner_product(v1.begin(), v1.end(), v2.begin(), Real(0.0))``
    # — a strictly left-to-right accumulation.

    This is deliberately NOT ``np.dot``. BLAS ``ddot`` sums with several
    unrolled accumulators, so for anything past a handful of elements it
    rounds differently from ``std::inner_product``. The line-search based
    optimizers compare gradient norms against tolerances near machine
    precision and publish their iterate trajectory, so the summation
    order is observable.
    """
    total = 0.0
    for i in range(v1.size):
        total += float(v1[i]) * float(v2[i])
    return total


class LineSearch(ABC):
    """Base class for line searches.

    # C++ parity: ``class LineSearch`` in
    # ql/math/optimization/linesearch.hpp:38-78 (v1.43).
    """

    __slots__ = ("_gradient", "_qpt", "_qt", "_search_direction", "_succeed", "_xtd")

    def __init__(self, eps: float = 0.0) -> None:
        # C++ parity: linesearch.hpp:41 — the parameter is UNNAMED in C++
        # and the body is empty, so ``eps`` is accepted and discarded.
        del eps
        self._search_direction: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self._xtd: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self._gradient: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self._qt: float = 0.0
        self._qpt: float = 0.0
        self._succeed: bool = True

    # --- inspectors -----------------------------------------------------

    @property
    def last_x(self) -> npt.NDArray[np.float64]:
        """Last accepted point. # C++ parity: linesearch.hpp:46 ``lastX``."""
        return self._xtd

    @property
    def last_function_value(self) -> float:
        """Cost at ``last_x``. # C++ parity: linesearch.hpp:48."""
        return self._qt

    @property
    def last_gradient(self) -> npt.NDArray[np.float64]:
        """Gradient at ``last_x``. # C++ parity: linesearch.hpp:50."""
        return self._gradient

    @property
    def last_gradient_norm2(self) -> float:
        """Squared gradient norm at ``last_x``. # C++ parity: linesearch.hpp:52."""
        return self._qpt

    @property
    def succeed(self) -> bool:
        """False once the search gave up on max iterations.

        # C++ parity: linesearch.hpp:54 ``succeed()``.
        """
        return self._succeed

    @property
    def search_direction(self) -> npt.NDArray[np.float64]:
        """Current search direction (read/write).

        # C++ parity: linesearch.hpp:67-68 — C++ exposes both a const and
        # a non-const ``searchDirection()``; the non-const one is how
        # ``LineSearchBasedMethod`` installs each new direction.
        """
        return self._search_direction

    @search_direction.setter
    def search_direction(self, direction: npt.NDArray[np.float64]) -> None:
        self._search_direction = direction.astype(np.float64, copy=True)

    # --- operations -----------------------------------------------------

    @abstractmethod
    def __call__(
        self,
        problem: Problem,
        ec_type: Type,
        end_criteria: EndCriteria,
        t_ini: float,
    ) -> tuple[float, Type]:
        """Perform the line search from ``problem.current_value``.

        # C++ parity: linesearch.hpp:57-60 — ``operator()``.

        C++ returns the step ``t`` and writes the fired criterion into the
        ``EndCriteria::Type&`` out-parameter. Python has no out-parameters,
        so ``ec_type`` is passed in and the (possibly updated) value is
        returned alongside ``t``.
        """
        ...

    def update(
        self,
        params: npt.NDArray[np.float64],
        direction: npt.NDArray[np.float64],
        beta: float,
        constraint: Constraint,
    ) -> float:
        """Step ``params`` along ``direction``, halving ``beta`` until feasible.

        # C++ parity: linesearch.cpp:26-45 — ``LineSearch::update``. This
        # duplicates ``Constraint::update`` (constraint.cpp:27-43) in
        # v1.43; the two are byte-for-byte the same algorithm with a
        # different failure message, and the duplication is preserved here
        # so a caller-visible difference in the message stays faithful.

        ``params`` is mutated in place; the accepted step length is
        returned.
        """
        diff = beta
        new_params = params + diff * direction
        valid = constraint.test(new_params)
        icount = 0
        while not valid:
            if icount > 200:
                qassert.fail("can't update linesearch")
            diff *= 0.5
            icount += 1
            new_params = params + diff * direction
            valid = constraint.test(new_params)
        params += diff * direction
        return diff

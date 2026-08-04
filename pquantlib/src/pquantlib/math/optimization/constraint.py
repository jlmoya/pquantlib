"""Optimization constraint hierarchy.

# C++ parity: ql/math/optimization/constraint.hpp (v1.42.1).

The C++ implementation uses the PIMPL pattern (``Constraint::Impl``
abstract inner class held by ``shared_ptr``). Python idiom collapses
that to a single abstract base ``Constraint`` whose subclasses
override ``test``, ``upper_bound``, ``lower_bound`` directly — there
is no need for a separate Impl layer.

L1-D ported the simple constraints used by the optimization
scaffolding: ``NoConstraint``, ``PositiveConstraint``,
``BoundaryConstraint``. ``NonhomogeneousBoundaryConstraint`` follows
here — it is the per-coordinate box that ``LBFGSB`` reads its bounds
from. ``CompositeConstraint`` is still deferred to a later cluster
that needs it.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from pquantlib import qassert

# C++ ``std::numeric_limits<double>::max()`` — used as the sentinel
# "unbounded" value when ``upperBound`` / ``lowerBound`` are not
# subclass-overridden.
_REAL_MAX: float = sys.float_info.max


class Constraint(ABC):
    """Abstract base for optimization constraints.

    # C++ parity: ``class Constraint`` (and its inner ``Impl``) in
    # ql/math/optimization/constraint.hpp (v1.42.1).

    The C++ class is concrete-but-empty (``empty()`` returns true when
    no Impl is set); pquantlib makes it abstract because Python's
    structural typing has no analogue to the "empty PIMPL"
    sentinel state. Callers that want an unconstrained problem use
    ``NoConstraint`` directly.
    """

    @abstractmethod
    def test(self, params: npt.NDArray[np.float64]) -> bool:
        """Return True iff ``params`` satisfies the constraint."""
        ...

    def upper_bound(self, params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Element-wise upper bound (defaults to +max for every component).

        # C++ parity: ``Constraint::Impl::upperBound`` default impl,
        # ql/math/optimization/constraint.hpp:44-47 (v1.42.1).
        """
        result = np.full(params.shape, _REAL_MAX, dtype=np.float64)
        qassert.require(
            params.size == result.size,
            f"upper bound size ({result.size}) not equal to params size ({params.size})",
        )
        return result

    def lower_bound(self, params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Element-wise lower bound (defaults to -max for every component).

        # C++ parity: ``Constraint::Impl::lowerBound`` default impl,
        # ql/math/optimization/constraint.hpp:49-52 (v1.42.1).
        """
        result = np.full(params.shape, -_REAL_MAX, dtype=np.float64)
        qassert.require(
            params.size == result.size,
            f"lower bound size ({result.size}) not equal to params size ({params.size})",
        )
        return result


class NoConstraint(Constraint):
    """Unconstrained — always satisfied.

    # C++ parity: ``class NoConstraint`` in
    # ql/math/optimization/constraint.hpp:79-89 (v1.42.1).
    """

    def test(self, params: npt.NDArray[np.float64]) -> bool:
        return True


class PositiveConstraint(Constraint):
    """All components strictly positive (``p_i > 0``).

    # C++ parity: ``class PositiveConstraint`` in
    # ql/math/optimization/constraint.hpp:92-111 (v1.42.1).
    """

    def test(self, params: npt.NDArray[np.float64]) -> bool:
        return bool(np.all(params > 0.0))

    def lower_bound(self, params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # C++ parity: constraint.hpp:103-105 — zero lower bound.
        return np.zeros(params.shape, dtype=np.float64)


class BoundaryConstraint(Constraint):
    """All components in the closed interval ``[low, high]``.

    # C++ parity: ``class BoundaryConstraint`` in
    # ql/math/optimization/constraint.hpp:114-137 (v1.42.1).
    """

    __slots__ = ("_high", "_low")

    def __init__(self, low: float, high: float) -> None:
        self._low: float = low
        self._high: float = high

    def test(self, params: npt.NDArray[np.float64]) -> bool:
        return bool(np.all((params >= self._low) & (params <= self._high)))

    def upper_bound(self, params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.full(params.shape, self._high, dtype=np.float64)

    def lower_bound(self, params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.full(params.shape, self._low, dtype=np.float64)


class NonhomogeneousBoundaryConstraint(Constraint):
    """The i-th component lies in its own interval ``[low_i, high_i]``.

    # C++ parity: ``class NonhomogeneousBoundaryConstraint`` in
    # ql/math/optimization/constraint.hpp:176-203 (v1.43).

    Unlike ``BoundaryConstraint`` the interval varies per coordinate, so
    the bounds are arrays rather than scalars. Use ``+/-sys.float_info.max``
    for a coordinate that is unbounded on that side — that is the sentinel
    the default ``Constraint`` returns, and the one ``LBFGSB`` recognises.
    """

    __slots__ = ("_high", "_low")

    def __init__(self, low: npt.NDArray[np.float64], high: npt.NDArray[np.float64]) -> None:
        low_arr = np.ascontiguousarray(low, dtype=np.float64)
        high_arr = np.ascontiguousarray(high, dtype=np.float64)
        qassert.require(
            low_arr.size == high_arr.size,
            "Upper and lower boundaries sizes are inconsistent.",
        )
        self._low: npt.NDArray[np.float64] = low_arr
        self._high: npt.NDArray[np.float64] = high_arr

    def test(self, params: npt.NDArray[np.float64]) -> bool:
        qassert.require(
            params.size == self._low.size,
            "Number of parameters and boundaries sizes are inconsistent.",
        )
        return bool(np.all((params >= self._low) & (params <= self._high)))

    def upper_bound(self, params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # C++ parity: constraint.hpp:194 — the stored array, ignoring ``params``.
        del params
        return self._high

    def lower_bound(self, params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # C++ parity: constraint.hpp:195 — the stored array, ignoring ``params``.
        del params
        return self._low

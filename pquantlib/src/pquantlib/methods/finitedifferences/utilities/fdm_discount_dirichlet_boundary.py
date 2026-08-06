"""FdmDiscountDirichletBoundary — a discounted cash flow on one grid face.

# C++ parity: ql/methods/finitedifferences/utilities/fdmdiscountdirichletboundary.{hpp,cpp}
# @ v1.43 (6b57206e0).

Thin adapter over :class:`FdmTimeDepDirichletBoundary` whose boundary
function is the anonymous-namespace ``DiscountedCashflowAtBoundary``
functor::

    f(t) = cashFlow * rTS->discount(maturityTime) / rTS->discount(t)

i.e. the terminal cash flow rolled back to ``t`` on the risk-free curve.
C++ implements this by *holding* an ``FdmTimeDepDirichletBoundary`` and
forwarding all five virtuals; the Python port keeps the same composition
rather than inheriting, so the delegation is one-to-one with the C++
source.
"""

from __future__ import annotations

from typing import final

from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.fdm_boundary_condition import (
    BoundaryConditionSide,
    FdmBoundaryCondition,
)
from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.utilities.fdm_time_dep_dirichlet_boundary import (
    FdmTimeDepDirichletBoundary,
)
from pquantlib.termstructures.yield_term_structure import YieldTermStructure


@final
class _DiscountedCashflowAtBoundary:
    """``cashFlow * P(maturity) / P(t)``.

    # C++ parity: anonymous-namespace ``class DiscountedCashflowAtBoundary``
    # in fdmdiscountdirichletboundary.cpp.
    """

    __slots__ = ("_cash_flow", "_maturity_time", "_r_ts")

    def __init__(
        self, maturity_time: float, value_on_boundary: float, r_ts: YieldTermStructure
    ) -> None:
        self._maturity_time: float = maturity_time
        self._cash_flow: float = value_on_boundary
        self._r_ts: YieldTermStructure = r_ts

    def __call__(self, t: float) -> float:
        return self._cash_flow * self._r_ts.discount(self._maturity_time) / self._r_ts.discount(t)


@final
class FdmDiscountDirichletBoundary(FdmBoundaryCondition):
    """Dirichlet face pinned to a discounted terminal cash flow.

    # C++ parity: ``class FdmDiscountDirichletBoundary : public
    # BoundaryCondition<FdmLinearOp>``.
    """

    def __init__(
        self,
        mesher: FdmMesher,
        r_ts: YieldTermStructure,
        maturity_time: float,
        value_on_boundary: float,
        direction: int,
        side: BoundaryConditionSide,
    ) -> None:
        self._bc: FdmTimeDepDirichletBoundary = FdmTimeDepDirichletBoundary(
            mesher,
            direction,
            side,
            value_on_boundary=_DiscountedCashflowAtBoundary(
                maturity_time, value_on_boundary, r_ts
            ),
        )

    def set_time(self, t: float) -> None:
        """# C++ parity: ``setTime(Time)`` — forwards to the held bc."""
        self._bc.set_time(t)

    def apply_before_applying(self, op: object) -> None:
        """# C++ parity: forwards to the held bc."""
        self._bc.apply_before_applying(op)

    def apply_before_solving(self, op: object, rhs: Array) -> None:
        """# C++ parity: forwards to the held bc."""
        self._bc.apply_before_solving(op, rhs)

    def apply_after_applying(self, u: Array) -> None:
        """# C++ parity: forwards to the held bc."""
        self._bc.apply_after_applying(u)

    def apply_after_solving(self, u: Array) -> None:
        """# C++ parity: forwards to the held bc."""
        self._bc.apply_after_solving(u)


__all__ = ["FdmDiscountDirichletBoundary"]

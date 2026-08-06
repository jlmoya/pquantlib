"""Boundary condition base for the modern (Fdm*) finite-difference framework.

# C++ parity: ql/methods/finitedifferences/boundarycondition.hpp
# (v1.43) — ``template <class Operator> class BoundaryCondition``,
# instantiated as ``BoundaryCondition<FdmLinearOp>``; and
# ql/methods/finitedifferences/utilities/fdmboundaryconditionset.hpp
# — ``typedef OperatorTraits<FdmLinearOp>::bc_set FdmBoundaryConditionSet``.

The C++ class template is parameterised on the operator type. Only one
instantiation matters for the modern framework — ``FdmLinearOp`` — so the
Python port is a plain ABC over ``FdmLinearOp`` / ``Array``.

The module is named ``fdm_boundary_condition`` rather than
``boundary_condition`` because the retired pre-1.0 FD framework (hosted in
``pquantlib-helpers``) already owns that name for its
``BoundaryCondition<TridiagonalOperator>`` instantiation; the two are
unrelated hierarchies in C++ too, sharing only the template.

``FdmBoundaryConditionSet`` is a C++ ``typedef`` for
``std::vector<shared_ptr<BoundaryCondition<FdmLinearOp>>>``; Python spells
it as a type alias over ``Sequence``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import IntEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pquantlib.math.array import Array


class BoundaryConditionSide(IntEnum):
    """Which end of the grid a boundary condition applies to.

    # C++ parity: ``BoundaryCondition<Operator>::Side`` —
    # ``enum Side { None, Upper, Lower }``. Renamed from the nested C++
    # ``Side`` because ``None`` is a Python keyword; the enumerator is
    # spelled ``NONE`` and the enum is module-level.
    """

    NONE = 0
    UPPER = 1
    LOWER = 2


class FdmBoundaryCondition(ABC):
    """Abstract boundary condition for an ``FdmLinearOp``-based problem.

    # C++ parity: ``class BoundaryCondition<FdmLinearOp>``.
    """

    Side = BoundaryConditionSide

    @abstractmethod
    def apply_before_applying(self, op: object) -> None:
        """Modify ``op`` before it is applied so ``v = L u`` satisfies the condition.

        # C++ parity: ``applyBeforeApplying(operator_type&)``.
        """

    @abstractmethod
    def apply_after_applying(self, u: Array) -> None:
        """Modify ``u`` in place so it satisfies the condition.

        # C++ parity: ``applyAfterApplying(array_type&)``.
        """

    @abstractmethod
    def apply_before_solving(self, op: object, rhs: Array) -> None:
        """Modify ``op``/``rhs`` before solving ``L u' = u``.

        # C++ parity: ``applyBeforeSolving(operator_type&, array_type&)``.
        """

    @abstractmethod
    def apply_after_solving(self, u: Array) -> None:
        """Modify ``u`` in place so it satisfies the condition.

        # C++ parity: ``applyAfterSolving(array_type&)``.
        """

    @abstractmethod
    def set_time(self, t: float) -> None:
        """Set the current time for time-dependent boundary conditions.

        # C++ parity: ``setTime(Time)``.
        """


#: # C++ parity: ``FdmBoundaryConditionSet`` in
#: ql/methods/finitedifferences/utilities/fdmboundaryconditionset.hpp.
FdmBoundaryConditionSet = Sequence[FdmBoundaryCondition]


__all__ = [
    "BoundaryConditionSide",
    "FdmBoundaryCondition",
    "FdmBoundaryConditionSet",
]

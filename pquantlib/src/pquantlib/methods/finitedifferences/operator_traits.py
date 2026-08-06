"""OperatorTraits — the type bundle a finite-difference operator induces.

# C++ parity: ql/methods/finitedifferences/operatortraits.hpp (v1.43).

``OperatorTraits<Operator>`` names five types that the FD machinery reads off
its operator: the operator itself, its array type, the matching boundary
condition, a set of those, and the matching step condition. In C++ they are
consumed at compile time by ``FiniteDifferenceModel``, ``MixedScheme`` and
friends; Python resolves the same things at runtime, so the class is spelled
as a namespace of type aliases plus the two aliases that *are* load-bearing at
runtime (``bc_set`` and ``condition_type``, which appear in signatures).

Two instantiations exist in the library, and both are provided:

* ``FdmOperatorTraits`` — ``OperatorTraits<FdmLinearOp>``, the modern
  framework, whose ``bc_set`` is the ``FdmBoundaryConditionSet`` of
  ``utilities/fdmboundaryconditionset.hpp``.
* the retired ``OperatorTraits<TridiagonalOperator>`` lives with the rest of
  the pre-1.0 framework in ``pquantlib-helpers`` and is not restated here.
"""

from __future__ import annotations

from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.fdm_boundary_condition import (
    FdmBoundaryCondition,
    FdmBoundaryConditionSet,
)
from pquantlib.methods.finitedifferences.operators.fdm_linear_op import FdmLinearOp
from pquantlib.methods.finitedifferences.step_conditions.step_condition import (
    StepCondition,
)


class OperatorTraits:
    """Type bundle induced by a finite-difference operator.

    # C++ parity: ``template <class Operator> class OperatorTraits``.
    """

    #: # C++ parity: ``typedef Operator operator_type``.
    operator_type: type[FdmLinearOp] = FdmLinearOp
    #: # C++ parity: ``typedef typename Operator::array_type array_type``.
    array_type = Array
    #: # C++ parity: ``typedef BoundaryCondition<operator_type> bc_type``.
    bc_type: type[FdmBoundaryCondition] = FdmBoundaryCondition
    #: # C++ parity: ``typedef std::vector<shared_ptr<bc_type>> bc_set``.
    bc_set = FdmBoundaryConditionSet
    #: # C++ parity: ``typedef StepCondition<array_type> condition_type``.
    condition_type: type[StepCondition] = StepCondition


#: The one instantiation the modern framework uses.
#: # C++ parity: ``OperatorTraits<FdmLinearOp>``.
FdmOperatorTraits = OperatorTraits


__all__ = ["FdmOperatorTraits", "OperatorTraits"]

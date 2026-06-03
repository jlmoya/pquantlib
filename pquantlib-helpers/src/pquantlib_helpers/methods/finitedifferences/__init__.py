"""Legacy (retired pre-1.0 QuantLib) finite-difference linear-algebra core.

# Retired-API compat layer — NOT a port of C++ QuantLib v1.42.1's *modern*
# Fdm* framework (that lives in pquantlib core). This is the old FD framework
# the dividend-option helpers depend on; the v1.42.1 headers for these classes
# are either deprecated stubs (``bsmoperator.hpp``) or marked
# ``[[deprecated]]`` (``mixedscheme.hpp`` / ``cranknicolson.hpp`` /
# ``expliciteuler.hpp`` / ``finitedifferencemodel.hpp``).

Java parity: ``org.jquantlib.methods.finitedifferences`` (jquantlib).
"""

from __future__ import annotations

from pquantlib_helpers.methods.finitedifferences.boundary_condition import (
    BoundaryConditionSet,
    DirichletBC,
    NeumannBC,
    Side,
)
from pquantlib_helpers.methods.finitedifferences.bsm_operator import BSMOperator
from pquantlib_helpers.methods.finitedifferences.crank_nicolson import CrankNicolson
from pquantlib_helpers.methods.finitedifferences.d_minus import DMinus
from pquantlib_helpers.methods.finitedifferences.d_plus import DPlus
from pquantlib_helpers.methods.finitedifferences.d_plus_minus import DPlusMinus
from pquantlib_helpers.methods.finitedifferences.d_zero import DZero
from pquantlib_helpers.methods.finitedifferences.explicit_euler import ExplicitEuler
from pquantlib_helpers.methods.finitedifferences.finite_difference_model import (
    FiniteDifferenceModel,
    StandardFiniteDifferenceModel,
    StepCondition,
)
from pquantlib_helpers.methods.finitedifferences.mixed_scheme import (
    BoundaryCondition,
    MixedScheme,
)
from pquantlib_helpers.methods.finitedifferences.operator import Operator
from pquantlib_helpers.methods.finitedifferences.pde import (
    Pde,
    PdeBSM,
    PdeConstantCoeff,
    PdeSecondOrderParabolic,
)
from pquantlib_helpers.methods.finitedifferences.pde_operator import (
    BSMTermOperator,
    GenericTimeSetter,
    OperatorFactory,
    PdeOperator,
)
from pquantlib_helpers.methods.finitedifferences.step_condition import (
    AmericanCondition,
    CurveDependentStepCondition,
    NullCondition,
    StepConditionSet,
)
from pquantlib_helpers.methods.finitedifferences.transformed_grid import (
    LogGrid,
    TransformedGrid,
)
from pquantlib_helpers.methods.finitedifferences.tridiagonal_operator import (
    TimeSetter,
    TridiagonalOperator,
)

__all__ = [
    "AmericanCondition",
    "BSMOperator",
    "BSMTermOperator",
    "BoundaryCondition",
    "BoundaryConditionSet",
    "CrankNicolson",
    "CurveDependentStepCondition",
    "DMinus",
    "DPlus",
    "DPlusMinus",
    "DZero",
    "DirichletBC",
    "ExplicitEuler",
    "FiniteDifferenceModel",
    "GenericTimeSetter",
    "LogGrid",
    "MixedScheme",
    "NeumannBC",
    "NullCondition",
    "Operator",
    "OperatorFactory",
    "Pde",
    "PdeBSM",
    "PdeConstantCoeff",
    "PdeOperator",
    "PdeSecondOrderParabolic",
    "Side",
    "StandardFiniteDifferenceModel",
    "StepCondition",
    "StepConditionSet",
    "TimeSetter",
    "TransformedGrid",
    "TridiagonalOperator",
]

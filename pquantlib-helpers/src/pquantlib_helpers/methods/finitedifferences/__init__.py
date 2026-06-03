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
)
from pquantlib_helpers.methods.finitedifferences.mixed_scheme import MixedScheme
from pquantlib_helpers.methods.finitedifferences.operator import Operator
from pquantlib_helpers.methods.finitedifferences.tridiagonal_operator import (
    TridiagonalOperator,
)

__all__ = [
    "BSMOperator",
    "CrankNicolson",
    "DMinus",
    "DPlus",
    "DPlusMinus",
    "DZero",
    "ExplicitEuler",
    "FiniteDifferenceModel",
    "MixedScheme",
    "Operator",
    "StandardFiniteDifferenceModel",
    "TridiagonalOperator",
]

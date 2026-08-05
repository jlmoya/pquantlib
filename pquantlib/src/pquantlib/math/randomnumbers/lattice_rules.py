"""Rank-1 lattice-rule generating vectors.

# C++ parity: ql/math/randomnumbers/latticerules.{hpp,cpp} (v1.43) —
# ``class LatticeRule``.

Cools, Kuo and Nuyens, "Constructing embedded lattice rules for multivariate
integration", SIAM J. Sci. Comp. 28 (2006). Four families (A, B, C, D) of
3 600 generating integers each are tabulated; ``get_rule`` copies one out.

The four vectors live as text resources next to this module (see
``sobol_tables``) because 14 400 literals do not belong in a source file.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Final

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.randomnumbers.sobol_tables import lattice_rule_vector

#: # C++ parity: ``Size ruleLength = 3600`` (latticerules.cpp:14462).
RULE_LENGTH: Final[int] = 3600


class LatticeRule:
    """The four tabulated lattice rules.

    # C++ parity: ``LatticeRule`` (latticerules.hpp:36-46) — a namespace with
    # one static function, kept as a class so the C++ name survives.
    """

    class Type(IntEnum):
        """# C++ parity: ``LatticeRule::type`` (latticerules.hpp:40)."""

        A = 0
        B = 1
        C = 2
        D = 3

    @staticmethod
    def get_rule(name: Type, n: int) -> Array:
        """Generating vector of rule ``name`` for ``n`` points.

        # C++ parity: ``LatticeRule::getRule`` (latticerules.cpp:14458-14493).

        ``n`` selects nothing — the same 3 600 integers come back for every
        admissible ``n``; the argument exists only for the range check, which
        C++ spells as ``N >= 1024 && N <= std::pow(2.9, 20)``. That bound is
        reproduced verbatim, oddity included: the comment says "2 to the 20"
        but the code writes 2.9, and the promised "check that N is a power of
        2" was never written.
        """
        qassert.require(
            n >= 1024 and n <= 2.9**20,
            "N must be between 2 to 10 and 2 to the 20 for these lattice rules ",
        )
        row = lattice_rule_vector(name.name.lower())
        qassert.require(
            len(row) == RULE_LENGTH,
            f"lattice rule {name.name} has {len(row)} entries, expected {RULE_LENGTH}",
        )
        return np.array(row, dtype=np.float64)


__all__ = ["RULE_LENGTH", "LatticeRule"]

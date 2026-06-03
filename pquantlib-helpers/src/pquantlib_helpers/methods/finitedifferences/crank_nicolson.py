"""CrankNicolson — theta=0.5 specialisation of MixedScheme.

# Retired-API compat layer — see package docstring.

Java parity: ``org.jquantlib.methods.finitedifferences.CrankNicolson<T>``.
C++ parity: ``ql/methods/finitedifferences/cranknicolson.hpp`` (v1.42.1,
``[[deprecated]]``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib_helpers.methods.finitedifferences.mixed_scheme import MixedScheme

if TYPE_CHECKING:
    from pquantlib_helpers.methods.finitedifferences.mixed_scheme import (
        BoundaryCondition,
    )
    from pquantlib_helpers.methods.finitedifferences.tridiagonal_operator import (
        TridiagonalOperator,
    )


class CrankNicolson(MixedScheme):
    """Crank-Nicolson scheme (``theta = 0.5``)."""

    def __init__(
        self,
        op: TridiagonalOperator,
        bcs: list[BoundaryCondition] | None = None,
    ) -> None:
        """Build a Crank-Nicolson evolver around ``op``."""
        super().__init__(op, 0.5, bcs)


__all__ = ["CrankNicolson"]

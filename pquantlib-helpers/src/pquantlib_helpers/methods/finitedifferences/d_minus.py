r"""DMinus — first-derivative backward-difference operator :math:`D_{-}`.

# Retired-API compat layer — see package docstring.

Discretises the first derivative with the first-order backward formula
:math:`\frac{\partial u_i}{\partial x} \approx \frac{u_i-u_{i-1}}{h} = D_{-} u_i`.

Java parity: ``org.jquantlib.methods.finitedifferences.DMinus``.
C++ parity: ``ql/methods/finitedifferences/dminus.hpp`` (old QuantLib).
"""

from __future__ import annotations

from pquantlib_helpers.methods.finitedifferences.tridiagonal_operator import (
    TridiagonalOperator,
)


class DMinus(TridiagonalOperator):
    r""":math:`D_{-}` matricial representation (backward first difference)."""

    def __init__(self, grid_points: int, h: float) -> None:
        """Build the backward-difference operator on ``grid_points`` nodes, step ``h``."""
        super().__init__(grid_points)
        # linear extrapolation at the boundaries.
        self.set_first_row(-1.0 / h, 1.0 / h)
        self.set_mid_rows(-1.0 / h, 1.0 / h, 0.0)
        self.set_last_row(-1.0 / h, 1.0 / h)


__all__ = ["DMinus"]

r"""DPlus — first-derivative forward-difference operator :math:`D_{+}`.

# Retired-API compat layer — see package docstring.

Discretises the first derivative with the first-order forward formula
:math:`\frac{\partial u_i}{\partial x} \approx \frac{u_{i+1}-u_i}{h} = D_{+} u_i`.

Java parity: ``org.jquantlib.methods.finitedifferences.DPlus``.
C++ parity: ``ql/methods/finitedifferences/dplus.hpp`` (old QuantLib).
"""

from __future__ import annotations

from pquantlib_helpers.methods.finitedifferences.tridiagonal_operator import (
    TridiagonalOperator,
)


class DPlus(TridiagonalOperator):
    r""":math:`D_{+}` matricial representation (forward first difference)."""

    def __init__(self, grid_points: int, h: float) -> None:
        """Build the forward-difference operator on ``grid_points`` nodes, step ``h``."""
        super().__init__(grid_points)
        # linear extrapolation at the boundaries.
        self.set_first_row(-1.0 / h, 1.0 / h)
        self.set_mid_rows(0.0, -1.0 / h, 1.0 / h)
        self.set_last_row(-1.0 / h, 1.0 / h)


__all__ = ["DPlus"]

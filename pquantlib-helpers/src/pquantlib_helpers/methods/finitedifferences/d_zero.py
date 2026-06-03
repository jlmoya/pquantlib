r"""DZero — first-derivative central-difference operator :math:`D_0`.

# Retired-API compat layer — see package docstring.

Discretises the first derivative with the second-order central formula
:math:`\frac{\partial u_i}{\partial x} \approx \frac{u_{i+1}-u_{i-1}}{2h} = D_0 u_i`.

Java parity: ``org.jquantlib.methods.finitedifferences.DZero``.
C++ parity: ``ql/methods/finitedifferences/dzero.hpp`` (old QuantLib).
"""

from __future__ import annotations

from pquantlib_helpers.methods.finitedifferences.tridiagonal_operator import (
    TridiagonalOperator,
)


class DZero(TridiagonalOperator):
    r""":math:`D_0` matricial representation (central first difference)."""

    def __init__(self, grid_points: int, h: float) -> None:
        """Build the central-difference operator on ``grid_points`` nodes, step ``h``."""
        super().__init__(grid_points)
        # linear extrapolation at the boundaries.
        self.set_first_row(-1.0 / h, 1.0 / h)
        self.set_mid_rows(-1.0 / (2.0 * h), 0.0, 1.0 / (2.0 * h))
        self.set_last_row(-1.0 / h, 1.0 / h)


__all__ = ["DZero"]

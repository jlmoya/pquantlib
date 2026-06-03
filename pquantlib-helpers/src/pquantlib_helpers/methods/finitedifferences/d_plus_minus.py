r"""DPlusMinus — second-derivative operator :math:`D_{+}D_{-}`.

# Retired-API compat layer — see package docstring.

Discretises the second derivative with the second-order central formula
:math:`\frac{\partial^2 u_i}{\partial x^2} \approx
\frac{u_{i+1}-2u_i+u_{i-1}}{h^2} = D_{+}D_{-} u_i`.

Java parity: ``org.jquantlib.methods.finitedifferences.DPlusMinus``.
C++ parity: ``ql/methods/finitedifferences/dplusdminus.hpp`` (old QuantLib).
"""

from __future__ import annotations

from pquantlib_helpers.methods.finitedifferences.tridiagonal_operator import (
    TridiagonalOperator,
)


class DPlusMinus(TridiagonalOperator):
    r""":math:`D_{+}D_{-}` matricial representation (central second difference)."""

    def __init__(self, grid_points: int, h: float) -> None:
        """Build the second-difference operator on ``grid_points`` nodes, step ``h``."""
        super().__init__(grid_points)
        h2 = h * h
        # boundaries left at zero (linear extrapolation).
        self.set_first_row(0.0, 0.0)
        self.set_mid_rows(1.0 / h2, -2.0 / h2, 1.0 / h2)
        self.set_last_row(0.0, 0.0)


__all__ = ["DPlusMinus"]

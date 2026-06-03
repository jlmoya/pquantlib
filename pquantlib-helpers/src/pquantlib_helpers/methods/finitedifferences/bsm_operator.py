"""BSMOperator — Black-Scholes-Merton differential operator (tridiagonal).

# Retired-API compat layer — see package docstring.

Java parity: ``org.jquantlib.methods.finitedifferences.BSMOperator``.
C++ parity: ``ql/methods/finitedifferences/bsmoperator.hpp`` — in v1.42.1 this
header is an **empty deprecated stub** ("this file is empty and will disappear
in a future release"), so the BSMOperator class no longer ships in modern
QuantLib. We reproduce its documented old-QuantLib / JQuantLib coefficient
formula, which builds a :class:`TridiagonalOperator` directly:

    sigma2 = sigma * sigma
    nu     = r - q - sigma2 / 2
    pd     = -(sigma2 / dx - nu) / (2 * dx)        # lower (sub-diagonal)
    pu     = -(sigma2 / dx + nu) / (2 * dx)        # upper (super-diagonal)
    pm     =  sigma2 / (dx * dx) + r               # diagonal
    setMidRows(pd, pm, pu)                          # first/last rows stay zero

The interior coefficients are cross-validated TIGHT against the C++ probe
(``cluster/ws3fd1.json`` ``bsm_operator`` block), which rebuilds exactly this
stencil from genuine C++ ``TridiagonalOperator`` arithmetic.

# DEFERRED: the second Java constructor ``BSMOperator(Array grid,
# GeneralizedBlackScholesProcess process, double residualTime)`` builds the
# operator on a non-uniform log-grid via ``PdeConstantCoeff<PdeBSM>`` /
# ``LogGrid``. Those Pde* classes are explicitly out of scope for this cluster
# (FD-alpha2). The dividend FD engine path uses the uniform-grid
# ``(size, dx, r, q, sigma)`` constructor, which is what we port here.
"""

from __future__ import annotations

from pquantlib_helpers.methods.finitedifferences.tridiagonal_operator import (
    TridiagonalOperator,
)


class BSMOperator(TridiagonalOperator):
    """Constant-coefficient Black-Scholes-Merton tridiagonal operator."""

    def __init__(self, size: int, dx: float, r: float, q: float, sigma: float) -> None:
        """Build the BSM operator on ``size`` uniform log-grid nodes (step ``dx``).

        ``r`` risk-free rate, ``q`` dividend yield, ``sigma`` volatility.
        Only the interior rows are populated (``set_mid_rows``); the first and
        last rows are left at zero, mirroring the old-QuantLib / Java source.
        """
        super().__init__(size)
        sigma2 = sigma * sigma
        nu = r - q - sigma2 / 2.0
        pd = -(sigma2 / dx - nu) / (2.0 * dx)
        pu = -(sigma2 / dx + nu) / (2.0 * dx)
        pm = sigma2 / (dx * dx) + r
        self.set_mid_rows(pd, pm, pu)


__all__ = ["BSMOperator"]

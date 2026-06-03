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

The second Java constructor ``BSMOperator(Array grid,
GeneralizedBlackScholesProcess process, double residualTime)`` — the log-grid
path FD-alpha1 deferred — is now provided as the :meth:`BSMOperator.from_grid`
classmethod (FD-alpha2). It builds the operator on a non-uniform log-grid via a
:class:`~pquantlib_helpers.methods.finitedifferences.pde.PdeConstantCoeff` over
:class:`~pquantlib_helpers.methods.finitedifferences.pde.PdeBSM` and the
non-uniform-grid stencil in
:meth:`~pquantlib_helpers.methods.finitedifferences.pde.PdeSecondOrderParabolic.generate_operator`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib_helpers.methods.finitedifferences.pde import PdeBSM, PdeConstantCoeff
from pquantlib_helpers.methods.finitedifferences.transformed_grid import LogGrid
from pquantlib_helpers.methods.finitedifferences.tridiagonal_operator import (
    TridiagonalOperator,
)

if TYPE_CHECKING:
    from pquantlib.math.array import Array
    from pquantlib.processes.generalized_black_scholes_process import (
        GeneralizedBlackScholesProcess,
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

    @classmethod
    def from_grid(
        cls,
        grid: Array,
        process: GeneralizedBlackScholesProcess,
        residual_time: float,
    ) -> BSMOperator:
        """Build the constant-coeff BSM operator on a non-uniform log-grid.

        C++/Java parity: ``BSMOperator(Array grid, process, residualTime)``.
        The BSM coefficients are frozen once at ``(residualTime, S0)`` (where
        ``S0`` is the process state-variable value) via :class:`PdeConstantCoeff`,
        then the non-uniform-grid stencil is generated over the ``log(grid)``.

        Implemented as a classmethod (not a second ``__init__`` overload) so the
        uniform-grid ``__init__`` signature stays type-clean under pyright strict.
        """
        op = cls.__new__(cls)
        TridiagonalOperator.__init__(op, int(grid.shape[0]))
        log_grid = LogGrid(grid)
        x = float(process.state_variable().value())
        cc = PdeConstantCoeff(PdeBSM(process), residual_time, x)
        cc.generate_operator(residual_time, log_grid, op)
        return op


__all__ = ["BSMOperator"]

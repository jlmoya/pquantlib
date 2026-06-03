"""Time-dependent PDE operators + the OperatorFactory for the legacy FD layer.

# Retired-API compat layer — see package docstring.

C++ parity: ``ql/methods/finitedifferences/pdeoperator.hpp`` (``GenericTimeSetter``,
``PdeOperator``) and ``ql/methods/finitedifferences/operatorfactory.hpp``.
Java parity: ``org.jquantlib.methods.finitedifferences`` —
``GenericTimeSetter``, ``PdeOperator``, ``BSMTermOperator``, ``OperatorFactory``.

A :class:`GenericTimeSetter` rebuilds a tridiagonal operator's coefficients at a
given time by re-running ``Pde.generate_operator`` on the (cached) transformed
grid. :class:`PdeOperator` is a :class:`TridiagonalOperator` that installs such a
time-setter and seeds itself at the residual time. :class:`BSMTermOperator` is
the BSM specialisation. :func:`OperatorFactory.get_operator` selects between the
time-dependent (term) operator and the constant-coefficient
:class:`BSMOperator`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib_helpers.methods.finitedifferences.bsm_operator import BSMOperator
from pquantlib_helpers.methods.finitedifferences.pde import PdeBSM
from pquantlib_helpers.methods.finitedifferences.tridiagonal_operator import (
    TridiagonalOperator,
)

if TYPE_CHECKING:
    from pquantlib.math.array import Array
    from pquantlib.processes.generalized_black_scholes_process import (
        GeneralizedBlackScholesProcess,
    )
    from pquantlib_helpers.methods.finitedifferences.pde import (
        PdeSecondOrderParabolic,
    )


class GenericTimeSetter:
    """Time-setter that regenerates an operator from a PDE on each ``set_time``.

    C++ parity: ``GenericTimeSetter<PdeClass>``. Java parity:
    ``GenericTimeSetter``. Conforms to the FD-alpha1
    :class:`~pquantlib_helpers.methods.finitedifferences.tridiagonal_operator.TimeSetter`
    Protocol (``set_time(t, op)``).
    """

    def __init__(self, grid: Array, pde: PdeSecondOrderParabolic) -> None:
        """Cache the transformed grid for ``pde`` and bind the PDE."""
        self._grid = pde.apply_grid_type(grid)
        self._pde = pde

    def set_time(self, t: float, op: TridiagonalOperator) -> None:
        """Rebuild ``op`` from the PDE at time ``t``."""
        if self._grid is None:
            raise ValueError("PDE did not provide a transformed grid")
        self._pde.generate_operator(t, self._grid, op)


class PdeOperator(TridiagonalOperator):
    """Tridiagonal operator driven by a PDE via an installed time-setter.

    C++ parity: ``PdeOperator<PdeClass>``. Java parity:
    ``PdeOperator<T extends PdeSecondOrderParabolic>``.
    """

    def __init__(
        self,
        pde: PdeSecondOrderParabolic,
        grid: Array,
        residual_time: float,
    ) -> None:
        """Build a ``len(grid)``-sized operator seeded at ``residual_time``.

        # C++ parity divergence: C++/Java take a PDE *class* plus a process and
        # reflectively instantiate the PDE. We accept a pre-built PDE instance
        # (Python needs no ``Class<? extends Pde>`` reflection); the only caller
        # (:class:`BSMTermOperator`) constructs :class:`PdeBSM` directly.
        """
        super().__init__(int(grid.shape[0]))
        self.time_setter = GenericTimeSetter(grid, pde)
        self.set_time(residual_time)


class BSMTermOperator(PdeOperator):
    """Time-dependent Black-Scholes-Merton operator on a log-grid.

    C++ parity: ``BSMTermOperator``. Java parity: ``BSMTermOperator``.
    """

    def __init__(
        self,
        grid: Array,
        process: GeneralizedBlackScholesProcess,
        residual_time: float,
    ) -> None:
        """Build a BSM term operator from ``process`` on ``grid`` at ``residual_time``."""
        super().__init__(PdeBSM(process), grid, residual_time)


class OperatorFactory:
    """Factory selecting the constant-coefficient or time-dependent BSM operator.

    C++ parity: ``OperatorFactory``. Java parity: ``OperatorFactory``.
    """

    @staticmethod
    def get_operator(
        process: GeneralizedBlackScholesProcess,
        grid: Array,
        residual_time: float,
        time_dependent: bool,
    ) -> TridiagonalOperator:
        """Return a BSM term operator (if ``time_dependent``) or constant BSMOperator."""
        if time_dependent:
            return BSMTermOperator(grid, process, residual_time)
        return BSMOperator.from_grid(grid, process, residual_time)


__all__ = [
    "BSMTermOperator",
    "GenericTimeSetter",
    "OperatorFactory",
    "PdeOperator",
]

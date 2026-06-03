"""PDE coefficient classes + grid-operator generation for the legacy FD layer.

# Retired-API compat layer — see package docstring.

C++ parity: ``ql/methods/finitedifferences/pde.hpp`` (the ``PdeSecondOrderParabolic``
CRTP base + ``PdeConstantCoeff``) and ``ql/methods/finitedifferences/pdebsm.hpp``
(``PdeBSM``). Java parity: ``org.jquantlib.methods.finitedifferences`` —
``Pde`` (interface), ``PdeSecondOrderParabolic`` (abstract), ``PdeConstantCoeff``,
``PdeBSM``.

A ``Pde`` exposes ``diffusion(t, x)``, ``drift(t, x)``, ``discount(t, x)``.
``PdeSecondOrderParabolic.generate_operator`` discretises the second-order
parabolic operator on a (possibly log-transformed) :class:`TransformedGrid`
into a :class:`TridiagonalOperator`, using the non-uniform-grid stencil:

    pd =  -(sigma^2 / dxm - nu) / dx
    pu =  -(sigma^2 / dxp + nu) / dx
    pm =   sigma^2 / (dxm * dxp) + r

This is the machinery FD-alpha1 deferred for the log-grid
``BSMOperator(grid, process, residualTime)`` path.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency
from pquantlib_helpers.methods.finitedifferences.transformed_grid import (
    LogGrid,
    TransformedGrid,
)

if TYPE_CHECKING:
    from pquantlib.math.array import Array
    from pquantlib.processes.generalized_black_scholes_process import (
        GeneralizedBlackScholesProcess,
    )
    from pquantlib_helpers.methods.finitedifferences.tridiagonal_operator import (
        TridiagonalOperator,
    )


@runtime_checkable
class Pde(Protocol):
    """Coefficient surface a PDE must expose to the FD layer.

    C++ parity: the ``Pde`` concept (``diffusion``/``drift``/``discount``).
    Java parity: ``interface Pde``.
    """

    def diffusion(self, t: float, x: float) -> float:
        """Diffusion coefficient ``sigma`` at ``(t, x)``."""
        ...

    def drift(self, t: float, x: float) -> float:
        """Drift coefficient ``nu`` at ``(t, x)``."""
        ...

    def discount(self, t: float, x: float) -> float:
        """Discount/reaction coefficient ``r`` at ``(t, x)``."""
        ...


class PdeSecondOrderParabolic(ABC):
    """Abstract second-order parabolic PDE with grid-operator generation.

    C++ parity: ``PdeSecondOrderParabolic``. Java parity:
    ``abstract class PdeSecondOrderParabolic implements Pde``.
    """

    @abstractmethod
    def diffusion(self, t: float, x: float) -> float:
        """Diffusion coefficient ``sigma`` at ``(t, x)``."""

    @abstractmethod
    def drift(self, t: float, x: float) -> float:
        """Drift coefficient ``nu`` at ``(t, x)``."""

    @abstractmethod
    def discount(self, t: float, x: float) -> float:
        """Discount/reaction coefficient ``r`` at ``(t, x)``."""

    def apply_grid_type(self, a: Array) -> TransformedGrid | None:
        """Wrap ``a`` in the transformed grid this PDE lives on (``None`` by default).

        C++ parity: ``applyGridType``. Overridden by :class:`PdeBSM` to return a
        :class:`LogGrid`.
        """
        return None

    def generate_operator(
        self, t: float, tg: TransformedGrid, op: TridiagonalOperator
    ) -> None:
        """Discretise the operator on transformed grid ``tg`` into ``op`` at time ``t``.

        C++ parity: ``PdeSecondOrderParabolic::generateOperator``. Only interior
        rows are written; the boundary rows are left as-is for the boundary
        conditions to set.
        """
        for i in range(1, tg.size() - 1):
            x = tg.grid(i)
            sigma = self.diffusion(t, x)
            nu = self.drift(t, x)
            r = self.discount(t, x)
            sigma2 = sigma * sigma

            pd = -(sigma2 / tg.dxm(i) - nu) / tg.dx(i)
            pu = -(sigma2 / tg.dxp(i) + nu) / tg.dx(i)
            pm = sigma2 / (tg.dxm(i) * tg.dxp(i)) + r
            op.set_mid_row(i, pd, pm, pu)


class PdeConstantCoeff(PdeSecondOrderParabolic):
    """Constant-coefficient freeze of another PDE evaluated once at ``(t, x)``.

    C++ parity: ``PdeConstantCoeff<PdeClass>``. Java parity:
    ``PdeConstantCoeff<T extends Pde>``. The wrapped PDE's coefficients are
    sampled once at the construction point and returned constant thereafter,
    yielding a constant-coefficient tridiagonal operator.
    """

    def __init__(self, pde: PdeSecondOrderParabolic, t: float, x: float) -> None:
        """Freeze ``pde``'s coefficients at ``(t, x)``.

        # C++ parity divergence: the C++/Java ``PdeConstantCoeff`` takes a PDE
        # *class* plus a process and reflectively instantiates it. We pass a
        # pre-built :class:`PdeSecondOrderParabolic` instance instead — Python
        # has no need for the reflective ``Class<? extends Pde>`` indirection,
        # and the call sites (:class:`BSMOperator`) construct the PDE directly.
        """
        self._diffusion = pde.diffusion(t, x)
        self._drift = pde.drift(t, x)
        self._discount = pde.discount(t, x)

    def diffusion(self, t: float, x: float) -> float:
        """Frozen diffusion coefficient."""
        return self._diffusion

    def drift(self, t: float, x: float) -> float:
        """Frozen drift coefficient."""
        return self._drift

    def discount(self, t: float, x: float) -> float:
        """Frozen discount coefficient."""
        return self._discount


class PdeBSM(PdeSecondOrderParabolic):
    """Black-Scholes-Merton PDE backed by a generalized BSM process.

    C++ parity: ``PdeBSM``. Java parity: ``PdeBSM``. Lives on a log-grid
    (:meth:`apply_grid_type` returns a :class:`LogGrid`).
    """

    def __init__(self, process: GeneralizedBlackScholesProcess) -> None:
        """Bind the BSM process supplying the coefficients."""
        self._process = process

    def diffusion(self, t: float, x: float) -> float:
        """Process diffusion ``sigma(t, x)``."""
        return self._process.diffusion_1d(t, x)

    def drift(self, t: float, x: float) -> float:
        """Process drift ``nu(t, x)``."""
        return self._process.drift_1d(t, x)

    def discount(self, t: float, x: float) -> float:
        """Continuously-compounded instantaneous forward (discount) rate at ``t``.

        C++ parity: ``PdeBSM::discount`` — snaps tiny ``t`` to zero then reads
        the risk-free instantaneous forward rate.
        """
        if abs(t) < 1e-8:
            t = 0.0
        return (
            self._process.risk_free_rate()
            .forward_rate(t, t, Compounding.Continuous, Frequency.NoFrequency, True)
            .rate()
        )

    def apply_grid_type(self, a: Array) -> TransformedGrid:
        """Return the log-grid this PDE lives on."""
        return LogGrid(a)


__all__ = [
    "Pde",
    "PdeBSM",
    "PdeConstantCoeff",
    "PdeSecondOrderParabolic",
]

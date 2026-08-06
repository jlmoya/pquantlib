"""Python mirror of the C++ probe fixture in v143_methods_schemes/probe.cpp.

# C++ parity: the ``ProbeOp`` / ``ProbeBC`` classes in
# migration-harness/cpp/probes/v143_methods_schemes/probe.cpp @ v1.43.

The schemes under test are pure algebra over an operator: every number they
produce is a linear combination of ``apply`` / ``apply_direction`` /
``apply_mixed`` / ``solve_splitting`` results. Testing them against a
production operator alone would conflate a scheme bug with an operator bug, and
— worse — a 1-D operator makes ``CraigSneydScheme``, ``HundsdorferScheme`` and
``ModifiedCraigSneydScheme`` algebraically identical, so it cannot tell them
apart at all.

So the primary fixture is this deliberately synthetic two-direction operator
with a real mixed term, transcribed loop-for-loop from the probe. Every
coefficient is a dyadic rational and every accumulation is written in the same
order as the C++, so any C++/Python difference has to come from the scheme.
"""

from __future__ import annotations

from typing import Final, cast

import numpy as np

from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.fdm_boundary_condition import FdmBoundaryCondition

#: # C++ parity: ``const Size PROBE_N = 7``.
PROBE_N: Final[int] = 7

#: # C++ parity: ``SYN_T`` / ``SYN_DT``.
SYN_T: Final[float] = 1.0
SYN_DT: Final[float] = 0.25


class ProbeOp:
    """Two-direction ``FdmLinearOpComposite`` with a mixed term.

    # C++ parity: ``class ProbeOp : public FdmLinearOpComposite``.

    Diffusion-shaped (negative diagonal, positive off-diagonals, weakly
    diagonally dominant) so ``a + dt*L*a`` decays exactly as a production FD
    generator does.
    """

    def __init__(self) -> None:
        self._mask: list[float] = [1.0] * PROBE_N
        self._tm: float = 0.0

    def size(self) -> int:
        return 2

    def set_time(self, t1: float, t2: float) -> None:
        self._tm = 0.5 * (t1 + t2)

    def mask_boundary(self) -> None:
        """Zero the first and last operator row (what a Dirichlet BC does)."""
        self._mask[0] = 0.0
        self._mask[PROBE_N - 1] = 0.0

    def _lower(self, d: int, i: int) -> float:
        return (1.0 - 0.0625 * i) if d == 0 else (0.375 + 0.015625 * i)

    def _diag(self, d: int, i: int) -> float:
        return (-2.5 - 0.25 * self._tm - 0.0625 * i) if d == 0 else (-1.25 - 0.5 * self._tm)

    def _upper(self, d: int, i: int) -> float:
        return (0.875 + 0.03125 * i) if d == 0 else (0.4375 - 0.03125 * i)

    def apply_direction(self, direction: int, r: Array) -> Array:
        out: Array = np.zeros(PROBE_N, dtype=np.float64)
        for i in range(PROBE_N):
            v = self._diag(direction, i) * float(r[i])
            if i > 0:
                v += self._lower(direction, i) * float(r[i - 1])
            if i + 1 < PROBE_N:
                v += self._upper(direction, i) * float(r[i + 1])
            out[i] = self._mask[i] * v
        return out

    def apply_mixed(self, r: Array) -> Array:
        out: Array = np.zeros(PROBE_N, dtype=np.float64)
        for i in range(PROBE_N):
            out[i] = (
                self._mask[i]
                * (0.0625 + 0.125 * self._tm)
                * (float(r[(i + 2) % PROBE_N]) - float(r[(i + PROBE_N - 2) % PROBE_N]))
            )
        return out

    def apply(self, r: Array) -> Array:
        return self.apply_direction(0, r) + self.apply_direction(1, r) + self.apply_mixed(r)

    def solve_splitting(self, direction: int, r: Array, dt: float) -> Array:
        """Solve ``(I + dt*L_direction) x = r`` by the Thomas algorithm.

        # C++ parity: ``ProbeOp::solve_splitting``, itself written as
        # ``TridiagonalOperator::solveFor``.
        """
        x: Array = np.zeros(PROBE_N, dtype=np.float64)
        temp = [0.0] * PROBE_N
        a = [0.0] * PROBE_N
        b = [0.0] * PROBE_N
        c = [0.0] * PROBE_N
        for i in range(PROBE_N):
            a[i] = dt * self._mask[i] * self._lower(direction, i)
            b[i] = 1.0 + dt * self._mask[i] * self._diag(direction, i)
            c[i] = dt * self._mask[i] * self._upper(direction, i)
        bet = b[0]
        x[0] = float(r[0]) / bet
        for j in range(1, PROBE_N):
            temp[j] = c[j - 1] / bet
            bet = b[j] - a[j] * temp[j]
            x[j] = (float(r[j]) - a[j] * float(x[j - 1])) / bet
        for j in range(PROBE_N - 2, 0, -1):
            x[j] -= temp[j + 1] * x[j + 1]
        x[0] -= temp[1] * x[1]
        return x

    def preconditioner(self, r: Array, dt: float) -> Array:
        return self.solve_splitting(0, r, dt)


class ProbeBC(FdmBoundaryCondition):
    """Dirichlet-shaped boundary condition over :class:`ProbeOp`.

    # C++ parity: ``class ProbeBC : public BoundaryCondition<FdmLinearOp>``.

    The operator hooks are idempotent (they zero the two boundary rows), the
    array hooks pin the two boundary entries; ``set_time`` moves the pinned
    value so a missing ``bcSet_.setTime`` call in a scheme is observable.
    """

    def __init__(self) -> None:
        self._v: float = 1.0

    def apply_before_applying(self, op: object) -> None:
        cast(ProbeOp, op).mask_boundary()

    def apply_after_applying(self, u: Array) -> None:
        u[0] = self._v
        u[-1] = -self._v

    def apply_before_solving(self, op: object, rhs: Array) -> None:
        cast(ProbeOp, op).mask_boundary()
        rhs[0] = self._v
        rhs[-1] = -self._v

    def apply_after_solving(self, u: Array) -> None:
        u[0] = 2.0 * self._v
        u[-1] = -2.0 * self._v

    def set_time(self, t: float) -> None:
        self._v = 1.0 + 0.5 * t


def probe_start() -> Array:
    """# C++ parity: ``Array probeStart()``."""
    return np.array([1.0, 1.5, 2.25, 1.75, 0.5, -0.75, 2.0], dtype=np.float64)


def make_bc_set(with_bc: bool) -> tuple[ProbeBC, ...]:
    """# C++ parity: ``makeBcSet`` — a fresh ProbeBC per call, as in C++."""
    return (ProbeBC(),) if with_bc else ()

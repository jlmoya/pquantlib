"""FdmExtOUJumpOp — FD operator for the Kluge ExtOU + jump SDE.

# C++ parity: ql/experimental/finitedifferences/fdmextoujumpop.{hpp,cpp}
# (v1.42.1).

For the two-factor Kluge ExtOU + jump process

    dX_t = a(b(t) - X_t) dt + sigma dW_t
    dY_t = -beta Y_{t-} dt + J_t dN_t,    omega(J) = eta exp(-eta J)

the PDE features an integro-differential cross-coupling: jumps in
``Y`` translate continuous PDF over ``Y'`` via the partial integral

    integro(f)(y) = lambda * (integral f(y' + y) omega(y') dy' - f(y)).

The C++ implementation discretises this with Gauss-Laguerre
quadrature of order ``integroIntegrationOrder`` (rule for x in [0, inf)
weighted by exp(-x); we factor the weight back out and use ys = y +
yInt/eta as the shifted query points). It interpolates linearly
into the Y-mesh using ``upper_bound``.

The Python port reproduces this exactly with
``scipy.special.roots_laguerre`` for the nodes/weights and the same
linear-interpolation scheme, building a sparse ``integro_part`` matrix.
"""

from __future__ import annotations

from typing import final

import numpy as np
import numpy.typing as npt
from scipy.sparse import (  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
    csr_matrix,
    lil_matrix,
)
from scipy.special import (  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
    roots_laguerre,
)

from pquantlib.experimental.finitedifferences.fdm_extended_ornstein_uhlenbeck_op import (
    FdmExtendedOrnsteinUhlenbeckOp,
)
from pquantlib.experimental.processes.ext_ou_with_jumps_process import (
    ExtOUWithJumpsProcess,
)
from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.operators.first_derivative_op import (
    FirstDerivativeOp,
)
from pquantlib.methods.finitedifferences.operators.triple_band_linear_op import (
    TripleBandLinearOp,
)
from pquantlib.termstructures.yield_term_structure import YieldTermStructure

_IntArray = npt.NDArray[np.int64]


def _gauss_laguerre_nodes_weights(order: int) -> tuple[Array, Array]:
    """Return Gauss-Laguerre nodes and weights of the given order.

    # C++ parity: ``GaussLaguerreIntegration(order)`` returns nodes
    # ``x_i`` and weights ``w_i`` for the integral over [0, inf)
    # with weight exp(-x).
    """
    nodes, weights = roots_laguerre(order)  # pyright: ignore[reportUnknownVariableType]
    return np.asarray(nodes, dtype=np.float64), np.asarray(weights, dtype=np.float64)


@final
class FdmExtOUJumpOp:
    """FD operator for the Kluge ExtOU + jump SDE (2-factor).

    # C++ parity: ``class FdmExtOUJumpOp : public FdmLinearOpComposite``.

    The operator is composed of three parts:

    * ``ouOp`` — pure ExtendedOU op along direction 0.
    * ``dyMap`` — first-derivative along direction 1 times ``-beta * y``
      (jump factor mean-reversion).
    * ``integroPart`` — sparse integro term for jumps (off-direction).
    """

    def __init__(
        self,
        mesher: FdmMesher,
        process: ExtOUWithJumpsProcess,
        r_ts: YieldTermStructure,
        integro_integration_order: int = 32,
    ) -> None:
        self._mesher: FdmMesher = mesher
        self._process: ExtOUWithJumpsProcess = process
        self._r_ts: YieldTermStructure = r_ts
        self._integro_order: int = integro_integration_order

        # X (direction 0) op uses the embedded ExtendedOU process.
        self._ou_op: FdmExtendedOrnsteinUhlenbeckOp = FdmExtendedOrnsteinUhlenbeckOp(
            mesher,
            process.get_extended_ornstein_uhlenbeck_process(),
            r_ts,
            direction=0,
        )

        # Y-axis mean reversion drift: D_y times -beta * y.
        layout = mesher.layout()
        size = layout.size()
        # Build the per-node y-array for the drift coefficient.
        y_drift = np.empty(size, dtype=np.float64)
        for iter_ in layout.iter():
            y_drift[iter_.index] = -process.beta() * mesher.location(iter_, 1)
        self._dy_map: TripleBandLinearOp = FirstDerivativeOp(1, mesher).mult(y_drift)

        # Pre-build the sparse integro part.
        self._integro_part: csr_matrix = self._build_integro_part()

    def _build_integro_part(self) -> csr_matrix:
        """Build the sparse integro matrix.

        # C++ parity: constructor body in fdmextoujumpop.cpp:74-101.

        For each grid node `i` with y-coordinate `y`:
        * diag entry receives ``-lambda``.
        * for each Gauss-Laguerre node ``yInt[k]`` with weight ``w[k]``,
          we form ``ys = y + yInt[k] / eta`` and find its location
          ``l`` such that ``yLoc[l] < ys <= yLoc[l+1]`` (clamp at end).
          We split the weighted contribution
          ``exp(-yInt[k]) * w[k] * lambda``
          between the linear-interpolation neighbours.
        """
        eta = self._process.eta()
        lam = self._process.jump_intensity()

        nodes, weights = _gauss_laguerre_nodes_weights(self._integro_order)

        layout = self._mesher.layout()
        n = layout.size()
        dim_y = layout.dim()[1]

        # 1-D locations along the Y axis (for the interpolation
        # search; only need the dim_y unique values).
        y_loc = np.empty(dim_y, dtype=np.float64)
        for iter_ in layout.iter():
            y_loc[iter_.coordinates[1]] = self._mesher.location(iter_, 1)

        integro = lil_matrix((n, n))

        for iter_ in layout.iter():
            diag = iter_.index
            integro[diag, diag] -= lam

            y = self._mesher.location(iter_, 1)
            y_index = iter_.coordinates[1]

            for k in range(nodes.size):
                # scipy.special.roots_laguerre returns "weighted" weights,
                # i.e., sum_i w_i f(x_i) ~= int f(x) exp(-x) dx. The C++
                # QuantLib GaussLaguerre returns "unweighted" weights and
                # multiplies by exp(-x_i) in fdmextoujumpop.cpp line 88
                # to recover the exp(-x) factor. With scipy, the weight
                # IS already exp-weighted, so we use it directly.
                weight = float(weights[k])

                ys = y + float(nodes[k]) / eta
                if ys > float(y_loc[-1]):
                    el = dim_y - 2
                else:
                    # Match C++ upper_bound logic:
                    # `l = upper_bound(yLoc.begin(), yLoc.end()-1, ys) - yLoc.begin() - 1`
                    # std::upper_bound returns the first iterator > ys
                    # over a search range yLoc[0..dim_y-1) -- so dim_y-1 inclusive (end-1).
                    el = int(np.searchsorted(y_loc[:-1], ys, side="right")) - 1
                    el = max(el, 0)
                    el = min(el, dim_y - 2)

                s = (ys - float(y_loc[el])) / (float(y_loc[el + 1]) - float(y_loc[el]))

                lower_neigh = layout.neighbourhood(iter_, 1, el - y_index)
                upper_neigh = layout.neighbourhood(iter_, 1, el + 1 - y_index)
                integro[diag, lower_neigh] += weight * lam * (1.0 - s)
                integro[diag, upper_neigh] += weight * lam * s

        return csr_matrix(integro.tocsr())

    # --- composite interface --------------------------------------------

    def size(self) -> int:
        """Number of directions in the operator.

        # C++ parity: ``size`` returns count of mesh axes.
        """
        return len(self._mesher.layout().dim())

    def set_time(self, t1: float, t2: float) -> None:
        """Recompute time-dependent coefficients.

        # C++ parity: ``setTime`` only updates the inner ExtOU op
        # (the dyMap + integroPart are time-independent).
        """
        self._ou_op.set_time(t1, t2)

    def apply(self, r: Array) -> Array:
        """Full apply: ouOp.apply + dyMap.apply + integro(r).

        # C++ parity: ``apply``.
        """
        return self._ou_op.apply(r) + self._dy_map.apply(r) + self._integro(r)

    def apply_mixed(self, r: Array) -> Array:
        """Cross-direction apply = integro term (off-diagonal).

        # C++ parity: ``apply_mixed`` returns ``integro(r)``.
        """
        return self._integro(r)

    def apply_direction(self, direction: int, r: Array) -> Array:
        """Apply only the ``direction``-axis part.

        # C++ parity: direction==0 -> ouOp; ==1 -> dyMap; else 0.
        """
        if direction == 0:
            return self._ou_op.apply_direction(0, r)
        if direction == 1:
            return self._dy_map.apply(r)
        return np.zeros(r.size, dtype=np.float64)

    def solve_splitting(self, direction: int, r: Array, a: float) -> Array:
        """Solve ``(I + a * L_dir) x = r`` along the given direction.

        # C++ parity: direction==0 -> ouOp.solve_splitting;
        # ==1 -> dyMap.solve_splitting; else returns r unchanged.
        """
        if direction == 0:
            return self._ou_op.solve_splitting(0, r, a)
        if direction == 1:
            return self._dy_map.solve_splitting(r, a, 1.0)
        return r

    def preconditioner(self, r: Array, dt: float) -> Array:
        """Preconditioner via direction-0 splitting solve.

        # C++ parity: ``preconditioner`` always uses direction 0.
        """
        return self._ou_op.solve_splitting(0, r, dt)

    def _integro(self, r: Array) -> Array:
        """Sparse matrix-vector with the integro part."""
        return np.asarray(self._integro_part @ r, dtype=np.float64)


__all__ = ["FdmExtOUJumpOp"]

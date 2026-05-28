"""ShortRateTree — recombining trinomial lattice for 1-factor short-rate models.

# C++ parity: nested ``OneFactorModel::ShortRateTree`` in
# ql/models/shortrate/onefactormodel.{hpp,cpp} (v1.42.1).

The C++ class is a nested member of ``OneFactorModel`` and uses the
``TreeLattice1D<ShortRateTree>`` CRTP base.  PQuantLib promotes it
to a stand-alone class because (a) Python doesn't have C++ nested-
class behaviours that benefit us here and (b) the two-stage init
(plain tree + numerical-fitting tree) makes the class fully owned by
``ShortRateTree`` rather than ``OneFactorModel``.

The lattice wraps a :class:`TrinomialTree` over the state variable
``x`` and a :class:`ShortRateDynamics` that maps ``x`` to the
short rate ``r``.  The per-node discount is ``exp(-r * dt(i))``.

Two construction modes (matching C++):

  * **Plain**: ``ShortRateTree(trinomial, dynamics, grid)`` — assumes
    the dynamics already produces the correct discount-bond fit at
    every state (Vasicek / CIR / HullWhite — closed-form fitting).
  * **Numerical fit**: ``ShortRateTree(trinomial, dynamics, phi, grid)``
    — calibrates a ``TermStructureFittingParameterImpl`` (``phi``) by
    Brent-solving for the theta(i) that makes the lattice reprice the
    market discount at each grid point. Used by BlackKarasinski.

# C++ parity: there's an optional ``spread`` field that adds a
# constant credit spread to the short rate. We carry it for parity.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pquantlib.math.array import Array
from pquantlib.math.solvers1d.brent import Brent
from pquantlib.methods.lattices.tree_lattice_1d import TreeLattice1D

if TYPE_CHECKING:
    from pquantlib.methods.lattices.trinomial_tree import TrinomialTree
    from pquantlib.models.parameter import TermStructureFittingParameterImpl
    from pquantlib.models.shortrate.onefactor.one_factor_model import (
        ShortRateDynamics,
    )
    from pquantlib.termstructures.yield_term_structure import YieldTermStructure
    from pquantlib.time.time_grid import TimeGrid


class ShortRateTree(TreeLattice1D):
    """Recombining trinomial lattice for a 1-factor short-rate model.

    # C++ parity: ``class OneFactorModel::ShortRateTree`` in
    # ql/models/shortrate/onefactormodel.hpp:74-117 (v1.42.1).
    """

    branches: int = 3

    def __init__(
        self,
        tree: TrinomialTree,
        dynamics: ShortRateDynamics,
        time_grid: TimeGrid,
        phi: TermStructureFittingParameterImpl | None = None,
        term_structure: YieldTermStructure | None = None,
    ) -> None:
        """Construct the lattice — either plain or with numerical fitting.

        # C++ parity: two ctors in onefactormodel.{hpp,cpp}:
        #   - plain: (tree, dynamics, timeGrid) — onefactormodel.cpp:79-83.
        #   - numerical-fit: (tree, dynamics, phi, timeGrid) —
        #     onefactormodel.cpp:56-77.

        Pass ``phi`` (and ``term_structure``) to enable numerical fitting.
        The fitter calibrates ``phi(t_i)`` by Brent-solving for the
        theta that makes the lattice reprice the curve's discount at
        each ``t_{i+1}``.
        """
        super().__init__(time_grid=time_grid, n_branches=3)
        self._tree: TrinomialTree = tree
        self._dynamics: ShortRateDynamics = dynamics
        self._spread: float = 0.0
        # # C++ parity: only numerically fit when both phi and curve
        # # are supplied (the plain ctor doesn't touch the cache).
        if phi is not None and term_structure is not None:
            self._numerically_fit(phi, term_structure)

    # --- inspectors -------------------------------------------------------

    def set_spread(self, spread: float) -> None:
        """Set an additive constant credit spread.

        # C++ parity: ``setSpread`` (onefactormodel.hpp:108-110).
        """
        self._spread = float(spread)

    @property
    def spread(self) -> float:
        return self._spread

    @property
    def dynamics(self) -> ShortRateDynamics:
        return self._dynamics

    # --- Tree contract delegated to the held TrinomialTree ---------------

    def size(self, i: int) -> int:
        # C++ parity: onefactormodel.hpp:85-87.
        return self._tree.size(i)

    def underlying(self, i: int, index: int) -> float:
        # C++ parity: onefactormodel.hpp:93-95.
        return self._tree.underlying(i, index)

    def descendant(self, i: int, index: int, branch: int) -> int:
        # C++ parity: onefactormodel.hpp:96-98.
        return self._tree.descendant(i, index, branch)

    def probability(self, i: int, index: int, branch: int) -> float:
        # C++ parity: onefactormodel.hpp:99-101.
        return self._tree.probability(i, index, branch)

    def discount(self, i: int, index: int) -> float:
        """One-step discount = exp(-(r + spread) * dt(i)).

        # C++ parity: ``discount`` (onefactormodel.hpp:88-92).
        """
        x = self._tree.underlying(i, index)
        r = self._dynamics.short_rate(self._time_grid[i], x) + self._spread
        return math.exp(-r * self._time_grid.dt(i))

    # --- numerical fitting machinery (BlackKarasinski-style) -------------

    def _numerically_fit(
        self,
        phi: TermStructureFittingParameterImpl,
        term_structure: YieldTermStructure,
    ) -> None:
        """Solve for theta(t_i) at each grid time via Brent.

        # C++ parity: onefactormodel.cpp:56-77 (the ``ShortRateTree``
        # numerical-impl ctor body).
        """
        phi.reset()
        value = 1.0
        v_min = -100.0
        v_max = 100.0
        n = self._time_grid.size() - 1
        for i in range(n):
            target_discount = term_structure.discount(self._time_grid[i + 1])
            # # C++ parity: onefactormodel.cpp:35-46 — the Helper class
            # # uses ``size_ = tree.size(i)`` and the current state
            # # prices at slice ``i``.
            size = self.size(i)
            state_prices = self.state_prices(i)
            phi.set(self._time_grid[i], 0.0)

            value = self._solve_slice(
                phi=phi,
                i=i,
                size=size,
                state_prices=state_prices,
                target=target_discount,
                seed=value,
                v_min=v_min,
                v_max=v_max,
            )
            phi.change(value)

    def _solve_slice(
        self,
        *,
        phi: TermStructureFittingParameterImpl,
        i: int,
        size: int,
        state_prices: Array,
        target: float,
        seed: float,
        v_min: float,
        v_max: float,
    ) -> float:
        """Run Brent on a single slice ``i`` — extracted to bind loop vars.

        # C++ parity: onefactormodel.cpp:67-73 — the Brent solve inside
        # the for-i loop of the numerical-fit ctor.
        """
        def finder(theta: float) -> float:
            phi.change(theta)
            acc = target
            for j in range(size):
                acc -= state_prices[j] * self.discount(i, j)
            return acc

        solver = Brent()
        solver.set_max_evaluations(1000)
        return solver.solve(finder, 1e-7, seed, v_min, v_max)


__all__ = ["ShortRateTree"]

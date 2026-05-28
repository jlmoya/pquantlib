"""TrinomialTree — recombining trinomial tree over a 1-D process.

# C++ parity: ql/methods/lattices/trinomialtree.{hpp,cpp} (v1.42.1).

The trinomial tree discretises a 1-D stochastic process on a given
``TimeGrid``. At each time slice ``i`` (>= 1) the lattice fans out
into the centred trinomial step at every existing node, and the
descendant indices recombine into a single contiguous index range.

Notation matches the C++ source:

- ``dx[i]`` — node spacing at slice ``i`` (a function of the process
  variance over ``dt[i-1]``).
- ``size(i)`` — number of nodes at slice ``i``.
- ``branchings_[i]`` — branching scheme that maps (index_at_i, branch)
  to (descendant_index_at_(i+1), probability). Stored as one
  :class:`_Branching` per time slice in the C++ source; we keep the
  same layout (a list ``branchings`` indexed by ``i`` in [0, n-1]).
- ``underlying(i, index) = x0 + (jMin(i) + index) * dx[i]``.

The branching is centred so that the **middle** branch lands at the
node whose underlying is closest to the conditional expectation
``E[x_{t+dt} | x_t = x]``. The branching probabilities ``(p1, p2, p3)``
match the second moment (variance) and the third moment is left to
recombine via the choice of centre.

C++ requires that the diffusion term of the SDE be independent of the
underlying — i.e. ``variance(t, 0.0, dt)`` evaluates to the slice
variance regardless of ``x``. We carry the same restriction (called
out in C++ trinomialtree.hpp:36-37 as the "warning" Doxygen tag).

The ``is_positive`` flag (off by default) clamps the lowest branch so
that ``x0 + (k-1)*dx[i+1] > 0`` — used by BlackKarasinski-style trees
where the state is the log of a positive quantity.

# C++ parity: nested ``TrinomialTree::Branching`` (trinomialtree.hpp:66-79)
# is ported as a private ``_Branching`` dataclass-like helper. We use
# lists + a single growing-bounds invariant; the C++ source uses
# ``std::vector<Integer>`` and ``std::vector<std::vector<Real>>``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.methods.lattices.tree import Tree

if TYPE_CHECKING:
    from pquantlib.processes.stochastic_process_1d import StochasticProcess1D
    from pquantlib.time.time_grid import TimeGrid

# Sentinels matching ``QL_MAX_INTEGER`` / ``QL_MIN_INTEGER`` (C++ uses
# ``std::numeric_limits<Integer>::max()`` / ``min()`` to initialise
# the ``kMin_``/``jMin_`` invariants — Python's ``sys.maxsize`` is
# the closest analogue).
_INTEGER_MAX = 2**31 - 1
_INTEGER_MIN = -(2**31)


class _Branching:
    """Branching scheme for a single time slice of a trinomial tree.

    # C++ parity: nested ``TrinomialTree::Branching`` in
    # trinomialtree.hpp:66-79 + trinomialtree.hpp:105-143 (inline impls).

    Each call to :meth:`add` registers the central-branch index ``k``
    for one parent node together with the three branching
    probabilities ``(p1, p2, p3)``. ``add`` maintains the global
    ``kMin/kMax`` over all parents, from which we derive
    ``jMin/jMax = kMin - 1 / kMax + 1`` (the bounds of the *child*
    slice indices).
    """

    __slots__ = ("_j_max", "_j_min", "_k", "_k_max", "_k_min", "_probs")

    def __init__(self) -> None:
        # C++ parity: trinomialtree.hpp:105-107 — initial invariants.
        # ``probs_`` is ``std::vector<std::vector<Real>>(3)`` —
        # three parallel arrays for p1, p2, p3.
        self._k: list[int] = []
        self._probs: list[list[float]] = [[], [], []]
        self._k_min: int = _INTEGER_MAX
        self._j_min: int = _INTEGER_MAX
        self._k_max: int = _INTEGER_MIN
        self._j_max: int = _INTEGER_MIN

    def add(self, k: int, p1: float, p2: float, p3: float) -> None:
        """Record a parent's central-branch index + three probabilities.

        # C++ parity: ``TrinomialTree::Branching::add`` (trinomialtree.hpp:131-143).
        """
        self._k.append(k)
        self._probs[0].append(p1)
        self._probs[1].append(p2)
        self._probs[2].append(p3)
        # Invariants: kMin/kMax across all parents; jMin/jMax for
        # the child slice (bounded by ``[kMin - 1, kMax + 1]``).
        if k < self._k_min:
            self._k_min = k
            self._j_min = k - 1
        if k > self._k_max:
            self._k_max = k
            self._j_max = k + 1

    def descendant(self, index: int, branch: int) -> int:
        """Child-slice index for parent ``index`` taking ``branch`` (0/1/2).

        # C++ parity: ``Branching::descendant`` (trinomialtree.hpp:109-112).
        """
        return self._k[index] - self._j_min - 1 + branch

    def probability(self, index: int, branch: int) -> float:
        """Branch probability at parent ``index`` and branch ``branch``.

        # C++ parity: ``Branching::probability`` (trinomialtree.hpp:114-117).
        """
        return self._probs[branch][index]

    def size(self) -> int:
        """Number of nodes at the *child* slice (one more on each end).

        # C++ parity: ``Branching::size`` (trinomialtree.hpp:119-121) —
        # ``jMax - jMin + 1``.
        """
        return self._j_max - self._j_min + 1

    @property
    def j_min(self) -> int:
        """Minimum child-slice index (``kMin - 1``).

        # C++ parity: ``Branching::jMin`` (trinomialtree.hpp:123-125).
        """
        return self._j_min

    @property
    def j_max(self) -> int:
        """Maximum child-slice index (``kMax + 1``).

        # C++ parity: ``Branching::jMax`` (trinomialtree.hpp:127-129).
        """
        return self._j_max


class TrinomialTree(Tree[float]):
    """Recombining trinomial tree over a 1-D process.

    # C++ parity: ``TrinomialTree`` (trinomialtree.hpp:41-79 +
    # trinomialtree.cpp:26-75).

    Construction populates ``branchings_`` and the per-slice ``dx_`` —
    everything else (``size`` / ``underlying`` / ``descendant`` /
    ``probability``) is read from those tables.
    """

    branches: int = 3

    def __init__(
        self,
        process: StochasticProcess1D,
        time_grid: TimeGrid,
        is_positive: bool = False,
    ) -> None:
        # # C++ parity: trinomialtree.cpp:26-75.
        # ``Tree::__init__`` records columns = time_grid.size().
        super().__init__(columns=time_grid.size())
        self._x0: float = process.x0()
        self._time_grid: TimeGrid = time_grid
        # dx[0] = 0.0 (root has no spacing yet); subsequent ``dx[i]``
        # are pushed by the loop below.
        self._dx: list[float] = [0.0]
        self._branchings: list[_Branching] = []

        n_time_steps = time_grid.size() - 1
        qassert.require(n_time_steps > 0, "null time steps for trinomial tree")

        j_min: int = 0
        j_max: int = 0

        for i in range(n_time_steps):
            t = time_grid[i]
            dt = time_grid.dt(i)

            # The diffusion must be independent of x (C++ "warning"
            # tag — variance(t, 0.0, dt) is the slice variance).
            v2 = process.variance_1d(t, 0.0, dt)
            v = math.sqrt(v2)
            # Trinomial spacing: dx = v * sqrt(3) — minimum
            # node spacing that admits a centred trinomial with
            # positive probabilities for typical drift/sigma ratios.
            self._dx.append(v * math.sqrt(3.0))

            branching = _Branching()
            for j in range(j_min, j_max + 1):
                x = self._x0 + j * self._dx[i]
                m = process.expectation_1d(t, x, dt)
                # Central-branch index = round((m - x0) / dx[i+1]).
                temp = math.floor((m - self._x0) / self._dx[i + 1] + 0.5)

                # Optional positivity floor — keep the lowest branch
                # node strictly positive (BlackKarasinski uses log
                # rates and needs r > 0).
                if is_positive:
                    while self._x0 + (temp - 1) * self._dx[i + 1] <= 0.0:
                        temp += 1

                # ``e`` is the centring residual: the diff between
                # the true expectation and the discrete centre.
                e = m - (self._x0 + temp * self._dx[i + 1])
                e2 = e * e
                e3 = e * math.sqrt(3.0)

                # Branching probabilities — match the first two moments
                # of the conditional distribution (and recombine the
                # third by virtue of the centre choice). Same formulas
                # as C++ trinomialtree.cpp:64-66.
                p1 = (1.0 + e2 / v2 - e3 / v) / 6.0
                p2 = (2.0 - e2 / v2) / 3.0
                p3 = (1.0 + e2 / v2 + e3 / v) / 6.0

                branching.add(temp, p1, p2, p3)

            self._branchings.append(branching)
            j_min = branching.j_min
            j_max = branching.j_max

    # --- inspectors -------------------------------------------------------

    @property
    def x0(self) -> float:
        """Initial state value.

        # C++ parity: protected ``x0_`` (trinomialtree.hpp:58).
        """
        return self._x0

    @property
    def time_grid(self) -> TimeGrid:
        """Underlying TimeGrid (alias property for the C++ ``timeGrid()``).

        # C++ parity: ``TrinomialTree::timeGrid()`` (trinomialtree.hpp:49).
        """
        return self._time_grid

    def dx(self, i: int) -> float:
        """Node spacing at slice ``i``.

        # C++ parity: ``TrinomialTree::dx(Size)`` (trinomialtree.hpp:48).
        """
        return self._dx[i]

    # --- Tree contract — implemented from the branchings table -----------

    def size(self, i: int) -> int:
        """Number of nodes at slice ``i``.

        # C++ parity: ``TrinomialTree::size`` (trinomialtree.hpp:84-86).
        """
        # Slice 0 is a single node; slice i (>= 1) reads its size from
        # the previous branching's child-bound width.
        if i == 0:
            return 1
        return self._branchings[i - 1].size()

    def underlying(self, i: int, index: int) -> float:
        """Underlying value at node ``(i, index)``.

        # C++ parity: ``TrinomialTree::underlying`` (trinomialtree.hpp:88-94).
        """
        if i == 0:
            return self._x0
        # Index 0 of slice i corresponds to j = j_min of the branching
        # that produced it (branchings[i-1]).
        return self._x0 + (self._branchings[i - 1].j_min + index) * self._dx[i]

    def descendant(self, i: int, index: int, branch: int) -> int:
        """Descendant index at slice ``i + 1``.

        # C++ parity: ``TrinomialTree::descendant`` (trinomialtree.hpp:96-99).
        """
        return self._branchings[i].descendant(index, branch)

    def probability(self, i: int, index: int, branch: int) -> float:
        """Branch probability at node ``(i, index)`` taking ``branch``.

        # C++ parity: ``TrinomialTree::probability`` (trinomialtree.hpp:101-103).
        """
        return self._branchings[i].probability(index, branch)


__all__ = ["TrinomialTree"]

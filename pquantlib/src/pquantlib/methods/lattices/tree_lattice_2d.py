"""TreeLattice2D — two-dimensional tree-based lattice.

# C++ parity: ql/methods/lattices/lattice2d.hpp (v1.43) —
#             ``template <class Impl, class T = TrinomialTree>
#              class TreeLattice2D``.

Two independent trinomial trees are laid out as a single index space —
node ``index`` at slice ``i`` means ``(index % size1(i), index // size1(i))``
— and the branch index is split the same way. The joint branch
probability is the product of the two marginals plus a correlation
correction ``rho * m[branch1][branch2] / 36``, where ``m`` is one of two
hard-coded 3x3 integer patterns selected by the *sign* of the
correlation (``rho`` itself enters as ``|correlation|``).

The lattice is primarily used for the G2 short-rate model, whose
``TwoFactorModel::ShortRateTree`` supplies the missing ``discount``.

Two C++ members have no useful 2-D meaning and are dead ends there too:

  * ``grid(Time)`` is overridden to ``QL_FAIL("not implemented")``
    (lattice2d.hpp:52, flagged "smelly" in the source);
  * ``underlying(i, index)`` simply does not exist on the 2-D lattice —
    C++ never instantiates ``TreeLattice1D`` here, so nothing asks for
    it. PQuantLib's ``Lattice`` derives from ``Tree``, which declares
    ``underlying`` abstract, so the port has to answer the question;
    it answers it the same way the C++ ``grid`` does, by failing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from pquantlib import qassert
from pquantlib.methods.lattices.tree_lattice import TreeLattice

if TYPE_CHECKING:
    from pquantlib.math.array import Array
    from pquantlib.methods.lattices.trinomial_tree import TrinomialTree

# C++ parity: lattice2d.hpp:77-98. The two 3x3 patterns are reflections of
# each other about the anti-diagonal; ``correlation < 0`` selects the second.
# Stored as tuples rather than a ``Matrix`` because they are compile-time
# constants in every practical sense and are only ever read element-wise.
_M_NEGATIVE_CORRELATION: Final[tuple[tuple[float, ...], ...]] = (
    (-1.0, -4.0, 5.0),
    (-4.0, 8.0, -4.0),
    (5.0, -4.0, -1.0),
)
_M_POSITIVE_CORRELATION: Final[tuple[tuple[float, ...], ...]] = (
    (5.0, -4.0, -1.0),
    (-4.0, 8.0, -4.0),
    (-1.0, -4.0, 5.0),
)


class TreeLattice2D(TreeLattice):
    """Two-dimensional lattice built from two trinomial trees.

    # C++ parity: ``TreeLattice2D<Impl, T>`` (lattice2d.hpp:41-57 +
    # the template definitions at lattice2d.hpp:62-131).

    C++ templates on the tree type ``T`` (defaulting to ``TrinomialTree``)
    and reads ``T::branches``; ``TrinomialTree`` is the only instantiation
    in the library and the hard-coded 3x3 ``m_`` patterns only make sense
    for a 3-branch tree, so the port pins the parameter to
    :class:`TrinomialTree`.
    """

    def __init__(
        self,
        tree1: TrinomialTree,
        tree2: TrinomialTree,
        correlation: float,
    ) -> None:
        # C++ parity: lattice2d.hpp:70-99 — the base gets tree1's time grid
        # and ``T::branches * T::branches`` branches (9 for trinomial trees).
        branches = tree1.branches
        super().__init__(time_grid=tree1.time_grid, n_branches=branches * branches)
        self._tree1: TrinomialTree = tree1
        self._tree2: TrinomialTree = tree2
        self._branches: int = branches
        # C++ parity: ``rho_(std::fabs(correlation))`` (lattice2d.hpp:75) —
        # the sign is consumed entirely by the ``m_`` selection below.
        self._rho: float = abs(correlation)
        self._m: tuple[tuple[float, ...], ...] = (
            _M_NEGATIVE_CORRELATION
            if (correlation < 0.0 and branches == 3)
            else _M_POSITIVE_CORRELATION
        )

    # --- inspectors -------------------------------------------------------

    @property
    def tree1(self) -> TrinomialTree:
        """First factor's tree.

        # C++ parity: protected ``tree1_`` (lattice2d.hpp:50).
        """
        return self._tree1

    @property
    def tree2(self) -> TrinomialTree:
        """Second factor's tree.

        # C++ parity: protected ``tree2_`` (lattice2d.hpp:50).
        """
        return self._tree2

    # --- Tree contract ----------------------------------------------------

    def size(self, i: int) -> int:
        """Joint node count = ``size1(i) * size2(i)``.

        # C++ parity: ``TreeLattice2D<Impl,T>::size`` (lattice2d.hpp:63-65).
        """
        return self._tree1.size(i) * self._tree2.size(i)

    def descendant(self, i: int, index: int, branch: int) -> int:
        """Joint descendant index at slice ``i + 1``.

        # C++ parity: ``TreeLattice2D<Impl,T>::descendant`` (lattice2d.hpp:102-115).
        """
        modulo = self._tree1.size(i)
        index1 = index % modulo
        index2 = index // modulo
        branch1 = branch % self._branches
        branch2 = branch // self._branches

        modulo = self._tree1.size(i + 1)
        return self._tree1.descendant(i, index1, branch1) + (
            self._tree2.descendant(i, index2, branch2) * modulo
        )

    def probability(self, i: int, index: int, branch: int) -> float:
        """Joint branch probability, with the correlation correction.

        # C++ parity: ``TreeLattice2D<Impl,T>::probability`` (lattice2d.hpp:117-131).
        """
        modulo = self._tree1.size(i)
        index1 = index % modulo
        index2 = index // modulo
        branch1 = branch % self._branches
        branch2 = branch // self._branches

        prob1 = self._tree1.probability(i, index1, branch1)
        prob2 = self._tree2.probability(i, index2, branch2)
        # C++ leaves the literal 36 in place with a "does this depend on
        # T::branches?" comment (lattice2d.hpp:129); carried verbatim.
        return prob1 * prob2 + self._rho * self._m[branch1][branch2] / 36.0

    # --- members with no 2-D meaning --------------------------------------

    def grid(self, t: float) -> Array:
        """Always fails — a 2-D lattice has no single underlying row.

        # C++ parity: ``Array grid(Time) const override { QL_FAIL("not
        # implemented"); }`` (lattice2d.hpp:52).
        """
        del t
        qassert.fail("not implemented")

    def underlying(self, i: int, index: int) -> float:
        """Always fails — see the module docstring.

        # C++ parity: no such member on ``TreeLattice2D``; PQuantLib's
        # ``Tree`` declares it abstract, so the 2-D lattice answers the
        # same way ``grid`` does.
        """
        del i, index
        qassert.fail("not implemented")


__all__ = ["TreeLattice2D"]

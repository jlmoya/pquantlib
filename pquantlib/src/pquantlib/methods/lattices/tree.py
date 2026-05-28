"""Tree base class for lattice methods.

# C++ parity: ql/methods/lattices/tree.hpp (v1.42.1) — ``template <class T>
#             class Tree`` (Curiously Recurring Template Pattern).

C++ uses CRTP — ``Tree<T>`` where ``T`` is the concrete derived class —
so the static-dispatch indirection happens at compile time without
virtual-call overhead. The pattern is a stand-in for late binding in
classes whose interface contract is enforced by convention rather
than by language features:

    template <class T>
    class Tree : public CuriouslyRecurringTemplate<T> {
      public:
        // Derived must provide:
        //   Real underlying(Size i, Size index) const;
        //   Size size(Size i) const;
        //   Size descendant(Size i, Size index, Size branch) const;
        //   Real probability(Size i, Size index, Size branch) const;
        ...
    };

Python's lookup is already dynamic, so we collapse CRTP to a normal
abstract base class with the four virtual methods as ``@abstractmethod``.
The C++ ``columns()`` constant — number of branches per node — is
exposed as the abstract ``branches`` class attribute (concrete trees
set ``branches = 2`` for binomial, ``branches = 3`` for trinomial).

PEP 695 generic syntax ``class Tree[T]: ...`` carries the ``T`` for
the underlying value type (typically ``float`` for Real-valued trees,
but ``np.ndarray`` for vector trees). The base does not constrain ``T``
beyond what the four abstract methods need.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Tree[T](ABC):
    """Tree approximating a single-factor diffusion.

    # C++ parity: ``Tree<T>`` (tree.hpp:50-57).

    Subclasses must implement ``size``, ``underlying``, ``descendant``,
    and ``probability``. The number of branches per node is exposed via
    the class attribute ``branches`` (set by the concrete subclass).
    """

    # C++ stores ``columns_`` (number of columns / time slices). Python
    # mirrors that with an instance attribute set by ``__init__``.
    def __init__(self, columns: int) -> None:
        # C++ parity: ``Tree(Size columns)`` (tree.hpp:53).
        self._columns: int = columns

    def columns(self) -> int:
        """Number of time-slice columns in the tree.

        # C++ parity: ``Tree::columns()`` (tree.hpp:54).
        """
        return self._columns

    # --- abstract contract -----------------------------------------------

    @abstractmethod
    def size(self, i: int) -> int:
        """Number of nodes at time-slice ``i``."""

    @abstractmethod
    def underlying(self, i: int, index: int) -> T:
        """Underlying value at node ``(i, index)``."""

    @abstractmethod
    def descendant(self, i: int, index: int, branch: int) -> int:
        """Node index reached from ``(i, index)`` via branch ``branch``
        at the next time slice ``i+1``."""

    @abstractmethod
    def probability(self, i: int, index: int, branch: int) -> float:
        """Probability of traversing ``branch`` from node ``(i, index)``."""

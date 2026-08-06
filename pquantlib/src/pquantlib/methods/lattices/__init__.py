"""Tree and lattice classes used by binomial/trinomial pricers.

# C++ parity: ql/methods/lattices/ (v1.43) — tree.hpp, binomialtree.hpp,
#             trinomialtree.hpp, lattice.hpp, lattice1d.hpp, lattice2d.hpp,
#             bsmlattice.hpp, tflattice.hpp.

The lattice hierarchy mirrors C++ one-for-one, minus the CRTP
indirection (Python's method lookup is already dynamic)::

    Lattice
      └── TreeLattice            (lattice.hpp)
            ├── TreeLattice1D    (lattice1d.hpp)
            │     └── BlackScholesLattice  (bsmlattice.hpp)
            │           └── TsiveriotisFernandesLattice  (tflattice.hpp)
            └── TreeLattice2D    (lattice2d.hpp)

Submodules are imported directly (``from pquantlib.methods.lattices.
binomial_tree import CoxRossRubinstein``); this module re-exports the
lattice bases only, because they are the ones referenced by name from
outside the package.
"""

from pquantlib.methods.lattices.tree_lattice import TreeLattice
from pquantlib.methods.lattices.tree_lattice_1d import TreeLattice1D
from pquantlib.methods.lattices.tree_lattice_2d import TreeLattice2D

__all__ = ["TreeLattice", "TreeLattice1D", "TreeLattice2D"]

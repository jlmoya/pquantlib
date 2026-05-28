"""Numerical-method scaffolding (lattices, MC paths, FD schemes).

# C++ parity: ql/methods/* + ql/numericalmethod.hpp + ql/discretizedasset.hpp
#             (v1.42.1).

The L5 layer ports the tree/lattice abstraction (``methods/lattices``)
plus the cross-cluster ``Protocols`` collected in :mod:`protocols`.
Concrete trees (Binomial / Trinomial / TFLattice), Monte Carlo paths
and finite-difference schemes are scheduled for later L5 stages.
"""

"""Short-rate model hierarchy.

# C++ parity: ql/models/shortrate/* (v1.42.1).

This namespace package gathers the abstract bases (``ShortRateModel``,
``OneFactorModel``, ``TwoFactorModel``) plus concrete models living in
sibling submodules.

L4-D scope (this branch): two-factor short-rate machinery
(``TwoFactorModel``, ``G2``). One-factor concretes land in L4-B; this
branch defines ``ShortRateModel`` locally as a stop-gap so that
``TwoFactorModel`` can subclass it. When L4-B merges, the merge
resolution will dedupe the definition.
"""

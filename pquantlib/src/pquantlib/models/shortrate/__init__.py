"""Concrete short-rate model hierarchy.

# C++ parity: ql/models/shortrate/ (v1.42.1).

Subpackages:

- ``onefactor`` — one-factor models (Vasicek, HullWhite, CIR, ECIR,
  etc.). Includes ``OneFactorModel`` (abstract) and
  ``OneFactorAffineModel`` (abstract, with closed-form A(t,T) + B(t,T)).

Two-factor models (G2++, etc.) deferred to a later cluster.
"""

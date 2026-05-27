"""One-factor short-rate models.

# C++ parity: ql/models/shortrate/onefactormodel.{hpp,cpp} +
# ql/models/shortrate/onefactormodels/* (v1.42.1).

Hierarchy:

- ``one_factor_model.OneFactorModel`` (abstract, extends
  ``ShortRateModel``).
- ``one_factor_affine_model.OneFactorAffineModel`` (abstract, adds
  closed-form ``A(t, T)`` and ``B(t, T)`` so that
  ``discount_bond(t, T, r) = A(t, T) * exp(-B(t, T) * r)``).
- Concretes: ``vasicek.Vasicek``, ``hull_white.HullWhite``,
  ``cox_ingersoll_ross.CoxIngersollRoss``,
  ``extended_cox_ingersoll_ross.ExtendedCoxIngersollRoss``.

BlackKarasinski deferred per L4-B carve-out (requires the lattice/tree
fitting infrastructure).
"""

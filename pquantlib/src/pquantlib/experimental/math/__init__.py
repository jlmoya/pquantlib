"""Experimental math foundations.

# C++ parity: ql/experimental/math/* (v1.42.1).

This sub-package hosts the experimental math foundations used by the latent
factor model and its copula policies:

  * copula policies — :mod:`gaussian_copula_policy`, :mod:`t_copula_policy`;
  * copula RNGs — :mod:`clayton_copula_rng`, :mod:`frank_copula_rng`,
    :mod:`farlie_gumbel_morgenstern_copula_rng`, :mod:`polar_student_t_rng`;
  * special functions — :mod:`convolved_student_t` (generalised Behrens-Fisher),
    :mod:`gaussian_noncentral_chisquared_polynomial`, :mod:`moore_penrose_inverse`;
  * interpolation / integration — :mod:`laplace_interpolation`,
    :mod:`multidim_integrator`, :mod:`multidim_quadrature`,
    :mod:`piecewise_function`, :mod:`piecewise_integral`;
  * the generic :mod:`latent_model` (the math template that the
    ``experimental.credit`` latent models specialise to the credit domain).
"""

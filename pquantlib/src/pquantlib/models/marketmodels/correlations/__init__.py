"""MarketModels correlations — instantaneous forward/swap correlation structures.

# C++ parity: ql/models/marketmodels/correlations/ (v1.42.1).

W9-B scope (all subclass the W9-A ``PiecewiseConstantCorrelation`` abstract):

- ``exp_correlations`` — ``exponential_forward_correlation`` free function
  (closed-form exponential instantaneous correlation) +
  ``ExponentialForwardCorrelation`` piecewise-constant structure.
- ``time_homogeneous_forward_correlation.TimeHomogeneousForwardCorrelation``
  — wraps a single forward-correlation matrix into the time-homogeneous
  lower-right-shifted family of per-step matrices (+ static
  ``evolved_matrices``).
- ``cot_swap_from_fwd_correlation.CotSwapFromFwdCorrelation`` —
  coterminal-swap correlation derived from a forward correlation via the
  swap<->forward Z-matrix sandwich ``corr(Z C Z^T)``.
"""

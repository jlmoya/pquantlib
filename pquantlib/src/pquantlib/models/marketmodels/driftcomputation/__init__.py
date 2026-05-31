"""MarketModels drift computation — reduced-factor BGM drift calculators.

# C++ parity: ql/models/marketmodels/driftcomputation/ (v1.42.1).

W9-B scope — the per-step drift ``mu * dt`` calculators for the various
market-model parameterizations (Joshi 2003, *Rapid Computation of Drifts in
a Reduced Factor Libor Market Model*; Joshi-Liesch for the swap variants):

- ``lmm_drift_calculator.LMMDriftCalculator`` — log-normal LIBOR market model
  (plain = covariance matrix directly; reduced = pseudo-square-root factor
  reduction).
- ``lmm_normal_drift_calculator.LMMNormalDriftCalculator`` — normal (not
  log-normal) forward-rate dynamics.
- ``smm_drift_calculator.SMMDriftCalculator`` — coterminal-swap market model.
- ``cms_mm_drift_calculator.CMSMMDriftCalculator`` — constant-maturity-swap
  market model.
"""

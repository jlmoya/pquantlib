"""MarketModels evolvers — forward-/swap-rate BGM evolvers (W10-B).

# C++ parity: ql/models/marketmodels/evolvers/ (v1.42.1).

Each evolver is a ``MarketModelEvolver`` (W9 abstract) that advances the rates
step-by-step under a ``MarketModel`` + a ``BrownianGenerator``, applying the
per-step drift (W9-B drift calculators) plus the pseudo-root diffusion.

LMM forward-rate evolvers:

- ``lognormal_fwd_rate_pc.LogNormalFwdRatePc`` — predictor-corrector (the
  canonical BGM evolver).
- ``lognormal_fwd_rate_euler.LogNormalFwdRateEuler`` — Euler scheme.
- ``lognormal_fwd_rate_euler_constrained.LogNormalFwdRateEulerConstrained`` —
  constrained Euler (Fries-Joshi proxy simulation; ``ConstrainedEvolver``).
- ``lognormal_fwd_rate_ipc.LogNormalFwdRateIpc`` — iterative predictor-corrector
  (terminal measure).
- ``lognormal_fwd_rate_balland.LogNormalFwdRateBalland`` — Balland drift
  approximation.
- ``lognormal_fwd_rate_iballand.LogNormalFwdRateIBalland`` — interpolated
  Balland (terminal measure).

Other parameterizations:

- ``normal_fwd_rate_pc.NormalFwdRatePc`` — normal (not log-normal) PC evolver.
- ``lognormal_cotswap_rate_pc.LogNormalCotSwapRatePc`` — coterminal-swap-rate PC.
- ``lognormal_cmswap_rate_pc.LogNormalCmSwapRatePc`` — CM-swap-rate PC.
- ``svdd_fwd_rate_pc.SVDDFwdRatePc`` — SVD-reduced displaced-diffusion PC with
  an external (uncorrelated) vol process.

Vol processes (``volprocesses`` subpackage):

- ``market_model_vol_process.MarketModelVolProcess`` — abstract base.
- ``square_root_andersen.SquareRootAndersen`` — Andersen QE discretization of a
  square-root variance process.
"""

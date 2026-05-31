"""MarketModels — LIBOR Market Model (BGM) core / foundations.

# C++ parity: ql/models/marketmodels/ (v1.42.1).

W9-A pilot scope — the abstract spine + curve-state geometry + mappings +
discounters + grid utilities that ALL downstream marketmodels clusters
(W9-B/C models+evolvers, W10, W11 products+callability) build against.

Abstract spine:

- ``evolution_description.EvolutionDescription`` — rate-time / evolution-time
  grid + first-alive-rate bookkeeping + numeraire-measure free functions.
- ``curve_state.CurveState`` — abstract yield-curve geometry + the three
  discount-ratio conversion free functions.
- ``market_model.MarketModel`` / ``MarketModelFactory`` — pseudo-root /
  covariance generator abstract base.
- ``evolver.MarketModelEvolver`` — abstract forward-rate evolver.
- ``multi_product.MarketModelMultiProduct`` (+ ``CashFlow``) — abstract
  product termsheet + cash-flow generator.
- ``pathwise_multi_product.MarketModelPathwiseMultiProduct`` (+ ``CashFlow``)
  — pathwise (Greeks-aware) product variant.
- ``brownian_generator.BrownianGenerator`` / ``BrownianGeneratorFactory`` —
  abstract Gaussian-increment generator.
- ``piecewise_constant_correlation.PiecewiseConstantCorrelation`` — abstract
  piecewise-constant instantaneous-correlation structure.

Curve-state concretes (``curvestates`` subpackage):

- ``LMMCurveState`` / ``CMSwapCurveState`` / ``CoterminalSwapCurveState``.

Discounters + mappings + utilities:

- ``discounter.MarketModelDiscounter`` — numeraire-rebased log-linear
  discounter.
- ``pathwise_discounter.MarketModelPathwiseDiscounter`` — pathwise variant
  (returns discount + its forward-rate derivatives).
- ``forward_forward_mappings.ForwardForwardMappings`` — fwd-tenor-to-fwd-tenor
  jacobian / Y-matrix / curve-state restriction helpers.
- ``swap_forward_mappings.SwapForwardMappings`` — swap<->forward jacobian +
  Z-matrix helpers + freezing-coefficient swaption implied vol.
- ``utilities`` — ``merge_times`` / ``is_in_subset`` / time-grid validators.

Deferred to later clusters (need not-yet-ported foundations):

- ``marketmodeldifferences`` free functions (``rate_vol_differences`` etc.)
  — depend on ``PiecewiseConstantVariance`` + concrete ``MarketModel``
  implementations (W9-B/C).
- ``swaptionImpliedVolatility`` on ``SwapForwardMappings`` — needs a concrete
  ``MarketModel`` with a ``pseudo_root``; the static jacobian/Z-matrix helpers
  it builds on ARE ported here.
"""

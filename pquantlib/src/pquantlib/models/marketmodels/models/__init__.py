"""MarketModels concrete volatility models (W10-A).

# C++ parity: ql/models/marketmodels/models/ (v1.42.1).

The concrete ``MarketModel`` volatility structures that the W10 evolvers and
caplet-coterminal calibration build against:

- ``flat_vol.FlatVol`` — flat-per-rate vol model; pseudo-roots from a flat vol
  vector + a ``PiecewiseConstantCorrelation`` (the workhorse concrete model).
- ``abcd_vol.AbcdVol`` — Abcd-parametric vol model.
- ``piecewise_constant_variance.PiecewiseConstantVariance`` — abstract
  piecewise-constant variance.
- ``piecewise_constant_abcd_variance.PiecewiseConstantAbcdVariance`` —
  abcd-form piecewise-constant variance.
- ``pseudo_root_facade.PseudoRootFacade`` — wraps calibrated pseudo-roots as a
  ``MarketModel``.
- ``cot_swap_to_fwd_adapter.CotSwapToFwdAdapter`` (+ factory) — coterminal-swap
  -> forward-rate dynamics adapter.
- ``fwd_to_cot_swap_adapter.FwdToCotSwapAdapter`` (+ factory) — forward-rate ->
  coterminal-swap dynamics adapter.
- ``fwd_period_adapter.FwdPeriodAdapter`` — fine-grid -> coarse-period adapter.
- ``volatility_interpolation_specifier.VolatilityInterpolationSpecifier`` —
  abstract synthetic-rate vol interpolator (for calibration).
- ``volatility_interpolation_specifier_abcd.VolatilityInterpolationSpecifierabcd``
  — abcd-form specifier.

Shared helpers:

- ``abcd_function.AbcdFunction`` — the Rebonato abcd instantaneous-volatility
  functional with covariance / variance integrals (the ``AbcdFunction`` from
  ``ql/termstructures/volatility/abcd.hpp``).
- ``pseudo_sqrt.rank_reduced_sqrt`` — spectral rank-reduced pseudo-square-root
  (``ql/math/matrixutilities/pseudosqrt.hpp`` ``rankReducedSqrt``).
"""

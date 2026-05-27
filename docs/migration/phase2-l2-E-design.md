# Phase 2 L2-E — Volatility termstructures

**Date:** 2026-05-26
**Status:** **closed** — merged into `main` via `b5d2519 merge: L2-E`; tagged as part of `pquantlib-phase2-complete` @ `b5d2519`. Final test delta: **+96** (5 commits).
**Predecessor:** `pquantlib-phase2-l2-A-complete` @ `4ace1f0`
**C++ ground truth:** QuantLib v1.42.1 @ `099987f0`
**Style:** lean — leans on [`phase2-design.md`](phase2-design.md) for ground rules.

## What landed

### Volatility abstracts
- `pquantlib.termstructures.volatility_term_structure.VolatilityTermStructure`.
- `pquantlib.termstructures.volatility.smile_section.SmileSection`.
- `pquantlib.termstructures.volatility.flat_smile_section.FlatSmileSection`.

### Black family (equity/FX)
- `pquantlib.termstructures.volatility.equity_fx.black_vol_term_structure.BlackVolTermStructure` (abstract) + `BlackVolatilityTermStructure` + `BlackVarianceTermStructure` (adapter intermediates).
- `pquantlib.termstructures.volatility.equity_fx.black_constant_vol.BlackConstantVol`.
- `pquantlib.termstructures.volatility.equity_fx.black_variance_curve.BlackVarianceCurve`.
- `pquantlib.termstructures.volatility.equity_fx.black_variance_surface.BlackVarianceSurface`.

### Local family
- `pquantlib.termstructures.volatility.equity_fx.local_vol_term_structure.LocalVolTermStructure` (abstract).
- `pquantlib.termstructures.volatility.equity_fx.local_constant_vol.LocalConstantVol`.
- `pquantlib.termstructures.volatility.equity_fx.local_vol_curve.LocalVolCurve`.
- `pquantlib.termstructures.volatility.equity_fx.local_vol_surface.LocalVolSurface`.

## Documented divergences

- **Moving-reference-date construction (mode 3)** deferred everywhere (`VolatilityTermStructure`, `BlackVolTermStructure`, `LocalVolTermStructure`, `BlackConstantVol`, `LocalConstantVol`). Awaits `ObservableSettings.evaluation_date` observer wiring.
- **`SmileSection` floating-via-global-eval-date deferred.** Date-anchored mode requires an explicit `reference_date`.
- **`SmileSection` option pricing methods** (`option_price` / `vega` / `density` / implied-vol conversion) deferred to Phase 3 (need `BlackFormula`).
- **`BlackVarianceCurve` `BlackVolTimeExtrapolation`** strategies (FlatVolatility / UseInterpolator / LinearVariance) collapsed to default FlatVolatility. Custom interpolator selection via `setInterpolation<Interpolator>` deferred (linear pinned).
- **`LocalVolSurface` flat-curve simplification** (zero risk-free, zero dividend, forward = spot). Full version with yield curves deferred until L2-B+L3 wire `discount_curve` and `dividend_curve`.

## Independence

L2-E was the only **truly independent** Phase 2 cluster — no cross-cluster Protocol dependencies. Self-contained against L2-A's `TermStructure` + L1-E's interpolations + L1-E's matrix utilities (numpy/scipy). Validated the design — when a cluster fits entirely under one L2-A abstract base, no Protocols are needed.

## Carve-outs

ZABR / SABR / XABR families (depend on optimization concretes deferred from Phase 1). Capfloor / Optionlet / Swaption volatility surfaces (specialized; Phase 4 alongside short-rate models). `HestonBlackVolSurface`, `GridModelLocalVolSurface`, Andreasen-Huge variants — model-coupled; Phase 4.

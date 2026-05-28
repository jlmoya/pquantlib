# Phase 5 L5-E — Exotic instruments + analytic engines

**Date:** 2026-05-28
**Status:** **closed** — merged via `d322fca merge: L5-E`; tagged as part of `pquantlib-phase5-complete` @ `d322fca`. Final test delta: **+97**. 8 commits.
**Predecessor:** `pquantlib-phase5-l5-A-complete` @ `aa19340`
**Style:** lean — leans on [`phase5-design.md`](phase5-design.md).

## What landed

### L1 carry-overs closed
- `FloatingTypePayoff` + `PercentageStrikePayoff` — payoff carve-outs needed for lookback + cliquet.
- **`BivariateCumulativeNormalDistribution`** via scipy.stats.multivariate_normal.cdf (Genz-Bretz at-or-above C++ Genz-2004 We04DP precision). `Dr78` alias points to same class. **Closes Phase 1 L1-B carve-out.**

### Exotic instrument families
- `AsianOption` family — `ContinuousAveragingAsianOption` (additive — L5-C ported Discrete with stricter validation; both coexist).
- `BarrierOption` + `BarrierType` IntEnum (DownIn / UpIn / DownOut / UpOut).
- `BasketOption` + `BasketPayoff` hierarchy (Min / Max / Average / Spread).
- `ContinuousFloatingLookbackOption` + `ContinuousFixedLookbackOption`.
- `CliquetOption`.
- `DigitalOption` alias (leverages L3-A CashOrNothingPayoff / AssetOrNothingPayoff).

### Analytic engines
- `AnalyticContinuousGeometricAveragePriceAsianEngine` (Kemna-Vorst).
- `AnalyticDiscreteGeometricAveragePriceAsianEngine` (Levy 1997 + `blackScholesTheta` helper inlined).
- `AnalyticBarrierEngine` (Reiner-Rubinstein 4-of-6 A/B/C/D/E/F dispatch with NaN traps mirrored).
- `AnalyticBinaryBarrierEngine` (Reiner-Rubinstein 8-branch table; degenerate KO/KI delegating to AnalyticEuropeanEngine for KI).
- `StulzEngine` (Stulz 1982 2-asset min/max basket with put-parity construction).
- `AnalyticContinuousFloatingLookbackEngine` (Conze-Viswanathan closed-form).

## Documented divergences

- All NPVs cross-validated TIGHT vs C++ (closed-form algebraic).
- L5-C's DiscreteAveragingAsianOption retained over L5-E's at merge (L5-C had stricter validation tests).
- `ContinuousAveragingAsianOption.is_expired` returns False (Settings.evaluationDate carve-out — same as VanillaOption).

## Carve-outs (Phase 6+ or permanent)

- MC engines for these (Phase 6 — barriers/basket/lookback MC).
- FD engines for barriers (FdBlackScholesBarrierEngine).
- Continuous-monitoring barrier with discrete-correction adjustments (BroadieGlassermanKou).
- DoubleBarrierOption, PartialTimeBarrierOption.
- HolderExtensibleOption, ComplexChooserOption, CompoundOption.
- 3+ asset baskets.
- Soft-barrier engines.
- AnalyticCompound, AnalyticChooser, etc.
- GapPayoff-only digital engines.

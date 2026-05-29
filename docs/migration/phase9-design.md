# Phase 9 — Cubic/Bicubic + Post-L8 ergonomics + SABR cube (design)

**Date:** 2026-05-28
**Status:** drafted; run-to-completion per user directive ("Phase 9 in its totality")
**Predecessor:** `pquantlib-phase8-complete` @ `dec05fb` — 2303/0/0, pyright + ruff clean
**C++ ground truth:** QuantLib v1.42.1 @ `099987f0`

## Scope

Three carve-out closures bundled into a **pilot + 2-parallel** topology because L9-A's cubic interpolation is a hard dependency of L9-C's `InterpolatedSmileSection`. Total ~22 classes / +90-110 tests target.

### L9-A pilot — Cubic + Bicubic spline interpolators (~5 classes)

Closes the L1-E carve-out for cubic-family interpolators. Delegates to `scipy.interpolate.{CubicSpline,PchipInterpolator,RectBivariateSpline}` to avoid re-implementing 820 LOC of templated cubic-spline machinery; C++ ground truth used at pillar nodes for TIGHT-tier exactness.

- `pquantlib.math.interpolations.cubic_interpolation.CubicInterpolation(x, y, derivative_approx, monotonic, left_condition, left_value, right_condition, right_value)` — top-level cubic with boundary conditions. C++ supports 4 derivative-approximation styles (Spline, SplineOM1, SplineOM2, Akima, Kruger, etc.) × 4 monotonicity-preserving filters. Python ports the `Spline + Natural BC + non-monotonic` core (the SciPy default) plus `Pchip` (monotonic Hyman/Fritsch-Carlson via `PchipInterpolator`). Other variants documented as carve-outs.
- `pquantlib.math.interpolations.cubic_interpolation.CubicNaturalSpline(x, y)` — convenience: natural BC, non-monotonic.
- `pquantlib.math.interpolations.cubic_interpolation.MonotonicCubicNaturalSpline(x, y)` — convenience: PCHIP (monotonic).
- `pquantlib.math.interpolations.cubic_interpolation.KrugerCubic(x, y)` — Kruger monotonic cubic (mentioned in `CubicInterpolation` styles; needs custom implementation since scipy doesn't ship it). Defer to carve-out if scope-tight.
- `pquantlib.math.interpolations.bicubic_spline.BicubicSpline(x, y, z)` — 2D bicubic via `scipy.interpolate.RectBivariateSpline(kx=3, ky=3)`.

**L9-A also opt-in upgrades L8-C's surfaces:** `CapFloorTermVolCurve` and `CapFloorTermVolSurface` gain an `interpolator=` kwarg (defaults still Linear/Bilinear for backward compatibility; users can pass `CubicNaturalSpline` / `BicubicSpline` to match C++ defaults).

### L9-B parallel — Post-L8 ergonomic follow-ups (~7-8 classes)

Closes the post-Phase-8 carry-overs in credit + yield-curve bootstrap.

- `pquantlib.pricingengines.credit.isda_cds_engine.IsdaCdsEngine(default_curve, recovery_rate, discount_curve, include_settlement_date_flows, numerical_fix, accrual_bias, forwards_in_coupon_period)` — ISDA-standard CDS engine (`ql/pricingengines/credit/isdacdsengine.hpp`). 3 step-style options (HalfDayBias, NoBias, Disabled).
- `pquantlib.instruments.make_cds.MakeCDS(...)` — fluent factory mirroring C++ `MakeCDS`. Builds a `CreditDefaultSwap` with sensible defaults + chainable `.with_*()` setters.
- `pquantlib.instruments.credit_default_swap.CreditDefaultSwap.implied_hazard_rate(target_npv, ...) -> float` — Brent root-finder over a `FlatHazardRate`.
- `pquantlib.instruments.credit_default_swap.CreditDefaultSwap.conventional_spread(...) -> float` — Brent root-finder for the conventional (par) spread.
- `pquantlib.termstructures.yield.piecewise_yield_curve.PiecewiseYieldCurve(traits, reference_date, instruments, day_counter, interpolator=Linear)` — concrete piecewise-bootstrapped yield curve wrapping `IterativeBootstrap[YieldTermStructure, Traits]` from L8-A.
- `pquantlib.termstructures.yield.yield_traits.Discount` / `ZeroYield` / `ForwardRate` traits — the 3 standard yield-curve bootstrap traits.
- `pquantlib.termstructures.credit.piecewise_default_curve.PiecewiseDefaultCurve` wire-up — replace the scaffold from Phase 8 L8-B with full bootstrap dispatch (uses L8-A `IterativeBootstrap[DefaultProbabilityTermStructure, Traits]`).

### L9-C parallel — SABR swaption smile cube (~10 classes)

Closes the Phase-8-deferred SABR cube + smile-section family.

- **Smile sections** (under `pquantlib.termstructures.volatility.smile_section.*`):
  - `smile_section.SmileSection` abstract — `volatility(strike) -> float`, `variance(strike) -> float`, `atm_level() -> float`, `optionPrice(strike, type, discount=1.0) -> float`, `digitalOptionPrice(strike, type) -> float`, `density(strike) -> float`.
  - `flat_smile_section.FlatSmileSection(option_date, vol, day_counter, reference_date=None, atm_level=None)`.
  - `interpolated_smile_section.InterpolatedSmileSection(option_date, strikes, vols, atm_level, day_counter, vol_type=ShiftedLognormal, displacement=0)` — uses L9-A's `CubicInterpolation`.
  - `sabr_smile_section.SabrSmileSection(option_date, forward, sabr_params, day_counter)` — closed-form Hagan 2002 evaluation.
  - `spreaded_smile_section.SpreadedSmileSection(base, vol_spread)`.
- **SABR math** (under `pquantlib.math.interpolations.*`):
  - `sabr_formula.sabr_volatility(strike, forward, expiry, alpha, beta, nu, rho) -> float` — Hagan 2002 closed-form, lognormal vol.
  - `sabr_formula.sabr_normal_volatility(strike, forward, expiry, alpha, beta, nu, rho) -> float` — bachelier-vol variant.
  - `sabr_interpolation.SabrInterpolation(strikes, vols, expiry, forward, alpha=None, beta=None, nu=None, rho=None, ...)` — fits SABR params to a strike-vol slice via `scipy.optimize.least_squares` over the 4 params (with optional pinned values).
- **Swaption vol cubes** (under `pquantlib.termstructures.volatility.swaption.*`):
  - `swaption_volatility_cube.SwaptionVolatilityCube` abstract (Xabr-style) — adds strike dimension to `SwaptionVolatilityDiscrete`.
  - `sabr_swaption_volatility_cube.SabrSwaptionVolatilityCube(atm_vol_structure, option_tenors, swap_tenors, strike_spreads, vol_spreads, swap_index_base, short_swap_index_base, vega_weighted_smile_fit, sabr_params_per_grid_point, ...)` — fitted SABR per (expiry, tenor) grid point.
  - `interpolated_swaption_volatility_cube.InterpolatedSwaptionVolatilityCube` — interpolated strike-vol grid per (expiry, tenor) — no SABR fit.

## Cluster topology

- **L9-A pilot first** because L9-C's `InterpolatedSmileSection` imports L9-A's `CubicInterpolation`. Pilot lands on main, both L9-B and L9-C branch off the post-L9-A tip.
- **L9-B and L9-C run in parallel** — credit ergonomics + SABR cube touch disjoint module trees.
- **No new cross-cluster Protocols needed** — `SmileSection` is a concrete abstract that L9-C consumers import directly; not a Protocol.

```
main (pquantlib-phase8-complete)
  │
  ├── L9-A pilot worktree → merge to main
  │     ↓ post-L9-A tip
  │     ├── L9-B parallel worktree → merge
  │     └── L9-C parallel worktree → merge
  ↓
pquantlib-phase9-complete
```

## Carve-outs (deferred — Phase 10+ or never)

### Cubic interpolation leftovers (L9-A)
- `Spline` derivative approximation with non-natural BCs (Clamped, NotAKnot, FirstDerivative, SecondDerivative).
- `SplineOM1`, `SplineOM2` (one-sided / second-derivative monotonicity).
- `Akima` cubic (we already have `AkimaCubic` from Phase 5 L5-A — keep separate).
- `Kruger` cubic — defer if scope-tight; small custom impl, but not on the surface critical path.
- `Hyman`-modified `FritschButland` filters.
- `MultiCubicSpline` (n-D).
- `ChebyshevInterpolation`.
- `BackwardFlatLinear` (composite).
- `AbcdInterpolation` (parametric vol model — superseded by SABR).

### Post-L8 leftovers (L9-B)
- `FaceValueAccrualClaim` (accrual-rebate-conventional CDS) — niche.
- Quanto CDS — out-of-scope domain.
- `experimental/credit/*` (CDO, basket CDS, CDS-on-CDS) — out-of-scope.
- `BootstrapError` / `LocalBootstrap` (alternative bootstrap algorithms — `IterativeBootstrap` is the production one).

### SABR cube leftovers (L9-C)
- `ZabrSmileSection` + `ZabrInterpolatedSmileSection` (ZABR is a SABR generalization with extra parameters; specialty).
- `Gaussian1dSwaptionVolatility` (1-factor model adaptation — needs L4's Gaussian1d model which is itself carved out).
- `CmsMarket` + `CmsMarketCalibration` (CMS-specific helpers).
- `KahaleSmileSection` (no-arbitrage smile reformulation; numeric stability work).
- `AtmAdjustedSmileSection` + `AtmSmileSection` (ATM-targeted adapters; small but niche).
- `SabrInterpolatedSmileSection` (an adapter combining SabrInterpolation fit + InterpolatedSmileSection wrapping — can be done by composition).
- `XabrSwaptionVolatilityCube` + general `Xabr` family (we land SABR as the only Xabr instantiation; others — like `Svi` — are speciality).

## Tolerance discipline

| Cluster | Tier | Justification |
|---|---|---|
| L9-A `CubicInterpolation` at pillars | EXACT | C++ pillar values match scipy at floating-point exactness. |
| L9-A intermediate-point evaluation | TIGHT | scipy.interpolate.CubicSpline is bit-identical to standard cubic-spline solvers; round-off ≤ 1e-14. |
| L9-A `BicubicSpline` | TIGHT | RectBivariateSpline matches C++ at known grids. |
| L9-B `IsdaCdsEngine` NPV | LOOSE | ISDA convention involves accrual + day-count + step approximations; matches C++ to ~1e-8. |
| L9-B `implied_hazard_rate` Brent | LOOSE | Brent tolerance set to 1e-10; matches C++ to ~1e-9. |
| L9-B `PiecewiseYieldCurve` | LOOSE | Bootstrap convergence — 1e-12 inner tolerance, ~1e-9 outer roundtrip. |
| L9-C `SabrFormula` Hagan 2002 | TIGHT | Closed-form; matches C++ to ~1e-14 (no log/atan branch differences). |
| L9-C `SabrInterpolation` fit | LOOSE | least_squares converges to the same optimum to ~1e-8; final params may have ~1e-6 noise. |
| L9-C `SabrSmileSection.volatility(K)` | TIGHT | Direct closed-form evaluation. |
| L9-C `SabrSwaptionVolatilityCube` at grid points | LOOSE | Grid-point ATM matches; off-grid interpolation matches C++ to LOOSE. |

## Decision log

| Decision | Rationale |
|---|---|
| **Pilot pattern restored** | L9-C's `InterpolatedSmileSection` consumes L9-A's `CubicInterpolation`. Mirrors L4-A → L4-B/C/D/E and L5-A → L5-B/C/D/E from earlier phases. |
| **`CubicInterpolation` delegates to scipy** | 820 LOC of templated C++ machinery → ~80 LOC of Python that calls `scipy.interpolate.CubicSpline`. Roundtrip-validated against C++ probe at pillar nodes. |
| **Monotonic cubic via PchipInterpolator** | scipy.PchipInterpolator IS Hyman/Fritsch-Carlson monotonic cubic. No custom math required. |
| **Defer `Kruger` cubic** | Small impl (~30 LOC) but not on critical path — record as carve-out. |
| **L9-A retrofits L8-C surfaces with opt-in `interpolator=` kwarg** | Backward compat is preferred: don't change defaults; let users opt in to C++-default cubic. |
| **L9-C `SabrInterpolation` uses scipy.optimize.least_squares** | C++ uses LM via QuantLib's optimizer; we already delegate LM to scipy in Phase 4 (precedent set). |
| **`SwaptionVolatilityCube` abstract over SABR + Interpolated** | Both Cube1 and Cube2 share the (expiry × tenor × strike-spread) grid layout. Sharing the parent is a small win. |
| **No new cross-cluster Protocol** | `SmileSection` is a concrete abstract with type-erased polymorphism (Python's bread-and-butter). Protocols pay off when *parallel* clusters need an isolation boundary; L9-C is the only consumer. |
| **No A6 pause between L9-A merge and L9-B/C dispatch** | User directive: "Phase 9 in its totality, don't stop until 100% done." See [`feedback-phase-runtocompletion`](../../) in agent memory. |

## Plan + executable tasks

See [`phase9-plan.md`](phase9-plan.md).

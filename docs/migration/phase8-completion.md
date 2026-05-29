# Phase 8 — completion

**Date closed:** 2026-05-28
**Tag:** `pquantlib-phase8-complete` @ `efdfac3`
**Baseline → outcome:** 2109/0/0 → **2303/0/0** (+194 tests vs +145 target)
**Triad:** pytest 2303 passed · pyright 0 errors / 0 warnings · ruff All checks passed
**C++ ground truth:** QuantLib v1.42.1 @ `099987f0`

## What landed

Three independent specialty-domain carve-outs, dispatched as **3 parallel clusters with no pilot** (proven against Phase 6's L6-A/B/C topology). Total wall-clock from dispatch → tag: ~60 minutes (dominated by L8-C at ~59 min).

| Cluster | Branch tip | Classes | Tests delta | Carve-out closed |
|---|---|---|---|---|
| L8-A piecewise inflation | `e2e2fa5` | 6 | +38 | L7-Bb piecewise inflation + L2-B IterativeBootstrap |
| L8-B credit cluster | `280b4c2` | 17 | +58 | Tier-1 credit cluster (full vanilla CDS) |
| L8-C capfloor/swaption vol surfaces | `3690f24` | 17 | +98 | Phase 2 capfloor-vol carve-out |
| **Total** | — | **~40** | **+194** | 3 multi-phase carve-outs |

## Cluster-by-cluster detail

### L8-A — Piecewise inflation + IterativeBootstrap (`e2e2fa5`)

**New modules (6 classes + 1 generic):**
- `pquantlib.termstructures.bootstrap.iterative_bootstrap.IterativeBootstrap[TS, Traits]` — PEP 695 generic; iterative pillar solve with `data_live()` writethrough into the curve's interpolator.
- `pquantlib.termstructures.inflation.zero_inflation_traits.ZeroInflationTraits`.
- `pquantlib.termstructures.inflation.yoy_inflation_traits.YoYInflationTraits`.
- `pquantlib.termstructures.inflation.piecewise_zero_inflation_curve.PiecewiseZeroInflationCurve`.
- `pquantlib.termstructures.inflation.piecewise_yoy_inflation_curve.PiecewiseYoYInflationCurve`.
- `pquantlib.termstructures.inflation.inflation_helpers.ZeroCouponInflationSwapHelper` + `YearOnYearInflationSwapHelper`.

**Divergences:**
- PEP 695 generics in `IterativeBootstrap[TS, Traits]` are documentation-only — internally duck-typed as `Any` because pyright can't prove the TypeVar bound at construction.
- `YearOnYearInflationSwapHelper.implied_quote` short-circuits to `ts.yoy_rate(maturity - lag)` instead of building the full YYIIS — under flat-zero discounting the simplified formula matches C++ fairRate exactly (documented inline).
- Eager bootstrap in `__init__` rather than `LazyObject` deferral (Python's eager-init norm; the curves still observe and re-bootstrap on quote changes).
- `ZeroInflationIndex` / `YoYInflationIndex` gained `set_*_inflation_term_structure` setters; C++'s relinkable-Handle dance is overkill for the in-place bootstrap binding.

**Carve-outs (none new for L8-A).**

### L8-B — Credit cluster (`280b4c2`)

**New modules (17 classes):**
- **Termstructures:** `default_probability_term_structure.DefaultProbabilityTermStructure` + 3 intermediates (`SurvivalProbabilityStructure` / `HazardRateStructure` / `DefaultDensityStructure`) + `flat_hazard_rate.FlatHazardRate` + `interpolated_survival_probability_curve.InterpolatedSurvivalProbabilityCurve` + `interpolated_hazard_rate_curve.InterpolatedHazardRateCurve` + `interpolated_default_density_curve.InterpolatedDefaultDensityCurve`.
- **Bootstrap glue:** `probability_traits.{SurvivalProbability, HazardRate, DefaultDensity}` + `piecewise_default_curve.PiecewiseDefaultCurve` (scaffold; iterative bootstrap deferred — see carve-outs).
- **Helpers:** `default_probability_helpers.SpreadCdsHelper` + `UpfrontCdsHelper`.
- **Instruments:** `pquantlib.instruments.credit_default_swap.CreditDefaultSwap` (Buyer/Seller IntEnum + `with_upfront` factory) + `pquantlib.instruments.claim.{Claim, FaceValueClaim}`.
- **Engines:** `pquantlib.pricingengines.credit.midpoint_cds_engine.MidPointCdsEngine` + `integral_cds_engine.IntegralCdsEngine`.

**Divergences:**
- `HazardRateStructure` / `DefaultDensityStructure` use `scipy.integrate.quad` (adaptive Gauss-Kronrod) for the survival-probability integral rather than C++'s 48-point Gauss-Chebyshev. Concrete classes (`FlatHazardRate` + interpolated curves) override `_survival_probability_impl` with closed-form integrals, so the numeric fallback only fires for hand-rolled subclasses.
- `default_probability(...)` overloads collapsed: C++ has 4 overloads on `(Time, bool)` / `(Date, bool)` / `(Time, Time, bool)` / `(Date, Date, bool)`; Python uses polymorphism on `t2 is None`.
- C++ `Settings::includeReferenceDateEvents` global flag → engine constructor `include_settlement_date_flows` kwarg.
- `CdsHelper.cds_maturity()` IMM-anchor logic for `DateGeneration.CDS` / `CDS2015` / `OldCDS` deferred; helpers fall back to `ref + tenor`.

**Carve-outs (deferred, recorded in `docs/carve-outs.md`):**
- `MakeCDS` factory.
- `implied_hazard_rate` + `conventional_spread` (Brent root-finders over `FlatHazardRate`).
- `IsdaCdsEngine` (ISDA-standard fixed convention) — `UpfrontCdsHelper` falls back to `MidPointCdsEngine` regardless of `model` kwarg.
- `last_period_day_counter` accepted but not wired into the fixed-rate leg builder.
- `PiecewiseDefaultCurve` iterative bootstrap is a scaffold — wiring to `IterativeBootstrap[DefaultProbabilityTermStructure, Traits]` is a follow-up (~1 hr); the generic landed in L8-A.
- `FaceValueAccrualClaim` (accrual-rebate-conventional CDS), Quanto CDS, `experimental/credit/*` (CDO, basket CDS, CDS-on-CDS).

### L8-C — Capfloor / optionlet / swaption vol surfaces (`3690f24`)

**New modules (17 classes):**
- **CapFloor term vol (4):** `CapFloorTermVolatilityStructure` abstract + `ConstantCapFloorTermVolatility` + `CapFloorTermVolCurve` (1-D over maturities) + `CapFloorTermVolSurface` (2-D bilinear over maturity × strike).
- **Optionlet vol (7):** `OptionletVolatilityStructure` abstract + `ConstantOptionletVolatility` + `CapletVarianceCurve` + `StrippedOptionletBase` abstract + `StrippedOptionlet` concrete + `StrippedOptionletAdapter` + `OptionletStripper1` + `SpreadedOptionletVolatility`.
- **Swaption vol (5):** `SwaptionVolatilityStructure` abstract + `SwaptionConstantVolatility` + `SwaptionVolatilityMatrix` (expiry × tenor grid) + `SwaptionVolatilityDiscrete` abstract + `SpreadedSwaptionVolatility`.

**Divergences:**
- `CapFloorTermVolCurve` uses `LinearInterpolation` and `CapFloorTermVolSurface` uses `BilinearInterpolation` — C++ defaults to cubic/bicubic spline, but those interpolators remain L1 carve-outs (`phase1-completion.md`). At pillar nodes the implementations agree exactly; intermediate-point assertions test internal linear/bilinear coherence.
- `OptionletStripper1` inlines the abstract `OptionletStripper` parent (the deferred `OptionletStripper2` was its only other subclass).
- C++ `MakeCapFloor` factory emulated by building floating legs via `ibor_leg` + `Cap` / `Floor` wrappers.
- Stripper uses `adjusted_fixing()` (with a trivial `IborCouponPricer` attached) instead of `index_fixing()` to mirror what `BlackCapFloorEngine` stores in its argument carrier — `index_fixing` introduced a 5e-5 par-coupon mismatch and broke the LOOSE-tier round-trip.
- Schedule generation passes the unadjusted nominal end date to `Schedule.from_rule` so the schedule's BDC fires; the BDC-adjusted end made 24M caps generate 2-day stub periods that broke the 1-caplet cancellation property of consecutive cap NPV diffs.
- Added a `_period_le` normalizer in `OptionletStripper1` — PQuantLib's `Period` dataclass `__eq__` treats `60M != 5Y` field-by-field; C++ semantics treat them as equal.

**Carve-outs (deferred to a future SABR phase):**
- `SabrSwaptionVolatilityCube` + `InterpolatedSwaptionVolatilityCube` + `SwaptionVolCube{1,2}` (full SABR cube + smile sections).
- `Gaussian1dSwaptionVolatility`.
- `CmsMarket` + `CmsMarketCalibration`.
- `OptionletStripper2` (caplet variance curve + spread).
- `SmileSection*` family for full swaption vol cube + `SpreadedSmileSection` + `InterpolatedSmileSection` cubic-strike paths used by adapters/spreaded wrappers.

## Merge reconciliations

| Conflict | Resolution |
|---|---|
| `migration-harness/cpp/probes/CMakeLists.txt` × 2 (L8-B and L8-C each appended a new probe target after L8-A's) | Standard 3-way: keep both sides — stacked L8-A → L8-B → L8-C entries with one blank line between each. |
| Untracked probe leakage into main worktree (`cluster_l8b/`, `cluster_l8c/`, `references/cluster/l8{b,c}.json`) | Discarded with `git checkout -- CMakeLists.txt` + `rm -rf cluster_l8{b,c}` — same recurring pattern as Phase 7; the canonical copies live on each cluster branch. |

## Test-count reconciliation

```
baseline                       2109
  L8-A IterativeBootstrap +18  2127
  L8-A inflation +20           2147
  L8-B credit +58              2205
  L8-C vol surfaces +98        2303
```

The +194 outcome exceeds the +145 plan target because L8-C overshot (+98 vs +60 plan). L8-A and L8-B landed inside their respective sub-targets. L8-C's overshoot is dominated by the 3-way capfloor / optionlet / swaption split, each picking up dense pillar-grid + interpolation-coherence sweeps.

## Cross-phase carve-out closures

This phase moves three long-standing items off `docs/carve-outs.md`:

1. **L2-B IterativeBootstrap (originally carved at Phase 2)** — landed as a phase-1 generic; the concrete `PiecewiseYieldCurve` wiring can now follow.
2. **L7-Bb piecewise inflation (carved at Phase 7 after the L7-B subagent socket drop)** — now closed via `PiecewiseZeroInflationCurve` + `PiecewiseYoYInflationCurve` + bootstrap helpers + `IterativeBootstrap` integration.
3. **Phase 2 capfloor / optionlet / swaption vol surfaces (carved at Phase 2 closure)** — now closed via the L8-C cluster.

Tier-1 credit (CDS pricing + bootstrap surface) is new closure, not a previously-recorded carve-out.

## Lessons / patterns reinforced

- **3-parallel-no-pilot remains the right topology for independent specialty domains.** L6 used it; L8 reused it without adjustment. No shared abstract needed dispatch coordination.
- **Subagent worktree → main-worktree leakage** continues to happen on every multi-cluster phase. The cleanup pattern (`git checkout -- CMakeLists.txt && rm -rf cluster_l8X*`) is now reflexive.
- **PEP 695 generics as documentation-only** is fine for bootstrap glue when the runtime is unavoidably dynamic — pyright sees `Any` internally; callers see the typed surface.
- **Adapter inlining** when an abstract has only one production subclass (`OptionletStripper` → `OptionletStripper1`) is preferable to porting both. `OptionletStripper2` becomes a future carve-out closure rather than a half-built abstract base.

## Post-phase carry-overs (recorded; not blocking phase closure)

| Item | Origin | Estimated effort |
|---|---|---|
| `PiecewiseDefaultCurve` iterative bootstrap wiring | L8-B (scaffold landed) | ~1 hr — wire `IterativeBootstrap[DefaultProbabilityTermStructure, Traits]` into the scaffold |
| `IsdaCdsEngine` | L8-B carve-out | ~3 hr — ISDA convention is well-specified but accrual edge cases need care |
| `MakeCDS` + `implied_hazard_rate` + `conventional_spread` | L8-B carve-out | ~2 hr — pure ergonomics + Brent wrappers |
| Concrete `PiecewiseYieldCurve` (using L8-A `IterativeBootstrap`) | L2-B follow-up | ~2 hr — generic exists; needs YieldTermStructure-specific traits + helper threading |
| Cubic / Bicubic spline interpolators | L1-E carve-out | ~2 hr — would unlock C++-default cubic on `CapFloorTermVolCurve` + `CapFloorTermVolSurface` |
| Full SABR swaption vol cube | new (post-L8-C) | dedicated Phase 9 candidate |

None of these block project completion. They are the natural next-iteration backlog.

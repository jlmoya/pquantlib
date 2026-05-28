# Phase 6 (high-impact carve-outs + final closure) — completion

**Date closed:** 2026-05-28
**Tag:** [`pquantlib-phase6-complete`](../../README.md#migration-status) @ `998fed3` (final close → [`pquantlib-final`](../../README.md#migration-status))
**Predecessor:** `pquantlib-phase5-complete` @ `d322fca`
**Test count:** 1883 → **1958/0/0** (+75). pyright + ruff clean.
**Design spec:** [`phase6-design.md`](phase6-design.md). **Plan:** [`phase6-plan.md`](phase6-plan.md).

## Cluster contribution table

| Cluster | Mode | Commits | Tests added | Closes |
|---|---|---|---|---|
| **L6-A LongstaffSchwartz American MC** | parallel | 3 | +41 | Phase 5 L5-C American MC carve-out |
| **L6-B BatesEngine** | parallel | 1 | +14 | Phase 4 L4-C BatesEngine carve-out |
| **L6-C DoubleBarrierOption + AnalyticDoubleBarrierEngine** | parallel | 3 | +20 | Phase 5 L5-E DoubleBarrier carve-out |
| **Total** | | **7** | **+75** | **~12 classes** |

## Notable Phase 6 scope decision

**The originally-planned "Python 3.14 modernization sweep" was deleted from Phase 6 scope** after audit confirmed the codebase was built with modern idioms from day 1:

- 0 uses of `Generic[T]` (all PEP 695 `class X[T]`).
- 0 uses of `Optional[X]` (all `X | None`).
- 0 uses of `Union[X, Y]` (all `X | Y`).
- 0 uses of `List` / `Dict` / `Tuple` from typing (all builtins).
- 0 uses of bare `@dataclass` (all `frozen=True, slots=True`).

Phase 6 narrowed to: 3 specific high-value carve-out closures + final closure tooling.

## Documented divergences (new in Phase 6)

### LongstaffSchwartz American MC (L6-A)
- **LsmBasisSystem reimplements the C++ recurrence directly** rather than wrapping numpy.polynomial — numpy's probabilist's HermiteE / unscaled Laguerre / Chebyshev2nd conventions don't match C++ out-of-the-box. Subtle gotcha: C++ `i / 2.0` for Hermite β is **float division** (0.5/1.0/1.5), not integer truncation.
- **LongstaffSchwartzPathPricer uses `numpy.linalg.lstsq`** instead of C++ `GeneralLinearLeastSquares` (same QR-based math).
- **MCAmericanEngine surfaces `exercise_probability()`** as a typed engine method instead of C++'s untyped `additionalResults["exerciseProbability"]` dict — pyright-strict friendly.
- `MakeMCAmericanEngine` builder skipped (Python kwargs); `RNG_Calibration` template arg collapsed.
- Path values deep-copied during calibration to defeat L5-C's in-place `PathGenerator` mutation.

### BatesEngine (L6-B)
- **scipy.integrate.quad over (0, +inf)** inherited from L4-C `AnalyticHestonEngine` — LOOSE tier.
- **Static `model() -> BatesModel` narrowing** vs C++ `dynamic_pointer_cast` at call site.
- **`(rel_tolerance, max_evaluations)` ctor overload dropped** (same as `AnalyticHestonEngine`).
- C++ `BatesModel.PositiveConstraint` on λ/δ forces λ=0 reduction probe to use `1e-12`; Python tests use `1e-30` (same constraint, finer resolution).

### DoubleBarrier (L6-C)
- **`_volatility` calls `BlackVolTermStructure.black_vol(last_date, ...)`** (Date-typed overload). Python only binds the Date overload; resolves identically to C++ via day-counter `year_fraction`.
- **`calculate()` restructured to flat `if/elif/elif/elif/else`** instead of nested option_type/barrier_type — ruff PLR5501 prefers it.

## Final closure tooling

Phase 6 also produced the final-closure tooling:

- **`docs/carve-outs.md`** — comprehensive carve-out documentation. 5 categories (specialty domains, engine-pair carry-overs, tooling-boundary replacements, deliberate-deferral follow-ups, items-not-in-C++). Per-item: C++ source location, rationale, access pattern.
- **`pquantlib-samples/`** populated with 4 end-to-end sample programs:
  - `vanilla_swap_pricing.py` — bootstrap curve + price 5y receiver swap.
  - `heston_calibration.py` — LM calibration of Heston to vanilla surface.
  - `american_option_mc.py` — Longstaff-Schwartz American put (matches paper).
  - `double_barrier_analytic.py` — Ikeda-Kunitomo + in-out parity check.
- **Per-cluster lean design docs** at `phase6-l6-{A,B,C}-design.md`.

## Carry-overs closed in Phase 6

- **Phase 4 L4-C**: BatesEngine.
- **Phase 5 L5-C**: LongstaffSchwartz American MC (+ `MCAmericanEngine`).
- **Phase 5 L5-E**: DoubleBarrierOption + AnalyticDoubleBarrierEngine.

Total carry-overs closed across Phases 4-6: **11** (LM + Simplex + Swaption + CapFloor instruments + Settings.evaluation_date + Schedule null-effective-date + RelativeDateBootstrapHelper + SmileSection floating-mode + TermStructure moving-mode + Sobol + Burley2020 + GammaFunction + AkimaCubic + BivariateCumulativeNormal + VanillaOption.implied_volatility + BlackKarasinski + OneFactorModel.tree() + TreeSwaption/TreeCapFloor + LongstaffSchwartz American MC + BatesEngine + DoubleBarrier).

## Carve-outs deferred indefinitely

See [`docs/carve-outs.md`](../carve-outs.md) for the comprehensive list — organized by category (specialty domains, engine-pair carry-overs, tooling-boundary replacements). Highlights:

- MarketModels (125 files of LMM).
- All inflation (instruments + indexes + termstructures + cashflows + engines).
- All credit (CDS + DefaultProbabilityTermStructure + engines).
- Capfloor / optionlet / swaption volatility surfaces.
- ZABR / SABR / XABR volatility model families.
- Specialty short-rate (Gaussian1d / GSR / MarkovFunctional).
- Specialty Heston variants (HestonSLV / BatesDoubleExp / GJR-GARCH / PiecewiseTimeDependentHeston).
- Multi-asset FD (~110 of 120 ql/methods/finitedifferences/ files).
- Multi-asset MC engines.
- TreeLattice2D + G2.tree().
- Exotic option variants (PartialTimeBarrier, SoftBarrier, HolderExtensible, ComplexChooser, Compound, 3+ asset baskets).
- Specialty bonds (BTP, CmsRateBond, CpiBond, ConvertibleBond).
- Specialty cashflows (Digital, Cms, CapFloored, AverageBma).
- 35 region-specialty ibors.

## Lessons learned

- **Modernization sweeps need an audit first.** Day-1 modern-idiom adoption made the planned Phase 6 sweep unnecessary; the audit took ~2 minutes and saved a substantial cluster.
- **Hooks placed during port pay off later.** L4-C left `add_on_term` in `AnalyticHestonEngine` knowing BatesEngine would need it; L6-B's BatesEngine port was a single commit (14 tests) because of that hook.
- **numpy / scipy convention differences are real.** L6-A's LsmBasisSystem had to reimplement Hermite/Laguerre/Chebyshev recurrences directly rather than wrap numpy.polynomial — the unit-normalization vs C++ definitions differed enough to fail probe matches. Future numerical ports should compare conventions up front.
- **`pquantlib-samples/` sample programs are valuable closure artifacts.** They serve as living documentation of the end-to-end pricing path and double as integration tests.

## What's next

**`pquantlib-final` tag** (immediately after this completion doc lands). Phase 6 = the final phase per the project plan.

Future work (not on the roadmap but possible follow-ups): dedicated inflation cluster, dedicated credit cluster, MarketModels cluster, capfloor-vol surface cluster, exotic-MC cluster.

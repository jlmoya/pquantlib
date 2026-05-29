# Phase 11 — Complete C++ QuantLib v1.42.1 closure (design)

**Date:** 2026-05-29
**Status:** drafted
**Predecessor:** `pquantlib-phase10-complete` @ `d3746e4` — 2652/0/0, pyright + ruff clean
**C++ ground truth:** QuantLib v1.42.1 @ `099987f0`
**Target:** literal 1:1 with the C++ v1.42.1 surface — every portable .hpp ported, every deliberate carve-out reconsidered and either ported or formally re-justified.

## Why this is a mega-phase

A normal phase ports ~15-50 classes across 3-5 parallel clusters in ~50-90 minutes wall-clock. **Phase 11 ports the remaining ~500 classes** (~670 class-bearing .hpp files), which is more than the sum of Phases 0-10 combined. Honest framing:

| | Class-bearing .hpp on the C++ surface |
|---|---|
| Total in `ql/` (excluding `all.hpp` aggregators) | ~1,320 |
| Already ported (estimated from 486 Python module files + consolidation factor) | ~650 |
| **Remaining for Phase 11** | **~670** |

Single-shot dispatch isn't realistic. Phase 11 is structured as **12 sequential waves**, each wave equivalent in scope to a normal phase (3-5 parallel clusters, ~25-50 classes, ~60-90 min wall-clock). Each wave terminates at an intermediate tag so partial progress is durable.

## Wave structure + tags

| Wave | Theme | Est. classes | Intermediate tag |
|---|---|---|---|
| **W1** | Specialty model completion (Markov + Bates variants + Heston SLV + Gaussian1d engines) | ~25 | `pquantlib-phase11-w1-complete` |
| **W2** | ZABR + smile + bootstrap + ergonomic follow-ups | ~15 | `pquantlib-phase11-w2-complete` |
| **W3** | `experimental/credit/*` — basket CDS + CDO + synthetic-CDO + N-th-to-default | ~45 | `pquantlib-phase11-w3-complete` |
| **W4** | `experimental/exoticoptions/*` + `experimental/barrieroption/*` + `experimental/varianceoption/*` | ~35 | `pquantlib-phase11-w4-complete` |
| **W5** | `experimental/finitedifferences/*` — multi-asset FD + Heston SLV FDM + 2D operators | ~25 | `pquantlib-phase11-w5-complete` |
| **W6** | `experimental/volatility/*` + `experimental/math/*` | ~46 | `pquantlib-phase11-w6-complete` |
| **W7** | `experimental/processes/*` + `experimental/commodities/*` + `experimental/inflation/*` + `experimental/variancegamma/*` | ~49 | `pquantlib-phase11-w7-complete` |
| **W8** | `experimental/{coupons,swaptions,callablebonds,catbonds,mcbasket,asian,fx,forward,basismodels,averageois,risk,lattices,models,shortrate,termstructures}/*` (long tail) | ~67 | `pquantlib-phase11-w8-complete` |
| **W9** | `marketmodels/products/*` + `marketmodels/callability/*` | ~50 | `pquantlib-phase11-w9-complete` |
| **W10** | `marketmodels/models/*` + `marketmodels/evolvers/*` | ~33 | `pquantlib-phase11-w10-complete` |
| **W11** | `marketmodels/{driftcomputation,correlations,curvestates,pathwisegreeks,browniangenerators}/*` + `marketmodels/*.hpp` root | ~42 | `pquantlib-phase11-w11-complete` |
| **W12** | Tooling-boundary closure + final audit + audit-driven gap fills | ~30 | `pquantlib-phase11-complete` (and `pquantlib-100-complete`) |
| **Total** | — | **~460** | — |

The 460 estimate is conservative; actual count could reach 500+ if `experimental/*` headers map 1:1 to classes (some are `all.hpp` rejected up front; some are typedefs / tags).

## Wave-by-wave detail

### W1 — Specialty model completion (~25 classes)

**Short-rate:**
- `MarkovFunctional` model (542 LOC; bootstrap-vs-swaption-strip calibration over a custom yield curve).
- `Gaussian1dGsrProcess` (process companion — separate from `GsrProcess` already in L10-B).
- `Gaussian1dCapFloorEngine`.
- `Gaussian1dFloatFloatSwaptionEngine`.
- `Gaussian1dNonStandardSwaptionEngine`.

**Equity stochastic-vol:**
- `BatesDetJumpModel`.
- `BatesDoubleExpModel`.
- `BatesDoubleExpDetJumpModel`.
- `GjrGarchModel` + `AnalyticGjrGarchEngine`.
- `HestonSlvFdmModel` (FDM stochastic local vol).
- `HestonSlvMcModel` (MC stochastic local vol).
- `PiecewiseTimeDependentHestonModel`.

**Carve-outs (W1):** none — port the full short-rate + Heston specialty set.

### W2 — ZABR + smile + bootstrap + ergonomic follow-ups (~15 classes)

- `ZabrInterpolation` 5-param fitter + `ZabrInterpolatedSmileSection`.
- ZABR `LocalVolatility` / `FullFd` / `ProjectedHedge` evaluation modes (requires a 1-D FD engine on the ZABR PDE — port the engine first, then wire the modes).
- `XabrSwaptionVolatilityCube<Model>` template generalization (port as a concrete dispatch with class-type parameter; SABR is already special-cased).
- `ConvexMonotoneInterpolation` (Hagan-West 2009).
- `BootstrapError` + `LocalBootstrap` (alternative bootstrap algorithms).
- `CmsMarket` + `CmsMarketCalibration` (CMS specialty helpers).
- `KahaleSmileSection.core_smile` deep-iteration + pathological-arbitrage testing.
- `AbcdCalibration` (separate helper from `AbcdInterpolation` already in L10-C).
- Specialty swap variants: `BMASwap` (already partially in core), `FloatFloatSwap`, `NonstandardSwap`.

### W3 — `experimental/credit/*` (~45 classes)

- N-th-to-default basket CDS (`NthToDefault`, `BasketCDS`).
- Synthetic CDO + tranche (`SyntheticCDO`, `CDOEngine`).
- CDS-on-CDS (`CDSOption`).
- Loss-distribution models (`LossDist`, `LossDistBucketing`, ...).
- Recovery rate models (`ConstantRecoveryRate`, `MidPointCDSEngineWithEvent`, ...).
- Continuous-time Markov stochastic process for credit transitions.
- `RandomDefaultModel` + Monte-Carlo CDO engine.
- Issuer + DefaultEvent infrastructure.
- Pool-of-issuers utilities.

### W4 — `experimental/exoticoptions/*` + `experimental/barrieroption/*` + `experimental/varianceoption/*` (~35 classes)

**experimental/exoticoptions (22):**
- `CompoundOption` + `AnalyticCompoundOptionEngine`.
- `ChooserOption`.
- `AmericanForwardOption`.
- `HimalayaOption` + `MCHimalayaEngine`.
- `EverestOption` + `MCEverestEngine`.
- `PagodaOption` + `MCPagodaEngine`.
- `PartialTimeBarrierOption` + `AnalyticPartialTimeBarrierEngine`.
- `TwoAssetBarrierOption` + 2 engines.
- `TwoAssetCorrelationOption` + 2 engines.
- `KirkSpreadOptionEngine`.
- `HolderExtensibleOption` + `WriterExtensibleOption` + engines.
- `AnalyticPdfHestonEngine`.
- Continuous arithmetic Asian engines (`Levy`, `Vecer`).

**experimental/barrieroption (10):**
- `DoubleBarrierOption` (already in L6-C) + remaining variants.
- `PartialBarrierOption`.
- `SoftBarrierOption`.
- `AnalyticBinaryBarrierEngine`, `BinomialDoubleBarrierEngine`.
- `VannaVolgaBarrierEngine`, `VannaVolgaDoubleBarrierEngine`.

**experimental/varianceoption (3):**
- `VarianceOption`.
- `IntegralHestonVarianceOptionEngine`.

### W5 — `experimental/finitedifferences/*` (~25 classes)

- Multi-asset FDM: `FdmHestonHullWhiteOp`, `FdmDupire1dOp`, etc.
- HestonSLV FDM engine on top of W1 `HestonSlvFdmModel`.
- `Fdm2dBlackScholesOp` (multi-asset Black-Scholes).
- `AdaptiveFdScheme`.
- `LocalVolatilityFDM` engines.
- `BatesFDM` engine.
- `FdmDividendHandler` for cash dividends.
- `FdmExtOUJumpOp` (extended OU with jumps).

### W6 — `experimental/volatility/*` + `experimental/math/*` (~46 classes)

**experimental/volatility (23):**
- `NoArbSabrInterpolation`, `NoArbSabrSmileSection` (NoArbSABR variant).
- `SabrVolSurface`.
- `SviInterpolation`, `SviVolatilityCube` (SVI — Stochastic Volatility Inspired).
- `VolatilityCubeBySabr`.
- `EquityFxVolatilityRatio` + cross-rate adjustments.

**experimental/math (23):**
- `LaplaceInterpolation`.
- `MultidimensionalIntegral`.
- `ParticleSwarmOptimization`.
- `DifferentialEvolution`.
- `FireflyAlgorithm` (heuristic optimizer).
- `BoundaryConstraint`, `NonLinearLeastSquare` (helpers).
- `LatentModel` (factor structure).
- `TCopula`, `Frank`, `Gumbel` (extended copulas beyond L1-B).

### W7 — `experimental/processes/*` + `experimental/commodities/*` + `experimental/inflation/*` + `experimental/variancegamma/*` (~49 classes)

**experimental/processes (7):**
- `VarianceGammaProcess`.
- `KlugeExtOUProcess` (Kluge + extended OU jump process for energy).
- `GsrProcess` (already in L10-B; verify completeness).
- `SquaredOuProcess`.

**experimental/commodities (23):**
- `CommodityIndex` + concretes.
- `CommodityCurve` (forward curves with cost-of-carry).
- `CommodityType` + `UnitOfMeasure`.
- `Energy{Commodity,Future,Swap}` instruments.
- `EnergyVanillaSwap`, `EnergyFutures{ ... }Engine` engines.
- `IsdaCdsHelper`-style for commodities.

**experimental/inflation (12):**
- Multi-curve inflation (separate fwd + ATM-vol curves).
- Cross-currency inflation swap.
- Inflation cap/floor specialty engines beyond L7-D.

**experimental/variancegamma (7):**
- `VarianceGammaModel`.
- `FFTVarianceGammaEngine`.
- `MCEuropeanVarianceGammaEngine`.

### W8 — `experimental/{coupons,swaptions,callablebonds,catbonds,mcbasket,asian,fx,forward,basismodels,averageois,risk,lattices,models,shortrate,termstructures}/*` long tail (~67 classes)

**experimental/coupons (8):**
- `OvernightIndexedSwap` extras (`MakeOIS` is core; experimental adds `OvernightIndexedSwapIndex` variants).
- `IsdaIborCoupon`.
- `LiborCoupon` extras.

**experimental/swaptions (4):**
- `IborCmsSpreadCoupon` swaption.
- `BasisSwapEngine`.

**experimental/callablebonds (7):**
- `CallableBond` + 3 engines (`TreeCallableBondEngine`, `BlackKarasinskiCallableBondEngine`, ...).

**experimental/catbonds (5):**
- `CatBond` instrument + Monte Carlo engine + EventSet infrastructure.

**experimental/mcbasket (8):**
- Generic basket MC pricer + path pricers (multi-asset MC framework).

**experimental/asian (3):**
- Continuous-Asian extensions beyond L5-E.

**experimental/fx (3):**
- `BlackDeltaCalculator`, `FxBlackVolSurface`.

**experimental/forward (2):**
- `NonStandardForwardSwap`.

**experimental/basismodels (4):**
- `TenorBasisSwap`, `LiborOisSpread` curves.

**experimental/averageois (4):**
- Average-OIS rate helpers + engine.

**experimental/risk (3):**
- Sensitivity / curve-bumping helpers.

**experimental/lattices (2):**
- Trinomial Bates lattice + 2D-lattice generalization.

**experimental/models (3):**
- `HwSwaptionEngine` (HW + Jamshidian sub-variant), `PolyakovTreeSwaptionEngine`.

**experimental/shortrate (3):**
- `HwModel`, `G2pp` extras (already partly in L4-D).

**experimental/termstructures (3):**
- Specialty term-structure extensions.

### W9 — `marketmodels/products/*` + `marketmodels/callability/*` (~50 classes)

**marketmodels/products (35):**
- `MarketModelMultiProduct` abstract + N concrete products: `MultiStepCoinitialSwaps`, `MultiStepCoterminalSwaps`, `MultiStepCoterminalSwaptions`, `MultiStepInverseFloater`, `MultiStepNothing`, `MultiStepOptionlets`, `MultiStepPathwiseInverseFloater`, `MultiStepPathwiseWrapper`, `MultiStepPeriodCapletSwaptions`, `MultiStepRatchet`, `MultiStepSwap`, `MultiStepSwaption`, `MultiStepTarn`, `OneStepForwards`, `OneStepOptionlets`, plus their pathwise variants and product testers.

**marketmodels/callability (15):**
- `BermudanSwaption`-style callability framework for LMM.
- `MarketModelBasisSystem`, `MarketModelNodeDataProvider`.
- `LongstaffSchwartzExerciseStrategy` for LMM.
- `UpperBoundEngine` (Andersen-Broadie).
- `LowerBoundEngine` (Andersen-Broadie + Belomestny).
- `CallabilityHelpers`.

### W10 — `marketmodels/models/*` + `marketmodels/evolvers/*` (~33 classes)

**marketmodels/models (19):**
- `LMMNormal` (lognormal forward rate dynamics).
- `LMMAB`, `LMMABCD` parameterizations.
- `MarketModel` abstract + 7 concrete models.
- `PiecewiseConstantAbcdVariance`.
- Volatility models for LMM.

**marketmodels/evolvers (14):**
- `MarketModelEvolver` abstract.
- `LogNormalForwardRatePcEvolver` (predictor-corrector).
- `LogNormalForwardRateIpcEvolver` (interpolated predictor-corrector).
- `LogNormalForwardRateBalland` evolver.
- `LogNormalForwardRateEulerEvolver`.
- `LogNormalCotSwapRateEvolver` (co-terminal swap-rate evolver).
- `NormalFwdRatePcEvolver`.
- Constrained evolvers.

### W11 — `marketmodels/{driftcomputation,correlations,curvestates,pathwisegreeks,browniangenerators}/*` + root (~42 classes)

**marketmodels/driftcomputation (5):**
- `LmmDriftCalculator`, `CmSwapDriftCalculator`, etc.

**marketmodels/correlations (4):**
- `ExponentialForwardCorrelation`, `CotSwapFromFwdCorrelation`, `TimeHomogeneousForwardCorrelation`.

**marketmodels/curvestates (4):**
- `LMMCurveState`, `CMSwapCurveState`, `CoterminalSwapCurveState`, `Cot/CMS curve states`.

**marketmodels/pathwisegreeks (5):**
- `BumpInstrumentJacobian`, `RatePseudoRootJacobian`, `RatePseudoRootJacobianAllElements`, etc.

**marketmodels/browniangenerators (3):**
- `SobolBrownianGenerator`, `MTBrownianGenerator`.

**marketmodels root + remaining (21):**
- `AccountingEngine`.
- `BrownianGenerator` (top-level abstract).
- `CurveState` abstract.
- `Discounter`, `EvolutionDescription`.
- `Evolver` abstract.
- `MarketModel`, `MarketModelMultiProduct`, `MarketModelPathwiseDiscounter`.
- `PathwiseAccountingEngine`.
- `ForwardForwardMappings`.
- `SwapForwardMappings`.
- `UtilityForReductionAlgorithms`.
- `PiecewiseConstantCorrelation`.

### W12 — Tooling-boundary closure + final audit (~30 classes)

**Tooling-boundary items** (per `docs/carve-outs.md` Category 3) — were *deliberately* replaced by numpy/scipy/arch package; Phase 11 ports them anyway for literal 1:1 with C++:

- `ConstantEstimator` (historical-vol constant estimator).
- `GARCHModel` (`models/volatility/garch.hpp`).
- `GarmanKlassSimpleSigma` + `GarmanKlassEstimator`.
- `Parkinson` + `ParkinsonExtended`.
- `SimpleLocalEstimator`.
- `YangZhang`.
- `HistoricalSyntheticInterestRateForwardCurve`.

Plus any items surfaced by **final audit**: a programmatic `diff` of `ql/**/*.hpp` against our Python tree to identify any silent gaps (a `find ql -name '*.hpp' | check_corresponding_python_module` script).

## Tolerance discipline (per-wave)

Each wave inherits the standard 3-tier tolerance discipline. Specific calls:

- **W1 MarkovFunctional** — LOOSE (bootstrap converges to ~1e-8).
- **W1 Heston SLV FDM** — LOOSE (FDM noise).
- **W3 CDO** — LOOSE (Monte-Carlo CDO engines).
- **W5 Multi-asset FDM** — LOOSE (2D operator splitting noise).
- **W9-W11 Market Models** — LOOSE for forward-rate Monte Carlo; TIGHT only for closed-form pieces (correlations, drift adjustments).
- **W12 Tooling-boundary historical vol estimators** — TIGHT for closed-form (Parkinson, GarmanKlass, YangZhang, Constant). GARCH fit — LOOSE.

## Cluster topology per wave

Each wave dispatches **3-5 parallel clusters** (no pilot needed — these are independent specialty closures, like Phase 6/8/10). When a wave does have a hard dependency (e.g., W2 ZABR FD modes need a ZABR PDE engine first), the wave internally uses a pilot pattern.

Per-wave wall-clock estimate: 60-90 min agent time + ~15 min orchestrator overhead (merges, conflicts, tag, doc-sweep).

**Total Phase 11 wall-clock estimate: 12-25 hours of agent time + ~3-5 hours of orchestrator overhead.**

## Risk + escape hatches

| Risk | Mitigation |
|---|---|
| A wave's C++ subtree has fundamental Python-incompatible mechanics (e.g., zero-overhead templates) | Document the specific divergence inline + port the runtime-equivalent. Pattern: see Phase 4 G2 / Phase 9 `IterativeBootstrap[TS, Traits]`. |
| `experimental/*` contains code that is deprecated / abandoned upstream | Identify via C++ comments + commit history. If genuinely abandoned, port with a `# pyright: ignore[deprecated]` marker + `[[carved-out-but-deprecated-upstream]]` link. Do not skip without documentation. |
| Test count balloons beyond available capacity per session | Each wave is committed + tagged independently. If session boundaries get hit mid-wave, the wave-N intermediate tag is the durable checkpoint. |
| Scipy / numpy don't have a clean equivalent for a particular C++ algorithm | Port the algorithm directly in Python (custom math). Pattern: see Phase 10 `HymanFilteredCubic` (no scipy delegation). |
| Subagent socket drops mid-wave (Phase 7 pattern) | Same as before: partial-commit + carve-out for the remainder of the cluster. Document in the wave's completion doc. |
| Some C++ items are tests, not production code (e.g., `ql/test/*`) | Out of scope. Phase 11 ports the library, not the C++ test suite. Our test parity vs `jquantlib-final` is the proxy for "how much library coverage we have." |

## Stopping criteria

Phase 11 is **complete** when:

1. Every `.hpp` in `ql/**/*.hpp` (excluding `all.hpp` aggregators) has a corresponding Python class or module, or a formal carve-out documented in `docs/carve-outs.md` with an explicit `cannot-be-ported-because-X` rationale.
2. `find ql -name '*.hpp' | check_corresponding_python_module` (the audit script) reports zero unflagged gaps.
3. Triad passes: pytest + pyright + ruff all clean.
4. Test-parity vs `jquantlib-final` ≥ 95% (target 100% conceptually; ≥ 95% acknowledges that some C++ classes have no JQuantLib counterpart and vice versa).

After all conditions: tag `pquantlib-100-complete` alongside `pquantlib-phase11-complete`.

## Decision log

| Decision | Rationale |
|---|---|
| **Single Phase 11 with 12 internal waves** (vs Phases 11-22 each treated as separate phases) | User explicitly asked for "Phase 11 ... everything ... nothing left out". Each wave still gets its own intermediate tag for durability. |
| **Wave order: specialty (W1-W2) → experimental (W3-W8) → marketmodels (W9-W11) → final closure (W12)** | Specialty depends only on existing core; experimental is grab-bag (any order works); marketmodels is the most self-contained heavy lift; W12 needs all of the above to do the gap-fill audit correctly. |
| **Port the tooling-boundary items (GARCH, YangZhang, Parkinson, ...) in W12** | The user explicitly asked for "no carve-outs left." This overrides the prior design decision (Phase 5/6) to delegate these to numpy/arch. Documented carve-outs become real ports. |
| **Multi-wave structure preserves run-to-completion semantics** | Each wave is independently dispatched. If session boundaries hit mid-phase, intermediate `pquantlib-phase11-wN-complete` tags are durable. The full `pquantlib-phase11-complete` tag only lands after all 12 waves close. |
| **Final audit script lives in `migration-harness/check-coverage.sh`** | Programmatic gap detection: walks `ql/**/*.hpp`, applies the consolidation rules (Thirty360 → 1 module; X-fwd headers → skip; all.hpp → skip), and emits a CSV of unported headers. Run at the end of W11; re-run after W12 to confirm zero gaps. |

## Operational rules (binding for all 12 waves)

Same as the rest of the project:

- Direct-to-main per wave (fast-forward only).
- `-s` sign-off, NO `Co-authored-by`.
- Every commit passes the triad.
- C++ probe + JSON reference per cluster.
- Tolerance tiers EXACT / TIGHT / LOOSE; per-test inline justification.
- Wave intermediate tag pushed immediately after the wave's last cluster merge + doc-sweep micro-update.
- Doc-sweep (per [[feedback-phase-doc-sweep]]) executed at end of each wave; the full 8-step sweep happens at Phase-11 closure.

## A6 pause behavior

Per [[feedback-phase-runtocompletion]] and the user's run-to-completion directive for Phase 11, A6 is **waived between waves** for the duration of Phase 11. Other pause triggers (A1-A4) remain active.

If a wave reveals an A2 (tolerance needs >1e-8) or A3 (C++ appears wrong) or A4 (new dep) condition, pause and report.

## Plan + executable tasks

See [`phase11-plan.md`](phase11-plan.md).

# Phase 11 — Complete C++ QuantLib v1.42.1 closure (executable plan)

**Goal:** Land every remaining ~500 classes (~670 .hpp files) from the C++ v1.42.1 surface, behind the final tag `pquantlib-phase11-complete` (and `pquantlib-100-complete`). Zero unflagged carve-outs.

**Predecessor:** `pquantlib-phase10-complete` @ `d3746e4` — 2652/0/0.

**Date:** 2026-05-29.

**Topology:** 12 sequential waves. Each wave = 3-5 parallel clusters (standard phase scope). Intermediate tag per wave.

---

## Wave loop (template — applied per wave Wn)

```
Task Wn.0  Spawn 3-5 worktrees off the current main tip
Task Wn.1  Dispatch 3-5 parallel cluster subagents
Task Wn.2  Merge clusters + resolve CMakeLists conflicts
Task Wn.3  Verify triad (pytest + pyright + ruff) after each merge
Task Wn.4  Push main + tag pquantlib-phase11-wN-complete
Task Wn.5  Micro doc-sweep: extend phase11-completion.md with wave N's contribution
```

Wave-level commit cadence: **3-5 feature commits per cluster + 1 merge commit per cluster + 1 doc-sweep micro-commit per wave**.

---

## W1 — Specialty model completion

**Clusters:**
- **L11-W1-A** Markov + Gaussian1d engines (~10 classes): `MarkovFunctional` + `Gaussian1dGsrProcess` + `Gaussian1dCapFloorEngine` + `Gaussian1dFloatFloatSwaptionEngine` + `Gaussian1dNonStandardSwaptionEngine`.
- **L11-W1-B** Bates family completion (~6 classes): `BatesDetJumpModel` + `BatesDoubleExpModel` + `BatesDoubleExpDetJumpModel` + engines.
- **L11-W1-C** Heston SLV + GjrGarch + time-dependent Heston (~9 classes): `HestonSlvFdmModel` + `HestonSlvMcModel` + `GjrGarchModel` + `PiecewiseTimeDependentHestonModel` + 4 engines.

Target +60 tests.

Intermediate tag: `pquantlib-phase11-w1-complete`.

---

## W2 — ZABR + smile + bootstrap + ergonomic follow-ups

**Clusters:**
- **L11-W2-A pilot** ZABR PDE FD engine (~3 classes): `FdmZabrOp` + `FdmZabrSolver` + `ZabrFdmEngine`. (Pilot because W2-B's ZABR FD modes consume this.)
- **L11-W2-B** ZABR fitter + cube generalization (~5 classes): `ZabrInterpolation` + `ZabrInterpolatedSmileSection` + ZABR `LocalVolatility`/`FullFd`/`ProjectedHedge` modes + `XabrSwaptionVolatilityCube<Model>` generalization.
- **L11-W2-C** Bootstrap + smile + CMS small follow-ups (~7 classes): `ConvexMonotoneInterpolation` + `BootstrapError` + `LocalBootstrap` + `CmsMarket` + `CmsMarketCalibration` + `AbcdCalibration` + KahaleSmileSection deep iter.

Target +30 tests.

Intermediate tag: `pquantlib-phase11-w2-complete`.

---

## W3 — `experimental/credit/*`

**Clusters:**
- **L11-W3-A pilot** Loss-distribution + recovery models (~10 classes): the math foundation for the CDO engines.
- **L11-W3-B** Basket CDS + NthToDefault (~10 classes).
- **L11-W3-C** Synthetic CDO + tranche + CDO engines (~15 classes).
- **L11-W3-D** CDS option + RandomDefaultModel + DefaultEvent infrastructure (~10 classes).

Target +120 tests.

Intermediate tag: `pquantlib-phase11-w3-complete`.

---

## W4 — `experimental/exoticoptions/*` + `experimental/barrieroption/*` + `experimental/varianceoption/*`

**Clusters:**
- **L11-W4-A** Multi-asset exotics (~12 classes): `HimalayaOption` + `EverestOption` + `PagodaOption` + `TwoAssetBarrierOption` + `TwoAssetCorrelationOption` + MC engines.
- **L11-W4-B** Compound + Chooser + extensible + spread (~10 classes): `CompoundOption` + `ChooserOption` + `Holder/WriterExtensibleOption` + `KirkSpreadOptionEngine` + `AnalyticPdfHestonEngine`.
- **L11-W4-C** Barrier specialties + variance options (~13 classes): `PartialBarrierOption` + `SoftBarrierOption` + `PartialTimeBarrierOption` + `VarianceOption` + `IntegralHestonVarianceOptionEngine` + `AnalyticBinaryBarrier` + `VannaVolgaBarrier` engines.

Target +90 tests.

Intermediate tag: `pquantlib-phase11-w4-complete`.

---

## W5 — `experimental/finitedifferences/*`

**Clusters:**
- **L11-W5-A pilot** Multi-asset FDM operators (~8 classes): `Fdm2dBlackScholesOp` + `FdmHestonHullWhiteOp` + `FdmDupire1dOp` + `FdmExtOUJumpOp` + `FdmDividendHandler` + `AdaptiveFdScheme`.
- **L11-W5-B** Multi-asset FDM engines (~9 classes): `Fdm2dBlackScholesVanillaEngine` + `FdmHestonHullWhiteVanillaEngine` + `FdmDupire1dEngine` + various solvers.
- **L11-W5-C** HestonSLV FDM + Bates FDM + LocalVolatility FDM engines (~8 classes): wires up to W1 HestonSlvFdmModel + closes the BatesFDM carve-out from L4.

Target +80 tests.

Intermediate tag: `pquantlib-phase11-w5-complete`.

---

## W6 — `experimental/volatility/*` + `experimental/math/*`

**Clusters:**
- **L11-W6-A** NoArbSABR + SVI + extended vol surfaces (~12 classes): `NoArbSabrInterpolation` + `NoArbSabrSmileSection` + `SviInterpolation` + `SviVolatilityCube` + `SabrVolSurface` + `EquityFxVolatilityRatio`.
- **L11-W6-B** Experimental math helpers (~12 classes): `LaplaceInterpolation` + `MultidimensionalIntegral` + `NonLinearLeastSquare` + `BoundaryConstraint` + `LatentModel` + extended copulas (T, Frank, Gumbel).
- **L11-W6-C** Heuristic optimizers (~12 classes): `ParticleSwarmOptimization` + `DifferentialEvolution` + `FireflyAlgorithm` + companion infrastructure.

Target +100 tests.

Intermediate tag: `pquantlib-phase11-w6-complete`.

---

## W7 — `experimental/processes/*` + `experimental/commodities/*` + `experimental/inflation/*` + `experimental/variancegamma/*`

**Clusters:**
- **L11-W7-A** VG + Kluge processes (~10 classes): `VarianceGammaProcess` + `KlugeExtOUProcess` + `SquaredOuProcess` + `VarianceGammaModel` + `FFTVarianceGammaEngine` + `MCEuropeanVarianceGammaEngine`.
- **L11-W7-B** Commodity infrastructure (~17 classes): `CommodityIndex` + `CommodityCurve` + `CommodityType` + `UnitOfMeasure` + `EnergyCommodity` + `EnergyFuture` + `EnergySwap` + 4 engines.
- **L11-W7-C** Experimental inflation (~12 classes): multi-curve inflation, cross-currency inflation swap, specialty cap/floor engines beyond L7-D.

Target +100 tests.

Intermediate tag: `pquantlib-phase11-w7-complete`.

---

## W8 — Long-tail `experimental/*`

**Clusters:**
- **L11-W8-A** Coupons + swaptions specialty (~12 classes): `IsdaIborCoupon` + `IborCmsSpreadCoupon` + specialty swap variants (`TenorBasisSwap`, `LiborOisSpread`, `NonStandardForwardSwap`).
- **L11-W8-B** Callable + cat + MC basket framework (~20 classes): `CallableBond` + 3 engines + `CatBond` + `EventSet` + generic basket MC pricer + path pricers.
- **L11-W8-C** Asian / FX / forward / basis / averageois (~16 classes): continuous-Asian extras + `BlackDeltaCalculator` + `FxBlackVolSurface` + `NonStandardForwardSwap` + Average-OIS rate helpers + engine.
- **L11-W8-D** Risk + lattices + experimental models / shortrate / termstructures (~14 classes): Sensitivity helpers + 2D lattices + `HwSwaptionEngine` + Polyakov tree + specialty TS extensions.

Target +130 tests.

Intermediate tag: `pquantlib-phase11-w8-complete`.

---

## W9 — `marketmodels/products/*` + `marketmodels/callability/*`

**Clusters:**
- **L11-W9-A pilot** `MarketModelMultiProduct` abstract + 5 base products (~10 classes): `OneStepForwards` + `OneStepOptionlets` + `MultiStepNothing` + `MultiStepOptionlets` + `MultiStepSwap`.
- **L11-W9-B** Coterminal swaps + swaptions products (~10 classes): `MultiStepCoterminalSwaps` + `MultiStepCoterminalSwaptions` + `MultiStepCoinitialSwaps` + `MultiStepSwaption`.
- **L11-W9-C** Path-dependent products (~10 classes): `MultiStepRatchet` + `MultiStepTarn` + `MultiStepInverseFloater` + `MultiStepPeriodCapletSwaptions` + `MultiStepPathwise{InverseFloater,Wrapper}`.
- **L11-W9-D** Callability framework (~15 classes): `BermudanSwaption`-LMM + `MarketModelBasisSystem` + `MarketModelNodeDataProvider` + `LongstaffSchwartzExerciseStrategy` LMM + `UpperBoundEngine` + `LowerBoundEngine` + `CallabilityHelpers`.

Target +130 tests.

Intermediate tag: `pquantlib-phase11-w9-complete`.

---

## W10 — `marketmodels/models/*` + `marketmodels/evolvers/*`

**Clusters:**
- **L11-W10-A pilot** `MarketModel` abstract + 2 core concretes (~6 classes): `MarketModel` abstract + `LMMNormal` + `LMMAB`.
- **L11-W10-B** Remaining LMM models (~13 classes): `LMMABCD` + `PiecewiseConstantAbcdVariance` + 5 more LMM model variants + LMM volatility helpers.
- **L11-W10-C** Forward-rate evolvers (~8 classes): `MarketModelEvolver` abstract + `LogNormalForwardRatePcEvolver` + `LogNormalForwardRateIpcEvolver` + `LogNormalForwardRateBalland` + `LogNormalForwardRateEulerEvolver`.
- **L11-W10-D** Swap-rate + normal evolvers + constrained (~6 classes): `LogNormalCotSwapRateEvolver` + `NormalFwdRatePcEvolver` + constrained evolver variants.

Target +90 tests.

Intermediate tag: `pquantlib-phase11-w10-complete`.

---

## W11 — `marketmodels/{driftcomputation,correlations,curvestates,pathwisegreeks,browniangenerators}/*` + root

**Clusters:**
- **L11-W11-A** Drift computation + correlations (~9 classes): `LmmDriftCalculator` + `CmSwapDriftCalculator` + `ExponentialForwardCorrelation` + `CotSwapFromFwdCorrelation` + `TimeHomogeneousForwardCorrelation` + `PiecewiseConstantCorrelation`.
- **L11-W11-B** Curve states + pathwise greeks (~9 classes): `LMMCurveState` + `CMSwapCurveState` + `CoterminalSwapCurveState` + pathwise Jacobians + bump-instrument Jacobian.
- **L11-W11-C** Brownian generators + AccountingEngine (~9 classes): `BrownianGenerator` abstract + `SobolBrownianGenerator` + `MTBrownianGenerator` + `AccountingEngine` + `PathwiseAccountingEngine`.
- **L11-W11-D** Root + mappings + reduction utilities (~15 classes): `MarketModel` (top-level) + `MarketModelMultiProduct` + `MarketModelPathwiseDiscounter` + `ForwardForwardMappings` + `SwapForwardMappings` + `UtilityForReductionAlgorithms` + `Discounter` + `EvolutionDescription` + `CurveState` abstract + `Evolver` abstract.

Target +120 tests.

Intermediate tag: `pquantlib-phase11-w11-complete`.

---

## W12 — Tooling-boundary closure + final audit + audit-driven gap fills

**Pre-W12 setup:** write the `migration-harness/check-coverage.sh` audit script. Walks `ql/**/*.hpp`, applies known consolidation rules, emits CSV of unported headers.

**Clusters:**
- **L11-W12-A** Historical vol estimators (~8 classes): `ConstantEstimator` + `Parkinson` + `ParkinsonExtended` + `GarmanKlassSimpleSigma` + `GarmanKlassEstimator` + `SimpleLocalEstimator` + `YangZhang`. **Reverses the L1-D / L5-A tooling-boundary carve-out** — implements the C++ math directly in Python (no scipy/arch delegation).
- **L11-W12-B** GARCH model + historical synthetic forward curve (~5 classes): `GARCHModel` + `HistoricalSyntheticInterestRateForwardCurve` + supporting infrastructure.
- **L11-W12-C** Audit-driven gap fills (~17 classes — exact count emerges from W12 audit pass): port whatever the audit script flags. Each port follows the standard cluster commit-batching discipline.
- **L11-W12-D** Final closure docs + carve-out re-baseline: `docs/carve-outs.md` rewritten — every entry either marked CLOSED-by-PhaseN with a cluster reference, or marked permanently-not-in-scope with explicit rationale (e.g., "C++ test-suite items, not library code").

Target +50 tests + 2 doc commits.

**Final tags (pushed together):**
- `pquantlib-phase11-w12-complete`
- `pquantlib-phase11-complete`
- **`pquantlib-100-complete`**

---

## Total scope summary

| Wave | Target tests | Cumulative pytest |
|---|---|---|
| baseline (post-Phase 10) | — | 2652 |
| W1 | +60 | 2712 |
| W2 | +30 | 2742 |
| W3 | +120 | 2862 |
| W4 | +90 | 2952 |
| W5 | +80 | 3032 |
| W6 | +100 | 3132 |
| W7 | +100 | 3232 |
| W8 | +130 | 3362 |
| W9 | +130 | 3492 |
| W10 | +90 | 3582 |
| W11 | +120 | 3702 |
| W12 | +50 | **3752** |

Target Phase 11 closure: **~3752 / 0 / 0** pytest, pyright clean, ruff clean.

vs `jquantlib-final` 3610 — this would put PQuantLib ahead of JQuantLib's final state on raw test count (because PQuantLib's experimental/* coverage exceeds JQuantLib's, which never got to experimental/*).

## Per-wave doc-sweep cadence

After each wave Wn:
1. Append wave-N contribution table row to `phase11-completion.md` (incrementally building the doc across all 12 waves).
2. Update `CLAUDE.md` headline state to reflect the new test count + `pquantlib-phase11-wN-complete` tag.
3. Update `README.md` badges to reflect the new tag + test count.
4. Update `memory/phase_status.md` to reflect the new wave.

**Full 8-step sweep happens at Phase-11 closure** (post-W12):
- `phase11-design.md` status → CLOSED.
- `phase11-completion.md` finalized.
- `CLAUDE.md` headline state + L11 bullet.
- `README.md` badges + new migration-status table row.
- `docs/migration/README.md` Phase 11 section.
- `docs/carve-outs.md` rebaseline (every entry CLOSED or permanently-not-in-scope).
- `memory/phase_status.md` + `MEMORY.md` updated.
- New tag `pquantlib-100-complete` referenced everywhere.

## Wall-clock + cost estimate

| Component | Estimate |
|---|---|
| Wave agent time (12 waves × ~75 min) | ~15 hours |
| Orchestrator merge + triad verification (12 waves × ~15 min) | ~3 hours |
| Inter-wave doc-sweep micro-commits (12 × ~5 min) | ~1 hour |
| Final 8-step doc-sweep + audit script run + final tag | ~1 hour |
| **Total estimated wall-clock** | **~20 hours of orchestrated work** |
| **Estimated session count** | 2-4 sessions depending on conversation limits |

## Stopping criteria recap

Phase 11 is complete when:

1. Every class-bearing `.hpp` in `ql/` has a Python counterpart **or** a formally documented permanent carve-out (Item-not-in-C++-library, test-suite item, or fundamentally non-portable).
2. `migration-harness/check-coverage.sh` reports zero unflagged gaps.
3. Triad clean.
4. Test count > 3600 (matching/exceeding `jquantlib-final`).
5. `docs/carve-outs.md` rewritten so every entry is either CLOSED-by-PhaseN or permanently-not-in-scope.
6. Three tags pushed: `pquantlib-phase11-w12-complete`, `pquantlib-phase11-complete`, `pquantlib-100-complete`.

---

## Execution control

Phase 11 is **NOT auto-executed** by writing this plan. The plan is a contract.

**To execute:** user says "execute Phase 11" or "start Phase 11 wave 1". I then dispatch W1's 3 clusters in parallel and follow the wave loop above.

**Between waves:** standard cadence — push tag, micro doc-sweep, then immediately spawn W(n+1) worktrees and dispatch the next wave. No A6 pause per the run-to-completion directive.

**At session boundaries:** intermediate `pquantlib-phase11-wN-complete` tag is the durable resume point. Next session reads `phase11-completion.md` to know which waves are done.

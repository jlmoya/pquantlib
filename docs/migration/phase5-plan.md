# Phase 5 — L5 tree/lattice + MC + FD + exotic instruments (executable plan)

**Status:** **closed** — all 5 clusters landed on `main`; tag `pquantlib-phase5-complete` @ `d322fca` on 2026-05-28. **1883/0/0** tests. See [`phase5-completion.md`](phase5-completion.md).

**Goal:** Land Phase 5's ~50 must-port classes on `main`, behind tag `pquantlib-phase5-complete`. Closes Phase 1 carry-overs (Sobol + GammaFunction + Akima cubic), Phase 3 `VanillaOption.implied_volatility`, Phase 4 BlackKarasinski + TreeSwaptionEngine + TreeCapFloorEngine.

**Predecessor:** `pquantlib-phase4-complete` @ `fab5a0d` — 1544/0/0, pyright + ruff clean.

**Date:** 2026-05-28.

---

## Task 0 — Spawn pilot worktree

```bash
git worktree add -b phase5-A ../pquantlib-phase5-A main
cd ../pquantlib-phase5-A && uv sync
```

---

## L5-A pilot (sequential, ~10 classes)

### Stages
- **S0** Probe scaffolding (Sobol seq + Gamma values + Akima cubic + lattice base behavior).
- **S1** Phase 1 carry-overs: `SobolRsg` + `Burley2020SobolRsg` (scipy.stats.qmc.Sobol wrapped) + `GammaFunction` (port C++ Lanczos approximation; replace `math.lgamma` delegate in Factorial) + `AkimaCubicInterpolation`.
- **S2** Tree base classes (`Tree[T]`, `Lattice`).
- **S3** DiscretizedAsset hierarchy (DiscretizedAsset / DiscretizedOption / DiscretizedDiscountBond).
- **S4** Cross-cluster Protocols (DiscretizedAssetProtocol / LatticeProtocol / PathGeneratorProtocol).

Target: +50-65 tests.

---

## L5-B/C/D/E parallel (each ~10 classes)

### L5-B: tree/lattice engines + BlackKarasinski
TrinomialTree + BlackScholesLattice + TreeLattice1D + TreeLattice2D + DiscretizedSwap/Swaption/CapFloor + **TreeSwaptionEngine** + **TreeCapFloorEngine** + **BlackKarasinski** + `ShortRateModel.tree(grid)` impls.

### L5-C: MC framework
Path + PathGenerator + MultiPath + BrownianBridge + McSimulation + MCEuropeanEngine + MCDiscreteArithmeticAveragePriceEngine + PathPricer.

### L5-D: FD framework + FdBlackScholesVanillaEngine
FdmMesher + UniformGridMesher + FdmBlackScholesMesher + FdmLinearOp + FdmBlackScholesOp + FdmStepConditionComposite + FdmAmericanStepCondition + FdmBackwardSolver + **FdBlackScholesVanillaEngine** + closes Phase 3 implied-vol carve-out.

### L5-E: exotic instruments + analytic engines
AsianOption (+ AverageType) + BarrierOption (+ BarrierType) + BasketOption (+ BasketType) + LookbackOption + CliquetOption + DigitalOption + analytic engines (AnalyticGeometricAveragePrice / AnalyticBarrier / AnalyticBinaryBarrier / StulzEngine / AnalyticContinuousFloatingLookback).

Target each cluster: ~35-45 tests.

---

## Expected outcomes

| Cluster | Classes | Tests delta (est.) |
|---|---|---|
| L5-A pilot | ~10 | +60 |
| L5-B trees + BlackKarasinski | ~12 | +50 |
| L5-C MC framework + engines | ~10 | +40 |
| L5-D FD framework + FD vanilla engine | ~10 | +40 |
| L5-E exotic instruments + engines | ~10 | +45 |
| **Total** | **~52** | **~235 → 1779/0/0** |

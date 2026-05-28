# Phase 5 (L5 tree/lattice + MC + FD + exotic instruments) — completion

**Date closed:** 2026-05-28
**Tag:** [`pquantlib-phase5-complete`](../../README.md#migration-status) @ `d322fca`
**Predecessor:** `pquantlib-phase4-complete` @ `fab5a0d`
**Test count:** 1544 → **1883/0/0** (+339). pyright + ruff clean.
**Design spec:** [`phase5-design.md`](phase5-design.md). **Plan:** [`phase5-plan.md`](phase5-plan.md).

## Cluster contribution table

| Cluster | Mode | Commits | Tests added | Coverage |
|---|---|---|---|---|
| **L5-A pilot** | sequential, 5 stages | 5 | +70 | Phase 1 carry-overs closed (SobolRsg + Burley2020 + GammaFunction Lanczos + AkimaCubic); Tree[T] + Lattice base; DiscretizedAsset hierarchy; 3 cross-cluster Protocols |
| **L5-B** trees + lattices | parallel | 7 | +54 | BinomialTree concretes refactored + TrinomialTree + TreeLattice1D + BlackScholesLattice + DiscretizedSwap/Swaption/CapFloor + TreeSwaptionEngine + TreeCapFloorEngine + BlackKarasinski + ShortRateModel.tree() |
| **L5-C** MC framework | parallel | 5 | +63 | Path + MultiPath + BrownianBridge + PathGenerator + MultiPathGenerator + McSimulation + MCVanillaEngine + MCEuropeanEngine + MCDiscreteArithmeticAveragePriceEngine + AnalyticGeometricAsianEngine + DiscreteAveragingAsianOption |
| **L5-D** FD framework | parallel | 1 | +64 | 18 modules: layout + meshers + operators + step conditions + schemes + solver + FdBlackScholesVanillaEngine + VanillaOption.implied_volatility |
| **L5-E** exotic instruments | parallel | 8 | +97 | 6 instrument families (Asian/Barrier/Basket/Lookback/Cliquet/Digital) + 6 analytic engines (Kemna-Vorst / Reiner-Rubinstein / Stulz / Conze-Viswanathan) + 2 payoffs + BivariateCumulativeNormalDistribution |
| **(post-merge fixup)** | sequential | 1 | (subsumed) | ContinuousAveragingAsianOption added additively (preserving L5-C strict Discrete validation) |
| **Total** | | **~27** | **+339** | **~50 classes** |

## Carry-overs closed

Phase 5 closed an unprecedented number of pre-existing carve-outs:

- **Phase 1 L1-D**: SobolRsg + Burley2020SobolRsg low-discrepancy generators.
- **Phase 1 L1-B**: GammaFunction (Lanczos approximation, replaces `math.lgamma` in Factorial); BivariateCumulativeNormalDistribution (Dr78 alias + Genz-Bretz via scipy).
- **Phase 1 L1-E**: AkimaCubicInterpolation.
- **Phase 3 L3-D**: `VanillaOption.implied_volatility` via FdBlackScholesVanillaEngine + Brent solver.
- **Phase 4 L4-B**: BlackKarasinski (needed TrinomialTree).
- **Phase 4 L4-B**: `OneFactorModel.tree()` across Vasicek/HW/CIR/ECIR/BK.
- **Phase 4 L4-E**: TreeSwaptionEngine + TreeCapFloorEngine.

Five distinct phases of carry-overs are now empty for the items that L5 was designed to close.

## Parallelization wins

Phase 5 was the largest L-layer to date by class count (~50) and by tests added (+339). Wall-clock with 4 parallel subagents: ~50 min. Pattern continues to scale.

L5-D landed as a single commit despite porting 18 modules — the agent batched aggressively.

## Merge reconciliation

- **L5-C and L5-E both ported Asian instruments**. L5-C's DiscreteAveragingAsianOption has stricter validation (unseasoned overrides, fixing-date sorting, negative-sum rejection); L5-E's was simpler but added ContinuousAveragingAsianOption. Resolved by keeping L5-C's Discrete + adding L5-E's Continuous additively (`is_expired = False` carve-out preserved).
- **CMakeLists.txt**: 4 parallel cluster entries stacked.
- **L5-E subagent file leakage** into main worktree (probe.cpp + JSON) — cleaned + reverted before the L5-B merge.

## Cumulative documented divergences (L1+L2+L3+L4+L5)

New in Phase 5:

### scipy wrapping (Sobol + Gamma + bivariate normal)
- `SobolRsg` wraps `scipy.stats.qmc.Sobol`. Joe-Kuo default direction integers; alternative C++ sets (Jaeckel default / Unit / SobolLevitan / Kuo / etc.) carved out.
- `Burley2020SobolRsg` uses scipy's Matousek LMS+shift Owen scrambling (statistically equivalent to C++ Burley 2020 hash, not bit-exact).
- `GammaFunction` Lanczos approximation matches C++.
- `AkimaCubicInterpolation` uses scipy.interpolate.Akima1DInterpolator (standard reflection rule); C++ QuantLib uses a non-standard nonlinear endpoint slope formula. Tests assert exact-knot interpolation + quadratic recovery instead of per-cell match.
- `BivariateCumulativeNormalDistribution` (Dr78 + Genz-Bretz alias) via scipy.stats.multivariate_normal.cdf.

### MC framework
- `MCEuropeanEngine` uses MersenneTwister + InverseCumulativeNormal (bit-exact vs C++ PseudoRandom). Low-discrepancy MC with Sobol carved out.
- `McSimulation.calculate` renamed to `run_mc` to avoid colliding with `PricingEngine.calculate` when MCVanillaEngine multi-inherits.
- `Path` mutated in place per call (C++ `mutable Sample<Path>`).
- `BrownianBridge` not supported in MultiPathGenerator (mirrors C++ `QL_FAIL`).

### FD framework
- 1-D Black-Scholes only; multi-asset FD (Heston/G2/Bates) carved out for Phase 6.
- scipy.sparse CSR for operator matrices; manual Thomas tridiagonal sweep for splitting.
- Uniform 1-D mesher only (no Concentrating1dMesher — costs ~3e-3 abs error at xGrid=200 vs C++).
- BoundaryCondition framework carved out (uses operator-truncation instead).

### Tree/lattice
- TreeLattice2D + G2.tree() carved out (heavy multi-factor lattice; needs follow-up).
- DiscretizedSwaption uses in-place SwaptionArguments rebuild rather than fresh snapped VanillaSwap reconstruction (~1bp delta at N=100).
- Python collapses C++ TreeLattice<Impl> + TreeLattice1D<Impl> CRTP into a single TreeLattice1D.

### Exotic instruments
- L5-C and L5-E both ported Asian; resolved by additive merge (L5-C Discrete strict; L5-E Continuous new).
- `AnalyticBinaryBarrier` 8-branch table with degenerate KO/KI cases delegating to `AnalyticEuropeanEngine` for KI.
- `AnalyticContinuousFloatingLookback` Conze-Viswanathan closed-form.
- `StulzEngine` 2-asset min/max basket with put-parity construction.

## Carve-outs (deferred to Phase 6+)

### Tree/lattice
- TreeLattice2D + G2.tree() — multi-factor lattice.
- Joshi4 / AdditiveEQP / Trigeorgis tree builders.
- TFLattice variants.

### MC
- LongstaffSchwartz American MC.
- Heston / G2 / Bates / HW MC engines.
- Low-discrepancy MC (Sobol path generators).
- All exotic MC engines (MCBarrier / MCBasket / MCLookback / MCCliquet).
- Multi-asset basket PathGenerator.

### FD
- Multi-asset FD (Heston / G2 / Bates / CIR / SABR).
- Time-dependent FD operators.
- Operator-splitting (Craig-Sneyd / Hundsdorfer / TR-BDF2 / MethodOfLines).
- BoundaryCondition framework (FdmDirichletBoundary / FdmNeumannBoundary).
- Concentrating1dMesher.

### Exotic
- DoubleBarrier / PartialTimeBarrier / SoftBarrier options.
- HolderExtensibleOption / ComplexChooserOption / CompoundOption.
- 3+ asset baskets.
- Soft-barrier engines.

### Permanent (out of scope)
- MarketModels (125 files of LMM machinery).
- ZABR / SABR / XABR volatility models.
- All inflation (instruments + indexes + termstructures + engines).
- All credit (CDS + DefaultProbabilityTermStructure + engines).
- Capfloor/optionlet/swaption volatility surfaces.
- Specialty short-rate (Gaussian1d / GSR / MarkovFunctional).
- Specialty Heston (PiecewiseTimeDependentHeston / HestonSLV / GJR-GARCH / BatesDoubleExp).
- Volatility models (GARCH / GarmanKlass).

## Lessons learned

- **Subagents independently porting "the same" type produced compatible-but-different APIs**. L5-C's Asian had stricter validation than L5-E's. The additive-merge approach (preserve the strict one + add the missing class) worked cleanly without test regressions.
- **scipy wrapping continues to pay off**. Phase 5 used scipy.stats.qmc (Sobol), scipy.interpolate (Akima), scipy.stats.multivariate_normal (bivariate normal), scipy.sparse + scipy.sparse.linalg (FD). Each saved an estimated 50-500 LOC of careful numerical porting.
- **18 modules in one commit (L5-D)** is fine when they're tightly coupled. The agent batched aggressively; commit message documents the structure.
- **L4-D's preference for analytic engines paid off in L5-B**: TreeSwaptionEngine + TreeCapFloorEngine had a clean implementation path because L4 had already wired in the model-supplied `discount_bond_option` for the analytic path; the tree path just extends to `model.tree(grid)`.

## Next: Phase 6 — Python 3.14 modernization + final closure

Phase 6 priorities:
1. **Modernization sweep**: PEP 695 generics where missing, `match-case` where natural, `@dataclass(frozen=True, slots=True)` where appropriate, type aliases via `type` statement.
2. **Remaining carve-outs** that benefit the most users:
   - LongstaffSchwartz American MC (closes American option pricing without FD).
   - Heston/G2/Bates MC engines.
   - DoubleBarrier instrument family + engines.
3. **`pquantlib-final` tag** with comprehensive carve-out documentation.

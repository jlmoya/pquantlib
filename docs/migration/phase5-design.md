# Phase 5 — L5 tree/lattice + MC + FD + exotic instruments (design)

**Date:** 2026-05-28
**Status:** drafted, awaiting ack to start
**Predecessor:** `pquantlib-phase4-complete` @ `fab5a0d` — 1544/0/0, pyright + ruff clean
**Sister-project anchor:** jquantlib `phase2-L5-experimental-plan.md`
**C++ ground truth:** QuantLib v1.42.1 @ `099987f0`

## Scope decision

C++'s `ql/methods/` (147 .hpp) + `ql/experimental/` (263 .hpp) is too broad for one phase. **Phase 5 ports the most-impactful carve-outs that close vanilla pricing gaps from Phases 1–4**. L6 (test-suite parity) becomes a separate phase before modernization.

Phase 5 closes when:

1. Every must-port class is ported with C++-cross-validated tests or annotated `# C++ parity:` with a deliberate divergence note.
2. `uv run pytest` + `uv run pyright` + `uv run ruff check` clean on `main`.
3. Tag `pquantlib-phase5-complete` is pushed.

## Scope (must-port subset: ~50 classes)

### L5-A pilot — foundations + Phase 1 carry-overs (sequential, ~10 classes)

- **Phase 1 closures**:
  - `pquantlib.math.randomnumbers.sobol_rsg.SobolRsg` (low-discrepancy).
  - `pquantlib.math.randomnumbers.burley_2020_sobol_rsg.Burley2020SobolRsg`.
  - `pquantlib.math.distributions.gamma_function.GammaFunction` — replaces `math.lgamma` delegate; closes Factorial LOOSE-tier carve-out.
  - `pquantlib.math.interpolations.akima_cubic_interpolation.AkimaCubicInterpolation` — closes one of the 8+ cubic-spline carve-outs.
- **Tree/Lattice foundations**:
  - `pquantlib.methods.lattices.tree.Tree[T]` — abstract PEP 695 generic.
  - `pquantlib.methods.lattices.lattice.Lattice` — abstract.
  - `pquantlib.methods.lattices.discretized_asset.DiscretizedAsset` — abstract.
  - `pquantlib.methods.lattices.discretized_option.DiscretizedOption` — abstract intermediate.
- **Cross-cluster Protocols**: `DiscretizedAssetProtocol`, `LatticeProtocol`, `PathGeneratorProtocol`.

### L5-B — tree/lattice engines + BlackKarasinski (parallel, ~10 classes)

- `pquantlib.methods.lattices.binomial_tree.BinomialTree` (refactor from L3-D inline impl).
- `pquantlib.methods.lattices.trinomial_tree.TrinomialTree`.
- `pquantlib.methods.lattices.bsm_lattice.BlackScholesLattice`.
- `pquantlib.methods.lattices.lattice_1d.TreeLattice1D`.
- `pquantlib.methods.lattices.lattice_2d.TreeLattice2D`.
- `pquantlib.methods.lattices.discretized_swap.DiscretizedSwap`.
- `pquantlib.methods.lattices.discretized_swaption.DiscretizedSwaption`.
- `pquantlib.methods.lattices.discretized_capfloor.DiscretizedCapFloor`.
- **`pquantlib.pricingengines.swaption.tree_swaption_engine.TreeSwaptionEngine`** — closes Phase 4 carve-out.
- **`pquantlib.pricingengines.capfloor.tree_capfloor_engine.TreeCapFloorEngine`** — closes Phase 4 carve-out.
- **`pquantlib.models.shortrate.onefactor.black_karasinski.BlackKarasinski`** — closes Phase 4 carve-out (needed TrinomialTree).
- `ShortRateModel.tree(grid)` implementation across the 1F-affine models.

### L5-C — MC framework + simple MC engines (parallel, ~10 classes)

- `pquantlib.methods.montecarlo.path.Path` (single-path container; thin numpy wrapper).
- `pquantlib.methods.montecarlo.path_generator.PathGenerator[Process]`.
- `pquantlib.methods.montecarlo.multi_path.MultiPath` + `MultiPathGenerator`.
- `pquantlib.methods.montecarlo.brownian_bridge.BrownianBridge`.
- `pquantlib.methods.montecarlo.path_pricer.PathPricer`.
- `pquantlib.methods.montecarlo.mc_simulation.McSimulation`.
- `pquantlib.pricingengines.mc_simulation.MCVanillaEngine` — base.
- `pquantlib.pricingengines.vanilla.mc_european_engine.MCEuropeanEngine`.
- `pquantlib.pricingengines.asian.mc_discrete_arithmetic_average_engine.MCDiscreteArithmeticAveragePriceEngine` (simple MC Asian).
- Statistics extension: convergence statistics (uses Sobol from L5-A).

### L5-D — FD framework + FdBlackScholesVanillaEngine (parallel, ~10 classes)

- `pquantlib.methods.finitedifferences.meshers.fdm_mesher.FdmMesher` abstract.
- `pquantlib.methods.finitedifferences.meshers.uniform_grid_mesher.UniformGridMesher`.
- `pquantlib.methods.finitedifferences.meshers.fdm_black_scholes_mesher.FdmBlackScholesMesher`.
- `pquantlib.methods.finitedifferences.operators.fdm_linear_op.FdmLinearOp` abstract.
- `pquantlib.methods.finitedifferences.operators.fdm_black_scholes_op.FdmBlackScholesOp`.
- `pquantlib.methods.finitedifferences.step_conditions.fdm_step_condition_composite.FdmStepConditionComposite`.
- `pquantlib.methods.finitedifferences.step_conditions.fdm_american_step_condition.FdmAmericanStepCondition`.
- `pquantlib.methods.finitedifferences.fdm_solver_desc.FdmSolverDesc` (config DTO).
- `pquantlib.methods.finitedifferences.fdm_backward_solver.FdmBackwardSolver`.
- **`pquantlib.pricingengines.vanilla.fd_black_scholes_vanilla_engine.FdBlackScholesVanillaEngine`** — closes Phase 3 `VanillaOption.implied_volatility` carve-out.

### L5-E — Exotic instruments + analytic engines (parallel, ~10 classes)

- `pquantlib.instruments.asian_option.AsianOption` + `ContinuousAveragingAsianOption` + `DiscreteAveragingAsianOption`. `AverageType` enum (Arithmetic / Geometric).
- `pquantlib.instruments.barrier_option.BarrierOption`. `BarrierType` enum (DownIn/DownOut/UpIn/UpOut).
- `pquantlib.instruments.basket_option.BasketOption`. `BasketType` (MinBasketPayoff / MaxBasketPayoff / AverageBasketPayoff / SpreadBasketPayoff).
- `pquantlib.instruments.lookback_option.{ContinuousFloatingLookbackOption, ContinuousFixedLookbackOption}`.
- `pquantlib.instruments.cliquet_option.CliquetOption`.
- `pquantlib.instruments.digital_option.{DigitalOption, AssetOrNothing}` — leverage L3-A payoffs.
- **Analytic engines**:
  - `pquantlib.pricingengines.asian.analytic_continuous_geometric_average_price_engine.AnalyticContinuousGeometricAveragePriceEngine`.
  - `pquantlib.pricingengines.asian.analytic_discrete_geometric_average_price_engine.AnalyticDiscreteGeometricAveragePriceEngine`.
  - `pquantlib.pricingengines.barrier.analytic_barrier_engine.AnalyticBarrierEngine` (Rich-Chesney/Reiner-Rubinstein closed-form).
  - `pquantlib.pricingengines.barrier.analytic_binary_barrier_engine.AnalyticBinaryBarrierEngine`.
  - `pquantlib.pricingengines.basket.stulz_engine.StulzEngine` (2-asset basket analytic).
  - `pquantlib.pricingengines.lookback.analytic_continuous_floating_lookback_engine.AnalyticContinuousFloatingLookbackEngine`.

## Carve-outs (deferred to Phase 6 or beyond)

### Tree/lattice deferrals
- Joshi4 / AdditiveEQP / Trigeorgis tree builders.
- ConvertibleBond tree engine.
- TF-lattice variants.

### MC deferrals
- Sobol-Brownian-Bridge integration.
- Heston MC, Bates MC, G2 MC, HW MC engines.
- All exotic MC engines (MCBarrier, MCBasket, MCLookback, MCCliquet).
- LongstaffSchwartz American MC.

### FD deferrals
- 95% of the 120 finite-differences .hpp files (multi-asset FD, FdBates, FdHeston, FdG2, FdHullWhite, etc.).
- All multi-asset FD engines (rainbow, spread, basket).

### Exotic deferrals
- Quanto / forward-start option engines.
- ComplexChooserOption, HolderExtensibleOption, CompoundOption.
- Double-barrier, partial-time-barrier, soft-barrier options.
- 2D-asset basket engines beyond StulzEngine.

### Specialty (Phase 6 or later)
- ZABR/SABR/XABR volatility model family.
- All inflation (instruments + indexes + termstructures + engines).
- All credit (CDS + DefaultProbabilityTermStructure + engines).
- Capfloor/optionlet/swaption volatility surfaces (Phase 2 carve-outs).
- Specialty short-rate (Gaussian1d / GSR / MarkovFunctional).
- MarketModels (125 files of LMM).
- HestonSLV, GJR-GARCH, Bates double-exp.

## Cluster topology

Same proven pattern: 1 sequential pilot + 4 parallel via subagents.

## Per-class TDD discipline + tolerance tiers

Identical to Phases 1-4. LOOSE tier for: MC engines (sampling variance), FD engines (discretization), tree-convergence tests (N=1000 trees should match analytic to ~1e-4).

## Decision log

| Decision | Rationale |
|---|---|
| **Phase 1 carry-overs (Sobol + GammaFunction + Akima cubic) in L5-A pilot**. | These unblock L5-C MC (Sobol) and improve numerical stability across L1+L2 (GammaFunction replaces lgamma in factorial). |
| **scipy.stats.qmc.Sobol** wraps Sobol RSG (rather than re-implementing Joe-Kuo primitive polynomials). | scipy provides bit-exact Joe-Kuo direction numbers; saves substantial porting effort. |
| **Tree/Lattice as PEP 695 generic `Tree[T]`**. | Mirrors C++ `template<class T>` cleanly. |
| **BlackScholesLattice + TrinomialTree under `methods/lattices`** (not `pricingengines`). | Matches C++ `ql/methods/lattices/`. Engines live in `pricingengines`. |
| **MC engines analytic-control-variate optional** (off by default for the simple MC). | Reduces L5-C scope; can be added per-engine later. |
| **FD engine restricted to 1-D Black-Scholes vanilla**. | Multi-asset FD (Heston/G2/Bates) defers to Phase 6 — needs the full FdmMesher hierarchy. |
| **Tree-based capfloor/swaption** use L4-B's `OneFactorAffineModel.tree(grid)` to lift the model onto a TrinomialTree. | Now-buildable thanks to L4 + L5-A foundations. |
| **`AverageType` / `BarrierType` / `BasketType`** as IntEnums (continuing the L1 pattern). | |

## Plan + executable tasks

See [`phase5-plan.md`](phase5-plan.md).

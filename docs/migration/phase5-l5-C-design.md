# Phase 5 L5-C — Monte Carlo framework + simple MC engines

**Date:** 2026-05-28
**Status:** **closed** — merged via L5-C merge; tagged as part of `pquantlib-phase5-complete` @ `d322fca`. Final test delta: **+63**. 5 commits.
**Predecessor:** `pquantlib-phase5-l5-A-complete` @ `aa19340`
**Style:** lean — leans on [`phase5-design.md`](phase5-design.md).

## What landed

### MC framework (`pquantlib.methods.montecarlo.*`)
- `Path` (single-path container; thin numpy wrapper).
- `MultiPath` + `MultiPathGenerator`.
- `BrownianBridge`.
- `PathGenerator(process, time_grid, rng, brownian_bridge=False)`.
- `PathPricer` abstract.
- `McSimulation` orchestrator (calls renamed to `run_mc` to avoid collision with PricingEngine.calculate).
- `MonteCarloModel` — aggregator over (process, generator, pricer, statistics).
- Gaussian sequence stack: bit-exact `MersenneTwisterUniformRng` + `InverseCumulativeNormal` from L1-D.

### MC engines (`pquantlib.pricingengines.*`)
- `MCVanillaEngine` abstract base (subclass of `GenericEngine`).
- `MCEuropeanEngine(process, time_steps, antithetic_variate, control_variate, required_samples, seed)`.
- `MCDiscreteArithmeticAveragePriceEngine` — simple MC Asian.

### New Asian instrument + analytic engines
- `DiscreteAveragingAsianOption` + `DiscreteAveragingAsianOptionArguments` (with strict validation: unseasoned overrides, fixing-date sorting, negative-sum rejection).
- `AverageType` IntEnum at `pquantlib.instruments.average_type`.
- `AnalyticDiscreteGeometricAveragePriceAsianEngine` (Levy 1997 + `blackScholesTheta` helper).
- `AnalyticContinuousGeometricAveragePriceAsianEngine` (Kemna-Vorst).

## Documented divergences

- **Did not wrap scipy.stats.qmc.Sobol** for MC — pquantlib's L1-D `MersenneTwisterUniformRng + InverseCumulativeNormal` stack is bit-exact vs C++ `PseudoRandom`. Low-discrepancy MC carved out for a future cluster.
- **`McSimulation.calculate` renamed to `run_mc`** to avoid collision with `PricingEngine.calculate` when `MCVanillaEngine` multi-inherits.
- **`Path` mutated in place per call** (C++ `mutable Sample<Path>`); tests snapshot floats before reading next.
- **`BrownianBridge` not supported in `MultiPathGenerator`** (mirrors C++ `QL_FAIL`).
- **`DiscreteAveragingAsianOption.is_expired`** returns `False` (Settings.evaluationDate carve-out).
- **`AnalyticDiscreteGeometricAveragePriceAsianEngine.theta` / `theta_per_day`** left as `None` (blackScholesTheta helper full port carved out).

## Carve-outs (Phase 6+)

- LongstaffSchwartz American MC.
- Heston / G2 / Bates / HW MC engines.
- Low-discrepancy MC (Sobol path generators).
- All exotic MC engines (MCBarrier, MCBasket, MCLookback, MCCliquet).
- Multi-asset basket PathGenerator.
- All-past-fixings Asian constructor (needs Settings.evaluationDate).

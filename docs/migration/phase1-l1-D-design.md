# Phase 1 L1-D — RNGs + simple optimization

**Date:** 2026-05-24
**Status:** drafted
**Predecessor:** `pquantlib-phase1-l1-A-complete` @ `03d0ce8`
**C++ ground truth:** QuantLib v1.42.1 @ `099987f0`
**Style:** lean — see [`phase1-l1-A-design.md`](phase1-l1-A-design.md) for ground rules.

## Goal

Port the **deterministic-sequence-bit-exact-against-C++** random number generators (Mersenne Twister, Knuth, Lecuyer, Ranlux, Xoshiro256**, BoxMuller Gaussian over MT) and the simple optimization constraint + cost-function + EndCriteria scaffolding. Defer Sobol (deep low-discrepancy table dep) + Burley2020 + InverseCumulative bridging + Halton + Faure + LM/Simplex/BFGS to follow-up.

## Must-port (tractable subset)

### RNGs (~7)
- `RandomNumberGenerator` Protocol.
- `MersenneTwisterUniformRng` — 624-element state-array, well-known well-defined algorithm. Bit-exact required.
- `KnuthUniformRng` — Knuth's lagged-Fibonacci.
- `LecuyerUniformRng` — L'Ecuyer's combined linear congruential.
- `Ranlux64UniformRng` (or RanluxUniformRng — pick the simpler one).
- `Xoshiro256StarStarUniformRng` — modern PRNG, simple state.
- `BoxMullerGaussianRng` — Gaussian over uniform.
- `SeedGenerator` — singleton seed source.

EXACT-tier cross-validation required for sequence outputs.

### Optimization (~7)
- `Constraint` abstract + `NoConstraint`, `PositiveConstraint`, `BoundaryConstraint`.
- `CostFunction` abstract.
- `EndCriteria` (max iterations + tolerance bundle).
- `OptimizationMethod` abstract.
- `Problem` (cost + constraint + current value bundle).

## Carve-outs

- SobolRsg + Burley2020SobolRsg + the full primitive-polynomials + Joe-Kuo init tables (deep low-discrepancy).
- HaltonRsg, FaureRsg.
- InverseCumulativeRng / InverseCumulativeRsg (depend on inverse cumulatives in L1-B).
- LowDiscrepancy / GenericLowDiscrepancy / GenericPseudoRandom.
- LevenbergMarquardt, Bfgs, ConjugateGradient, Simplex, SimulatedAnnealing, DifferentialEvolution — defers.
- LineSearch + subclasses.

## Approach

Single mega-probe `math/l1D_probe.cpp` emitting first-N-values for each RNG seeded with a fixed seed. Optimization scaffolding has trivial behavioral tests (no probe needed).

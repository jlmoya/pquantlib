# Phase 3 L3-D — Equity options + processes

**Date:** 2026-05-27
**Status:** **closed** — merged into `main` via `f08e0dc merge: L3-D`; tagged as part of `pquantlib-phase3-complete` @ `aacc2c2`. Final test delta: **+97** (vs +43 target). 4 commits.
**Predecessor:** `pquantlib-phase3-l3-A-complete` @ `e72bcdf`
**C++ ground truth:** QuantLib v1.42.1 @ `099987f0`
**Style:** lean — leans on [`phase3-design.md`](phase3-design.md) for ground rules.

## What landed

### Stochastic processes (under `pquantlib.processes.*`)
- `StochasticProcess` (abstract, multi-D).
- `StochasticProcess1D` (abstract, 1-D).
- `EulerDiscretization` (drift/diffusion stepper; uses `typing.overload` to satisfy both multi-D + 1-D abstract bases).
- `GeneralizedBlackScholesProcess` (risk-free + dividend + Black-vol curves).
- `BlackScholesProcess` (no dividends).
- `BlackProcess` (no rates — Black 76).
- `BlackScholesMertonProcess` (full BSM).
- `StochasticProcessDiscretization` + `StochasticProcess1DDiscretization` (abstract bases for discretizations).

### Instruments
- `pquantlib.instruments.vanilla_option.VanillaOption`.
- `pquantlib.instruments.european_option.EuropeanOption` (forced `EuropeanExercise`).

### Engines
- `pquantlib.pricingengines.vanilla.analytic_european_engine.AnalyticEuropeanEngine` (closed-form via `black_formula` + Greeks: delta/gamma/vega/theta/rho).
- `pquantlib.pricingengines.vanilla.binomial_engine.BinomialVanillaEngine` (parameterized by `TreeBuilder` enum: CRR / JarrowRudd / Tian / LeisenReimer).
- `pquantlib.pricingengines.vanilla.black_calculator.BlackCalculator` (helper — replaces C++ Visitor-pattern `BlackCalculator::Calculator`).

## Documented divergences

- **`StochasticProcess` inherits `Observable` only** (not `Observer` — Python's Observer is a structural Protocol satisfied by providing `update()`).
- **`EulerDiscretization`** uses `typing.overload` to satisfy both multi-D and 1-D abstract bases with one body. C++ achieves via multiple-inheritance + name-overloading.
- **`BlackCalculator` replaces the C++ Visitor pattern** (`BlackCalculator::Calculator`) with `isinstance` dispatch.
- **`BinomialVanillaEngine` collapses C++ `template<class T>`** into a single class parameterised by `TreeBuilder` enum. Bypasses the entire `Tree` / `Lattice` / `DiscretizedAsset` class hierarchy with direct numpy-array backward induction.
- **`GeneralizedBlackScholesProcess.evolve_1d`** bypasses `StochasticProcess1D.evolve_1d` default in the Euler-fallback branch (which routes through `expectation_1d` → "not implemented"); calls discretization drift directly to match C++ semantics.
- **1-D scalar overloads renamed** `drift_1d` / `diffusion_1d` / `expectation_1d` / `variance_1d` / `std_deviation_1d` / `evolve_1d` to coexist with multi-D `drift(t, NDArray) -> NDArray` signatures.

## Carve-outs

- Entire lattice class hierarchy: `Tree`, `BlackScholesLattice`, `DiscretizedAsset`, `DiscretizedVanillaOption`, `TimeGrid` plumbing — Phase 4.
- Joshi4 / AdditiveEQP / Trigeorgis trees — Phase 4.
- `VanillaOption.implied_volatility` — depends on `FdBlackScholesVanillaEngine`, Phase 4.
- `LocalVolSurface` full Dupire with rates/dividends — needs L4 wiring.
- vega/rho/dividend_rho for the binomial engine — Phase 4 (need additional propagation through the tree).
- All Heston / Bates / GJR-GARCH / Hull-White / G2 / CEV / SABR engines — Phase 4 (model-coupled).
- MC + FD engines — Phase 4.

# Phase 5 L5-D — Finite-difference framework + FdBlackScholesVanillaEngine

**Date:** 2026-05-28
**Status:** **closed** — merged via `e767108 merge: L5-D`; tagged as part of `pquantlib-phase5-complete` @ `d322fca`. Final test delta: **+64**. 1 commit (aggressive batching).
**Predecessor:** `pquantlib-phase5-l5-A-complete` @ `aa19340`
**Style:** lean — leans on [`phase5-design.md`](phase5-design.md).

## What landed (18 modules)

### Layout + meshers
- `FdmLinearOpLayout(dim)` — multi-D index layout.
- `FdmMesher` abstract + `Fdm1dMesher` abstract.
- `Uniform1dMesher` + `UniformGridMesher` (multi-D).
- `FdmBlackScholesMesher` — concrete 1-D log-spot mesh anchored at strike.
- `FdmMesherComposite`.

### Operators
- `TripleBandLinearOp` — banded matrix base (scipy.sparse CSR storage).
- `FirstDerivativeOp`, `SecondDerivativeOp` — central-difference stencils.
- `FdmBlackScholesOp` — 1-D BSM operator: -0.5σ²S²∂²/∂S² - (r-q)S∂/∂S + r·.

### Schemes + step conditions + solver
- `ExplicitEulerScheme`, `ImplicitEulerScheme`, `CrankNicolsonScheme`.
- `FdmSchemeDesc` factory.
- `FdmAmericanStepCondition` (early-exercise barrier).
- `FdmStepConditionComposite`.
- `FdmSolverDesc` (config DTO).
- `FdmBackwardSolver` (back-propagates from maturity to t=0).

### Engine (closes Phase 3 carve-out)
- **`FdBlackScholesVanillaEngine(process, t_grid=100, x_grid=100, damping_steps=0, scheme=CrankNicolson)`** — supports European + American.
- **`VanillaOption.implied_volatility(target_price, process, accuracy, max_evaluations, min_vol, max_vol)`** — Brent solver dispatching to AnalyticEuropean (European) or FdBlackScholesVanillaEngine (American/Bermudan). **Closes Phase 3 L3-D carve-out.**

## Documented divergences

- **scipy.sparse CSR for operator matrices**; `apply` uses numpy fancy-indexing for speed.
- **`solve_splitting` Thomas tridiagonal sweep** manually implemented (1-D reverseIndex is identity, so C++ permutation collapses).
- **`BlackConstantVol(volatility=Quote)`** makes the implied-vol clone-and-reprice loop trivial.
- **Uniform-mesh only** — Concentrating1dMesher carved out (~3e-3 abs error at xGrid=200 vs C++).
- **No BoundaryCondition framework** — uses operator-truncation (FdmDirichletBoundary/FdmNeumannBoundary carved out).
- **No FD grid-Greek extraction** — Greeks come from AnalyticEuropean for European; American Greeks deferred.

## Carve-outs (Phase 6+)

- Multi-asset FD (Heston / G2 / Bates / CIR / SABR).
- Multi-D operator splittings (Craig-Sneyd / Hundsdorfer / TR-BDF2 / MethodOfLines).
- Time-dependent operators (FdmTimeDepBlackScholesOp).
- Concentrating1dMesher.
- Local-vol + quanto branches.
- Dividend handling.
- Iterative BiCGstab / GMRES solvers.

## Tolerance ladder

- EXACT: integer index ops.
- TIGHT: stencil coefficients (algebraic).
- LOOSE 5e-3: FD-vs-analytic European convergence at xGrid=200.
- LOOSE 5e-2: American Put vs C++ reference (reflects uniform-mesh carve-out).

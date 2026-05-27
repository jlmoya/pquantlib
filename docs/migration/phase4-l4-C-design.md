# Phase 4 L4-C — Heston + Bates equity stochastic-volatility

**Date:** 2026-05-27
**Status:** **closed** — merged via `fd042e1 merge: L4-C`; tagged as part of `pquantlib-phase4-complete` @ `fab5a0d`. Final test delta: **+53**. 3 commits.
**Predecessor:** `pquantlib-phase4-l4-A-complete` @ `657b707`
**Style:** lean — leans on [`phase4-design.md`](phase4-design.md).

## What landed

### Processes
- `pquantlib.processes.heston_process.HestonProcess` — 2-D (S, V), 2-dim noise. Multi-D via `StochasticProcess` (L3-D).
- `pquantlib.processes.bates_process.BatesProcess` — HestonProcess + Merton jumps.

### Models
- `pquantlib.models.equity.heston_model.HestonModel(process)`.
- `pquantlib.models.equity.heston_model_helper.HestonModelHelper` — `BlackCalibrationHelper` for vanilla European options under Heston.
- `pquantlib.models.equity.bates_model.BatesModel(process)`.

### Engines
- `pquantlib.pricingengines.vanilla.analytic_heston_engine.AnalyticHestonEngine(model, integration_order=144)` — Gatheral characteristic function + `scipy.integrate.quad`.

## Documented divergences

- **scipy.integrate.quad over (0, +inf) replaces 144-pt Gauss-Laguerre** — LOOSE tier (1e-8 abs/rel) on NPV; put-call parity still holds TIGHT (algebraic).
- **Only Gatheral ComplexLogFormula** ported. BranchCorrection / AndersenPiterbarg / AngledContour / OptimalCV control-variate forms + AP_Helper / OptimalAlpha classes deferred to Phase 5.
- **HestonProcess.drift sidesteps L2 forward_rate(t1==t2)** by using a 1e-4 finite window (matches the L3-D GBSM workaround) — introduces ~1e-13 noise in drift_s, matching the C++ probe.
- **HestonProcess.Discretization enum dropped**; FullTruncation semantics only.
- **HestonProcess.pdf / Broadie-Kaya exact-sampling CF dropped** (no L4-C path exercises them; MC out of scope).
- **BatesProcess.evolve override + CumulativeNormalDistribution / InverseCumulativePoisson members dropped** (MC step out of scope).
- **BatesModel.generate_arguments uses a first-pass guard** so HestonModel.__init__ can complete with a plain HestonProcess before the jump slots are populated.
- **HestonModelHelper takes a single Quote-typed s0** (vs C++'s two overloads).

## Carve-outs

- **BatesEngine** (jump-aware analytic) — `add_on_term` hook is in place; calibrating with a BatesModel against the Heston engine simply reduces to Heston. Will land in L5.
- **BatesDetJumpModel, BatesDoubleExpModel, BatesDoubleExpDetJumpModel**.
- **PiecewiseTimeDependentHestonModel, HestonSLVFDMModel, HestonSLVMCModel, GJRGARCHModel**.
- **HestonProcess discretizations beyond FullTruncation** (BroadieKaya exact sampling) — MC-only.

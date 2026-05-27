# Phase 4 — L4 models (design)

**Date:** 2026-05-27
**Status:** drafted, awaiting ack to start
**Predecessor:** `pquantlib-phase3-complete` @ `aacc2c2` — 1284/0/0, pyright + ruff clean
**Sister-project anchor:** jquantlib `phase2-L4-models-plan.md`
**C++ ground truth:** QuantLib v1.42.1 @ `099987f0`

## Goal

Port the L4 layer: **financial models** (short-rate + equity stochastic-vol) + their **calibration machinery** + **analytic pricing engines** that use them. L4 builds on L1 math (optimizers) + L2 termstructures (yield curves, vol surfaces) + L3 instruments (Swaption, CapFloor, VanillaOption — instruments already ported; engines were carved out).

Phase 4 closes when:

1. Every must-port L4 class (listed below) is ported with C++-cross-validated tests or annotated `# C++ parity:` with a deliberate divergence note.
2. `uv run pytest` + `uv run pyright` + `uv run ruff check` clean on `main`.
3. Tag `pquantlib-phase4-complete` is pushed.
4. Closure doc `phase4-completion.md` lists cumulative divergences, carve-outs, lessons learned.

## Scope (must-port subset)

**Estimated total: ~40 classes.** L4 leans heavily on closed-form analytic engines (avoiding the full lattice/tree/MC machinery deferred from Phase 3).

### L4-A pilot — foundations + Phase 1 optimizer carry-overs (sequential, ~10 classes)

- **`pquantlib.math.optimization.levenberg_marquardt.LevenbergMarquardt`** — closes Phase 1 carry-over. Wraps `scipy.optimize.least_squares(method='lm')`.
- **`pquantlib.math.optimization.simplex.Simplex`** — closes Phase 1 carry-over. Wraps `scipy.optimize.minimize(method='Nelder-Mead')`.
- `pquantlib.models.parameter.Parameter` — model-parameter abstraction. Holds value(s) + Constraint + ParameterTransformation.
- `pquantlib.models.parameter.ConstantParameter` / `NullParameter` / `PiecewiseConstantParameter` / `TermStructureFittedParameter`.
- `pquantlib.models.model.Model` — abstract base. Holds Parameter list + Constraint. `params()` / `set_params(...)`.
- `pquantlib.models.model.TermStructureConsistentModel` — abstract intermediate.
- `pquantlib.models.model.CalibratedModel` — abstract; provides `calibrate(...)` orchestration.
- `pquantlib.models.calibration_helper.CalibrationHelper` — abstract base for calibration instruments.
- `pquantlib.models.calibration_helper.BlackCalibrationHelper` — abstract specialization (carries Black vol quote).
- **Cross-cluster Protocols**: `ModelProtocol`, `CalibrationHelperProtocol`, `ShortRateModelProtocol`.

### L4-B — short-rate models (parallel, ~8 classes)

- `pquantlib.models.shortrate.short_rate_model.ShortRateModel` — abstract subclass of `CalibratedModel`.
- `pquantlib.models.shortrate.onefactor.one_factor_model.OneFactorModel` — abstract.
- `pquantlib.models.shortrate.onefactor.one_factor_affine_model.OneFactorAffineModel` — abstract; provides closed-form `discount(t)` and `discount_bond(t, T, x)`.
- `pquantlib.models.shortrate.onefactor.vasicek.Vasicek` — original Vasicek (1977).
- `pquantlib.models.shortrate.onefactor.hull_white.HullWhite` — extended-Vasicek with mean-reversion + term-structure fit.
- `pquantlib.models.shortrate.onefactor.cox_ingersoll_ross.CoxIngersollRoss` — CIR (1985).
- `pquantlib.models.shortrate.onefactor.black_karasinski.BlackKarasinski` — log-normal short rate.
- `pquantlib.models.shortrate.onefactor.extended_cox_ingersoll_ross.ExtendedCoxIngersollRoss` (bonus port if scope allows).

### L4-C — equity stochastic-volatility (parallel, ~6 classes)

- `pquantlib.processes.heston_process.HestonProcess` — subclass of `StochasticProcess` from L3-D. Multi-D (S, V).
- `pquantlib.models.equity.heston_model.HestonModel` — subclass of `CalibratedModel`. Wraps a `HestonProcess`; exposes `theta` / `kappa` / `sigma` / `rho` / `v0` parameters.
- `pquantlib.models.equity.heston_model_helper.HestonModelHelper` — `BlackCalibrationHelper` for vanilla European options under Heston.
- `pquantlib.processes.bates_process.BatesProcess` — Heston + Merton jumps.
- `pquantlib.models.equity.bates_model.BatesModel`.
- `pquantlib.pricingengines.vanilla.analytic_heston_engine.AnalyticHestonEngine` — characteristic-function + numerical integration. Calibration target.

### L4-D — two-factor short-rate + multi-currency (parallel, ~6 classes)

- `pquantlib.models.shortrate.two_factor_model.TwoFactorModel` — abstract.
- `pquantlib.processes.g2_process.G2Process` — Gaussian 2-factor process (sub of `StochasticProcess`).
- `pquantlib.models.shortrate.twofactor.g2.G2` — Brigo-Mercurio 2-factor Gaussian model.
- `pquantlib.models.shortrate.twofactor.g2.G2ForwardProcess` (helper).
- `pquantlib.processes.hull_white_forward_process.HullWhiteForwardProcess` — forward-measure HW process (bonus).
- `pquantlib.processes.cox_ingersoll_ross_process.CoxIngersollRossProcess` — for CIR-based engines.

### L4-E — calibration helpers + analytic swaption/capfloor engines (parallel, ~10 classes)

- `pquantlib.models.calibration_helper.SwaptionHelper` — BlackCalibrationHelper concrete for swaptions.
- `pquantlib.models.calibration_helper.CapHelper` — concrete for caps.
- `pquantlib.pricingengines.swaption.black_swaption_engine.BlackSwaptionEngine` — model-free, uses swaption vol cube.
- `pquantlib.pricingengines.swaption.bachelier_swaption_engine.BachelierSwaptionEngine` (or rolled into BlackSwaption with displacement).
- `pquantlib.pricingengines.swaption.jamshidian_swaption_engine.JamshidianSwaptionEngine` — analytic under one-factor affine (HW, Vasicek, ExtendedCIR).
- `pquantlib.pricingengines.swaption.g2_swaption_engine.G2SwaptionEngine` — analytic via 2-D numerical integration on G2.
- `pquantlib.pricingengines.swaption.tree_swaption_engine.TreeSwaptionEngine` (defer — needs full lattice machinery).
- `pquantlib.pricingengines.capfloor.analytic_capfloor_engine.AnalyticCapFloorEngine` — uses model `discount_bond_option`.
- `pquantlib.pricingengines.capfloor.black_capfloor_engine.BlackCapFloorEngine` — model-free, uses OptionletVolatilityStructure.
- `pquantlib.pricingengines.capfloor.bachelier_capfloor_engine.BachelierCapFloorEngine`.

Also instruments deferred from Phase 3 that get a pricing engine in Phase 4:
- `pquantlib.instruments.swaption.Swaption` — instrument port (was carved out from L3).
- `pquantlib.instruments.cap_floor.CapFloor` (+ `Cap` + `Floor`).

## Carve-outs (deferred)

### MarketModels (Phase 5 or never)
All 125 files under `models/marketmodels/` — LIBOR Market Model machinery. Very specialized; defer entirely.

### Volatility models (Phase 5)
GARCH, GarmanKlass, ConstantEstimator, SimpleLocalEstimator under `models/volatility/`.

### Specialty short-rate (Phase 5+ or on demand)
- `Gaussian1dModel` + GSR + MarkovFunctional (depend on numerical integration framework + Gaussian process abstractions deferred).
- `BatesDoubleExpModel` / `BatesDoubleExpEngine`.
- `PiecewiseTimeDependentHestonModel`.
- `HestonSLVFDMModel` / `HestonSLVMCModel` (stochastic-local-vol with FDM/MC engines).
- `GJRGARCHModel`.

### Tree/lattice engines (Phase 5)
- TreeSwaptionEngine, TreeCapFloorEngine, DiscretizedSwaption, DiscretizedCapFloor.
- Joshi4 / AdditiveEQP / Trigeorgis trees.
- All MC and FD engines.

### Calibration specialties
- BasketGenerator-based calibration.
- Gaussian1d-based calibrators.

## Cluster topology

Same proven pattern as Phases 1-3: **1 sequential pilot + 4 parallel via subagents**.

L4-A pilot ports the optimizer carry-overs (so L4-E's calibration helpers + L4-B/C/D's model calibration paths all unblock at once), plus the Model/CalibrationHelper bases + cross-cluster Protocols.

**Pre-port discipline (Phase 4 lesson)**: identify shared types upfront. For Phase 4: `Parameter` + concrete sub-types (`ConstantParameter`, etc.) are needed by both L4-A (defining Model) and L4-B/C/D (using them). L4-A pilot ports all Parameter concretes.

## Per-class TDD discipline

Identical to Phases 1-3. Five-step loop: probe → JSON → failing pytest → port → green → commit `feat(<topic>): port <ClassName>` with `-s` sign-off, no `Co-authored-by`.

## Tolerance discipline

Same three tiers. **L4 specifics**:
- Model calibration: LOOSE tier (optimizer convergence). Document target tolerance per test.
- Analytic engines (closed-form): TIGHT.
- Heston engine: LOOSE (numerical integration of complex characteristic function).

## Pause triggers (binding)

Same A1-A8 set. **A4 watch**: Phase 4 may need scipy.linalg.solve_continuous_lyapunov for G2 calibration; if so, document.

## Decision log

| Decision | Rationale |
|---|---|
| **Close Phase 1 optimizer carry-overs in L4-A pilot** (LM + Simplex). | They're required by every model calibration path; unblock all L4 calibration at once. |
| **Heston engine via scipy quad over characteristic function** rather than the custom Gauss-Laguerre quadrature in C++. | Python's scipy.integrate.quad is well-tested and accurate enough; avoid porting the Gauss-Laguerre weights table. |
| **Tree engines deferred** (TreeSwaptionEngine, TreeCapFloorEngine). | Depend on the lattice/Tree/DiscretizedAsset machinery deferred from Phase 3. Land together in Phase 5 when full lattice port happens. |
| **MarketModels deferred entirely**. | 125 files of specialty LIBOR Market Model machinery. Defer to Phase 5 or beyond. |
| **`Parameter` concretes ported in L4-A**, not split per cluster. | Avoid duplication that happened with Compounding/InterestRate in Phase 2. |
| **`Swaption` + `CapFloor` instruments ported in L4-E** (alongside their engines) rather than retroactively in a Phase 3 follow-up. | Cleaner to ship instrument + engine pair together. |
| **`AnalyticHestonEngine`** ported in L4-C (alongside HestonModel) since they're tightly coupled, even though Engines belong in L3-style organization. |

## Plan + executable tasks

See [`phase4-plan.md`](phase4-plan.md) for the bite-sized executable task list (per-cluster subagent prompts).

## Linked

- [`phase3-completion.md`](phase3-completion.md) — Phase 3 closure summary; Phase 4 builds on its 1284-test baseline.
- jquantlib `phase2-L4-models-plan.md` — sister-project anchor (delta-only).

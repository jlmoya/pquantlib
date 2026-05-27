# Phase 4 L4-A pilot — completion

**Date closed:** 2026-05-27
**Tag:** `pquantlib-phase4-l4-A-complete` @ `657b707`
**Predecessor:** `pquantlib-phase3-complete` @ `aacc2c2`
**Test delta:** 1284 → 1351 (+67; exceeded +60 target). Triad clean.
**Successor:** merged into `pquantlib-phase4-complete` @ `fab5a0d` via 4 parallel clusters.

## Stages closed

| Stage | Topic | Tests | Approach |
|---|---|---|---|
| 0 | foundations mega-probe | — | C++ probe → `l4a/foundations.json` (LM + Simplex on Rosenbrock) |
| 1 | Phase 1 carry-overs: LevenbergMarquardt + Simplex | +15 | scipy.optimize.least_squares + scipy.optimize.minimize wrappers |
| 2 | Parameter hierarchy | +17 | Null + Constant + PiecewiseConstant + TermStructureFitted |
| 3 | Model abstract bases | +16 | Model + TermStructureConsistentModel + CalibratedModel (with calibrate() orchestration) |
| 4 | CalibrationHelper bases | +11 | CalibrationHelper + BlackCalibrationHelper |
| 5 | Cross-cluster Protocols | +8 | Model / CalibrationHelper / ShortRateModel |

## Notable decisions

- **scipy-backed optimizers**: LM via `scipy.optimize.least_squares(method='lm')`, Simplex via `scipy.optimize.minimize(method='Nelder-Mead')`. EndCriteria.Type translation: scipy LM `gtol`/`ftol`/`xtol` → `ZeroGradientNorm` / `StationaryFunctionValue` / `StationaryPoint`.
- **Constraint handling**: penalty-based (large finite residual when infeasible) — matches C++ `QL_MAX_REAL` callback convention.
- **`use_cost_functions_jacobian` flag** accepted on LM ctor for API parity but ignored.
- **`Parameter::Impl`** preserved as a strategy class (not collapsed) because `TermStructureFittingParameter::NumericalImpl` carries non-parameter state.
- **`Model` ABC** introduced for typing surface (no C++ counterpart).
- **`_ProjectedConstraintAdapter` / `_AndConstraint`** inlined as private classes in `model.py` (only consumer is L4-A's CalibratedModel.calibrate orchestration).

## Files of note (relative to repo root)

- `pquantlib/src/pquantlib/math/optimization/{levenberg_marquardt,simplex}.py` — Phase 1 carry-overs closed.
- `pquantlib/src/pquantlib/models/parameter.py` — Parameter hierarchy.
- `pquantlib/src/pquantlib/models/model.py` — Model + CalibratedModel + TermStructureConsistentModel.
- `pquantlib/src/pquantlib/models/calibration_helper.py` — base classes.
- `pquantlib/src/pquantlib/models/protocols.py` — cross-cluster Protocols.
- `migration-harness/cpp/probes/l4a/foundations_probe.cpp` + `migration-harness/references/l4a/foundations.json`.

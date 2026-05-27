# Phase 4 — L4 models (executable plan)

> **For agentic workers:** sub-cluster work uses the proven Phase 1-3 pattern — sequential per-stub TDD for L4-A pilot; parallel via subagents for B/C/D/E.

**Goal:** Land Phase 4's ~40 must-port classes on `main`, behind tag `pquantlib-phase4-complete`. Unblocks Phase 3 carve-outs (Swaption + CapFloor + Heston engines).

**Predecessor:** `pquantlib-phase3-complete` @ `aacc2c2` — 1284/0/0, pyright + ruff clean.

**Date:** 2026-05-27.

---

## Task 0 — Spawn pilot worktree

```bash
git worktree add -b phase4-A ../pquantlib-phase4-A main
cd ../pquantlib-phase4-A
uv sync
uv run pytest -q  # confirm 1284/0/0
```

---

## L4-A pilot — foundations + optimizer carry-overs (sequential)

**Goal**: ~10 classes + Phase 1 carry-overs (LM + Simplex). Tag `pquantlib-phase4-l4-A-complete` when done.

### Stage 0 — Probe scaffolding
File: `migration-harness/cpp/probes/l4a/foundations_probe.cpp` covering LevenbergMarquardt + Simplex on a known Rosenbrock-style problem + parameter constraint transforms.

### Stage 1 — Phase 1 optimizer carry-overs
- `LevenbergMarquardt` (wraps `scipy.optimize.least_squares(method='lm')`).
- `Simplex` (wraps `scipy.optimize.minimize(method='Nelder-Mead')`).

Both subclass `OptimizationMethod` from Phase 1 L1-D. Tests cross-validated against C++ probe optimum.

### Stage 2 — Parameter hierarchy
- `Parameter` abstract + `NullParameter` + `ConstantParameter` + `PiecewiseConstantParameter` + `TermStructureFittedParameter`.

### Stage 3 — Model abstract bases
- `Model`, `TermStructureConsistentModel`, `CalibratedModel`.

### Stage 4 — CalibrationHelper bases
- `CalibrationHelper` abstract + `BlackCalibrationHelper` abstract.

### Stage 5 — Cross-cluster Protocols
- `ModelProtocol`, `CalibrationHelperProtocol`, `ShortRateModelProtocol`.

### L4-A closure
FF-merge, tag `pquantlib-phase4-l4-A-complete`, push, cleanup.

Estimated test delta: **+60**.

---

## L4-B/C/D/E — parallel dispatch

Spawn 4 worktrees off `pquantlib-phase4-l4-A-complete`. Dispatch 4 Agent calls in one tool-use turn.

### L4-B subagent prompt (short-rate models, ~8 classes)
Targets: `ShortRateModel` + `OneFactorModel` + `OneFactorAffineModel` + Vasicek + HullWhite + CIR + BlackKarasinski + ExtendedCIR.

Probe: discount-bond + bond-option closed-form values for each model at known parameters.

Target test delta: ~35.

### L4-C subagent prompt (equity stochastic-vol, ~6 classes)
Targets: HestonProcess + HestonModel + HestonModelHelper + BatesProcess + BatesModel + AnalyticHestonEngine.

Probe: AnalyticHestonEngine call/put price at textbook Heston parameters (κ=2, θ=0.04, σ=0.3, ρ=-0.7, v0=0.04).

Target test delta: ~35.

### L4-D subagent prompt (two-factor short-rate, ~6 classes)
Targets: TwoFactorModel + G2Process + G2 + G2ForwardProcess + HullWhiteForwardProcess + CoxIngersollRossProcess.

Probe: G2 discount-bond + G2 swaption value at known parameters.

Target test delta: ~30.

### L4-E subagent prompt (calibration helpers + analytic engines + Swaption/CapFloor instruments, ~10 classes)
Targets: Swaption + CapFloor instruments (carve-outs from L3) + SwaptionHelper + CapHelper + BlackSwaptionEngine + JamshidianSwaptionEngine + G2SwaptionEngine + AnalyticCapFloorEngine + BlackCapFloorEngine + BachelierCapFloorEngine.

Probe: Swaption price under Black, Bachelier, Jamshidian (HW), G2 — same option, 4 engines.

Target test delta: ~40.

---

## Task 1 — Spawn worktrees + dispatch

Standard 4-worktree spawn + uv sync + 4 parallel Agent calls.

## Task 2 — Merge + tag

```bash
git merge --no-ff phase4-B -m "merge: L4-B (short-rate models)"
git merge --no-ff phase4-C -m "merge: L4-C (Heston + Bates equity stochastic-vol)"
git merge --no-ff phase4-D -m "merge: L4-D (G2 two-factor + multi-process)"
git merge --no-ff phase4-E -m "merge: L4-E (Swaption + CapFloor instruments + analytic engines + calibration helpers)"
```

Resolve conflicts (CMakeLists.txt + possibly cross-cluster Protocol params).

Tag `pquantlib-phase4-complete`, push, cleanup.

## Expected outcomes

| Cluster | Class count | Test delta (est.) |
|---|---|---|
| L4-A pilot | ~10 | +60 |
| L4-B short-rate | ~8 | +35 |
| L4-C Heston/Bates | ~6 | +35 |
| L4-D G2 + multi-process | ~6 | +30 |
| L4-E calibration + engines + instruments | ~10 | +40 |
| **Total Phase 4** | **~40** | **~200 → 1484/0/0 cumulative** |

## Linked

- [`phase4-design.md`](phase4-design.md) — binding spec.
- [`phase3-completion.md`](phase3-completion.md) — Phase 3 closure + carry-overs (Swaption/CapFloor instruments closed in L4-E).

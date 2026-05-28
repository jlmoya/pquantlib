# Phase 6 L6-A — LongstaffSchwartz American MC (design)

**Date:** 2026-05-28
**Status:** drafted
**Predecessor:** `pquantlib-phase5-complete` @ `d322fca` — 1883/0/0, pyright + ruff clean
**C++ ground truth:** QuantLib v1.42.1 @ `099987f0`
**Style:** lean — leans on [`phase6-design.md`](phase6-design.md).

## What this closes

The Phase 5 L5-C carve-out for the LongstaffSchwartz American MC engine. This is the
most commonly-cited gap in early-exercise pricing — the Longstaff-Schwartz 1998 paper
American put at S=36 K=40 r=6% σ=20% T=1 (25 weekly exercise dates) is the canonical
literature reference for cross-validation.

## Classes

| Class | C++ source | Tier |
|---|---|---|
| `pquantlib.methods.montecarlo.lsm_basis_system.LsmBasisSystem` | `ql/methods/montecarlo/lsmbasissystem.{hpp,cpp}` | TIGHT (basis evaluation) |
| `pquantlib.methods.montecarlo.early_exercise_path_pricer.EarlyExercisePathPricer[PathT]` | `ql/methods/montecarlo/earlyexercisepathpricer.hpp` | abstract |
| `pquantlib.methods.montecarlo.longstaff_schwartz_path_pricer.LongstaffSchwartzPathPricer[PathT]` | `ql/methods/montecarlo/longstaffschwartzpathpricer.hpp` | LOOSE (regression + MC variance) |
| `pquantlib.pricingengines.mc_longstaff_schwartz_engine.MCLongstaffSchwartzEngine` | `ql/pricingengines/mclongstaffschwartzengine.hpp` | abstract |
| `pquantlib.pricingengines.vanilla.mc_american_engine.MCAmericanEngine` | `ql/pricingengines/vanilla/mcamericanengine.{hpp,cpp}` | LOOSE (MC + regression) |
| `pquantlib.pricingengines.vanilla.mc_american_engine.AmericanPathPricer` | same as above | TIGHT (basis system construction) |

`LongstaffSchwartzMultiPathPricer` (the multi-asset variant) is deferred — there is no
multi-asset PathGenerator in pquantlib yet (Phase 5 carve-out) and no multi-asset MC
engine consumes it, so porting the multi-path pricer in isolation is dead code.

## Polynomial basis support

C++ ships seven polynomial types: `Monomial`, `Laguerre`, `Hermite`, `Hyperbolic`,
`Legendre`, `Chebyshev`, `Chebyshev2nd`. The C++ `pathBasisSystem` uses
`GaussianOrthogonalPolynomial::weightedValue(n, x) = sqrt(w(x)) * value(n, x)` (with
`w(x)` the orthogonality weight) — this is for numerical stability in the regression.

`AmericanPathPricer`'s constructor restricts the set to `Monomial`, `Laguerre`,
`Hermite`, `Hyperbolic`, `Chebyshev2nd` (no Legendre / Chebyshev1st). The Python port:

- **Implements** `Monomial`, `Laguerre`, `Hermite`, `Chebyshev2nd` (the four allowed by
  C++ AmericanPathPricer that are in numpy.polynomial or have closed-form weights).
- **Defers** `Hyperbolic` (custom orthogonality, not in numpy.polynomial — would need
  bespoke port of `GaussHyperbolicPolynomial` from `ql/math/integrals/gaussianorthogonalpolynomial.cpp`).
- **Defers** `Legendre`, `Chebyshev` (not in AmericanPathPricer's allowed set anyway).

For the four implemented types we provide our own `weighted_value(n, x)` using the
explicit recurrences from `gaussianorthogonalpolynomial.cpp` (so the basis matches C++
bit-for-bit). The numpy.polynomial modules are used only for sanity cross-checks in
tests, not in production code, because their convention (probabilist's HermiteE,
unscaled Laguerre, etc.) doesn't match what C++ ships out of the box.

## Algorithm

`LongstaffSchwartzPathPricer.__call__` works in two phases:

- **Calibration phase** (first `n_calibration_samples` calls): record the path verbatim;
  return 0.0 as a placeholder NPV. The `MCLongstaffSchwartzEngine.calculate` calls
  `addSamples(n_calibration_samples)` to load them, then calls `calibrate()`.

- **Calibration step** (`calibrate()`): backward induction from `t = len-2` down to
  `t = 1`. At each step:
  1. Compute path-by-path immediate exercise values.
  2. Filter ITM paths (exercise > 0).
  3. Solve the regression `Y = X · coef` where `X[j, l] = v_l(state(path_j, t))`,
     `Y[j] = discount_t_to_t+1 * price[j]` (the discounted continuation value carried
     back). Use `numpy.linalg.lstsq`.
  4. Per-path update: if regressed continuation < exercise, replace `price` with
     `exercise`.

- **Pricing phase** (subsequent calls): forward through the path; at each early-exercise
  point evaluate the trained regression to compute continuation; if continuation <
  exercise, take immediate exercise.

The discount factors `dF[i] = TS.discount(t[i+1]) / TS.discount(t[i])` are pre-computed
in the constructor.

## Construction wiring (MCAmericanEngine.calculate)

C++ uses a single template chain. The Python port collapses to a concrete inheritance
chain `MCAmericanEngine ← MCLongstaffSchwartzEngine ← (GenericEngine + McSimulation)`:

1. `MCAmericanEngine.__init__(process, time_steps=..., samples=..., polynom_order=2,
   polynom_type=PolynomialType.Monomial, calibration_samples=2048, seed=0,
   antithetic_variate=False, control_variate=False)`.
2. `calculate()` (inherited from `MCLongstaffSchwartzEngine`):
   - Build the `LongstaffSchwartzPathPricer` (subclass hook `lsm_path_pricer()`).
   - Build a calibration `PathGenerator` + `MonteCarloModel`, drive it for
     `calibration_samples` paths (pricer is in calibration mode).
   - Call `path_pricer_.calibrate()`.
   - Run the canonical sampling loop via `McSimulation.run_mc(required_samples,
     required_tolerance, max_samples)` (pricer now in evaluation mode).
   - Fill `results.value` + `error_estimate`.

`MCAmericanEngine.lsm_path_pricer` instantiates an `AmericanPathPricer` (with the
basis + polynomial order) wrapped by a `LongstaffSchwartzPathPricer[Path]`.

`time_grid` for an American option: take `process.time(exercise.last_date())` as the
single mandatory time, then `TimeGrid(0, t, time_steps)` (the C++ code in
`MCLongstaffSchwartzEngine::timeGrid` for `American`).

For Bermudan exercise, the C++ uses `TimeGrid(required_times, steps)` with the
mandatory times being all exercise dates. We mirror this via
`TimeGrid.with_mandatory_and_steps`.

`AmericanPathPricer.state(path, t) = path[t] * scaling`, where `scaling = 1.0 /
strike` for `StrikedTypePayoff`. C++ rescales for numerical stability — we mirror
exactly.

## Probe coverage

Single probe at `migration-harness/cpp/probes/cluster_l6a/probe.cpp` →
`migration-harness/references/cluster/l6a.json`. Captures:

- `lsm_monomial_order_3`: monomial basis values at x ∈ {0.5, 1.0, 1.5} for order
  0…3 → confirms TIGHT-tier match.
- `lsm_laguerre_order_3`: same, Laguerre weighted-value at x ∈ {0.5, 1.0, 1.5}.
- `lsm_hermite_order_3`: same, Hermite weighted-value at x ∈ {0.5, 1.0, 1.5}.
- `lsm_chebyshev2nd_order_3`: same, Chebyshev2nd at x ∈ {-0.5, 0.0, 0.5}.
- `mc_american_put_lsm_1998`: Longstaff-Schwartz 1998 paper Table 1 reference (S=36,
  K=40, r=6%, σ=20%, T=1, 50 timesteps, Monomial order 2, calibration_samples=2048,
  samples=4096, seed=42) → expect ~4.478 with LOOSE 3-sigma band.
- `analytic_european_put_reference`: BSM European put at the same params (S=36, K=40,
  r=6%, q=0, σ=20%, T=1) for cross-comparison — gives ~3.844 (American >= European by
  early-exercise premium ≈ 0.6).

Both LSM basis evaluation and the C++ MC reference value bake into one C++ probe
JSON. The Python tests:

- `test_lsm_basis_system.py`: TIGHT match against probe values per polynomial type.
- `test_longstaff_schwartz_path_pricer.py`: LOOSE convergence on a synthetic linear
  basis + simple put payoff (regression coefficients stable under seed).
- `test_mc_american_engine.py`:
  - LOOSE: NPV within 3-sigma of probe `mc_american_put_lsm_1998` value AND within
    0.05 of the 1998 paper reference 4.478.
  - LOOSE: American >= European (early-exercise premium > 0).
  - LOOSE: same seed → reproducible NPV.
  - LOOSE: deep-OTM call (S=36 K=80) ≈ 0 (smoke test).
  - LOOSE: control variate (with `analytic European` as CV) reduces standard error.

Target test delta: 15-25 tests.

## Documented divergences

- `LsmBasisSystem`: emits a list of callables (matching C++ `std::vector<std::function>`)
  but wrapped in a dedicated `LsmBasisSystem` class. Static methods (`path_basis_system`,
  `multi_path_basis_system`) mirror the C++ static names.
- `PolynomialType.Legendre`, `Chebyshev`, `Hyperbolic` raise `NotImplementedError`
  at construction with a pointer to the Phase 6 carve-out.
- `LongstaffSchwartzPathPricer.calibrate()` uses `numpy.linalg.lstsq` instead of the
  C++ `GeneralLinearLeastSquares` — same QR-based solution, just letting numpy do the
  decomposition. Coefficients are a numpy float64 array indexed in basis order.
- `MCAmericanEngine.controlVariate` (with analytic European as CV) is supported but the
  multi-RNG calibration plumbing from C++ (`RNG_Calibration` template arg with separate
  seed) is collapsed — the calibration uses the same RNG as pricing, seeded with
  `seed + 1768237423` to keep calibration and pricing path sequences distinct (mirrors
  C++ `seedCalibration_ = seed + 1768237423L` when seed != 0).
- `Path` is mutated in place per generator call (already a Phase 5 L5-C divergence).
  Calibration must clone the path values into a stored ndarray, not the reference, to
  keep history. Done via `paths_.append(numpy.copy(path.values))`.

## Carve-outs (Phase 6+)

- `LongstaffSchwartzMultiPathPricer` (multi-asset MC) — no consumer.
- `Hyperbolic`, `Legendre`, `Chebyshev` (1st-kind) polynomial bases — out of
  AmericanPathPricer's allowed set or needs custom orthogonality weights.
- `MakeMCAmericanEngine` builder — Python kwargs make builders unnecessary (mirrors
  L5-C's decision for `MakeMCEuropeanEngine`).
- Calibration with a different RNG family than pricing (`RNG_Calibration` C++ template
  arg) — niche use case.

## Files

```
pquantlib/src/pquantlib/methods/montecarlo/lsm_basis_system.py
pquantlib/src/pquantlib/methods/montecarlo/early_exercise_path_pricer.py
pquantlib/src/pquantlib/methods/montecarlo/longstaff_schwartz_path_pricer.py
pquantlib/src/pquantlib/pricingengines/mc_longstaff_schwartz_engine.py
pquantlib/src/pquantlib/pricingengines/vanilla/mc_american_engine.py

pquantlib/tests/methods/montecarlo/test_lsm_basis_system.py
pquantlib/tests/methods/montecarlo/test_longstaff_schwartz_path_pricer.py
pquantlib/tests/pricingengines/vanilla/test_mc_american_engine.py

migration-harness/cpp/probes/cluster_l6a/probe.cpp
migration-harness/cpp/probes/CMakeLists.txt  (+ cluster_l6a entry)
migration-harness/references/cluster/l6a.json
```

## Commits (planned)

1. `feat(montecarlo): port LsmBasisSystem (4 polynomial types)`
2. `feat(montecarlo): port LongstaffSchwartzPathPricer[PathT] + EarlyExercisePathPricer`
3. `feat(pricingengines): port MCLongstaffSchwartzEngine + MCAmericanEngine + AmericanPathPricer`

Each commit:

- Adds the C++ probe entry + JSON file delta if its content is new.
- Adds the Python file + tests, all green.
- Triad (pytest + pyright + ruff) green.
- `-s` Signed-off-by, no `Co-authored-by`.

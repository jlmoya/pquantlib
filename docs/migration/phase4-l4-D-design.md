# Phase 4 L4-D — G2 two-factor + multi-process suite

**Date:** 2026-05-27
**Status:** **closed** — merged via `bdf44fa merge: L4-D`; tagged as part of `pquantlib-phase4-complete` @ `fab5a0d`. Final test delta: **+48**. 3 commits.
**Predecessor:** `pquantlib-phase4-l4-A-complete` @ `657b707`
**Style:** lean — leans on [`phase4-design.md`](phase4-design.md).

## What landed

### Processes
- `pquantlib.processes.g2_process.G2Process(a, sigma, b, eta, rho)` — Gaussian 2-factor (subclass of `StochasticProcess`).
- `pquantlib.processes.g2_forward_process.G2ForwardProcess` — forward-measure G2.
- `pquantlib.processes.hull_white_forward_process.HullWhiteForwardProcess` — forward-measure HW (bonus port).
- `pquantlib.processes.cox_ingersoll_ross_process.CoxIngersollRossProcess` (also ported by L4-B; merge dedup).
- `pquantlib.processes.ornstein_uhlenbeck_process.OrnsteinUhlenbeckProcess` (also ported by L4-B; merge dedup).
- `pquantlib.processes.forward_measure_process.ForwardMeasureProcess` (1D + multi-D).

### Models
- `pquantlib.models.shortrate.two_factor_model.TwoFactorModel` (abstract).
- `pquantlib.models.shortrate.twofactor.g2.G2(termStructure, a, sigma, b, eta, rho)` — Brigo-Mercurio G2++ short-rate model. Closed-form `discount(t)`, `discount_bond(now, maturity, x0, x1)`, `discount_bond_option(type, strike, maturity, bond_maturity)` (Black-76), `swaption(...)` via 1-D SegmentIntegral × inner Brent solve.

## Documented divergences

- **`TermStructureConsistentModel` drops `__slots__`** + adds `_set_term_structure` late-bind helper to enable the G2 diamond (CalibratedModel + TermStructureConsistentModel). L4-B made the same `__slots__` drop; merged. The `_set_term_structure` helper was added in main session post-merge to support G2's tests.
- **`G2` omits the explicit `AffineModel` base** since pquantlib uses `ShortRateModelProtocol` (Protocol).
- **`G2.swaption` takes unpacked primitives** instead of a `Swaption::arguments` struct (Swaption is L4-E scope; this avoids cross-cluster dep).
- **`G2.FittingParameter` inlined as a plain callable** since phi(t) is closed-form.
- **CIR `variance(t0, x0, dt)` and `diffusion(t, x)`** preserve C++ quirks (uses stored `x0_`; reports `sigma` not `sigma*sqrt(x)`).
- **HullWhiteForwardProcess.alpha / M_T / expectation_1d** at LOOSE tier post-merge — forward-rate-near-t1==t2 numerical-diff sensitivity (~1e-12 rel between independent align fixes).

## Align commits

- Same `align(termstructures): forward_rate(t,t)` fix as L4-B — independent, identical patch. Merged cleanly.
- `align(models.model): drop __slots__ + add diamond-cooperative ctor on TermStructureConsistentModel` — same content as L4-B's; merged.

## Carve-outs

- **`TwoFactorModel.tree(grid)`** raises `LibraryException` — needs Lattice2D + TrinomialTree (Phase 5).
- **`StochasticProcessArray`** — only used by `ShortRateDynamics.process()`; not consumed by any L4 path.

# Phase 4 L4-B — Short-rate models (1-factor)

**Date:** 2026-05-27
**Status:** **closed** — merged via L4-B merge; tagged as part of `pquantlib-phase4-complete` @ `fab5a0d`. Final test delta: **+49**. 6 commits.
**Predecessor:** `pquantlib-phase4-l4-A-complete` @ `657b707`
**Style:** lean — leans on [`phase4-design.md`](phase4-design.md).

## What landed

### Processes (under `pquantlib.processes.*`)
- `OrnsteinUhlenbeckProcess(speed, vol, x0, level)`.
- `CoxIngersollRossProcess(speed, vol, x0, level)`.

### Model abstracts (under `pquantlib.models.shortrate.*`)
- `ShortRateModel` (abstract subclass of `CalibratedModel`).
- `OneFactorModel`.
- `OneFactorAffineModel` (closed-form `A(t,T)`, `B(t,T)`, `discount_bond_option` via Jamshidian).

### Concretes (under `pquantlib.models.shortrate.onefactor.*`)
- `Vasicek(r0, a, b, sigma, lambda_=0.0)` — Vasicek (1977).
- `HullWhite(termStructure, a=0.1, sigma=0.01)` — extended-Vasicek with phi(t) closed-form + 4-arg/5-arg `discount_bond_option` + `convexity_bias`.
- `CoxIngersollRoss(r0, theta, k, sigma)` — CIR (1985).
- `ExtendedCoxIngersollRoss(termStructure, theta, k, sigma, r0)`.

## Documented divergences

- `TermStructureConsistentModel.__init__(term_structure=None)` cooperative-super() escape for diamond MRO (HW/ECIR multi-inherit `CalibratedModel + TermStructureConsistentModel`).
- `OneFactorAffineModel._a(t,T)` / `_b(t,T)` — renamed from C++ `A`/`B` to avoid clashing with Vasicek's `a()`/`b()` accessors.
- `NonCentralCumulativeChiSquareDistribution` delegates to `scipy.stats.ncx2.cdf`; CIR/ECIR option tests use LOOSE (1e-8 rel) instead of TIGHT (1e-12 rel) with inline justification.
- CIR process preserves C++ idiosyncrasies (`diffusion(t,x)=sigma` constant; `variance` uses ctor-time `x0_`/`level_`).
- Vasicek small-`a` degenerate `A=0` branch preserved.
- **`tree(TimeGrid)` on all OneFactorModel subclasses raises** `LibraryException("L4-B carve-out")` — TrinomialTree/TreeLattice1D/ShortRateTree are a separate cluster.

## Align commits (closes pre-existing L2-B bug)

- `align(termstructures): forward_rate(t,t) returns instantaneous rate, not zero` — fixes L2-B equal-times finite-difference branch returning `t_for_rate=0` which raised LibraryException. Surfaced by HullWhite reading the instantaneous forward at t1==t2. **L4-D made an identical independent fix; merged cleanly.**
- `align(models): drop __slots__ from TermStructureConsistentModel` — enables HW/ECIR diamond multi-inheritance.

## Carve-outs

- `BlackKarasinski` — log-normal short rate, no closed-form discount bond; depends on TrinomialTree/lattice. Per task spec ("ok to skip"). Phase 5+.
- `tree(TimeGrid)` on all OneFactorModel subclasses — same reason.
- `NonCentralCumulativeChiSquareSankaranApprox` + inverse-non-central chi-square solver — not needed by L4-B.

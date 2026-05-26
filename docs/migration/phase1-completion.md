# Phase 1 — Completion summary

**Closed:** 2026-05-24
**Tag:** `pquantlib-phase1-complete` @ `8b64830`
**Predecessor tag:** `pquantlib-phase1-l1-A-complete` @ `03d0ce8`
**C++ ground truth:** QuantLib v1.42.1 @ `099987f0`

## Final state

- **581 / 0 / 0** pytest, **pyright strict** clean, **ruff lint + format** clean.
- **L1 layer covered**: foundations, time, daycounters, 41 calendars, math primitives (constants, closeness, rounding, factorial, error function, beta, bernstein, pascal), 12 copulas, 3 normal distributions, 2 statistics aggregators, 5 currencies, 9 Solver1D concretes, 5 simple integrals, 5 deterministic RNGs (all EXACT-tier bit-exact) + BoxMuller, 7 optimization scaffolding modules, 4 interpolations + bilinear, Cholesky.

## Cluster contributions

| Cluster | Tests added | Tag/merge |
|---|---:|---|
| L1-A (pilot) | 415 | `pquantlib-phase1-l1-A-complete` @ `03d0ce8` |
| L1-B (copulas + normal + statistics + currencies) | +50 | merge commit `cbd55ac` |
| L1-C (Solver1D + simple integrals) | +34 | merge commit `6580db9` |
| L1-D (RNGs + optimization scaffolding) | +52 | landed in `5370a08` |
| L1-E (interpolations + Cholesky) | +30 | merge commit `8b64830` |
| **Total** | **581** | `8b64830` (this tag) |

## Parallelization summary

L1-B/C/D/E were dispatched as **4 isolated-worktree subagents in parallel**, each given:
- A pre-emitted reference JSON at `migration-harness/references/cluster/{b,c,d,e}.json` (committed in `59ec380` ahead of dispatch).
- A lean design doc listing must-port targets + explicit carve-outs.
- The L1-A docs as binding ground rules.

Total wall-clock from cluster-design-write to all-4-merged: **~25 minutes**.

The 5 RNGs (MT19937, Knuth, Lecuyer, Ranlux3, Xoshiro256**) + BoxMuller all achieved **EXACT-tier bit-exact sequence match** against the C++ probe (seed = 42, 5 samples each).

## Documented divergences (cumulative across L1)

Captured in the individual cluster designs and spec-review doc. Most notable:
- `math.lgamma` instead of GammaFunction.logValue (Factorial, Beta).
- `math.erf` instead of Sun-Microsystems polynomial fit (ErrorFunction).
- `datetime.strptime` instead of boost::date_time (DateParser.parse_formatted).
- `numpy.ndarray[float64]` as Array/Matrix typing alias (no custom wrapper class).
- `scipy.linalg.cholesky` wraps C++ Cholesky (L1-E).
- Schedule's `Settings.evaluation_date` fallback not ported.
- Business252 requires explicit calendar (no Brazil() default).
- Pre-Cholesky `flexible=True` deferred (eigenvalue-based pseudo-Cholesky for non-PSD matrices).

## Carve-outs deferred to follow-up

L1-B:
- GaussianCopula (depends on BivariateCumulativeNormal — defer).
- Maddock-class inverse cumulatives, StochasticCollocationInvCDF, non-central chi-squared.
- Bivariate cumulative normal Dr78 + We04DP, bivariate cumulative student.
- Histogram, sequence/risk/convergence statistics.

L1-C:
- Full GaussianOrthogonalPolynomial hierarchy (Hermite, Laguerre, Chebyshev, Gegenbauer, Jacobi, Hyperbolic, ~12 subclasses).
- MultiDimGaussianIntegration, TwoDimensionalIntegral, GaussianQuadMultidimIntegrator.
- DiscreteSimpson, DiscreteTrapezoid, TanhSinh, ExpSinh, Filon, Patterson.
- OdeFctWrapper.

L1-D:
- Sobol + Burley2020 + the primitive-polynomials / Joe-Kuo init tables (deep low-discrepancy).
- Halton, Faure, RandomizedLDS.
- InverseCumulativeRng / InverseCumulativeRsg, GenericPseudoRandom, GenericLowDiscrepancy.
- LevenbergMarquardt, BFGS, ConjugateGradient, Simplex, SimulatedAnnealing, DifferentialEvolution.
- LineSearch + subclasses (Armijo, Goldstein).
- ZigguratGaussianRng, CLGaussianRng, InverseCumulativeRng-based Gaussian.

L1-E:
- 8+ cubic-spline variants (Akima, Kruger, FritschButland, Harmonic, LogCubic*, MonotonicCubic, ParabolicCubic, ConvexMonotone).
- XABR family (SABR + variants + ChebyshevInterpolation + KernelInterpolation + Lagrange).
- 2-D extras (BicubicSpline, KernelInterpolation2D, MultiCubicSpline).
- QRDecomposition, EigenvalueDecomposition, SymmetricSchurDecomposition, TqrEigenDecomposition.
- HouseholderReflection/Transformation, OrthogonalProjections.
- BiCGStab, GMRES, SparseILUPreconditioner, SparseMatrix.
- PseudoSqrt, CovarianceDecomposition, FrobeniusCostFunction, HypersphereCostFunction, BasisIncompleteOrdered, Expm.
- ModifiedBesselFunction, FastFourierTransform, RegularisedIncompleteBeta.

## What's next

- **Phase 2 (L2)**: termstructures + indexes. Mirrors jquantlib `phase2-L2-termstructures-indexes-plan.md`.
- Follow-up clusters for the L1 carve-outs (especially Sobol, GammaFunction, full cubic-spline family, optimization methods) will land as sub-clusters of L2 or as a dedicated "L1-completion" cluster — TBD by user.

## Lessons learned (cumulative)

(Full discussion in `phase1-l1-A-completion.md`; condensed pointers here.)

1. **Subagent fan-out** is the right pattern for parallelizable porting work — used twice (Stage 4 calendars and L1-B/C/D/E clusters), each time with zero merge conflicts.
2. **Mega-probe + committed reference JSON before dispatch** = subagents can verify locally without touching C++.
3. **Lean design docs work** for clusters that lean on a pilot's discipline doc (L1-A).
4. **EXACT-tier RNG cross-validation is achievable** — all 5 PRNGs matched bit-exactly on first try by porting the C++ source verbatim (Schrage's method, Bays-Durham, SplitMix64 seed expansion).
5. **`scipy` is the natural dep** for linear algebra (Cholesky, LU, SVD). Wrapping numpy/scipy is cleaner than reimplementing C++ algorithms.

## Links

- [Phase 1 design](phase1-design.md) — original L1 scope
- [L1-A completion](phase1-l1-A-completion.md) — pilot-cluster summary
- [L1-A design](phase1-l1-A-design.md) — binding ground rules
- [L1-A spec review](phase1-l1-A-spec-review.md), [L1-A code review](phase1-l1-A-code-review.md)
- [L1-B design](phase1-l1-B-design.md), [L1-C design](phase1-l1-C-design.md), [L1-D design](phase1-l1-D-design.md), [L1-E design](phase1-l1-E-design.md)

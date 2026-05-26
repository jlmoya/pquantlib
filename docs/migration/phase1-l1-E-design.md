# Phase 1 L1-E — Interpolations + matrix utilities + remaining math root

**Date:** 2026-05-24
**Status:** **closed** — merged into `main` via `8b64830 merge: L1-E`; tagged as part of `pquantlib-phase1-complete` @ `edcadbc`. Final test delta: **+30** (Array/Matrix typing aliases + Interpolation abstract + 4 1-D interpolations + Bilinear 2-D + scipy-backed Cholesky). Added `scipy>=1.13` as a workspace dependency.
**Predecessor:** `pquantlib-phase1-l1-A-complete` @ `03d0ce8`
**C++ ground truth:** QuantLib v1.42.1 @ `099987f0`
**Style:** lean — see [`phase1-l1-A-design.md`](phase1-l1-A-design.md) for ground rules.

## Goal

Port the **most common** 1-D and 2-D interpolations, the matrix-utility classes that delegate cleanly to numpy/scipy, and the remaining math root classes that didn't make it into Stage 5.

## Must-port (tractable subset)

### Interpolations (~6 — simple 1-D + 2-D)
- `Interpolation` abstract + `Extrapolator` abstract.
- `LinearInterpolation` — straight piecewise-linear, ~30 LOC.
- `LogLinearInterpolation` — linear-in-log-space.
- `BackwardFlatInterpolation`, `ForwardFlatInterpolation` — piecewise constant.
- `CubicInterpolation` (natural cubic spline only — defer 8+ XABR / Akima / Kruger / FritschButland variants).
- `BilinearInterpolation` — 2-D linear.

### Matrix utilities (~5 — backed by numpy/scipy)
- `Array` typing alias `npt.NDArray[np.float64]`.
- `Matrix` typing alias `npt.NDArray[np.float64]` (rank-2).
- `CholeskyDecomposition` — wrap `scipy.linalg.cholesky`.
- `LUDecomposition` — wrap `scipy.linalg.lu`.
- `SVD` — wrap `scipy.linalg.svd`.
- `Identity` — `numpy.eye`.

### Math root (~5)
- `Constants` (already landed in Stage 5).
- `IntervalPrice` — `@dataclass(frozen=True, slots=True)` (open/close/high/low).
- `Prices` — list-of-IntervalPrice utilities.
- `LinearRegression` — `numpy.linalg.lstsq` wrapper.
- `GeneralLinearLeastSquares` — same.

## Carve-outs

- 8+ Cubic variants: AkimaCubicInterpolation, KrugerCubic, FritschButland, Harmonic, LogCubic*, MonotonicCubic, ParabolicCubic, ConvexMonotone.
- XABR* / Abcd* / SABR / Chebyshev / Kernel / Lagrange interpolations — defers.
- BicubicSpline / KernelInterpolation2D / MultiCubicSpline — defers.
- QRDecomposition, EigenvalueDecomposition, SymmetricSchurDecomposition, TqrEigenDecomposition, HouseholderReflection/Transformation, OrthogonalProjections, BiCGStab, GMRES, SparseILUPreconditioner, SparseMatrix — defers; numpy/scipy can be wrapped follow-up.
- PseudoSqrt, CovarianceDecomposition, FrobeniusCostFunction, HypersphereCostFunction, BasisIncompleteOrdered, Expm.
- ModifiedBesselFunction, FastFourierTransform, RegularisedIncompleteBeta, IntegralForm helpers.

## Approach

Mega-probe covers the 1-D interpolations + 2-D bilinear at known (x, y) points; matrix utilities tested behaviorally (numpy/scipy delegates).

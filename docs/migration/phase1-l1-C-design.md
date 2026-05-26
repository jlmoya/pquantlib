# Phase 1 L1-C — Solvers1D + simple integrals

**Date:** 2026-05-24
**Status:** drafted
**Predecessor:** `pquantlib-phase1-l1-A-complete` @ `03d0ce8`
**C++ ground truth:** QuantLib v1.42.1 @ `099987f0`
**Style:** lean — see [`phase1-l1-A-design.md`](phase1-l1-A-design.md) for ground rules.

## Goal

Port the 9 `Solver1D` concretes (Bisection, Brent, FalsePosition, Newton, NewtonSafe, FiniteDifferenceNewtonSafe, Halley, Ridder, Secant) plus the simple integrals (Simpson, Trapezoid, GaussLegendre, SegmentIntegral, KronrodIntegral, GaussLobattoIntegral). Defer the Gauss-quadrature-polynomial hierarchy and multi-dim integration to a follow-up.

## Must-port (tractable subset)

### Solvers1D base + 9 concretes
- `Solver1D[T]` abstract base (C++ uses CRTP `Solver1D<Impl>` template). Python uses normal abstract method.
- Bisection, Brent, FalsePosition, FiniteDifferenceNewtonSafe, Halley, Newton, NewtonSafe, Ridder, Secant.

Each solver is ~50-100 LOC of Newton-style iteration. All depend on a `bracket(min, max)` + `solve(f, accuracy, guess, step)` method shape.

### Integrals (~6 simple)
- `Integrator` abstract.
- `SimpsonIntegral` (composite Simpson's rule).
- `TrapezoidIntegral` (composite trapezoid).
- `SegmentIntegral` (uniform-segment).
- `GaussLegendreIntegration` (the simple non-polynomial-hierarchy form).
- `KronrodIntegral` (adaptive).
- `GaussLobattoIntegral` (4-point).

### ODE (~1)
- `OdeFctWrapper` — single thin functor wrapping a `Callable[[float, float], float]`.

## Carve-outs

- GaussianOrthogonalPolynomial + the full 12+ subclass hierarchy (Hermite, Laguerre, Chebyshev, Gegenbauer, Jacobi, Hyperbolic) — substantial port, defers to a follow-up.
- MultiDimGaussianIntegration, TwoDimensionalIntegral, GaussianQuadMultidimIntegrator — defers.
- DiscreteSimpsonIntegral / DiscreteTrapezoidIntegral / TanhSinh / ExpSinh / Filon / Patterson — defers.

## Approach

Single mega-probe `math/l1C_probe.cpp` covering all 9 solvers + 6 integrals against known closed-form benchmarks.

# Phase 2 L2-B — Yield curves

**Date:** 2026-05-26
**Status:** **closed** — merged into `main` via `13fc008 merge: L2-B`; tagged as part of `pquantlib-phase2-complete` @ `b5d2519`. Final test delta: **+50** (649 → 699 before subsequent merges). 7 commits.
**Predecessor:** `pquantlib-phase2-l2-A-complete` @ `4ace1f0`
**C++ ground truth:** QuantLib v1.42.1 @ `099987f0`
**Style:** lean — leans on [`phase2-design.md`](phase2-design.md) for ground rules.

## What landed

- `pquantlib.termstructures.yield_term_structure.YieldTermStructure` — abstract; provides `discount(t|date)` / `zero_rate(...)` / `forward_rate(...)`.
- `pquantlib.termstructures.yield_.zero_yield_structure.ZeroYieldStructure` — abstract intermediate.
- `pquantlib.termstructures.yield_.flat_forward.FlatForward` (+ `FlatForward.from_rate(...)` classmethod).
- `pquantlib.termstructures.yield_.interpolated_zero_curve.InterpolatedZeroCurve`.
- `pquantlib.termstructures.yield_.interpolated_forward_curve.InterpolatedForwardCurve`.
- `pquantlib.termstructures.yield_.interpolated_discount_curve.InterpolatedDiscountCurve`.
- PEP 695 type aliases: `ZeroCurve` / `ForwardCurve` / `DiscountCurve` over the interpolated variants with `LinearInterpolation` default.
- `pquantlib.termstructures.yield_.{forward,zero,discount}_spreaded_term_structure.*` — 3 spreaded variants.
- `pquantlib.termstructures.yield_.implied_term_structure.ImpliedTermStructure`.
- **Prerequisites (alignment commits)**: `pquantlib.time.compounding.Compounding` IntEnum; `pquantlib.interest_rate.InterestRate` class.

## Documented divergences

- Subpackage named `yield_` (trailing underscore — `yield` is a Python keyword). Same as L2-C; agreed at merge.
- `Compounding` placed under `pquantlib.time` (next to `Frequency`) rather than C++'s namespace root.
- `InterestRate` uses NaN + `is_null()` rather than C++ `Null<Real>()`.
- C++ template `<Interpolator>` → Python `InterpolationFactory = Callable[[Array, Array], Interpolation]`.
- Settlement-days (moving-mode) constructors deferred until `Settings.evaluation_date` observable lands.
- `DiscountSpreadedTermStructure` exists as a type alias to `InterpolatedSpreadDiscountCurve` (C++'s canonical name); the design's scalar-Quote-spread description was a divergence from C++ (which takes a term-structure of spreads).
- `ForwardSpreadedTermStructure` despite name uses the zero rate (not instantaneous forward) per C++ verbatim.

## Carve-outs

`PiecewiseYieldCurve` (full bootstrap), `BootstrapTraits`, `FittedBondDiscountCurve`, `CompositeZeroYieldStructure`, `MultiCurve`, all spline-fitting variants. Deferred to L2-completion follow-up or Phase 3+.

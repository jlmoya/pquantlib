# Phase 5 L5-A pilot — completion

**Date closed:** 2026-05-28
**Tag:** `pquantlib-phase5-l5-A-complete` @ `aa19340`
**Predecessor:** `pquantlib-phase4-complete` @ `fab5a0d`
**Test delta:** 1544 → 1614 (+70; exceeded +60 target). Triad clean.
**Successor:** merged into `pquantlib-phase5-complete` @ `d322fca`.

## Stages closed

| Stage | Topic | Tests | Approach |
|---|---|---|---|
| 0 | foundations mega-probe | — | C++ probe → `l5a/foundations.json` |
| 1 | Phase 1 carry-overs: Sobol + Burley2020 + GammaFunction + AkimaCubic | +25 | scipy wrappers + Lanczos port |
| 2 | Tree[T] + Lattice base classes | +10 | PEP 695 generic |
| 3 | DiscretizedAsset + DiscretizedOption + DiscretizedDiscountBond | +15 | abstract hierarchy for tree engines |
| 4 | Cross-cluster Protocols | +8 | DiscretizedAsset / Lattice / PathGenerator |

## Notable decisions

- **scipy.stats.qmc.Sobol wraps SobolRsg** — Joe-Kuo default direction integers; alternative C++ sets carved out. Scipy emits the origin as its first draw while C++ starts at index 1; the wrapper advances scipy past the origin to align.
- **scipy.stats.qmc.Sobol(scramble=True)** for Burley2020 — Matousek LMS+shift Owen scrambling (statistically equivalent to C++ Burley 2020 hash; not bit-exact).
- **GammaFunction Lanczos port** uses the same coefficients as C++. `Factorial.get(n > 27)` now delegates to `GammaFunction.log_value(n+1)` instead of `math.lgamma` — same LOOSE tier as before.
- **AkimaCubicInterpolation** uses scipy.interpolate.Akima1DInterpolator (standard reflection rule); C++ uses a non-standard nonlinear endpoint slope formula. Tests assert exact-knot interpolation + quadratic recovery on y=x² inputs.
- **Full CubicInterpolation family** carved out (FritschButland, Kruger, Harmonic, MonotonicAkima, Spline, Parabolic) — these need bespoke porting effort.

## Files of note

- `pquantlib/src/pquantlib/math/randomnumbers/{sobol_rsg.py, burley_2020_sobol_rsg.py}`.
- `pquantlib/src/pquantlib/math/distributions/gamma_function.py` (+ factorial.py update).
- `pquantlib/src/pquantlib/math/interpolations/akima_cubic_interpolation.py`.
- `pquantlib/src/pquantlib/methods/lattices/{tree.py, lattice.py, discretized_asset.py, discretized_option.py, discretized_discount_bond.py}`.
- `pquantlib/src/pquantlib/methods/protocols.py`.
- `migration-harness/cpp/probes/l5a/foundations_probe.cpp` + `migration-harness/references/l5a/foundations.json`.

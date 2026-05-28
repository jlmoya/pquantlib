# Phase 6 L6-B — BatesEngine (analytic jump-diffusion engine)

**Date:** 2026-05-28
**Status:** in-progress on `phase6-B` worktree; will be merged FF into `main`.
**Predecessor:** `pquantlib-phase5-complete` @ `d322fca` — 1883/0/0.
**Style:** lean — leans on [`phase6-design.md`](phase6-design.md).

## Scope

Closes the Phase 4 L4-C carve-out for `BatesEngine`. L4-C ported the
`BatesProcess` / `BatesModel` parameter machinery and added the
`add_on_term(phi, t, j)` hook to `AnalyticHestonEngine`; this cluster
plugs in the jump compensator and validates against C++ probes.

## Classes landed

- `pquantlib.pricingengines.vanilla.bates_engine.BatesEngine(model, integration_order=144)`
  — subclass of `AnalyticHestonEngine`. Overrides `add_on_term(phi, t, j)` to
  return the Bates Merton-jump CF compensator in the Sepp form:

      delta2 = 0.5 * delta^2
      g      = complex(1, phi) if j == 1 else complex(0, phi)
      add_on = t * lambda * ( exp(nu*g + delta2*g^2)
                              - 1
                              - g * (exp(nu + delta2) - 1) )

  The `-g*(exp(nu+delta2) - 1)` piece is the martingale compensator
  that pairs with `BatesProcess.drift`'s `-lambda*m` adjustment.

Narrows `model() -> BatesModel` so jump-parameter introspection
(`lambda_()`, `nu()`, `delta()`) is reachable without an explicit
cast.

## Carve-outs

- **BatesDetJumpEngine** — requires `BatesDetJumpModel`
  (`kappa_lambda`, `theta_lambda` parameters) which L4-C carved out.
  Skipped per phase6-design carve-out list.
- **BatesDoubleExpEngine** — requires `BatesDoubleExpModel`
  (`p`, `nuUp`, `nuDown`) which L4-C carved out. Skipped.
- **BatesDoubleExpDetJumpEngine** — requires both of the above. Skipped.
- **Gauss-Laguerre integration** — inherited LOOSE tier from L4-C
  (scipy.integrate.quad on `(0, +inf)`).

## C++ ground truth

- Pin: `v1.42.1` @ `099987f0`.
- Probe: `migration-harness/cpp/probes/cluster_l6b/probe.cpp` →
  `migration-harness/references/cluster/l6b.json`.
- Probe coverage:
  - **Zero-jump reduction** — BatesEngine with `lambda=delta=1e-12`
    at the Albrecher-Mayer-Schoutens-Tistaert (AMST) testbed (`rho=-0.7`).
    Reference values match the L4-C `heston_engine` block exactly.
  - **AMST with jumps on** (`lambda=0.1, nu=-0.05, delta=0.1`) —
    ATM call/put, OTM call (K=80, 120), and K=90/110 to lock the
    jump-induced skew.
  - **Bates 1996 testbed** (`rho=-0.5`) — ATM call/put.

## Tolerance tiers

| Test                                        | Tier  | Reason                                                                |
| ------------------------------------------- | ----- | --------------------------------------------------------------------- |
| Zero-jump reduction (Bates vs Heston in Py) | TIGHT | Same Fj integrand code path; add_on ~= 0 +0j at lambda=1e-30          |
| C++ probe values (jumps on, AMST + B96)     | LOOSE | scipy.quad vs Gauss-Laguerre divergence at the 7th-8th sig figure     |
| Put-call parity (jumps on)                  | TIGHT | Algebraic identity; martingale jump compensator preserves it          |
| BatesModel param round-trip                 | TIGHT | `ConstantParameter` exact evaluation at t=0                           |

## Test delta

- New: `pquantlib/tests/pricingengines/vanilla/test_bates_engine.py` — 14 tests.
- Cumulative: **1883 → 1897** (`+14`).

## Divergences from C++

- **scipy.integrate.quad over (0, +inf)** replaces 144-pt
  Gauss-Laguerre — inherited from L4-C's `AnalyticHestonEngine`.
- **PositiveConstraint on lambda/delta** at the C++ `BatesModel`
  level means the zero-jump reduction probe uses `lambda=delta=1e-12`
  instead of `0.0` — the Python test uses `1e-30` (same constraint
  enforcement, finer resolution).
- **Static narrowing of `model()`** to `BatesModel` (vs C++'s
  call-site `dynamic_pointer_cast`).
- **Constructor `(rel_tolerance, max_evaluations)` overload** is
  dropped — same treatment as `AnalyticHestonEngine`.

## Operational notes

Single commit. Triad green at each step:

- `uv run pytest`  →  1897/0/0
- `uv run pyright`  →  0 errors, 0 warnings
- `uv run ruff check`  →  All checks passed!

Plan + tasks: this cluster has a single class and was scoped tight
enough that the implementation + tests landed in one commit; no
separate plan file produced.

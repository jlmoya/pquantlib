# Phase 6 — High-impact carve-outs + final closure (executable plan)

**Goal:** Land Phase 6's ~12 high-impact carve-out closures + final closure tooling. Tag `pquantlib-final` post-merge.

**Predecessor:** `pquantlib-phase5-complete` @ `d322fca` — 1883/0/0, pyright + ruff clean.

**Date:** 2026-05-28.

---

## Task 0 — Spawn 3 worktrees + dispatch (no pilot)

```bash
git worktree add -b phase6-A ../pquantlib-phase6-A main
git worktree add -b phase6-B ../pquantlib-phase6-B main
git worktree add -b phase6-C ../pquantlib-phase6-C main
for w in A B C; do cd ../pquantlib-phase6-$w && uv sync; done
```

Dispatch 3 Agent calls in one tool-use turn.

## Cluster scopes

### L6-A — LongstaffSchwartz American MC (~5 classes)

- `LsmBasisSystem` — polynomial basis (Monomial / Chebyshev / Hermite / Laguerre via `numpy.polynomial`).
- `LongstaffSchwartzPathPricer[Path]` — PEP 695 generic.
- `LongstaffSchwartzMultiPathPricer`.
- `MCAmericanEngine(process, time_steps, samples, polynom_type, polynom_order, calibration_samples=2048)`.

Probe: Longstaff-Schwartz 1998 American put reference value (4.478 at S=36/K=40/T=1/r=6%/σ=20%).

Target test delta: ~25.

### L6-B — BatesEngine (~3 classes)

- `BatesEngine(model, integration_order=144)` — extends L4-C's `AnalyticHestonEngine` with Bates jump-CF via `add_on_term` hook.

Probe: Bates 1996 textbook value for vanilla European call under jump-diffusion.

Target test delta: ~10.

### L6-C — DoubleBarrier instrument + analytic engine (~4 classes)

- `DoubleBarrierOption(barrier_type, barrier_lo, barrier_hi, rebate, payoff, exercise)` + `DoubleBarrierType` IntEnum (KnockIn / KnockOut / KIKO / KOKI).
- `AnalyticDoubleBarrierEngine(process)` — Ikeda-Kunitomo (1992) closed-form series.
- `AnalyticDoubleBarrierBinaryEngine(process)` — cash-or-nothing double barrier.

Probe: Hull/Haug textbook double-barrier values.

Target test delta: ~20.

## Final closure (sequential, post-merge)

1. **Comprehensive carve-out doc** at `docs/carve-outs.md` — per-phase + per-category list of every C++ v1.42.1 class NOT ported, with rationale + access via wrapping if applicable.
2. **`pquantlib-samples/` populated** with end-to-end sample programs:
   - `vanilla_swap_pricing.py` — bootstrap curve from deposits/futures/swaps, price a 10y swap.
   - `swaption_calibration.py` — calibrate HW to swaption surface, price a swaption.
   - `heston_calibration.py` — calibrate Heston to vanilla option surface.
   - `american_option_mc.py` — Longstaff-Schwartz MC American put.
   - `double_barrier_analytic.py` — Ikeda-Kunitomo double-barrier.
3. **`docs/migration/phase6-completion.md`** + per-cluster lean designs (L6-A/B/C).
4. **CLAUDE.md** + **README.md** updates with final-state numbers + sample-program links.
5. **`pquantlib-final` tag** with comprehensive commit message + push.

## Expected outcomes

| Cluster | Classes | Tests delta (est.) |
|---|---|---|
| L6-A LongstaffSchwartz American MC | ~5 | +25 |
| L6-B BatesEngine | ~3 | +10 |
| L6-C DoubleBarrier + engine | ~4 | +20 |
| **Total Phase 6 code** | **~12** | **~55 → 1938/0/0** |
| Sample programs | 5 | (smoke tests only) |

# Phase 10 — Vol surface tail + Gaussian1d short-rate + ZABR / interpolator tail (executable plan)

**Goal:** Land Phase 10's ~15 must-port classes on `main`, behind tag `pquantlib-phase10-complete`. Closes 3 carve-out clusters (residual smile sections + OptionletStripper2; Gaussian1d short-rate model + GSR + Gaussian1d swaption vol; Hyman/Chebyshev/MultiCubic interpolators + ABCD + ZABR family).

**Predecessor:** `pquantlib-phase9-complete` @ `37e67e0` — 2464/0/0.

**Date:** 2026-05-29.

**Topology:** 3-parallel-no-pilot (each cluster is an independent extension of existing layers).

---

## Task 0 — Spawn 3 worktrees

```bash
for w in A B C; do
  git worktree add -b phase10-$w ../pquantlib-phase10-$w main
  cd ../pquantlib-phase10-$w && uv sync
done
```

---

## L10-A — Vol surface tail (~6 classes)

KahaleSmileSection + AtmSmileSection + AtmAdjustedSmileSection + SabrInterpolatedSmileSection + OptionletStripper2 + Halton multi-start for SabrInterpolation.

Target +30 tests.

## L10-B — Gaussian1d short-rate cluster (~3 classes)

Gaussian1dModel abstract + GSR + Gaussian1dSwaptionVolatility (MarkovFunctional + Gaussian1d engines deferred).

Target +20 tests.

## L10-C — Interpolator tail + ZABR (~6 classes)

HymanFilteredCubic + ChebyshevInterpolation + MultiCubicSpline + AbcdInterpolation + zabr_volatility + ZabrSmileSection (+ optionally ZabrInterpolatedSmileSection).

Target +30 tests.

---

## Task 1 — Dispatch L10-A/B/C in parallel

3 Agent calls in a single message.

## Task 2 — Merge + tag

Standard non-FF merges + CMakeLists conflict resolution. Tag `pquantlib-phase10-complete`. Push.

## Task 3 — Phase 10 doc sweep

8-step sweep per `feedback-phase-doc-sweep` memory:
- `phase10-design.md` status → CLOSED + completion link
- `phase10-completion.md` (new) per-cluster table + divergences
- `CLAUDE.md` headline state + L10 bullet
- `README.md` badges + new migration-status row
- `docs/migration/README.md` Phase 10 closed-phase section above Phase 9
- `docs/carve-outs.md` mark Phase-9 vol-tail + specialty-short-rate + L1-E + ZABR carve-outs CLOSED; refresh project-totals
- memory `phase_status.md` + `MEMORY.md` updated

No A6 pause per user directive.

## Expected outcomes

| Cluster | Classes | Tests delta (est.) |
|---|---|---|
| L10-A vol surface tail | ~6 | +30 |
| L10-B Gaussian1d short-rate | ~3 | +20 |
| L10-C interpolator tail + ZABR | ~6 | +30 |
| **Total** | **~15** | **~80 → ~2544/0/0** |

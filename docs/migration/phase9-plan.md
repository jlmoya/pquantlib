# Phase 9 — Cubic/Bicubic + Post-L8 ergonomics + SABR cube (executable plan)

**Goal:** Land Phase 9's ~22 must-port classes on `main`, behind tag `pquantlib-phase9-complete`. Closes 3 carve-outs (L1-E cubic-family interpolators; post-L8 credit/bootstrap ergonomics; Phase-8 SABR cube + smile-section family).

**Predecessor:** `pquantlib-phase8-complete` @ `dec05fb` — 2303/0/0.

**Date:** 2026-05-28.

**Topology:** L9-A pilot → L9-B + L9-C parallel (L9-A is a hard dep of L9-C `InterpolatedSmileSection`).

---

## Task 0 — Spawn L9-A pilot worktree

```bash
git worktree add -b phase9-A ../pquantlib-phase9-A main
cd ../pquantlib-phase9-A && uv sync
```

---

## L9-A — Cubic + Bicubic spline interpolators (pilot, ~5 classes)

Closes the L1-E carve-out. scipy delegation. Opt-in upgrade for L8-C surfaces.

Target +20-25 tests.

## Task 1 — Dispatch L9-A pilot subagent

Single Agent call.

## Task 2 — Merge L9-A + verify triad

`git merge --no-ff phase9-A` + triad green.

---

## Task 3 — Spawn L9-B + L9-C parallel worktrees

```bash
for w in B C; do
  git worktree add -b phase9-$w ../pquantlib-phase9-$w main
  cd ../pquantlib-phase9-$w && uv sync
done
```

(Both off the post-L9-A main tip.)

## L9-B — Post-L8 ergonomic follow-ups (~7-8 classes)

IsdaCdsEngine + MakeCDS + implied_hazard_rate + conventional_spread + PiecewiseYieldCurve + yield traits (Discount/ZeroYield/ForwardRate) + PiecewiseDefaultCurve bootstrap wiring.

Target +30-35 tests.

## L9-C — SABR swaption smile cube (~10 classes)

SmileSection family (5: abstract + Flat + Interpolated + Sabr + Spreaded) + SABR math (3: formula + interpolation + normal-vol formula) + SwaptionVolCube abstract + SabrSwaptionVolatilityCube + InterpolatedSwaptionVolatilityCube.

Target +40-50 tests.

## Task 4 — Dispatch L9-B + L9-C in parallel

2 Agent calls in a single message.

## Task 5 — Merge L9-B + L9-C + tag

Standard non-FF merges + CMakeLists conflict resolution. Tag `pquantlib-phase9-complete`. Push.

---

## Expected outcomes

| Cluster | Classes | Tests delta (est.) |
|---|---|---|
| L9-A cubic/bicubic | ~5 | +20-25 |
| L9-B post-L8 ergonomics | ~7-8 | +30-35 |
| L9-C SABR cube | ~10 | +40-50 |
| **Total** | **~22** | **~90-110 → ~2393-2413/0/0** |

## Task 6 — Phase 9 doc sweep

No A6 pause per user directive. Sweep:
- `phase9-design.md` status → CLOSED
- `phase9-completion.md` (new) per-cluster contribution table + divergences
- `CLAUDE.md` headline state + L9 bullet
- `README.md` badges + migration-status table row
- `docs/migration/README.md` Phase 9 section
- `docs/carve-outs.md` mark L1-E cubic + post-L8 + Phase-8 SABR carve-outs CLOSED
- memory `phase_status.md` + `MEMORY.md` updated

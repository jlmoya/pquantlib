# Phase 8 — Piecewise inflation + credit + capfloor-vol surfaces (executable plan)

**Goal:** Land Phase 8's ~50 must-port classes on `main`, behind tag `pquantlib-phase8-complete`. Closes 3 carve-outs (L7-Bb piecewise inflation; full credit cluster; capfloor/optionlet/swaption vol surfaces).

**Predecessor:** `pquantlib-phase7-complete` @ `b7ac1a6` — 2109/0/0.

**Date:** 2026-05-28.

---

## Task 0 — Spawn 3 worktrees (no pilot)

```bash
for w in A B C; do
  git worktree add -b phase8-$w ../pquantlib-phase8-$w main
  cd ../pquantlib-phase8-$w && uv sync
done
```

---

## L8-A — Piecewise inflation curves + helpers (~8 classes)

Closes the L7-Bb follow-up from Phase 7.

Target +25 tests.

## L8-B — Credit cluster (~18 classes)

DefaultProbabilityTermStructure abstracts + concretes + traits + PiecewiseDefaultCurve + helpers + CDS instrument + Claim + 3 CDS engines (MidPoint + Integral + ISDA).

Target +60 tests.

## L8-C — Capfloor / optionlet / swaption vol surfaces (~18 classes)

CapFloorTermVolatilityStructure family + OptionletVolatilityStructure family + OptionletStripper1 + SwaptionVolatilityStructure family + SwaptionVolatilityMatrix.

Target +60 tests.

---

## Task 1 — Spawn + dispatch

3 parallel Agent calls.

## Task 2 — Merge + tag

Standard non-FF merges + CMakeLists conflict resolution. Tag `pquantlib-phase8-complete`.

## Expected outcomes

| Cluster | Classes | Tests delta (est.) |
|---|---|---|
| L8-A piecewise inflation | ~8 | +25 |
| L8-B credit | ~18 | +60 |
| L8-C capfloor/swaption vol | ~18 | +60 |
| **Total** | **~44** | **~145 → 2254/0/0** |

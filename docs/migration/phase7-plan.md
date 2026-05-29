# Phase 7 — Inflation cluster (executable plan)

**Status:** **closed** — tagged `pquantlib-phase7-complete` @ `3a7228e` on 2026-05-28. **2109/0/0** tests. See [`phase7-completion.md`](phase7-completion.md).

**Goal:** Land ~30 must-port inflation classes on `main`, behind tag `pquantlib-phase7-complete`. Closes the inflation Tier-1 carve-out from `docs/carve-outs.md`.

**Predecessor:** `pquantlib-final` @ `45f4668` — 1958/0/0.

**Date:** 2026-05-28.

---

## Task 0 — Spawn pilot worktree

```bash
git worktree add -b phase7-A ../pquantlib-phase7-A main
cd ../pquantlib-phase7-A && uv sync
```

---

## L7-A pilot (sequential, ~10 classes)

### Stages
- **S0** Probe scaffolding (EUHICP / UKRPI / USCPI defaults + Seasonality eval).
- **S1** `InflationIndex` + `ZeroInflationIndex` + `YoYInflationIndex` abstracts.
- **S2** 5 region concretes: EUHICP, FRHICP, UKRPI, UKHICP, USCPI.
- **S3** `InflationTermStructure` + `ZeroInflationTermStructure` + `YoYInflationTermStructure` abstracts.
- **S4** `Seasonality` + `MultiplicativePriceSeasonality`.
- **S5** Cross-cluster Protocols (`InflationIndexProtocol`, `InflationTermStructureProtocol`).

Target: +50 tests.

### L7-A closure
FF-merge, tag `pquantlib-phase7-l7-A-complete`, push.

---

## L7-B/C/D parallel (each ~8-10 classes)

### L7-B: curves + bootstrap
InterpolatedZero/YoY inflation curves + Piecewise variants + ZeroInflationTraits + YoYInflationTraits + ZeroCouponInflationSwapHelper + YearOnYearInflationSwapHelper.

Target: +35 tests.

### L7-C: cashflows + pricers
InflationCoupon abstract + ZeroInflationCashFlow + CPICoupon + YoYInflationCoupon + CappedFlooredInflationCoupon family + InflationCouponPricer abstract + CPICouponPricer + BlackCPICouponPricer + YoYInflationCouponPricer + BlackYoYInflationCouponPricer.

Target: +30 tests.

### L7-D: instruments + vol surfaces + engines
CPISwap + ZeroCouponInflationSwap + YearOnYearInflationSwap + CPICapFloor + YoYInflationCapFloor + CPIVolatilitySurface abstract + ConstantCPIVolatility + YoYOptionletVolatilitySurface abstract + ConstantYoYOptionletVolatility + YoY*BlackCapFloorEngine variants.

Target: +35 tests.

---

## Task 1 — Spawn worktrees + dispatch

3 parallel subagents off `pquantlib-phase7-l7-A-complete`. Standard pattern.

## Task 2 — Merge + tag

```bash
git merge --no-ff phase7-B -m "merge: L7-B (inflation curves + bootstrap helpers)"
git merge --no-ff phase7-C -m "merge: L7-C (inflation cashflows + pricers)"
git merge --no-ff phase7-D -m "merge: L7-D (inflation instruments + vol surfaces + engines)"
```

Tag `pquantlib-phase7-complete`, push.

## Expected outcomes

| Cluster | Classes | Tests delta (est.) |
|---|---|---|
| L7-A pilot | ~10 | +50 |
| L7-B curves + bootstrap | ~8 | +35 |
| L7-C cashflows + pricers | ~8 | +30 |
| L7-D instruments + vol + engines | ~10 | +35 |
| **Total Phase 7** | **~36** | **~150 → 2108/0/0 cumulative** |

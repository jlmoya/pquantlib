# Phase 7 (inflation cluster) — completion

**Date closed:** 2026-05-28
**Tag:** [`pquantlib-phase7-complete`](../../README.md#migration-status) @ `3a7228e`
**Predecessor:** `pquantlib-final` @ `45f4668`
**Test count:** 1958 → **2109/0/0** (+151). pyright + ruff clean.
**Design spec:** [`phase7-design.md`](phase7-design.md). **Plan:** [`phase7-plan.md`](phase7-plan.md).

## Cluster contribution table

| Cluster | Mode | Commits | Tests added | Coverage |
|---|---|---|---|---|
| **L7-A pilot** | sequential | 6 | +60 | InflationIndex hierarchy + 5 region concretes (EUHICP/FRHICP/UKRPI/UKHICP/USCPI) + termstructure abstracts + Seasonality + 2 cross-cluster Protocols |
| **L7-B partial** | parallel (socket-dropped) | 2 | +12 | Interpolated{Zero,YoY}InflationCurve only; Piecewise + traits + helpers deferred |
| **L7-C** | parallel | 2 | +44 | InflationCoupon hierarchy + CPICoupon + YoYInflationCoupon + CappedFlored variants + InflationCouponPricer + Black variants |
| **L7-D** | parallel | 4 | +35 | ZeroCouponInflationSwap + YearOnYearInflationSwap + CPISwap + CPICapFloor + YoYInflationCapFloor + Cap/Floor/Collar + ConstantCPI/YoY vol surfaces + 3 YoY engines (Bachelier/Black/UnitDisplacedBlack) |
| **Total** | | **~14** | **+151** | **~32 classes** |

## What landed

The vanilla inflation pricing path is functional end-to-end through analytic engines:

- **Inflation indexes**: EUHICP, FRHICP, UKRPI, UKHICP, USCPI + YoY siblings + Region enum + InterpolationType (AsIndex/Flat/Linear) + lagged_fixing / lagged_yoy_rate helpers.
- **Inflation termstructures**: abstracts (zero + YoY) + Interpolated curves + Seasonality + MultiplicativePriceSeasonality.
- **Inflation cashflows**: InflationCoupon abstract + ZeroInflationCashFlow + CPICoupon + CPICashFlow + YoYInflationCoupon + CappedFloredInflationCoupon family + InflationCouponPricer + CPICouponPricer + Black variants + Bachelier variant.
- **Inflation instruments**: ZeroCouponInflationSwap, YearOnYearInflationSwap, CPISwap (shell), CPICapFloor (shell), YoYInflationCapFloor + Cap / Floor / Collar.
- **Inflation vol surfaces**: CPIVolatilitySurface + YoYOptionletVolatilitySurface abstracts + ConstantCPIVolatility + ConstantYoYOptionletVolatility.
- **Inflation engines**: YoYInflationCapFloorEngine base + Black / Bachelier / UnitDisplacedBlack analytic concretes.

## Merge reconciliations

1. **L7-B subagent socket dropped mid-cluster.** Got 2 commits (probe + interpolated curves). Piecewise{Zero,YoY}InflationCurve + ZeroInflationTraits + YoYInflationTraits + ZeroCouponInflationSwapHelper + YearOnYearInflationSwapHelper carved out to a follow-up cluster (L7-Bb).
2. **L7-C and L7-D both created `indexes/inflation/cpi.py` independently**. L7-C's strict InterpolationType + lagged_fixing / lagged_yoy_rate retained; L7-D's `is_interpolated` + `effective_interpolation_type` helpers grafted in additively.
3. **L7-B tests needed `cast()` annotations** for `ref["nodes"] / ref["samples"]` dict-accessor types (pyright strict). Single align commit landed.
4. **CMakeLists.txt**: 3 cluster entries stacked.
5. **Phase 6 sample programs removed** as part of Phase 7 prep — they were speculative API usage I introduced in the Phase 6 doc-sweep without testing (54 pyright errors). Will rebuild as proper smoke-tested samples later.

## Documented divergences

- **`Region` is an IntEnum** rather than C++'s class-per-region hierarchy. Payload mirrors `ql/indexes/region.cpp` static `Data` tuples.
- **`ZeroInflationIndex.maturity_date()`** is a Python-side convenience absent on C++ — returns `inflationPeriod(d, freq).second`.
- **`InflationTermStructure.observation_lag()` and `.nominal_term_structure()`** are `[[deprecated]]` / dropped in v1.42.1 — re-instated per L7-A spec because downstream L7-C cashflows + L7-D engines still need them.
- **CPI engines fetch forward via `index.yoy_inflation_term_structure().yoy_rate(fixing_date)`** rather than C++'s template-typed `Handle<YoYInflationTermStructure>`.
- **CPISwap shell requires pre-built legs** — CPILeg / IborLeg builders defer.

## Carve-outs (deferred to follow-up clusters)

### L7-Bb follow-up
- `PiecewiseZeroInflationCurve` + `PiecewiseYoYInflationCurve`.
- `ZeroInflationTraits` + `YoYInflationTraits` (bootstrap traits).
- `ZeroCouponInflationSwapHelper` + `YearOnYearInflationSwapHelper` (rate helpers).

### Other inflation deferrals
- AUCPI + ZACPI region indexes (least-used).
- CPIBond instrument.
- CPILeg / IborLeg builders for CPISwap.
- `MakeYoYInflationCapFloor` factory.
- Inflation cap/floor multi-curve / cross-currency variants.
- Inflation forward-measure processes (experimental SABR-on-inflation).
- `experimental/inflation/*` (additional inflation experimentals).

## Lessons learned

- **Subagent socket failures are a real risk on long-running clusters.** L7-B had 4 commits planned + 4 stages of work; ~40 min in the socket dropped. Recovery pattern: commit partial work as a sub-deliverable + carve out remaining scope. The 18 tests L7-B did land are net-positive value.
- **Cross-cluster `cpi.py` creation by two subagents** is a recurring pattern (Phase 2 Compounding / InterestRate; Phase 3 BondForwardPosition / Position; Phase 7 InterpolationType helpers). Mitigation for Phase 8+: pre-port any shared helper module in the pilot.
- **Phase 6 sample programs** were speculative without API verification — they introduced 54 pyright errors hidden until L7-A's subagent ran the full triad and flagged them. Future closure-tooling commits should be triad-validated before push.

## What's next

- **L7-Bb follow-up** to close the piecewise bootstrap (small cluster, ~8 classes).
- **Phase 8 candidates from the carve-out catalog** (credit cluster, MarketModels, multi-asset MC, etc.).

`pquantlib-final` tag still represents the canonical "planned migration complete" state; Phase 7 is an opt-in extension that ports the highest-ROI specialty domain (inflation).

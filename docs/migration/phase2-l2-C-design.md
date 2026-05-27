# Phase 2 L2-C — Indexes + rate helpers

**Date:** 2026-05-26
**Status:** **closed** — merged into `main` via `e015cd7 merge: L2-C`; tagged as part of `pquantlib-phase2-complete` @ `b5d2519`. Final test delta: **+77** (4 commits).
**Predecessor:** `pquantlib-phase2-l2-A-complete` @ `4ace1f0`
**C++ ground truth:** QuantLib v1.42.1 @ `099987f0`
**Style:** lean — leans on [`phase2-design.md`](phase2-design.md) for ground rules.

## What landed

### Index hierarchy (4 abstracts)

- `pquantlib.indexes.interest_rate_index.InterestRateIndex`
- `pquantlib.indexes.ibor_index.IborIndex`
- `pquantlib.indexes.overnight_index.OvernightIndex`
- `pquantlib.indexes.swap_index.SwapIndex`

### 8 IBOR / overnight concretes (~10 modules including helpers)

`Euribor`, `Libor` base + `USDLibor` + `GBPLibor` + `DailyTenor*` helpers, `Eonia`, `Sofr`, `Sonia`, `FedFunds`, `Estr`.

### 2 swap indexes

`EuriborSwapIsdaFixA`, `UsdLiborSwapIsdaFixAm`.

### 7 rate helpers

`DepositRateHelper`, `FraRateHelper`, `FuturesRateHelper`, `SwapRateHelper`, `OISRateHelper`, `BondHelper`, `FxSwapRateHelper`. All under `pquantlib.termstructures.yield_.*_rate_helper`.

## Documented divergences

- `Euribor` ported as a **single multi-tenor class** with classmethod shortcuts (`Euribor.three_months()`) rather than per-tenor C++ subclasses (`Euribor3M`, etc.). Same for `USDLibor`, `GBPLibor`. Tenors stored as `Period`.
- Rate helpers use **explicit `initialize_dates(evaluation_date)`** argument rather than registering with the (still-absent) `Settings.evaluation_date` observable.
- Tests use an inline `FlatForwardMock` satisfying `YieldTermStructureProtocol` rather than depending on L2-B's not-yet-on-branch concrete curves — proves the Protocol design.

## Carve-outs (deferred to L3)

- **`SwapIndex.forecast_fixing()` + `underlying_swap()`** — need `VanillaSwap` from L3.
- **`SwapRateHelper.implied_quote()`** — needs `MakeVanillaSwap` + `DiscountingSwapEngine`.
- **`OISRateHelper.implied_quote()`** — needs `OvernightIndexedSwap` + `MakeOIS`.
- **`BondHelper.implied_quote()`** — needs `Bond` + `DiscountingBondEngine`.
- **`FraRateHelper(useIndexedCoupon=True)`** — needs L2-D `IborCoupon` wired through.
- **Region-specialty libors** beyond USDLibor/GBPLibor (CHFLibor, JPYLibor, AUDLibor, etc.).
- **Swap-index variants** beyond ISDA-Fix A / AM (ISDA-Fix B, IFR, P.M., CHF/EUR/GBP/JPY).

# Phase 3 L3-C — Swaps + L2-C carry-over closures

**Date:** 2026-05-27
**Status:** **closed** — merged into `main` via `2a64c3e merge: L3-C`; tagged as part of `pquantlib-phase3-complete` @ `aacc2c2`. Final test delta: **+41** (vs +35 target). 4 commits.
**Predecessor:** `pquantlib-phase3-l3-A-complete` @ `e72bcdf`
**C++ ground truth:** QuantLib v1.42.1 @ `099987f0`
**Style:** lean — leans on [`phase3-design.md`](phase3-design.md) for ground rules.

## What landed

### New ports
- `pquantlib.instruments.swap.Swap` (abstract).
- `pquantlib.instruments.fixed_vs_floating_swap.FixedVsFloatingSwap` (abstract intermediate).
- `pquantlib.instruments.vanilla_swap.VanillaSwap` (fixed-vs-Ibor).
- `pquantlib.instruments.overnight_indexed_swap.OvernightIndexedSwap`.
- `pquantlib.instruments.zero_coupon_swap.ZeroCouponSwap` (single-payment fixed-vs-IBOR with compound IBOR float leg).
- `pquantlib.instruments.make_vanilla_swap.make_vanilla_swap(...)` (free-function factory).
- `pquantlib.instruments.make_ois.make_ois(...)` (free-function factory).
- `pquantlib.pricingengines.swap.discounting_swap_engine.DiscountingSwapEngine` (NPV per leg + fair_rate via NPV-balancing solver + fair_spread + BPS).

### L2-C carry-overs closed
- `SwapRateHelper.implied_quote()` — was raising; now implements via `make_vanilla_swap` + `DiscountingSwapEngine`.
- `OISRateHelper.implied_quote()` — was raising; now implements via `make_ois` + `DiscountingSwapEngine`.
- `SwapIndex.forecast_fixing()` + `SwapIndex.underlying_swap()` — was raising; now closed.

## Documented divergences

- **`VanillaSwap` auto-attaches `IborCouponPricer()`** to the floating leg (C++ relies on the global `PricerCleaner` registry).
- **`IborCouponPricer` now implements C++ par-coupons forecast** (C++ default `Settings::usingAtParCoupons=true`). Fixed a 1.8e-5 rel drift in 5y swap NPV caught by L3-C testing.
- **`IborIndexProtocol`** gained `maturity_date(value_date)` (additive).
- **`YieldTermStructure.{discount,zero_rate,forward_rate}` parameter names renamed `arg`→`t`** to align with `YieldTermStructureProtocol`. (Caused a post-merge alignment when L3-E's Protocol docstring still referenced `arg` — caught by pyright; fixed in a single `align(...)` commit.)
- **`Leg` (list) vs `LegInput` (Sequence covariant)** for input-boundary widening.
- **`MakeOIS` settlement-days default switches on `index.name()`** (SONIA/CORRA) rather than C++ `dynamic_pointer_cast`.

## Carve-outs

- **`MultipleResetsCoupon` + `CompoundingMultipleResetsPricer`** — `ZeroCouponSwap` uses a private `_CompoundedIborCashFlow` approximation in their place.
- **`RateAveraging.Simple`** for OIS (arithmetic-averaged variant).
- **`lookback_days` / `lockout_days` / `apply_observation_shift` / `telescopic_value_dates` / `payment_lag`** plumbing through OIS leg builder.
- **Visitor-based Swap dispatch** (C++ uses Visitor; Python uses isinstance).
- **Swaption** (Phase 4 — needs short-rate models).
- **Specialty swaps** (BMA / Float/Float / NonStandard / MultipleResets / EquityTotalReturn) — Phase 5.

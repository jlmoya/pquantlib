# Phase 2 L2-D — Cashflows

**Date:** 2026-05-26
**Status:** **closed** — merged into `main` via `a9f23b0 merge: L2-D`; tagged as part of `pquantlib-phase2-complete` @ `b5d2519`. Final test delta: **+50** (post-dedup of L2-B-duplicated `InterestRate`/`Compounding`). 6 commits.
**Predecessor:** `pquantlib-phase2-l2-A-complete` @ `4ace1f0`
**C++ ground truth:** QuantLib v1.42.1 @ `099987f0`
**Style:** lean — leans on [`phase2-design.md`](phase2-design.md) for ground rules.

## What landed

### Cashflow hierarchy
- `pquantlib.cashflows.cash_flow.CashFlow` (abstract).
- `pquantlib.cashflows.simple_cash_flow.SimpleCashFlow` (+ `Redemption` + `AmortizingPayment`).
- `pquantlib.cashflows.coupon.Coupon` (abstract).

### Fixed
- `pquantlib.cashflows.fixed_rate_coupon.FixedRateCoupon`.
- `pquantlib.cashflows.fixed_rate_leg.fixed_rate_leg` (free function).

### Floating
- `pquantlib.cashflows.floating_rate_coupon.FloatingRateCoupon` (abstract).
- `pquantlib.cashflows.ibor_coupon.IborCoupon`.
- `pquantlib.cashflows.overnight_indexed_coupon.OvernightIndexedCoupon`.
- `pquantlib.cashflows.ibor_leg.ibor_leg`.
- `pquantlib.cashflows.overnight_leg.overnight_leg`.

### Pricers
- `pquantlib.cashflows.coupon_pricer.CouponPricer` (abstract) + `IborCouponPricer` + `BlackIborCouponPricer` + `CompoundingOvernightIndexedCouponPricer` + `set_coupon_pricer`.

### Aggregator + duration
- `pquantlib.cashflows.duration.Duration` (IntEnum: Simple/Macaulay/Modified).
- `pquantlib.cashflows.cash_flows.CashFlows` (static methods: NPV / IRR / bps / simple/macaulay/modified duration / convexity).

## Documented divergences

- **C++ Builder pattern → free-function leg generators**: `FixedRateLeg(...).withNotionals(...).withSchedule(...)` → `fixed_rate_leg(schedule, notionals, rates, day_counter, ...)`. Same for `ibor_leg` / `overnight_leg`.
- **`Settings.evaluationDate()` global state NOT ported** — callers pass `settlement_date` explicitly; `CashFlow.has_occurred` returns `False` for `ref_date=None`.
- **`Visitor.accept()` cashflow dispatch NOT ported** — `bps` reimplemented without visitor.
- **`LazyObject` / `performCalculations` deferred-evaluation pattern replaced by eager Python evaluation** — `rate()` recomputes each call via pricer.
- **`BlackIborCouponPricer` cap/floor methods raise** — they need `OptionletVolatilityStructure` (deferred). Plain swaplet behaves identically to `IborCouponPricer`.
- **`OvernightIndexedCoupon` simplified** to `RateAveraging.Compound` with no lookback/lockout/observation-shift/compound-spread-daily/telescopic-value-dates.

## Merge reconciliation

L2-D originally ported `Compounding` + `InterestRate` independently to `pquantlib.cashflows.{compounding,interest_rate}`. L2-B had already placed them at `pquantlib.time.compounding` + `pquantlib.interest_rate` (matching C++ root placement). Reconciled at merge by:

- Removing L2-D's duplicate `compounding.py` + `interest_rate.py`.
- Rewriting L2-D's imports + attribute accesses (`.rate` → `.rate()`, etc.) to use L2-B's method-call API.
- Renaming `_between` method calls to `_dates` (L2-B's naming).
- Deleting L2-D's duplicate `test_interest_rate.py` (L2-B's covers the same surface).

## Carve-outs

- **`IborCoupon::Settings`** global toggle (par vs indexed coupons).
- **`CashFlows::atmRate` / `zSpread` / `basisPointValue` / `yieldValueBasisPoint`** + start/maturity/previous/next-cashflow helpers (visitor-driven + curve-dependent).
- **`CmsCouponPricer` + `MeanRevertingPricer` hierarchy**.
- **`OptionletVolatilityStructure`** cap/floor pricing surface.
- **`RateAveraging.Simple`** (arithmetic-averaged OIS pricer).
- **`AverageBmaCoupon` / `CmsCoupon` / `DigitalCoupon` / `CappedFlooredCoupon` / `CappedFlooredOvernightIndexedCoupon`** + inflation coupons + equity cashflow + multiple-resets coupon.

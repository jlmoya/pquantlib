# Phase 3 L3-B — Bonds

**Date:** 2026-05-27
**Status:** **closed** — merged into `main` via `4b86...` merge; tagged as part of `pquantlib-phase3-complete` @ `aacc2c2`. Final test delta: **+81** (1037 → 1118 pre-other-merges, vs +30 target). 6 commits.
**Predecessor:** `pquantlib-phase3-l3-A-complete` @ `e72bcdf`
**C++ ground truth:** QuantLib v1.42.1 @ `099987f0`
**Style:** lean — leans on [`phase3-design.md`](phase3-design.md) for ground rules.

## What landed

- `pquantlib.instruments.bond.Bond` (abstract) + `BondArguments` + `BondResults` + `BondPrice` (value object).
- `pquantlib.instruments.bonds.fixed_rate_bond.FixedRateBond`.
- `pquantlib.instruments.bonds.zero_coupon_bond.ZeroCouponBond`.
- `pquantlib.instruments.bonds.floating_rate_bond.FloatingRateBond`.
- `pquantlib.instruments.bonds.amortizing_fixed_rate_bond.AmortizingFixedRateBond`.
- `pquantlib.instruments.callability.Callability` + `CallabilitySchedule` (data carriers).
- `pquantlib.instruments.bond_forward.BondForward` (instrument; collapsed Forward hierarchy formula inline).
- `pquantlib.pricingengines.bond.discounting_bond_engine.DiscountingBondEngine`.
- **Extended `pquantlib.cashflows.cash_flows.CashFlows`** with leg-walking helpers (`next_cashflow_date`, `previous_cashflow_date`, accrued helpers) — prerequisite for Bond.

## Documented divergences

- **`Bond.yield_rate`** (Python `yield` is reserved). Brent solver at LOOSE tier (1e-8); C++ uses NewtonSafe with derivative, harder in Python without templated function-with-derivative.
- **`BondPrice` carries `None`** not `Null<Real>()` sentinel.
- **`DiscountingBondEngine` takes concrete `YieldTermStructure`** (not `Handle<YieldTermStructure>`); uses `typing.cast` to bridge to `YieldTermStructureProtocol` on `CashFlows.npv_curve` callsites where richer signature isn't structurally assignable.
- **`Bond` observes `ObservableSettings().evaluation_date`** (now-wired in L3-A) — same semantics as C++ `Settings::evaluationDate()`.
- **`BondForward` collapses the Forward base hierarchy** — inlines the formula + a minimal `BondForwardPosition` enum. L3-E's `Position` enum can replace this later (no current conflict).

## Carve-outs

- Cap/floor caps/floors on `FloatingRateBond` (need `OptionletVolatilityStructure` from L4).
- `ex_coupon_period` threading into `fixed_rate_leg` / `ibor_leg` (L2-D builder carve-outs).
- `first_period_day_counter` / `payment_lag` on FixedRateBond / AmortizingFixedRateBond.
- BondForward engine (uses no engine; computes NPV inline).
- Callability visitor.accept dispatch.
- `BondFunctions` namespace (Bond delegates to `CashFlows` + `InterestRate` directly).
- Specialty bonds (BTP, CmsRateBond, ConvertibleBond, CpiBond, AmortizingCmsRateBond, AmortizingFloatingRateBond) — on-demand.

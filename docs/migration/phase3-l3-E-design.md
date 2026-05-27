# Phase 3 L3-E — Forwards + FRAs + L2-C FraRateHelper closure

**Date:** 2026-05-27
**Status:** **closed** — merged into `main` via `6267450 merge: L3-E`; tagged as part of `pquantlib-phase3-complete` @ `aacc2c2`. Final test delta: **+28** (vs +20 target). 4 commits.
**Predecessor:** `pquantlib-phase3-l3-A-complete` @ `e72bcdf`
**C++ ground truth:** QuantLib v1.42.1 @ `099987f0`
**Style:** lean — leans on [`phase3-design.md`](phase3-design.md) for ground rules.

## What landed

### New ports
- `pquantlib.instruments.forward.Forward` (abstract).
- `pquantlib.instruments.forward.ForwardTypePayoff` (concrete; replaces C++ `Position::Type` casting).
- `pquantlib.instruments.forward.Position` (IntEnum: Long / Short).
- `pquantlib.instruments.fx_forward.FxForward` + `FxForwardArguments` + `FxForwardResults`.
- `pquantlib.instruments.forward_rate_agreement.ForwardRateAgreement` (FRA on an Ibor index).
- `pquantlib.pricingengines.forward.discounting_fwd_engine.DiscountingFwdEngine` (port of C++ `DiscountingFxForwardEngine`).

### L2-C carry-over closed
- `FraRateHelper(useIndexedCoupon=True)` — was raising via deferred-guard; now branches to `index.fixing(fixing_date, forecast_todays_fixing=True)` when set. `initialize_dates` sets `_latest_relevant_date = index.maturity_date(earliest)` to mirror C++.

## Documented divergences

- **`Forward.settlement_date`** accepts an explicit `evaluation_date` kwarg (defaults to discount-curve `reference_date`) — PQuantLib has no global `Settings.evaluationDate` reference here.
- **`Forward.is_expired`** short-circuits to False without a discount curve.
- **`FxForward.setup_arguments`** infers eval date from the engine's source-currency curve `reference_date`.
- **`FxForward.is_expired`** returns False unconditionally (deferred until Settings.evaluation_date integration in pricing engine).
- **`ForwardRateAgreement`** collapses C++'s two constructors into one Python signature: `maturity_date=None` triggers the indexed-coupon branch.
- **`LazyObject.calculate`** modified to set `_calculated=True` BEFORE invoking `_perform_calculations` (C++ parity) with rollback on exception — supports bootstrap recursion `Forward.forward_value → calculate → perform_calculations → forward_value`.
- **`YieldTermStructureProtocol` slimmed** to discount-only (`zero_rate` / `forward_rate` removed; the richer signatures on concrete `YieldTermStructure` were not structurally assignable to the Protocol). Tracked as a separate `align(termstructures): slim YieldTermStructureProtocol to discount-only` commit. (See [`phase3-completion.md`](phase3-completion.md) for the post-merge parameter-name alignment that this triggered.)

## Carve-outs

None — all 4 scope classes shipped, plus the L2-C carry-over closed.

## Position enum (and L3-B alignment)

L3-B independently inlined a `BondForwardPosition` enum into `BondForward` while L3-E was developing the canonical `Position`. No merge conflict — L3-B's inline values are compatible. Future BondForward refactor can adopt `pquantlib.instruments.forward.Position` without breaking changes.

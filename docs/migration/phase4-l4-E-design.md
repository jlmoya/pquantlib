# Phase 4 L4-E — Swaption + CapFloor instruments + analytic engines + calibration helpers

**Date:** 2026-05-27
**Status:** **closed** — merged via `fab5a0d merge: L4-E`; tagged as `pquantlib-phase4-complete` @ `fab5a0d`. Final test delta: **+43**. 5 commits.
**Predecessor:** `pquantlib-phase4-l4-A-complete` @ `657b707`
**Style:** lean — leans on [`phase4-design.md`](phase4-design.md).

## What landed

### Instruments (closes Phase 3 carve-outs)
- `pquantlib.instruments.swaption.Swaption(swap, exercise, settlement_type, settlement_method)`.
- `pquantlib.instruments.cap_floor.CapFloor` abstract + `Cap` + `Floor` concretes.

### Calibration helpers
- `pquantlib.models.swaption_helper.SwaptionHelper` — `BlackCalibrationHelper` concrete for swaptions.
- `pquantlib.models.cap_helper.CapHelper` — concrete for caps.

### Analytic engines
- `pquantlib.pricingengines.swaption.black_swaption_engine.BlackSwaptionEngine(termStructure, vol)` — model-free, lognormal Black.
- `pquantlib.pricingengines.swaption.bachelier_swaption_engine.BachelierSwaptionEngine(termStructure, vol)` — model-free, normal Bachelier.
- `pquantlib.pricingengines.swaption.jamshidian_swaption_engine.JamshidianSwaptionEngine(model)` — analytic under any `OneFactorAffineModel` (HW / Vasicek / CIR / ExtendedCIR). Uses Jamshidian decomposition: Brent solve for r* → portfolio of bond options.
- `pquantlib.pricingengines.swaption.g2_swaption_engine.G2SwaptionEngine(model, range_, intervals)` — analytic G2 swaption via 1-D `SegmentIntegral` × inner Brent solve (Brigo-Mercurio §4.2).
- `pquantlib.pricingengines.capfloor.black_capfloor_engine.BlackCapFloorEngine(termStructure, vol)`.
- `pquantlib.pricingengines.capfloor.bachelier_capfloor_engine.BachelierCapFloorEngine(termStructure, vol)`.
- `pquantlib.pricingengines.capfloor.analytic_capfloor_engine.AnalyticCapFloorEngine(model)` — uses `model.discount_bond_option` for each caplet/floorlet (HW/Vasicek/CIR/ExtendedCIR).

## Documented divergences

- **`IborCouponPricer` par-coupons forecast** now implements C++ default (`Settings::usingAtParCoupons=true`). Fixed a 1.8e-5 rel drift in 5y swap NPV. Landed as a separate `align(coupon_pricer)` commit splitting coupon vs index fixing_days.
- **`G2SwaptionEngine`** uses pquantlib SegmentIntegral (1-D) + nested Brent solve rather than C++ Gauss-Hermite quadrature. LOOSE tier on output.
- **`JamshidianSwaptionEngine`** restricted to `OneFactorAffineModel` (Protocol satisfied by HW/Vasicek/CIR/ExtendedCIR at merge). LOOSE tier (Brent convergence).
- **`AnalyticCapFloorEngine`** test mock `_MiniHullWhite._B(t, T)` uses math-convention names (capital T for terminal date — matches C++ `ql/models/shortrate/onefactormodels/hullwhite.cpp`). `noqa: N802, N803` inline with parity rationale.
- **`BlackSwaptionEngine` / `BachelierSwaptionEngine`** take either a flat vol Quote or a SwaptionVolatilityStructure (the latter is a stub — full surface defers to L5 capfloor-vol cluster).

## Carve-outs

- **TreeSwaptionEngine, TreeCapFloorEngine** — needs full lattice machinery (Phase 5).
- **All MC engines** (Heston MC, G2 MC, capfloor MC) — Phase 5.
- **All FD engines** (FdHullWhiteSwaptionEngine, FdG2SwaptionEngine, etc.) — Phase 5.
- **Gaussian1d engines** (Gaussian1dSwaptionEngine, Gaussian1dCapFloorEngine, Gaussian1dFloatFloatSwaptionEngine, Gaussian1dNonstandardSwaptionEngine) — Phase 5.
- **FloatFloatSwaption, NonStandardSwaption** instruments.
- **MakeCapFloor, MakeSwaption** Builder factories — defer until consumers need them (Phase 5).
- **CmsCapFloor + CpiCapFloor + InflationCapFloor** + their engines — all-inflation defer.
- **SwaptionVolatilityStructure** surface family — capfloor-vol cluster in Phase 5.
- **CapFloorTermVolatilityStructure** + concretes.

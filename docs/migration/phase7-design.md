# Phase 7 — Inflation cluster (design)

**Date:** 2026-05-28
**Status:** drafted, awaiting ack to start
**Predecessor:** `pquantlib-final` @ `45f4668` — 1958/0/0, pyright + ruff clean
**C++ ground truth:** QuantLib v1.42.1 @ `099987f0`

## Scope

Inflation is a self-contained specialty domain deferred from Phases 2 (termstructures) + L2-D (cashflows) + L3 (instruments) + L4 (engines). It is the highest-ROI Tier 1 carve-out per `docs/carve-outs.md`.

C++ surface (~30 .hpp files):
- `ql/indexes/inflation/*` (7 region indexes + the `InflationIndex` abstract at `ql/index.hpp` is already in L2-A).
- `ql/termstructures/inflationtermstructure.hpp` (abstract).
- `ql/termstructures/inflation/*` (7 files: helpers + traits + 2 interpolated curves + 2 piecewise curves + seasonality).
- `ql/cashflows/{cpicoupon,cpicouponpricer,inflationcoupon,inflationcouponpricer,yoyinflationcoupon,zeroinflationcashflow,capflooredinflationcoupon}.hpp`.
- `ql/instruments/{cpicapfloor,cpiswap,inflationcapfloor,makeyoyinflationcapfloor,yearonyearinflationswap,zerocouponinflationswap}.hpp`.
- `ql/termstructures/volatility/inflation/{constantcpivolatility,cpivolatilitystructure,yoyinflationoptionletvolatilitystructure}.hpp`.
- `ql/pricingengines/inflation/inflationcapfloorengines.hpp` (multiple engines in one header).

Phase 7 closes when:

1. Every must-port class is ported with C++-cross-validated tests or annotated `# C++ parity:` with a deliberate divergence note.
2. `uv run pytest` + `uv run pyright` + `uv run ruff check` clean on `main`.
3. Tag `pquantlib-phase7-complete` is pushed.

## Scope (must-port subset: ~30 classes)

### L7-A pilot — foundations (sequential, ~10 classes)

- **Indexes** (under `pquantlib.indexes.inflation.*`):
  - `InflationIndex` abstract (subclass of `Index` from L2-A). Holds family_name, frequency, region, revised, interpolated, currency, availability_lag, last_fixing.
  - `ZeroInflationIndex` abstract (CPI-style single fixing).
  - `YoYInflationIndex` abstract (year-on-year ratio of two zero-inflation fixings).
- **Region concretes** (5 of 7 — drop AUCPI + ZACPI as least-used):
  - `EUHICPIndex`, `FRHICPIndex`, `UKRPIIndex`, `UKHICPIndex`, `USCPIIndex`.
- **Termstructure abstracts**:
  - `InflationTermStructure` abstract (subclass of `TermStructure` from L2-A). Holds nominal_term_structure handle, observation_lag, frequency, base_rate, seasonality.
  - `ZeroInflationTermStructure` abstract.
  - `YoYInflationTermStructure` abstract.
- **`Seasonality`** + `MultiplicativePriceSeasonality` (used by all curves).
- **Cross-cluster Protocols**: `InflationIndexProtocol`, `InflationTermStructureProtocol`.

### L7-B — curves + bootstrap (parallel, ~8 classes)

- `pquantlib.termstructures.inflation.interpolated_zero_inflation_curve.InterpolatedZeroInflationCurve`.
- `pquantlib.termstructures.inflation.interpolated_yoy_inflation_curve.InterpolatedYoYInflationCurve`.
- `pquantlib.termstructures.inflation.piecewise_zero_inflation_curve.PiecewiseZeroInflationCurve`.
- `pquantlib.termstructures.inflation.piecewise_yoy_inflation_curve.PiecewiseYoYInflationCurve`.
- `pquantlib.termstructures.inflation.inflation_traits.ZeroInflationTraits` + `YoYInflationTraits`.
- `pquantlib.termstructures.inflation.inflation_helpers.ZeroCouponInflationSwapHelper` + `YearOnYearInflationSwapHelper`.

### L7-C — cashflows + coupons (parallel, ~8 classes)

- `pquantlib.cashflows.inflation_coupon.InflationCoupon` abstract.
- `pquantlib.cashflows.zero_inflation_cashflow.ZeroInflationCashFlow`.
- `pquantlib.cashflows.cpi_coupon.CPICoupon`.
- `pquantlib.cashflows.yoy_inflation_coupon.YoYInflationCoupon`.
- `pquantlib.cashflows.capflored_inflation_coupon.{CappedFlooredInflationCoupon, CappedFlooredCPICoupon, CappedFlooredYoYInflationCoupon}`.
- `pquantlib.cashflows.inflation_coupon_pricer.InflationCouponPricer` abstract.
- `pquantlib.cashflows.cpi_coupon_pricer.CPICouponPricer` + `BlackCPICouponPricer`.
- `pquantlib.cashflows.yoy_inflation_coupon_pricer.YoYInflationCouponPricer` + `BlackYoYInflationCouponPricer`.

### L7-D — instruments + vol surfaces + engines (parallel, ~10 classes)

- **Instruments** (under `pquantlib.instruments.*`):
  - `CPISwap(payer_or_receiver, nominal, ...)`.
  - `ZeroCouponInflationSwap(type, nominal, start_date, maturity, calendar, ...)`.
  - `YearOnYearInflationSwap(type, nominal, ...)`.
  - `CPICapFloor(type, nominal, ...)`.
  - `YoYInflationCapFloor(type, ...)` + `YoYInflationCap` + `YoYInflationFloor` + `YoYInflationCollar`.
- **Volatility termstructures** (under `pquantlib.termstructures.volatility.inflation.*`):
  - `CPIVolatilitySurface` abstract.
  - `ConstantCPIVolatility`.
  - `YoYOptionletVolatilitySurface` abstract.
  - `ConstantYoYOptionletVolatility`.
- **Engines**:
  - `pquantlib.pricingengines.inflation.{YoYInflationBachelierCapFloorEngine, YoYInflationBlackCapFloorEngine, YoYInflationUnitDisplacedBlackCapFloorEngine}` — analytic Bachelier / Black / shifted-Black for YoY cap/floor.

## Carve-outs (deferred — Phase 8+ or never)

- **AUCPI + ZACPI** region indexes (least-used).
- **`ql/experimental/inflation/*`** (additional inflation experimentals).
- **InflationCapFloor multi-curve / cross-currency variants**.
- **CPI bond** (`CPIBond`) — would fit naturally but the closure of the inflation swap path already provides the discount tooling; CPIBond defers to a focused follow-up.
- **`MakeYoYInflationCapFloor`** factory — defer (free-function pattern not as commonly used for inflation).
- **Inflation forward measure processes** (used by experimental SABR-on-inflation).

## Cluster topology

Same proven pattern: **1 sequential pilot + 3 parallel via subagents**.

**L7-A** ports the abstract bases + region concretes + Seasonality + Protocols.

**L7-B/C/D** dispatch in parallel after L7-A merges:
- **L7-B** curves + helpers (depends on `InflationIndex` + `InflationTermStructure` from L7-A).
- **L7-C** cashflows + pricers (depends on `InflationIndex` from L7-A).
- **L7-D** instruments + vol surfaces + engines (depends on cashflows from L7-C + curves from L7-B at merge time — uses Protocols).

## Per-class TDD + tolerance + commits

Same as prior phases. EXACT / TIGHT (1e-14, 1e-12) / LOOSE (1e-8). LOOSE for any inflation-curve-bootstrapping convergence; TIGHT for closed-form CPI cap/floor analytic.

## Decision log

| Decision | Rationale |
|---|---|
| **Drop AUCPI + ZACPI** | Least-used region indexes; can be ported on demand following the L7-A region-concrete pattern. |
| **Drop CPIBond from this phase** | The inflation swap path provides the curve + cashflow tooling; CPIBond gets a focused follow-up cluster if needed. |
| **Drop MakeYoYInflationCapFloor** | Free-function factory pattern from L2-D (`fixed_rate_leg`, etc.) is already established; users can build inflation legs directly. |
| **Seasonality in L7-A pilot** | Used by all inflation curves; must be available before L7-B can port piecewise bootstrap. |
| **scipy.interpolate.Akima1DInterpolator** for Interpolated*InflationCurve | Already imported for AkimaCubic in Phase 5; pattern reuses. |
| **`InflationCouponPricer` analytic engines wrap `pquantlib.pricingengines.black_formula`** | Same path as L3's IborCouponPricer; consistent. |
| **Use `InflationIndexProtocol` for cross-cluster typing** | Standard Phase-X pattern; lets L7-C cashflows reference indexes structurally. |

## Plan + executable tasks

See [`phase7-plan.md`](phase7-plan.md).

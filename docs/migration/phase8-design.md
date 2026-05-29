# Phase 8 — Piecewise inflation + credit + capfloor-vol surfaces (design)

**Date:** 2026-05-28
**Status:** **CLOSED** — tagged `pquantlib-phase8-complete` @ `efdfac3` — 2303/0/0
**Predecessor:** `pquantlib-phase7-complete` @ `b7ac1a6` — 2109/0/0, pyright + ruff clean
**C++ ground truth:** QuantLib v1.42.1 @ `099987f0`
**Completion summary:** [`phase8-completion.md`](phase8-completion.md)

## Scope

Three independent specialty-domain carve-outs from `docs/carve-outs.md`, dispatched as **3 parallel clusters with no pilot** (each cluster is independent of the others — they share no abstract bases beyond what already exists on `main`).

## Scope (must-port subset: ~50 classes)

### L8-A — Piecewise inflation curves + helpers (parallel, ~8 classes)

Closes the L7-Bb follow-up carve-out from Phase 7.

- `pquantlib.termstructures.inflation.zero_inflation_traits.ZeroInflationTraits` — bootstrap traits (initial_value, guess, min_value_after, max_value_after, update_guess).
- `pquantlib.termstructures.inflation.yoy_inflation_traits.YoYInflationTraits`.
- `pquantlib.termstructures.inflation.piecewise_zero_inflation_curve.PiecewiseZeroInflationCurve(reference_date, calendar, day_counter, observation_lag, frequency, nominal_yts, instruments, base_rate=None)`.
- `pquantlib.termstructures.inflation.piecewise_yoy_inflation_curve.PiecewiseYoYInflationCurve`.
- `pquantlib.termstructures.inflation.inflation_helpers.ZeroCouponInflationSwapHelper(quote, lag, maturity, calendar, convention, day_counter, index, yTS)` — subclass of `BootstrapHelper`.
- `pquantlib.termstructures.inflation.inflation_helpers.YearOnYearInflationSwapHelper`.

### L8-B — Credit cluster (parallel, ~18 classes)

- **Term structures** (under `pquantlib.termstructures.credit.*`):
  - `default_probability_term_structure.DefaultProbabilityTermStructure` abstract (subclass of `TermStructure`).
  - `survival_probability_structure.SurvivalProbabilityStructure` abstract intermediate.
  - `hazard_rate_structure.HazardRateStructure` abstract intermediate.
  - `default_density_structure.DefaultDensityStructure` abstract intermediate.
  - `flat_hazard_rate.FlatHazardRate` concrete.
  - `interpolated_survival_probability_curve.InterpolatedSurvivalProbabilityCurve`.
  - `interpolated_hazard_rate_curve.InterpolatedHazardRateCurve`.
  - `interpolated_default_density_curve.InterpolatedDefaultDensityCurve`.
  - `probability_traits.SurvivalProbability` / `HazardRate` / `DefaultDensity` traits.
  - `piecewise_default_curve.PiecewiseDefaultCurve` (parameterized by traits).
  - `default_probability_helpers.SpreadCdsHelper` + `UpfrontCdsHelper`.
- **Instruments**:
  - `pquantlib.instruments.credit_default_swap.CreditDefaultSwap(side, notional, spread, schedule, ...)`.
  - `pquantlib.instruments.claim.Claim` + `FaceValueClaim` (recovery convention).
- **Engines** (under `pquantlib.pricingengines.credit.*`):
  - `midpoint_cds_engine.MidPointCdsEngine(default_curve, recovery_rate, discount_curve, include_settlement_date_flows)` — midpoint Riemann CDS engine.
  - `integral_cds_engine.IntegralCdsEngine` (Riemann integral; finer).
  - `isda_cds_engine.IsdaCdsEngine` (ISDA-standard fixed convention).

Defer: `MakeCDS` factory, accrual rebate.

### L8-C — Capfloor / optionlet / swaption volatility surfaces (parallel, ~18 classes)

- **CapFloor vol** (under `pquantlib.termstructures.volatility.capfloor.*`):
  - `cap_floor_term_volatility_structure.CapFloorTermVolatilityStructure` abstract.
  - `constant_capfloor_term_vol.ConstantCapFloorTermVolatility`.
  - `cap_floor_term_vol_curve.CapFloorTermVolCurve` (1-D over maturities).
  - `cap_floor_term_vol_surface.CapFloorTermVolSurface` (2-D over maturity × strike).
- **Optionlet vol** (under `pquantlib.termstructures.volatility.optionlet.*`):
  - `optionlet_volatility_structure.OptionletVolatilityStructure` abstract (subclass of `VolatilityTermStructure`).
  - `constant_optionlet_vol.ConstantOptionletVolatility`.
  - `caplet_variance_curve.CapletVarianceCurve`.
  - `stripped_optionlet_base.StrippedOptionletBase` abstract.
  - `stripped_optionlet.StrippedOptionlet` concrete container.
  - `stripped_optionlet_adapter.StrippedOptionletAdapter` — adapts a StrippedOptionletBase to OptionletVolatilityStructure.
  - `optionlet_stripper_1.OptionletStripper1(cap_floor_term_vol_surface, ibor_index, ...)` — strips caplets from cap term vols using Black/Bachelier.
  - `spreaded_optionlet_vol.SpreadedOptionletVolatility`.
- **Swaption vol** (under `pquantlib.termstructures.volatility.swaption.*`):
  - `swaption_volatility_structure.SwaptionVolatilityStructure` abstract.
  - `swaption_constant_vol.SwaptionConstantVolatility`.
  - `swaption_volatility_matrix.SwaptionVolatilityMatrix` (expiry × tenor grid).
  - `swaption_volatility_discrete.SwaptionVolatilityDiscrete` abstract.
  - `spreaded_swaption_vol.SpreadedSwaptionVolatility`.

Defer: `CmsMarket` / `CmsMarketCalibration` (CMS-specific helpers), `Gaussian1dSwaptionVolatility` (1-factor model adaptation), `InterpolatedSwaptionVolatilityCube` + `SabrSwaptionVolatilityCube` + `SwaptionVolCube{1,2}` (full SABR cube — large, deferred to a Phase 9 SABR cluster).

## Cluster topology

**No pilot needed** — each cluster is an independent extension of existing layers:
- L8-A extends Phase 7's inflation termstructures with the piecewise bootstrap pattern from L2-B.
- L8-B extends L2's termstructure abstracts to the credit domain.
- L8-C extends L2-E's volatility termstructure foundation to interest-rate vol surfaces.

3 worktrees spawn directly off `pquantlib-phase7-complete`; 3 subagents dispatch in parallel.

## Carve-outs (deferred — Phase 9+ or never)

### Inflation (L8-A leftovers)
- Inflation cap/floor multi-curve / cross-currency variants.
- Inflation forward-measure processes.

### Credit (L8-B leftovers)
- `MakeCDS` factory.
- Accrual-rebate-conventional CDS.
- Quanto CDS.
- Constant-recovery vs claim-with-recovery model variants.
- `experimental/credit/*` — CDS-on-CDS, basket CDS, CDO.

### Capfloor / swaption vol (L8-C leftovers)
- `SabrSwaptionVolatilityCube` + `InterpolatedSwaptionVolatilityCube` + `SwaptionVolCube{1,2}` — full SABR cube.
- `Gaussian1dSwaptionVolatility`.
- `CmsMarket` + `CmsMarketCalibration`.
- `OptionletStripper2` (uses caplet variance curve + spread).
- `SmileSection*` family for full swaption vol cube.

## Decision log

| Decision | Rationale |
|---|---|
| **No pilot** | Each cluster is an independent extension; no shared new abstract bases. |
| **3-parallel-no-pilot** | Lowest-overhead dispatch; pattern proven on Phase 6 (L6-A/B/C had no pilot). |
| **Defer full SABR swaption cube to a Phase 9 SABR cluster** | SABR is a Tier-2 specialty that benefits from its own focused phase including SABR vol model + variants. |
| **Defer experimental credit (CDO, basket CDS) entirely** | Out of scope for vanilla credit; users requiring these should port the specific instruments individually. |
| **PiecewiseDefaultCurve parameterized by traits** | Mirrors C++ template-on-traits; Python uses IntEnum or class-type dispatch (subagent choice). |
| **MidPoint CDS engine first; ISDA secondary** | MidPoint is the textbook Riemann CDS engine; ISDA adds calendar/convention complications that benefit from a separate focus. |

## Plan + executable tasks

See [`phase8-plan.md`](phase8-plan.md).

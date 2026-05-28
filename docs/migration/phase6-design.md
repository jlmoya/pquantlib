# Phase 6 — High-impact carve-outs + final closure (design)

**Date:** 2026-05-28
**Status:** drafted, awaiting ack to start
**Predecessor:** `pquantlib-phase5-complete` @ `d322fca` — 1883/0/0, pyright + ruff clean
**C++ ground truth:** QuantLib v1.42.1 @ `099987f0`

## Scope decision: modernization sweep deleted

Original Phase 6 plan included a "Python 3.14 modernization sweep" (PEP 695 generics, match-case, frozen+slots dataclasses, type statements). **Audit showed this work is already done** — the codebase was built from day 1 with modern idioms:

- 0 uses of `Generic[T]` (all PEP 695 `class X[T]`).
- 0 uses of `Optional[X]` (all `X | None`).
- 0 uses of `Union[X, Y]` (all `X | Y`).
- 0 uses of `List` / `Dict` / `Tuple` from typing (all builtins).
- 0 uses of bare `@dataclass` (all `@dataclass(frozen=True, slots=True)` where appropriate).

Phase 6 narrows to two pieces:

1. **High-impact carve-outs**: 3 specific closures that meaningfully extend the pricing surface.
2. **Final closure**: comprehensive carve-out doc, sample programs, `pquantlib-final` tag.

## Scope: ~12 classes across 3 parallel clusters

### L6-A — LongstaffSchwartz American MC (parallel, ~5 classes)

C++ source: `ql/methods/montecarlo/longstaffschwartzpathpricer.hpp` + `ql/methods/montecarlo/lsmbasissystem.hpp`.

Closes the American-option-via-MC gap (Phase 5 L5-C deferred LongstaffSchwartz American MC).

- `pquantlib.methods.montecarlo.lsm_basis_system.LsmBasisSystem` — polynomial basis function set (Monomial / Chebyshev / Hermite / Laguerre). Use `numpy.polynomial` for the basis evaluation.
- `pquantlib.methods.montecarlo.longstaff_schwartz_path_pricer.LongstaffSchwartzPathPricer[Path]` (PEP 695 generic) — regression-based exercise rule. Forward pass collects exercise-vs-continuation values; backward pass regresses on the basis system per exercise date.
- `pquantlib.methods.montecarlo.longstaff_schwartz_multi_path_pricer.LongstaffSchwartzMultiPathPricer` — multi-asset variant.
- `pquantlib.pricingengines.vanilla.mc_american_engine.MCAmericanEngine(process, time_steps, samples, seed, polynom_type, polynom_order, calibration_samples=2048)`.

Probe: AmericanOption put (S=36, K=40, r=6%, q=0, σ=20%, T=1, Bermudan dates every 0.04) → Longstaff-Schwartz 1998 paper reference value (4.478). LOOSE tier (MC variance + regression variance).

### L6-B — BatesEngine (parallel, ~3 classes)

C++ source: `ql/pricingengines/vanilla/batesengine.hpp`.

L4-C ported BatesProcess + BatesModel but skipped BatesEngine (the jump-aware analytic). The `add_on_term` hook in AnalyticHestonEngine is already in place; BatesEngine just adds the Merton-jump characteristic function term.

- `pquantlib.pricingengines.vanilla.bates_engine.BatesEngine(model, integration_order=144)` — extends `AnalyticHestonEngine` with the Bates jump-CF `add_on_term`.
- Tests against the Bates 1996 reference values (jumps mean-reverting log-normal).

### L6-C — DoubleBarrier instrument + engine (parallel, ~4 classes)

C++ source: `ql/instruments/doublebarrieroption.hpp` + `ql/pricingengines/barrier/analyticdoublebarrierengine.hpp`.

- `pquantlib.instruments.double_barrier_option.DoubleBarrierOption(barrier_type, barrier_lo, barrier_hi, rebate, payoff, exercise)`. `DoubleBarrierType` IntEnum (KnockIn / KnockOut / KIKO / KOKI).
- `pquantlib.pricingengines.barrier.analytic_double_barrier_engine.AnalyticDoubleBarrierEngine(process)` — Ikeda-Kunitomo 1992 closed-form series.
- `pquantlib.pricingengines.barrier.analytic_double_barrier_binary_engine.AnalyticDoubleBarrierBinaryEngine(process)` — for cash-or-nothing double-barriers.

Probe: textbook double-barrier values from Hull / Haug.

## Final closure (sequential, post-merge)

After L6-A/B/C land:

1. **`docs/carve-outs.md`** — comprehensive carve-out documentation. Per-phase + per-category list of every class deferred from C++ v1.42.1, with rationale + how to access via wrapping if applicable.
2. **`pquantlib-samples/`** package — populate with end-to-end sample programs (vanilla swap pricing, swaption calibration, Heston calibration, American option MC, double-barrier analytic).
3. **`docs/migration/phase6-completion.md`** + per-cluster designs.
4. **`pquantlib-final` tag** — final closure tag with comprehensive commit message.

## Carve-outs (deferred indefinitely — out of scope for pquantlib-final)

These are large, specialized domains that defer permanently unless a future explicit phase targets them:

### Specialty pricing
- MarketModels (125 .hpp files of LMM machinery).
- ZABR / SABR / XABR volatility model family.
- HestonSLV (stochastic-local-vol) / Bates double-exp / GJR-GARCH.
- PiecewiseTimeDependentHeston.
- LongstaffSchwartz multi-asset MC (this phase ports the 1-D path-pricer; multi-asset variant defers).

### Specialty term structures
- All inflation (CPI/YoY) — instruments + indexes + termstructures + capfloor + cashflows.
- All credit (CDS instruments + DefaultProbabilityTermStructure + bootstrap + DiscountingCDSEngine).
- Capfloor / optionlet / swaption volatility surfaces (Phase 2 carve-outs).
- Advanced curve construction (FittedBondDiscountCurve / MultiCurve / GlobalBootstrap / spline-fitting variants).
- Specialty short-rate (Gaussian1d / GSR / MarkovFunctional).

### Specialty cashflows
- DigitalCoupon / DigitalIborCoupon / DigitalCmsCoupon.
- CmsCoupon / CmsSpreadCoupon.
- AverageBmaCoupon.
- CapFlooredCoupon + CapFlooredOvernightIndexedCoupon.
- 35 region-specialty ibors beyond the 8 ported.

### Specialty bonds
- BTP (Italian govt), CmsRateBond, ConvertibleBond, CpiBond.

### Multi-asset numerical methods
- Multi-asset FD (Heston / G2 / Bates / CIR / SABR).
- Time-dependent FD operators + operator-splitting (HV / MS / CS / TR-BDF2).
- TreeLattice2D + G2.tree().
- Joshi4 / AdditiveEQP / Trigeorgis tree builders.
- Multi-asset PathGenerator (basket).

### Other exotic instruments
- PartialTimeBarrier / SoftBarrier (single-barrier).
- HolderExtensibleOption / ComplexChooserOption / CompoundOption.
- 3+ asset baskets (StulzEngine handles 2-asset).
- FloatingStrikeLookback (we have FloatingFixed; floating-strike has a different formula).
- Volatility models (GARCH / GarmanKlass / ConstantEstimator).

## Decision log

| Decision | Rationale |
|---|---|
| **Drop the modernization sweep entirely**. | Codebase is already modern; audit confirms 0 legacy `Generic[T]` / `Optional` / `Union` / `List[X]`. The day-1 idiom adoption made this work unnecessary. |
| **3 parallel clusters, no pilot**. | L5-A's foundations cover everything; no new abstract bases needed. |
| **Final closure as a sequential post-merge step**, not a separate cluster. | Doc + sample writing benefits from main-session context (whole-project view) more than parallel-subagent scope. |
| **`pquantlib-final` tag in this phase**. | Phase 7 was envisioned for further carve-outs but the user has been clear on cumulative scope; `pquantlib-final` after Phase 6 reflects "everything Phase 1-6 ported" cleanly. |
| **MCAmericanEngine** as the most impactful carve-out closure. | American MC is the most commonly-cited gap; ports the same Longstaff-Schwartz algorithm everyone has heard of. |
| **BatesEngine via add_on_term hook**. | Cheap to close — hook in place from Phase 4. |
| **AnalyticDoubleBarrierEngine** — Ikeda-Kunitomo (1992). | Closed-form; doesn't depend on Phase 6-deferred numerical machinery. |

## Plan + executable tasks

See [`phase6-plan.md`](phase6-plan.md).

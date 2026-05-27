# Phase 3 — L3 instruments + pricingengines (design)

**Date:** 2026-05-27
**Status:** drafted, awaiting ack to start
**Predecessor:** `pquantlib-phase2-complete` @ `b5d2519` — 922/0/0, pyright + ruff clean
**Sister-project anchor:** jquantlib `phase2-L3-instruments-pricingengines-plan.md` (a 76-class delta port off a 2007 skeleton — PQuantLib's L3 is **larger** because we start from scratch)
**C++ ground truth:** QuantLib v1.42.1 @ `099987f0`

## Goal

Port the L3 layer of QuantLib v1.42.1 to Python: the **vanilla pricing path** for bonds, swaps, equity options, and FX forwards. L3 wires L1 math + L2 termstructures/indexes/cashflows into pricable instruments. Phase 3 closes when the L3 surface needed to value (bond NPV, swap NPV, European equity option price, FX forward fair-value) is in place and probe-validated.

Phase 3 closes when:

1. Every must-port L3 class (listed below) is ported with C++-cross-validated tests or annotated `# C++ parity:` with a deliberate divergence note.
2. `uv run pytest` + `uv run pyright` + `uv run ruff check` clean on `main`.
3. Tag `pquantlib-phase3-complete` is pushed.
4. Completion doc `phase3-completion.md` lists cumulative divergences, carve-outs, lessons learned.

## Scope (must-port subset)

**Estimated total: ~50 classes** across 5 clusters. C++ has ~250 .hpp files under `ql/instruments/` + `ql/pricingengines/`; the must-port subset is the minimum to drive bond/swap/European-option/forward pricing. Everything else (exotic products, model-coupled engines, MC engines, FD engines, swaption/capfloor engines) carves out to Phase 4 (models) or Phase 5 (experimental).

### L3-A pilot — foundations (sequential, ~14 classes)

Establishes the API conventions that L3-B/C/D/E will target via Python `Protocol`s, plus the `Settings.evaluation_date` observable wiring that's been deferred since L2.

- **`Settings.evaluation_date` observable** — finally wire it. Adds `evaluation_date` field to `ObservableSettings` with observer notification on mutation. Unblocks: TermStructure moving mode (carried over from L2-A), `RelativeDateBootstrapHelper` (carried over from L2-A), SmileSection floating mode (carried over from L2-E), and several L3 pricing engines that rely on the global eval date.
- `pquantlib.instruments.instrument.Instrument` — abstract base. Carries an optional `PricingEngine`; `npv()` triggers `engine.calculate()`. Subclasses define `setup_arguments(args)` + `fetch_results(results)`.
- `pquantlib.pricingengines.pricing_engine.PricingEngine` — abstract; `calculate()` virtual.
- `pquantlib.pricingengines.generic_engine.GenericEngine[ArgsT, ResultsT]` — PEP 695 generic. `arguments` + `results` fields; `set_pricing_engine(engine)` plumbing.
- `pquantlib.pricingengines.black_formula` — `black_formula(option_type, strike, forward, std_dev, discount=1.0, displacement=0.0)`, `bachelier_black_formula(...)`, `black_formula_implied_std_dev(...)`, `bachelier_black_formula_implied_vol(...)`.
- `pquantlib.option.Option` — abstract subclass of `Instrument`. Holds `payoff` + `exercise`.
- `pquantlib.option.OneAssetOption` — abstract single-underlying option.
- `pquantlib.exercise.Exercise` — abstract. `Exercise.Type` enum (European/American/Bermudan).
- `pquantlib.exercise.EuropeanExercise(expiry_date)`.
- `pquantlib.exercise.AmericanExercise(earliest_exercise_date, latest_exercise_date, payoff_at_expiry=False)`.
- `pquantlib.exercise.BermudanExercise(exercise_dates, payoff_at_expiry=False)`.
- `pquantlib.payoffs.Payoff` — abstract. `description()` virtual.
- `pquantlib.payoffs.TypePayoff` — abstract (Call/Put discriminant).
- `pquantlib.payoffs.StrikedTypePayoff` — abstract (adds strike).
- `pquantlib.payoffs.PlainVanillaPayoff(option_type, strike)`.
- `pquantlib.payoffs.CashOrNothingPayoff` + `AssetOrNothingPayoff`.
- **Protocols** (Python-only): `InstrumentProtocol`, `PricingEngineProtocol`, `StochasticProcessProtocol`. Used by L3-B/C/D/E to reference cross-cluster types without import cycles.

### L3-B — bonds (parallel, ~8 classes)

- `Bond` — abstract subclass of `Instrument`. Holds settlement_days, calendar, issue_date, cashflows leg. Provides `clean_price` / `dirty_price` / `yield_rate` / `accrued_amount` / `next_cashflow_date` / etc.
- `FixedRateBond` — coupons from L2-D `fixed_rate_leg`.
- `ZeroCouponBond` — single redemption cashflow.
- `FloatingRateBond` — coupons from L2-D `ibor_leg`.
- `AmortizingFixedRateBond` — variable notional.
- `Callability` + `CallabilitySchedule` (simple, just the data carriers; embedded-option pricing deferred).
- `pquantlib.pricingengines.bond.discounting_bond_engine.DiscountingBondEngine` — NPV = sum of cashflows discounted by yield curve.
- `BondForward` (instrument; pricing engine deferred).

### L3-C — swaps (parallel, ~10 classes)

- `Swap` — abstract subclass of `Instrument`. Holds two legs.
- `VanillaSwap` — fixed-vs-Ibor.
- `FixedVsFloatingSwap` — abstract intermediate.
- `OvernightIndexedSwap` — fixed-vs-overnight.
- `ZeroCouponSwap` — single bullet payment.
- `MakeVanillaSwap` — factory (free function in Python, mirrors C++ Make pattern).
- `MakeOIS` — factory.
- `pquantlib.pricingengines.swap.discounting_swap_engine.DiscountingSwapEngine` — fair rate by NPV-balancing.
- **Closes carry-over**: `SwapRateHelper.implied_quote()`, `OISRateHelper.implied_quote()` (deferred from L2-C; now buildable).
- **Closes carry-over**: `SwapIndex.forecast_fixing()` + `underlying_swap()` (deferred from L2-C).

### L3-D — equity vanilla options + processes (parallel, ~12 classes)

- `pquantlib.processes.stochastic_process.StochasticProcess` — abstract; drift/diffusion/evolve.
- `pquantlib.processes.stochastic_process_1d.StochasticProcess1D` — abstract.
- `pquantlib.processes.euler_discretization.EulerDiscretization` — drift/diffusion stepper.
- `pquantlib.processes.generalized_black_scholes_process.GeneralizedBlackScholesProcess` — Black-Scholes-Merton with risk-free curve + dividend curve + Black vol surface.
- `pquantlib.processes.black_scholes_process.BlackScholesProcess` — no dividends specialization.
- `pquantlib.processes.black_process.BlackProcess` — no rates (Black 76 — futures).
- `pquantlib.processes.black_scholes_merton_process.BlackScholesMertonProcess` — full BSM.
- `pquantlib.instruments.vanilla_option.VanillaOption` — concrete option (`OneAssetOption` subclass).
- `pquantlib.instruments.european_option.EuropeanOption` — fixed European exercise.
- `pquantlib.pricingengines.vanilla.analytic_european_engine.AnalyticEuropeanEngine` — closed-form BSM via `black_formula`.
- `pquantlib.pricingengines.vanilla.binomial_engine.BinomialVanillaEngine` — Cox-Ross-Rubinstein, Jarrow-Rudd, Tian, Leisen-Reimer (parameterized).

### L3-E — forwards + FRAs (parallel, ~6 classes)

- `Forward` — abstract subclass of `Instrument`.
- `FxForward` — concrete; fair-value via discount curves of both currencies.
- `ForwardRateAgreement` — FRA on an Ibor index.
- `pquantlib.pricingengines.forward.discounting_fwd_engine.DiscountingFwdEngine` — generic discount-curve-based forward pricer.
- `pquantlib.pricingengines.forward.replicating_variance_swap_engine.ReplicatingVarianceSwapEngine` — bonus port if scope allows; otherwise defer.
- **Closes carry-over**: `FraRateHelper(useIndexedCoupon=True)` (deferred from L2-C; needs L2-D `IborCoupon`).

## Carve-outs (deferred)

Everything outside the must-port subset. Each lands either as a Phase 4/5 cluster or a dedicated follow-up.

### Exotic instruments (deferred to Phase 5)

- All Asian / Barrier / Basket / Cliquet / Lookback / Quanto / DoubleBarrier / ComplexChooser / Compound / HolderExtensible / Variance / DigitalCmsCoupon options.
- All swaption variants (`Swaption`, `NonStandardSwaption`, `FloatFloatSwaption`, `MakeSwaption`).
- All capfloor instruments (`CapFloor`, `MakeCapFloor`).
- CDS / `CreditDefaultSwap` / `MakeCDS`.
- Convertible bonds.
- BMA swap, Float/Float swap, NonStandardSwap, MultipleResetsSwap, EquityTotalReturnSwap.
- CPI cap/floor + inflation instruments.

### Model-coupled engines (deferred to Phase 4)

- All Heston / Bates / GJR-GARCH / Hull-White / G2 / CEV / SABR engines under `pricingengines/vanilla/`.
- All MC engines (Monte Carlo).
- All FD (finite-difference) engines.
- Tree / Lattice engines (`BinomialEngine` ported in L3-D as a baseline; tree-based pricing for American/Bermudan options is in Phase 4).
- All barrier / lookback / basket MC + FD engines.

### Specialized processes (deferred)

`HestonProcess`, `BatesProcess`, `GJRGARCHProcess`, `G2Process`, `Hull-White`, `CIR`, `Vasicek` processes — Phase 4.

### Bond specialty

`BTP` (Italian govt), `CmsRateBond`, `ConvertibleBond`, `CpiBond`, `AmortizingCmsRateBond`, `AmortizingFloatingRateBond`. Defer to Phase 5 or on-demand.

## Cluster topology

L3 uses the proven Phase 1+2 pattern: **1 sequential pilot + 4 parallel via subagents**.

```
L3-A (sequential pilot, foundations + Settings.evaluation_date) ──► closes
                  │
                  └──► L3-B / L3-C / L3-D / L3-E (4 parallel subagents)
                                  │
                                  └──► merge to main, tag pquantlib-phase3-complete
```

**Cross-cluster Protocols** defined in L3-A: `InstrumentProtocol`, `PricingEngineProtocol`, `StochasticProcessProtocol`. L3-B's `DiscountingBondEngine` takes `InstrumentProtocol`; L3-C's swap engine same; L3-D's `AnalyticEuropeanEngine` takes `StochasticProcessProtocol`. Structural matching at merge time.

**`Settings.evaluation_date` wiring in L3-A** unblocks several deferred items. After L3-A merges, retroactively close: TermStructure moving mode, `RelativeDateBootstrapHelper`, SmileSection floating mode. Done via `align(...)` commits in L3-A pilot, not deferred to clusters.

## Per-class TDD discipline

Identical to Phases 1 + 2 (see [`phase1-l1-A-design.md`](phase1-l1-A-design.md) for full detail). Five-step loop:

1. Read C++ source under `migration-harness/cpp/quantlib/ql/`.
2. Write probe at `migration-harness/cpp/probes/<topic>/<class>_probe.cpp`.
3. Write failing pytest loading probe JSON via `pquantlib.testing.reference_reader.load(...)`.
4. Implement.
5. Verify pass; commit `feat(<topic>): port <ClassName>` with `-s` sign-off, no `Co-authored-by`.

## Tolerance discipline

Same three tiers (EXACT / TIGHT 1e-14 abs + 1e-12 rel / LOOSE 1e-8). Per-test exceptions require inline written justification.

**New for Phase 3**: pricing engines often involve interpolation + summation + transcendental eval, so LOOSE may be common. Document tier choice per test if not TIGHT.

## Pause triggers (binding)

Same A1-A8 set. **A4 watch**: Phase 3 doesn't anticipate new deps, but if a model-coupled engine in a sub-cluster needs scipy.optimize beyond what we already use, flag.

## Decision log

| Decision | Rationale |
|---|---|
| **Wire `Settings.evaluation_date` observable in L3-A pilot** (not deferred further). | Multiple L1/L2 deferrals depend on it; L3 pricing engines need it. Unblock all at once. |
| **`Payoff` / `Exercise` / `Option` hierarchy at `pquantlib.payoffs` / `pquantlib.exercise` / `pquantlib.option`** rather than under `pquantlib.instruments.*`. Matches C++ where these are at namespace root (`ql/payoffs.hpp`, `ql/exercise.hpp`, `ql/option.hpp`). |
| **Single `BinomialVanillaEngine` parameterized by tree-builder enum** (CRR / JarrowRudd / Tian / LeisenReimer) rather than 4 separate engine classes. C++ uses templates. |
| **`MakeVanillaSwap` / `MakeOIS` as free functions** with keyword args, mirroring L2-D's leg-generator pattern. C++ Builder. |
| **`StochasticProcess` ported at `pquantlib.processes.*`** (under top-level), not under `pquantlib.math.processes`. C++ has them at `ql/stochasticprocess.hpp` + `ql/processes/*` root-level. |
| **Forward instrument's NPV-via-discount-curves**, not the more general C++ "forward + spot + cost-of-carry + dividend" form. Vanilla path only. |
| **Carry-overs from L1/L2 closed in L3 cluster commits** (not in L3-A pilot): SwapRateHelper.implied_quote in L3-C alongside `DiscountingSwapEngine`; FraRateHelper(useIndexedCoupon=True) in L3-E. |

## Plan + executable tasks

See [`phase3-plan.md`](phase3-plan.md) for the bite-sized executable task list (per-cluster subagent prompts).

## Linked

- [`phase2-completion.md`](phase2-completion.md) — Phase 2 closure summary; Phase 3 builds on its 922-test baseline.
- [`phase1-l1-A-design.md`](phase1-l1-A-design.md) — TDD ground rules + tolerance discipline (referenced across all phases).
- jquantlib `phase2-L3-instruments-pricingengines-plan.md` — sister-project anchor (delta-only port; broader scope here).

# Phase 3 — L3 instruments + pricingengines (executable plan)

> **For agentic workers:** sub-cluster work uses the proven Phase 1+2 pattern — sequential per-stub TDD for L3-A pilot; `superpowers:subagent-driven-development` for parallel B/C/D/E.

**Status:** **closed** — all 5 clusters landed on `main`; tag `pquantlib-phase3-complete` @ `aacc2c2` on 2026-05-27. **1284/0/0** tests. See [`phase3-completion.md`](phase3-completion.md).

**Goal:** Land Phase 3's ~50 must-port classes on `main`, behind tag `pquantlib-phase3-complete`. Drives Phase 4 (models) + Phase 5 (experimental).

**Predecessor:** `pquantlib-phase2-complete` @ `b5d2519` — 922/0/0, pyright + ruff clean.

**Date:** 2026-05-27.

---

## Task 0 — Spawn pilot worktree

```bash
cd /Users/josemoya/Projects/PycharmProjects/pquantlib
git worktree add -b phase3-A ../pquantlib-phase3-A main
cd ../pquantlib-phase3-A
uv sync
uv run pytest -q  # confirm 922/0/0
uv run pyright && uv run ruff check
```

DoD: worktree at `../pquantlib-phase3-A`, triad green.

---

## L3-A pilot — foundations (sequential)

**Goal**: ~14 classes + `Settings.evaluation_date` observable wiring (unblocks deferred items from L1+L2). Tag `pquantlib-phase3-l3-A-complete` when done.

### Stage 0 — Probe scaffolding

Single mega-probe `migration-harness/cpp/probes/l3a/foundations_probe.cpp` covering:
- `black_formula` at known (forward, strike, std_dev) inputs.
- `bachelier_black_formula` at the same.
- Implied-vol roundtrip.
- `PlainVanillaPayoff(Call/Put, K)(S)` at sample S values.
- `CashOrNothingPayoff` + `AssetOrNothingPayoff` at sample S.

Commit: `infra(harness): L3-A foundations mega-probe`.

### Stage 1 — Settings.evaluation_date observable

File: `pquantlib/src/pquantlib/patterns/observable_settings.py` — extend `ObservableSettings` with:
- `evaluation_date: Date | None` field.
- Property setter that calls `notify_observers()` on mutation.
- `ObservableSettings` mixes `Observable` (multi-inheritance with `Singleton` metaclass).

**Retroactive cleanup** (align commits):
- `pquantlib.time.schedule.from_rule` — remove the "null effective date" error for backward rule; instead read from `ObservableSettings().evaluation_date` as fallback.
- `pquantlib.termstructures.term_structure.TermStructure` — add moving-reference-date mode (settlement_days + calendar) via `register_with(ObservableSettings())`.
- `pquantlib.termstructures.bootstrap_helper` — add `RelativeDateBootstrapHelper` subclass.
- `pquantlib.termstructures.volatility.smile_section.SmileSection` — add floating-via-eval-date mode.

Expected test delta: +12 (3 for settings + 3 for schedule + 3 for TS + 3 for SmileSection).

Stage 1 commit: `feat(patterns): wire Settings.evaluation_date observable + retroactive cleanups`.

### Stage 2 — Payoff hierarchy

Files under `pquantlib/src/pquantlib/payoffs.py` (single module):
- `Payoff` (abstract) — `description() -> str`, `name() -> str`, `__call__(price) -> float`.
- `TypePayoff` — abstract; `OptionType` enum (Call/Put).
- `StrikedTypePayoff` — abstract; adds strike.
- `PlainVanillaPayoff(option_type, strike)` — `max(option_type * (S - K), 0)`.
- `CashOrNothingPayoff(option_type, strike, cash_payoff)`.
- `AssetOrNothingPayoff(option_type, strike)`.
- `GapPayoff(option_type, strike, second_strike)`.
- `SuperFundPayoff(strike, second_strike)`.
- `SuperSharePayoff(strike, second_strike, cash_payoff)`.

Expected test delta: +18 (3 per concrete + abstract checks).

Stage 2 commit: `feat(payoffs): port Payoff hierarchy (PlainVanilla + Cash/Asset OrNothing + Gap + SuperFund + SuperShare)`.

### Stage 3 — Exercise hierarchy

File: `pquantlib/src/pquantlib/exercise.py`:
- `Exercise` — abstract; `Type` enum (European/American/Bermudan); `last_date()` / `dates()` / `type()`.
- `EuropeanExercise(date)`.
- `AmericanExercise(earliest, latest=None, payoff_at_expiry=False)`.
- `BermudanExercise(dates, payoff_at_expiry=False)`.

Expected test delta: +12.

Stage 3 commit: `feat(exercise): port Exercise hierarchy (European/American/Bermudan)`.

### Stage 4 — Instrument + PricingEngine

Files:
- `pquantlib/src/pquantlib/instruments/instrument.py` — `Instrument` abstract (Observer + Observable + LazyObject behavior). `set_pricing_engine(engine)`, `npv()`, `error_estimate()`, `is_expired()`, `valuation_date()`, `additional_results()`. Calculate on demand; invalidate on update.
- `pquantlib/src/pquantlib/pricingengines/pricing_engine.py` — `PricingEngine` abstract (Observable). `calculate()` virtual.
- `pquantlib/src/pquantlib/pricingengines/generic_engine.py` — `GenericEngine[ArgsT, ResultsT]` (PEP 695 generic) extends `PricingEngine`. Concrete engines subclass with their own Args/Results dataclasses.

Expected test delta: +12.

Stage 4 commit: `feat(instruments): port Instrument + PricingEngine + GenericEngine`.

### Stage 5 — BlackFormula

File: `pquantlib/src/pquantlib/pricingengines/black_formula.py`. Free functions:
- `black_formula(option_type, strike, forward, std_dev, discount=1.0, displacement=0.0) -> float`.
- `black_formula_implied_std_dev(option_type, strike, forward, black_price, discount=1.0, displacement=0.0, guess=None, accuracy=1e-6, max_iterations=100) -> float`.
- `bachelier_black_formula(option_type, strike, forward, std_dev, discount=1.0) -> float`.
- `bachelier_black_formula_implied_vol(option_type, strike, forward, ttm, black_price, discount=1.0) -> float`.
- `black_formula_standard_deviation_derivative(...)`, `black_formula_volatility_derivative(...)` (Black vega).

Expected test delta: +14 (2-3 per function).

Stage 5 commit: `feat(pricingengines): port BlackFormula family (lognormal + bachelier + implied-vol solvers + derivatives)`.

### Stage 6 — Option + OneAssetOption + cross-cluster Protocols

Files:
- `pquantlib/src/pquantlib/option.py` — `Option` abstract (subclass of `Instrument`); `payoff` + `exercise` fields.
- `pquantlib/src/pquantlib/instruments/one_asset_option.py` — `OneAssetOption` abstract subclass of `Option`.
- `pquantlib/src/pquantlib/instruments/protocols.py` — `InstrumentProtocol`, `PricingEngineProtocol`, `StochasticProcessProtocol`.

Expected test delta: +8.

Stage 6 commit: `feat(option): port Option + OneAssetOption + cross-cluster Protocols`.

### L3-A closure

```bash
cd ../pquantlib-phase3-A
uv run pytest -q && uv run pyright && uv run ruff check
git checkout main
git merge --ff-only phase3-A
git push origin main
git tag -a pquantlib-phase3-l3-A-complete -m "L3-A pilot complete: 14 classes + Settings.evaluation_date wiring + retroactive cleanups, +N tests."
git push origin pquantlib-phase3-l3-A-complete
git worktree remove ../pquantlib-phase3-A
```

Estimated test delta: **+76** (922 → 998).

---

## L3-B/C/D/E — parallel dispatch

Pattern: spawn 4 worktrees off `pquantlib-phase3-l3-A-complete`, dispatch 4 subagents in one tool-use turn.

### Common subagent instructions

Reused across L3-B/C/D/E prompts:

- **Worktree**: `/Users/josemoya/Projects/PycharmProjects/pquantlib-phase3-<X>`.
- **C++ headers / library**: read-only at `/Users/josemoya/Projects/PycharmProjects/pquantlib/migration-harness/cpp/{quantlib,build/quantlib}/`. **Never write to main worktree.**
- **Probe compile**: one-off `clang++` against main worktree's libQuantLib (compile command in prompt).
- **TDD discipline**: 5-step loop, `-s` sign-off, NO `Co-authored-by`, tolerance tiers as specified.
- **Triad must pass per commit**: `uv run pytest -q && uv run pyright && uv run ruff check`.

### L3-B subagent prompt (bonds, ~8 classes)

Targets:
1. `pquantlib.instruments.bond.Bond` (abstract).
2. `pquantlib.instruments.bonds.fixed_rate_bond.FixedRateBond` (uses `fixed_rate_leg` from L2-D).
3. `pquantlib.instruments.bonds.zero_coupon_bond.ZeroCouponBond`.
4. `pquantlib.instruments.bonds.floating_rate_bond.FloatingRateBond` (uses `ibor_leg` from L2-D).
5. `pquantlib.instruments.bonds.amortizing_fixed_rate_bond.AmortizingFixedRateBond`.
6. `pquantlib.instruments.callability.Callability` + `CallabilitySchedule` (data only).
7. `pquantlib.pricingengines.bond.discounting_bond_engine.DiscountingBondEngine` (NPV via `CashFlows.npv` from L2-D).
8. `pquantlib.instruments.bond_forward.BondForward` (instrument only; engine deferred).

Probe: `migration-harness/cpp/probes/cluster_l3b/probe.cpp` covering:
- FixedRateBond clean/dirty price at known curve.
- ZeroCouponBond price = discount(maturity).
- DiscountingBondEngine NPV vs CashFlows.npv direct call.
- accrued_amount on a partial coupon.

Expected test delta: ~25-35.

### L3-C subagent prompt (swaps + finishes L2-C carry-overs, ~10 classes)

Targets:
1. `pquantlib.instruments.swap.Swap` (abstract).
2. `pquantlib.instruments.fixed_vs_floating_swap.FixedVsFloatingSwap` (abstract intermediate).
3. `pquantlib.instruments.vanilla_swap.VanillaSwap` (fixed leg + ibor leg).
4. `pquantlib.instruments.overnight_indexed_swap.OvernightIndexedSwap` (fixed leg + overnight leg).
5. `pquantlib.instruments.zero_coupon_swap.ZeroCouponSwap`.
6. `pquantlib.instruments.make_vanilla_swap.make_vanilla_swap(...)` (free function factory).
7. `pquantlib.instruments.make_ois.make_ois(...)` (free function factory).
8. `pquantlib.pricingengines.swap.discounting_swap_engine.DiscountingSwapEngine` (computes NPV per leg + fair_rate + fair_spread + BPS).
9. **L2-C carry-over**: finish `SwapRateHelper.implied_quote()` (now buildable with `MakeVanillaSwap` + `DiscountingSwapEngine`).
10. **L2-C carry-over**: finish `OISRateHelper.implied_quote()`.
11. **L2-C carry-over**: finish `SwapIndex.forecast_fixing()` + `underlying_swap()`.

Probe: `migration-harness/cpp/probes/cluster_l3c/probe.cpp` covering:
- VanillaSwap NPV (fixed-vs-Euribor3M at a known flat curve).
- DiscountingSwapEngine fair_rate (NPV-balancing).
- OvernightIndexedSwap NPV with known OIS curve.

Expected test delta: ~30-40.

### L3-D subagent prompt (equity vanilla options + processes, ~12 classes)

Targets:

**Processes** (under `pquantlib.processes.*`):
1. `pquantlib.processes.stochastic_process.StochasticProcess` (abstract).
2. `pquantlib.processes.stochastic_process_1d.StochasticProcess1D` (abstract).
3. `pquantlib.processes.euler_discretization.EulerDiscretization`.
4. `pquantlib.processes.generalized_black_scholes_process.GeneralizedBlackScholesProcess` (risk-free + dividend + Black-vol curves).
5. `pquantlib.processes.black_scholes_process.BlackScholesProcess` (no dividends).
6. `pquantlib.processes.black_process.BlackProcess` (no rates — Black 76).
7. `pquantlib.processes.black_scholes_merton_process.BlackScholesMertonProcess`.

**Instruments** (under `pquantlib.instruments.*`):
8. `pquantlib.instruments.vanilla_option.VanillaOption` (`OneAssetOption` subclass).
9. `pquantlib.instruments.european_option.EuropeanOption` (forced `EuropeanExercise`).

**Engines** (under `pquantlib.pricingengines.vanilla.*`):
10. `pquantlib.pricingengines.vanilla.analytic_european_engine.AnalyticEuropeanEngine` (closed-form via `black_formula`).
11. `pquantlib.pricingengines.vanilla.binomial_engine.BinomialVanillaEngine` (parameterized by tree-builder enum: CRR / JarrowRudd / Tian / LeisenReimer).

Probe: `migration-harness/cpp/probes/cluster_l3d/probe.cpp` covering:
- AnalyticEuropeanEngine call/put price at known (S=100, K=100, T=1, r=5%, q=2%, σ=20%) — textbook BSM value.
- Greeks (delta/gamma/vega/theta/rho).
- BinomialVanillaEngine (CRR) converges to AnalyticEuropeanEngine for N=1000 (LOOSE tier).
- BlackScholesProcess.expectation / variance / evolve at known inputs.

Expected test delta: ~35-50.

### L3-E subagent prompt (forwards + FRAs, ~6 classes)

Targets:
1. `pquantlib.instruments.forward.Forward` (abstract).
2. `pquantlib.instruments.fx_forward.FxForward`.
3. `pquantlib.instruments.forward_rate_agreement.ForwardRateAgreement` (Ibor-based).
4. `pquantlib.pricingengines.forward.discounting_fwd_engine.DiscountingFwdEngine`.
5. **L2-C carry-over**: finish `FraRateHelper(useIndexedCoupon=True)` using L2-D `IborCoupon` (now buildable).

Probe: `migration-harness/cpp/probes/cluster_l3e/probe.cpp` covering:
- FxForward fair-value via two discount curves.
- ForwardRateAgreement fair-rate at known forecast curve.
- DiscountingFwdEngine NPV.

Expected test delta: ~15-25.

---

## Task 1 — Spawn 4 worktrees + dispatch

```bash
git worktree add -b phase3-B ../pquantlib-phase3-B pquantlib-phase3-l3-A-complete
git worktree add -b phase3-C ../pquantlib-phase3-C pquantlib-phase3-l3-A-complete
git worktree add -b phase3-D ../pquantlib-phase3-D pquantlib-phase3-l3-A-complete
git worktree add -b phase3-E ../pquantlib-phase3-E pquantlib-phase3-l3-A-complete
for w in B C D E; do
  cd ../pquantlib-phase3-$w && uv sync
done
cd /Users/josemoya/Projects/PycharmProjects/pquantlib
```

Dispatch 4 `Agent` calls in one tool-use turn (parallel).

## Task 2 — Two-stage review

Same pattern as Phase 1 L1-A: post-merge, dispatch spec-compliance + code-quality reviewer subagents. BLOCKER findings → preceding align commit; MAJOR → fix-up commit; MINOR → defer to follow-up.

## Task 3 — FF-merge each cluster to main + tag

```bash
git merge --no-ff phase3-B -m "merge: L3-B (bonds)"
git merge --no-ff phase3-C -m "merge: L3-C (swaps + L2-C carry-overs)"
git merge --no-ff phase3-D -m "merge: L3-D (equity options + processes)"
git merge --no-ff phase3-E -m "merge: L3-E (forwards + L2-C FRA carry-over)"
uv run pytest -q && uv run pyright && uv run ruff check
git tag -a pquantlib-phase3-complete -m "Phase 3 (L3 instruments + pricingengines) complete: ~50 classes, +N tests."
git push origin main
git push origin pquantlib-phase3-complete
```

Worktree cleanup as in Phase 2.

## Task 4 — Write closure docs

- `docs/migration/phase3-completion.md` — 5-cluster contribution table + parallelization timings + cumulative divergences + carve-outs + lessons learned.
- `docs/migration/phase3-l3-A-completion.md` — pilot closure.
- `docs/migration/phase3-l3-{B,C,D,E}-design.md` — lean per-cluster designs.
- Update `CLAUDE.md`, `README.md`, `docs/migration/README.md`, memory.

## Expected outcomes

| Cluster | Class count | Test delta (est.) |
|---|---|---|
| L3-A pilot | ~14 | +76 |
| L3-B (bonds) | ~8 | +30 |
| L3-C (swaps + L2-C carry-overs) | ~10 | +35 |
| L3-D (equity options + processes) | ~12 | +43 |
| L3-E (forwards + L2-C FRA carry-over) | ~6 | +20 |
| **Total Phase 3** | **~50** | **~204 → 1126/0/0 cumulative** |

Stretch goal: hold L3-B/C/D/E parallel dispatch under 40 min wall-clock (Phase 2 was 35 min; L3 classes are denser so allow more headroom).

## Linked

- [`phase3-design.md`](phase3-design.md) — binding design spec.
- [`phase2-completion.md`](phase2-completion.md) — Phase 2 closure + carry-overs to close in L3-C/L3-E.
- [`phase1-l1-A-plan.md`](phase1-l1-A-plan.md) — TDD template + probe scaffolding examples.

# Phase 2 — L2 termstructures + indexes + cashflows (executable plan)

> **For agentic workers:** sub-cluster work uses the proven Phase 1 pattern — `superpowers:subagent-driven-development` for parallel B/C/D/E; sequential per-stub TDD for L2-A pilot.

**Goal:** Land Phase 2's ~70 must-port classes on `main`, behind tag `pquantlib-phase2-complete`. Drives Phase 3 vanilla-pricing-engine work.

**Predecessor:** `pquantlib-phase1-complete` @ `edcadbc` — 581/0/0, pyright + ruff clean.

**Date:** 2026-05-26.

---

## Task 0 — Spawn pilot worktree

```bash
cd /Users/josemoya/Projects/PycharmProjects/pquantlib
git worktree add -b phase2-A ../pquantlib-phase2-A main
cd ../pquantlib-phase2-A
uv sync
uv run pytest  # confirm 581/0/0 baseline
uv run pyright
uv run ruff check
```

DoD: worktree at `../pquantlib-phase2-A`, baseline triad green.

---

## L2-A pilot — foundations (sequential)

**Goal**: ~10 classes establishing the Protocol scaffolding that L2-B/C/D/E will reference. Tag `pquantlib-phase2-l2-A-complete` when done.

### Stage 0 — Probe scaffolding

#### Task A.0.1 — Mega-probe for L2-A foundations

File: `migration-harness/cpp/probes/quotes/foundations_probe.cpp`

Coverage: SimpleQuote getter/setter/observer dispatch + DerivedQuote chained transform + CompositeQuote two-input transform + BootstrapError NPV behavior.

JSON output: `migration-harness/references/quotes/foundations.json`.

Commit: `infra(harness): L2-A foundations mega-probe`

### Stage 1 — Quotes

#### Task A.1.1 — `Quote` abstract

File: `pquantlib/src/pquantlib/quotes/quote.py`

Defines `Quote` abstract base (`@runtime_checkable` Protocol-ish, but with abstract methods for `value()` + `is_valid()`). Implements `Observable` mixin from `pquantlib.patterns`.

Expected test delta: +2 (abstract-method-raises + observer-registration).

#### Task A.1.2 — `SimpleQuote`

File: `pquantlib/src/pquantlib/quotes/simple_quote.py`

Concrete leaf. Mutable value with `set_value()` triggering observer dispatch.

Expected test delta: +5 (set/get/reset, observer dispatch on set, invalid sentinel).

#### Task A.1.3 — `DerivedQuote`

File: `pquantlib/src/pquantlib/quotes/derived_quote.py`

Wraps another Quote + a `Callable[[float], float]`. Observer chain.

Expected test delta: +4.

#### Task A.1.4 — `CompositeQuote`

File: `pquantlib/src/pquantlib/quotes/composite_quote.py`

Wraps two Quotes + a `Callable[[float, float], float]`.

Expected test delta: +4.

Stage 1 commit: `feat(quotes): port Quote hierarchy (Quote + Simple + Derived + Composite)`. Expected test delta: **+15**.

### Stage 2 — Termstructure core abstractions

#### Task A.2.1 — `TermStructure` abstract

File: `pquantlib/src/pquantlib/termstructures/term_structure.py`

Defines `TermStructure` abstract base with `reference_date()`, `max_date()`, `day_counter()`, `time_from_reference()` + observer plumbing. Three construction modes (impl-1, impl-2, impl-3 per C++) collapsed via `Convention`-like enum dispatch.

Expected test delta: +6 (3 construction modes × 2 trivial-call asserts).

#### Task A.2.2 — `Extrapolator`

File: `pquantlib/src/pquantlib/termstructures/extrapolator.py`

Trivial mixin enabling extrapolation. Minimal port — just an `enable_extrapolation()` flag.

Expected test delta: +2.

Stage 2 commit: `feat(termstructures): port TermStructure abstract + Extrapolator`. Test delta: **+8**.

### Stage 3 — Bootstrap scaffolding

#### Task A.3.1 — `BootstrapHelper` abstract

File: `pquantlib/src/pquantlib/termstructures/bootstrap_helper.py`

Defines `BootstrapHelper` abstract with `quote()` / `pillar_date()` / `latest_date()` / `quote_error()` / `set_term_structure()`. Used by L2-C rate helpers and L2-B `PiecewiseYieldCurve`.

Expected test delta: +3.

#### Task A.3.2 — `BootstrapError`

File: `pquantlib/src/pquantlib/termstructures/bootstrap_error.py`

Functor adapter from `BootstrapHelper.quote_error()` to `Solver1D`-compatible `Callable[[float], float]`.

Expected test delta: +3.

Stage 3 commit: `feat(termstructures): port BootstrapHelper + BootstrapError scaffolding`. Test delta: **+6**.

### Stage 4 — Index core + IndexManager

#### Task A.4.1 — `Index` abstract

File: `pquantlib/src/pquantlib/indexes/index.py`

Defines `Index` abstract with `name()` / `is_valid_fixing_date()` / `fixing(date)` / `add_fixing(date, value)` / `clear_fixings()`. Mixes `Observable`.

Expected test delta: +4.

#### Task A.4.2 — `IndexManager` singleton

File: `pquantlib/src/pquantlib/indexes/index_manager.py`

Singleton holding the `TimeSeries[float]` fixing history per index name. Uses `pquantlib.patterns.Singleton`.

Expected test delta: +5.

Stage 4 commit: `feat(indexes): port Index abstract + IndexManager singleton`. Test delta: **+9**.

### Stage 5 — Cross-cluster Protocols

#### Task A.5.1 — Protocols module

File: `pquantlib/src/pquantlib/termstructures/protocols.py`

```python
from typing import Protocol, runtime_checkable
# YieldTermStructureProtocol: zero_rate, forward_rate, discount, reference_date, max_date, day_counter
# IborIndexProtocol: name, tenor, fixing_days, currency, calendar, day_counter, business_day_convention, end_of_month, forecast_fixing
# OvernightIndexProtocol: name, currency, calendar, day_counter
# SwapIndexProtocol: name, tenor, fixing_days, currency, fixed_leg_tenor, fixed_leg_convention, fixed_leg_day_counter, ibor_index
```

Expected test delta: +4 (structural-typing assertions — `isinstance` on a mock concrete).

Stage 5 commit: `feat(termstructures): cross-cluster Protocols (Yield/Ibor/Overnight/Swap)`. Test delta: **+4**.

### L2-A closure

```bash
cd ../pquantlib-phase2-A
uv run pytest -q  # expect 581 + 42 = 623/0/0
uv run pyright
uv run ruff check
git checkout main
git merge --ff-only phase2-A
git push origin main
git tag -a pquantlib-phase2-l2-A-complete -m "L2-A foundations complete: 10 classes, +42 tests"
git push origin pquantlib-phase2-l2-A-complete
git worktree remove ../pquantlib-phase2-A
```

DoD: pilot landed; tag pushed; 4 parallel cluster worktrees ready to spawn off this tip.

---

## L2-B/C/D/E — parallel dispatch

**Pattern**: emit mega-probes for all 4 clusters in one infra commit on `main`, then spawn 4 isolated-worktree subagents from `pquantlib-phase2-l2-A-complete`. Same mechanic as Phase 1 L1-B/C/D/E.

### Task 1 — Mega-probes (4 clusters, single commit)

Files:
- `migration-harness/cpp/probes/cluster_l2b/yield_curves_probe.cpp` — FlatForward / ZeroCurve / ForwardCurve / DiscountCurve / PiecewiseYieldCurve(IterativeBootstrap) reference values at known curve nodes.
- `migration-harness/cpp/probes/cluster_l2c/indexes_probe.cpp` — 8 ibor concretes' default-market params + a Euribor.fixing(...) sample + SwapIndex .underlying_swap() composition + 7 rate-helper bootstraps.
- `migration-harness/cpp/probes/cluster_l2d/cashflows_probe.cpp` — FixedRateCoupon/IborCoupon/OvernightIndexedCoupon amount + leg construction + CashFlows.npv / IRR / duration / convexity.
- `migration-harness/cpp/probes/cluster_l2e/volatility_probe.cpp` — BlackConstantVol/BlackVarianceCurve/BlackVarianceSurface eval at known points + LocalConstantVol/LocalVolCurve/LocalVolSurface eval.

JSON outputs: `migration-harness/references/cluster/{l2b,l2c,l2d,l2e}.json`.

Commit: `infra(harness): L2-B/C/D/E mega-probes (4-cluster parallel batch)`.

### Task 2 — Spawn 4 worktrees + dispatch subagents

```bash
git worktree add -b phase2-B ../pquantlib-phase2-B pquantlib-phase2-l2-A-complete
git worktree add -b phase2-C ../pquantlib-phase2-C pquantlib-phase2-l2-A-complete
git worktree add -b phase2-D ../pquantlib-phase2-D pquantlib-phase2-l2-A-complete
git worktree add -b phase2-E ../pquantlib-phase2-E pquantlib-phase2-l2-A-complete
```

Dispatch 4 subagents in parallel (one `Agent` call per cluster, all in one tool-use turn).

Each subagent receives:
- Worktree path.
- Cluster-specific reference JSON path.
- Per-cluster must-port class list.
- TDD loop reminder (5-step cycle).
- Triad-must-pass requirement (`pytest + pyright + ruff check`).
- Commit format: `feat(<topic>): port <ClassName>` with `-s` sign-off, no `Co-authored-by`.

Estimated wall-clock: ~25-35 min (Phase 1 L1-B/C/D/E ran ~25 min with smaller per-cluster scope).

### L2-B subagent prompt (yield curves, ~13 classes)

Targets in order (each depends on previous):

1. `pquantlib.termstructures.yield_term_structure.YieldTermStructure` (abstract).
2. `pquantlib.termstructures.yield.flat_forward.FlatForward`.
3. `pquantlib.termstructures.yield.interpolated_zero_curve.InterpolatedZeroCurve`.
4. `pquantlib.termstructures.yield.interpolated_forward_curve.InterpolatedForwardCurve`.
5. `pquantlib.termstructures.yield.interpolated_discount_curve.InterpolatedDiscountCurve`.
6. `pquantlib.termstructures.yield.zero_curve.ZeroCurve` (linear-interp alias via `type` statement, PEP 695).
7. `pquantlib.termstructures.yield.forward_curve.ForwardCurve` (alias).
8. `pquantlib.termstructures.yield.discount_curve.DiscountCurve` (alias).
9. `pquantlib.termstructures.yield.bootstrap_traits.Discount` / `ZeroRate` / `ForwardRate` (3 trait classes).
10. `pquantlib.termstructures.yield.piecewise_yield_curve.PiecewiseYieldCurve` (parameterized class).
11. `pquantlib.termstructures.yield.forward_spreaded.ForwardSpreadedTermStructure`.
12. `pquantlib.termstructures.yield.zero_spreaded.ZeroSpreadedTermStructure`.
13. `pquantlib.termstructures.yield.discount_spreaded.DiscountSpreadedTermStructure`.
14. `pquantlib.termstructures.yield.implied_term_structure.ImpliedTermStructure`.

Expected test delta: **~30-40** (3-4 tests per class).

DoD: subagent pushes to `phase2-B` branch; reports back with test counts + triad-green confirmation.

### L2-C subagent prompt (indexes + rate helpers, ~21 classes)

Targets:

1. `pquantlib.indexes.interest_rate_index.InterestRateIndex` (abstract).
2. `pquantlib.indexes.ibor_index.IborIndex` (abstract).
3. `pquantlib.indexes.overnight_index.OvernightIndex` (abstract).
4. `pquantlib.indexes.swap_index.SwapIndex` (abstract).
5. 8 ibor concretes: `Euribor`, `USDLibor`, `GBPLibor`, `Eonia`, `Sofr`, `Sonia`, `FedFunds`, `Estr` (each ~30 LOC, IntEnum dispatch where C++ has multi-tenor variants).
6. 2 swap indexes: `EuriborSwapIsdaFixA`, `UsdLiborSwapIsdaFixAm`.
7. 7 rate helpers: `DepositRateHelper`, `FraRateHelper`, `FuturesRateHelper`, `SwapRateHelper`, `OISRateHelper`, `BondHelper`, `FxSwapRateHelper` (each ~40-60 LOC; subclass `BootstrapHelper` from L2-A).

Expected test delta: **~50-65** (each ibor concrete ~3 tests, each rate helper ~5).

### L2-D subagent prompt (cashflows, ~15 classes)

Targets (use `IborIndexProtocol` / `OvernightIndexProtocol` from L2-A; concrete linking happens at merge):

1. `pquantlib.cashflows.cash_flow.CashFlow` (abstract).
2. `pquantlib.cashflows.simple_cash_flow.SimpleCashFlow`.
3. `pquantlib.cashflows.coupon.Coupon` (abstract).
4. `pquantlib.cashflows.interest_rate.InterestRate` (`@dataclass(frozen=True, slots=True)`).
5. `pquantlib.cashflows.fixed_rate_coupon.FixedRateCoupon`.
6. `pquantlib.cashflows.fixed_rate_leg.fixed_rate_leg` (free function module).
7. `pquantlib.cashflows.floating_rate_coupon.FloatingRateCoupon` (abstract).
8. `pquantlib.cashflows.ibor_coupon.IborCoupon`.
9. `pquantlib.cashflows.overnight_indexed_coupon.OvernightIndexedCoupon`.
10. `pquantlib.cashflows.ibor_leg.ibor_leg` (free function).
11. `pquantlib.cashflows.overnight_leg.overnight_leg` (free function).
12. `pquantlib.cashflows.coupon_pricer.CouponPricer` (abstract).
13. `pquantlib.cashflows.ibor_coupon_pricer.IborCouponPricer`.
14. `pquantlib.cashflows.black_ibor_coupon_pricer.BlackIborCouponPricer`.
15. `pquantlib.cashflows.cash_flows.CashFlows` (static-method aggregator: NPV, IRR, Macaulay/modified duration, convexity).
16. `pquantlib.cashflows.duration.Duration` (enum: Simple/Macaulay/Modified).

Expected test delta: **~45-55**.

### L2-E subagent prompt (vol termstructures, ~11 classes)

Targets:

1. `pquantlib.termstructures.volatility_term_structure.VolatilityTermStructure` (abstract).
2. `pquantlib.termstructures.volatility.smile_section.SmileSection` (abstract).
3. `pquantlib.termstructures.volatility.flat_smile_section.FlatSmileSection`.
4. `pquantlib.termstructures.volatility.equity_fx.black_vol_term_structure.BlackVolTermStructure` (abstract).
5. `pquantlib.termstructures.volatility.equity_fx.black_constant_vol.BlackConstantVol`.
6. `pquantlib.termstructures.volatility.equity_fx.black_variance_curve.BlackVarianceCurve`.
7. `pquantlib.termstructures.volatility.equity_fx.black_variance_surface.BlackVarianceSurface`.
8. `pquantlib.termstructures.volatility.equity_fx.local_vol_term_structure.LocalVolTermStructure` (abstract).
9. `pquantlib.termstructures.volatility.equity_fx.local_constant_vol.LocalConstantVol`.
10. `pquantlib.termstructures.volatility.equity_fx.local_vol_curve.LocalVolCurve`.
11. `pquantlib.termstructures.volatility.equity_fx.local_vol_surface.LocalVolSurface`.

Expected test delta: **~25-35**.

---

## Task 3 — Two-stage review (post-parallel-merge)

Same pattern as Phase 1 L1-A: dispatch two reviewer subagents after all 4 clusters merge:

1. **Spec-compliance reviewer** — checks each ported class against its C++ source for signature parity, behavioral parity, divergence notes (`# C++ parity:`).
2. **Code-quality reviewer** — checks ruff + pyright cleanliness, idiomatic Python (no anti-patterns), no dead code, no half-finished implementations, proper test coverage.

Both report into `docs/migration/phase2-spec-review.md` + `phase2-code-review.md`. BLOCKER findings fix as preceding commits before tag. MAJOR fix-ups before tag. MINOR can defer to follow-up.

---

## Task 4 — FF-merge each cluster to main + tag

```bash
cd /Users/josemoya/Projects/PycharmProjects/pquantlib
# In the order returned (or all at once if disjoint)
git merge --no-ff phase2-B -m "merge: L2-B (yield curves)"
git merge --no-ff phase2-C -m "merge: L2-C (indexes + rate helpers)"
git merge --no-ff phase2-D -m "merge: L2-D (cashflows + leg generators + pricers + aggregator)"
git merge --no-ff phase2-E -m "merge: L2-E (vol termstructures minimum)"
uv run pytest && uv run pyright && uv run ruff check  # all green
git tag -a pquantlib-phase2-complete -m "Phase 2 (L2 termstructures + indexes + cashflows) complete: ~70 classes, +N tests."
git push origin main
git push origin pquantlib-phase2-complete
```

Worktree cleanup:

```bash
git worktree remove ../pquantlib-phase2-B
git worktree remove ../pquantlib-phase2-C
git worktree remove ../pquantlib-phase2-D
git worktree remove ../pquantlib-phase2-E
git branch -D phase2-A phase2-B phase2-C phase2-D phase2-E
```

---

## Task 5 — Write closure docs

Files:

- `docs/migration/phase2-completion.md` — Phase 2 closure summary, 5-cluster contribution table, parallelization timings, cumulative divergences from Phase 1+2, carve-outs, lessons learned.
- `docs/migration/phase2-l2-A-completion.md` — pilot closure summary.
- `docs/migration/phase2-l2-{B,C,D,E}-design.md` — lean per-cluster design docs (mirror Phase 1's L1-B/C/D/E lean style).

Update:

- `CLAUDE.md` — current-state section (next phase = L3 instruments + pricingengines).
- `README.md` — migration-status table + "What's available today" section.
- `docs/migration/README.md` — Phase index.
- `memory/MEMORY.md` + `memory/project_python_translation_choices.md` — any new Python idiomatic decisions worth preserving.

---

## Expected outcomes

| Cluster | Class count | Test delta (est.) |
|---|---|---|
| L2-A pilot | ~10 | +42 |
| L2-B (yield curves) | ~13 | +35 |
| L2-C (indexes + rate helpers) | ~21 | +55 |
| L2-D (cashflows + pricers + aggregator) | ~15 | +50 |
| L2-E (vol termstructures minimum) | ~11 | +30 |
| **Total Phase 2** | **~70** | **~212 → 793/0/0 cumulative** |

Stretch goal: hold under 35 min wall-clock for L2-B/C/D/E parallel dispatch (matches Phase 1 envelope despite ~25% more classes per cluster).

## Linked

- [`phase2-design.md`](phase2-design.md) — binding design spec (read first).
- [`phase1-l1-A-plan.md`](phase1-l1-A-plan.md) — Phase 1 plan reference (TDD loop template, probe scaffolding examples).
- [`phase1-completion.md`](phase1-completion.md) — Phase 1 closure metrics + carve-outs (some Phase 1 carve-outs may be needed mid-Phase-2; flag any A4 trigger early).

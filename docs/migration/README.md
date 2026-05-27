# PQuantLib Migration Documentation

Each phase has 3-4 docs:

- **`phase<N>-design.md`** — binding spec approved before any code is written. Sections: scope, approach, worktree topology, tolerance discipline, pause triggers, decision log.
- **`phase<N>-plan.md`** — executable, bite-sized task list with checkboxes. Per-task: exact file paths, code snippets, expected test deltas.
- **`phase<N>-progress.md`** — running log of cluster landings (optional, mainly for multi-day phases).
- **`phase<N>-completion.md`** — closure summary: tags, test counts, latent-bug fixes surfaced, deferred items, lessons learned.

## Sister-project equivalence

For every PQuantLib phase doc, there's a corresponding JQuantLib doc at `/Users/josemoya/eclipse-workspace/jquantlib/docs/migration/`. Cross-reference when designing a phase — the Java port already made most of the scope decisions and learned most of the lessons.

| PQuantLib | JQuantLib equivalent |
|---|---|
| Phase 0 (bootstrap) | n/a (JQuantLib inherited a 2007-era skeleton) |
| Phase 1 (L1 math) | jquantlib `phase2-l1-plan.md` + `phase2-l1-{A,B,C,D,E}-*-plan.md` |
| Phase 2 (L2 termstructures + indexes) | jquantlib `phase2-L2-termstructures-indexes-plan.md` |
| Phase 3 (L3 instruments + pricingengines) | jquantlib `phase2-L3-instruments-pricingengines-plan.md` |
| Phase 4 (L4 models) | jquantlib `phase2-L4-models-plan.md` |
| Phase 5 (L5 experimental + L6 test-suite parity) | jquantlib `phase2-L5-experimental-plan.md` + `phase2-L6-test-suite-parity-plan.md` |
| Phase 6 (Python 3.14 modernization) | jquantlib `jdk25-modernization-design.md` (analogue: dataclasses, match-case, PEP 695 generics, t-strings) |
| Phase 7 (final closure) | jquantlib `phase2-complete` + `truly-complete` + `final` tags |

## Phase index (current state — 2026-05-26)

### Phase 0 — Bootstrap (closed)

- [`phase0-design.md`](phase0-design.md) — binding spec.
- Tag: `pquantlib-phase0-bootstrap` @ `85018e5`.

### Phase 1 — L1 math primitives + time + foundations (closed)

- [`phase1-design.md`](phase1-design.md) — binding spec (closed; outcome appendix at top).
- [`phase1-completion.md`](phase1-completion.md) — closure summary, 5-cluster contribution table, parallelization notes, cumulative documented divergences, carve-outs, lessons learned.
- **L1-A pilot cluster** (sequential, 6 stages — `pquantlib-phase1-l1-A-complete` @ `03d0ce8`):
  - [`phase1-l1-A-design.md`](phase1-l1-A-design.md) — design spec.
  - [`phase1-l1-A-plan.md`](phase1-l1-A-plan.md) — executable plan.
  - [`phase1-l1-A-progress.md`](phase1-l1-A-progress.md) — stage-by-stage log.
  - [`phase1-l1-A-completion.md`](phase1-l1-A-completion.md) — closure summary.
  - [`phase1-l1-A-spec-review.md`](phase1-l1-A-spec-review.md) — spec-compliance review (PASS, 4 NITs).
  - [`phase1-l1-A-code-review.md`](phase1-l1-A-code-review.md) — code-quality review (0 BLOCKER, 3 MAJOR, 6 MINOR; 3 MAJOR landed as preceding fixups).
- **L1-B / C / D / E** (parallel cluster subagents — landed into `main` and tagged together as `pquantlib-phase1-complete` @ `edcadbc`):
  - [`phase1-l1-B-design.md`](phase1-l1-B-design.md) — copulas + simple distributions/statistics + currencies (+50 tests; merge `cbd55ac`).
  - [`phase1-l1-C-design.md`](phase1-l1-C-design.md) — Solver1D + simple integrals (+34 tests; merge `6580db9`).
  - [`phase1-l1-D-design.md`](phase1-l1-D-design.md) — RNGs (EXACT-tier bit-exact) + optimization scaffolding (+52 tests; tip `5370a08`).
  - [`phase1-l1-E-design.md`](phase1-l1-E-design.md) — interpolations + matrix utilities (numpy/scipy delegates) (+30 tests; merge `8b64830`).
- Final test count: **581/0/0**. pyright + ruff clean.

### Phase 2 — L2 termstructures + indexes + cashflows (closed)

- [`phase2-design.md`](phase2-design.md) — binding spec (closed; outcome appendix at top).
- [`phase2-plan.md`](phase2-plan.md) — executable plan.
- [`phase2-completion.md`](phase2-completion.md) — closure summary, 5-cluster contribution table, parallelization notes, cumulative documented divergences from Phase 1+2, carve-outs, lessons learned.
- **L2-A pilot cluster** (sequential, 6 stages — `pquantlib-phase2-l2-A-complete` @ `4ace1f0`):
  - [`phase2-l2-A-completion.md`](phase2-l2-A-completion.md) — closure summary.
- **L2-B / C / D / E** (4 parallel cluster subagents — landed into `main` and tagged together as `pquantlib-phase2-complete` @ `b5d2519`):
  - [`phase2-l2-B-design.md`](phase2-l2-B-design.md) — yield curves (FlatForward + Interpolated{Zero,Forward,Discount} + spreaded + Implied) (+50 tests; merge `13fc008`).
  - [`phase2-l2-C-design.md`](phase2-l2-C-design.md) — indexes + rate helpers (Euribor / Libor / Eonia / Sofr / Sonia / FedFunds / Estr + Deposit/FRA/Futures/Swap/OIS/Bond/FxSwap helpers) (+77 tests; merge `e015cd7`).
  - [`phase2-l2-D-design.md`](phase2-l2-D-design.md) — cashflows (Coupon hierarchy + IborCoupon + OvernightIndexedCoupon + legs + pricers + CashFlows aggregator) (+50 tests post-dedup; merge `a9f23b0`).
  - [`phase2-l2-E-design.md`](phase2-l2-E-design.md) — vol termstructures (SmileSection + BlackVol/LocalVol family) (+96 tests; merge `b5d2519`).
- Tags: `pquantlib-phase2-l2-A-complete` @ `4ace1f0` (pilot), `pquantlib-phase2-complete` @ `b5d2519` (final).
- Final test count: **922/0/0**. pyright + ruff clean.

### Phase 3 — L3 instruments + pricingengines (closed)

- [`phase3-design.md`](phase3-design.md) — binding spec (closed; outcome appendix at top).
- [`phase3-plan.md`](phase3-plan.md) — executable plan.
- [`phase3-completion.md`](phase3-completion.md) — closure summary, 5-cluster contribution table, parallelization notes, cumulative documented divergences from Phase 1+2+3, carve-outs, lessons learned.
- **L3-A pilot cluster** (sequential, 6 stages — `pquantlib-phase3-l3-A-complete` @ `e72bcdf`):
  - [`phase3-l3-A-completion.md`](phase3-l3-A-completion.md) — closure summary (Settings.evaluation_date wired + 4 retroactive cleanups + Payoff + Exercise + Instrument + PricingEngine + BlackFormula + Option + Protocols).
- **L3-B / C / D / E** (4 parallel cluster subagents — landed into `main` and tagged together as `pquantlib-phase3-complete` @ `aacc2c2`):
  - [`phase3-l3-B-design.md`](phase3-l3-B-design.md) — bonds (Bond + 4 concretes + DiscountingBondEngine + BondForward + Callability) (+81 tests).
  - [`phase3-l3-C-design.md`](phase3-l3-C-design.md) — swaps (Swap + VanillaSwap + OIS + ZeroCoupon + Make-factories + DiscountingSwapEngine) + 3 L2-C carry-overs closed (+41 tests).
  - [`phase3-l3-D-design.md`](phase3-l3-D-design.md) — equity options + processes (StochasticProcess hierarchy + GBSM family + VanillaOption + EuropeanOption + AnalyticEuropeanEngine + BinomialVanillaEngine + BlackCalculator) (+97 tests).
  - [`phase3-l3-E-design.md`](phase3-l3-E-design.md) — forwards + FRAs (Forward + Position + FxForward + ForwardRateAgreement + DiscountingFwdEngine) + L2-C FraRateHelper carry-over closed (+28 tests).
- Tags: `pquantlib-phase3-l3-A-complete` @ `e72bcdf` (pilot), `pquantlib-phase3-complete` @ `aacc2c2` (final).
- Final test count: **1284/0/0**. pyright + ruff clean.

### Phase 4 — L4 models (not yet started)

Pending; mirror jquantlib's `phase2-L4-models-plan.md`. Expected scope: short-rate models + Heston + Hull-White + G2 + their calibration engines + the full optimizer suite (LM / BFGS / Simplex from Phase 1 carve-outs).

Will unblock Phase 3 carve-outs: Swaption + CapFloor pricing engines; BlackIborCouponPricer cap/floor branches; `VanillaOption.implied_volatility`; all Heston / Bates / GJR-GARCH equity option engines.

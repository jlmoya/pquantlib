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

### Phase 4 — L4 models (closed)

- [`phase4-design.md`](phase4-design.md) — binding spec (closed; outcome appendix at top).
- [`phase4-plan.md`](phase4-plan.md) — executable plan.
- [`phase4-completion.md`](phase4-completion.md) — closure summary, 6-row cluster contribution table, parallelization notes, cumulative documented divergences from Phase 1+2+3+4, carve-outs, lessons learned.
- **L4-A pilot cluster** (sequential, 6 stages — `pquantlib-phase4-l4-A-complete` @ `657b707`):
  - [`phase4-l4-A-completion.md`](phase4-l4-A-completion.md) — closure summary (LM + Simplex carry-overs closed; Parameter + Model + CalibrationHelper bases + 3 Protocols).
- **L4-B / C / D / E** (4 parallel cluster subagents — landed into `main` and tagged together as `pquantlib-phase4-complete` @ `fab5a0d`):
  - [`phase4-l4-B-design.md`](phase4-l4-B-design.md) — short-rate models (Vasicek + HW + CIR + ExtendedCIR + OU/CIR processes) (+49).
  - [`phase4-l4-C-design.md`](phase4-l4-C-design.md) — equity stochastic-vol (Heston + Bates + AnalyticHestonEngine via scipy.quad) (+53).
  - [`phase4-l4-D-design.md`](phase4-l4-D-design.md) — G2++ two-factor + multi-process suite (+48).
  - [`phase4-l4-E-design.md`](phase4-l4-E-design.md) — Swaption + CapFloor instruments (Phase 3 carve-outs closed) + analytic engines (Black/Bachelier swaption + Jamshidian + G2Swaption + Black/Bachelier/AnalyticCapFloor) + SwaptionHelper + CapHelper (+43).
- Tags: `pquantlib-phase4-l4-A-complete` @ `657b707` (pilot), `pquantlib-phase4-complete` @ `fab5a0d` (final).
- Final test count: **1544/0/0**. pyright + ruff clean.

### Phase 5 — L5 tree/lattice + MC + FD + exotic instruments (closed)

- [`phase5-design.md`](phase5-design.md) — binding spec (closed; outcome appendix at top).
- [`phase5-plan.md`](phase5-plan.md) — executable plan.
- [`phase5-completion.md`](phase5-completion.md) — closure summary, 6-row cluster contribution table, parallelization notes, cumulative documented divergences, carve-outs, lessons learned. Notes the FIVE phases of carry-overs Phase 5 closed.
- **L5-A pilot cluster** (sequential, 5 stages — `pquantlib-phase5-l5-A-complete` @ `aa19340`):
  - [`phase5-l5-A-completion.md`](phase5-l5-A-completion.md) — closure summary (Phase 1 carry-overs + Tree/Lattice base + DiscretizedAsset + Protocols).
- **L5-B / C / D / E** (4 parallel cluster subagents — landed and tagged together as `pquantlib-phase5-complete` @ `d322fca`):
  - [`phase5-l5-B-design.md`](phase5-l5-B-design.md) — trees + lattices + tree engines + BlackKarasinski (closes Phase 4 carve-outs) (+54).
  - [`phase5-l5-C-design.md`](phase5-l5-C-design.md) — Monte Carlo framework + MCEuropeanEngine + MCAsianEngine (+63).
  - [`phase5-l5-D-design.md`](phase5-l5-D-design.md) — Finite-difference framework + FdBlackScholesVanillaEngine + VanillaOption.implied_volatility (closes Phase 3 carve-out) (+64).
  - [`phase5-l5-E-design.md`](phase5-l5-E-design.md) — exotic instruments + 6 analytic engines + BivariateCumulativeNormalDistribution (closes Phase 1 carve-out) (+97).
- Tags: `pquantlib-phase5-l5-A-complete` @ `aa19340` (pilot), `pquantlib-phase5-complete` @ `d322fca` (final).
- Final test count: **1883/0/0**. pyright + ruff clean.

### Phase 11 — Full C++ v1.42.1 closure (closed; 12-wave mega-phase) — **`pquantlib-100-complete`**

- [`phase11-design.md`](phase11-design.md) + [`phase11-plan.md`](phase11-plan.md) — the 12-wave binding spec (delegation-philosophy-revised).
- [`phase11-completion.md`](phase11-completion.md) — per-wave contribution tables (W1–W12) + the Phase-11 grand total + the W12 coverage-audit triage.
- [`phase11-w5-resume-checkpoint.md`](phase11-w5-resume-checkpoint.md) — the W5 computer-restart checkpoint (resolved).
- **W1–W8** closed the entire specialty-model + `experimental/*` surface (credit / exotic / finite-difference / volatility / math / processes / commodities / inflation / long-tail). **W9–W11** ported the **entire MarketModels/BGM/LMM domain** (~111 files: core + models + evolvers + calibration + products + callability + pathwise greeks) with two passing canonical end-to-end tests. **W12** ran the coverage audit + filled the core cashflows CMS/CappedFloored/Digital gap.
- Tags: `pquantlib-phase11-complete` + **`pquantlib-100-complete`**. Test count: **4048/0/0** (112% of jquantlib-final 3610). 2652 → 4048 (+1396) across ~520 classes / ~50 subagent clusters.
- Result: **functional 1:1 with C++ QuantLib v1.42.1** — every class ported or documented (representation-mismatch / superseded-legacy / permanently-delegated). See [`../carve-outs.md`](../carve-outs.md) Statistics for the full accounting.

### Phase 10 — Vol surface tail + Gaussian1d short-rate + interpolator tail / ZABR (closed; opt-in extension beyond pquantlib-final)

- [`phase10-design.md`](phase10-design.md) — binding spec (closed).
- [`phase10-plan.md`](phase10-plan.md) — executable plan.
- [`phase10-completion.md`](phase10-completion.md) — closure summary, 3-cluster contribution table, merge reconciliations, divergences, follow-up carve-outs.
- **L10-A parallel** (+77 tests): KahaleSmileSection + AtmSmileSection + AtmAdjustedSmileSection + SabrInterpolatedSmileSection + OptionletStripper2 + SabrInterpolation Halton multi-start + HaltonRsg (closes Phase 9 vol-tail residuals + L1-D HaltonRsg carry-over).
- **L10-B parallel** (+41 tests): Gaussian1dModel abstract + Gsr concrete + GsrProcess + Gaussian1dSwaptionVolatility (closes Tier-1 specialty short-rate; MarkovFunctional deferred).
- **L10-C parallel** (+70 tests): HymanFilteredCubic + ChebyshevInterpolation + MultiCubicSpline + AbcdInterpolation + zabr_volatility + ZabrSmileSection (closes L1-E interpolator tail + ZABR family closed-form).
- Tag: `pquantlib-phase10-complete` @ `d3746e4`. Test count: **2652/0/0**.

### Phase 9 — Cubic/Bicubic + post-L8 ergonomics + SABR cube (closed; opt-in extension beyond pquantlib-final)

- [`phase9-design.md`](phase9-design.md) — binding spec (closed).
- [`phase9-plan.md`](phase9-plan.md) — executable plan.
- [`phase9-completion.md`](phase9-completion.md) — closure summary, 3-cluster contribution table, merge reconciliations, divergences, follow-up carve-outs.
- **L9-A pilot** (+40 tests): CubicInterpolation + CubicNaturalSpline + MonotonicCubicNaturalSpline + BicubicSpline + Interpolation2D abstract + opt-in `interpolator=` kwarg on L8-C CapFloorTermVolCurve/Surface (closes L1-E cubic-family carve-out).
- **L9-B parallel** (+41 tests): PiecewiseYieldCurve + Discount/ZeroYield/ForwardRate traits + PiecewiseDefaultCurve bootstrap wiring + IsdaCdsEngine + implied_hazard_rate + conventional_spread + MakeCDS factory (closes post-L8 credit/bootstrap ergonomics).
- **L9-C parallel** (+80 tests): SABR closed-form (Hagan 2002) + SabrInterpolation (scipy least-squares) + SmileSection abstract + Flat/Interpolated/Sabr/Spreaded SmileSection + SwaptionVolatilityCube abstract + SabrSwaptionVolatilityCube + InterpolatedSwaptionVolatilityCube (closes Phase-8 SABR cube + smile-section family carve-out).
- Tag: `pquantlib-phase9-complete` @ `7784e94`. Test count: **2464/0/0**.

### Phase 8 — Piecewise inflation + credit + capfloor-vol surfaces (closed; opt-in extension beyond pquantlib-final)

- [`phase8-design.md`](phase8-design.md) — binding spec (closed).
- [`phase8-plan.md`](phase8-plan.md) — executable plan.
- [`phase8-completion.md`](phase8-completion.md) — closure summary, 3-cluster contribution table, merge reconciliations, divergences, follow-up carve-outs.
- **L8-A** (+38 tests): PiecewiseZero/YoYInflationCurve + IterativeBootstrap[TS, Traits] + Zero/YoYInflationTraits + ZeroCouponInflationSwap/YearOnYearInflationSwap helpers (closes L7-Bb + L2-B carve-outs).
- **L8-B** (+58 tests): DefaultProbabilityTermStructure family (4 abstracts + 3 interpolated curves + FlatHazardRate) + probability traits + PiecewiseDefaultCurve scaffold + Spread/UpfrontCdsHelper + CreditDefaultSwap + Claim + MidPoint/Integral CDS engines (closes Tier-1 credit cluster).
- **L8-C** (+98 tests): CapFloorTermVolatilityStructure family + OptionletVolatilityStructure family + OptionletStripper1 + SwaptionVolatilityStructure family + SwaptionVolatilityMatrix (closes Phase 2 capfloor-vol surface carve-out).
- Tag: `pquantlib-phase8-complete` @ `efdfac3`. Test count: **2303/0/0**.

### Phase 7 — Inflation cluster (closed; opt-in extension beyond pquantlib-final)

- [`phase7-design.md`](phase7-design.md) — binding spec (closed).
- [`phase7-plan.md`](phase7-plan.md) — executable plan.
- [`phase7-completion.md`](phase7-completion.md) — closure summary, 4-cluster contribution table, merge reconciliations (incl. L7-B subagent socket drop), carve-outs.
- **L7-A pilot** (+60 tests): InflationIndex hierarchy + 5 region concretes (EUHICP/FRHICP/UKRPI/UKHICP/USCPI) + termstructure abstracts + Seasonality + 2 Protocols.
- **L7-B partial** (+12 tests): InterpolatedZero/YoYInflationCurve — Piecewise + traits + helpers deferred to L7-Bb follow-up.
- **L7-C** (+44 tests): InflationCoupon hierarchy + CPI/YoY coupons + pricers.
- **L7-D** (+35 tests): inflation swaps + cap/floor instruments + vol surfaces + 3 YoY analytic engines.
- Tag: `pquantlib-phase7-complete` @ `3a7228e`. Test count: **2109/0/0**.

### Phase 6 — high-impact carve-outs + final closure (closed)

- [`phase6-design.md`](phase6-design.md) — binding spec (closed; outcome appendix at top).
- [`phase6-plan.md`](phase6-plan.md) — executable plan.
- [`phase6-completion.md`](phase6-completion.md) — closure summary, 3-cluster contribution table, scope decision (modernization sweep deleted after audit), final closure tooling.
- **L6-A** LongstaffSchwartz American MC (closes Phase 5 carve-out) — +41 tests.
- **L6-B** BatesEngine (closes Phase 4 carve-out) — +14 tests.
- **L6-C** DoubleBarrierOption + AnalyticDoubleBarrierEngine (closes Phase 5 carve-out) — +20 tests.
- Tags: `pquantlib-phase6-complete` @ `998fed3` (the planned-migration milestone). _`pquantlib-final` originally pointed at the Phase 6 closure commit `45f4668`; it was re-pointed to the Phase 11 terminal (`1fdb1db`) on 2026-05-31 to denote the true end of the project._
- Final test count: **1958/0/0**. pyright + ruff clean.
- **`docs/carve-outs.md`** — comprehensive per-category carve-out documentation.
- **`pquantlib-samples/`** — 4 end-to-end sample programs.

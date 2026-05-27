# Phase 2 — L2 termstructures + indexes + cashflows (design)

**Date:** 2026-05-26
**Status:** **closed** — tagged `pquantlib-phase2-complete` @ `b5d2519` on 2026-05-26. **922/0/0** pytest, pyright + ruff clean. Closure summary at [`phase2-completion.md`](phase2-completion.md).
**Predecessor:** `pquantlib-phase1-complete` @ `edcadbc` — 581/0/0, pyright + ruff clean
**Sister-project anchor:** jquantlib `phase2-L2-termstructures-indexes-plan.md` (note: jquantlib's L2 is a 73-class *delta* port off a 2007 skeleton — PQuantLib's L2 is **larger** because we start from scratch)
**C++ ground truth:** QuantLib v1.42.1 @ `099987f0`

## Outcome (filled in at closure)

Phase 2 shipped as 5 clusters across two stages of work:

- **L2-A pilot** (sequential, 6 stages): 649/0/0 tests. Tagged `pquantlib-phase2-l2-A-complete` @ `4ace1f0`. See [`phase2-l2-A-completion.md`](phase2-l2-A-completion.md).
- **L2-B / L2-C / L2-D / L2-E** (4 parallel cluster subagents, ~35 min wall-clock): +273 tests, total 922/0/0.

Cross-cluster Protocols defined in L2-A (`YieldTermStructureProtocol`, `IborIndexProtocol`, `OvernightIndexProtocol`, `SwapIndexProtocol`) glued the 4 parallel clusters at merge time with zero integration code — structural typing matched concretes to consumers automatically.

Two merge reconciliations (documented in [`phase2-completion.md`](phase2-completion.md)): `Compounding` + `InterestRate` duplicated across L2-B and L2-D (resolved by keeping L2-B's canonical placements); CMakeLists.txt with 4 parallel `add_executable` additions.

## Goal

Port the L2 layer of QuantLib v1.42.1 to Python: **term structures** (yield curves + minimum volatility surfaces), **indexes** (rate indexes + 8 must-have ibor concretes + 2 swap indexes), **cashflows** (coupons + leg generators + aggregator), and **quotes** (Quote handle abstraction). Phase 2 closes when the L2 surface needed to drive Phase 3 vanilla pricing engines (bonds, plain-vanilla swaps, European options) is in place and probe-validated.

Phase 2 closes when:

1. Every must-port L2 class (listed below) is ported with C++-cross-validated tests or annotated `# C++ parity:` with a deliberate divergence note.
2. `uv run pytest` + `uv run pyright` + `uv run ruff check` clean on `main`.
3. Tag `pquantlib-phase2-complete` is pushed.
4. Completion doc `phase2-completion.md` lists cumulative divergences, carve-outs, and lessons learned.

## Scope (must-port subset)

**Estimated total: ~70 classes across 5 clusters (A pilot + B/C/D/E parallel).** Comparable in scale to Phase 1's L1 work (~120 classes ported across 5 stages + 4 parallel clusters).

### L2-A pilot — foundations (sequential, ~10 classes)

Establishes the API conventions that L2-B/C/D/E will target via Python `Protocol`s (no inheritance — structural typing lets parallel clusters link at merge time without sequencing).

- `quotes`: `Quote` abstract + `SimpleQuote` + `DerivedQuote` + `CompositeQuote` (4 classes).
- `termstructures`: `TermStructure` abstract + `Extrapolator` (2 classes).
- `termstructures.bootstrap`: `BootstrapHelper` + `BootstrapError` (2 classes).
- `indexes`: `Index` abstract + `IndexManager` singleton (2 classes).
- **Protocols** (Python-only; no C++ analogue): `YieldTermStructureProtocol`, `IborIndexProtocol`, `OvernightIndexProtocol`, `SwapIndexProtocol`. Used by L2-B/C/D/E to reference cross-cluster types without import cycles.

### L2-B — yield curves (parallel, ~13 classes)

- `YieldTermStructure` abstract.
- `FlatForward` — constant-rate curve.
- `InterpolatedZeroCurve` / `InterpolatedForwardCurve` / `InterpolatedDiscountCurve` (templated in C++ on interpolation type; Python uses a single class parameterized by an `Interpolation` instance).
- `ZeroCurve` / `ForwardCurve` / `DiscountCurve` — linear-interpolation aliases for the above.
- `BootstrapTraits` — `Discount`, `ZeroRate`, `ForwardRate` (the three pillars for `PiecewiseYieldCurve`).
- `PiecewiseYieldCurve` — generic curve bootstrap shell (concrete bootstrap binds at construction time via rate-helper list, supplied by L2-C concretes at merge time).
- `ForwardSpreadedTermStructure` + `ZeroSpreadedTermStructure` + `DiscountSpreadedTermStructure` (3 spread overlays).
- `ImpliedTermStructure` — forward-shifted view of an existing curve.

### L2-C — indexes + rate helpers (parallel, ~21 classes)

- `InterestRateIndex` abstract.
- `IborIndex` abstract + `OvernightIndex` abstract + `SwapIndex` abstract.
- 8 ibor concretes: `Euribor`, `USDLibor`, `GBPLibor`, `Eonia`, `Sofr`, `Sonia`, `FedFunds`, `Estr`.
- 2 swap index concretes: `EuriborSwapIsdaFixA`, `UsdLiborSwapIsdaFixAm`.
- Rate helpers: `DepositRateHelper`, `FraRateHelper`, `FuturesRateHelper`, `SwapRateHelper`, `OISRateHelper`, `BondHelper`, `FxSwapRateHelper` (7 helpers).

### L2-D — cashflows (parallel, ~15 classes)

- `CashFlow` abstract.
- `Coupon` abstract.
- `SimpleCashFlow` — fixed-amount-on-date.
- `InterestRate` — rate + day counter + compounding tuple (`@dataclass(frozen=True, slots=True)`).
- `FixedRateCoupon` + `FixedRateLeg` (leg generator).
- `FloatingRateCoupon` abstract + `IborCoupon` + `OvernightIndexedCoupon` + `IborLeg` + `OvernightLeg` (4 floating concretes + 2 leg generators).
- `CouponPricer` abstract + `IborCouponPricer` + `BlackIborCouponPricer` (3 pricers).
- `CashFlows` — aggregator (NPV / IRR / Macaulay duration / modified duration / convexity static methods).
- `Duration` enum (Simple/Macaulay/Modified).

### L2-E — volatility termstructures (parallel, ~11 classes)

Minimum needed for vanilla European-option pricing in Phase 3.

- `VolatilityTermStructure` abstract.
- `SmileSection` abstract + `FlatSmileSection` (2 classes).
- `BlackVolTermStructure` abstract + `BlackConstantVol` + `BlackVarianceCurve` + `BlackVarianceSurface` (4 classes).
- `LocalVolTermStructure` abstract + `LocalConstantVol` + `LocalVolCurve` + `LocalVolSurface` (4 classes).

## Carve-outs (deferred)

These are present in C++ v1.42.1 but **not ported in Phase 2**. They land either as L2-completion clusters (Phase 5 or before) or as part of higher layers that depend on them.

### Inflation (entire surface)

`ql/termstructures/inflation/*` (8 headers), `ql/indexes/inflation/*` (8 headers), `ql/termstructures/volatility/inflation/*` (4 headers), `ql/cashflows/{cpi_coupon, cpi_coupon_pricer, capflooredinflationcoupon}.hpp`. Specialized domain; not needed for vanilla bond / swap / equity-option pricing. Deferred to a dedicated Phase 5 inflation cluster.

### Credit (entire surface)

`ql/termstructures/credit/*` (11 headers) — `DefaultProbabilityTermStructure`, `PiecewiseDefaultCurve`, hazard-rate / survival-probability traits, `CdsHelper`. Deferred to Phase 4 (models) or Phase 5 (experimental) as appropriate.

### Specialized volatility

- ZABR/SABR/XABR volatility models — depend on optimization concretes deferred from Phase 1 (LM/BFGS/Simplex). Land as a follow-up after Phase 1 carve-outs close.
- `ql/termstructures/volatility/capfloor/*` (5 headers) + `ql/termstructures/volatility/optionlet/*` (11 headers) + `ql/termstructures/volatility/swaption/*` (13 headers) — specialized interest-rate volatility. Defer to Phase 4 (models) where they're naturally consumed.
- `BlackVolSurface` Andreasen-Huge variants, `GridModelLocalVolSurface`, `HestonBlackVolSurface` — all in volatility/equityfx but model-coupled. Defer to Phase 4.

### Ibor specialty / region-specialized indexes

35 of the 43 ibor concretes are region-specialty (AUDLibor, BBSW, Bibor, BKBM, CADLibor, CDI, Cdor, CHFLibor, Corra, Custom, Destr, DKKLibor, EURLibor, Jibar, JPYLibor, Kofr, Libor base + tenor variants, Mosprime, NZDLibor, NZOCR, Pribor, Robor, Saron, SEKLibor, Shibor, Swestr, Thbfix, Tibor, Tona, Tonar, TRLibor, Wibor, Zibor, BMA, Equity, Aonia). Port on demand when an instrument or test-suite case in Phase 3+ needs one. The 8 chosen for L2-C cover ~90% of standard textbook curve-building scenarios.

### Specialized cashflows

`DigitalCoupon`, `DigitalIborCoupon`, `DigitalCmsCoupon`, `CmsCoupon`, `CmsSpreadCoupon`, `AverageBmaCoupon`, `CapFlooredCoupon`, `EquityCashflow`, `Dividend`, `ConundrumPricer`, `LinearTsrPricer`, `Yoy*` — defer to L3 (instruments + pricing engines) or specialized clusters.

### Yield-curve specialty

`FittedBondDiscountCurve`, `CompositeZeroYieldStructure`, `MultiCurve`, `GlobalBootstrap`, `LocalBootstrap`, `MultipleResetsSwapHelper`, `OvernightIndexFutureRateHelper`, `QuantoTermStructure`, `NonlinearFittingMethods`, `CubicBSplinesFitting`, `ExponentialSplinesFitting`, `NaturalCubicFitting`, `SpreadFittingMethod`, `SpreadTraits`, `CPIBondHelper`, `InterpolatedSimpleZeroCurve`, `PiecewiseSpreadYieldCurve`, `PiecewiseForwardSpreadedTermStructure`, `PiecewiseZeroSpreadedTermStructure`, `ForwardStructure` — defer to L2-completion cluster (advanced curve construction).

## Cluster topology

L2 uses the proven Phase 1 pattern: **1 sequential pilot + 4 parallel via subagents**.

```
L2-A (sequential pilot, foundations) ──────────────► closes
                  │
                  └──► L2-B / L2-C / L2-D / L2-E (4 parallel subagents)
                                  │
                                  └──► all merge to main, tag pquantlib-phase2-complete
```

**Sequencing rationale.** L2-A defines the Protocols + abstract bases that the parallel clusters reference. By using `typing.Protocol` (structural typing) rather than concrete inheritance, the parallel clusters can implement their concretes without needing each other's code at compile time. At merge time, structural matching automatically links L2-D's `IborCoupon` to L2-C's `IborIndex` — no glue cluster required.

**Worktree topology.** One worktree per cluster: `../pquantlib-phase2-A`, `-B`, `-C`, `-D`, `-E`. A is sequential. B/C/D/E dispatch as subagents off the `phase2-A` tip (same pattern as Phase 1 L1-B/C/D/E off `pquantlib-phase1-l1-A-complete`).

## Per-class TDD discipline

Identical to Phase 1 (see [`phase1-l1-A-design.md`](phase1-l1-A-design.md) for full detail). Five-step loop:

1. Read C++ source under `migration-harness/cpp/quantlib/ql/<subdir>/`.
2. Write probe at `migration-harness/cpp/probes/<topic>/<class>_probe.cpp`. Emit reference JSON to `migration-harness/references/<topic>/<class>.json`.
3. Write failing pytest under `pquantlib/tests/<topic>/test_<class>.py` loading the JSON via `pquantlib.testing.reference_reader.load(...)`. Compare via `pquantlib.testing.tolerance.{exact,tight,loose,custom}`.
4. Implement at `pquantlib/src/pquantlib/<topic>/<class>.py`.
5. Verify pass; commit `feat(<topic>): port <ClassName>` with `-s` Signed-off-by, no `Co-authored-by`.

**Cluster-batch commits**: each parallel cluster lands as one merge commit on `main` (`merge: L2-X (...)`). Individual stub commits land on the cluster branch.

## Tolerance discipline

Same three tiers as Phase 1:

- **EXACT** (`struct.pack('!d', x)`): bit-identical. Used where the algorithm is reproducible (e.g. interpolation between known nodes; flat-rate compounding).
- **TIGHT** (`math.isclose(abs_tol=1e-14, rel_tol=1e-12)`): default for closed-form math (e.g. discount factor from continuous compounding).
- **LOOSE** (`math.isclose(abs_tol=1e-8, rel_tol=1e-8)`): bootstrapped values, iterative solvers, anywhere accumulated rounding is expected.

**Per-test exceptions require inline written justification.** Bootstrapping in particular may need a custom tolerance (e.g. `1e-10` after solver convergence); document with `# tolerance: solver convergence target` at the assert.

## Pause triggers (binding)

Same A1-A8 set from Phase 1:

| Trigger | Condition |
|---------|-----------|
| A1 | Phase scope > 1000 classes |
| A2 | Tolerance looser than 1e-8 needed |
| A3 | Cross-validation suggests v1.42.1 itself is wrong |
| A4 | Stub needs a Python dep not in the locked workspace |
| A6 | End of every phase — report summary, wait for ack |

No A4 issues anticipated for L2 (numpy/scipy already in workspace; no new deps planned).

## Decision log

| Decision | Rationale |
|---|---|
| Use Python `Protocol` (structural typing) for cross-cluster abstract types, not abstract base classes. | Lets L2-B/C/D/E port concretes in parallel without import-cycle / wait-on-sibling concerns. Structural typing matches concretes at merge time automatically. |
| Drop tenor-specialized ibors (35 of 43) from must-port. | Region specialty; port on demand when a Phase 3+ instrument needs one. The 8 chosen cover ~90% of textbook curve-building. |
| Defer inflation + credit termstructures + specialty volatility entirely. | Specialized domains that don't block vanilla pricing engines. Each gets a dedicated follow-up cluster. |
| Single Python `PiecewiseYieldCurve` parameterized by interpolation type + bootstrap trait, not 9 separate templates. | C++ uses `PiecewiseYieldCurve<Traits, Interpolator>` (template). Python's runtime polymorphism + the `Interpolation` abstract base from L1-E lets one class cover all 9 combos. |
| `InterestRate` as `@dataclass(frozen=True, slots=True)`. | C++ uses a small POD-like struct. Frozen+slots matches the immutability + cheap-copy semantics. |
| Leg generators as **free functions in a module**, not Builder classes. | C++ uses Builder classes (`FixedRateLeg`, `IborLeg`) chained with `.withNotionals(...)` etc. Python's keyword-args + `@dataclass(frozen=True, slots=True)` config objects are cleaner. Documented inline. |
| Concrete `Yield*Curve` / `BlackVariance*Curve` aliases via `type` statements (PEP 695) instead of subclassing. | Eliminates a layer of indirection; matches Python idioms. |

## Plan + executable tasks

See [`phase2-plan.md`](phase2-plan.md) for the bite-sized executable task list (one checkbox per stub, with exact file paths + expected test deltas).

## Linked

- [`phase1-completion.md`](phase1-completion.md) — Phase 1 closure summary; Phase 2 builds on its 581-test baseline.
- [`phase1-l1-A-design.md`](phase1-l1-A-design.md) — TDD ground rules + tolerance discipline (referenced by every Phase 1 sub-cluster and now Phase 2).
- jquantlib `phase2-L2-termstructures-indexes-plan.md` — sister-project anchor; PQuantLib's scope is broader because we start from scratch.

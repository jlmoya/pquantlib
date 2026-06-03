# Phase 12 — completion (sibling-packages migration)

**Started:** 2026-06-02 · **Closed:** 2026-06-03
**Predecessor:** `pquantlib-phase11-complete` + `pquantlib-100-complete` + `pquantlib-final` @ `1fdb1db` — 4048/0/0
**Final:** `pquantlib-siblings-complete` — **4237 passed / 0 failed / 3 skipped**
**C++ ground truth:** QuantLib v1.42.1 @ `099987f0` (where a C++ equivalent exists); JQuantLib Java output for the retired-API / JQuantLib-original buckets.

**Phase 12 ported the three JQuantLib sibling Maven modules — `jquantlib-contrib`, `jquantlib-helpers`, `jquantlib-samples` — into the PQuantLib workspace members `pquantlib-contrib`, `pquantlib-helpers`, `pquantlib-samples`.** These were never in the Phase 0–11 scope (which closed PQuantLib *core* at functional 1:1 with v1.42.1); they are separate workspace members against a different source set. The phase is self-contained: 5 sequenced waves, 4048 → 4237 tests (+189 / +3 skips), ~45 classes. See [`phase12-design.md`](phase12-design.md) for the binding spec + [`docs/carve-outs.md`](../carve-outs.md) for the ConvertibleBond closure + sample/follow-up accounting.

## The API-era reconciliation (the framing that made this completable)

The three siblings were authored against a **much older QuantLib API** — JQuantLib forked QuantLib c. 2008 (QL ~0.9/1.0). PQuantLib faithfully mirrors **v1.42.1**. Several depended-on classes therefore split into three provenance buckets, each with its own ground-truth rule. Conflating them produced two prior dead-ends; keeping them separate is what made the phase completable. **PQuantLib core stays pure v1.42.1; period-specific retired-API classes are hosted by the sibling package that needs them — exactly what a `contrib`/`helpers` package is for, and exactly how the Java siblings are structured.**

| Bucket | Examples | Status in v1.42.1 | Where it lives | Ground truth |
|---|---|---|---|---|
| **(A) Retired-API** | `DividendVanillaOption`, `BinomialDividendVanillaEngine`, `BlackScholesDividendLattice`, the pre-1.0 FD framework (`FDVanillaEngine`/`FDDividendEuropeanEngine`/`FDDividendAmericanEngine`) | **Removed** from v1.42.1 (replaced by `DividendSchedule` + `AnalyticDividendEuropeanEngine` + `FdBlackScholesVanillaEngine(dividends=…)`) | `pquantlib-helpers` (compat layer on v1.42.1 primitives) | C++ v1.42.1 probes for the *pricing* where the 3 surviving FD-scheme classes exist; otherwise TIGHT vs JQuantLib Java same-engine output |
| **(B) JQuantLib-original** | `XorShiftRandom` | **Never in C++** (Marsaglia 2003 xorshift) | `pquantlib-contrib` | JQuantLib Java seeded sequences (`# C++ parity:` note documents the no-C++-equivalent divergence) |
| **(C) Real v1.42.1 class, prior carve-out** | `ConvertibleFixedCouponBond`, `ConvertibleZeroCouponBond`, `SoftCallability`, `BinomialConvertibleEngine` | **Present** (`ql/instruments/bonds/convertiblebonds.hpp` + experimental convertible engine), deferred at Phase 11 | PQuantLib **core** (closes the carve-out) | C++ v1.42.1 convertible-bond probes |

## Wave summary

| Wave | Tag | Package | Provenance | Scope |
|---|---|---|---|---|
| W-S1 | `pquantlib-siblings-ws1-complete` @ `4b3d923` | `pquantlib-contrib` | B | `XorShiftRandom` (pilot) |
| W-S2 | `pquantlib-siblings-ws2-complete` @ `e17d705` | `pquantlib-helpers` | A | dividend-option compat primitives |
| W-S3 | `pquantlib-siblings-ws3-complete` @ `3653f6e` | `pquantlib-helpers` | A | 6 helper builders + the legacy FD framework + 2 test suites |
| W-S4 | `pquantlib-siblings-ws4-complete` @ `be001ae` | pquantlib **core** | C | ConvertibleBond subsystem (closes the carve-out) |
| W-S5 | `pquantlib-siblings-complete` @ `553d4c1`+ | `pquantlib-samples` | mixed | sample programs + `AllSamples` runner + pytest smoke suite |

W-S1 doubled as the pilot (proves the per-package skeleton + test wiring + review loop). W-S2 preceded W-S3 (helpers depend on the compat layer). W-S4 preceded the ConvertibleBonds sample in W-S5.

## W-S1 — `XorShiftRandom` → `pquantlib-contrib` (pilot, bucket B)

Marsaglia 2003 xorshift64 RNG. **No C++ equivalent** — a JQuantLib convenience RNG; cross-validated against JQuantLib Java seeded output, with a `# C++ parity:` note documenting the deliberate "C++ is source of truth" divergence. Java `long` signed-64 + `>>>` unsigned shift reproduced via explicit `& 0xFFFFFFFFFFFFFFFF` masking. Proved out the contrib-package skeleton.

## W-S2 — dividend-option compat primitives → `pquantlib-helpers` (bucket A)

`DividendVanillaOption` + `BinomialDividendVanillaEngine` + `BlackScholesDividendLattice`. These classes were **removed from C++ v1.42.1**; hosted in the helpers package (as the Java siblings do, not in jquantlib core), built on v1.42.1 core primitives, cross-validated TIGHT vs JQuantLib Java same-engine output (`migration-harness/references/cluster/ws2_java.json`). **Faithfully reproduces a real JQuantLib bug** — the lattice drops the dividend amount — for behavioral parity with the Java sibling.

## W-S3 — 6 helper builders + the legacy FD framework (bucket A)

The 6 dividend-option helper builders (CRR/FD × European/American + 2 bases) + the retired pre-1.0 QuantLib FD framework. PQuantLib core ships only the modern `Fdm*` framework, so the legacy FD layer also lives in `pquantlib-helpers`.

- **W-S3a:** `CRRDividendOptionHelper` + CRR European/American helpers (1:1, TIGHT vs Java).
- **W-S3-FD** (3 clusters, ~35 classes): the retired FD framework —
  - α1: `TridiagonalOperator`, `BSMOperator`, D-operators, `FiniteDifferenceModel`/schemes (C++-validated TIGHT for the 3 scheme classes that survive in v1.42.1).
  - α2: boundary/step conditions + PDE grid-operator layer.
  - β: the engine layer `FDVanillaEngine` → `FDMultiPeriodEngine`/`FDStepConditionEngine` → `FDDividendEuropeanEngine`/`FDDividendAmericanEngine` + `StandardSystemFiniteDifferenceModel` + `SampledCurve`. End-to-end TIGHT vs JQuantLib Java FD output.
- **W-S3c:** `FDDividendOptionHelper` + FD European/American helpers (1:1, TIGHT vs Java).

All 6 helpers (CRR/FD × European/American + 2 bases) cross-validate TIGHT.

**Bug found + fixed during β (behavioral-parity reproduction):** Java's `FDMultiPeriodEngine` **swaps the `(gridPoints, timeSteps)` ctor params** so `timeStepPerPeriod` binds to `gridPoints`. Reproducing this faithfully made the European engine bit-invariant to `timeSteps` and reach TIGHT vs Java. A bit-invariance regression guard + a note documenting the `time_dependent` control-operator limitation landed as a review follow-up.

## W-S4 — ConvertibleBond subsystem → pquantlib **CORE** (bucket C — closes the carve-out)

Genuine C++ v1.42.1 — closes the carve-out previously at `docs/carve-outs.md:172`. Ported: `SoftCallability`, `ConvertibleFixedCouponBond` + `ConvertibleZeroCouponBond`, `DiscretizedConvertible` (Tsiveriotis–Fernandes credit-spread decomposition), `TsiveriotisFernandesLattice`, `BinomialConvertibleEngine`. Cross-validated TIGHT vs C++ `BinomialConvertibleEngine<CoxRossRubinstein>` at the identical N=801 on 3 scenarios (European / American / American+SoftCallability+Put+FixedDividend). A review follow-up registered `ConvertibleZeroCouponBond` as an observer of its redemption cashflow. `ConvertibleFloatingRateBond` is a straightforward follow-on (IborLeg in place of FixedRateLeg) — not yet ported.

## W-S5 — sample programs → `pquantlib-samples` (mixed)

Foundation (`StopClock` timing helper, `AllSamples` runner with COMPLETE/INCOMPLETE/PENDING tuples, pytest smoke suite mirroring Java `TestSamples`) + the sample programs. Each Java sample → one Python module with an importable `run()` that prints to stdout (console-behaviour parity), an `if __name__ == "__main__":` entry, and a smoke test; PENDING samples → `@pytest.mark.skip` mirroring Java `@Ignore`.

- **12 COMPLETE:** calendars, dates, swap, repo, convertible_bonds (via W-S4), processes, yield_curve_term_structures, fra, discrete_hedging, sobol_chart_sample (matplotlib, env-guarded headless), volatility_term_structures, bermudan_swaption.
- **1 INCOMPLETE:** equity_options — the American analytic approximation engines (Barone-Adesi/Whaley, Bjerksund/Stensland, Ju Quadratic, Integral) are not ported, so those rows print `N/A`; mirrors the Java sibling.
- **3 PENDING** (genuinely blocked by a missing core class, re-verified against Python-idiom names; `run()` raises `NotImplementedError`):
  - **bonds** — needs a functional `FixedRateBondHelper` / `BondHelper.implied_quote()` (the latter is an explicit deferred stub never closed). Java AllSamples also kept Bonds pending.
  - **replication** — needs `CompositeInstrument` (not ported; the Java original was itself an incomplete stub).
  - **cox_ross_with_hull_white** — the extended binomial-tree family IS ported, but no engine wires the extended trees to a diffusion process, and there is no equity-style `HullWhiteProcess`. The Java original threw `UnsupportedOperationException` for the same reason.

## Bugs found + follow-ups surfaced

**Found + fixed:**
- W-S3-FD param-swap (above) — reproduced faithfully for Java behavioral parity.
- W-S2 lattice dividend-drop bug — reproduced faithfully for Java behavioral parity.

**Surfaced core follow-ups** (documented in `docs/carve-outs.md`; not re-discovered here):
1. `JamshidianSwaptionEngine` → `OneFactorAffineModel.discount_bond` has a multi-factor signature issue (`'float' object is not subscriptable`) that surfaces during Hull-White calibration; the `bermudan_swaption` sample uses the tree-engine calibration path instead. Small core follow-up.
2. `PiecewiseYieldCurve` does not re-bootstrap on quote change (bootstrap-once limitation, noted during W-S5 review).
3. `ConvertibleFloatingRateBond` not yet ported (W-S4 follow-on).

## Phase 12 grand total

4048 → **4237 passed / 0 failed / 3 skipped** (+189 tests / +3 PENDING-sample skips) across 5 sequenced waves / ~45 classes. The three sibling packages are now populated: `pquantlib-contrib` (XorShiftRandom), `pquantlib-helpers` (the dividend-option compat layer + the legacy FD framework + 6 helper builders), `pquantlib-samples` (12 COMPLETE + 1 INCOMPLETE + 3 PENDING samples). ConvertibleBond is closed in core (bucket C). The retired-API dividend/FD classes live in `pquantlib-helpers` per the API-era reconciliation — core stays pure v1.42.1. Tag `pquantlib-siblings-complete`.

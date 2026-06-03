# Phase 12 — Sibling-packages migration (design)

**Date:** 2026-06-02
**Status:** approved (brainstorm complete)
**Predecessor:** `pquantlib-phase11-complete` + `pquantlib-100-complete` + `pquantlib-final` @ `1fdb1db` — 4048/0/0, pyright + ruff clean
**C++ ground truth:** QuantLib v1.42.1 @ `099987f0` (where applicable)
**Target:** port the three JQuantLib sibling projects — `jquantlib-contrib`, `jquantlib-helpers`, `jquantlib-samples` — into the already-skeletoned PQuantLib workspace members `pquantlib-contrib`, `pquantlib-helpers`, `pquantlib-samples`, at the same fidelity and discipline as Phases 0–11.

Source root for the Java originals: `/Users/josemoya/eclipse-workspace/jquantlib/{jquantlib-contrib,jquantlib-helpers,jquantlib-samples}`.

---

## Why this is a distinct effort, not a Phase-11 reopening

Phase 11 closed PQuantLib **core** at functional 1:1 with C++ v1.42.1 (with documented carve-outs in `docs/carve-outs.md`). The three sibling packages are *separate Maven modules* in the Java repo and *separate workspace members* here. They were never in the Phase 0–11 scope. This is a new, self-contained migration phase against a different source set.

## The API-era reconciliation (binding framing)

The three siblings were authored against a **much older QuantLib API** — JQuantLib forked QuantLib c. 2008 (QL ~0.9/1.0). PQuantLib faithfully mirrors **v1.42.1**. Several classes the siblings depend on therefore fall into three distinct provenance buckets, and each bucket has its own ground-truth rule. Conflating them is what produced two prior dead-ends; keeping them separate is what makes this completable.

| Bucket | Examples | Status in v1.42.1 | Where it lives in the port | Ground truth |
|---|---|---|---|---|
| **(A) Retired-API** | `DividendVanillaOption`, `FDDividendEuropeanEngine`, `FDDividendAmericanEngine`, `FDEngineAdapter`, `BinomialDividendVanillaEngine`, `BlackScholesDividendLattice` | **Removed** — `DividendVanillaOption` has no class in `ql/` (only a stale test-suite reference); the FD/binomial dividend engines are gone entirely. v1.42.1 replaced them with `DividendSchedule` + `AnalyticDividendEuropeanEngine` + `CashDividendEuropeanEngine` + `FdBlackScholesVanillaEngine(dividends=…)`. | **Inside the sibling package** (`pquantlib-helpers`), as a compat layer built on v1.42.1 primitives. Mirrors the Java layout exactly: `BlackScholesDividendLattice` and `BinomialDividendVanillaEngine` already live in `jquantlib-helpers`, **not** jquantlib core. | C++ v1.42.1 `AnalyticDividendEuropeanEngine` / `FdBlackScholesVanillaEngine(dividends)` probes — real C++ ground truth for the *pricing*, even though the *class shape* is retired-API. |
| **(B) JQuantLib-original** | `XorShiftRandom` | **Never in C++** (Marsaglia 2003 xorshift; a JQuantLib convenience RNG). | `pquantlib-contrib`. | JQuantLib Java output (seeded sequences). Divergence from the "C++ is source of truth" rule documented inline with a `# C++ parity:` note explaining there is no C++ equivalent. |
| **(C) Real v1.42.1 class, prior carve-out** | `ConvertibleFixedCouponBond`, `SoftCallability`/`CallabilitySchedule`, `BinomialConvertibleEngine` | **Present** (`ql/instruments/bonds/convertiblebonds.hpp` + experimental convertible engine), but deferred at Phase 11 (`docs/carve-outs.md:172`). | PQuantLib **core** (closes the carve-out). | C++ v1.42.1 convertible-bond probes. |

**Key principle:** PQuantLib core stays pure v1.42.1. Period-specific retired-API classes are hosted by the sibling package that needs them — which is precisely what a `contrib`/`helpers` package is for, and precisely how the Java siblings are structured.

---

## Source inventory

| Project | Java files | LOC | Content |
|---|---|---|---|
| `jquantlib-contrib` | 1 | 55 | `XorShiftRandom` (bucket B). |
| `jquantlib-helpers` | 10 | 1578 | 6 helper builders (`CRR`/`FD` × `European`/`American` + `CRRDividendOptionHelper`/`FDDividendOptionHelper` bases) + `BlackScholesDividendLattice` + `BinomialDividendVanillaEngine` + 2 test suites (`CRRDividendOptionTest`, `FDDividendOptionTest`). |
| `jquantlib-samples` | 21 | 3829 | 16 sample programs + `AllSamples` runner + `TestSamples` + 3 util files (`StopClock`, `ReplicationError`, `ReplicationPathPricer`). |

### Helpers compat-layer surface (from reading `CRRDividendOptionHelper`)
Each helper `extends DividendVanillaOption`; builds a `BlackScholesMertonProcess` from flat r/q/vol `SimpleQuote`s (`FlatForward` + `BlackConstantVol`); attaches a `BinomialDividendVanillaEngine<CoxRossRubinstein>` (or the FD engine); exposes `vega`/`rho`/`impliedVolatility` via bump-and-revalue. All of `FdBlackScholesVanillaEngine`, `BinomialVanillaEngine`, `GeneralizedBlackScholesProcess`, `Dividend` cashflow, `ExtendedTian`/CRR lattices, `FlatForward`, `BlackConstantVol`, `VanillaOption` confirmed present in pquantlib core.

### Samples bucketing (from dependency audit)
Java `AllSamples` already stratifies into **complete / incomplete / pending**; we mirror that. Preliminary classification (the audit used narrow grep; each "blocked" item is re-verified against Python-idiom names at dispatch, so several are likely buildable):

- **Buildable-now (≈8):** Calendars, Dates, Swap, Repo, YieldCurveTermStructures, Processes, Replication, SobolChartSample (charts → matplotlib).
- **Buildable-now but stubby in Java:** BermudanSwaption, FRA.
- **Blocked pending re-verify / dep landing:** EquityOptions (FD + Barone-Adesi/Bjerksund-Stensland/Ju-quadratic/Integral engines — likely mostly present), Bonds (bond-helper curve bootstrap), VolatilityTermStructures (`ImpliedVolTermStructure`), DiscreteHedging (`ReplicationError` util), CoxRossWithHullWhite, ConvertibleBonds (bucket C — unblocked by W-S4).

---

## Wave structure + tags

Sequenced so dependency producers land before consumers. Each wave = one commit (or a small cluster-batch), green on `uv run pytest` + `uv run pyright` + `uv run ruff check`, two-stage review (spec compliance + code quality), direct-to-main FF-only.

| Wave | Theme | Package | Provenance | Intermediate tag |
|---|---|---|---|---|
| **W-S1** | `XorShiftRandom` (pilot) | `pquantlib-contrib` | B | `pquantlib-siblings-ws1-complete` |
| **W-S2** | Dividend-option compat primitives: `DividendVanillaOption` + `BinomialDividendVanillaEngine` + `BlackScholesDividendLattice` | `pquantlib-helpers` | A | `pquantlib-siblings-ws2-complete` |
| **W-S3** | 6 helper builders + 2 test suites (1:1) | `pquantlib-helpers` | A | `pquantlib-siblings-ws3-complete` |
| **W-S4** | ConvertibleBond subsystem: `ConvertibleFixedCouponBond` + `SoftCallability`/`CallabilitySchedule` + `BinomialConvertibleEngine` | pquantlib **core** | C | `pquantlib-siblings-ws4-complete` |
| **W-S5** | Sample programs + `__main__` entries + pytest smoke, in complete/incomplete/pending buckets | `pquantlib-samples` | mixed | `pquantlib-siblings-complete` |

W-S1 doubles as the pilot that proves the per-package skeleton, test wiring, and review loop before fan-out. W-S2 must precede W-S3 (helpers depend on the compat layer). W-S4 must precede the ConvertibleBonds sample in W-S5. Within W-S5, buildable-now samples can fan out in parallel.

---

## Sample-program form (binding for W-S5)

Each Java sample → one Python module under `pquantlib_samples/` with:
- an importable function (e.g. `run()`) that performs the computation and **prints to stdout** like the Java original (parity of console behaviour);
- an `if __name__ == "__main__":` entry so it runs as a script (`uv run python -m pquantlib_samples.<name>`);
- a **pytest smoke test** under `pquantlib-samples/tests/` asserting the sample runs to completion (mirrors `TestSamples.testCompleteSamples` / `testIncompleteSamples`; `pending` bucket → `@pytest.mark.skip` mirroring Java `@Ignore`).
- Charts (SobolChartSample) → matplotlib, **saved to a file** under a temp/output dir, the plotting call **skipped in CI** (guarded so headless test runs don't require a display).

`AllSamples` → a Python module exposing `complete`, `incomplete`, `pending` tuples of sample modules; the smoke tests iterate these, preserving the Java categorization.

Util files: `StopClock` → a small timing context manager / helper; `ReplicationError` + `ReplicationPathPricer` → ported as the DiscreteHedging sample's support (MC replication, Derman–Kamal).

---

## Discipline (inherited from Phases 0–11, binding)

- C++ v1.42.1 source of truth where a C++ equivalent exists; provenance buckets above govern the exceptions, each documented inline with `# C++ parity:`.
- TDD + cross-validation: every functional change backed by a probe emitting reference JSON consumed by tests via `pquantlib.testing.tolerance` (`exact`/`tight`/`loose`), per-test justification for any exception.
- Subagent fan-out (up to 5 concurrent, worktree-isolated where files would collide) off a sequential pilot; two-stage review (spec compliance + code quality).
- Direct-to-main, FF-only, no PRs, no `Co-authored-by` trailer, `-s` Signed-off-by, unsigned commits. One wave (or cluster-batch) = one commit, all gates green.
- Divergence found mid-stub → separate preceding `align(<module>): …` commit.

## Closure (binding 8-step doc sweep)

On `pquantlib-siblings-complete`: refresh README + CLAUDE.md + this design's completion doc (`phase12-completion.md`) + `docs/carve-outs.md` (remove ConvertibleBond from the deferred list; note the retired-API compat layer) + memory (`phase_status.md`). Tag `pquantlib-siblings-complete`.

## Open risks

- **Audit accuracy:** the samples "blocked" list was grep-derived; re-verify each missing class against Python-idiom names at W-S5 dispatch before declaring a sample blocked.
- **XorShift integer semantics:** Java `long` is signed 64-bit with `>>>` unsigned shift; Python ints are unbounded. Match via explicit `& 0xFFFFFFFFFFFFFFFF` masking / numpy `uint64`, cross-validated against Java seeded output.
- **Convertible engine provenance:** confirm whether v1.42.1 ships `BinomialConvertibleEngine` in `ql/` or `ql/experimental/`; port from wherever the v1.42.1 canonical implementation lives.

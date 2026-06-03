# Phase 12 — Sibling-packages migration (executable plan)

> **For agentic workers:** use `superpowers:subagent-driven-development` (fresh subagent per cluster + two-stage review) to execute. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Port `jquantlib-contrib`, `jquantlib-helpers`, `jquantlib-samples` into the PQuantLib workspace members `pquantlib-contrib`, `pquantlib-helpers`, `pquantlib-samples`, behind the final tag `pquantlib-siblings-complete`.

**Architecture:** Provenance-bucketed ground truth (design §"API-era reconciliation"): retired dividend-option API → compat layer inside `pquantlib-helpers`; `XorShiftRandom` → `pquantlib-contrib` (Java-output truth); `ConvertibleBond` → pquantlib core (closes carve-out). Same probe-driven TDD + two-stage review + direct-to-main discipline as Phases 0–11.

**Tech stack:** Python 3.14, numpy/scipy, pytest 9+, pyright strict, ruff. uv workspace. C++ v1.42.1 @ `099987f0` probes where a C++ equivalent exists.

**Predecessor:** `pquantlib-final` @ `1fdb1db` — 4048/0/0.
**Date:** 2026-06-02.
**Source root (Java):** `/Users/josemoya/eclipse-workspace/jquantlib/{jquantlib-contrib,jquantlib-helpers,jquantlib-samples}`.

---

## Probe-driven TDD loop (template — applied per functional cluster)

```
1. Read the Java original(s) + the matching C++ v1.42.1 header(s) where applicable.
2. Write/extend the C++ probe at migration-harness/cpp/probes/cluster_<name>/probe.cpp,
   build via migration-harness/build-cpp.sh, run it to emit
   migration-harness/references/cluster/<name>.json.
   (Bucket-B XorShiftRandom: no C++ — emit the reference from JQuantLib Java output instead.)
3. Write the failing pytest that loads reference_reader.load("cluster/<name>") and
   asserts via tolerance.{exact,tight,loose}. Run it — confirm it FAILS (import/AttributeError).
4. Implement the Python module(s) (snake_case files, PascalCase classes).
5. Run pytest -> green; run pyright -> clean; run ruff check -> clean.
6. Commit (one cluster = one commit, -s sign-off, no Co-authored-by trailer).
```

Verify triad after every commit: `uv run pytest && uv run pyright && uv run ruff check`.

---

## W-S1 — contrib pilot: `XorShiftRandom` (bucket B)

Single cluster. Establishes per-package test wiring; smallest surface.

**Java source:** `jquantlib-contrib/.../math/randomnumbers/XorShiftRandom.java`
**Target module:** `pquantlib-contrib/src/pquantlib_contrib/math/randomnumbers/xorshift_random.py`
**Test:** `pquantlib-contrib/tests/math/randomnumbers/test_xorshift_random.py`
**Reference:** `migration-harness/references/cluster/ws1.json` (Java-output, NOT C++)

- [ ] **Step 1 — emit the Java reference sequence.** Add `migration-harness/java/Ws1Emitter.java`: instantiate `XorShiftRandom(seed)` for a fixed seed (e.g. `42`) and print, as JSON, the first 16 `nextLong()` values and first 16 `nextDouble()` values. Run it against the jquantlib classpath; capture stdout to `migration-harness/references/cluster/ws1.json`. (This is the agreed bucket-B ground truth: there is no C++ equivalent.)
- [ ] **Step 2 — failing test.** Load `ws1.json`; assert `XorShiftRandom(42).next_long()` reproduces each reference `nextLong` via `tolerance.exact` (integers, bit-identical) and `next_double()` via `tolerance.exact`. Run → FAIL (module missing).
- [ ] **Step 3 — implement.** Port the algorithm with explicit 64-bit wraparound to match Java signed `long` + `>>>`:

```python
_MASK64 = (1 << 64) - 1

class XorShiftRandom:
    """Marsaglia 2003 xorshift64 RNG.

    # C++ parity: none — XorShiftRandom is a JQuantLib-original convenience
    # RNG with no QuantLib v1.42.1 equivalent. Ground truth is JQuantLib Java
    # output. Java `long` is signed 64-bit with `>>>` (unsigned) shift; we
    # reproduce that with explicit & _MASK64 masking on unsigned state.
    """
    def __init__(self, seed: int) -> None:
        self._x = seed & _MASK64

    def next_long_unsigned(self) -> int:
        x = self._x
        x ^= (x << 13) & _MASK64
        x ^= x >> 7
        x ^= (x << 17) & _MASK64
        self._x = x & _MASK64
        return self._x

    def next_long(self) -> int:
        u = self.next_long_unsigned()
        return u - (1 << 64) if u >= (1 << 63) else u  # signed view, matches Java

    def next_double(self) -> float:
        return (self.next_long_unsigned() >> 11) / float(1 << 53)
```

- [ ] **Step 4 — green triad + commit.** `feat(contrib/ws1): XorShiftRandom (Marsaglia xorshift64, Java-output cross-validated)`.
- [ ] **Step 5 — tag** `pquantlib-siblings-ws1-complete`.

**Risk:** confirm the signed-vs-unsigned reference convention — the Java test (if any) and `nextLong()` return signed; `next()`/`nextDouble()` derive from the unsigned shift. The two helper methods above keep both views explicit. Re-derive from the Java if the `exact` assertion fails.

---

## W-S2 — helpers compat primitives (bucket A)

Single cluster (the three classes are tightly coupled). Hosted **in `pquantlib-helpers`**, built on pquantlib-core v1.42.1 primitives.

**Java sources:**
- `jquantlib-helpers/.../instruments/` ⟵ retired core `DividendVanillaOption` (re-created here)
- `jquantlib-helpers/.../methods/lattices/BlackScholesDividendLattice.java`
- `jquantlib-helpers/.../pricingengines/vanilla/BinomialDividendVanillaEngine.java`

**Target modules (in `pquantlib-helpers/src/pquantlib_helpers/`):**
- `instruments/dividend_vanilla_option.py` — `DividendVanillaOption(OneAssetOption)` carrying a `DividendSchedule` (list of `Dividend` cashflows) + payoff + exercise.
- `methods/lattices/black_scholes_dividend_lattice.py` — `BlackScholesDividendLattice` (subclasses the core BS lattice; subtracts escrowed PV-of-dividends at each node).
- `pricingengines/vanilla/binomial_dividend_vanilla_engine.py` — `BinomialDividendVanillaEngine[Tree]` building a CRR/Tian tree over a GBSM process and applying dividends.

**Core primitives to reuse (verified present):** `OneAssetOption`/`VanillaOption` (`option.py`), `Dividend`/`FixedDividend` (`cashflows/dividend.py`), `BinomialVanillaEngine`, CRR + `ExtendedTian` lattices, `GeneralizedBlackScholesProcess`. **Verify at dispatch:** whether `DividendSchedule` needs adding (audit said MISSING) — if so add a thin `dividend_schedule.py` = `list[Dividend]` alias + builder in `pquantlib_helpers` (it is retired-API glue, not core).

### Cross-validation strategy (corrected — same-algorithm peer)

These are **bucket-A retired-API** engines: C++ v1.42.1 deleted them, so there is no C++ same-method peer. A tree/FD price compared against a C++ *closed form* disagrees by **O(1/N) discretization error** (~1e-3 at the helper's N, ~10⁸ steps needed for 1e-8) **plus a possible dividend-model difference** (escrowed-spot vs dividend-node-shift) that no N closes. So tree-vs-analytic is the wrong gate. Instead:

- **Primary gate — JQuantLib Java output, same algorithm + same N → TIGHT (target EXACT).** The Python port reproduces the Java `BinomialDividendVanillaEngine` / `FDDividend*Engine` step-for-step. This is the real parity test and needs **no A2 waiver**.
- **Secondary economic sanity — C++ `AnalyticDividendEuropeanEngine`, European only**, at a documented ~1e-3 absolute tolerance via `tolerance.custom(..., reason="CRR/FD tree discretization vs C++ closed-form dividend engine at N=<steps>; O(1/N) convergence")`. This is a sanity bound, **not** the correctness gate, so its >1e-8 tolerance is acceptable by design rather than an A2 exception.

**References:**
- `references/cluster/ws2.json` (C++ v1.42.1): the test scenarios from the two Java suites priced via `AnalyticDividendEuropeanEngine` (European) + `FdBlackScholesVanillaEngine` with a `DividendSchedule` (American/escrowed). Used for the **secondary** sanity check only.
- `references/cluster/ws2_java.json` (JQuantLib Java): the **same** scenarios priced via the Java `BinomialDividendVanillaEngine` / `FDDividend*Engine` at fixed `N` (NPV + greeks). Used for the **primary** TIGHT/EXACT gate. Emitted by a small `migration-harness/java/Ws2Emitter.java` run against the jquantlib classpath (the retired engines live there).

- [ ] **Step 1a** — extract the scenarios from the two Java test files (`CRRDividendOptionTest`, `FDDividendOptionTest`); write the C++ `probe.cpp`; build; emit `ws2.json` (secondary sanity).
- [ ] **Step 1b** — write `Ws2Emitter.java` pricing the same scenarios with the Java retired engines at fixed `N`; emit `ws2_java.json` (primary gate).
- [ ] **Step 2** — failing tests: **primary** assertions against `ws2_java.json` via `tolerance.tight` (drop to `exact` where the arithmetic is deterministic across JVM↔CPython); **secondary** European assertions against `ws2.json` via `tolerance.custom(abs=~1e-3, reason=...)`. Run → FAIL.
- [ ] **Step 3** — implement the three classes; match the Java tree construction (node count `N`, up/down/prob, dividend handling) exactly so the primary gate holds at TIGHT/EXACT.
- [ ] **Step 4** — green triad + commit `feat(helpers/ws2): DividendVanillaOption + BinomialDividendVanillaEngine + BlackScholesDividendLattice (retired-API compat layer, Java same-algorithm cross-validated)`.
- [ ] **Step 5** — tag `pquantlib-siblings-ws2-complete`.

**A2 note:** the only >1e-8 tolerance here is the *secondary* economic sanity check, which is loose **by design** (tree-vs-closed-form), not a correctness gate. The correctness gate (primary) is TIGHT/EXACT, so **no A2 pause is expected in W-S2**. If the primary Java gate itself cannot reach TIGHT (e.g. an unavoidable JVM↔CPython float-order difference), *that* is the real surprise → pause and report.

---

## W-S3 — helpers builders 1:1 (bucket A)

Single cluster. Depends on W-S2.

**Java sources → Target modules (`pquantlib_helpers/helpers/`):**
- `CRRDividendOptionHelper.java` → `crr_dividend_option_helper.py` (abstract base; `extends DividendVanillaOption`)
- `FDDividendOptionHelper.java` → `fd_dividend_option_helper.py` (abstract base)
- `CRREuropeanDividendOptionHelper.java` → `crr_european_dividend_option_helper.py`
- `CRRAmericanDividendOptionHelper.java` → `crr_american_dividend_option_helper.py`
- `FDEuropeanDividendOptionHelper.java` → `fd_european_dividend_option_helper.py`
- `FDAmericanDividendOptionHelper.java` → `fd_american_dividend_option_helper.py`

**Tests (1:1 port of the Java suites):**
- `CRRDividendOptionTest.java` → `pquantlib-helpers/tests/helpers/test_crr_dividend_option.py`
- `FDDividendOptionTest.java` → `pquantlib-helpers/tests/helpers/test_fd_dividend_option.py`

Each base builds a `BlackScholesMertonProcess` from flat r/q/vol `SimpleQuote`s (`FlatForward` + `BlackConstantVol`), attaches the W-S2 engine, exposes `vega`/`rho` (bump-and-revalue) + `implied_volatility`. Default `NullCalendar` + `Actual360` (per Java). Java constructor overload sets → Python keyword defaults (`cal=NullCalendar()`, `dc=Actual360()`).

- [ ] **Step 1** — port the two test suites verbatim (same scenarios). Numeric assertions follow the W-S2 strategy: **primary** TIGHT/EXACT against `ws2_java.json` (same-algorithm Java output), **secondary** European sanity at ~1e-3 against `ws2.json` (C++ analytic). Run → FAIL.
- [ ] **Step 2** — implement the 2 bases + 4 concrete helpers.
- [ ] **Step 3** — green triad + commit `feat(helpers/ws3): 6 dividend-option helper builders + test suites (1:1)`.
- [ ] **Step 4** — tag `pquantlib-siblings-ws3-complete`.

---

## W-S4 — ConvertibleBond subsystem into core (bucket C)

Closes `docs/carve-outs.md:172`. Largest wave. May split into 2 clusters (instrument vs engine) if context-heavy.

**C++ v1.42.1 sources:** `ql/instruments/bonds/convertiblebonds.{hpp,cpp}`, `ql/instruments/callabilityschedule.hpp`, and the binomial convertible engine (**verify location at dispatch:** `ql/pricingengines/bond/` vs `ql/experimental/`).

**Target modules (pquantlib core, `pquantlib/src/pquantlib/`):**
- `instruments/callability_schedule.py` — `Callability`, `SoftCallability`, `CallabilitySchedule` (**verify** whether already present in core first; audit suggested missing).
- `instruments/bonds/convertible_bonds.py` — `ConvertibleBond` base + `ConvertibleFixedCouponBond` (+ `ConvertibleZeroCouponBond`, `ConvertibleFloatingRateBond` if the sample needs only the fixed-coupon variant, port just that one and note the others as follow-ons).
- `pricingengines/bond/binomial_convertible_engine.py` — `BinomialConvertibleEngine[Tree]` (Tsiveriotis–Fernandes credit-adjusted tree).

### Cross-validation strategy (same-method tree-vs-tree)

Unlike W-S2, `BinomialConvertibleEngine` **is a real v1.42.1 class**, so the proper peer is the **C++ same engine, same method, same `N`** — tree-vs-identical-tree, which agrees step-for-step (no model difference, only shared discretization that is identical on both sides). So the gate is **TIGHT** against C++, *not* a loose tree-vs-analytic comparison.

- **Primary gate — C++ `BinomialConvertibleEngine`, same `N` → TIGHT.** Match the C++ tree construction (steps, lattice, Tsiveriotis–Fernandes credit-adjusted rollback) exactly. No A2 waiver expected.
- No analytic closed form exists for a callable convertible, so there is no separate "analytic" sanity tier; the C++ same-engine value *is* the reference.

**Probe `cluster_ws4`** (C++ v1.42.1): construct the `ConvertibleBonds` sample's instrument (conversion ratio, credit spread, dividends, callability) and price via `BinomialConvertibleEngine` at fixed `N`; emit NPV (and, if cheap, a few intermediate tree diagnostics) to `references/cluster/ws4.json`.

- [ ] **Step 1** — verify which of `CallabilitySchedule`/convertible engine already exist in core (grep Python-idiom names); scope to only the genuinely-missing pieces.
- [ ] **Step 2** — write `probe.cpp` (fixed `N`, record it in the JSON so the Python test uses the identical step count); build; emit `ws4.json`.
- [ ] **Step 3** — failing test asserting convertible NPV via `tolerance.tight` against the **same-`N`** C++ value. (If TIGHT proves unreachable despite matching `N` and method, that signals an actual algorithm divergence to fix — not a tolerance to loosen. A documented `custom` >1e-8 tolerance here would be an **A2 pause**, but it should not be needed.)
- [ ] **Step 4** — implement callability + instrument + engine, matching the C++ tree step-for-step.
- [ ] **Step 5** — green triad + commit(s) `feat(core/ws4): ConvertibleFixedCouponBond + CallabilitySchedule + BinomialConvertibleEngine (closes carve-out, C++ same-method cross-validated)`.
- [ ] **Step 6** — tag `pquantlib-siblings-ws4-complete`.

---

## W-S5 — sample programs (mixed provenance)

Fan-out by bucket. Each Java sample → one `pquantlib_samples/<name>.py` module with `run()` + `if __name__ == "__main__":`, printing to stdout like the Java original.

**Util (port first — shared deps):** `pquantlib_samples/util/{stop_clock.py,replication_error.py,replication_path_pricer.py}`.
- `StopClock` → timing context-manager / helper.
- `ReplicationError` + `ReplicationPathPricer` → MC replication support for DiscreteHedging (Derman–Kamal).

**Runner:** `pquantlib_samples/all_samples.py` exposing `COMPLETE`, `INCOMPLETE`, `PENDING` tuples of sample modules (mirrors Java `AllSamples`).

**Smoke tests:** `pquantlib-samples/tests/test_samples.py`:

```python
import importlib, pytest
from pquantlib_samples import all_samples

@pytest.mark.parametrize("mod", all_samples.COMPLETE)
def test_complete_samples_run(mod):
    importlib.import_module(f"pquantlib_samples.{mod}").run()

@pytest.mark.parametrize("mod", all_samples.INCOMPLETE)
def test_incomplete_samples_run(mod):
    importlib.import_module(f"pquantlib_samples.{mod}").run()

@pytest.mark.skip(reason="pending bucket — mirrors Java @Ignore")
@pytest.mark.parametrize("mod", all_samples.PENDING)
def test_pending_samples(mod):  # pragma: no cover
    importlib.import_module(f"pquantlib_samples.{mod}").run()
```

**Cluster W-S5-A — buildable-now (parallel):** Calendars, Dates, Swap, Repo, YieldCurveTermStructures, Processes, Replication, SobolChartSample.
- SobolChartSample: chart via matplotlib `savefig` to a temp path, **guarded** so CI/headless skips the plotting call (e.g. `if os.environ.get("PQL_SAMPLES_PLOT"):`). The `run()` still computes the Sobol sequence unconditionally.

**Cluster W-S5-B — dep re-verify then port (parallel):** EquityOptions, Bonds, VolatilityTermStructures, BermudanSwaption, FRA, CoxRossWithHullWhite, DiscreteHedging.
- [ ] **First step for each:** re-verify the audit's "blocked" classes against Python-idiom names in pquantlib core. If present → port to COMPLETE/INCOMPLETE bucket. If genuinely absent and out of scope → place in PENDING with an inline note citing `docs/carve-outs.md`.

**Cluster W-S5-C — ConvertibleBonds** (depends on W-S4): port to the COMPLETE bucket like the Java original.

Per-sample TDD: smoke test (run-to-completion) is mandatory; add a numeric cross-val (reuse an existing core probe, or extend `cluster_ws5`) where the sample prints a known computed value.

- [ ] Commit cadence: one commit per cluster (or per sample for the heavier ones). Final: `feat(samples/ws5): sample programs + AllSamples runner + smoke suite`.
- [ ] **Tag** `pquantlib-siblings-complete`.

---

## Closure — binding 8-step doc sweep

- [ ] `docs/migration/phase12-completion.md` (per-wave contribution, final test count).
- [ ] `docs/carve-outs.md` — remove `ConvertibleBond` from the deferred bond list (W-S4 ported it); add a short note documenting the retired dividend-option compat layer in `pquantlib-helpers`.
- [ ] `README.md` + `CLAUDE.md` — update Current-state to reflect the three siblings ported + new test count.
- [ ] Memory `phase_status.md` — siblings complete.
- [ ] Confirm `uv run pytest && uv run pyright && uv run ruff check` clean on `main`.
- [ ] Tag `pquantlib-siblings-complete`; push `main` + tags.

---

## Self-review notes (spec coverage)

- Design buckets A/B/C → W-S2+W-S3 (A) / W-S1 (B) / W-S4 (C). ✓
- "compat layer in pquantlib-helpers" → W-S2 target paths under `pquantlib_helpers/`. ✓
- "samples as runnable scripts + pytest smoke, complete/incomplete/pending" → W-S5 form + runner + smoke suite. ✓
- ConvertibleBonds = COMPLETE sample (user chose port-subsystem) → W-S4 → W-S5-C. ✓
- XorShift Java-output ground truth → W-S1 step 1. ✓
- **A2 (corrected):** correctness gates now use **same-algorithm peers** — Java same-engine output for the retired W-S2/W-S3 dividend engines (TIGHT/EXACT), C++ same-`N` engine for the W-S4 convertible (TIGHT). No A2 pause is expected. The only >1e-8 tolerance is W-S2's *secondary* economic sanity check (tree-vs-C++-analytic, European-only), which is loose **by design**, not a correctness exception. A2 fires only if a *primary* same-algorithm gate cannot reach TIGHT — which would indicate a real bug to fix, not a tolerance to relax.
- **Audit-accuracy risk:** W-S5-B first step re-verifies each "blocked" class before bucketing.

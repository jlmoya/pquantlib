# Phase 6 — L6-C: DoubleBarrier instrument + AnalyticDoubleBarrierEngine

**Branch:** `phase6-C`
**Worktree:** `/Users/josemoya/Projects/PycharmProjects/pquantlib-phase6-C`
**Baseline:** 1883/0/0

## Scope

Close Phase 5 `DoubleBarrierOption` carve-out: introduce the instrument
+ Ikeda-Kunitomo (1992) closed-form analytic engine for KnockIn /
KnockOut variants. KIKO / KOKI are NOT supported by the analytic engine
(C++ `QL_FAIL` on those branches) so the engine merely rejects them
with `LibraryException`.

## C++ sources

* `ql/instruments/doublebarriertype.hpp` — `DoubleBarrier::Type` enum
  (`KnockIn`, `KnockOut`, `KIKO`, `KOKI`).
* `ql/instruments/doublebarrieroption.{hpp,cpp}` — instrument +
  arguments + base engine.
* `ql/pricingengines/barrier/analyticdoublebarrierengine.{hpp,cpp}` —
  Ikeda-Kunitomo series engine.
* `ql/pricingengines/barrier/analyticdoublebarrierbinaryengine.{hpp,cpp}` —
  C.H.Hui series engine for one-touch binary double barriers.
  **DEFERRED** (see "Optional binary engine" below).

## Python modules added

### `pquantlib.instruments.double_barrier_option`

* `DoubleBarrierType(IntEnum)` — `KnockIn=0` / `KnockOut=1` / `KIKO=2` /
  `KOKI=3`. Integer values match C++ declaration order.
* `DoubleBarrierOptionArguments(OptionArguments)` — adds
  `barrier_type`, `barrier_lo`, `barrier_hi`, `rebate` slots. `validate`
  asserts barrier_type ∈ enum, both barriers non-None, rebate non-None.
* `DoubleBarrierOption(barrier_type, barrier_lo, barrier_hi, rebate,
  payoff, exercise)` — subclass of `OneAssetOption`. Mirrors
  `BarrierOption` shape. `is_expired() → False` (Settings.evaluation_date
  is a Phase 1 carve-out, matches existing pattern). `setup_arguments`
  populates the four fields onto a `DoubleBarrierOptionArguments`.
  `implied_volatility` is **carved out** (matches Phase 3
  `VanillaOption.implied_volatility` carve-out — requires
  `ImpliedVolatilityHelper`, not ported).

### `pquantlib.pricingengines.barrier.analytic_double_barrier_engine`

* `AnalyticDoubleBarrierEngine(process, series=5)` —
  `GenericEngine[DoubleBarrierOptionArguments, OneAssetOptionResults]`.
* Constructor registers with `process` (observer pattern, same as
  existing single-barrier engine).
* `calculate()`:
  - Rejects non-European exercise with `LibraryException`.
  - Rejects non-`PlainVanillaPayoff` with `LibraryException`.
  - Rejects `strike <= 0` with `LibraryException`.
  - Rejects `spot <= 0` with `LibraryException`.
  - Rejects already-triggered barrier with `LibraryException`.
  - For `KnockIn`: `vanillaEquivalent - callKO/putKO`, clipped at 0.
  - For `KnockOut`: `callKO()` / `putKO()` Ikeda-Kunitomo series.
  - For `KIKO`/`KOKI`: `LibraryException("unsupported double-barrier
    type")` — matches C++ `QL_FAIL` exactly.
* Helper methods mirror C++ accessors directly:
  `_underlying`, `_strike`, `_residual_time`, `_volatility`,
  `_volatility_squared`, `_barrier_lo`, `_barrier_hi`, `_std_deviation`,
  `_risk_free_rate`, `_risk_free_discount`, `_dividend_yield`,
  `_cost_of_carry`, `_dividend_discount`, `_vanilla_equivalent`,
  `_call_ko`, `_put_ko`, `_call_ki`, `_put_ki`.
* `_vanilla_equivalent` uses `BlackCalculator` at forward.
* Series indexed `n ∈ [-series, +series]` (2*series+1 terms).
* The math-symbol variables in `_call_ko`/`_put_ko` (`d1`/`d2`/`d3`/`d4`
  for call, `y1`/`y2`/`y3`/`y4` for put, `acc1`/`acc2`, `mu1`,
  `bsigma`, `L2n`, `U2n`, `rend`, `kov`) are kept as-is per the
  math-symbol-name allowance, with `# noqa: N802, N803, N806`.

### Optional binary engine: DEFERRED

`AnalyticDoubleBarrierBinaryEngine` is 307 lines of CHHui-series code
that supports both European and American exercise styles, eight binary
payoff combinations (cash/asset × KnockIn/KnockOut × Call/Put with
strike comparisons), and produces Greeks via numerical differentiation
on top of the closed-form value. Per the task spec ("Defer if
non-trivial") and the layer budget (~4 classes), this is carved out
to a follow-up cluster. Documented in `phase6-L6-C-completion.md` and
listed in `docs/carve-outs.md` (to be created in Phase 6 closure).

## Probe

`migration-harness/cpp/probes/cluster_l6c/probe.cpp` →
`migration-harness/references/cluster/l6c.json`. CMakeLists entry
added under the comment block `# === L6-C cluster ===`.

Cases (single underlying setup unless otherwise noted; `T=1y`,
day-counter `Actual365Fixed`, calendar `NullCalendar`, reference
`15 June 2026`):

1. **Haug textbook KO call** — S=100, K=100, L=80, U=120, r=5%, q=0%,
   σ=20%, rebate=0, series=5. KnockOut Call.
2. **In-out parity (KnockIn = European − KnockOut)** — same params as
   case 1, but KnockIn Call. Cross-checked TIGHT via direct closed-form
   vs vanilla-minus-KO.
3. **Asymmetric L=90 / U=130 KO call** — S=100, K=100, L=90, U=130,
   r=5%, q=2%, σ=25%.
4. **KO put** — S=100, K=100, L=80, U=120, r=5%, q=0%, σ=20%, rebate=0.
5. **KI put** — same as case 4 but KnockIn.
6. **Convergence (series=5 vs series=10)** — case 1 with both series
   values, asserting TIGHT match (the Ikeda-Kunitomo series converges
   geometrically; ≥5 terms is well within machine precision for these
   barrier widths).

The probe also emits the European vanilla NPV (via
`AnalyticEuropeanEngine`) for the same params to allow the
in-out-parity test to load both values from a single JSON.

## Tolerance

Closed-form Ikeda-Kunitomo series with rapid (geometric) convergence
→ **TIGHT** (`abs_tol=1e-14`, `rel_tol=1e-12`). No LOOSE-tier
expected.

## Tests added (planned)

| Test file | Cases |
|---|---|
| `tests/instruments/test_double_barrier_option.py` | 6 — type ints, constructor field round-trip, `is_expired() → False`, `setup_arguments` populates correctly, `DoubleBarrierOptionArguments.validate` accepts good / rejects missing barriers / rejects bad enum value. |
| `tests/pricingengines/barrier/test_analytic_double_barrier_engine.py` | 11 — 4 textbook KO (call symmetric + asymmetric, put symmetric) + 2 in-out-parity (call + put) + 1 series-convergence (case 6) + 1 non-European rejection + 1 KIKO rejection + 1 KOKI rejection + 1 triggered-barrier rejection. |

Target delta: **~17 tests**. Slight under-shoot vs plan's ~20 is fine
— the analytic engine has fewer code paths than the single-barrier
engine (no 8 strike-vs-barrier × call/put combinatorics).

## Commit sequence

1. **`feat(instruments): port DoubleBarrierType + DoubleBarrierOption`** —
   instrument + enum + arguments + 6 instrument tests. Triad green.
2. **`feat(barrier): port AnalyticDoubleBarrierEngine`** — Ikeda-Kunitomo
   closed-form engine + 11 engine tests + probe + CMakeLists entry +
   `cluster/l6c.json` reference. Triad green.
3. **`docs(migration): close L6-C`** — adds
   `docs/migration/phase6-L6-C-completion.md` summarizing the cluster.

## Carve-outs documented

* `AnalyticDoubleBarrierBinaryEngine` — 307 lines C.H.Hui series for
  binary double-barriers. Deferred to follow-up.
* `DoubleBarrierOption.implied_volatility` — needs
  `ImpliedVolatilityHelper`, same Phase 3 carve-out as
  `VanillaOption`.
* `DoubleBarrier.KIKO` / `DoubleBarrier.KOKI` analytic-engine support
  — C++ `QL_FAIL`s on these, so the Python engine matches by raising.
  These barrier types are valid on the *instrument*; only the
  closed-form analytic engine refuses them. They would require a
  Monte-Carlo or finite-difference engine to price.

## Cross-cluster blockers

None — all foundations (OneAssetOption, GBSP, BlackCalculator,
CumulativeNormalDistribution, PlainVanillaPayoff, EuropeanExercise) are
already on `main`.

# Phase 6 — L6-C completion

**Cluster:** L6-C — DoubleBarrier instrument + AnalyticDoubleBarrierEngine.
**Branch:** `phase6-C`.
**Commits (2):**

1. `feat(instruments): port DoubleBarrierOption + DoubleBarrierType (L6-C)`
2. `feat(barrier): port AnalyticDoubleBarrierEngine (L6-C)`

(A third closure commit will land alongside this completion doc.)

## Outcome

* **Pre-cluster baseline:** 1883/0/0.
* **Post-cluster:** 1903/0/0 — **delta +20** (= 7 instrument + 13 engine).
* **Tolerance:** TIGHT across all engine cross-validations
  (Ikeda-Kunitomo series converges geometrically; series=5 is
  bit-identical to series=10 for textbook barrier widths).
* **Triad:** pytest + pyright + ruff all green at every commit.

## What landed

### `pquantlib.instruments.double_barrier_option`

* `DoubleBarrierType(IntEnum)` — `KnockIn=0` / `KnockOut=1` / `KIKO=2` /
  `KOKI=3`. Integer values match C++ declaration order.
* `DoubleBarrierOptionArguments(OptionArguments)` — barrier_type +
  barrier_lo + barrier_hi + rebate; validation mirrors C++
  `QL_REQUIRE` chain.
* `DoubleBarrierOption(barrier_type, barrier_lo, barrier_hi, rebate,
  payoff, exercise)` — `OneAssetOption` subclass.

### `pquantlib.pricingengines.barrier.analytic_double_barrier_engine`

* `AnalyticDoubleBarrierEngine(process, series=5)` — KnockOut family
  via Ikeda-Kunitomo (1992) closed-form series; KnockIn via the
  vanilla-minus-KO identity using a `BlackCalculator` at forward.
* KIKO / KOKI raise `LibraryException("unsupported double-barrier
  type: ...")` matching the C++ `QL_FAIL`.

## Probe + reference

* `migration-harness/cpp/probes/cluster_l6c/probe.cpp` — captures 7
  reference values (textbook KO call/put, KI call/put, European vanilla
  call/put for in-out-parity, KO call at series=10, asymmetric KO
  call).
* `migration-harness/references/cluster/l6c.json` — committed.
* CMakeLists entry: `# === L6-C cluster: DoubleBarrierOption +
  AnalyticDoubleBarrierEngine ===`.

## Divergences

* `_volatility` calls `BlackVolTermStructure.black_vol(last_date, ...)`
  whereas C++ calls `blackVol(residualTime(), ...)`. The Python API
  binds the `Date` overload only; both routes resolve to identical
  variance via the day-counter's `year_fraction`. Inline comment
  documents the choice.
* `calculate()` is restructured to a single flat `if/elif/elif/elif/else`
  chain instead of nested `if (option_type) { switch (barrier_type)
  ... }` — ruff PLR5501 prefers it and the branch table is small enough
  that flat is clearer.

## Carved out (deferred)

| Item | Reason | Suggested follow-up |
|---|---|---|
| `AnalyticDoubleBarrierBinaryEngine` (C.H.Hui 1996 series) | 307 lines, depends on binary payoffs + American exercise + numerical-diff Greeks; "Defer if non-trivial" per task brief | Separate cluster e.g. `L6-D` once binary single-barrier engine demand emerges |
| `DoubleBarrierOption.implied_volatility` helper | Same Phase 3 carve-out as `VanillaOption.implied_volatility` — needs `ImpliedVolatilityHelper` plumbing | Re-evaluate when an FD double-barrier engine lands |
| `DoubleBarrierType.KIKO` / `KOKI` analytic-engine support | C++ `QL_FAIL`s on those branches — the closed-form Ikeda-Kunitomo series doesn't decompose to them | Requires MC or FD engine — outside L6-C scope |

## Cross-cluster blockers

None. All foundations were already on `main` (OneAssetOption,
GBSP, BlackCalculator, CumulativeNormalDistribution,
PlainVanillaPayoff, EuropeanExercise).

## Hand-off

* Branch `phase6-C` is fast-forward ready against `main`.
* Worktree clean; no carry-overs into Phase 6 closure.

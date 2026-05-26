# Phase 1 — L1 math primitives + time + foundations (design)

**Date:** 2026-05-23
**Status:** **closed** — tagged `pquantlib-phase1-complete` @ `edcadbc` on 2026-05-24. **581/0/0** pytest, pyright + ruff clean. Closure summary at [`phase1-completion.md`](phase1-completion.md).
**Predecessor:** `pquantlib-phase0-bootstrap` @ `85018e5`
**Sister-project anchor:** jquantlib `phase2-l1-plan.md` + 4 sub-cluster plans (B/C/D/E)
**C++ ground truth:** QuantLib v1.42.1 @ `099987f0`

## Outcome (filled in at closure)

Phase 1 shipped as 5 clusters across two stages of work:

- **L1-A pilot** (sequential, 6 stages): 415/0/0 tests. Tagged `pquantlib-phase1-l1-A-complete` @ `03d0ce8`. See [`phase1-l1-A-completion.md`](phase1-l1-A-completion.md).
- **L1-B / L1-C / L1-D / L1-E** (4 parallel cluster subagents, ~25 min wall-clock): +166 tests, total 581/0/0.

The design's "~500 ported class" target was not hit — actual realized scope was tighter (the must-port subset per cluster). Carve-outs documented in [`phase1-completion.md`](phase1-completion.md).

## Goal

Port the L1 layer of QuantLib v1.42.1 to Python: math primitives, time machinery, day counters, currencies, plus the small but pervasive foundation modules (exceptions, patterns, utility types) that the rest of the library depends on.

Phase 1 closes when:

1. Every L1 class is either ported with C++-cross-validated tests, or annotated `# C++ parity:` with a deliberate divergence note.
2. The full pytest suite is green at every tolerance tier.
3. `uv run pyright`, `uv run ruff check`, `uv run ruff format --check` all clean.
4. Tag `pquantlib-phase1-complete` pushed.
5. `docs/migration/phase1-completion.md` written.

## Why this is bigger than jquantlib's L1

JQuantLib Phase 2 L1 was a **gap-fill** of 132 classes on top of a pre-existing 2007-era Java port. PQuantLib is **greenfield** — there's no existing skeleton, so the L1 scope is the *full* C++ math+time+foundations layer, not just the missing bits.

Estimated scope (sized from jquantlib's complete L1 footprint, which faithfully mirrors C++ v1.42.1):

| Sub-layer | Approx. class count | Notes |
|---|---|---|
| `math` (root + transcendental + functions) | ~30 | Constants, Closeness, ErrorFunction, Beta, Factorial, Rounding (4 variants), BernsteinPolynomial, PascalTriangle, GeneralLinearLeastSquares, Quadratic, BSpline, etc. |
| `math.copulas` | 13 | Full Archimedean + extreme-value family |
| `math.distributions` | ~30 | Normal/Bivariate/Student/Poisson/Binomial/Gamma/ChiSq families + Maddock/Moro inverse |
| `math.integrals` | ~45 | Gauss quadrature family + Simpson/Trapezoid + Kronrod + Filon + ExpSinh/TanhSinh + DiscreteSimpson/Trapezoid + GaussianOrthogonalPolynomial hierarchy |
| `math.interpolations` | ~50 | Linear/Cubic family + Akima/Kruger/FritschButland/Harmonic/Lagrange/Chebyshev/SABR/XABR + Log* variants + 2D (Bilinear/Bicubic/Kernel) |
| `math.matrixutilities` | ~25 | Mostly delegated to numpy/scipy where possible; port only what C++ uses idiomatically (Cholesky/LU/QR/SVD wrappers, PseudoSqrt, EigenvalueDecomposition, SymmetricSchurDecomposition, SparseMatrix, BiCGStab, GMRES, HouseholderReflection/Transformation, OrthogonalProjections) |
| `math.ode` | 1 | OdeFctWrapper — likely scipy-backed |
| `math.optimization` | ~30 | LM/BFGS/CG/Simplex/SimulatedAnnealing/DifferentialEvolution + Constraint hierarchy + LineSearch family + EndCriteria + Problem/CostFunction interfaces |
| `math.randomnumbers` | ~30 | Mersenne Twister + Sobol (incl. Burley2020) + Halton + Faure + KnuthUniform + LecuyerUniform + Ranlux64/Ranlux + Xoshiro256** + GaussianRng family (BoxMuller, CL, Ziggurat) + InverseCumulative wiring + PrimitivePolynomials/LatticeRules |
| `math.solvers1D` | 9 | Bisection, Brent, FalsePosition, FiniteDifferenceNewtonSafe, Halley, Newton, NewtonSafe, Ridder, Secant |
| `math.statistics` | ~15 | GeneralStatistics + Incremental + Gaussian/Risk/Sequence stats + Histogram + Convergence/Discrepancy stats + StatsHolder + DoublingConvergenceSteps |
| `time` (Date, Calendar, Period, Schedule, ASX/ECB/IMM, etc.) | ~20 | Date arithmetic is the foundation everything else depends on |
| `time.calendars` | ~45 | Full sovereign + exchange + joint/null/bespoke set |
| `daycounters` | ~12 | Actual360, Actual364, Actual36525, Actual365Fixed, Actual366, ActualActual, Business252, OneDayCounter, SimpleDayCounter, Thirty360, Thirty365 + base DayCounter |
| `patterns` + `util` | ~12 | Observer/Observable, LazyObject, Singleton, ObservableSettings, Visitor/Visitable family, Pair, ComparablePair, Std |
| `exceptions` | 1 | LibraryException (+ `QL.require` / `QL.fail` helpers as a tiny module, not a class) |
| `currencies` | ~9 | Money + ExchangeRate + ExchangeRateManager + per-continent factories (Africa/America/Asia/Europe/Oceania) |

**Total: ~450 classes.** This is the L1 *target* — but many are tiny (10–80 lines), and several blocks (matrixutilities, ode, random sequence machinery) are heavily delegated to numpy/scipy/mpmath instead of literally translated.

If audit (pre-cluster work in L1-A) reveals scope materially exceeds 500 ported classes, **trigger A1** and re-scope. We will not silently grow.

## Scope (in)

- All classes listed in the table above, ported with `# C++ parity:` line references.
- The shared support layer that lands as part of L1-A (the pilot):
  - `pquantlib.exceptions.LibraryException` + `pquantlib.exceptions.qassert.require` / `.fail` helpers (mirror C++ `QL_REQUIRE` / `QL_FAIL`).
  - `pquantlib.testing.tolerance` with `exact`, `tight`, `loose`, `custom(reason=...)`.
  - `pquantlib.testing.reference_reader.load("<topic>/<class>")` loader for the JSON-encoded C++ probe outputs.
  - Submodule clone at `migration-harness/cpp/quantlib/` pinned to `099987f0`.
  - `migration-harness/cpp/probes/CMakeLists.txt` template + at least one real probe used by L1-A.
- Per-cluster design + plan docs (`docs/migration/phase1-l1-<A..E>-design.md` + `…-plan.md`).
- Closure doc `docs/migration/phase1-completion.md` summarizing what landed, what was deferred with rationale, and any divergences from C++.

## Scope (out — deferred to Phase 2+)

- Term structures (yield curves, vol surfaces, default curves) — Phase 2.
- Indexes (IBOR, OIS, inflation, swap) — Phase 2.
- Instruments + pricing engines — Phase 3.
- Models — Phase 4.
- Experimental + L6 test-suite parity — Phase 5.
- Any class in `ql/experimental/` (regardless of layer) — deferred until Phase 5.

## Approach

### Cluster decomposition

Mirror jquantlib's 5-worktree pattern, but **L1-A is intentionally bigger** because it carries the foundation modules (exceptions, testing helpers, harness, Date/Calendar/DayCounter pilot) that every other cluster will depend on. L1-B…E land in parallel after L1-A merges.

| Cluster | Scope | Approx. classes | Worktree | Depends on |
|---|---|---|---|---|
| **L1-A (pilot)** | Foundations (`exceptions`, `testing`, `patterns`, `util`) + `time` (Date, Calendar core, Period, Schedule, DateGeneration, BusinessDayConvention, Frequency, ASX, ECB, IMM) + `daycounters` (all 12) + `time.calendars` (full set, ~45) + first batch of `math` root (Constants, Closeness, Rounding 4 variants, Factorial, ErrorFunction, Beta, BernsteinPolynomial, PascalTriangle) + harness wiring (submodule clone, first probe, reference loader) | ~120 | `pquantlib-phase1-A` | — |
| **L1-B** | `math.copulas` (13) + `math.distributions` (all ~30) + `math.statistics` (all ~15) + `currencies` (9) | ~67 | `pquantlib-phase1-B` | A |
| **L1-C** | `math.integrals` (~45) + `math.solvers1D` (9) + `math.ode` (1) | ~55 | `pquantlib-phase1-C` | A |
| **L1-D** | `math.randomnumbers` (~30) + `math.optimization` (~30) | ~60 | `pquantlib-phase1-D` | A |
| **L1-E** | `math.matrixutilities` (~25) + `math.interpolations` (~50) + remaining `math` root + `math.functions` + `math.transcendental` | ~75 | `pquantlib-phase1-E` | A, E may pull from numpy/scipy heavily |

Note the cluster count totals to ~377 vs the ~450 estimate above: the gap is *audit slack*. L1-A will do a precise per-class audit during its pilot phase and re-allocate before B/C/D/E dispatch.

### Sequencing

```
L1-A (pilot, alone)
  └─ merge to main
        ├─ L1-B ┐
        ├─ L1-C ├─ in parallel (up to 4 worktrees)
        ├─ L1-D ┤
        └─ L1-E ┘
              └─ merge to main (FF, no PR), in landing order
                    └─ tag pquantlib-phase1-complete
```

### Per-cluster discipline

Same pattern as jquantlib's `phase2-l1-*-plan.md`:

1. Worktree spawned at `../pquantlib-phase1-<X>` off `main`.
2. Implementer subagent (`Agent` → `general-purpose` or a custom L1 implementer) executes the cluster plan task-by-task. Subagent uses the `superpowers:subagent-driven-development` skill where applicable.
3. **Two-stage review** before merge:
   - Spec-compliance reviewer agent: verifies every ported class has a `# C++ parity:` line, a backing C++ probe (or documented exemption), and tier-correct tolerance.
   - `pr-review-toolkit:code-reviewer` agent: idiomatic Python, type strictness, ruff/pyright clean.
4. Fix-up loop until both reviewers are clean.
5. FF-merge to `main`. No PR. Worktree + branch cleaned up local+remote.

### TDD + cross-validation contract

Every functional change follows this loop:

1. Write a C++ probe at `migration-harness/cpp/probes/<topic>/<class>_probe.cpp`.
2. Add it to the harness CMakeLists.
3. Run `./migration-harness/generate-references.sh <topic>/<class>_probe`.
4. The probe writes `migration-harness/references/<topic>/<class>.json`.
5. Write the pytest test that loads the JSON and compares via tolerance helpers — **fail first** (red).
6. Implement the Python port.
7. Test goes green.
8. Commit (`feat(<topic>): port <ClassName>` or per-batch equivalent).

**No inline hand-derived expected values.** If a value can't be obtained from a C++ probe (e.g., it's a property like idempotence, not a numeric output), use mpmath as the secondary ground truth and document inline.

## Python-specific translation cheatsheet (L1-flavored, extends CLAUDE.md)

| C++ idiom | Java analogue (jquantlib) | Python idiom (pquantlib) |
|---|---|---|
| `QL_REQUIRE(cond, msg)` | `QL.require(cond, msg)` | `if not cond: raise LibraryException(msg)` |
| `Array` (boost::numeric::ublas vector) | custom `Array.java` over `Cells.$` raw double[] | `numpy.ndarray[float64]` (rank-1) directly; no wrapper class. Add a typing alias `Array = npt.NDArray[np.float64]` in `pquantlib.math.array`. |
| `Matrix` (boost::numeric::ublas matrix) | custom `Matrix.java` | `numpy.ndarray[float64]` (rank-2); typing alias `Matrix = npt.NDArray[np.float64]`. |
| `Date` (custom serial-number class) | custom `Date.java` (serial-day arithmetic, 1901-01-01 epoch) | port `pquantlib.time.Date` — do NOT use stdlib `datetime`. Same serial-day arithmetic. |
| `Calendar` (polymorphic, holiday list) | abstract class + per-country impl | abstract class + per-country impl (Python class inheritance). |
| `DayCounter` (polymorphic) | abstract class + ~12 concrete | abstract class + ~12 concrete. |
| C++ template `Singleton<T>` | per-class `getInstance()` | `pquantlib.patterns.singleton.Singleton[T]` as a base class using `__init_subclass__` for safety. |
| `Observable` / `Observer` (boost::signals2) | hand-rolled with WeakReference | hand-rolled with `weakref.WeakSet` of observers. |
| C++ enum (`Frequency`, `BusinessDayConvention`, `TimeUnit`, `Weekday`) | Java enum | `enum.IntEnum` (matches C++ integral values exactly for boundary tests). |
| C++ template `Solver1D<Impl>` (CRTP) | Java abstract class + subclass | abstract `Solver1D` + subclasses. No CRTP gymnastics needed in Python. |
| C++ template `GenericPseudoRandom<URNG, IC>` | Java generic class | use a `@dataclass(frozen=True, slots=True)` config + functions. No need for the C++ template machinery. |
| C++ functor `boost::function<Real(Real)>` | `Ops.DoubleOp` SAM interface | `Callable[[float], float]` from `typing`. |
| Inner anonymous classes (jquantlib `XABRSpecs` etc.) | `static class` inside parent | nested class or module-level helper class. |
| `std::vector<Real>` | `double[]` or `List<Double>` | prefer `npt.NDArray[np.float64]` for performance; use `list[float]` only where order-preserving append is needed. |
| `std::pair` | `Pair<L,R>` | `tuple[A, B]` or `@dataclass(frozen=True, slots=True)` if it needs a name. |
| C++ pre-2011 result-via-output-param (`f(x, &out)`) | Java return-value-class | return a tuple, or a `@dataclass(frozen=True, slots=True)` result struct. |

## Decision log (Phase 1)

| # | Decision | Why |
|---|---|---|
| 1 | L1-A is the pilot and absorbs the entire foundation surface (exceptions, testing, patterns, util, time, daycounters, calendars, first math) | Foundations must land first so B/C/D/E can depend on them. Bundling them into A means no half-built infra blocking the parallel clusters. |
| 2 | Greenfield port the full L1, not a gap-fill | PQuantLib has no prior code — there are no "already-present" classes to filter out. Audit happens within L1-A to refine the precise count before B/C/D/E dispatch. |
| 3 | `Array` / `Matrix` are typing aliases over `np.ndarray`, not wrapper classes | Python idiom; matches numpy ecosystem. jquantlib's `Cells.$` raw-double pain was a Java artifact (no proper double-array generics) that doesn't exist in Python with numpy. Documented as a deliberate divergence in the port of `ql/math/array.hpp`. |
| 4 | `Date` is a custom class, not stdlib `datetime` | C++ QuantLib uses serial-day arithmetic with specific epoch + leap-year handling needed for cross-validation to be bit-exact. stdlib `datetime` would force constant translation. Port verbatim. |
| 5 | `Observer` / `Observable` ported (not "modernized away") | Used pervasively in termstructures (L2) and instruments (L3). Replacing it with `asyncio` or pub/sub libraries is a Phase 6 modernization concern, not L1. |
| 6 | `math.matrixutilities.Matrix`/`Array` linear algebra delegates to numpy/scipy where applicable | numpy provides correctness-equivalent and faster ops. Port: keep API shape (`CholeskyDecomposition`, `LUDecomposition`, `SVD`, etc.) but back the actual factorization with `scipy.linalg`. Document divergence per file. |
| 7 | Random number generators ported bit-exactly | Sobol, MersenneTwister, Xoshiro256** are deterministic — their sequence MUST match C++ for cross-validation to work in upstream stages (Monte Carlo pricing). EXACT tier required. |
| 8 | `mpmath` is the secondary ground truth where probes can't be written | E.g., for properties like idempotence or extreme-precision transcendentals. Documented inline. |
| 9 | Currencies land in L1-B alongside copulas/distributions | They're independent of the other math clusters and L2 (`termstructures`) will want them immediately. Small enough not to need their own cluster. |
| 10 | C++ submodule cloned at the start of L1-A | First probe writing requires it. The clone command is one line; pre-A is the natural moment. |
| 11 | Pyright `extraPaths` + `venvPath` config (from Phase 0 verification) is permanent | Without it, pyright doesn't resolve workspace-member imports. Documented in `phase0-completion.md`. |
| 12 | A cluster commit is allowed to bundle multiple tiny stubs (e.g., all 4 `Rounding` variants in one commit) if they share a probe and a test file | Reduces commit noise. Constraint: the commit must still be reversible standalone (no cross-cluster dependencies in one commit). |

## Tolerance discipline (binding for Phase 1)

Tier assignments are made *per test*, not per class. Defaults:

| Predicate | Default tier | When to override |
|---|---|---|
| Closed-form algebraic identity (e.g., `Beta(a,b) * (a+b) / (a*b) = ...`) | **EXACT** | Only if FP non-associativity is provably introduced by a known intermediate. |
| Closed-form transcendental (e.g., `ErrorFunction`, `NormalDistribution.cdf`) | **TIGHT** | Loosen only with mpmath-justified rationale. |
| Special-function series approximations (e.g., `IncompleteGamma`, `RegularisedIncompleteBeta`) | **TIGHT** | Loosen to LOOSE if the C++ algorithm and the mpmath reference agree only to ~10 digits. |
| Iterative methods (`Brent`, `Newton`, `LevenbergMarquardt`, `DifferentialEvolution`) | **LOOSE** | Document termination tolerance + step count match in inline comment. |
| Quadrature methods (`GaussKronrod`, `GaussLegendre`) | **TIGHT** at convergence, **LOOSE** for partial sums | Per-test inline note. |
| Random sequence generators | **EXACT** for the first N values, where N is the count emitted by the C++ probe (typically 100–1000) | Any divergence requires escalation (A3). |
| Date arithmetic | **EXACT** (integer-valued under the hood) | Never override. |
| DayCounter year-fraction | **TIGHT** | Loosen only for `ActualActual.ISMA` corner cases per documented C++ behavior. |

`pquantlib.testing.tolerance` will expose:

```python
def exact(actual: float, expected: float, *, reason: str | None = None) -> bool: ...
def tight(actual: float, expected: float, *, reason: str | None = None) -> bool: ...
def loose(actual: float, expected: float, *, reason: str | None = None) -> bool: ...
def custom(actual: float, expected: float, *, abs_tol: float, rel_tol: float, reason: str) -> bool: ...
```

All four are assertion helpers (raise on mismatch with a structured message). `reason` is required for `custom` and optional-but-encouraged for `exact`/`tight`/`loose` overrides.

## Worktree topology

- 5 worktrees at most: `../pquantlib-phase1-{A,B,C,D,E}` (sibling of repo root, NOT nested inside).
- Each branched off `main` at the L1-A merge commit.
- FF-merge to `main` in landing order.
- Worktree + branch removed local+remote post-merge (use `commit-commands:clean_gone` or equivalent).
- L1-A runs alone first; the pilot's discoveries (audit refinements, divergence patterns) feed into the B–E plan files before they dispatch.

## Pause triggers (Phase 1)

Same A1–A8 as Phase 0, plus:

| ID | Condition | Action |
|---|---|---|
| A1' | Audit during L1-A shows the true ported-class count exceeds **500** | Pause, present a re-scoping proposal (defer entire `math.experimental.*`, split L1-E into two clusters, etc.) |
| A2' | A test needs tolerance looser than `LOOSE` (1e-8) | Pause, document why; consider whether the C++ algorithm itself converges this poorly, or whether the test set-up is wrong |
| A3' | C++ probe output and a Python implementation match to fewer than 8 digits, AND mpmath agrees with Python (not C++) | Pause — this may be a v1.42.1 bug. Log per A3. |
| A4' | A cluster needs a Python dependency outside `numpy`, `mpmath`, `scipy` | Pause and request approval. Likely candidates that would still need approval: `numba`, `cython`, `pyarrow`. |
| A5' | An L2-bound API change is required at L1 time (e.g., termstructure-aware interpolation hook) | Pause. Don't fold L2 concerns into L1. |
| A6 | End of Phase 1 | Report summary, wait for ack before Phase 2 design. |

## Definition of done (Phase 1)

- [ ] L1-A merged to `main` with foundation modules + time + daycounters + calendars + first math batch landed.
- [ ] L1-B, L1-C, L1-D, L1-E each merged to `main` (any landing order).
- [ ] `uv run pytest`: full suite green at every tier.
- [ ] `uv run pyright`: 0 errors across all `src/` directories.
- [ ] `uv run ruff check` + `uv run ruff format --check`: clean.
- [ ] Every ported class has a `# C++ parity:` line citing v1.42.1 source.
- [ ] Every numerical test loads its expected values from `migration-harness/references/<topic>/<class>.json` (no inline hand-derived values, except those explicitly justified inline per A3/A3').
- [ ] `phase1-completion.md` written with: actually-ported count vs estimate, list of deferred classes with rationale, list of inline divergences from C++, list of cluster-specific lessons.
- [ ] Tag `pquantlib-phase1-complete` pushed.

## Next-step preview (immediate)

After this design is acked:

1. Write `docs/migration/phase1-l1-A-design.md` (the pilot cluster's binding spec).
2. Write `docs/migration/phase1-l1-A-plan.md` (the pilot cluster's executable task list).
3. Clone the C++ submodule + build it (one-time setup per `migration-harness/README.md`).
4. Dispatch L1-A.

The B/C/D/E design + plan docs are drafted *after* L1-A lands so they can incorporate audit refinements and discovered patterns.

# Claude Code bootstrap — PQuantLib migration

## What this repo is

PQuantLib is a Python port of the C++ [QuantLib](https://github.com/lballabio/QuantLib) quantitative finance library — a sister project to **JQuantLib** (Java port at `/Users/josemoya/eclipse-workspace/jquantlib`, tagged `jquantlib-final`).

This port is brand-new (started 2026-05-23). It uses the **same migration patterns** that JQuantLib refined over its multi-session migration:

- C++ v1.42.1 is source of truth (pinned commit `099987f0`)
- Tier-stratified tolerances (EXACT / TIGHT / LOOSE) with per-test inline justification
- Cross-validation against C++ probe values (no inline hand-derived expected values)
- Subagent-driven development with two-stage review (spec compliance + code quality)
- Worktree-parallel implementer dispatch (up to 5 concurrent)
- Direct-to-main per layer (no PRs; solo single-owner repo)
- Phase tags + completion docs in `docs/migration/`

## Read this first, every session

**`docs/migration/phase0-design.md`** is the binding design spec for the bootstrap phase. Read it before doing anything.

Once Phase 0 is closed (project skeleton green + lint+typecheck+test pass on `hello world`), Phase 1 starts on the math-primitives layer. The phase-1 design will mirror jquantlib's Phase 2 L1 (math/utilities/time/patterns).

## Ground-truth principle

**C++ QuantLib v1.42.1 is source of truth.** Where Python idioms force divergence (no inheritance from generics, dataclass-vs-Pydantic, async vs threads), the divergence is documented inline with a `# C++ parity:` comment citing the C++ source line.

Pin: `v1.42.1` @ `099987f0ca2c11c505dc4348cdb9ce01a598e1e5` (2026-04-16).

## Current state

- **Phase:** **Phase 1 complete** — all 5 L1 clusters closed and tagged `pquantlib-phase1-complete` @ `edcadbc`.
- **Branch:** `main`. No active feature branches. Workflow for the next phase will mirror what L1 used: per-cluster worktrees, push direct to `main` per cluster (FF-only).
- **Workspace:** uv-managed 4-package monorepo (`pquantlib` + `pquantlib-{contrib,helpers,samples}`). scipy added as a dep in L1-E.
- **Python:** 3.14. **Type checker:** pyright strict. **Lint+format:** ruff. **Test framework:** pytest 8+, **currently 581/0/0**.
- **L1 layer landed (Phase 1):** foundations (exceptions, qassert, testing, patterns), time core (Date, Period, Calendar abstract + 4 trivial + 41 sovereign/exchange, Schedule + MakeSchedule, IMM/ASX/ECB, TimeGrid/TimeSeries), day counters (DayCounter abstract + 11 concretes), 8 first-math modules, 12 copulas, 3 normal distributions, 2 statistics aggregators, 5 currencies, 9 Solver1D concretes, 5 simple integrals, 5 deterministic RNGs (all EXACT-tier bit-exact) + BoxMuller, 7 optimization scaffolding modules, 4 interpolations + bilinear, scipy-backed Cholesky.
- **Parallelization wins:** Stage 4 (41 calendars) and L1-B/C/D/E (~166 tests across 4 clusters) each ran in ~25 min wall-clock via subagent fan-out. Pattern documented in `phase1-completion.md`.
- **Cumulative L1 carve-outs** (deferred): full GaussianOrthogonalPolynomial hierarchy, Sobol/Burley2020 low-discrepancy, LM/BFGS/Simplex optimizers, 8+ cubic spline variants, QR/Eigen/SVD/SparseMatrix utilities, GammaFunction (factorial/beta still LOOSE-tier vs `math.lgamma`). Listed in `phase1-completion.md`.
- **Next phase:** **Phase 2 (L2)** — termstructures + indexes, mirroring jquantlib's `phase2-L2-termstructures-indexes-plan.md`.

## Sibling repo (read-only reference)

`/Users/josemoya/eclipse-workspace/jquantlib` (tag `jquantlib-final`) is the Java port that this project mirrors. When in doubt about migration discipline, look at how jquantlib handled the equivalent C++ class. Don't blindly translate Java → Python — adapt to Python idioms — but match the layer sequencing (L1 math → L2 termstructures+indexes → L3 instruments+pricingengines → L4 models → L5 experimental → L6 test-suite parity).

## Operational rules (binding)

Same as JQuantLib's:

- **Push direct to `main` per cluster** (fast-forward only, no squash). No PRs.
- **No `Co-authored-by: Claude` trailer.** `-s` Signed-off-by trailer yes. Unsigned commits (no GPG/SSH).
- **One stub (or cluster-batch) = one commit.** Every commit passes `uv run pytest` + `uv run pyright` + `uv run ruff check`.
- **TDD + cross-validation.** Every functional change is backed by a C++ probe at `migration-harness/cpp/probes/` that emits reference values to `migration-harness/references/<topic>.json`. Tests load the JSON and compare via the tolerance helpers in `pquantlib.testing.tolerance`.
- **Tolerance tiers:**
  - `exact(actual, expected)` — bit-identical via `struct.pack('!d', x)` comparison.
  - `tight(actual, expected)` — `math.isclose(abs_tol=1e-14, rel_tol=1e-12)`.
  - `loose(actual, expected)` — `math.isclose(abs_tol=1e-8, rel_tol=1e-8)`.
  - Per-test exceptions require inline written justification.
- **Divergence found mid-stub** → separate preceding `align(<module>): ...` commit, not folded into the implementation commit.
- **API changes to match v1.42.1 are automatic.** No per-change approval needed.

## Python-specific translation cheatsheet (Java → Python)

| Java concept | Python analogue |
|---|---|
| `JUnit @Test` | `pytest` `def test_xxx()` |
| `JUnit assertEquals(expected, actual, tol)` | `pquantlib.testing.tolerance.tight(actual, expected)` |
| `QL.require(cond, msg)` | `if not cond: raise LibraryException(msg)` |
| `LibraryException` | `pquantlib.exceptions.LibraryException` (RuntimeError subclass) |
| `@Deprecated` | `@deprecated("reason")` (PEP 702, 3.13+) |
| `@SuppressWarnings("deprecation")` | `# pyright: ignore[reportDeprecated]` |
| Record (JDK 16+) | `@dataclass(frozen=True, slots=True)` |
| Sealed interface (JDK 17+) | `Union[A, B, C]` + `typing.assert_never(default_arm)` |
| Pattern matching (JDK 21+) | `match-case` (PEP 634+) |
| `var` | type inference — just omit annotation on locals |
| `JDK 25 t-strings` | Python 3.14 PEP 750 t-strings |
| `Maven mvn test` | `uv run pytest` |
| `Maven multi-module` | uv workspace (`[tool.uv.workspace]` in root pyproject) |
| Javadoc | Google-style or reStructuredText docstrings |
| `Cells.$ raw double[] access` | `numpy.ndarray[float64]` (cleaner — no Address-mapping needed) |
| `JQuantMath correctly-rounded transcendentals` | `mpmath` (already arbitrary-precision); or `numpy` for batch |
| `Java Date` | Use `pquantlib.time.Date` (port the C++ Date class, don't use stdlib `datetime`) |

## When to pause and ask the user

Default: autonomous work. Pause only for these triggers (full list per JQuantLib's design discipline):

| Trigger | Condition |
|---------|-----------|
| A1 | Phase scope > 1000 classes |
| A2 | Tolerance looser than 1e-8 needed |
| A3 | Cross-validation suggests v1.42.1 itself is wrong |
| A4 | Stub needs a Python dep not in the locked workspace |
| A6 | End of every phase — report summary, wait for ack |

## Environment gotchas

- **uv workspace:** run `uv run <cmd>` from repo root; uv resolves the correct member package automatically. To target one package: `uv run --package pquantlib pytest`.
- **C++ clone:** the `migration-harness/cpp/quantlib/` submodule is independent. Build it once with `migration-harness/build-cpp.sh` (creates `migration-harness/cpp/build/`).
- **GH account:** the `jlmoya` GitHub account owns this repo. If multiple `gh` accounts are configured, run `gh auth switch -u jlmoya` first.
- **Remote URL is SSH** (`git@github.com:jlmoya/pquantlib.git`).
- **PyCharm:** the `.idea/` and `.venv/` directories are PyCharm-managed; don't commit anything inside `.venv/`.

## Quick resume checklist for a fresh session

1. `git status`, `git branch --show-current` — confirm on `main`.
2. Read `docs/migration/phase<current>-design.md` to know what's in scope.
3. Read `docs/migration/phase<current>-plan.md` if it exists (executable plan).
4. `uv sync` — ensure dependencies match the lockfile.
5. `uv run pytest` — confirm a known-green baseline before changing anything.
6. `uv run pyright` — confirm types are clean.
7. `uv run ruff check` — confirm lint is clean.
8. Pick the next task from the plan; dispatch via subagent if it's an implementer task; do it inline if it's design/coordination work.

## Mapping JQuantLib phases to PQuantLib phases

JQuantLib's journey (for reference; we don't have to repeat all of it):
- Phase 1: 80 stubs in 61 existing packages (Java had a pre-existing 2007-era port; we're starting from scratch so this is different)
- Phase 2 L1-L6: forward closure of ~458 missing classes against C++ test-suite parity
- Phase 3: closure of all carry-forwards (UOE stubs, @Ignore tests, TODO/FIXME)
- JDK 25 modernization (W1-W4 — cosmetic, pattern matching, records, sealed types)
- Final state: tag `jquantlib-final` @ 3610/0/0/21 BUILD SUCCESS

PQuantLib mapping (proposed; ratify in `phase0-design.md`):
- **Phase 0:** project skeleton bootstrap (this one)
- **Phase 1:** L1 — math primitives (`Array` via numpy, `Date`, `Calendar`, `DayCounter`, distributions, integrals, interpolations, RNGs)
- **Phase 2:** L2 — termstructures + indexes
- **Phase 3:** L3 — instruments + pricingengines
- **Phase 4:** L4 — models
- **Phase 5:** L5 — experimental + L6 test-suite parity
- **Phase 6:** Modernization sweep (use Python 3.14 features idiomatically: t-strings, PEP 695 generics, match-case, async where natural)
- **Phase 7:** Final closure + carve-out documentation + tag `pquantlib-final`

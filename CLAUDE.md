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

- **Phase:** **Phase 11 in progress — W1–W5 closed.** Latest tag `pquantlib-phase11-w5-complete` @ `6e0f8d2` (3299/0/0). W5 (experimental/finitedifferences) merged cleanly on resume after the computer-restart checkpoint. **Next: W6** (experimental/volatility + experimental/math) per [`docs/migration/phase11-plan.md`](docs/migration/phase11-plan.md).
- **Branch:** `main` @ `6e0f8d2`. No active feature branches between waves.
- **Workspace:** uv-managed 4-package monorepo. Dependencies: numpy, scipy, pytest, pyright, ruff. No new deps in Phase 11.
- **Python:** 3.14. **Type checker:** pyright strict. **Lint+format:** ruff. **Test framework:** pytest 9+, **main currently 3299/0/0**.
- **L1 layer (Phase 1):** foundations, time core, day counters, 8 first-math modules, copulas + distributions + statistics, currencies, Solver1D + integrals, deterministic RNGs, optimization scaffolding, interpolations + matrix utils.
- **L2 layer (Phase 2):** quotes, termstructures core + 4 cross-cluster Protocols, Index + IndexManager, Compounding + InterestRate, YieldTermStructure + concrete curves, InterestRateIndex hierarchy + 8 ibor concretes + 2 swap indexes + 7 rate helpers, cashflows + CashFlows aggregator + Duration, volatility termstructures (Black/Local Constant/Curve/Surface).
- **L3 layer (Phase 3):** Settings.evaluation_date observable; Payoff + Exercise + Option + Instrument + PricingEngine + GenericEngine + BlackFormula + 3 cross-cluster Protocols; Bond + 4 concretes + DiscountingBondEngine + BondForward; Swap + VanillaSwap + OIS + ZeroCoupon + make_vanilla_swap + make_ois + DiscountingSwapEngine; StochasticProcess + GBSM family + VanillaOption + EuropeanOption + AnalyticEuropeanEngine + BinomialVanillaEngine + BlackCalculator; Forward + Position + FxForward + ForwardRateAgreement + DiscountingFwdEngine.
- **L4 layer (Phase 4):** **LevenbergMarquardt + Simplex** (closes Phase 1 carry-over) via scipy wrappers; Parameter hierarchy + Model + CalibratedModel + TermStructureConsistentModel + CalibrationHelper bases + 3 cross-cluster Protocols (Model / CalibrationHelper / ShortRateModel); ShortRateModel + OneFactorModel + OneFactorAffineModel + **Vasicek + HullWhite + CoxIngersollRoss + ExtendedCoxIngersollRoss**; HestonProcess + HestonModel + HestonModelHelper + BatesProcess + BatesModel + **AnalyticHestonEngine** (scipy.quad over Gatheral CF); TwoFactorModel + G2Process + G2ForwardProcess + HullWhiteForwardProcess + CoxIngersollRossProcess + OrnsteinUhlenbeckProcess + **G2++**; **Swaption + CapFloor instruments** (closes Phase 3 carve-out); SwaptionHelper + CapHelper; **BlackSwaptionEngine + BachelierSwaptionEngine + JamshidianSwaptionEngine + G2SwaptionEngine + BlackCapFloorEngine + BachelierCapFloorEngine + AnalyticCapFloorEngine**.
- **L5 layer landed (Phase 5):** Sobol + Burley2020 + GammaFunction Lanczos + AkimaCubic + BivariateCumulativeNormalDistribution (5 Phase-1 carry-overs closed); Tree[T] + Lattice base + DiscretizedAsset hierarchy + 3 cross-cluster Protocols; BinomialTree concretes refactored + TrinomialTree + TreeLattice1D + BlackScholesLattice + DiscretizedSwap/Swaption/CapFloor; **TreeSwaptionEngine + TreeCapFloorEngine + BlackKarasinski** (3 Phase-4 carve-outs closed); MC framework (Path + MultiPath + BrownianBridge + PathGenerator + McSimulation + MonteCarloModel) + MCVanillaEngine + MCEuropeanEngine + MCDiscreteArithmeticAveragePriceEngine + Analytic Geometric Asian engines + DiscreteAveragingAsianOption; FD framework (18 modules: FdmLinearOpLayout + meshers + TripleBandLinearOp + FirstDeriv/SecondDeriv + FdmBlackScholesOp + schemes + step conditions + solver) + **FdBlackScholesVanillaEngine + VanillaOption.implied_volatility** (Phase 3 carve-out closed); 6 exotic instrument families (Asian/Barrier/Basket/Lookback/Cliquet/Digital) + 6 analytic engines (Kemna-Vorst / Reiner-Rubinstein / Stulz / Conze-Viswanathan); 2 additional payoffs (FloatingType / PercentageStrike).
- **Parallelization wins:** Phase 1 (~25 min), Phase 2 (~35 min), Phase 3 (~50 min), Phase 4 (~50 min), Phase 5 (~50 min) all via subagent fan-out off a sequential pilot. Cross-cluster Protocols proved out as integration glue across all 5 phases. Pattern documented in `phase{1,2,3,4,5}-completion.md`.
- **Cumulative L1+L2+L3+L4+L5 carve-outs** (deferred): mostly permanent out-of-scope items now — MarketModels (125 files), ZABR/SABR/XABR vol, all inflation, all credit, capfloor/optionlet/swaption volatility surfaces, specialty short-rate (Gaussian1d/GSR/MarkovFunctional), specialty Heston variants, volatility models. Phase 6 still-open items: LongstaffSchwartz American MC, Heston/G2/Bates MC engines, multi-asset FD, double-barrier/partial-barrier/soft-barrier, compound/chooser options.
- **L6 layer landed (Phase 6):** **Phase 4 BatesEngine** (closes L4-C carve-out via add_on_term hook); **LongstaffSchwartz American MC** (LsmBasisSystem + LongstaffSchwartzPathPricer + MCAmericanEngine — closes L5-C carve-out; matches Longstaff-Schwartz 1998 paper reference 4.478); **DoubleBarrierOption + AnalyticDoubleBarrierEngine** (Ikeda-Kunitomo 1992 — closes L5-E carve-out). The originally-planned "Python 3.14 modernization sweep" was deleted from Phase 6 scope after audit confirmed the codebase was built with modern idioms from day 1 (0 uses of Generic[T] / Optional / Union / List/Dict/Tuple / bare @dataclass).
- **Final closure tooling:** `docs/carve-outs.md` (comprehensive per-category carve-out documentation: specialty domains, engine-pair carry-overs, tooling-boundary replacements, deliberate-deferral follow-ups, items-not-in-C++); `pquantlib-samples/` populated with 4 end-to-end sample programs (vanilla_swap_pricing, heston_calibration, american_option_mc, double_barrier_analytic).
- **Project complete:** **`pquantlib-final` tag** marks the end of the planned migration. Phases 0-6 closed; ~340 classes ported across ~1958 tests (54.2% of jquantlib-final 3610). Vanilla pricing + calibration end-to-end + American MC + analytic exotics covered.
- **L7 layer landed (Phase 7 — opt-in extension):** inflation cluster — InflationIndex hierarchy + InflationTermStructure abstracts + Seasonality + 2 cross-cluster Protocols (L7-A pilot); InterpolatedZero/YoYInflationCurve (L7-B); inflation cashflows + pricers (L7-C); inflation instruments + vol surfaces + ZCIIS/YYIIS/CPISwap/YoYInflationCapFloor + CPI/YoY vol surface abstracts + Constant variants + Bachelier/Black/UnitDisplaced YoY cap/floor engines (L7-D). Tag `pquantlib-phase7-complete` @ `3a7228e`. ~33 classes / +151 tests.
- **L8 layer landed (Phase 8 — opt-in extension):** **Piecewise inflation** (PiecewiseZero/YoYInflationCurve + ZeroCouponInflationSwap/YearOnYearInflationSwap helpers + Zero/YoYInflationTraits + IterativeBootstrap[TS, Traits]; closes L7-Bb piecewise + L2-B IterativeBootstrap); **Credit cluster** (DefaultProbabilityTermStructure + 3 intermediates + FlatHazardRate + 3 interpolated curves + probability traits + PiecewiseDefaultCurve scaffold + Spread/UpfrontCdsHelper + CreditDefaultSwap + Claim + MidPoint/Integral CDS engines; closes Tier-1 credit); **Capfloor/optionlet/swaption vol surfaces** (CapFloorTermVolatilityStructure family + OptionletVolatilityStructure family + OptionletStripper1 + SwaptionVolatilityStructure family + SwaptionVolatilityMatrix; closes Phase 2 capfloor-vol). Tag `pquantlib-phase8-complete` @ `efdfac3`. ~40 classes / +194 tests.
- **L9 layer landed (Phase 9 — opt-in extension):** **Cubic + Bicubic spline interpolators** (CubicInterpolation + CubicNaturalSpline + MonotonicCubicNaturalSpline + BicubicSpline + Interpolation2D abstract + opt-in `interpolator=` kwarg on L8-C CapFloorTermVolCurve/Surface; closes L1-E cubic-family); **Post-L8 ergonomics** (PiecewiseYieldCurve + Discount/ZeroYield/ForwardRate traits + PiecewiseDefaultCurve bootstrap wiring + IsdaCdsEngine + implied_hazard_rate + conventional_spread + MakeCDS factory); **SABR swaption smile cube** (sabr_volatility + sabr_normal_volatility (Hagan 2002) + SabrInterpolation (scipy.optimize.least_squares) + SmileSection abstract + Flat/Interpolated/Sabr/Spreaded SmileSection + SwaptionVolatilityCube abstract + SabrSwaptionVolatilityCube + InterpolatedSwaptionVolatilityCube; closes Phase-8 SABR carve-out). Tag `pquantlib-phase9-complete` @ `7784e94`. ~22 classes / +161 tests.
- **L10 layer landed (Phase 10 — opt-in extension):** **Vol surface tail** (KahaleSmileSection + AtmSmileSection + AtmAdjustedSmileSection + SabrInterpolatedSmileSection + OptionletStripper2 + SabrInterpolation Halton multi-start + HaltonRsg — closes Phase 9 vol-tail residuals + L1-D HaltonRsg carry-over); **Gaussian1d short-rate** (Gaussian1dModel abstract + Gsr concrete + GsrProcess + Gaussian1dSwaptionVolatility — closes Tier-1 specialty short-rate; MarkovFunctional deferred); **Interpolator tail + ZABR** (HymanFilteredCubic + ChebyshevInterpolation + MultiCubicSpline + AbcdInterpolation + zabr_volatility (Hagan ZABR ShortMaturityLognormal/Normal) + ZabrSmileSection — closes L1-E interpolator tail + ZABR closed-form family). Tag `pquantlib-phase10-complete` @ `d3746e4`. ~17 classes / +188 tests.

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

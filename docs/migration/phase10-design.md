# Phase 10 — Vol surface tail + Gaussian1d short-rate + ZABR / interpolator tail (design)

**Date:** 2026-05-29
**Status:** **CLOSED** — tagged `pquantlib-phase10-complete` @ `d3746e4` — 2652/0/0
**Predecessor:** `pquantlib-phase9-complete` @ `37e67e0` — 2464/0/0, pyright + ruff clean
**C++ ground truth:** QuantLib v1.42.1 @ `099987f0`
**Completion summary:** [`phase10-completion.md`](phase10-completion.md)

## Scope

Three independent specialty-domain closures, dispatched as **3 parallel clusters with no pilot** (proven against Phase 6 + Phase 8 topology — each cluster is an independent extension of existing layers with no shared new abstract bases).

### L10-A — Vol surface tail (~6 classes / +30 tests)

Closes the L9-C residual carve-outs around smile sections + the OptionletStripper2 follow-up.

- `pquantlib.termstructures.volatility.kahale_smile_section.KahaleSmileSection(base: SmileSection, atm: float | None = None, exponential_extrapolation: bool = False, deletable_arbitrage_points: bool = False, k_min: float = ..., k_max: float = ...)` — no-arbitrage smile reformulation. Filters/repairs the base smile so that butterfly arbitrage (`d2C/dK2 ≥ 0`) and call-spread arbitrage (`dC/dK ≤ 0`) hold. Iterative: walks strikes outward from ATM, optionally deleting violating points + exponentially extrapolating.
- `pquantlib.termstructures.volatility.atm_smile_section.AtmSmileSection(base: SmileSection, atm: float)` — ATM-locked adapter; `atm_level()` returns the fixed value.
- `pquantlib.termstructures.volatility.atm_adjusted_smile_section.AtmAdjustedSmileSection(base: SmileSection, atm: float | None = None)` — shifts the base smile so that ATM volatility matches a target.
- `pquantlib.termstructures.volatility.sabr_interpolated_smile_section.SabrInterpolatedSmileSection(option_date, forward, strikes, vols, alpha=None, beta=None, nu=None, rho=None, alpha_is_fixed=False, beta_is_fixed=False, nu_is_fixed=False, rho_is_fixed=False, vega_weighted=False, ..., day_counter)` — composition: fits `SabrInterpolation` (L9-C) then wraps the fitted params in a `SabrSmileSection` (L9-C). Convenience entry point for the SABR workflow.
- `pquantlib.termstructures.volatility.optionlet.optionlet_stripper_2.OptionletStripper2(cap_floor_vol_curve, optionlet_stripper_1, switch_strike=None, accuracy=1e-5, max_iterations=100)` — strips ATM caplet vols out of a `CapFloorTermVolCurve` using an already-stripped `OptionletStripper1` reference grid. Brent root-finder per option date.
- `pquantlib.math.interpolations.sabr_interpolation.SabrInterpolation` — add `Halton multi-start` option: `max_guesses: int = 1` and an `endCriteria` aware loop that tries up to `max_guesses` Halton-sampled initial params and keeps the best RMS. C++: `SABRWrapper` / `XABRCoeffHolder::reset()`.

### L10-B — Gaussian1d short-rate cluster (~3 classes / +20 tests)

Closes most of the specialty-short-rate carve-out from Phase 4. **MarkovFunctional deferred to a future cluster** — 542 LOC of bootstrap-against-swaption-strip calibration; needs its own focused effort.

- `pquantlib.models.shortrate.gaussian1d_model.Gaussian1dModel(termStructure)` abstract — base for Gaussian1d short-rate variants. `numeraire(t, y, yts)` + `zerobond(T, t, y, yts)` + `forwardRate(d, t, y, yts)` + `swapRate(d, tenor, t, y, yts)` + `swapAnnuity(d, tenor, t, y, yts)` virtual methods. `stateProcess() -> StochasticProcess1D`. Used by the AMC / Gaussian1dSwaptionEngine path.
- `pquantlib.models.shortrate.gsr.GSR(termStructure, volstepdates: list[Date], volatilities: list[float], reversion: float | list[float], T: float)` — Gaussian short-rate model with piecewise volatility + mean-reversion. `Gaussian1dModel` subclass. C++ has parametric reversion + volatility lattice.
- `pquantlib.termstructures.volatility.swaption.gaussian1d_swaption_volatility.Gaussian1dSwaptionVolatility(calendar, business_day_convention, index_base, day_counter, model: Gaussian1dModel, swaption_engine: SwaptionEngine)` — implied-volatility surface backed out of a Gaussian1d model. Returns the Black vol implied from each swaption NPV.

### L10-C — Interpolator tail + ZABR (~6 classes / +30 tests)

Closes most remaining L1-E interpolator carve-outs + the ZABR family.

- `pquantlib.math.interpolations.hyman_filter.HymanFilteredCubic(x, y)` — Hyman-1983 monotonicity filter applied to a natural cubic spline. **Custom math**, not scipy delegation. Alternative to PCHIP (which matches Fritsch-Carlson). Documented in Phase 9 carve-outs as the C++ canonical monotonic cubic.
- `pquantlib.math.interpolations.chebyshev_interpolation.ChebyshevInterpolation(n: int, f: Callable[[float], float] | None = None, points: PointsType = SecondKind)` — Chebyshev nodes-based interpolation. C++: `ChebyshevInterpolation(size_t n, std::function<Real(Real)> f, PointsType = SecondKind)`. Delegate to `numpy.polynomial.chebyshev.Chebyshev.interpolate`.
- `pquantlib.math.interpolations.multi_cubic_spline.MultiCubicSpline(grid: list[list[float]], values: ndarray)` — n-D cubic. Delegate to `scipy.interpolate.RegularGridInterpolator(method='cubic')` for 2D+; fall back to scipy `CubicSpline` for 1D.
- `pquantlib.math.interpolations.abcd_interpolation.AbcdInterpolation(x, y, a=-0.06, b=0.17, c=0.54, d=0.17, ..., end_criteria=None, optimization_method=None)` — parametric volatility curve `σ(t) = (a + b*t) * exp(-c*t) + d`. Fits 4 params via scipy.optimize.least_squares.
- `pquantlib.termstructures.volatility.zabr_formula.zabr_volatility(strike, forward, expiry, alpha, beta, nu, rho, gamma) -> float` — ZABR closed-form vol. ZABR adds one extra parameter γ (gamma) controlling the elasticity of the variance dynamics; γ=1 collapses to SABR. C++: `ql/termstructures/volatility/zabr.hpp` `ZabrModel::lognormalVolatility`.
- `pquantlib.termstructures.volatility.zabr_smile_section.ZabrSmileSection(option_date, forward, zabr_params: tuple[float, float, float, float, float], day_counter, evaluation: Evaluation = FullFd)` — ZABR smile (5 params: α, β, ν, ρ, γ). `Evaluation` IntEnum (ShortMaturityLognormal / ShortMaturityNormal / Local / FullFd / ProjectedHedge). Port `ShortMaturityLognormal` (closed-form) + `ShortMaturityNormal`; defer the FD evaluations.
- `pquantlib.termstructures.volatility.zabr_interpolated_smile_section.ZabrInterpolatedSmileSection` — fitted ZABR per slice; composition of `ZabrInterpolation` (the fitter) + `ZabrSmileSection`.

## Cluster topology

- **3-parallel-no-pilot** — each cluster extends independent layers:
  - L10-A extends Phase 9's smile-section + Phase 8's optionlet machinery.
  - L10-B extends Phase 4's short-rate model hierarchy.
  - L10-C extends Phase 9's interpolation + smile-section family.
- 3 worktrees spawn directly off `pquantlib-phase9-complete`; 3 subagents dispatch in parallel.

```
main (pquantlib-phase9-complete)
  │
  ├── L10-A parallel worktree → merge
  ├── L10-B parallel worktree → merge
  └── L10-C parallel worktree → merge
  ↓
pquantlib-phase10-complete
```

## Carve-outs (deferred — Phase 11+ or never)

### Vol surface (L10-A leftovers)
- `Gaussian1dSmileSection` (depends on Gaussian1d model — actually moved to L10-B by L10-A reorg).
- `SwaptionVolatilityDelta` family (delta-vs-strike representation).
- `InterpolatedSwaptionVolatilityCube2` variants beyond Cube{1, 2}.

### Specialty short-rate (L10-B leftovers)
- `MarkovFunctional` (542 LOC; bootstrap-against-swaption-strip calibration — its own future cluster).
- `Gaussian1dCapFloorEngine` + `Gaussian1dFloatFloatSwaptionEngine` + `Gaussian1dNonStandardSwaptionEngine` (engines on top of the Gaussian1d model — many more LOC).
- `Gaussian1dGsrProcess` (process companion).

### Interpolator / ZABR (L10-C leftovers)
- `ConvexMonotoneInterpolation` (Hagan-West 2009 convex-monotone curve — specialty).
- `XabrSwaptionVolatilityCube` template generalization (we land SABR-as-only-Xabr in Phase 9; ZABR doesn't currently feed a cube).
- `ZabrInterpolation` (the fitter) — port if simple; defer if calibrating 5 params is too brittle.
- ZABR `Local` / `FullFd` / `ProjectedHedge` evaluation paths (require an FD engine; FD engine carve-out from L5).

### Bigger items (always carve-out unless dedicated phase)
- `MarketModels` (LIBOR Market Model — ~125 files).
- `experimental/credit/*` (CDO, basket CDS, CDS-on-CDS — ~30 files).
- `BootstrapError` / `LocalBootstrap` (alt bootstrap algorithms — IterativeBootstrap is the production one).

## Tolerance discipline

| Cluster | Tier | Justification |
|---|---|---|
| L10-A `KahaleSmileSection` arbitrage repair | LOOSE | Iterative; convergence depends on starting smile pathology. |
| L10-A `AtmSmileSection` / `AtmAdjustedSmileSection` | TIGHT | Closed-form delegation to base smile. |
| L10-A `SabrInterpolatedSmileSection` | LOOSE | Fits SABR (LOOSE) + wraps (TIGHT). End-to-end LOOSE. |
| L10-A `OptionletStripper2` | LOOSE | Brent caplet-by-caplet root-find; ~1e-7. |
| L10-A Halton multi-start | LOOSE | Best-of-N RMS; final params vary by ~1e-6. |
| L10-B `Gaussian1dModel` virtual surface | LOOSE | Numerical integration over 1-D state grid. |
| L10-B `GSR` zero bond / numeraire | LOOSE | Affine-form closed-form; matches C++ to 1e-8. |
| L10-B `Gaussian1dSwaptionVolatility` implied vol | LOOSE | Black/Bachelier inversion over Gaussian1d engine NPV. |
| L10-C `HymanFiltered` cubic at pillars | EXACT | Pillar interpolation matches input exactly. |
| L10-C `HymanFiltered` intermediates | TIGHT | Hyman-1983 filter math is closed-form. |
| L10-C `ChebyshevInterpolation` | TIGHT | Closed-form Chebyshev nodes + barycentric eval. |
| L10-C `MultiCubicSpline` | LOOSE | scipy `RegularGridInterpolator(method='cubic')` may differ from C++ multi-index spline by interior smoothness. |
| L10-C `AbcdInterpolation` fit | LOOSE | Least-squares over 4 params; final params ~1e-6 noise. |
| L10-C ZABR closed-form lognormal/normal | TIGHT | Closed-form Hagan-style; matches C++ to ~1e-14. |
| L10-C `ZabrSmileSection.volatility(K)` | TIGHT | Direct closed-form. |

## Decision log

| Decision | Rationale |
|---|---|
| **3-parallel-no-pilot** | Mirrors Phase 6 + Phase 8 — independent specialty domains, no shared abstracts. |
| **Defer `MarkovFunctional`** | 542 LOC of sophisticated calibration; needs its own cluster. Carve-out recorded. |
| **L10-C ports only ZABR `ShortMaturityLognormal` + `ShortMaturityNormal` evaluations** | The FD evaluations (`Local` / `FullFd` / `ProjectedHedge`) require an FD engine on the ZABR PDE; deferred. |
| **Halton multi-start added as a SabrInterpolation option, not a separate class** | C++ does this via the `XABRCoeffHolder::reset()` loop inside the same fitter; mirroring keeps the API one-method-call. |
| **`MultiCubicSpline` delegates to `RegularGridInterpolator(method='cubic')`** | scipy provides multi-D cubic interpolation that matches our pillar-tier discipline; the C++ implementation is 571 LOC of templated machinery. |
| **`ChebyshevInterpolation` uses numpy.polynomial.chebyshev** | numpy ships first/second-kind nodes + barycentric eval + Newton-form construction; avoids a 60-LOC custom port. |
| **`AbcdInterpolation` ported despite being SABR-superseded** | C++ ABCD is still used for swap-curve shape parameterization in some legacy workflows. Small (~120 LOC) and self-contained. |
| **No A6 pause between cluster merges** | User directive: "Phase 10 in its totality." See [[feedback-phase-runtocompletion]] in agent memory. |

## Plan + executable tasks

See [`phase10-plan.md`](phase10-plan.md).

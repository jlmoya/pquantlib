# Phase 10 — completion

**Date closed:** 2026-05-29
**Tag:** `pquantlib-phase10-complete` @ `d3746e4`
**Baseline → outcome:** 2464/0/0 → **2652/0/0** (+188 tests vs +80 target)
**Triad:** pytest 2652 passed · pyright 0 errors / 0 warnings · ruff All checks passed
**C++ ground truth:** QuantLib v1.42.1 @ `099987f0`

## What landed

Three independent specialty-domain closures, dispatched as **3 parallel clusters with no pilot** (proven against Phase 6 + Phase 8 topology). Run-to-completion per user directive ("Phase 10 in its totality") — no A6 pause between cluster merges.

| Cluster | Branch tip | Classes | Tests delta | Carve-out closed |
|---|---|---|---|---|
| L10-A vol surface tail | `ded3a6d` | ~7 (+ HaltonRsg bonus) | +77 | Phase 9 vol-tail residuals + L1-D HaltonRsg carry-over |
| L10-B Gaussian1d short-rate | `39418a4` | ~4 (incl. GsrProcess) | +41 | Tier-1 specialty short-rate (MarkovFunctional deferred) |
| L10-C interpolator tail + ZABR | `2f117bc` | ~6 | +70 | L1-E interpolator tail + ZABR family closed-form |
| **Total** | — | **~17** | **+188** | 4 multi-phase carve-outs |

## Cluster-by-cluster detail

### L10-A — Vol surface tail (`794cc29` + `197e29f` + `ded3a6d`)

**New modules:**
- `pquantlib.math.randomnumbers.halton.HaltonRsg` — low-discrepancy sequence (closes a never-previously-ported L1-D carry-over).
- `pquantlib.math.interpolations.sabr_interpolation.SabrInterpolation` extended with `max_guesses: int = 1` + `multi_start_seed` kwargs implementing Halton multi-start (C++ `XABRCoeffHolder::reset()` analogue).
- `pquantlib.termstructures.volatility.atm_smile_section.AtmSmileSection` — ATM-locked adapter.
- `pquantlib.termstructures.volatility.atm_adjusted_smile_section.AtmAdjustedSmileSection` — shifts base smile to match target ATM.
- `pquantlib.termstructures.volatility.kahale_smile_section.KahaleSmileSection` — no-arbitrage smile reformulation (butterfly + call-spread arbitrage repair, optional exponential extrapolation).
- `pquantlib.termstructures.volatility.sabr_interpolated_smile_section.SabrInterpolatedSmileSection` — composition wrapper (fits SabrInterpolation + wraps in SabrSmileSection).
- `pquantlib.termstructures.volatility.optionlet.optionlet_stripper_2.OptionletStripper2` — strips ATM caplet vols out of `CapFloorTermVolCurve` using `OptionletStripper1`'s grid as reference. Brent root-finder per option date.

**Divergences:**
- `HaltonRsg` uses `numpy.random.default_rng` for jitter (C++ uses MersenneTwister-backed `RandomSequenceGenerator`); invisible to consumers because no downstream test probes the C++ bit pattern.
- `KahaleSmileSection` inlines `SmileSectionUtils` (moneyness-grid + AF detection) rather than porting it as a separate class. Default moneyness grid is log-moneyness shocks.
- `SabrInterpolatedSmileSection` drops C++ `Handle<Quote>` ctor + `hasFloatingStrikes` (absolute strikes only). LazyObject deferred — fits eagerly in `__init__`.
- `OptionletStripper2` hand-rolls per-caplet repricing via `black_formula` directly because PQuantLib's `BlackCapFloorEngine` doesn't accept an `OptionletVolatilityStructure` ctor overload. Matches the C++ `SpreadedOptionletVolatility(adapter, spreadQuote)` semantics exactly.

**Carve-outs (deferred):**
- `KahaleSmileSection.core_smile` (deep `interpolate=True` iteration with `delete_arbitrage_points`) — structurally implemented but only lightly exercised; C++ test-suite has pathological-arbitrage cases.

### L10-B — Gaussian1d short-rate (`a86b5a7` + `d04afd2` + `39418a4`)

**New modules:**
- `pquantlib.processes.gsr_process.GsrProcess` + `GsrProcessCore` — deterministic-drift Ornstein-Uhlenbeck with piecewise σ. Caspers 2013 (ssrn:2246013) closed-form decomposition.
- `pquantlib.models.shortrate.gaussian1d_model.Gaussian1dModel` abstract — base for Gaussian1d short-rate variants. Virtual `numeraire` / `zerobond` / `forward_rate` / `swap_rate` / `swap_annuity` / `state_process` / `y_grid` methods.
- `pquantlib.models.shortrate.onefactor.gsr.Gsr` — Gaussian short-rate with piecewise volatility + (potentially piecewise) mean-reversion. Subclass of `Gaussian1dModel` + `CalibratedModel`.
- `pquantlib.termstructures.volatility.swaption.gaussian1d_swaption_volatility.Gaussian1dSwaptionVolatility` — implied-vol surface backed out of a Gaussian1d model via Brenner-Subrahmanyan inversion of the engine NPV.

**Divergences:**
- **Affine-form Gsr math:** Cache-heavy `_GsrProcessCore` ports the Caspers 2013 closed forms verbatim. Reversions with `|kappa| < 1e-4` switch to algebraic limits (mirrors C++).
- **Numerical-integration grid:** `Gaussian1dModel.y_grid` returns the standardized state grid exactly per C++ — bit-true match against the probe.
- **Diamond MRO:** `Gaussian1dModel.__init__` skips cooperative `super().__init__()` because subclasses (Gsr) also inherit `CalibratedModel` whose ctor needs `n_arguments` — cooperative chain wouldn't thread that through. Bases initialized in-place (matches G2's pattern). Documented inline.
- `Gaussian1dSwaptionVolatility` engine arg is required (no C++-style default to `Gaussian1dSwaptionEngine` — that engine is deferred).
- C++ catch-all returns 0 on Brenner-Subrahmanyan inversion underflow — PQuantLib matches.

**Carve-outs (deferred):**
- `MarkovFunctional` (542 LOC; second Gaussian1dModel concrete — own future cluster).
- `Gaussian1dGsrProcess` (process companion — separate from `GsrProcess`).
- `Gaussian1dCapFloorEngine`, `Gaussian1dFloatFloatSwaptionEngine`, `Gaussian1dNonStandardSwaptionEngine` (Gaussian1d-driven engines).
- C++ `Gaussian1dSmileSection` IborIndex ctor (depends on `Gaussian1dCapFloorEngine`).

### L10-C — Interpolator tail + ZABR (`e4b820e` + `2a199cd` + `2f117bc`)

**New modules:**
- `pquantlib.math.interpolations.hyman_filter.HymanFilteredCubic` — Hyman-1983 monotonicity filter on a natural cubic spline (custom math; not scipy delegation). Alternative to PCHIP (which is Fritsch-Carlson).
- `pquantlib.math.interpolations.chebyshev_interpolation.ChebyshevInterpolation` — Chebyshev nodes-based interpolation. Delegate to `numpy.polynomial.chebyshev`. C++-style canonical `[-1, 1]` with Python-side linear remap convenience.
- `pquantlib.math.interpolations.multi_cubic_spline.MultiCubicSpline` — n-D cubic. Delegate to `scipy.interpolate.RegularGridInterpolator(method='cubic')`.
- `pquantlib.math.interpolations.abcd_interpolation.AbcdInterpolation` — parametric volatility curve `σ(t) = (a + b*t) * exp(-c*t) + d`. Fits 4 params via `scipy.optimize.least_squares`.
- `pquantlib.math.interpolations.zabr_formula.zabr_volatility(strike, forward, expiry, alpha, beta, nu, rho, gamma, mode)` — ZABR closed-form. Ports `ShortMaturityLognormal` + `ShortMaturityNormal`; FD modes raise `LibraryException`.
- `pquantlib.termstructures.volatility.zabr_smile_section.ZabrSmileSection` — ZABR smile (5 params: α, β, ν, ρ, γ).

**Divergences:**
- **HymanFilteredCubic vs PCHIP:** Both monotonicity-preserving, but Hyman-natural-spline-then-filter (this port) differs from Fritsch-Carlson PCHIP by ~1e-2 at intermediate points; pillar values match both. Same algorithm as C++ → TIGHT cross-validation.
- **ChebyshevInterpolation x_min/x_max remap:** C++ class is canonical-`[-1, 1]` only; Python adds a linear remap convenience.
- **MultiCubicSpline BC:** scipy `RegularGridInterpolator(method='cubic')` uses Hermite cubic with one-sided three-point boundary slopes; C++ composes natural-BC `CubicSpline`s; ~10% diff at interior on coarse 4×4 grids, TIGHT at pillars.
- **AbcdInterpolation optimizer:** Python scipy TRF converges to the global minimum on noiseless data (residuals 1e-13); C++ projected-LM stops at a local minimum (rms ~1.8e-3). Python recovers truth `(a, b, c, d)`; C++ doesn't. **Same "Python is more accurate" pattern as Phase 9 `conventional_spread`.**
- **ZABR short-maturity vs SABR:** `zabr_volatility(γ=1)` is the *leading-order* SABR (closed-form `x(K)` transform, no `d` correction). Not full Hagan 2002 — the `sabr_volatility` from L9-C is.
- **ZABR γ≠1 ODE:** Python `scipy.integrate.solve_ivp(RK45, rtol=1e-5, atol=1e-8)` vs C++ `AdaptiveRungeKutta<Real>(1e-8, 1e-5, 0.0)`; ~1e-5 drift per probe call.

**Carve-outs (deferred):**
- ZABR `LocalVolatility`, `FullFd`, `ProjectedHedge` evaluation modes (require a Dupire 1-D / 2-D PDE engine on the ZABR PDE).
- `ZabrInterpolation` (5-param fitter) — same TRF-vs-projected-LM divergence risk as AbcdInterpolation.
- `ZabrInterpolatedSmileSection` — depends on the deferred fitter.
- `XabrSwaptionVolatilityCube` ZABR specialisation.

## Merge reconciliations

| Conflict | Resolution |
|---|---|
| `migration-harness/cpp/probes/CMakeLists.txt` × 2 (L10-B + L10-C each appended after L10-A's entry) | Standard 3-way: keep both sides, stacked L10-A → L10-B → L10-C entries. |
| No probe leakage into main worktree this phase — subagents respected the worktree boundary cleanly. | — |

## Test-count reconciliation

```
baseline                                2464
  L10-A vol surface tail +77            2541
  L10-B Gaussian1d short-rate +41       2582
  L10-C interpolator tail + ZABR +70    2652
```

The +188 outcome more than doubles the +80 plan target because:
- L10-A overshot (+77 vs +30) — the smile-section family + Halton multi-start + HaltonRsg all carry rich test grids.
- L10-C overshot (+70 vs +30) — 6 new classes each with multi-pillar + intermediate + cross-validation test sweeps.
- L10-B landed exactly inside its target (+41 vs +20-25).

## Cross-phase carve-out closures

This phase moves four long-standing items off `docs/carve-outs.md`:

1. **L1-D HaltonRsg (originally carry-over)** — landed as a side-benefit of L10-A's Halton multi-start.
2. **Phase 9 vol-tail residuals (L9-C carve-out)** — `KahaleSmileSection` + `AtmSmileSection` + `AtmAdjustedSmileSection` + `SabrInterpolatedSmileSection` + `OptionletStripper2` all closed.
3. **Tier-1 specialty short-rate (Phase 4 carve-out)** — `Gaussian1dModel` + `GSR` + `Gaussian1dSwaptionVolatility` now ported. `MarkovFunctional` remains.
4. **L1-E interpolator tail (Phase 1 carve-out)** — `HymanFilteredCubic` + `ChebyshevInterpolation` + `MultiCubicSpline` + `AbcdInterpolation` all closed.

Bonus: **ZABR family closed-form** added as a new domain (`ZabrSmileSection` + `zabr_volatility`).

## Lessons / patterns reinforced

- **Python-scipy optimizers continue to be quantitatively better than C++ projected-LM** on ill-posed parametric fits. `AbcdInterpolation` is the third instance (after L9-B `conventional_spread` and L9-C `SabrInterpolation` recovered-params noise). When the Python and C++ probes disagree by orders of magnitude, the bias is usually toward Python correctness — but document inline and verify the math contract independently.
- **3-parallel-no-pilot remains the right topology for independent specialty domains.** L6, L8, L10 all used it. No regressions on the merge mechanics.
- **HaltonRsg port discovered in the middle of L10-A** — the subagent realised the Halton multi-start needed HaltonRsg, then ported HaltonRsg itself rather than carving out. This kind of opportunistic L1 closure is fine when the dependency is small and the carve-out backlog is otherwise empty.
- **MRO diamond pattern documented** — `Gaussian1dModel` skips cooperative `super().__init__()` because of `CalibratedModel`'s ctor requirement; same precedent as G2 from Phase 4. Worth recording in `[[project-python-translation-choices]]`.

## Post-phase carry-overs (recorded; not blocking phase closure)

| Item | Origin | Estimated effort |
|---|---|---|
| `MarkovFunctional` | L10-B carve-out | dedicated future cluster (~542 LOC of bootstrap-vs-swaption-strip calibration) |
| Gaussian1d engines (`Gaussian1dCapFloorEngine` + 2 others) | L10-B carve-out | ~6 hr each — meaningful work but mechanical |
| `ZabrInterpolation` (5-param fitter) + `ZabrInterpolatedSmileSection` | L10-C carve-out | ~2 hr (fitter) + ~1 hr (wrapper) |
| ZABR FD modes (`Local`, `FullFd`, `ProjectedHedge`) | L10-C carve-out | dedicated future cluster (depends on FD engine on the ZABR PDE) |
| `XabrSwaptionVolatilityCube` template generalization | L9-C + L10-C carve-out | ~3 hr — pure plumbing |
| `KahaleSmileSection.core_smile` deep-iteration testing | L10-A carve-out | ~2 hr — needs pathological-arbitrage synthetic test cases |
| `MarketModels` cluster | persistent carve-out | dedicated future cluster (~125 files) |
| `experimental/credit/*` (CDO, basket CDS) | persistent carve-out | dedicated future cluster (~30 files) |

None of these block project completion. They are the natural next-iteration backlog.

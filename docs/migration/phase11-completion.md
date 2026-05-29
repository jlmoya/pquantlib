# Phase 11 — completion (in progress)

**Started:** 2026-05-29
**Predecessor:** `pquantlib-phase10-complete` @ `d3746e4` — 2652/0/0
**C++ ground truth:** QuantLib v1.42.1 @ `099987f0`

Living document — updated incrementally as each wave closes.

## Wave summary

| Wave | Tag | Tests delta | Cumulative | Date |
|---|---|---|---|---|
| W1 | `pquantlib-phase11-w1-complete` @ `07712fa` | +123 | **2775** | 2026-05-29 |

## W1 — Specialty model completion

4 parallel clusters off `pquantlib-phase10-complete`.

| Cluster | Branch tip | Classes | Tests | Carve-out closed / deferred |
|---|---|---|---|---|
| W1-A MarkovFunctional | `1f0aa63` | 3 (MarkovFunctional + MarkovFunctionalSwaptionEngine + Gaussian1dGsrProcess) | +24 | closes MarkovFunctional carve-out (caplet calibration / Kahale-SABR pretreatment / Bermudan exercise / AdjustYts / AdjustDigitals deferred) |
| W1-B Gaussian1d engines | `b66589b` | 3 engines + 4 instruments (FloatFloatSwap/Swaption + NonstandardSwap/Swaption) | +8 | closes L10-B Gaussian1d engines carve-out (BasketGeneratingEngine.calibrationBasket / Probabilities enum / OAS / RebatedExercise / CMS dispatch deferred) |
| W1-C Bates variants | `47cb885` | 3 models + 2 engines | +43 | closes Bates variant carve-out (AnalyticBatesDoubleExpDetJumpEngine still deferred — model landed, engine pending) |
| W1-D Heston SLV + GjrGarch + time-dep | `f272df5` | 4 models + 2 engines + FixedLocalVolSurface | +48 | closes specialty-Heston carve-out (HestonSlvFdmModel scaffold-only — full FDM scheduled for W5-C; PTD engine AndersenPiterbarg branch deferred) |

**Total W1:** 14 classes / +123 tests / 2652 → 2775.

**Notable divergences (per cluster summaries):**
- MarkovFunctional: numpy `hermgauss` weights already fold `exp(-x²)` whereas C++ `GaussHermiteIntegration` returns bare weights — divides by √π directly (fixed numeraire 1.5x off → 0.1% off C++ truth).
- Gaussian1d engines: C++ uses Lagrange-BC cubic; PQuantLib uses natural-BC (scipy.CubicSpline). Empirical deltas: cap NPV rel ~5e-7, FloatFloat swaption rel ~0.24%, NonStandard swaption rel ~3e-5.
- Bates: `BatesDoubleExp(lambda→0)` collapses to Heston (TIGHT); `BatesDetJump(thetaLambda==lambda)` collapses to base Bates (TIGHT).
- Heston SLV FDM scaffold returns unit leverage; full Fokker-Planck FDM solver scheduled for W5-C.

**Merge reconciliations:**
- 3 successive CMakeLists.txt conflicts as each branch appended its probe entry. Standard resolution: keep both sides, stacked W1-A → W1-B → W1-C → W1-D entries.
- `docs/carve-outs.md` modified by W1-D (clean merge — appended Specialty-Heston-variants status).
- 3 base-class vol files modified by W1-A (clean merge — backwards-compatible signature widenings).
- `gaussian1d_model.py` modified by W1-B (clean merge — added staticmethods).

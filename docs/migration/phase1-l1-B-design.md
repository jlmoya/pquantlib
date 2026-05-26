# Phase 1 L1-B — Copulas + simple distributions/statistics + currencies

**Date:** 2026-05-24
**Status:** **closed** — merged into `main` via `cbd55ac merge: L1-B`; tagged as part of `pquantlib-phase1-complete` @ `edcadbc`. Final test delta: **+50** (415 → 465 before subsequent merges).
**Predecessor:** `pquantlib-phase1-l1-A-complete` @ `03d0ce8`
**C++ ground truth:** QuantLib v1.42.1 @ `099987f0`
**Style:** lean — leans on the L1-A design + plan for the per-class TDD loop, tolerance discipline, commit format. Read [`phase1-l1-A-design.md`](phase1-l1-A-design.md) for binding ground rules.

## Goal

Port the copula family (13 classes, all closed-form 2-D bivariate copulas), the **simple** distribution helpers, key statistics aggregators, and the currency layer. Defer complex distributions (incl. inverse cumulatives, bivariate cumulative student, stochastic-collocation inverse CDF) to a follow-up cluster.

## Must-port (tractable subset)

### Copulas (~13)
AliMikhailHaqCopula, ClaytonCopula, FarlieGumbelMorgensternCopula, FrankCopula, GalambosCopula, GaussianCopula, GumbelCopula, HuslerReissCopula, IndependentCopula, MarshallOlkinCopula, MaxCopula, MinCopula, PlackettCopula.

All trivial — single `operator()(x, y)` returning a closed-form expression with input validation `x, y ∈ [0, 1]`. Ports cleanly as `@dataclass(frozen=True, slots=True)` with `__call__`.

### Distributions (~6 — subset)
- `CumulativeNormalDistribution` — via `math.erf` (LOOSE-tier).
- `NormalDistribution` — pdf = `(1/√(2π)) exp(-(x-μ)²/(2σ²))`.
- `MoroInverseCumulativeNormal` — Moro 1995 algorithm, ~30 LOC table-driven.
- `InverseCumulativeNormal` — Acklam algorithm.
- `BivariateCumulativeNormalDistributionDr78` — Drezner 1978.
- `CumulativeStudentDistribution` — closed-form via `math.lgamma`.

### Statistics (~3 — subset)
- `GeneralStatistics` — running mean / variance / kurtosis / skew.
- `IncrementalStatistics` — running aggregator (Welford-style).
- `Histogram` — simple binning.

### Currencies (~9)
Free-function or module-level `Currency` definitions. Static-data ports.

## Carve-outs (deferred to follow-up)

- Bivariate cumulative student distribution (complex numerical integration).
- Non-central chi-squared distributions (Sankaran approximation).
- StochasticCollocationInvCDF.
- Maddock-class inverse cumulatives.
- `Maddock*` variants.
- Gaussian / Risk / Sequence / Convergence statistics (depend on RNGs).

## Approach

- Single mega-probe `math/first_batch_l1B_probe.cpp` covering copulas + simple distributions + statistics value tables.
- Currencies probed separately (or via static-data assertion — no probe needed since each currency is a frozen ISO descriptor).
- Subagent dispatch: 1 cluster agent, gets ~30 must-port targets.

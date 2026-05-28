# Phase 5 L5-B — Trees + lattices + tree engines + BlackKarasinski

**Date:** 2026-05-28
**Status:** **closed** — merged via L5-B merge; tagged as part of `pquantlib-phase5-complete` @ `d322fca`. Final test delta: **+54**. 7 commits.
**Predecessor:** `pquantlib-phase5-l5-A-complete` @ `aa19340`
**Style:** lean — leans on [`phase5-design.md`](phase5-design.md).

## What landed

### Concrete trees
- `BinomialTree[Concrete]` refactor (CRR / JarrowRudd / Tian / LeisenReimer) — extracts from L3-D BinomialVanillaEngine inline impl.
- `TrinomialTree` (+ inner `_Branching`).
- `TreeLattice1D` (collapses C++ `TreeLattice<Impl>` + `TreeLattice1D<Impl>` CRTP into a single class).
- `BlackScholesLattice` — equity binomial lattice.

### Discretized concretes
- `DiscretizedSwap` + `DiscretizedSwaption` + `DiscretizedCapFloor` (subclasses of L5-A bases).
- `ShortRateTree` — Hull-White-style trinomial discretization of a short-rate model.

### Engines (closes Phase 4 carve-outs)
- `TreeSwaptionEngine(model, time_steps)` — generic across `ShortRateModel`.
- `TreeCapFloorEngine(model, time_steps)`.

### Closes Phase 4 carve-out: BlackKarasinski
- `BlackKarasinski(termStructure, a=0.1, sigma=0.1)` — log-normal short rate via trinomial tree (no closed-form discount bond).

### ShortRateModel.tree(grid)
- Concrete implementation across Vasicek / HullWhite / CoxIngersollRoss / ExtendedCoxIngersollRoss / BlackKarasinski.

## Documented divergences

- Python collapses C++ TreeLattice CRTP hierarchy.
- `DiscretizedSwaption` rebuilds `SwaptionArguments` in place rather than constructing a fresh snapped `VanillaSwap` — introduces ~1bp delta vs C++ at N=100 (still within 5% of Jamshidian).
- `peizer_pratt_method2_inversion` factored to `math/distributions/binomial_distribution`.
- `CouponAdjustment` enum on `DiscretizedAsset`.

## Carve-outs

- TreeLattice2D + G2.tree() (multi-factor heavy).
- TFLattice variants.
- Joshi4 / AdditiveEQP / Trigeorgis binomial builders.
- Full BinomialDistribution CDF/PDF.

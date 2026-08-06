"""Tests for the ql/models/marketmodels/ coverage tail.

Cross-validated against ``migration-harness/references/v143/models/marketmodels``
(produced by ``migration-harness/cpp/probes/v143_models_marketmodels/probe.cpp``).

C++ parity:
  ql/models/marketmodels/pathwiseaccountingengine.{hpp,cpp}
    — PathwiseVegasAccountingEngine, PathwiseVegasOuterAccountingEngine
  ql/models/marketmodels/models/flatvol.{hpp,cpp} — FlatVolFactory
  ql/models/marketmodels/historicalforwardratesanalysis.hpp
    — HistoricalForwardRatesAnalysisImpl (pipeline coverage lives in
      test_historical_analysis.py)
  ql/math/matrixutilities/{pseudosqrt,symmetricschurdecomposition}.cpp
    — rankReducedSqrt's eigenvalue ordering
  @ v1.43.

**No evaluation-date fixture.** The probe sets no evaluation date
(probe.cpp:12), because nothing it exercises reads
``Settings::instance().evaluationDate()``: the engines and ``rankReducedSqrt``
take no dates at all, and the ``FlatVolFactory`` block builds its
``FlatForward`` with an EXPLICIT reference date (probe.cpp:387). The absence of
an ``ObservableSettings`` fixture here is therefore a verified fact, not an
oversight.

**Deterministic driving noise.** Both engines are Monte-Carlo. The probe drives
them from a fixed table of Gaussian increments (probe.cpp:153) rather than an
MT/Sobol generator, and emits that table, so this module reads the SAME numbers
and reproduces every path exactly. The alternative — an MT-driven engine — would
inherit the documented ``InverseCumulativeNormal`` Halley-refinement gap (~1e-16
per variate, compounded through three ``exp()`` steps), which is not a
defensible basis for a TIGHT comparison of an engine.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.models.marketmodels.brownian_generator import (
    BrownianGenerator,
    BrownianGeneratorFactory,
)
from pquantlib.models.marketmodels.evolution_description import (
    EvolutionDescription,
    money_market_measure,
)
from pquantlib.models.marketmodels.evolvers.lognormal_fwd_rate_euler import (
    LogNormalFwdRateEuler,
)
from pquantlib.models.marketmodels.market_model import MarketModel, MarketModelFactory
from pquantlib.models.marketmodels.models.flat_vol import FlatVol, FlatVolFactory
from pquantlib.models.marketmodels.models.pseudo_root_facade import PseudoRootFacade
from pquantlib.models.marketmodels.models.pseudo_sqrt import (
    SalvagingAlgorithm,
    rank_reduced_sqrt,
)
from pquantlib.models.marketmodels.pathwise_accounting_engine import (
    PathwiseVegasAccountingEngine,
    PathwiseVegasOuterAccountingEngine,
)
from pquantlib.models.marketmodels.pathwise_multi_product import (
    MarketModelPathwiseMultiProduct,
)
from pquantlib.models.marketmodels.products.pathwise_product_caplet import (
    MarketModelPathwiseMultiCaplet,
    MarketModelPathwiseMultiDeflatedCap,
    MarketModelPathwiseMultiDeflatedCaplet,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import exact, tight
from pquantlib.time.date import Date
from pquantlib.time.month import Month

# --- the probe's world (probe.cpp block_world) ------------------------------
_FACTORS = 2
_N_RATES = 3
_N_STEPS = 3
_MAX_PATHS = 5

_RATE_TIMES = [0.5, 1.0, 1.5, 2.0]
_ACCRUALS = [0.5, 0.5, 0.5]
_PAYMENT_TIMES = [1.0, 1.5, 2.0]
_STRIKES = [0.04, 0.04, 0.04]


@pytest.fixture(scope="module")
def ref() -> dict[str, Any]:
    return load_reference("v143/models/marketmodels")


# --- deterministic Brownian generator (probe.cpp FixedBrownianGenerator) -----
class _FixedBrownianGenerator(BrownianGenerator):
    """Replays a fixed (path x step*factor) table of Gaussian increments."""

    def __init__(self, factors: int, steps: int, table: list[list[float]]) -> None:
        self._factors = factors
        self._steps = steps
        self._table = table
        self._path = 0
        self._step = 0
        self._started = False

    def next_step(self, output: list[float]) -> float:
        row = self._table[self._path % len(self._table)]
        for f in range(self._factors):
            output[f] = row[self._step * self._factors + f]
        self._step += 1
        return 1.0

    def next_path(self) -> float:
        if self._started:
            self._path += 1
        else:
            self._started = True
        self._step = 0
        return 1.0

    def number_of_factors(self) -> int:
        return self._factors

    def number_of_steps(self) -> int:
        return self._steps


class _FixedBrownianGeneratorFactory(BrownianGeneratorFactory):
    def __init__(self, table: list[list[float]]) -> None:
        self._table = table

    def create(self, factors: int, steps: int) -> BrownianGenerator:
        return _FixedBrownianGenerator(factors, steps, self._table)


# --- world builders ---------------------------------------------------------
def _gaussians(ref: dict[str, Any]) -> list[list[float]]:
    return [
        [ref[f"gauss_{p}_{i}"] for i in range(_N_STEPS * _FACTORS)]
        for p in range(_MAX_PATHS)
    ]


def _pseudo_roots(ref: dict[str, Any]) -> list[Any]:
    return [
        np.array(
            [
                [ref[f"pr_{s}_{i}_{f}"] for f in range(_FACTORS)]
                for i in range(_N_RATES)
            ],
            dtype=np.float64,
        )
        for s in range(_N_STEPS)
    ]


def _model(ref: dict[str, Any]) -> MarketModel:
    return PseudoRootFacade(
        _pseudo_roots(ref),
        _RATE_TIMES,
        [ref[f"init_{i}"] for i in range(_N_RATES)],
        [ref[f"disp_{i}"] for i in range(_N_RATES)],
    )


def _vega_bumps(ref: dict[str, Any]) -> list[list[Any]]:
    n = int(ref["number_bumps"])
    return [
        [
            np.array(
                [
                    [ref[f"vb_{step}_{b}_{i}_{f}"] for f in range(_FACTORS)]
                    for i in range(_N_RATES)
                ],
                dtype=np.float64,
            )
            for b in range(n)
        ]
        for step in range(_N_STEPS)
    ]


def _fresh_evolver(ref: dict[str, Any], model: MarketModel) -> LogNormalFwdRateEuler:
    """A new evolver, hence a new generator positioned at path 0."""
    numeraires = money_market_measure(EvolutionDescription(_RATE_TIMES))
    return LogNormalFwdRateEuler(
        model, _FixedBrownianGeneratorFactory(_gaussians(ref)), numeraires
    )


def _caplets() -> MarketModelPathwiseMultiCaplet:
    return MarketModelPathwiseMultiCaplet(
        _RATE_TIMES, _ACCRUALS, _PAYMENT_TIMES, _STRIKES
    )


def _caplets_deflated() -> MarketModelPathwiseMultiDeflatedCaplet:
    return MarketModelPathwiseMultiDeflatedCaplet(
        _RATE_TIMES, _ACCRUALS, _PAYMENT_TIMES, _STRIKES
    )


def _caps_deflated(ref: dict[str, Any]) -> MarketModelPathwiseMultiDeflatedCap:
    # 2 caps over 3 rates -> numberOfProducts() != numberOfRates.
    return MarketModelPathwiseMultiDeflatedCap(
        _RATE_TIMES, _ACCRUALS, _PAYMENT_TIMES, ref["strike"], [(0, 2), (1, 3)]
    )


def _run_inner(
    ref: dict[str, Any], product: MarketModelPathwiseMultiProduct, paths: int
) -> tuple[list[float], list[float]]:
    model = _model(ref)
    engine = PathwiseVegasAccountingEngine(
        _fresh_evolver(ref, model),
        product,
        model,
        _vega_bumps(ref),
        ref["initial_numeraire_value"],
    )
    means: list[float] = []
    errors: list[float] = []
    engine.multiple_path_values(means, errors, paths)
    return means, errors


def _run_outer(
    ref: dict[str, Any],
    product: MarketModelPathwiseMultiProduct,
    paths: int,
    *,
    elementary: bool,
) -> tuple[list[float], list[float]]:
    model = _model(ref)
    engine = PathwiseVegasOuterAccountingEngine(
        _fresh_evolver(ref, model),
        product,
        model,
        _vega_bumps(ref),
        ref["initial_numeraire_value"],
    )
    means: list[float] = []
    errors: list[float] = []
    if elementary:
        engine.multiple_path_values_elementary(means, errors, paths)
    else:
        engine.multiple_path_values(means, errors, paths)
    return means, errors


def _assert_series(
    actual: list[float], ref: dict[str, Any], prefix: str, size_key: str
) -> None:
    exact(float(len(actual)), ref[size_key])
    for i, a in enumerate(actual):
        tight(a, ref[f"{prefix}_{i}"])


# --- the derived bookkeeping the engines depend on --------------------------


def test_world_measure_and_alive_indices(ref: dict[str, Any]) -> None:
    evolution = EvolutionDescription(_RATE_TIMES)
    numeraires = money_market_measure(evolution)
    alive = evolution.first_alive_rate()
    for i in range(_N_STEPS):
        exact(float(numeraires[i]), ref[f"numeraire_{i}"])
        exact(float(alive[i]), ref[f"first_alive_{i}"])
    # RatePseudoRootJacobian requires aliveIndex == numeraire (discretely
    # compounding money-market account); the probe world satisfies it.
    assert numeraires == alive


def test_vega_bump_construction(ref: dict[str, Any]) -> None:
    # Reproduce the C++ test-suite's vegaBumps recipe and check it against the
    # probe's, so a mis-shaped bump set cannot silently pass the engine tests.
    vega_bump_size = 1e-2
    bump_increment = 1 + _N_STEPS // 3
    factors_to_test = min(2, _FACTORS)
    model_bump = np.zeros((_N_RATES, _FACTORS), dtype=np.float64)
    built: list[list[Any]] = []
    for step in range(_N_STEPS):
        built.append([])
        for k in range(0, _N_RATES, bump_increment):
            for f in range(factors_to_test):
                for m in range(_N_STEPS):
                    if step == m and k >= step:
                        model_bump[k, f] = vega_bump_size
                    built[step].append(model_bump.copy())
                    model_bump[k, f] = 0.0

    exact(float(len(built[0])), ref["number_bumps"])
    for step in range(_N_STEPS):
        for b in range(len(built[step])):
            for i in range(_N_RATES):
                for f in range(_FACTORS):
                    exact(
                        float(built[step][b][i, f]), ref[f"vb_{step}_{b}_{i}_{f}"]
                    )


# --- PathwiseVegasAccountingEngine ------------------------------------------


def test_inner_engine_one_path(ref: dict[str, Any]) -> None:
    # numberOfPaths == 1, so the means ARE path 1's singlePathValues output.
    means, _ = _run_inner(ref, _caplets(), 1)
    _assert_series(means, ref, "inner_undefl_p1_mean", "inner_undefl_p1_size")


def test_inner_engine_one_path_standard_errors_are_zero(
    ref: dict[str, Any],
) -> None:
    # At one path the C++ standard error is sqrt(v*v - v*v) == 0 in exact
    # arithmetic. The reference is NOT comparable here: the QuantLib build
    # contracts `means[j]*means[j]` into an FMA, so `sumsq/n - mean*mean` lands
    # a fraction of an ulp either side of zero and std::sqrt returns either a
    # tiny positive number or NaN. pquantlib evaluates the product at double
    # precision first and gets the exact 0.0.
    #
    # The bound below is DERIVED, not tuned: the contracted expression differs
    # from the uncontracted one by at most one ulp of v*v, i.e. 2^-52 * v^2, so
    # the reported error is at most sqrt(2^-52) * |v| = 2^-26 * |v| ~=
    # 1.49e-8 * |v|. Observed worst case in the reference: 8.09e-9 * |v|.
    means, errors = _run_inner(ref, _caplets(), 1)
    fma_bound = 2.0**-26
    for i, e in enumerate(errors):
        exact(e, 0.0)
        cpp = ref[f"inner_undefl_p1_err_{i}"]
        if not math.isnan(cpp):
            assert abs(cpp) <= fma_bound * abs(means[i]) + 1e-300, (i, cpp, means[i])


def test_inner_engine_two_paths(ref: dict[str, Any]) -> None:
    # Pins the second path too: mean_2 = (path1 + path2)/2.
    means, errors = _run_inner(ref, _caplets(), 2)
    _assert_series(means, ref, "inner_undefl_p2_mean", "inner_undefl_p2_size")
    _assert_series(errors, ref, "inner_undefl_p2_err", "inner_undefl_p2_size")


def test_inner_engine_undeflated_five_paths(ref: dict[str, Any]) -> None:
    # doDeflation_ == true: MarketModelPathwiseDiscounter::getFactors runs and
    # its derivatives feed fullDerivatives_.
    means, errors = _run_inner(ref, _caplets(), 5)
    _assert_series(means, ref, "inner_undefl_p5_mean", "inner_undefl_p5_size")
    _assert_series(errors, ref, "inner_undefl_p5_err", "inner_undefl_p5_size")


def test_inner_engine_deflated_five_paths(ref: dict[str, Any]) -> None:
    # doDeflation_ == false: the else-branch, fullDerivatives_ taken raw.
    assert _caplets_deflated().already_deflated()
    means, errors = _run_inner(ref, _caplets_deflated(), 5)
    _assert_series(means, ref, "inner_defl_p5_mean", "inner_defl_p5_size")
    _assert_series(errors, ref, "inner_defl_p5_err", "inner_defl_p5_size")


def test_inner_engine_caps_products_differ_from_rates(ref: dict[str, Any]) -> None:
    # 2 products, 3 rates: exercises `i*entriesPerProduct + ...` indexing.
    caps = _caps_deflated(ref)
    assert caps.number_of_products() == 2
    assert caps.number_of_products() != _N_RATES
    means, errors = _run_inner(ref, caps, 5)
    _assert_series(means, ref, "inner_caps_p5_mean", "inner_caps_p5_size")
    _assert_series(errors, ref, "inner_caps_p5_err", "inner_caps_p5_size")


# --- PathwiseVegasOuterAccountingEngine -------------------------------------


def test_outer_engine_elementary_caps(ref: dict[str, Any]) -> None:
    # The un-contracted elementary vegas: the only witness to the index
    # arithmetic m + l*factors + k*rates*factors.
    means, errors = _run_outer(ref, _caps_deflated(ref), 5, elementary=True)
    _assert_series(means, ref, "outer_caps_p5_elem_mean", "outer_caps_p5_elem_size")
    _assert_series(errors, ref, "outer_caps_p5_elem_err", "outer_caps_p5_elem_size")


def test_outer_engine_combined_caps(ref: dict[str, Any]) -> None:
    means, errors = _run_outer(ref, _caps_deflated(ref), 5, elementary=False)
    _assert_series(means, ref, "outer_caps_p5_mean", "outer_caps_p5_size")
    _assert_series(errors, ref, "outer_caps_p5_err", "outer_caps_p5_size")


def test_outer_engine_elementary_undeflated(ref: dict[str, Any]) -> None:
    means, errors = _run_outer(ref, _caplets(), 5, elementary=True)
    _assert_series(means, ref, "outer_undefl_p5_elem_mean", "outer_undefl_p5_elem_size")
    _assert_series(errors, ref, "outer_undefl_p5_elem_err", "outer_undefl_p5_elem_size")


def test_outer_engine_combined_undeflated(ref: dict[str, Any]) -> None:
    means, errors = _run_outer(ref, _caplets(), 5, elementary=False)
    _assert_series(means, ref, "outer_undefl_p5_mean", "outer_undefl_p5_size")
    _assert_series(errors, ref, "outer_undefl_p5_err", "outer_undefl_p5_size")


def test_inner_and_outer_engines_agree(ref: dict[str, Any]) -> None:
    # The C++ test-suite's own cross-check (marketmodel.cpp testPathwiseVegas,
    # tolerance 1e-8): contracting the vega bumps as early as possible and as
    # late as possible must give the same answer. Asserted here at TIGHT
    # because both sides are driven by the identical fixed variates.
    caps = _caps_deflated(ref)
    inner_means, _ = _run_inner(ref, caps, 5)
    outer_means, _ = _run_outer(ref, _caps_deflated(ref), 5, elementary=False)
    assert len(inner_means) == len(outer_means)
    for a, b in zip(inner_means, outer_means, strict=True):
        tight(a, b)


def test_outer_engine_errors_for_vegas_are_zero(ref: dict[str, Any]) -> None:
    # C++ resizes `errors` and only fills the price + delta slots, leaving the
    # vega standard errors at the value-initialized 0.0 — "post linear
    # combinations, errors are not meaningful" (pathwiseaccountingengine.cpp).
    n_bumps = int(ref["number_bumps"])
    per_product = 1 + _N_RATES + n_bumps
    _, errors = _run_outer(ref, _caps_deflated(ref), 5, elementary=False)
    for p in range(2):
        for bump in range(n_bumps):
            exact(errors[p * per_product + 1 + _N_RATES + bump], 0.0)


# --- FlatVolFactory ---------------------------------------------------------


def _flat_vol_factory(ref: dict[str, Any]) -> FlatVolFactory:
    # probe.cpp block_flat_vol_factory: FlatForward at an EXPLICIT reference
    # date, so no evaluation date is involved on either side.
    reference_date = Date.from_ymd(15, Month.June, 2026)
    exact(float(reference_date.serial_number()), ref["fvf_ref_date_serial"])
    curve = FlatForward.from_rate(reference_date, 0.04, Actual365Fixed())
    times = [ref[f"fvf_time_{i}"] for i in range(5)]
    vols = [ref[f"fvf_vol_{i}"] for i in range(5)]
    return FlatVolFactory(
        ref["fvf_long_term_correlation"],
        ref["fvf_beta"],
        times,
        vols,
        curve,
        ref["fvf_displacement"],
    )


@pytest.mark.parametrize("factors", [1, 2, 3])
def test_flat_vol_factory_create(ref: dict[str, Any], factors: int) -> None:
    tag = f"fvf_f{factors}"
    model = _flat_vol_factory(ref).create(EvolutionDescription(_RATE_TIMES), factors)
    assert isinstance(model, FlatVol)
    exact(float(model.number_of_rates()), ref[f"{tag}_n_rates"])
    exact(float(model.number_of_factors()), ref[f"{tag}_n_factors"])
    exact(float(model.number_of_steps()), ref[f"{tag}_n_steps"])
    # initial rates come off the curve; displacements are uniform. Both are
    # eigensolver-independent, so TIGHT.
    for i in range(_N_RATES):
        tight(model.initial_rates()[i], ref[f"{tag}_init_{i}"])
        tight(model.displacements()[i], ref[f"{tag}_disp_{i}"])
    # covariance = A @ A.T is invariant under the eigensolver's choice of basis
    # within an eigenspace; the raw A is not (C++ Jacobi vs numpy LAPACK), so
    # the covariance is what pins the displaced-vol formula + correlations.
    for s in range(model.number_of_steps()):
        cov = model.covariance(s)
        for i in range(_N_RATES):
            for j in range(_N_RATES):
                tight(float(cov[i, j]), ref[f"{tag}_cov_{s}_{i}_{j}"])
    total = model.total_covariance(model.number_of_steps() - 1)
    for i in range(_N_RATES):
        for j in range(_N_RATES):
            tight(float(total[i, j]), ref[f"{tag}_totcov_{i}_{j}"])


def test_flat_vol_factory_is_a_market_model_factory(ref: dict[str, Any]) -> None:
    assert isinstance(_flat_vol_factory(ref), MarketModelFactory)


def test_flat_vol_factory_rebroadcasts_curve_updates(ref: dict[str, Any]) -> None:
    # C++ FlatVolFactory::update() -> notifyObservers(). The pquantlib factory
    # keeps that wiring even though MarketModelFactory itself is a plain ABC.
    factory = _flat_vol_factory(ref)
    seen: list[int] = []

    class _Sink:
        def update(self) -> None:
            seen.append(1)

    sink = _Sink()
    factory.register_with(sink)
    factory.update()
    assert seen == [1]


def test_flat_vol_factory_volatility_is_interpolated(ref: dict[str, Any]) -> None:
    # rateTimes[0..2] = 0.5, 1.0, 1.5 are knots of the (times, vols) grid, so
    # the displaced vol at rate i is initialRates[i]*vols[i+1]/(f+displacement)
    # exactly — a direct check that the LinearInterpolation lookup uses the
    # rate's FIXING time and not its payment time.
    factors = 3
    model = _flat_vol_factory(ref).create(EvolutionDescription(_RATE_TIMES), factors)
    displacement = ref["fvf_displacement"]
    for i in range(_N_RATES):
        f0 = model.initial_rates()[i]
        vol = ref[f"fvf_vol_{i + 1}"]  # times 0.5, 1.0, 1.5 -> indices 1, 2, 3
        displaced = f0 * vol / (f0 + displacement)
        # step 0 covariance diagonal for rate i is displaced^2 * min(t_i, 0.5).
        cov0 = model.covariance(0)
        tight(float(cov0[i, i]), displaced * displaced * min(_RATE_TIMES[i], 0.5))


# --- rank_reduced_sqrt eigenvalue ordering ----------------------------------


def _assert_matrix(m: Any, ref: dict[str, Any], tag: str) -> None:
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            tight(float(m[i, j]), ref[f"{tag}_{i}_{j}"])


def test_rank_reduced_sqrt_identity_tie_break(ref: dict[str, Any]) -> None:
    # The minimal witness for the SymmetricSchurDecomposition ordering:
    # std::sort with std::greater<> on pair<Real, vector<Real> > is
    # LEXICOGRAPHIC, so a tied eigenvalue is broken by comparing eigenvectors
    # descending — (1,0) beats (0,1) and C++ returns the identity. Reversing an
    # ascending eigensolver's output returns the exchange matrix instead, and
    # B @ B.T is the identity either way, so only the FULL matrix catches it.
    i2 = np.eye(2, dtype=np.float64)
    _assert_matrix(rank_reduced_sqrt(i2, 2, 1.0, SalvagingAlgorithm.NONE), ref, "rrs_i2_none")
    _assert_matrix(
        rank_reduced_sqrt(i2, 2, 1.0, SalvagingAlgorithm.SPECTRAL),
        ref,
        "rrs_i2_spectral",
    )


def test_rank_reduced_sqrt_repeated_eigenvalue_diagonal(ref: dict[str, Any]) -> None:
    d3 = np.diag([2.0, 2.0, 1.0]).astype(np.float64)
    _assert_matrix(
        rank_reduced_sqrt(d3, 3, 1.0, SalvagingAlgorithm.NONE), ref, "rrs_d3_none"
    )


def test_rank_reduced_sqrt_repeated_eigenvalue_nontrivial(ref: dict[str, Any]) -> None:
    # b3 has eigenvalues {2, 2, 1}; the eigenvalue-2 eigenspace is spanned by
    # e0 and (0,1,1)/sqrt(2). The ordering rule resolves the TIE correctly, so
    # columns 0 and 1 match C++ exactly.
    #
    # Column 2 (the simple eigenvalue 1, eigenvector (0, 1, -1)/sqrt(2)) is
    # determined only up to an overall sign: C++'s sign convention flips iff
    # the eigenvector's FIRST component is negative
    # (symmetricschurdecomposition.cpp:131-133), and here it is exactly 0, so
    # the convention is vacuous and the Jacobi sweep's arbitrary sign survives.
    # numpy's LAPACK sweep picks the other one. That is a residual eigensolver
    # divergence, NOT a tolerance question: the column is asserted up to sign,
    # and the sign-invariant B @ B.T is asserted exactly.
    b3 = np.array(
        [[2.0, 0.0, 0.0], [0.0, 1.5, 0.5], [0.0, 0.5, 1.5]], dtype=np.float64
    )
    b = rank_reduced_sqrt(b3, 3, 1.0, SalvagingAlgorithm.NONE)
    for i in range(3):
        for j in (0, 1):
            tight(float(b[i, j]), ref[f"rrs_b3_none_{i}_{j}"])
    cpp_col2 = np.array([ref[f"rrs_b3_none_{i}_2"] for i in range(3)])
    sign = 1.0 if float(b[1, 2]) * cpp_col2[1] >= 0.0 else -1.0
    for i in range(3):
        tight(sign * float(b[i, 2]), cpp_col2[i])
    reconstructed = b @ b.T
    for i in range(3):
        for j in range(3):
            tight(float(reconstructed[i, j]), float(b3[i, j]))


def test_rank_reduced_sqrt_generic_control(ref: dict[str, Any]) -> None:
    # No ties: the ordering rule is irrelevant, so this isolates the
    # eigen-decomposition itself.
    g3 = np.array(
        [[1.0, 0.3, 0.1], [0.3, 2.0, 0.2], [0.1, 0.2, 3.0]], dtype=np.float64
    )
    _assert_matrix(
        rank_reduced_sqrt(g3, 3, 1.0, SalvagingAlgorithm.NONE), ref, "rrs_g3_none"
    )
    _assert_matrix(
        rank_reduced_sqrt(g3, 2, 1.0, SalvagingAlgorithm.NONE), ref, "rrs_g3_rank2"
    )


def test_rank_reduced_sqrt_semidefinite_spectral(ref: dict[str, Any]) -> None:
    s3 = np.array(
        [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 4.0]], dtype=np.float64
    )
    _assert_matrix(
        rank_reduced_sqrt(s3, 3, 1.0, SalvagingAlgorithm.SPECTRAL),
        ref,
        "rrs_s3_spectral",
    )

"""Cross-validation of the LFM covariance parameterizations against C++ v1.43.

Covers ``LfmCovarianceParameterization``, ``LfmCovarianceProxy`` and
``LfmHullWhiteParameterization``.

Expected values come from ``migration-harness/references/v143/legacy/lmm.json``
(``migration-harness/cpp/probes/v143_legacy_lmm/probe.cpp``).

EVALUATION DATE: the Hull-White section builds an ``Euribor1Y`` index and a
``CapletVarianceCurve`` off real dates. ``probe.cpp`` sets
``Settings::instance().evaluationDate() = TARGET().adjust(Date(4, September,
2005))`` (see ``makeIndex1Y``, probe.cpp), so the autouse fixture below pins
the same date and restores the previous one afterwards.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Final

import numpy as np
import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_actual import ActualActual, Convention
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.ibor.euribor import Euribor1Y
from pquantlib.legacy.libormarketmodels.lfm_covar_proxy import LfmCovarianceProxy
from pquantlib.legacy.libormarketmodels.lfm_hull_white_param import (
    LfmHullWhiteParameterization,
)
from pquantlib.legacy.libormarketmodels.lfm_process import LiborForwardModelProcess
from pquantlib.legacy.libormarketmodels.lm_exp_corr_model import (
    LmExponentialCorrelationModel,
)
from pquantlib.legacy.libormarketmodels.lm_fixed_vol_model import LmFixedVolatilityModel
from pquantlib.legacy.libormarketmodels.lm_lin_exp_vol_model import (
    LmLinearExponentialVolatilityModel,
)
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.termstructures.volatility.optionlet.caplet_variance_curve import (
    CapletVarianceCurve,
)
from pquantlib.termstructures.yield_.interpolated_zero_curve import (
    InterpolatedZeroCurve,
)
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import tight
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.time_unit import TimeUnit

# probe.cpp kAnchor; the evaluation date is TARGET().adjust of it.
ANCHOR: Final[Date] = Date.from_ymd(4, Month.September, 2005)
SIZE: Final[int] = 6
A: Final[float] = 0.23
B: Final[float] = 0.17
C: Final[float] = 0.11
D: Final[float] = 0.29
EXP_RHO: Final[float] = 0.13
HW_LEN: Final[int] = 10

# probe.cpp makeCapVolCurve()
CAP_VOLS: Final[list[float]] = [
    14.40, 17.15, 16.81, 16.64, 16.17, 15.78, 15.40, 15.21, 14.86, 14.54,
]

# probe.cpp: Hull & White factor loadings plus the extra normalisation that
# makes the eigenvectors orthogonal.
HW_COMPONENTS: Final[list[float]] = [
    0.85549771, 0.46707264, 0.22353259,
    0.91915359, 0.37716089, 0.11360610,
    0.96438280, 0.26413316, -0.01412414,
    0.97939148, 0.13492952, -0.15028753,
    0.95970595, -0.00000000, -0.28100621,
    0.97939148, -0.13492952, -0.15028753,
    0.96438280, -0.26413316, -0.01412414,
    0.91915359, -0.37716089, 0.11360610,
    0.85549771, -0.46707264, 0.22353259,
]


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/legacy/lmm")


@pytest.fixture(autouse=True)
def _pinned_evaluation_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """Pin the probe's evaluation date (probe.cpp ``makeIndex1Y`` / ``makeIndex6M``)."""
    settings = ObservableSettings()
    previous = settings.evaluation_date
    settings.evaluation_date = Euribor1Y(None).fixing_calendar().adjust(ANCHOR)
    try:
        yield
    finally:
        settings.evaluation_date = previous


def _assert_close(actual: Any, expected: Any, label: str, *, abs_tol: float | None = None) -> None:
    act = np.asarray(actual, dtype=np.float64)
    exp = np.asarray(expected, dtype=np.float64)
    assert act.shape == exp.shape, f"{label}: shape {act.shape} != {exp.shape}"
    for idx, (a, e) in enumerate(zip(act.ravel(), exp.ravel(), strict=True)):
        if abs_tol is None:
            tight(float(a), float(e), reason=f"{label}[{idx}]")
        else:
            assert abs(float(a) - float(e)) <= abs_tol, f"{label}[{idx}]: {a} != {e}"


# --- LfmCovarianceProxy, analytic branch --------------------------------------


@pytest.fixture(scope="module")
def analytic_proxy(cpp: dict[str, Any]) -> LfmCovarianceProxy:
    vol = LmLinearExponentialVolatilityModel(
        cpp["lm_linexp_volatility"]["fixing_times"], A, B, C, D
    )
    corr = LmExponentialCorrelationModel(SIZE, EXP_RHO)
    return LfmCovarianceProxy(vol, corr)


def test_proxy_size_and_factors_come_from_the_correlation_model(
    cpp: dict[str, Any], analytic_proxy: LfmCovarianceProxy
) -> None:
    ref = cpp["lfm_covariance_proxy"]
    assert analytic_proxy.size() == ref["size"] == SIZE
    assert analytic_proxy.factors() == ref["factors"] == SIZE
    assert analytic_proxy.correlation_model().size() == SIZE
    assert analytic_proxy.volatility_model().size() == SIZE


def test_proxy_rejects_mismatched_model_sizes() -> None:
    vol = LmLinearExponentialVolatilityModel([0.5, 1.0, 1.5], A, B, C, D)
    corr = LmExponentialCorrelationModel(5, EXP_RHO)
    with pytest.raises(LibraryException, match="different size"):
        LfmCovarianceProxy(vol, corr)


@pytest.mark.parametrize(("t", "key"), [(0.37, "diffusion_t0_37"), (2.55, "diffusion_t2_55")])
def test_proxy_diffusion_matches_cpp(
    cpp: dict[str, Any], analytic_proxy: LfmCovarianceProxy, t: float, key: str
) -> None:
    _assert_close(analytic_proxy.diffusion(t), cpp["lfm_covariance_proxy"][key], key)


@pytest.mark.parametrize(("t", "key"), [(0.37, "covariance_t0_37"), (2.55, "covariance_t2_55")])
def test_proxy_covariance_matches_cpp(
    cpp: dict[str, Any], analytic_proxy: LfmCovarianceProxy, t: float, key: str
) -> None:
    _assert_close(analytic_proxy.covariance(t), cpp["lfm_covariance_proxy"][key], key)


def test_proxy_covariance_equals_diffusion_times_its_transpose(
    analytic_proxy: LfmCovarianceProxy,
) -> None:
    """C++ test-suite testSimpleCovarianceModels, tolerance 1e-14.

    ``covariance`` is computed analytically (vol * rho * vol) and ``diffusion``
    from the correlation pseudo-root, so the identity is a genuine cross-check
    of the two independent routes.
    """
    t = 0.31
    while t < 4.6:
        d = analytic_proxy.diffusion(t)
        recon = analytic_proxy.covariance(t) - d @ d.T
        assert np.max(np.abs(recon)) <= 1e-14, f"t={t}"
        t += 0.31


def test_proxy_integrated_covariance_analytic_branch_matches_cpp(
    cpp: dict[str, Any], analytic_proxy: LfmCovarianceProxy
) -> None:
    """Time-independent correlation + analytic integratedVariance -> fast path."""
    ref = cpp["lfm_covariance_proxy"]
    got = [
        [analytic_proxy.integrated_covariance_scalar(i, j, 2.3) for j in range(SIZE)]
        for i in range(SIZE)
    ]
    _assert_close(got, ref["integrated_covariance_analytic_t2_3"], "proxy.ic_analytic")


# --- LfmCovarianceProxy, numerical fallback + base-class routine ---------------


@pytest.fixture(scope="module")
def numeric_proxy(cpp: dict[str, Any]) -> LfmCovarianceProxy:
    """LmFixedVolatilityModel has no integratedVariance, forcing quadrature."""
    ref = cpp["lfm_covariance_proxy_numeric"]
    vol = LmFixedVolatilityModel(
        np.asarray(ref["volatilities"], dtype=np.float64), ref["start_times"]
    )
    corr = LmExponentialCorrelationModel(int(ref["size"]), 0.31)
    return LfmCovarianceProxy(vol, corr)


def test_proxy_integrated_covariance_falls_back_to_quadrature(
    cpp: dict[str, Any], numeric_proxy: LfmCovarianceProxy
) -> None:
    """The C++ try/catch around the analytic call is load-bearing.

    Tolerance: TIGHT. Both sides run the SAME algorithm (64 sub-intervals of
    GaussKronrodAdaptive(1e-10, 10000) with identical nodes and the identical
    bisection rule), so agreement is a floating-point-summation question, not
    a quadrature-accuracy one.
    """
    ref = cpp["lfm_covariance_proxy_numeric"]
    n = int(ref["size"])
    got = [
        [numeric_proxy.integrated_covariance_scalar(i, j, 0.5) for j in range(n)]
        for i in range(n)
    ]
    _assert_close(got, ref["integrated_covariance_numeric_t0_5"], "proxy.ic_numeric")


def test_proxy_integrated_covariance_over_a_piecewise_integrand(
    cpp: dict[str, Any], numeric_proxy: LfmCovarianceProxy
) -> None:
    """t = 1.05 crosses a start-time boundary, so the integrand really is piecewise.

    Only the ``min(i, j) >= 1`` block is pinned: below it C++'s scalar
    ``LmFixedVolatilityModel::volatility`` reads out of bounds (see
    ``test_lm_volatility_models.py``), so there is no defined C++ answer.
    """
    ref = cpp["lfm_covariance_proxy_numeric"]
    n = int(ref["size"])
    got = [
        [numeric_proxy.integrated_covariance_scalar(i, j, 1.05) for j in range(1, n)]
        for i in range(1, n)
    ]
    _assert_close(
        got, ref["integrated_covariance_numeric_t1_05_upper_block"], "proxy.ic_numeric_hi"
    )


def test_base_class_integrated_covariance_matches_cpp(
    cpp: dict[str, Any], numeric_proxy: LfmCovarianceProxy
) -> None:
    """``LfmCovarianceParameterization.integrated_covariance`` — the matrix form.

    Uses the VECTOR volatility accessor, which zero-fills already-fixed
    forwards, so the whole matrix is well defined at t = 1.05.
    """
    ref = cpp["lfm_covariance_proxy_numeric"]
    _assert_close(
        numeric_proxy.integrated_covariance(1.05),
        ref["base_integrated_covariance_t1_05"],
        "proxy.base_ic",
    )


def test_base_class_integrated_covariance_rejects_a_state_vector(
    numeric_proxy: LfmCovarianceProxy,
) -> None:
    with pytest.raises(LibraryException, match="can not handle given x"):
        numeric_proxy.integrated_covariance(1.0, np.asarray([0.01, 0.02, 0.03]))


# --- LfmHullWhiteParameterization ---------------------------------------------


@pytest.fixture(scope="module")
def hw_setup() -> tuple[LiborForwardModelProcess, CapletVarianceCurve]:
    """probe.cpp ``makeIndex1Y`` + ``makeCapVolCurve`` + a size-10 process."""
    settings = ObservableSettings()
    previous = settings.evaluation_date
    try:
        day_counter = Actual360()
        today = Euribor1Y(None).fixing_calendar().adjust(ANCHOR)
        settings.evaluation_date = today
        spot = Euribor1Y(None).fixing_calendar().advance(
            today, Euribor1Y(None).fixing_days(), TimeUnit.Days
        )
        curve = InterpolatedZeroCurve(
            [spot, Date.from_ymd(4, Month.September, 2018)], [0.01, 0.08], day_counter
        )
        index = Euribor1Y(curve)

        # The caplet curve is built off a size-(len+1) process, exactly as
        # the C++ test-suite does.
        wide_process = LiborForwardModelProcess(HW_LEN + 1, index)
        cap_vol = CapletVarianceCurve(
            reference_date=today,
            dates=[wide_process.fixing_dates()[i + 1] for i in range(HW_LEN)],
            caplet_vol_curve=[v / 100.0 for v in CAP_VOLS],
            day_counter=ActualActual(Convention.ISDA),
        )
        return LiborForwardModelProcess(HW_LEN, index), cap_vol
    finally:
        settings.evaluation_date = previous


def test_hull_white_process_grid_matches_cpp(
    cpp: dict[str, Any], hw_setup: tuple[LiborForwardModelProcess, CapletVarianceCurve]
) -> None:
    ref = cpp["lfm_hull_white_1factor"]
    process, _ = hw_setup
    _assert_close(process.fixing_times(), ref["process_fixing_times"], "hw.fixing_times")
    assert [d.serial_number() for d in process.fixing_dates()] == ref[
        "process_fixing_date_serials"
    ]


@pytest.fixture(scope="module")
def hw1(
    hw_setup: tuple[LiborForwardModelProcess, CapletVarianceCurve],
) -> LfmHullWhiteParameterization:
    process, cap_vol = hw_setup
    return LfmHullWhiteParameterization(process, cap_vol)


def test_hull_white_one_factor_inspectors(
    cpp: dict[str, Any], hw1: LfmHullWhiteParameterization
) -> None:
    ref = cpp["lfm_hull_white_1factor"]
    assert hw1.size() == ref["size"] == HW_LEN
    assert hw1.factors() == ref["factors"] == 1


def test_hull_white_lambda_bootstrapping_matches_cpp(
    cpp: dict[str, Any], hw1: LfmHullWhiteParameterization
) -> None:
    """C++ test-suite testLambdaBootstrapping pins these to 1e-10 against
    hand-computed values; here they are pinned to the probe at TIGHT.
    """
    ref = cpp["lfm_hull_white_1factor"]
    cov0 = hw1.covariance(0.0)
    got = [float(np.sqrt(cov0[i, i])) for i in range(1, HW_LEN)]
    _assert_close(got, ref["lambdas"], "hw1.lambdas")


@pytest.mark.parametrize(
    ("method", "t", "key"),
    [
        ("diffusion", 0.0, "diffusion_t0"),
        ("diffusion", 2.5, "diffusion_t2_5"),
        ("covariance", 0.0, "covariance_t0"),
        ("covariance", 2.5, "covariance_t2_5"),
        ("integrated_covariance", 3.3, "integrated_covariance_t3_3"),
    ],
)
def test_hull_white_one_factor_matrices_match_cpp(
    cpp: dict[str, Any],
    hw1: LfmHullWhiteParameterization,
    method: str,
    t: float,
    key: str,
) -> None:
    _assert_close(
        getattr(hw1, method)(t), cpp["lfm_hull_white_1factor"][key], f"hw1.{key}"
    )


def test_hull_white_rejects_a_multi_factor_request_without_a_correlation_matrix(
    hw_setup: tuple[LiborForwardModelProcess, CapletVarianceCurve],
) -> None:
    process, cap_vol = hw_setup
    with pytest.raises(LibraryException, match="correlation matrix must be given"):
        LfmHullWhiteParameterization(process, cap_vol, None, 3)


@pytest.fixture(scope="module")
def hw3_correlation() -> Any:
    comp = np.asarray(HW_COMPONENTS, dtype=np.float64).reshape(9, 3)
    return comp @ comp.T


@pytest.fixture(scope="module")
def hw3(
    hw_setup: tuple[LiborForwardModelProcess, CapletVarianceCurve], hw3_correlation: Any
) -> LfmHullWhiteParameterization:
    process, cap_vol = hw_setup
    return LfmHullWhiteParameterization(process, cap_vol, hw3_correlation, 3)


def test_hull_white_three_factor_inspectors_and_input(
    cpp: dict[str, Any], hw3: LfmHullWhiteParameterization, hw3_correlation: Any
) -> None:
    ref = cpp["lfm_hull_white_3factor"]
    assert hw3.size() == ref["size"] == HW_LEN
    assert hw3.factors() == ref["factors"] == 3
    _assert_close(hw3_correlation, ref["correlation_input"], "hw3.correlation_input")
    assert hw3.diffusion(0.0).shape == (HW_LEN, 3)


@pytest.mark.parametrize(
    ("method", "t", "key"),
    [
        ("diffusion", 0.0, "diffusion_t0"),
        ("diffusion", 2.5, "diffusion_t2_5"),
        ("covariance", 0.0, "covariance_t0"),
        ("integrated_covariance", 3.3, "integrated_covariance_t3_3"),
    ],
)
def test_hull_white_three_factor_matrices_match_cpp(
    cpp: dict[str, Any],
    hw3: LfmHullWhiteParameterization,
    method: str,
    t: float,
    key: str,
) -> None:
    _assert_close(
        getattr(hw3, method)(t), cpp["lfm_hull_white_3factor"][key], f"hw3.{key}"
    )


def test_hull_white_rejects_a_correlation_matrix_of_the_wrong_size(
    hw_setup: tuple[LiborForwardModelProcess, CapletVarianceCurve],
) -> None:
    process, cap_vol = hw_setup
    with pytest.raises(LibraryException, match="wrong dimesion"):
        LfmHullWhiteParameterization(process, cap_vol, np.eye(4), 1)


def test_hull_white_rejects_more_factors_than_forwards(
    hw_setup: tuple[LiborForwardModelProcess, CapletVarianceCurve],
    hw3_correlation: Any,
) -> None:
    process, cap_vol = hw_setup
    with pytest.raises(LibraryException, match="too many factors"):
        LfmHullWhiteParameterization(process, cap_vol, hw3_correlation, HW_LEN)

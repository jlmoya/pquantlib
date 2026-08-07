"""Abcd helper classes + the XABR fit-error formula, against C++ v1.43.

Reference: ``migration-harness/references/v143/ts/abcd.json``
(probe ``migration-harness/cpp/probes/v143_ts_abcd/probe.cpp``).

Four things are pinned here, all of which live in the C++ ``termstructures``
subsystem even though PQuantLib files them under ``math/``:

* ``AbcdSquared``                    — abcd.hpp:93 / abcd.cpp:99-105
* ``AbcdParametersTransformation``   — abcdcalibration.hpp:69 / .cpp:36-52
* ``AbcdError``                      — abcdcalibration.hpp:44
* the shared fit-error formula       — abcdcalibration.cpp:179-196 and
  xabrinterpolation.hpp:270-285

**Why the formula needed its own probe.** C++ reports

    error    = sqrt( n * SUM_i w_i e_i^2 / (n - 1) )
    maxError = max_i |e_i|                              (UNWEIGHTED)

with weights that default to a uniform ``1/n``, so the unweighted case is the
SAMPLE standard deviation ``sqrt(SUM e^2 / (n-1))``. PQuantLib's
``SabrInterpolation``, ``ZabrInterpolation``, ``AbcdInterpolation`` and
``AbcdCalibration`` all reported ``sqrt(mean(e^2))`` instead — smaller by
``sqrt(n/(n-1))`` in the unweighted case, and wrong in a second way when
vega-weighted, where they also weighted ``maxError`` (C++ never does).
``sabr_interpolation.py`` already contained the CORRECT helper
(``xabr_interpolation_error``), written for and used by
``no_arb_sabr_interpolation.py`` and ``svi_interpolation.py``, but its own
class never called it.

Every C++ case below fixes ALL model parameters, so C++ short-circuits the
optimisation entirely (xabrinterpolation.hpp:161-168, abcdcalibration.cpp:117-122)
and no ``LevenbergMarquardt`` runs. That matters: PQuantLib's
``LevenbergMarquardt`` is a scipy delegation whose C++-parity tests are
xfailed, so anything downstream of it could not be pinned TIGHT. Nothing here
is downstream of it.

No evaluation date is involved — every quantity is a pure function of its
arguments, and the probe sets none.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from pquantlib.math.interpolations.abcd_calibration import (
    AbcdCalibration,
    AbcdError,
    AbcdParametersTransformation,
)
from pquantlib.math.interpolations.sabr_interpolation import SabrInterpolation
from pquantlib.models.marketmodels.models.abcd_function import (
    AbcdFunction,
    AbcdSquared,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.testing import reference_reader, tolerance

CPP: dict[str, Any] = reference_reader.load("v143/ts/abcd")


def test_reference_is_v143() -> None:
    assert CPP["quantlib_version"] == "1.43"


# --- AbcdSquared ------------------------------------------------------------


@pytest.mark.parametrize("case", CPP["abcd_squared"], ids=range(len(CPP["abcd_squared"])))
def test_abcd_squared_matches_cpp(case: dict[str, Any]) -> None:
    sq = AbcdSquared(
        case["a"], case["b"], case["c"], case["d"], case["T"], case["S"]
    )
    tolerance.tight(sq(case["t"]), case["value"])


@pytest.mark.parametrize("case", CPP["abcd_squared"], ids=range(len(CPP["abcd_squared"])))
def test_abcd_squared_equals_instantaneous_covariance(case: dict[str, Any]) -> None:
    """C++ ``AbcdSquared::operator()`` IS ``AbcdFunction::covariance(t,T,S)``.

    # C++ parity: abcd.cpp:104 delegates to the three-argument overload at
    # abcd.cpp:43-45, which PQuantLib names ``instantaneous_covariance``.
    """
    fn = AbcdFunction(case["a"], case["b"], case["c"], case["d"])
    tolerance.tight(
        fn.instantaneous_covariance(case["t"], case["T"], case["S"]),
        case["covariance"],
    )
    tolerance.tight(
        fn.instantaneous_covariance(case["t"], case["T"], case["S"]),
        case["instantaneousCovariance"],
    )


def test_abcd_squared_is_not_the_square_of_f_when_t_differs_from_s() -> None:
    """Guard against reading the class name literally.

    The C++ name says "squared" but the body is ``f(T-t) * f(S-t)``; a port
    that returns ``f(t)**2`` passes every ``T == S`` case and fails the rest.
    """
    asymmetric = [c for c in CPP["abcd_squared"] if c["T"] != c["S"] and c["t"] > 0.0]
    assert asymmetric, "probe must include a T != S case"
    case = asymmetric[0]
    sq = AbcdSquared(case["a"], case["b"], case["c"], case["d"], case["T"], case["S"])
    fn = AbcdFunction(case["a"], case["b"], case["c"], case["d"])
    square_of_f = fn(case["t"]) ** 2
    assert abs(sq(case["t"]) - square_of_f) > 1e-6 * abs(square_of_f)


# --- AbcdParametersTransformation -------------------------------------------


@pytest.mark.parametrize(
    "case", CPP["abcd_transformation"], ids=range(len(CPP["abcd_transformation"]))
)
def test_abcd_transformation_direct_matches_cpp(case: dict[str, Any]) -> None:
    t = AbcdParametersTransformation()
    got = t.direct(np.asarray(case["x"], dtype=np.float64))
    for g, e in zip(got, case["direct"], strict=True):
        tolerance.tight(float(g), e)


@pytest.mark.parametrize(
    "case", CPP["abcd_transformation"], ids=range(len(CPP["abcd_transformation"]))
)
def test_abcd_transformation_roundtrips_from_unconstrained(case: dict[str, Any]) -> None:
    t = AbcdParametersTransformation()
    back = t.inverse(t.direct(np.asarray(case["x"], dtype=np.float64)))
    for g, e in zip(back, case["inverse_of_direct"], strict=True):
        tolerance.tight(float(g), e)
    for g, x, e in zip(back, case["x"], case["roundtrip_residual"], strict=True):
        tolerance.tight(float(g) - x, e)


@pytest.mark.parametrize(
    "case",
    CPP["abcd_transformation_inverse"],
    ids=range(len(CPP["abcd_transformation_inverse"])),
)
def test_abcd_transformation_inverse_matches_cpp(case: dict[str, Any]) -> None:
    t = AbcdParametersTransformation()
    got = t.inverse(np.asarray(case["x"], dtype=np.float64))
    for g, e in zip(got, case["inverse"], strict=True):
        tolerance.tight(float(g), e)


@pytest.mark.parametrize(
    "case",
    CPP["abcd_transformation_inverse"],
    ids=range(len(CPP["abcd_transformation_inverse"])),
)
def test_abcd_transformation_roundtrips_from_constrained(case: dict[str, Any]) -> None:
    t = AbcdParametersTransformation()
    back = t.direct(t.inverse(np.asarray(case["x"], dtype=np.float64)))
    for g, e in zip(back, case["direct_of_inverse"], strict=True):
        tolerance.tight(float(g), e)


def test_direct_lands_inside_the_abcd_feasible_region() -> None:
    """``direct`` exists to enforce c > 0, d > 0, a + d > 0 by construction.

    # C++ parity: abcdcalibration.cpp:37-43 — ``a = exp(x0) - d`` makes
    # ``a + d = exp(x0)`` positive for every finite ``x0``.
    """
    t = AbcdParametersTransformation()
    for case in CPP["abcd_transformation"]:
        a, _b, c, d = t.direct(np.asarray(case["x"], dtype=np.float64))
        assert c > 0.0
        assert d > 0.0
        assert a + d > 0.0


# --- AbcdCalibration diagnostics --------------------------------------------


def _calibration(case: dict[str, Any]) -> AbcdCalibration:
    return AbcdCalibration(
        case["times"],
        case["blackVols"],
        case["a"],
        case["b"],
        case["c"],
        case["d"],
        a_is_fixed=True,
        b_is_fixed=True,
        c_is_fixed=True,
        d_is_fixed=True,
        vega_weighted=case["vegaWeighted"],
    )


@pytest.mark.parametrize("name", sorted(CPP["abcd_calibration"]))
def test_abcd_calibration_values_match_cpp(name: str) -> None:
    case = CPP["abcd_calibration"][name]
    calib = _calibration(case)
    calib.compute()
    for t, e in zip(case["times"], case["values"], strict=True):
        tolerance.tight(calib.value(t), e)


@pytest.mark.parametrize("name", sorted(CPP["abcd_calibration"]))
def test_abcd_calibration_error_matches_cpp(name: str) -> None:
    """``sqrt(n * sum w_i e_i^2 / (n-1))`` — abcdcalibration.cpp:179-187."""
    case = CPP["abcd_calibration"][name]
    calib = _calibration(case)
    calib.compute()
    tolerance.tight(calib.error(), case["error"])


@pytest.mark.parametrize("name", sorted(CPP["abcd_calibration"]))
def test_abcd_calibration_max_error_matches_cpp(name: str) -> None:
    """``max |e_i|``, UNWEIGHTED even when vega-weighted.

    # C++ parity: abcdcalibration.cpp:189-196 never touches ``weights_``.
    """
    case = CPP["abcd_calibration"][name]
    calib = _calibration(case)
    calib.compute()
    tolerance.tight(calib.max_error(), case["maxError"])


@pytest.mark.parametrize("name", sorted(CPP["abcd_calibration"]))
def test_abcd_calibration_errors_vector_matches_cpp(name: str) -> None:
    case = CPP["abcd_calibration"][name]
    calib = _calibration(case)
    calib.compute()
    for g, e in zip(calib.errors(), case["errors"], strict=True):
        tolerance.tight(g, e)


@pytest.mark.parametrize("name", sorted(CPP["abcd_calibration"]))
def test_abcd_calibration_k_matches_cpp(name: str) -> None:
    case = CPP["abcd_calibration"][name]
    calib = _calibration(case)
    calib.compute()
    for g, e in zip(calib.k(case["times"], case["blackVols"]), case["k"], strict=True):
        tolerance.tight(g, e)


def test_max_error_is_identical_weighted_and_unweighted() -> None:
    """The vega weights must not reach ``maxError``.

    The n=3 pair differs only in ``vegaWeighted``; C++ reports the SAME
    ``maxError`` for both and a DIFFERENT ``error``. A port that weights both
    passes neither half of this test.
    """
    unw = CPP["abcd_calibration"]["n3_unweighted"]
    veg = CPP["abcd_calibration"]["n3_vegaweighted"]
    tolerance.exact(veg["maxError"], unw["maxError"])
    assert veg["error"] != unw["error"]

    a = _calibration(unw)
    a.compute()
    b = _calibration(veg)
    b.compute()
    tolerance.exact(b.max_error(), a.max_error())
    assert b.error() != a.error()


def test_error_uses_n_minus_one_not_n() -> None:
    """Pin the denominator directly from the C++ numbers.

    For the unweighted cases ``w_i == 1/n``, so C++'s
    ``sqrt(n * sum w e^2 / (n-1))`` collapses to ``sqrt(sum e^2 / (n-1))``.
    Recomputing that from the emitted per-point values must reproduce
    ``error`` exactly, while ``sqrt(mean(e^2))`` must not — three different
    ``n`` are checked so the leading ``n`` and the trailing ``n-1`` cannot both
    be explained by some other convention.
    """
    for name in ("n2_unweighted", "n3_unweighted", "n6_unweighted"):
        case = CPP["abcd_calibration"][name]
        e = [v - b for v, b in zip(case["values"], case["blackVols"], strict=True)]
        n = len(e)
        ss = sum(x * x for x in e)
        tolerance.tight(math.sqrt(ss / (n - 1)), case["error"])
        assert not math.isclose(math.sqrt(ss / n), case["error"], rel_tol=1e-6)


# --- AbcdError --------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    CPP["abcd_error_via_public_api"],
    ids=range(len(CPP["abcd_error_via_public_api"])),
)
def test_abcd_error_value_matches_cpp(case: dict[str, Any]) -> None:
    """``AbcdError::value(x)`` == ``error()`` evaluated at ``direct(x)``.

    # C++ parity: abcdcalibration.hpp:47-54.
    """
    ref = CPP["abcd_calibration"]["n6_unweighted"]
    calib = AbcdCalibration(ref["times"], ref["blackVols"])
    cost = AbcdError(calib)
    tolerance.tight(cost.value(np.asarray(case["x"], dtype=np.float64)), case["value"])


@pytest.mark.parametrize(
    "case",
    CPP["abcd_error_via_public_api"],
    ids=range(len(CPP["abcd_error_via_public_api"])),
)
def test_abcd_error_values_matches_cpp(case: dict[str, Any]) -> None:
    """``AbcdError::values(x)`` == ``errors()`` at ``direct(x)``.

    # C++ parity: abcdcalibration.hpp:55-63.
    """
    ref = CPP["abcd_calibration"]["n6_unweighted"]
    calib = AbcdCalibration(ref["times"], ref["blackVols"])
    cost = AbcdError(calib)
    got = cost.values(np.asarray(case["x"], dtype=np.float64))
    for g, e in zip(got, case["values"], strict=True):
        tolerance.tight(float(g), e)


def test_abcd_error_mutates_the_calibration_it_points_at() -> None:
    """C++ writes the trial parameters into the calibration; so must Python.

    # C++ parity: abcdcalibration.hpp:49-52 assigns ``abcd_->a_`` … ``d_``
    # before reading ``error()``. It is observable, and a port that evaluates
    # the cost against a local copy silently breaks ``compute()``.
    """
    ref = CPP["abcd_calibration"]["n6_unweighted"]
    case = CPP["abcd_error_via_public_api"][0]
    calib = AbcdCalibration(ref["times"], ref["blackVols"])
    AbcdError(calib).value(np.asarray(case["x"], dtype=np.float64))
    for got, expected in zip(
        (calib.a(), calib.b(), calib.c(), calib.d()), case["direct"], strict=True
    ):
        tolerance.tight(got, expected)


def test_abcd_error_value_is_not_the_cost_function_default() -> None:
    """``AbcdError`` overrides ``value``; the base-class default is different.

    ``CostFunction::value`` defaults to ``sqrt(mean(values(x)^2))``
    (costfunction.hpp:38-43), whereas ``AbcdError::value`` returns
    ``error()``, which carries the extra ``n/(n-1)`` factor. Pinning the gap
    keeps a future refactor from "simplifying" the override away.
    """
    ref = CPP["abcd_calibration"]["n6_unweighted"]
    case = CPP["abcd_error_via_public_api"][0]
    calib = AbcdCalibration(ref["times"], ref["blackVols"])
    cost = AbcdError(calib)
    x = np.asarray(case["x"], dtype=np.float64)
    vals = cost.values(x)
    n = vals.size
    default = math.sqrt(float(np.sum(vals * vals)) / n)
    tolerance.tight(cost.value(x), case["value"])
    tolerance.tight(cost.value(x) / default, math.sqrt(n * n / (n - 1)))


# --- XABR interpolation error ----------------------------------------------


def _sabr(case: dict[str, Any]) -> SabrInterpolation:
    return SabrInterpolation(
        strikes=case["strikes"],
        volatilities=case["vols"],
        expiry_time=case["t"],
        forward=case["forward"],
        alpha=case["alpha"],
        beta=case["beta"],
        nu=case["nu"],
        rho=case["rho"],
        alpha_is_fixed=True,
        beta_is_fixed=True,
        nu_is_fixed=True,
        rho_is_fixed=True,
        vega_weighted=case["vegaWeighted"],
        volatility_type=VolatilityType.ShiftedLognormal,
    )


@pytest.mark.parametrize("name", sorted(CPP["sabr_interpolation_error"]))
def test_sabr_model_vols_match_cpp(name: str) -> None:
    """The SABR formula itself, so the error formula is separable from it."""
    case = CPP["sabr_interpolation_error"][name]
    interp = _sabr(case)
    for k, e in zip(case["strikes"], case["model"], strict=True):
        tolerance.tight(interp(k), e)


@pytest.mark.parametrize("name", sorted(CPP["sabr_interpolation_error"]))
def test_sabr_rms_error_matches_cpp(name: str) -> None:
    """``SABRInterpolation::rmsError()`` — sabrinterpolation.hpp:183.

    Reaches ``XABRInterpolationImpl::interpolationError``
    (xabrinterpolation.hpp:270-274) through the all-parameters-fixed
    short-circuit at hpp:161-168, so no optimiser is involved.
    """
    case = CPP["sabr_interpolation_error"][name]
    tolerance.tight(_sabr(case).rms_error(), case["rmsError"])


@pytest.mark.parametrize("name", sorted(CPP["sabr_interpolation_error"]))
def test_sabr_max_error_matches_cpp(name: str) -> None:
    """``maxError()`` is UNWEIGHTED — xabrinterpolation.hpp:276-285."""
    case = CPP["sabr_interpolation_error"][name]
    tolerance.tight(_sabr(case).max_error(), case["maxError"])


def test_sabr_max_error_ignores_the_vega_weights() -> None:
    """Same data, ``vegaWeighted`` toggled: C++ moves ``rms`` but not ``max``."""
    unw = CPP["sabr_interpolation_error"]["sabr_n3_unweighted"]
    veg = CPP["sabr_interpolation_error"]["sabr_n3_vegaweighted"]
    tolerance.exact(veg["maxError"], unw["maxError"])
    assert veg["rmsError"] != unw["rmsError"]
    tolerance.exact(_sabr(veg).max_error(), _sabr(unw).max_error())
    assert _sabr(veg).rms_error() != _sabr(unw).rms_error()


def test_sabr_rms_error_uses_n_minus_one() -> None:
    """Denominator check, straight off the emitted residuals.

    Unweighted, ``w_i == 1/n``, so C++'s ``sqrt(n * sum w e^2 / (n-1))`` is
    ``sqrt(sum e^2 / (n-1))``. ``sqrt(mean(e^2))`` — what this port used to
    return — is smaller by ``sqrt(n/(n-1))`` and must NOT match.
    """
    for name in ("sabr_n2_unweighted", "sabr_n3_unweighted", "sabr_n6_unweighted"):
        case = CPP["sabr_interpolation_error"][name]
        e = case["errors"]
        n = len(e)
        ss = sum(x * x for x in e)
        tolerance.tight(math.sqrt(ss / (n - 1)), case["rmsError"])
        assert not math.isclose(math.sqrt(ss / n), case["rmsError"], rel_tol=1e-6)

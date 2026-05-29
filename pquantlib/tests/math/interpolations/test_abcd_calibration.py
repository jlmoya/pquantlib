"""Tests for AbcdCalibration — Rebonato (a, b, c, d) volatility-curve fit.

# C++ parity: ql/termstructures/volatility/abcdcalibration.{hpp,cpp}
# (v1.42.1).

The W2-B probe (cluster/w2b.json :: abcd_calibration) records both the
input curve and the C++ ``AbcdCalibration::compute()`` result. The Python
``scipy.optimize.least_squares(trf)`` arm typically converges to the
*global* minimum on noiseless abcd-shape data while the C++
``LevenbergMarquardt`` may settle at a local minimum (the L10-C
``AbcdInterpolation`` test documents this divergence). We assert the
Python recovery is correct against the truth parameters rather than
the C++ fit.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.interpolations.abcd_calibration import AbcdCalibration
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("cluster/w2b")


def _make() -> AbcdCalibration:
    block = reference_reader.load("cluster/w2b")["abcd_calibration"]
    times = list(block["times"])
    vols = list(block["vols"])
    # C++ probe uses (-0.05, 0.15, 0.50, 0.16) starting point.
    calib = AbcdCalibration(times, vols, a=-0.05, b=0.15, c=0.50, d=0.16)
    calib.compute()
    return calib


# --- recovery ---------------------------------------------------------------


def test_recovers_truth_parameters_on_noiseless_input(cpp: dict[str, Any]) -> None:
    """Python TRF converges to the truth (a, b, c, d) on noiseless data.

    Same shape as the L10-C AbcdInterpolation test — scipy TRF finds
    the global minimum where C++ LM settles at a local minimum.
    """
    block = cpp["abcd_calibration"]
    calib = _make()
    tolerance.loose(calib.a(), float(block["a_true"]),
                    reason="trf recovers truth on noiseless data")
    tolerance.loose(calib.b(), float(block["b_true"]),
                    reason="trf recovers truth on noiseless data")
    tolerance.loose(calib.c(), float(block["c_true"]),
                    reason="trf recovers truth on noiseless data")
    tolerance.loose(calib.d(), float(block["d_true"]),
                    reason="trf recovers truth on noiseless data")
    # Tight residual on a 4-param fit to 6 noiseless data points → ~1e-13.
    assert calib.error() < 1e-8
    assert calib.max_error() < 1e-8


def test_pillar_values_match_input_to_loose(cpp: dict[str, Any]) -> None:
    """At each input time the fitted abcd reproduces the market vol.

    Tighter than the cross-validation check — the Python solver achieves
    residuals near scipy's xtol=ftol=1e-12.
    """
    block = cpp["abcd_calibration"]
    times = [float(e) for e in block["times"]]
    vols = [float(e) for e in block["vols"]]
    calib = _make()
    for t, v in zip(times, vols, strict=True):
        tolerance.custom(
            calib.value(t), v,
            abs_tol=1e-10, rel_tol=1e-10,
            reason="trf xtol=ftol=1e-12 residual floor",
        )


# --- construction + validation ----------------------------------------------


def test_validates_negative_time_inputs() -> None:
    with pytest.raises(LibraryException, match="non-negative times"):
        AbcdCalibration([1.0, -0.5, 2.0], [0.1, 0.15, 0.2])


def test_validates_mismatched_lengths() -> None:
    with pytest.raises(LibraryException, match="length mismatch"):
        AbcdCalibration([0.5, 1.0, 2.0], [0.1, 0.15])


def test_validates_initial_guess_against_abcd_constraints() -> None:
    """C++ ``AbcdMathFunction::validate`` checks (c >= 0, d >= 0, a+d >= 0)."""
    # c < 0
    with pytest.raises(LibraryException, match="c parameter"):
        AbcdCalibration([0.5, 1.0], [0.1, 0.15], a=-0.06, b=0.17, c=-0.1, d=0.17)
    # d < 0
    with pytest.raises(LibraryException, match="d parameter"):
        AbcdCalibration([0.5, 1.0], [0.1, 0.15], a=-0.06, b=0.17, c=0.54, d=-0.1)
    # a + d < 0
    with pytest.raises(LibraryException, match="a\\+d"):
        AbcdCalibration([0.5, 1.0], [0.1, 0.15], a=-0.5, b=0.17, c=0.54, d=0.1)


# --- fixed parameters -------------------------------------------------------


def test_all_fixed_yields_initial_guess() -> None:
    """If all four params are fixed, ``compute()`` returns the initial guess."""
    times = [0.25, 0.5, 1.0, 2.0]
    vols = [0.10, 0.12, 0.15, 0.13]
    calib = AbcdCalibration(
        times, vols,
        a=-0.06, b=0.17, c=0.54, d=0.17,
        a_is_fixed=True, b_is_fixed=True, c_is_fixed=True, d_is_fixed=True,
    )
    calib.compute()
    assert calib.a() == -0.06
    assert calib.b() == 0.17
    assert calib.c() == 0.54
    assert calib.d() == 0.17
    assert calib.converged()


def test_partial_fix_pins_only_designated_params() -> None:
    """Fixing ``d`` leaves it at the guess; others may move."""
    block = reference_reader.load("cluster/w2b")["abcd_calibration"]
    times = list(block["times"])
    vols = list(block["vols"])
    calib = AbcdCalibration(
        times, vols,
        a=-0.05, b=0.15, c=0.50, d=0.17,  # d fixed at truth
        d_is_fixed=True,
    )
    calib.compute()
    assert calib.d() == 0.17  # untouched
    # a/b/c may converge to truth or to a local minimum;
    # check the residual is small.
    assert calib.error() < 1e-4


# --- k() adjustment factor --------------------------------------------------


def test_k_returns_per_time_adjustment() -> None:
    """``k(t, vols) = vols / model(t)`` per the C++ helper."""
    block = reference_reader.load("cluster/w2b")["abcd_calibration"]
    times = list(block["times"])
    vols = list(block["vols"])
    calib = _make()
    ks = calib.k(times, vols)
    assert len(ks) == len(times)
    # On noiseless data, k(t) ≈ 1 because model(t) ≈ vols.
    for k in ks:
        assert abs(k - 1.0) < 1e-6


def test_k_rejects_mismatched_lengths() -> None:
    calib = _make()
    with pytest.raises(LibraryException, match="length mismatch"):
        calib.k([1.0, 2.0], [0.1])


# --- diagnostics ------------------------------------------------------------


def test_end_criteria_populated_after_compute() -> None:
    calib = _make()
    msg = calib.end_criteria()
    assert isinstance(msg, str)
    assert msg != "uncomputed"


def test_value_evaluates_at_arbitrary_t() -> None:
    """value(t) returns the Rebonato form ``(a + b t) exp(-c t) + d``."""
    calib = _make()
    a, b, c, d = calib.a(), calib.b(), calib.c(), calib.d()
    for t in [0.1, 1.5, 4.0, 7.0]:
        expected = (a + b * t) * np.exp(-c * t) + d
        assert abs(calib.value(t) - expected) < 1e-12

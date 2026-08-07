"""Tests for AbcdCalibration — Rebonato (a, b, c, d) volatility-curve fit.

# C++ parity: ql/termstructures/volatility/abcdcalibration.{hpp,cpp}
# (v1.42.1).

The W2-B probe (cluster/w2b.json :: abcd_calibration) records both the
input curve and the C++ ``AbcdCalibration::compute()`` result.

NOTE ON THE PROBE'S INPUT. ``vols`` in that block is the INSTANTANEOUS
``AbcdMathFunction::operator()`` evaluated at ``times`` with the ``*_true``
parameters — but ``AbcdCalibration`` fits ``abcdBlackVolatility``, the AVERAGE
vol over ``[0, t]`` (abcdcalibration.cpp:163-165 -> abcd.hpp:105-108). The
input is therefore NOT reproducible by the model, which is why C++'s
``*_fitted`` are nowhere near ``*_true`` and its residual floor is ~2.6e-3
rather than ~1e-13. That is a property of the probe's synthetic data, not of
either optimiser.

An earlier revision of this module read that mismatch as "scipy TRF finds the
global minimum where C++ LM settles at a local minimum", and asserted the
Python fit against ``*_true``. It passed only because ``AbcdCalibration.value``
was returning the instantaneous ``abcd_value`` instead of
``abcdBlackVolatility`` — i.e. the port was fitting a different function. With
that fixed, Python reproduces the C++ fit to ~2e-7 on every parameter and
~4e-10 relative on ``error()``, so these tests now assert against C++.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.interpolations.abcd_calibration import AbcdCalibration
from pquantlib.models.marketmodels.models.abcd_function import AbcdFunction
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


def test_recovers_the_cpp_fitted_parameters(cpp: dict[str, Any]) -> None:
    """Python reproduces the C++ ``compute()`` result on the probe's data.

    Tolerance: 1e-6 ABSOLUTE on the parameters, looser than LOOSE (1e-8 rel).
    Derivation: the two runs use different optimisers — C++
    ``LevenbergMarquardt`` (MINPACK ``lmdif``) against
    ``scipy.optimize.least_squares`` ``trf`` with
    ``xtol = ftol = gtol = 1e-12``. Near a minimum the cost is locally
    quadratic, so a parameter displacement ``dp`` perturbs the cost by
    ~``H dp^2``; the two runs agree on ``error()`` to 3.9e-11 relative, which
    admits a parameter displacement of order ``sqrt(3.9e-11) ~ 6e-6``. The
    measured worst-case ABSOLUTE gap is 1.9e-7 (on ``a``, whose magnitude is
    only 0.0108, so its RELATIVE gap is 1.7e-5 — which is why the bound is
    stated absolutely rather than relatively). ``error()`` itself, the quantity
    both optimisers minimise, is checked at 1e-9 relative below.
    """
    block = cpp["abcd_calibration"]
    calib = _make()
    for got, key in (
        (calib.a(), "a_fitted"),
        (calib.b(), "b_fitted"),
        (calib.c(), "c_fitted"),
        (calib.d(), "d_fitted"),
    ):
        tolerance.custom(
            got, float(block[key]), abs_tol=1e-6, rel_tol=0.0,
            reason="different optimisers at the same minimum; see docstring",
        )
    tolerance.custom(
        calib.error(), float(block["error"]), abs_tol=0.0, rel_tol=1e-9,
        reason="the minimised quantity itself",
    )
    tolerance.custom(
        calib.max_error(), float(block["max_error"]), abs_tol=0.0, rel_tol=1e-6,
        reason="max residual, sensitive to the parameter gap above",
    )


def test_pillar_values_match_the_cpp_fitted_curve(cpp: dict[str, Any]) -> None:
    """``value(t)`` at the input times reproduces C++'s ``fitted_at_pillars``.

    Tolerance 1e-6 relative: the fitted curve inherits the ~2e-7 parameter gap
    derived in :func:`test_recovers_the_cpp_fitted_parameters`.
    """
    block = cpp["abcd_calibration"]
    times = [float(e) for e in block["times"]]
    calib = _make()
    for t, e in zip(times, block["fitted_at_pillars"], strict=True):
        tolerance.custom(
            calib.value(t), float(e), abs_tol=1e-9, rel_tol=1e-6,
            reason="inherits the optimiser-gap bound; see the parameter test",
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

    # a/b/c must actually move, and the fit must beat the do-nothing baseline.
    # No magic residual bound: the achievable floor here is set by the probe's
    # synthetic data (instantaneous-form vols fitted with the average-vol
    # model — see the module docstring), and pinning it would be pinning the
    # optimiser, not the port.
    baseline = AbcdCalibration(
        times, vols,
        a=-0.05, b=0.15, c=0.50, d=0.17,
        a_is_fixed=True, b_is_fixed=True, c_is_fixed=True, d_is_fixed=True,
    )
    baseline.compute()
    assert calib.error() < baseline.error()
    assert (calib.a(), calib.b(), calib.c()) != (-0.05, 0.15, 0.50)


# --- k() adjustment factor --------------------------------------------------


def test_k_returns_per_time_adjustment() -> None:
    """``k(t, vols) = vols / model(t)`` per the C++ helper."""
    block = reference_reader.load("cluster/w2b")["abcd_calibration"]
    times = list(block["times"])
    vols = list(block["vols"])
    calib = _make()
    ks = calib.k(times, vols)
    assert len(ks) == len(times)
    # k(t) = vols[i] / value(t_i) by definition (abcdcalibration.cpp:168-178).
    for t, v, k in zip(times, vols, ks, strict=True):
        tolerance.tight(k, float(v) / calib.value(float(t)))


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


def test_value_is_the_average_black_vol_not_the_instantaneous_form() -> None:
    """``value(t)`` is ``AbcdFunction(a,b,c,d).volatility(0, t, t)``.

    # C++ parity: ``AbcdCalibration::value`` (abcdcalibration.cpp:163-165)
    # returns ``abcdBlackVolatility``, which abcd.hpp:105-108 defines as
    # ``AbcdFunction(a,b,c,d).volatility(0., u, u)`` — the AVERAGE vol over
    # ``[0, t]``. An earlier revision returned the INSTANTANEOUS
    # ``(a + b t) exp(-c t) + d`` here; the second assertion pins the two
    # apart so the regression cannot come back silently.
    """
    calib = _make()
    a, b, c, d = calib.a(), calib.b(), calib.c(), calib.d()
    model = AbcdFunction(a, b, c, d)
    for t in [0.1, 1.5, 4.0, 7.0]:
        tolerance.tight(calib.value(t), model.volatility(0.0, t, t))
        instantaneous = (a + b * t) * float(np.exp(-c * t)) + d
        assert abs(calib.value(t) - instantaneous) > 1e-4

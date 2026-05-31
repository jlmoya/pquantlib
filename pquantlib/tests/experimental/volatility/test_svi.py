"""Cross-validate the SVI smile family against the W6-A C++ probe.

Reference: ``migration-harness/references/cluster/w6a.json`` — ``svi``
section. Tests:
  * ``svi_total_variance`` at known log-moneyness points (TIGHT).
  * ``svi_volatility`` / ``SviSmileSection.volatility`` at strikes
    (TIGHT — pure closed-form on both sides).
  * variance at strike = forward*exp(m) equals ``a + b*sigma`` (TIGHT,
    matching the C++ ``testSviSmileSection`` 1e-10 assertion).
  * ``SviInterpolation`` recovers synthetic SVI params (multi-start).
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from pquantlib.experimental.volatility.svi_interpolation import (
    SviInterpolation,
    svi_total_variance,
    svi_volatility,
)
from pquantlib.experimental.volatility.svi_smile_section import SviSmileSection
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("cluster/w6a")


@pytest.fixture(scope="module")
def svi(cpp: dict[str, Any]) -> dict[str, Any]:
    return cpp["svi"]


def test_total_variance_matches_cpp(svi: dict[str, Any]) -> None:
    a, b, sigma, rho, m = (
        svi["a"], svi["b"], svi["sigma"], svi["rho"], svi["m"]
    )
    for k, cpp_tv in zip(svi["k_points"], svi["total_variance"], strict=True):
        tolerance.tight(
            svi_total_variance(a, b, sigma, rho, m, float(k)), float(cpp_tv)
        )


def test_smile_volatility_matches_cpp(svi: dict[str, Any]) -> None:
    sec = SviSmileSection(
        forward=svi["forward"],
        svi_params=(svi["a"], svi["b"], svi["sigma"], svi["rho"], svi["m"]),
        exercise_time=svi["tte"],
    )
    for k, cpp_vol in zip(svi["strikes"], svi["volatility"], strict=True):
        tolerance.tight(sec.volatility(float(k)), float(cpp_vol))


def test_free_function_matches_smile(svi: dict[str, Any]) -> None:
    sec = SviSmileSection(
        forward=svi["forward"],
        svi_params=(svi["a"], svi["b"], svi["sigma"], svi["rho"], svi["m"]),
        exercise_time=svi["tte"],
    )
    for k in svi["strikes"]:
        fn = svi_volatility(
            float(k), svi["forward"], svi["tte"],
            svi["a"], svi["b"], svi["sigma"], svi["rho"], svi["m"],
        )
        tolerance.tight(fn, sec.volatility(float(k)))


def test_variance_at_atm_m_equals_a_plus_b_sigma(svi: dict[str, Any]) -> None:
    """At strike = forward*exp(m), total variance collapses to a+b*sigma.

    # C++ parity: ``testSviSmileSection`` (svivolatility.cpp:57-58) —
    # ``variance(strike) == a + b*sigma`` to 1e-10.
    """
    sec = SviSmileSection(
        forward=svi["forward"],
        svi_params=(svi["a"], svi["b"], svi["sigma"], svi["rho"], svi["m"]),
        exercise_time=svi["tte"],
    )
    strike = svi["forward"] * math.exp(svi["m"])
    tolerance.tight(sec.variance(strike), float(svi["variance_at_atm_m"]))
    tolerance.tight(sec.variance(strike), svi["a"] + svi["b"] * svi["sigma"])


def test_smile_atm_level(svi: dict[str, Any]) -> None:
    sec = SviSmileSection(
        forward=svi["forward"],
        svi_params=(svi["a"], svi["b"], svi["sigma"], svi["rho"], svi["m"]),
        exercise_time=svi["tte"],
    )
    tolerance.exact(sec.atm_level(), svi["forward"])


def test_smile_date_constructor(svi: dict[str, Any]) -> None:
    """Date-anchored constructor with Actual365Fixed default day counter.

    # C++ parity: ``testSviSmileSection`` date overload — variance at
    # strike = forward*exp(m) still equals a+b*sigma when the day count
    # gives 11/365.
    """
    from pquantlib.time.date import Date  # noqa: PLC0415
    from pquantlib.time.month import Month  # noqa: PLC0415
    from pquantlib.time.period import Period  # noqa: PLC0415
    from pquantlib.time.time_unit import TimeUnit  # noqa: PLC0415

    reference = Date.from_ymd(15, Month.January, 2024)
    option_date = reference + Period(11, TimeUnit.Days)
    sec = SviSmileSection(
        forward=svi["forward"],
        svi_params=(svi["a"], svi["b"], svi["sigma"], svi["rho"], svi["m"]),
        exercise_date=option_date,
        reference_date=reference,
    )
    tolerance.exact(sec.atm_level(), svi["forward"])
    # Actual365Fixed(reference, reference+11d) == 11/365 == tte, so the
    # variance at strike = forward*exp(m) collapses to a+b*sigma.
    strike = svi["forward"] * math.exp(svi["m"])
    tolerance.tight(sec.variance(strike), svi["a"] + svi["b"] * svi["sigma"])


def test_interpolation_recovers_synthetic_params(svi: dict[str, Any]) -> None:
    """SVI fit recovers synthetic (a, b, sigma, rho, m) via multi-start.

    The raw-SVI slice is multi-modal; the Halton multi-start (default 50
    guesses, mirroring C++ ``maxGuesses``) recovers the generating slice
    essentially exactly. We assert recovery in *vol* space at LOOSE and
    the params at LOOSE since the slice is identifiable here.
    """
    a, b, sigma, rho, m = (
        svi["a"], svi["b"], svi["sigma"], svi["rho"], svi["m"]
    )
    f, tte = svi["forward"], svi["tte"]
    strikes = [60.0, 80.0, 100.0, f, 150.0, 180.0, 200.0]
    vols = [svi_volatility(k, f, tte, a, b, sigma, rho, m) for k in strikes]
    fit = SviInterpolation(strikes, vols, tte, f, multi_start_seed=7)
    assert fit.converged()
    assert fit.rms_error() < 1e-6
    tolerance.loose(fit.a(), a)
    tolerance.loose(fit.b(), b)
    tolerance.loose(fit.sigma(), sigma)
    tolerance.loose(fit.rho(), rho)
    tolerance.loose(fit.m(), m)
    # Vols recovered everywhere.
    for k in strikes:
        tolerance.loose(
            fit.value(k), svi_volatility(k, f, tte, a, b, sigma, rho, m)
        )


def test_interpolated_smile_section_recovers_slice(svi: dict[str, Any]) -> None:
    """SviInterpolatedSmileSection fits a synthetic slice + wraps it.

    The fitted smile reproduces the generating vols (LOOSE) and exposes
    the recovered SVI params; min/max strike come from the strike grid.
    """
    from pquantlib.experimental.volatility.svi_interpolated_smile_section import (  # noqa: PLC0415
        SviInterpolatedSmileSection,
    )
    from pquantlib.time.date import Date  # noqa: PLC0415
    from pquantlib.time.month import Month  # noqa: PLC0415
    from pquantlib.time.period import Period  # noqa: PLC0415
    from pquantlib.time.time_unit import TimeUnit  # noqa: PLC0415

    a, b, sigma, rho, m = svi["a"], svi["b"], svi["sigma"], svi["rho"], svi["m"]
    f, tte = svi["forward"], svi["tte"]
    strikes = [60.0, 80.0, 100.0, f, 150.0, 180.0, 200.0]
    vols = [svi_volatility(k, f, tte, a, b, sigma, rho, m) for k in strikes]

    # Choose a reference + option date giving exercise_time == tte (11/365).
    reference = Date.from_ymd(15, Month.January, 2024)
    option_date = reference + Period(11, TimeUnit.Days)
    sec = SviInterpolatedSmileSection(
        option_date=option_date,
        forward=f,
        strikes=strikes,
        vols=vols,
        reference_date=reference,
        multi_start_seed=7,
    )
    assert sec.converged()
    tolerance.exact(sec.min_strike(), strikes[0])
    tolerance.exact(sec.max_strike(), strikes[-1])
    tolerance.exact(sec.atm_level(), f)
    for k in strikes:
        tolerance.loose(sec.volatility(k), svi_volatility(k, f, tte, a, b, sigma, rho, m))


def test_check_svi_parameters_rejects_bad() -> None:
    from pquantlib.exceptions import LibraryException  # noqa: PLC0415
    from pquantlib.experimental.volatility.svi_interpolation import (  # noqa: PLC0415
        check_svi_parameters,
    )

    # b < 0 rejected.
    with pytest.raises(LibraryException):
        check_svi_parameters(0.1, -0.1, 0.3, 0.0, 0.0, 1.0)
    # |rho| >= 1 rejected.
    with pytest.raises(LibraryException):
        check_svi_parameters(0.1, 0.2, 0.3, 1.0, 0.0, 1.0)
    # sigma <= 0 rejected.
    with pytest.raises(LibraryException):
        check_svi_parameters(0.1, 0.2, 0.0, 0.0, 0.0, 1.0)
    # b(1+|rho|) > 4 rejected.
    with pytest.raises(LibraryException):
        check_svi_parameters(0.1, 5.0, 0.3, 0.5, 0.0, 1.0)

"""Cross-validate NoArbSABR (Doust 2012) against the W6-A C++ probe.

Reference: ``migration-harness/references/cluster/w6a.json`` —
``noarbsabr`` + ``d0_checkpoints`` sections. Tests:

  * :class:`D0Interpolator` reproduces the explicit absorption-matrix
    checkpoints from ``testAbsorptionMatrix`` (EXACT on the implied
    integer absorption count; TIGHT on the d0 fraction).
  * :class:`NoArbSabrModel` ``absorptionProbability`` /
    ``optionPrice`` / ``digitalOptionPrice`` / ``density`` match C++
    (TIGHT — same Gauss-Lobatto quadrature + scipy Bessel/gamma).
  * :func:`no_arb_sabr_volatility` matches the C++ implied vol (TIGHT).
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.experimental.volatility.no_arb_sabr import (
    NSIM,
    D0Interpolator,
    NoArbSabrModel,
    no_arb_sabr_volatility,
)
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("cluster/w6a")


@pytest.fixture(scope="module")
def na(cpp: dict[str, Any]) -> dict[str, Any]:
    return cpp["noarbsabr"]


@pytest.fixture(scope="module")
def model(na: dict[str, Any]) -> NoArbSabrModel:
    return NoArbSabrModel(
        na["tau"], na["forward"], na["alpha"], na["beta"], na["nu"], na["rho"]
    )


# --- D0Interpolator absorption matrix -------------------------------------


def test_d0_checkpoints_reproduce_absorption_counts(cpp: dict[str, Any]) -> None:
    """Every ``testAbsorptionMatrix`` checkpoint round-trips exactly.

    # C++ parity: ``checkD0`` (noarbsabr.cpp:32-45) requires
    # ``|d0*nsim - absorptions| < 0.1``. We assert the rounded count is
    # exact and the d0 fraction matches the C++ probe at TIGHT.
    """
    for row in cpp["d0_checkpoints"]:
        d = D0Interpolator(
            row["forward"], row["tau"], row["alpha"],
            row["beta"], row["nu"], row["rho"],
        )
        d0 = d()
        tolerance.tight(d0, float(row["d0"]))
        # Implied integer absorption count is exact.
        assert abs(d0 * NSIM - float(row["absorptions"])) < 0.1


# --- model diagnostics + pricing ------------------------------------------


def test_absorption_probability_matches_cpp(
    model: NoArbSabrModel, na: dict[str, Any]
) -> None:
    tolerance.tight(model.absorption_probability(), float(na["absorption_probability"]))


def test_numerical_forward_matches_cpp(
    model: NoArbSabrModel, na: dict[str, Any]
) -> None:
    tolerance.tight(model.numerical_forward(), float(na["numerical_forward"]))


def test_option_price_matches_cpp(model: NoArbSabrModel, na: dict[str, Any]) -> None:
    for k, cpp_price in zip(na["strikes"], na["option_price"], strict=True):
        tolerance.tight(model.option_price(float(k)), float(cpp_price))


def test_digital_option_price_matches_cpp(
    model: NoArbSabrModel, na: dict[str, Any]
) -> None:
    for k, cpp_dig in zip(
        na["strikes"], na["digital_option_price"], strict=True
    ):
        tolerance.tight(model.digital_option_price(float(k)), float(cpp_dig))


def test_density_matches_cpp(model: NoArbSabrModel, na: dict[str, Any]) -> None:
    for k, cpp_d in zip(na["strikes"], na["density"], strict=True):
        tolerance.tight(model.density(float(k)), float(cpp_d))


# --- implied volatility ----------------------------------------------------


def test_volatility_matches_cpp(na: dict[str, Any]) -> None:
    """No-arb implied vol matches C++ (TIGHT for the bulk, LOOSE in the
    deep wings where the Black inversion accuracy floor dominates)."""
    for k, cpp_vol in zip(na["strikes"], na["noarb_volatility"], strict=True):
        actual = no_arb_sabr_volatility(
            float(k), na["forward"], na["tau"],
            na["alpha"], na["beta"], na["nu"], na["rho"],
        )
        # The C++ vol is itself the result of a 1e-6-accuracy Black
        # implied-vol solve; deep-wing strikes inherit that floor, so we
        # use LOOSE (1e-8) which still pins ~7 significant figures.
        tolerance.loose(actual, float(cpp_vol))


def test_consistency_with_hagan_near_zero_absorption(na: dict[str, Any]) -> None:
    """Doust figure-3 params have ~0 absorption, so no-arb prices track
    the closed-form Hagan SABR prices closely.

    # C++ parity: ``testConsistencyWithHagan`` (noarbsabr.cpp:74-118).
    """
    model = NoArbSabrModel(
        na["tau"], na["forward"], na["alpha"], na["beta"], na["nu"], na["rho"]
    )
    assert 0.0 <= model.absorption_probability() < 1e-10
    for k, cpp_sabr_price in zip(
        na["strikes"], na["sabr_option_price"], strict=True
    ):
        noarb = model.option_price(float(k))
        # C++ asserts |sabrPrice - noarbsabrPrice| < 1e-5 across the
        # strike range; assert the same agreement here.
        assert abs(noarb - float(cpp_sabr_price)) < 1e-5


# --- bounds / error paths --------------------------------------------------


def test_model_rejects_out_of_bounds() -> None:
    from pquantlib.exceptions import LibraryException  # noqa: PLC0415

    # expiryTime > 30 rejected.
    with pytest.raises(LibraryException):
        NoArbSabrModel(31.0, 0.05, 0.026, 0.5, 0.4, -0.1)
    # forward <= 0 rejected.
    with pytest.raises(LibraryException):
        NoArbSabrModel(1.0, -0.05, 0.026, 0.5, 0.4, -0.1)
    # beta out of [0.01, 0.99].
    with pytest.raises(LibraryException):
        NoArbSabrModel(1.0, 0.05, 0.026, 0.999, 0.4, -0.1)
    # nu out of [0.01, 0.80].
    with pytest.raises(LibraryException):
        NoArbSabrModel(1.0, 0.05, 0.026, 0.5, 0.9, -0.1)

"""ZabrModel cross-validated against C++ v1.43.

Reference: migration-harness/references/v143/zabr/model.json
Probe:     migration-harness/cpp/probes/v143_zabr_model/probe.cpp

Four parameter sets: gamma below, at and above 1 on otherwise identical
inputs (so any difference is attributable to the gamma machinery alone),
plus a beta = 1 set that reaches the logarithmic branch of ``y(K)``.

The probe emits the same quantity twice — once from N scalar calls and
once from one vector call — because ``ZabrModel::x`` integrates the
Andreasen-Huge ODE incrementally across the sorted strikes and the two
therefore differ (order 1e-9 relative). Both are asserted separately.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.termstructures.volatility.zabr import ZabrModel
from pquantlib.testing import reference_reader, tolerance

_PARAMS: dict[str, tuple[float, float, float, float, float, float, float]] = {
    "gamma_1_00": (5.0, 0.03, 0.08, 0.70, 0.20, -0.30, 1.00),
    "gamma_0_75": (5.0, 0.03, 0.08, 0.70, 0.20, -0.30, 0.75),
    "gamma_1_30": (5.0, 0.03, 0.08, 0.70, 0.20, -0.30, 1.30),
    "beta_1_gamma_0_85": (2.0, 0.05, 0.20, 1.00, 0.30, 0.20, 0.85),
}

_FD_KEYS = ("gamma_1_00", "gamma_0_75")


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/zabr/model")


def _model(key: str) -> ZabrModel:
    return ZabrModel(*_PARAMS[key])


@pytest.mark.parametrize("key", list(_PARAMS))
def test_inspectors_match_cpp(cpp: dict[str, Any], key: str) -> None:
    """``nu()`` returns the TRANSFORMED nu, not the constructor argument."""
    block = cpp[key]
    m = _model(key)
    tolerance.tight(m.expiry_time(), float(block["expiry_time"]))
    tolerance.tight(m.forward(), float(block["forward"]))
    tolerance.tight(m.alpha(), float(block["alpha"]))
    tolerance.tight(m.beta(), float(block["beta"]))
    tolerance.tight(m.rho(), float(block["rho"]))
    tolerance.tight(m.gamma(), float(block["gamma"]))
    tolerance.tight(m.nu(), float(block["nu_transformed"]))
    # Guard the transform itself: at gamma != 1 the stored nu must NOT be
    # the input nu.
    if block["gamma"] != 1.0:
        assert abs(m.nu() - float(block["nu_input"])) > 1e-6


@pytest.mark.parametrize("key", list(_PARAMS))
def test_scalar_volatilities_match_cpp(cpp: dict[str, Any], key: str) -> None:
    """One strike at a time, from deep ITM through far OTM including ATM."""
    block = cpp[key]
    m = _model(key)
    for i, strike in enumerate(block["strikes"]):
        tolerance.tight(m.lognormal_volatility(strike), float(block["lognormal_vol_scalar"][i]))
        tolerance.tight(m.normal_volatility(strike), float(block["normal_vol_scalar"][i]))
        tolerance.tight(m.local_volatility(strike), float(block["local_vol_scalar"][i]))


@pytest.mark.parametrize("key", list(_PARAMS))
def test_vector_volatilities_match_cpp(cpp: dict[str, Any], key: str) -> None:
    """The whole ladder in one call — the incremental-integration path."""
    block = cpp[key]
    m = _model(key)
    strikes = list(block["strikes"])
    for got, expected in zip(
        m.lognormal_volatility_vector(strikes), block["lognormal_vol_vector"], strict=True
    ):
        tolerance.tight(got, float(expected))
    for got, expected in zip(
        m.normal_volatility_vector(strikes), block["normal_vol_vector"], strict=True
    ):
        tolerance.tight(got, float(expected))
    for got, expected in zip(
        m.local_volatility_vector(strikes), block["local_vol_vector"], strict=True
    ):
        tolerance.tight(got, float(expected))


@pytest.mark.parametrize("key", ["gamma_0_75", "gamma_1_30", "beta_1_gamma_0_85"])
def test_scalar_and_vector_paths_are_distinct(cpp: dict[str, Any], key: str) -> None:
    """At gamma != 1 the two spellings must NOT coincide.

    ``x(vector)`` carries the Runge-Kutta state across strikes while
    ``x(scalar)`` restarts from ``u(0) = 0`` each time. A port that
    implements one and reuses it for the other would pass the two tests
    above only if the reference itself were degenerate; this asserts it
    is not.
    """
    block = cpp[key]
    assert block["lognormal_vol_scalar"] != block["lognormal_vol_vector"]


@pytest.mark.parametrize("key", _FD_KEYS)
def test_fd_price_vector_matches_cpp(cpp: dict[str, Any], key: str) -> None:
    """1-D Dupire PDE prices for the whole strike vector in one solve."""
    block = cpp[key]
    m = _model(key)
    for got, expected in zip(
        m.fd_price_vector(list(block["fd_strikes"])), block["fd_price_vector"], strict=True
    ):
        tolerance.tight(got, float(expected))


@pytest.mark.parametrize("key", _FD_KEYS)
def test_fd_price_scalar_uses_its_own_grid(cpp: dict[str, Any], key: str) -> None:
    """Pricing a strike alone puts it on a different grid than the vector call.

    ``start = min(1e-5, K_first*0.5)`` and ``end = max(0.10, K_last*1.5)``
    are functions of the strike vector, so the 0.12 strike in
    ``fd_strikes`` stretches the vector call's grid to 0.18 while each
    scalar call keeps its own. The reference values differ by up to 6e-4
    relative at 0.06, well outside any tolerance tier.
    """
    block = cpp[key]
    m = _model(key)
    for i, strike in enumerate(block["fd_strikes"]):
        tolerance.tight(m.fd_price(strike), float(block["fd_price_scalar"][i]))
    assert block["fd_price_scalar"] != block["fd_price_vector"]


@pytest.mark.slow
@pytest.mark.parametrize("key", _FD_KEYS)
def test_full_fd_price_matches_cpp(cpp: dict[str, Any], key: str) -> None:
    """2-D Hundsdorfer PDE, priced below / at / above the forward.

    LOOSE tier rather than TIGHT: the five implicit-Euler damping steps
    that precede the Hundsdorfer sweep solve the full 2-D operator with
    BiCGstab at ``rel_tol = 1e-8`` (the C++ ``ImplicitEulerScheme``
    default), so the two runs' rolled-back grids can only agree to that
    order. Measured agreement is ~6e-10 relative, i.e. two orders inside
    the tier.
    """
    block = cpp[key]
    m = _model(key)
    for i, strike in enumerate(block["full_fd_strikes"]):
        tolerance.loose(m.full_fd_price(strike), float(block["full_fd_price"][i]))

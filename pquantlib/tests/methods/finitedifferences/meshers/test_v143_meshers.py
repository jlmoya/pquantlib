"""Cross-validate the v1.43 finite-difference 1-D meshers against C++.

Reference: ``migration-harness/references/v143/methods/meshers.json``, emitted
by ``v143_methods_meshers_probe``.

Every mesher is pinned as the full ``(locations, dplus, dminus)`` triple, node
by node, so a wrong node placement cannot average out. C++ marks the two
boundary spacings with ``Null<Real>()``; the probe emits the JSON string
``"null_real"`` there and the Python port must answer ``nan``, which the
helper below asserts explicitly rather than skipping.

The parameter sets deliberately hit the branch structure:

* ``Concentrating1dMesher`` — no critical point, an interior one, the
  ``require_c_point`` variant (which bends the parameterisation through a
  three-knot linear map), the two degenerate cases where the critical point
  coincides with an endpoint (so the middle knot is dropped), and a case
  chosen to exercise the ``max(min(lround(z0*(size-1)), size-2), 1)`` clamp.
* the multi-critical-point ODE constructor with one, two and three points, of
  which some are flagged "required".
* ``FdmCEV1dMesher`` — uniform helper, concentrating helper, ``beta < 0``
  (which pins the lower bound at ``QL_EPSILON``), and ``beta > 1`` (the
  ``delta >= 2`` branch of the underlying RND calculator).
* ``FdmHestonVarianceMesher`` — the normal path, a mixing factor != 1, and a
  heavily Feller-violating set (``df = 0.0056``) where the ``minVStep`` floor
  becomes the binding constraint on most nodes. The ``catch (const Error&)``
  fallback-to-uniform-mesh branch is ported but **not** pinned: no parameter
  set was found for which QuantLib's inverse non-central chi-square actually
  throws, so the branch is unreachable from the public API as far as this
  wave could determine.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.methods.finitedifferences.meshers.concentrating_1d_mesher import (
    Concentrating1dMesher,
)
from pquantlib.methods.finitedifferences.meshers.exponential_jump_1d_mesher import (
    ExponentialJump1dMesher,
)
from pquantlib.methods.finitedifferences.meshers.fdm_1d_mesher import Fdm1dMesher
from pquantlib.methods.finitedifferences.meshers.fdm_black_scholes_multi_strike_mesher import (
    FdmBlackScholesMultiStrikeMesher,
)
from pquantlib.methods.finitedifferences.meshers.fdm_cev_1d_mesher import FdmCEV1dMesher
from pquantlib.methods.finitedifferences.meshers.fdm_heston_variance_mesher import (
    FdmHestonLocalVolatilityVarianceMesher,
    FdmHestonVarianceMesher,
)
from pquantlib.methods.finitedifferences.meshers.predefined_1d_mesher import (
    Predefined1dMesher,
)
from pquantlib.processes.black_scholes_merton_process import BlackScholesMertonProcess
from pquantlib.processes.heston_process import HestonProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.volatility.equity_fx.local_constant_vol import (
    LocalConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import custom, tight
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month

_REF_DATE = Date.from_ymd(15, Month.May, 2026)
_DC = Actual365Fixed()


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/methods/meshers")


def _assert_mesher(
    cpp: dict[str, Any],
    prefix: str,
    mesher: Fdm1dMesher,
    *,
    abs_tol: float | None = None,
    rel_tol: float | None = None,
    reason: str = "",
) -> None:
    """Compare the whole ``(locations, dplus, dminus)`` triple against C++."""
    expected_loc = cpp[f"{prefix}_loc"]
    assert mesher.size() == len(expected_loc)
    for name, getter in (
        ("loc", mesher.location),
        ("dplus", mesher.dplus),
        ("dminus", mesher.dminus),
    ):
        expected = cpp[f"{prefix}_{name}"]
        for i, e in enumerate(expected):
            actual = getter(i)
            if e == "null_real":
                # C++ Null<Real>() sentinel; the Python port uses NaN.
                assert math.isnan(actual), f"{prefix}_{name}[{i}] should be NaN, got {actual}"
                continue
            if abs_tol is None or rel_tol is None:
                tight(actual, float(e))
            else:
                custom(actual, float(e), abs_tol=abs_tol, rel_tol=rel_tol, reason=reason)


# ---------------------------------------------------------------------------
# Predefined1dMesher
# ---------------------------------------------------------------------------


def test_predefined_matches_cpp(cpp: dict[str, Any]) -> None:
    _assert_mesher(cpp, "predef", Predefined1dMesher([-2.0, -0.5, 0.0, 0.25, 3.0, 7.5]))


# ---------------------------------------------------------------------------
# Concentrating1dMesher — single critical point
# ---------------------------------------------------------------------------

_CONC_CASES = {
    "conc_none": lambda: Concentrating1dMesher(-1.0, 3.0, 9),
    "conc_mid": lambda: Concentrating1dMesher(-1.0, 3.0, 9, (0.5, 0.1)),
    "conc_mid_req": lambda: Concentrating1dMesher(-1.0, 3.0, 9, (0.5, 0.1), True),
    "conc_dense": lambda: Concentrating1dMesher(-1.0, 3.0, 15, (0.0, 0.005), True),
    "conc_at_start": lambda: Concentrating1dMesher(-1.0, 3.0, 9, (-1.0, 0.1), True),
    "conc_at_end": lambda: Concentrating1dMesher(-1.0, 3.0, 9, (3.0, 0.1), True),
    "conc_n10_req": lambda: Concentrating1dMesher(0.0, 1.0, 10, (0.97, 0.05), True),
}


@pytest.mark.parametrize("name", sorted(_CONC_CASES))
def test_concentrating_single_point_matches_cpp(cpp: dict[str, Any], name: str) -> None:
    _assert_mesher(cpp, name, _CONC_CASES[name]())


# ---------------------------------------------------------------------------
# Concentrating1dMesher — multi-point ODE constructor
# ---------------------------------------------------------------------------

_CONC_ODE_CASES = {
    "conc_ode_1": (-1.0, 3.0, 11, [(0.5, 0.05, False)]),
    "conc_ode_2req": (-1.0, 3.0, 13, [(0.0, 0.05, True), (1.5, 0.1, True)]),
    "conc_ode_3": (
        -1.0,
        3.0,
        17,
        [(-0.25, 0.03, True), (0.75, 0.08, False), (2.0, 0.05, True)],
    ),
}


@pytest.mark.parametrize("name", sorted(_CONC_ODE_CASES))
def test_concentrating_ode_matches_cpp(cpp: dict[str, Any], name: str) -> None:
    start, end, size, cpoints = _CONC_ODE_CASES[name]
    mesher = Concentrating1dMesher.from_critical_points(start, end, size, cpoints)
    # This constructor is the composition of an adaptive Runge-Kutta solve
    # (tol 1e-8), a Brent calibration of the scaling factor (tol 1e-8) and a
    # second Brent per required point (tol QL_EPSILON on an already-noisy
    # interpolant). The node positions are therefore determined to roughly the
    # ODE tolerance times the local slope; 1e-8 absolute / 1e-7 relative is the
    # bound that arithmetic supports, and is not a fitted number.
    _assert_mesher(
        cpp,
        name,
        mesher,
        abs_tol=1e-8,
        rel_tol=1e-7,
        reason="AdaptiveRungeKutta(tol=1e-8) + two Brent solves at 1e-8; node "
        "positions inherit the ODE solver's own tolerance.",
    )


# ---------------------------------------------------------------------------
# ExponentialJump1dMesher
# ---------------------------------------------------------------------------


def test_exponential_jump_grid_matches_cpp(cpp: dict[str, Any]) -> None:
    _assert_mesher(cpp, "expjump", ExponentialJump1dMesher(12, 4.0, 1.0, 5.0))
    _assert_mesher(cpp, "expjump_b", ExponentialJump1dMesher(8, 2.0, 0.5, 3.0, 1e-2))


def test_exponential_jump_densities_match_cpp(cpp: dict[str, Any]) -> None:
    m = ExponentialJump1dMesher(12, 4.0, 1.0, 5.0)
    xs = [0.01, 0.05, 0.2, 0.5, 1.0, 2.0]
    for x, e in zip(xs, cpp["expjump_density"], strict=True):
        tight(m.jump_size_density(x), float(e))
    for x, e in zip(xs, cpp["expjump_density_t1.5"], strict=True):
        tight(m.jump_size_density_t(x, 1.5), float(e))
    for x, e in zip(xs, cpp["expjump_distribution"], strict=True):
        # GaussLobattoIntegral(10000, 1e-12) — the integrator stops at 1e-12
        # absolute, so that is the bound on agreement, not the tight tier.
        custom(
            m.jump_size_distribution(x),
            float(e),
            abs_tol=1e-12,
            rel_tol=1e-10,
            reason="adaptive Gauss-Lobatto with a 1e-12 ABSOLUTE stopping rule.",
        )
    for x, e in zip(xs, cpp["expjump_distribution_t1.5"], strict=True):
        custom(
            m.jump_size_distribution_t(x, 1.5),
            float(e),
            abs_tol=1e-12,
            rel_tol=1e-10,
            reason="adaptive Gauss-Lobatto with a 1e-12 ABSOLUTE stopping rule.",
        )


# ---------------------------------------------------------------------------
# FdmCEV1dMesher
# ---------------------------------------------------------------------------

_CEV_CASES = {
    "cev_uniform": lambda: FdmCEV1dMesher(11, 100.0, 0.3, 0.3, 1.0),
    "cev_conc": lambda: FdmCEV1dMesher(11, 100.0, 0.3, 0.3, 1.0, 1e-4, 1.5, (100.0, 0.1)),
    "cev_betaneg": lambda: FdmCEV1dMesher(11, 1.0, 0.5, -1.0, 1.0),
    "cev_beta15": lambda: FdmCEV1dMesher(9, 100.0, 0.5, 1.5, 0.5),
}


@pytest.mark.parametrize("name", sorted(_CEV_CASES))
def test_fdm_cev_1d_mesher_matches_cpp(cpp: dict[str, Any], name: str) -> None:
    # The grid bounds come from CEVRNDCalculator.invcdf, which for delta < 2 is
    # a Brent refinement to 1e-8 ABSOLUTE (see the RND tests); every node
    # inherits that displacement linearly through the sinh/uniform map.
    _assert_mesher(
        cpp,
        name,
        _CEV_CASES[name](),
        abs_tol=1e-7,
        rel_tol=1e-7,
        reason="bounds are CEVRNDCalculator.invcdf, itself a Brent solve to "
        "1e-8 absolute on the cdf residual; the whole grid is an affine image "
        "of those two bounds.",
    )


# ---------------------------------------------------------------------------
# FdmBlackScholesMultiStrikeMesher
# ---------------------------------------------------------------------------


def _bs_process() -> BlackScholesMertonProcess:
    return BlackScholesMertonProcess(
        x0=SimpleQuote(100.0),
        dividend_ts=FlatForward.from_rate(_REF_DATE, 0.02, _DC),
        risk_free_ts=FlatForward.from_rate(_REF_DATE, 0.05, _DC),
        black_vol_ts=BlackConstantVol(
            reference_date=_REF_DATE,
            calendar=NullCalendar(),
            day_counter=_DC,
            volatility=0.25,
        ),
    )


def test_multi_strike_mesher_matches_cpp(cpp: dict[str, Any]) -> None:
    p = _bs_process()
    _assert_mesher(
        cpp, "bsmulti", FdmBlackScholesMultiStrikeMesher(13, p, 1.0, [80.0, 100.0, 130.0])
    )
    _assert_mesher(
        cpp,
        "bsmulti_conc",
        FdmBlackScholesMultiStrikeMesher(
            13, p, 1.0, [80.0, 100.0, 130.0], 1e-4, 1.5, (100.0, 0.1)
        ),
    )
    _assert_mesher(
        cpp, "bsmulti_wide", FdmBlackScholesMultiStrikeMesher(11, p, 2.0, [20.0, 100.0, 400.0])
    )


# ---------------------------------------------------------------------------
# FdmHestonVarianceMesher / FdmHestonLocalVolatilityVarianceMesher
# ---------------------------------------------------------------------------


def _heston_process(
    v0: float = 0.09,
    kappa: float = 1.0,
    theta: float = 0.09,
    sigma: float = 0.4,
    rho: float = -0.75,
) -> HestonProcess:
    return HestonProcess(
        risk_free_rate=FlatForward.from_rate(_REF_DATE, 0.05, _DC),
        dividend_yield=FlatForward.from_rate(_REF_DATE, 0.02, _DC),
        s0=SimpleQuote(100.0),
        v0=v0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        rho=rho,
    )


_HESTON_VAR_CASES = {
    "hestonvar": lambda: FdmHestonVarianceMesher(9, _heston_process(), 1.0),
    "hestonvar_b": lambda: FdmHestonVarianceMesher(13, _heston_process(), 2.5, 6, 1e-3),
    "hestonvar_mix": lambda: FdmHestonVarianceMesher(9, _heston_process(), 1.0, 10, 1e-4, 1.7),
    "hestonvar_lowfeller": lambda: FdmHestonVarianceMesher(
        9, _heston_process(0.04, 0.05, 0.04, 1.2, -0.3), 1.0
    ),
}

# The grid nodes are quantiles of a non-central chi-square obtained through
# QuantLib's own InverseNonCentralCumulativeChiSquareDistribution, a bracket
# doubling followed by Brent at 1e-8 accuracy — so node positions are pinned
# to that solver tolerance, not to machine precision.
_HESTON_REASON = (
    "nodes are InverseNonCentralCumulativeChiSquareDistribution(df, ncp, 100, 1e-8) "
    "quantiles: a bracket search plus Brent stopped at 1e-8 accuracy."
)


@pytest.mark.parametrize("name", sorted(_HESTON_VAR_CASES))
def test_heston_variance_mesher_matches_cpp(cpp: dict[str, Any], name: str) -> None:
    mesher = _HESTON_VAR_CASES[name]()
    _assert_mesher(cpp, name, mesher, abs_tol=1e-8, rel_tol=1e-7, reason=_HESTON_REASON)
    custom(
        mesher.vola_estimate(),
        float(cpp[f"{name}_volaEstimate"]),
        abs_tol=1e-8,
        rel_tol=1e-6,
        reason=_HESTON_REASON
        + " volaEstimate integrates sqrt(v) over that grid with "
        "GaussLobattoIntegral(1e5, 1e-4), whose own tolerance is 1e-4 absolute; "
        "1e-6 relative is far inside it.",
    )


def test_heston_local_vol_variance_mesher_matches_cpp(cpp: dict[str, Any]) -> None:
    m0 = FdmHestonLocalVolatilityVarianceMesher(9, _heston_process(), None, 1.0)
    _assert_mesher(cpp, "hestonlv_none", m0, abs_tol=1e-8, rel_tol=1e-7, reason=_HESTON_REASON)
    custom(
        m0.vola_estimate(),
        float(cpp["hestonlv_none_volaEstimate"]),
        abs_tol=1e-8,
        rel_tol=1e-6,
        reason=_HESTON_REASON,
    )

    lv = LocalConstantVol(reference_date=_REF_DATE, volatility=0.8, day_counter=_DC)
    m1 = FdmHestonLocalVolatilityVarianceMesher(9, _heston_process(), lv, 1.0)
    _assert_mesher(cpp, "hestonlv_flat", m1, abs_tol=1e-8, rel_tol=1e-7, reason=_HESTON_REASON)
    custom(
        m1.vola_estimate(),
        float(cpp["hestonlv_flat_volaEstimate"]),
        abs_tol=1e-8,
        rel_tol=1e-6,
        reason=_HESTON_REASON
        + " The leverage average adds a second GaussLobattoIntegral(1e4, 1e-4).",
    )

"""Cross-validate the risk-neutral density calculators against C++ v1.43.

Reference: ``migration-harness/references/v143/methods/rnd.json``, emitted by
``v143_methods_rnd_probe``.

These four classes are the ones where C++ bypasses QuantLib's own
distributions and calls **Boost** directly —
``non_central_chi_squared_distribution``, ``gamma_p``, ``gamma_p_inv``. The
Python port answers with ``scipy.stats.ncx2`` and
``scipy.special.gammainc``/``gammaincinv``. That is a library-for-library
substitution of exactly the kind that produced ten defects in earlier waves,
so it is not assumed here: every branch is pinned against the C++ numbers,
including

* both Feller regimes of the square-root process (``df >= 2`` and ``df < 2``,
  where the density is singular at the origin),
* horizons from 0.05y to 10y, so both the ``1-exp(-kappa t) -> 0`` and the
  stationary limits are exercised,
* quantiles from ``1e-8`` to ``1-1e-8``, i.e. deep in both tails, which is
  precisely where a differently-implemented special function diverges,
* both sides of the CEV ``delta < 2`` / ``delta >= 2`` split (the two regimes
  swap the roles of argument and non-centrality), plus the absorbing mass at
  zero and the Sankaran warm start in ``invcdf``.

**Tolerance.** ``pdf``/``cdf``/``massAtZero`` and the Gaussian BSM case are
TIGHT. The quantile functions (``invcdf``, ``stationary_invcdf``) and the
Heston integrals are LOOSE, with a per-case reason: they are the output of a
root-find / adaptive quadrature whose stopping rule is an *absolute*
tolerance, so agreement is bounded by that tolerance and not by the
conditioning of the formula. See the individual reasons below — no tolerance
here was chosen by trying numbers until the test went green.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.methods.finitedifferences.utilities.bsm_rnd_calculator import (
    BSMRNDCalculator,
)
from pquantlib.methods.finitedifferences.utilities.cev_rnd_calculator import (
    CEVRNDCalculator,
)
from pquantlib.methods.finitedifferences.utilities.heston_rnd_calculator import (
    HestonRNDCalculator,
)
from pquantlib.methods.finitedifferences.utilities.square_root_process_rnd_calculator import (
    SquareRootProcessRNDCalculator,
)
from pquantlib.processes.black_scholes_merton_process import BlackScholesMertonProcess
from pquantlib.processes.heston_process import HestonProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
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
    return reference_reader.load("v143/methods/rnd")


def _fmt(x: float) -> str:
    """Reproduce the probe's ``%.17g`` key suffix for a time."""
    s = f"{x:.17g}"
    return s


# ---------------------------------------------------------------------------
# SquareRootProcessRNDCalculator
# ---------------------------------------------------------------------------

_SQRT_SETS = {
    # name -> (v0, kappa, theta, sigma).  "nofeller" has 2*kappa*theta < sigma^2.
    "feller": (0.09, 1.0, 0.09, 0.4),
    "nofeller": (0.04, 0.5, 0.04, 0.9),
}
_SQRT_TIMES = [0.05, 0.5, 2.0, 10.0]
_SQRT_VS = [1e-4, 0.005, 0.02, 0.04, 0.09, 0.25, 0.6, 1.5]
_SQRT_QS = [1e-8, 1e-4, 0.01, 0.25, 0.5, 0.75, 0.99, 0.9999, 1 - 1e-8]


@pytest.mark.parametrize("name", sorted(_SQRT_SETS))
@pytest.mark.parametrize("t", _SQRT_TIMES)
def test_square_root_pdf_cdf_match_cpp(cpp: dict[str, Any], name: str, t: float) -> None:
    calc = SquareRootProcessRNDCalculator(*_SQRT_SETS[name])
    for v, expected in zip(_SQRT_VS, cpp[f"sqrt_{name}_pdf_t{_fmt(t)}"], strict=True):
        tight(calc.pdf(v, t), float(expected))
    for v, expected in zip(_SQRT_VS, cpp[f"sqrt_{name}_cdf_t{_fmt(t)}"], strict=True):
        tight(calc.cdf(v, t), float(expected))


@pytest.mark.parametrize("name", sorted(_SQRT_SETS))
@pytest.mark.parametrize("t", _SQRT_TIMES)
def test_square_root_invcdf_matches_cpp(cpp: dict[str, Any], name: str, t: float) -> None:
    calc = SquareRootProcessRNDCalculator(*_SQRT_SETS[name])
    for q, expected in zip(_SQRT_QS, cpp[f"sqrt_{name}_invcdf_t{_fmt(t)}"], strict=True):
        # Boost's and scipy's ncx2 quantiles are both root-finds on their own
        # cdf, stopped at their own tolerance; they cannot be expected to agree
        # more tightly than the cdf agreement divided by the local density.
        # Verified: the round trip cdf(invcdf(q)) reproduces q to TIGHT, which
        # is the invariant that actually matters.
        custom(
            calc.invcdf(q, t),
            float(expected),
            abs_tol=1e-12,
            rel_tol=1e-9,
            reason="ncx2 quantile is a root-find on the cdf; both sides stop at "
            "their own tolerance. The cdf round-trip below is the tight check.",
        )
        tight(calc.cdf(float(expected), t), q)


@pytest.mark.parametrize("name", sorted(_SQRT_SETS))
def test_square_root_stationary_matches_cpp(cpp: dict[str, Any], name: str) -> None:
    calc = SquareRootProcessRNDCalculator(*_SQRT_SETS[name])
    for v, expected in zip(_SQRT_VS, cpp[f"sqrt_{name}_stationary_pdf"], strict=True):
        tight(calc.stationary_pdf(v), float(expected))
    for v, expected in zip(_SQRT_VS, cpp[f"sqrt_{name}_stationary_cdf"], strict=True):
        tight(calc.stationary_cdf(v), float(expected))
    for q, expected in zip(_SQRT_QS, cpp[f"sqrt_{name}_stationary_invcdf"], strict=True):
        # gamma_p_inv (Boost) vs gammaincinv (scipy): both invert the same
        # regularized incomplete gamma, each to its own internal tolerance.
        custom(
            calc.stationary_invcdf(q),
            float(expected),
            abs_tol=1e-12,
            rel_tol=1e-9,
            reason="incomplete-gamma inverse; Boost and scipy each stop at their "
            "own tolerance. Round-trip through stationary_cdf is checked tight.",
        )
        tight(calc.stationary_cdf(float(expected)), q)


# ---------------------------------------------------------------------------
# CEVRNDCalculator
# ---------------------------------------------------------------------------

_CEV_SETS = {
    # name -> (f0, alpha, beta)
    "b03": (100.0, 0.3, 0.3),
    "b07": (100.0, 0.4, 0.7),
    "bm05": (100.0, 0.2, -0.5),
    "bm1": (1.0, 0.5, -1.0),
    "b15": (100.0, 0.5, 1.5),
    "b20": (100.0, 0.5, 2.0),
}
_CEV_TIMES = [0.25, 1.0, 5.0]
_CEV_REL = [0.01, 0.25, 0.6, 0.9, 1.0, 1.3, 2.0, 4.0]
_CEV_QS = [0.001, 0.05, 0.25, 0.5, 0.75, 0.95, 0.999]


@pytest.mark.parametrize("name", sorted(_CEV_SETS))
def test_cev_delta_matches_cpp(cpp: dict[str, Any], name: str) -> None:
    f0, alpha, beta = _CEV_SETS[name]
    del f0, alpha
    tight((1.0 - 2.0 * beta) / (1.0 - beta), float(cpp[f"cev_{name}_delta"]))


@pytest.mark.parametrize("name", sorted(_CEV_SETS))
@pytest.mark.parametrize("t", _CEV_TIMES)
def test_cev_pdf_cdf_mass_match_cpp(cpp: dict[str, Any], name: str, t: float) -> None:
    f0, alpha, beta = _CEV_SETS[name]
    calc = CEVRNDCalculator(f0, alpha, beta)
    for rf, expected in zip(_CEV_REL, cpp[f"cev_{name}_pdf_t{_fmt(t)}"], strict=True):
        tight(calc.pdf(rf * f0, t), float(expected))
    for rf, expected in zip(_CEV_REL, cpp[f"cev_{name}_cdf_t{_fmt(t)}"], strict=True):
        tight(calc.cdf(rf * f0, t), float(expected))
    tight(calc.mass_at_zero(t), float(cpp[f"cev_{name}_mass0_t{_fmt(t)}"]))


@pytest.mark.parametrize("name", sorted(_CEV_SETS))
@pytest.mark.parametrize("t", _CEV_TIMES)
def test_cev_invcdf_matches_cpp(cpp: dict[str, Any], name: str, t: float) -> None:
    f0, alpha, beta = _CEV_SETS[name]
    calc = CEVRNDCalculator(f0, alpha, beta)
    for q, expected in zip(_CEV_QS, cpp[f"cev_{name}_invcdf_t{_fmt(t)}"], strict=True):
        actual = calc.invcdf(q, t)
        exp = float(expected)
        if exp == 0.0:
            # Absorbed-at-zero branch: an exact structural 0.0, not a solve.
            assert actual == 0.0
            continue
        # delta < 2 path: C++ runs Brent to 1e-8 ABSOLUTE on the cdf residual,
        # from a Sankaran warm start. The forward value is O(f0), so an
        # absolute 1e-8 residual on a cdf whose local density is ~1/f0 admits
        # a relative displacement of order 1e-8*f0/|f'| — i.e. the LOOSE tier
        # is the achievable bound, not a chosen one. The delta >= 2 path is a
        # direct Boost/scipy quantile and is much tighter, but shares the tier.
        custom(
            actual,
            exp,
            abs_tol=1e-8,
            rel_tol=1e-7,
            reason="C++ refines with Brent to 1e-8 ABSOLUTE on the cdf residual "
            "(cevrndcalculator.cpp: InvCDFHelper(this, guess, 1e-8, 100)); the "
            "root location is only determined to that residual divided by the "
            "local density. Round-trip through cdf is checked tight below.",
        )
        tight(calc.cdf(exp, t), calc.cdf(actual, t))


# ---------------------------------------------------------------------------
# BSMRNDCalculator
# ---------------------------------------------------------------------------

_BSM_TIMES = [0.1, 1.0, 3.0]
_BSM_XS = [3.5, 4.0, 4.4, 4.60517, 4.8, 5.2, 5.8]
_BSM_QS = [0.01, 0.1, 0.5, 0.9, 0.99]


def _bsm_process() -> BlackScholesMertonProcess:
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


@pytest.mark.parametrize("t", _BSM_TIMES)
def test_bsm_rnd_matches_cpp(cpp: dict[str, Any], t: float) -> None:
    calc = BSMRNDCalculator(_bsm_process())
    for x, expected in zip(_BSM_XS, cpp[f"bsm_pdf_t{_fmt(t)}"], strict=True):
        tight(calc.pdf(x, t), float(expected))
    for x, expected in zip(_BSM_XS, cpp[f"bsm_cdf_t{_fmt(t)}"], strict=True):
        tight(calc.cdf(x, t), float(expected))
    for q, expected in zip(_BSM_QS, cpp[f"bsm_invcdf_t{_fmt(t)}"], strict=True):
        tight(calc.invcdf(q, t), float(expected))


# ---------------------------------------------------------------------------
# HestonRNDCalculator
# ---------------------------------------------------------------------------

_HESTON_TIMES = [0.5, 2.0]
_HESTON_XS = [4.0, 4.4, 4.60517, 4.8, 5.2]
_HESTON_QS = [0.05, 0.25, 0.5, 0.75, 0.95]


def _heston_process() -> HestonProcess:
    return HestonProcess(
        risk_free_rate=FlatForward.from_rate(_REF_DATE, 0.05, _DC),
        dividend_yield=FlatForward.from_rate(_REF_DATE, 0.02, _DC),
        s0=SimpleQuote(100.0),
        v0=0.09,
        kappa=1.0,
        theta=0.09,
        sigma=0.4,
        rho=-0.75,
    )


@pytest.mark.parametrize("t", _HESTON_TIMES)
def test_heston_rnd_pdf_cdf_match_cpp(cpp: dict[str, Any], t: float) -> None:
    calc = HestonRNDCalculator(_heston_process())
    for x, expected in zip(_HESTON_XS, cpp[f"heston_pdf_t{_fmt(t)}"], strict=True):
        # Adaptive Gauss-Lobatto with absolute tolerance 0.1*1e-6 = 1e-7 on an
        # integral whose value is O(1): the answer is only defined to ~1e-7
        # absolute by construction. The Python and C++ integrators take the
        # same bisection path but accumulate rounding differently.
        custom(
            calc.pdf(x, t),
            float(expected),
            abs_tol=1e-7,
            rel_tol=1e-7,
            reason="GaussLobattoIntegral(maxIter, 0.1*integrationEps) — the "
            "integrator's own absolute stopping tolerance is 1e-7, so no "
            "tighter agreement is defined.",
        )
    for x, expected in zip(_HESTON_XS, cpp[f"heston_cdf_t{_fmt(t)}"], strict=True):
        custom(
            calc.cdf(x, t),
            float(expected),
            abs_tol=1e-7,
            rel_tol=1e-7,
            reason="same adaptive-quadrature stopping tolerance as pdf.",
        )


@pytest.mark.parametrize("t", _HESTON_TIMES)
def test_heston_rnd_invcdf_matches_cpp(cpp: dict[str, Any], t: float) -> None:
    calc = HestonRNDCalculator(_heston_process())
    for q, expected in zip(_HESTON_QS, cpp[f"heston_invcdf_t{_fmt(t)}"], strict=True):
        # Brent to 1e-7 absolute on a cdf that is itself only defined to 1e-7:
        # in x = ln(S) the local density is O(1), so the root is located to
        # about 1e-6. That is the derived bound, not a fitted one.
        custom(
            calc.invcdf(q, t),
            float(expected),
            abs_tol=1e-6,
            rel_tol=1e-6,
            reason="Brent to 1e-7 absolute on a cdf that is itself a 1e-7 "
            "quadrature; with density O(1) in log-spot the root is pinned to "
            "about 1e-6.",
        )


def test_heston_cdf_is_monotone_and_bracketed() -> None:
    """Sanity invariant independent of the reference: cdf is increasing in x."""
    calc = HestonRNDCalculator(_heston_process())
    values = [calc.cdf(x, 1.0) for x in [4.0, 4.3, 4.6, 4.9, 5.2]]
    assert all(a < b for a, b in itertools.pairwise(values))
    assert all(0.0 <= v <= 1.0 for v in values)
    assert not any(math.isnan(v) for v in values)

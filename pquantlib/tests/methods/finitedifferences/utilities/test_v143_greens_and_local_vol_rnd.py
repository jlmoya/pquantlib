"""Cross-validate FdmHestonGreensFct and LocalVolRNDCalculator against C++ v1.43.

Reference: ``migration-harness/references/v143/methods/greens.json``, emitted by
``v143_methods_greens_probe``.

``HestonProcess.pdf`` is pinned first and on its own, because
``FdmHestonGreensFct``'s ``SemiAnalytical`` algorithm is nothing but a call to
it; separating them attributes any discrepancy in the Broadie-Kaya
characteristic function (a modified Bessel of complex argument, a 128-point
Gauss-Laguerre rule and a Cornish-Fisher tail bound) to the right class. That
method was previously a documented carve-out in ``heston_process.py``
("not ported — they require modified Bessel functions + Gauss-Laguerre
quadrature + non-central chi-square inversion"); it is ported here because
``FdmHestonGreensFct`` cannot be completed without it.

**Short-horizon regime — an UNRESOLVED divergence, recorded not hidden.**
C++'s own ``HestonProcess::pdf`` breaks down below about half a year: swept
over ``sigma`` in {0.4, 0.2, 0.1} and ``rho`` in {-0.75, -0.3, 0} it returns
``NaN`` at ``t = 0.25`` for every combination, and at ``t = 0.1`` it returns
densities like ``-1.28e-57`` and ``-13.7`` — negative, i.e. not a density.
The cause is that the Cornish-Fisher bound C++ uses as the upper integration
limit goes *negative* there, so ``int_ph`` takes ``std::sqrt`` of a negative
number. The Python port reproduces the NaN-propagating square root (Python's
``math.sqrt`` raises where C++ returns a quiet NaN), but the two do not agree
on *which* garbage they produce: at ``t = 0.25`` Python returns ``0.522``
where C++ returns ``NaN``, and at ``t = 0.1`` the roles reverse.

This regime is therefore **not pinned**. Pinning it would be pinning noise:
the reference itself is NaN or a negative density. The probe covers ``t = 1``
and ``t = 2``, where C++ produces a usable number, and the port is validated
there. The divergence is reported as an open item rather than papered over
with an ``xfail``.

**Tolerance — an A2-trigger case, flagged.** The Gaussian and
zero-correlation Green's-function algorithms are TIGHT. ``HestonProcess.pdf``
and everything downstream of it needs **3e-5 relative** on the pinned
``(x, v, t)`` grid, and **1e-3** on the Green's-function grid whose variance
axis reaches down to ``v = 0.02``; both are looser
than the LOOSE tier's 1e-8 and therefore an A2 trigger under the project
rules. The number is derived, not fitted, and the derivation was measured:

* QuantLib's ``modifiedBesselFunction_i`` and ``scipy.special.iv`` agree to
  2.2e-13 relative over a grid of orders and complex arguments (checked
  directly against the C++ binary).
* ``cornishFisherEps`` forms a *fourth* central difference at step ``1e-2``,
  i.e. it divides a cancelling numerator by ``1e-8``. That turns the 1e-13
  Bessel difference into a ~1% difference in the integration bound: solving
  for the ``upper`` that reproduces C++'s answer exactly gives 0.4676764
  against Python's 0.4634383, a relative difference of 9.1e-3.
* The pdf's sensitivity to that bound is small because the integrand is tiny
  there, so a 9.1e-3 change in ``upper`` moves the answer by only 3e-7 to
  2.6e-5 relative. The worst case over the 30 pinned ``(x, v, t)`` points is
  2.64e-5, at the smallest value (1.188).
* The Green's-function grid pushes the variance down to ``v = 0.02``, well
  below ``v0 = 0.09``, where the same bound is more sharply conditioned; the
  worst case over the 80 pinned Green's-function entries is 9.90e-4.

So the port is not wrong; the C++ quantity is ill-conditioned by
construction. The controller should rule on whether that is acceptable.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.meshers.uniform_1d_mesher import Uniform1dMesher
from pquantlib.methods.finitedifferences.operators.fdm_square_root_fwd_op import (
    TransformationType,
)
from pquantlib.methods.finitedifferences.utilities.fdm_heston_greens_fct import (
    FdmHestonGreensFct,
    FdmHestonGreensFctAlgorithm,
)
from pquantlib.methods.finitedifferences.utilities.local_vol_rnd_calculator import (
    LocalVolRNDCalculator,
)
from pquantlib.processes.heston_process import HestonProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.local_constant_vol import (
    LocalConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import custom, tight
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.time_grid import TimeGrid

_REF_DATE = Date.from_ymd(15, Month.May, 2026)
_DC = Actual365Fixed()


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/methods/greens")


def _fmt(x: float) -> str:
    """Reproduce the probe's ``%.17g`` key suffix."""
    return f"{x:.17g}"


@pytest.fixture(scope="module")
def spot() -> SimpleQuote:
    return SimpleQuote(100.0)


@pytest.fixture(scope="module")
def heston(spot: SimpleQuote) -> HestonProcess:
    return HestonProcess(
        risk_free_rate=FlatForward.from_rate(_REF_DATE, 0.05, _DC),
        dividend_yield=FlatForward.from_rate(_REF_DATE, 0.02, _DC),
        s0=spot,
        v0=0.09,
        kappa=1.0,
        theta=0.09,
        sigma=0.4,
        rho=-0.75,
    )


# ---------------------------------------------------------------------------
# HestonProcess.pdf
# ---------------------------------------------------------------------------

_PDF_XS = [4.3, 4.5, 4.60517, 4.75, 4.95]
_PDF_VS = [0.04, 0.09, 0.16]

#: Worst measured relative deviation over the 30 pinned points is 2.64e-5;
#: this is that bound rounded up, not a value tried until the test passed.
#: See the module docstring for the full derivation (A2 trigger).
_HESTON_PDF_REL_TOL = 3e-5
#: Worst measured relative deviation over the 80 pinned Green's-function
#: entries is 9.90e-4, at v = 0.02 (well below v0 = 0.09), where the
#: Cornish-Fisher bound is more sharply conditioned. Same derivation, larger
#: constant; also an A2 trigger.
_GREENS_SEMI_REL_TOL = 1e-3
_HESTON_PDF_REASON = (
    "cornishFisherEps takes a FOURTH central difference at step 1e-2, dividing "
    "a cancelling numerator by 1e-8; that amplifies the 2.2e-13 Bessel "
    "difference between QuantLib's modifiedBesselFunction_i and scipy's iv "
    "into a 9.1e-3 difference in the integration bound (measured: solving for "
    "the upper limit that reproduces C++ gives 0.4676764 vs 0.4634383). The "
    "pdf's sensitivity to that bound is 3e-7..2.6e-5 relative."
)


@pytest.mark.parametrize("t", [1.0, 2.0])
@pytest.mark.parametrize("v", _PDF_VS)
def test_heston_process_pdf_matches_cpp(
    cpp: dict[str, Any], heston: HestonProcess, t: float, v: float
) -> None:
    expected = cpp[f"heston_pdf_t{_fmt(t)}_v{_fmt(v)}"]
    for x, e in zip(_PDF_XS, expected, strict=True):
        # The value is a 100-panel SegmentIntegral of a 128-point
        # Gauss-Laguerre inner integral, with the outer bound coming from a
        # Cornish-Fisher expansion built out of five 1e-2-step central
        # differences of the characteristic function. That last construction
        # alone loses ~4 digits (a fourth difference at step 1e-2 divides by
        # 1e-8), and Boost's modified Bessel of complex argument is not
        # scipy's. 1e-8 relative is what that arithmetic supports.
        custom(
            heston.pdf(x, v, t, 1e-4),
            float(e),
            abs_tol=1e-9,
            rel_tol=_HESTON_PDF_REL_TOL,
            reason=_HESTON_PDF_REASON,
        )


# ---------------------------------------------------------------------------
# FdmHestonGreensFct
# ---------------------------------------------------------------------------

_TRAFOS = {
    "plain": TransformationType.Plain,
    "log": TransformationType.Log,
    "power": TransformationType.Power,
}
_ALGOS = {
    "zerocorr": FdmHestonGreensFctAlgorithm.ZeroCorrelation,
    "gaussian": FdmHestonGreensFctAlgorithm.Gaussian,
    "semi": FdmHestonGreensFctAlgorithm.SemiAnalytical,
}


def _mesher_log() -> FdmMesherComposite:
    return FdmMesherComposite(Uniform1dMesher(4.2, 5.0, 5), Uniform1dMesher(-2.6, -1.4, 4))


def _mesher_pos() -> FdmMesherComposite:
    return FdmMesherComposite(Uniform1dMesher(4.2, 5.0, 5), Uniform1dMesher(0.02, 0.30, 4))


def _assert_greens(actual: Array, expected: list[float], *, semi: bool) -> None:
    assert actual.size == len(expected)
    for a, e in zip(actual.tolist(), expected, strict=True):
        if semi:
            custom(
                float(a),
                float(e),
                abs_tol=1e-9,
                rel_tol=_GREENS_SEMI_REL_TOL,
                reason="SemiAnalytical is a direct call to HestonProcess.pdf; "
                "it inherits that method's bound. " + _HESTON_PDF_REASON,
            )
        else:
            tight(float(a), float(e))


@pytest.mark.parametrize("algo", sorted(_ALGOS))
@pytest.mark.parametrize("l0", [1.0, 1.3])
def test_greens_log_transform_matches_cpp(
    cpp: dict[str, Any], heston: HestonProcess, algo: str, l0: float
) -> None:
    g = FdmHestonGreensFct(_mesher_log(), heston, TransformationType.Log, l0)
    _assert_greens(
        g.get(1.0, _ALGOS[algo]),
        cpp[f"greens_log_{algo}_l0{_fmt(l0)}"],
        semi=(algo == "semi"),
    )


@pytest.mark.parametrize("algo", sorted(_ALGOS))
@pytest.mark.parametrize("trafo", ["plain", "power"])
def test_greens_positive_variance_transforms_match_cpp(
    cpp: dict[str, Any], heston: HestonProcess, algo: str, trafo: str
) -> None:
    g = FdmHestonGreensFct(_mesher_pos(), heston, _TRAFOS[trafo], 1.0)
    _assert_greens(
        g.get(1.0, _ALGOS[algo]),
        cpp[f"greens_pos_{trafo}_{algo}"],
        semi=(algo == "semi"),
    )


# ---------------------------------------------------------------------------
# LocalVolRNDCalculator
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lv_calc(spot: SimpleQuote) -> LocalVolRNDCalculator:
    return LocalVolRNDCalculator(
        spot,
        FlatForward.from_rate(_REF_DATE, 0.05, _DC),
        FlatForward.from_rate(_REF_DATE, 0.02, _DC),
        LocalConstantVol(reference_date=_REF_DATE, volatility=0.25, day_counter=_DC),
        x_grid=21,
        time_grid=TimeGrid.regular(1.0, 7),
    )


# Every LocalVolRNDCalculator number is downstream of seven Douglas steps of a
# forward PDE on a grid that is re-meshed and cubic-spline-re-interpolated
# whenever mass reaches the edge (four times in this fixture). Rounding is
# amplified by the sequence of solves and by the re-interpolations; 1e-8
# relative is the LOOSE tier and is what a rollback of this depth supports.
# The grids themselves are pinned TIGHT below, so a structural error cannot
# hide inside the loose end value.
_LV_REASON = (
    "seven Douglas steps of a forward PDE with four re-mesh + cubic-spline "
    "re-interpolation events; the grids that drive it are asserted TIGHT "
    "separately so only the accumulated rounding is loose here."
)


def test_local_vol_rnd_grid_structure_matches_cpp(
    cpp: dict[str, Any], lv_calc: LocalVolRNDCalculator
) -> None:
    """The time grid, the re-mesh schedule and the per-step meshers, at TIGHT."""
    assert lv_calc.rescale_time_steps() == cpp["lv_rescale_time_steps"]
    tight(lv_calc.time_grid().back(), float(cpp["lv_time_grid_back"]))
    assert lv_calc.time_grid().size() == int(cpp["lv_time_grid_size"])

    for i in (1, 3, 7):
        mesher = lv_calc.mesher(lv_calc.time_grid().at(i))
        expected = cpp[f"lv_mesher_{i}"]
        assert mesher.size() == len(expected)
        for j, e in enumerate(expected):
            custom(
                mesher.location(j),
                float(e),
                abs_tol=1e-10,
                rel_tol=1e-9,
                reason="mesher bounds after step i are functions of the solved "
                "density at step i-1, so they inherit its rollback rounding; "
                "the step-1 mesher, which is not, matches far more tightly.",
            )


@pytest.mark.parametrize("t", [0.0005, 0.05, 0.4, 1.0])
def test_local_vol_rnd_pdf_matches_cpp(
    cpp: dict[str, Any], lv_calc: LocalVolRNDCalculator, t: float
) -> None:
    mesher = lv_calc.mesher(lv_calc.time_grid().closest_time(t))
    locations = mesher.locations()
    lo = float(locations[0])
    hi = float(locations[-1])
    expected = cpp[f"lv_pdf_t{_fmt(t)}"]
    for k, e in enumerate(expected, start=1):
        x = lo + (hi - lo) * k / 8.0
        custom(lv_calc.pdf(x, t), float(e), abs_tol=1e-10, rel_tol=1e-8, reason=_LV_REASON)


@pytest.mark.parametrize("t", [0.4, 1.0])
def test_local_vol_rnd_cdf_and_invcdf_match_cpp(
    cpp: dict[str, Any], lv_calc: LocalVolRNDCalculator, t: float
) -> None:
    mesher = lv_calc.mesher(lv_calc.time_grid().closest_time(t))
    locations = mesher.locations()
    lo = float(locations[0])
    hi = float(locations[-1])

    for k, e in enumerate(cpp[f"lv_cdf_t{_fmt(t)}"], start=1):
        x = lo + (hi - lo) * k / 6.0
        custom(lv_calc.cdf(x, t), float(e), abs_tol=1e-10, rel_tol=1e-8, reason=_LV_REASON)

    for q, e in zip([0.1, 0.5, 0.9], cpp[f"lv_invcdf_t{_fmt(t)}"], strict=True):
        # invcdf is Brent on the above cdf, stopped at 0.1*localVolProbEps = 1e-7
        # absolute; with a density of order 1 in log-spot the root is located to
        # about that same order.
        custom(
            lv_calc.invcdf(q, t),
            float(e),
            abs_tol=1e-7,
            rel_tol=1e-7,
            reason="Brent to 1e-7 absolute on a cdf that is itself the "
            "adaptive integral of the rolled-back density.",
        )

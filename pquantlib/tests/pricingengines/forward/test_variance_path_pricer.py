"""Cross-validate :class:`VariancePathPricer` and :class:`Integrand` against C++ v1.43.

Reference: ``migration-harness/references/v143/pe/mcforward.json``, cases
``vs_pathpricer_direct`` and ``vs_pathpricer_direct_termvol``, produced by
``migration-harness/cpp/probes/v143_pe_mcforward/probe.cpp``.

Why two cases and not one
-------------------------
Under ``BlackConstantVol`` the integrand of ``VariancePathPricer`` is the
constant ``sigma ** 2``. Every quadrature rule, every path index and every
interpolation convention then returns ``sigma ** 2``, so the first case proves
only the ``/ t`` normalisation and the integration bounds — it cannot
distinguish a correct integration from a wrong one.

The second case is built on a ``BlackVarianceCurve`` (pillars 3M / 1Y / 2Y at
vols 0.15 / 0.25 / 0.30, ``force_monotone_variance=False``), so the local
volatility genuinely varies **in time** over the path and the answer depends on
*where in time* the integrand is sampled. That pins ``SegmentIntegral``'s
midpoint rule and the ``int(t / dt)`` interval count.

What is still NOT pinned, stated so nobody assumes otherwise: the truncating
**path index** ``i = int(u / dt)``. Both surfaces here are strike-independent —
``BlackConstantVol`` trivially, and ``BlackVarianceCurve`` because its Dupire
local vol is a function of ``t`` alone — so ``local_vol(u, path[k])`` does not
depend on ``k`` and floor / round / nearest all give the same answer.
Discriminating the index needs a strike-dependent local-vol surface and its own
C++ reference case. That is an open coverage gap, recorded rather than papered
over.

Tolerance
---------
TIGHT-tier absolute, LOOSE-tier relative. Measured agreement is 0.0 (bit-exact)
for the constant-vol case and 1.05e-14 relative for the term-structure case.
The residual there is not the integration — it is the Dupire step from Black
variance to local variance, which differentiates an interpolated variance curve
and so loses a few ULP; the ``termvol_curve`` case in the same reference pins
that surface separately and Python matches its Black vols bit-exactly and its
local vols to 4.6e-15 absolute.

Evaluation date
---------------
``probe.cpp`` line 462 sets ``Settings::instance().evaluationDate() =
Date(15, June, 2023)``; the fixture below pins the same date and restores the
previous value in teardown.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.methods.montecarlo.path import Path
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.forward.variance_path_pricer import (
    Integrand,
    VariancePathPricer,
)
from pquantlib.processes.black_scholes_merton_process import BlackScholesMertonProcess
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.volatility.equity_fx.black_variance_curve import (
    BlackVarianceCurve,
)
from pquantlib.termstructures.volatility.equity_fx.black_vol_term_structure import (
    BlackVolTermStructure,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_grid import TimeGrid
from pquantlib.time.time_unit import TimeUnit

# probe.cpp:150 — const Date TODAY(15, June, 2023);
TODAY = Date.from_ymd(15, Month.June, 2023)
# probe.cpp:151-154
RISK_FREE = 0.04
DIVIDEND = 0.015
VOL = 0.22
SPOT = 95.0

_ABS_TOL = 1.0e-14
_REL_TOL = 1.0e-8


@pytest.fixture(autouse=True)
def _eval_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """probe.cpp:462 — Settings::instance().evaluationDate() = TODAY."""
    settings = ObservableSettings()
    saved = settings.evaluation_date
    settings.evaluation_date = TODAY
    yield
    settings.evaluation_date = saved


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/pe/mcforward")


def _day_counter() -> Actual365Fixed:
    return Actual365Fixed()


def _process(vol: BlackVolTermStructure) -> GeneralizedBlackScholesProcess:
    """probe.cpp:167-181 — bsmProcess() / bsmProcessTermVol()."""
    dc = _day_counter()
    return BlackScholesMertonProcess(
        x0=SimpleQuote(SPOT),
        dividend_ts=FlatForward.from_rate(reference_date=TODAY, forward_rate=DIVIDEND, day_counter=dc),
        risk_free_ts=FlatForward.from_rate(reference_date=TODAY, forward_rate=RISK_FREE, day_counter=dc),
        black_vol_ts=vol,
    )


def _constant_vol() -> BlackVolTermStructure:
    return BlackConstantVol(
        reference_date=TODAY,
        calendar=NullCalendar(),
        day_counter=_day_counter(),
        volatility=VOL,
    )


def _term_vol() -> BlackVolTermStructure:
    """probe.cpp:172-180 — BlackVarianceCurve at 3M/1Y/2Y, 0.15/0.25/0.30."""
    return BlackVarianceCurve(
        reference_date=TODAY,
        dates=[
            TODAY + Period(3, TimeUnit.Months),
            TODAY + Period(1, TimeUnit.Years),
            TODAY + Period(2, TimeUnit.Years),
        ],
        black_vol_curve=[0.15, 0.25, 0.30],
        day_counter=_day_counter(),
        force_monotone_variance=False,
    )


_VOL_BY_KIND = {"constant": _constant_vol, "termStructure": _term_vol}


def _rebuild_path(case: dict[str, Any]) -> Path:
    grid = TimeGrid.regular(float(case["inputs"]["t"]), int(case["inputs"]["n"]))
    return Path(grid, np.asarray(case["inputs"]["path"], dtype=np.float64))


@pytest.mark.parametrize("case_name", ["vs_pathpricer_direct", "vs_pathpricer_direct_termvol"])
def test_variance_path_pricer_matches_cpp(cpp: dict[str, Any], case_name: str) -> None:
    """``VariancePathPricer(path)`` reproduces C++ v1.43 on a fixed path."""
    case = cpp[case_name]
    process = _process(_VOL_BY_KIND[str(case["inputs"]["volKind"])]())
    actual = VariancePathPricer(process)(_rebuild_path(case))
    tolerance.custom(
        actual,
        float(case["expected"]["value"]),
        abs_tol=_ABS_TOL,
        rel_tol=_REL_TOL,
        reason=(
            "SegmentIntegral midpoint rule over the squared local vol; the "
            "residual is the Dupire differentiation of the interpolated "
            "variance curve, not the quadrature"
        ),
    )


def test_constant_vol_case_is_exactly_sigma_squared(cpp: dict[str, Any]) -> None:
    """Sanity anchor: under constant vol the realised variance IS ``sigma ** 2``.

    This is what makes the constant-vol case weak on its own, and stating it
    here keeps the next reader from mistaking it for a strong check.
    """
    tolerance.tight(
        float(cpp["vs_pathpricer_direct"]["expected"]["value"]),
        VOL * VOL,
        reason="constant-vol integrand is path-independent",
    )


def test_integrand_is_the_squared_local_vol_at_the_truncated_index(
    cpp: dict[str, Any],
) -> None:
    """``Integrand(u) == local_vol(u, path[int(u / dt)]) ** 2``.

    What this does and does not prove, stated plainly because the distinction
    cost a wrong assertion here first:

    * It pins the composition — ``Integrand`` calls the process's *local*
      volatility (not the Black volatility) at the sampled time and squares it.
    * It does **not** discriminate the truncating path index, because both
      surfaces in the reference are strike-independent.
    * It does **not** discriminate the sampled time either, because
      ``BlackVarianceCurve`` interpolates linearly in total variance and so has
      a local vol that is piecewise constant between pillars.

    Both limitations are asserted below rather than left implicit. What
    genuinely pins the quadrature is the *integral* in
    :func:`test_variance_path_pricer_matches_cpp`, which spans the 3M pillar.
    Pinning the path index would need a strike-dependent local-vol surface and
    its own C++ reference case — recorded as an open gap, not papered over with
    an uncross-validated test.
    """
    case = cpp["vs_pathpricer_direct_termvol"]
    path = _rebuild_path(case)
    process = _process(_term_vol())
    integrand = Integrand(path, process)
    dt = path.time_grid.dt(0)

    # Just past the midpoint of cell 3: truncation gives index 3, rounding 4.
    u = 3.6 * dt
    sigma = process.diffusion_1d(u, path[3])
    tolerance.tight(integrand(u), sigma * sigma, reason="squared local vol at the sampled time")

    # Two documented limitations, asserted so they cannot silently stop being
    # true and quietly turn this test into a stronger claim than it is.
    #
    # (a) BlackVarianceCurve interpolates linearly in TOTAL VARIANCE, so its
    #     Dupire local vol is piecewise constant in time between pillars. Two
    #     times inside the same segment give the same local vol, so this test
    #     cannot pin the sampled time either — only crossing a pillar changes
    #     the value, which is why the INTEGRAL over [0, 1] (which crosses the
    #     3M pillar at t ~ 0.2521) is sensitive to the quadrature while a
    #     single point evaluation is not.
    assert math.isclose(process.diffusion_1d(3.0 * dt, path[3]), sigma, abs_tol=1e-18)

    # (b) The surface is strike-independent, so the path index is not pinned.
    assert path[3] != path[4]
    assert math.isclose(process.diffusion_1d(u, path[3]), process.diffusion_1d(u, path[4]), abs_tol=1e-15), (
        "BlackVarianceCurve local vol is strike-independent, so the path index is not pinned here"
    )


def test_empty_path_is_rejected() -> None:
    """C++: ``QL_REQUIRE(!path.empty(), "the path cannot be empty")``."""
    process = _process(_constant_vol())
    empty = Path(TimeGrid([], []), np.asarray([], dtype=np.float64))
    with pytest.raises(LibraryException, match="path cannot be empty"):
        VariancePathPricer(process)(empty)

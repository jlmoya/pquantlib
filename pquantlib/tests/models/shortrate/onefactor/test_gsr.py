"""Tests for the Gsr (Gaussian short-rate, forward-measure) model.

Cross-validates against ``migration-harness/references/cluster/l10b.json``.

C++ parity:
- ``ql/models/shortrate/onefactormodels/gsr.{hpp,cpp}`` @ v1.42.1
- ``ql/models/shortrate/onefactormodels/gaussian1dmodel.{hpp,cpp}`` @ v1.42.1
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.models.shortrate.onefactor.gsr import Gsr
from pquantlib.processes.gsr_process import GsrProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import loose, tight
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture
def reference_data() -> dict[str, Any]:
    return load_reference("cluster/l10b")


def _build_curve() -> FlatForward:
    """Flat 3% Continuous yield curve on Actual/365 Fixed at 15-May-2026."""
    return FlatForward(
        Date.from_ymd(15, Month.May, 2026),
        SimpleQuote(0.03),
        Actual365Fixed(),
    )


def _build_gsr() -> Gsr:
    """Build a Gsr matching the C++ probe setup.

    sigma_i = 0.01 for all i (4 pieces over 1Y/2Y/5Y step dates),
    reversion = 0.05 (constant), T = 60.
    """
    today = Date.from_ymd(15, Month.May, 2026)
    step_dates = [
        today + 365,
        today + 730,
        today + 1826,
    ]
    return Gsr(
        term_structure=_build_curve(),
        volstepdates=step_dates,
        volatilities=[0.01, 0.01, 0.01, 0.01],
        reversion=0.05,
        T=60.0,
    )


def test_gsr_inspectors(reference_data: dict[str, Any]) -> None:
    """volatility[0], reversion[0], numeraire_time round-trip."""
    ref = reference_data["gsr_model_const"]
    gsr = _build_gsr()
    tight(float(gsr.volatility()[0]), ref["sigma_first"])
    tight(float(gsr.reversion()[0]), ref["reversion_first"])
    tight(gsr.numeraire_time(), ref["numeraire_time"])


def test_gsr_zerobond_at_t0_matches_curve(reference_data: dict[str, Any]) -> None:
    """``zerobond(T, 0, 0) = curve.discount(T)`` by construction.

    TIGHT — at t=0, the model just delegates to the curve.
    """
    ref = reference_data["gsr_model_const"]
    gsr = _build_gsr()
    tight(gsr.zerobond(5.0, 0.0, 0.0), ref["zerobond_5_0_0"])
    tight(gsr.zerobond(10.0, 0.0, 0.0), ref["zerobond_10_0_0"])
    tight(gsr.zerobond(5.0, 0.0, 0.0), ref["curve_discount_5"])
    tight(gsr.zerobond(10.0, 0.0, 0.0), ref["curve_discount_10"])


def test_gsr_zerobond_at_t_gt_0(reference_data: dict[str, Any]) -> None:
    """``zerobond(T, t, y)`` at t > 0 — exercises the affine-form formula.

    LOOSE: the closed-form involves piecewise integrals plus
    floating-point cancellation in the std-deviation / expectation
    inverter; ~1e-12 agreement (better than LOOSE) is typical.
    """
    ref = reference_data["gsr_model_const"]
    gsr = _build_gsr()
    loose(gsr.zerobond(5.0, 1.0, 0.0), ref["zerobond_5_1_0"])
    loose(gsr.zerobond(5.0, 1.0, 0.5), ref["zerobond_5_1_0p5"])
    loose(gsr.zerobond(5.0, 1.0, -0.5), ref["zerobond_5_1_m0p5"])
    loose(gsr.zerobond(10.0, 2.0, 0.5), ref["zerobond_10_2_0p5"])


def test_gsr_numeraire(reference_data: dict[str, Any]) -> None:
    """``numeraire(t, y)`` — at t=0 equals curve discount to T_fwd.

    LOOSE for t > 0 (delegates through ``zerobond``).
    """
    ref = reference_data["gsr_model_const"]
    gsr = _build_gsr()
    # At t=0, exact curve-discount delegation.
    tight(gsr.numeraire(0.0, 0.0), ref["numeraire_0_0"])
    loose(gsr.numeraire(1.0, 0.0), ref["numeraire_1_0"])
    loose(gsr.numeraire(1.0, 0.5), ref["numeraire_1_0p5"])
    loose(gsr.numeraire(5.0, 0.0), ref["numeraire_5_0"])


def test_gsr_y_grid_matches_probe(reference_data: dict[str, Any]) -> None:
    """``y_grid(stdDevs=4, gridPoints=8, T=1, t=0, y=0)`` matches C++ exactly.

    TIGHT — the grid is a deterministic ratio of (e_t_T + std_t_T * j *
    h - e_0_T) / std_0_T. Because the single-piece GsrProcess
    expectation correction includes the tf term, the e's don't fully
    cancel in t=0 — but at t=0 the formula collapses to ``j * h``
    grid scaled by std_t_T / std_0_T (which is 1 for t=0). The probe
    used a SINGLE-PIECE GSR with no step dates; we reconstruct that here.
    """
    ref = reference_data["gsr_y_grid"]
    gsr_single = Gsr(
        term_structure=_build_curve(),
        volstepdates=[],
        volatilities=[0.01],
        reversion=0.05,
    )
    grid = gsr_single.y_grid(std_devs=4.0, grid_points=8, T=1.0, t=0.0, y=0.0)
    assert grid.size == ref["size"]
    for got, expected in zip(grid.tolist(), ref["values"], strict=True):
        tight(float(got), float(expected))


def test_gsr_state_process_is_gsrprocess() -> None:
    """The state process must be a ``GsrProcess`` instance."""
    gsr = _build_gsr()
    sp = gsr.state_process()
    assert isinstance(sp, GsrProcess)


def test_gsr_numeraire_time_setter() -> None:
    """``set_numeraire_time`` updates the underlying process."""
    gsr = _build_gsr()
    assert gsr.numeraire_time() == 60.0
    gsr.set_numeraire_time(40.0)
    assert gsr.numeraire_time() == 40.0


def test_gsr_piecewise_reversion_validation() -> None:
    """Reversion must be a scalar or a list of size n+1."""
    today = Date.from_ymd(15, Month.May, 2026)
    step_dates = [today + 365, today + 730]
    # 3 vols, only 2 reversions — invalid.
    with pytest.raises(LibraryException, match="1 or n\\+1 reversions"):
        Gsr(
            term_structure=_build_curve(),
            volstepdates=step_dates,
            volatilities=[0.01, 0.01, 0.01],
            reversion=[0.05, 0.05],
        )


def test_gsr_volatility_count_validation() -> None:
    """Volatilities must be n+1 for n step dates."""
    today = Date.from_ymd(15, Month.May, 2026)
    step_dates = [today + 365, today + 730]
    with pytest.raises(LibraryException, match="n\\+1 volatilities"):
        Gsr(
            term_structure=_build_curve(),
            volstepdates=step_dates,
            volatilities=[0.01],  # should be 3
            reversion=0.05,
        )


def test_gsr_piecewise_reversion_constructs() -> None:
    """Piecewise reversion (n+1 values) is accepted and exposed."""
    today = Date.from_ymd(15, Month.May, 2026)
    step_dates = [today + 365, today + 730]
    gsr = Gsr(
        term_structure=_build_curve(),
        volstepdates=step_dates,
        volatilities=[0.01, 0.01, 0.01],
        reversion=[0.05, 0.06, 0.07],
    )
    revs = gsr.reversion()
    assert len(revs) == 3
    tight(float(revs[0]), 0.05)
    tight(float(revs[1]), 0.06)
    tight(float(revs[2]), 0.07)


def test_gsr_volstep_must_be_positive() -> None:
    """First vol step must be > 0 (post-reference-date)."""
    today = Date.from_ymd(15, Month.May, 2026)
    # Vol step on the reference date is invalid (T(today) = 0).
    with pytest.raises(LibraryException, match="must be positive"):
        Gsr(
            term_structure=_build_curve(),
            volstepdates=[today],
            volatilities=[0.01, 0.01],
            reversion=0.05,
        )


def test_gsr_volsteps_must_be_increasing() -> None:
    """Step dates must yield strictly increasing times."""
    today = Date.from_ymd(15, Month.May, 2026)
    with pytest.raises(LibraryException, match="strictly increasing"):
        Gsr(
            term_structure=_build_curve(),
            volstepdates=[today + 730, today + 365],
            volatilities=[0.01, 0.01, 0.01],
            reversion=0.05,
        )


def test_gsr_zerobond_at_t0_with_external_yts() -> None:
    """When ``yts`` is provided and t=0, the model defers to that curve."""
    gsr = _build_gsr()
    alt_curve = FlatForward(
        Date.from_ymd(15, Month.May, 2026),
        SimpleQuote(0.04),  # different rate
        Actual365Fixed(),
    )
    # zerobond(5, 0, 0, yts) should match alt_curve.discount(5).
    tight(gsr.zerobond(5.0, 0.0, 0.0, alt_curve), alt_curve.discount(5.0))


def test_gsr_zerobond_decay_with_state() -> None:
    """At fixed (t, T), higher y -> lower zerobond (call payoff intuition).

    The G(t, T) > 0 and the exponential ``exp(-x G(t,T) ...)`` is
    decreasing in x; x is monotone in y. So zerobond is decreasing in
    y at fixed (t, T).
    """
    gsr = _build_gsr()
    p_neg = gsr.zerobond(5.0, 1.0, -1.0)
    p_zero = gsr.zerobond(5.0, 1.0, 0.0)
    p_pos = gsr.zerobond(5.0, 1.0, 1.0)
    assert p_pos < p_zero < p_neg

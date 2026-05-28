"""Tests for ``pquantlib.methods.montecarlo.path_generator.PathGenerator``.

# C++ parity: ql/methods/montecarlo/pathgenerator.hpp (v1.42.1).
"""

from __future__ import annotations

import math

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.methods.montecarlo.gaussian_sequence_generator import (
    make_pseudo_random_rsg,
)
from pquantlib.methods.montecarlo.path_generator import PathGenerator
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.time_grid import TimeGrid


def _make_bsm() -> GeneralizedBlackScholesProcess:
    """Standard textbook BSM process: S=100, r=5%, q=0%, sigma=20%."""
    dc = Actual365Fixed()
    cal = NullCalendar()
    ref = Date.from_ymd(15, Month.May, 2026)
    spot = SimpleQuote(100.0)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.0, day_counter=dc)
    vol = BlackConstantVol(
        reference_date=ref, calendar=cal, day_counter=dc, volatility=0.20
    )
    return GeneralizedBlackScholesProcess(
        x0=spot, dividend_ts=div, risk_free_ts=rf, black_vol_ts=vol
    )


def test_path_generator_dimension_and_grid() -> None:
    process = _make_bsm()
    gsg = make_pseudo_random_rsg(4, 42)
    pg = PathGenerator(process, length=1.0, time_steps=4, generator=gsg)
    assert pg.size() == 4
    assert pg.dimension() == 4
    assert len(pg.time_grid) == 5  # 4 steps → 5 grid points


def test_path_generator_dimension_mismatch_rejected() -> None:
    process = _make_bsm()
    gsg = make_pseudo_random_rsg(3, 42)  # only 3 dims for 4 steps
    with pytest.raises(Exception, match="dimensionality"):
        PathGenerator(process, length=1.0, time_steps=4, generator=gsg)


def test_path_starts_at_x0_and_evolves() -> None:
    process = _make_bsm()
    gsg = make_pseudo_random_rsg(4, 42)
    pg = PathGenerator(process, length=1.0, time_steps=4, generator=gsg)
    sample = pg.next()
    path = sample.value
    # First point is x0 = 100.
    assert path.front() == 100.0
    # BSM evolves multiplicatively: subsequent points are positive.
    for i in range(1, path.length()):
        assert path[i] > 0.0


def test_path_generator_with_time_grid() -> None:
    process = _make_bsm()
    grid = TimeGrid.regular(2.0, 4)
    gsg = make_pseudo_random_rsg(4, 42)
    pg = PathGenerator.with_time_grid(process, grid, gsg)
    sample = pg.next()
    assert sample.value.length() == 5
    assert sample.value.front() == 100.0


def test_antithetic_reuses_negated_increments() -> None:
    """Antithetic path uses the negated last Gaussian increments.

    For BSM ``evolve_1d`` is ``x0 * exp(drift + sqrt(var) * dw)`` with the
    same ``drift`` and ``sqrt(var)`` for both calls. Therefore
    ``log(forward[1]) - log(x0) - drift_step1`` and ``-(log(anti[1]) -
    log(x0) - drift_step1)`` should be equal (the sigma * sqrt(dt) * dw
    bits negate exactly).

    NOTE: ``PathGenerator`` mutates its internal ``Path`` in place per
    call (matching C++ ``mutable Sample<Path>``); the test snapshots
    individual floats before invoking ``antithetic()``.
    """
    process = _make_bsm()
    gsg = make_pseudo_random_rsg(4, 42)
    pg = PathGenerator(process, length=1.0, time_steps=4, generator=gsg)
    fwd_path = pg.next().value
    fwd_step1 = fwd_path[1]
    anti_path = pg.antithetic().value
    anti_step1 = anti_path[1]
    # Step-1 drift = (r - q) * dt - 0.5 * sigma^2 * dt
    #             = 0.05 * 0.25 - 0.5 * 0.04 * 0.25 = 0.0075
    drift = 0.0075
    log_x0 = math.log(100.0)
    fwd_noise = math.log(fwd_step1) - log_x0 - drift
    anti_noise = math.log(anti_step1) - log_x0 - drift
    # The Brownian-noise contributions are exact negatives.
    assert abs(fwd_noise + anti_noise) < 1e-12


def test_brownian_bridge_path_still_starts_at_x0() -> None:
    process = _make_bsm()
    gsg = make_pseudo_random_rsg(4, 42)
    pg = PathGenerator(
        process,
        length=1.0,
        time_steps=4,
        generator=gsg,
        brownian_bridge=True,
    )
    sample = pg.next()
    assert sample.value.front() == 100.0

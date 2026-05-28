"""Tests for ``MultiPathGenerator`` over a multi-D process.

# C++ parity: ql/methods/montecarlo/multipathgenerator.hpp (v1.42.1).
"""

from __future__ import annotations

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.methods.montecarlo.gaussian_sequence_generator import (
    make_pseudo_random_rsg,
)
from pquantlib.methods.montecarlo.multi_path_generator import MultiPathGenerator
from pquantlib.processes.heston_process import HestonProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.time_grid import TimeGrid


def _make_heston() -> HestonProcess:
    dc = Actual365Fixed()
    ref = Date.from_ymd(15, Month.May, 2026)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.0, day_counter=dc)
    return HestonProcess(
        risk_free_rate=rf,
        dividend_yield=div,
        s0=SimpleQuote(100.0),
        v0=0.04,
        kappa=2.0,
        theta=0.04,
        sigma=0.2,
        rho=-0.5,
    )


def test_multi_path_generator_dimension_factors_times_steps() -> None:
    process = _make_heston()
    grid = TimeGrid.regular(1.0, 4)
    # factors=2, steps=4 → dim = 8
    gsg = make_pseudo_random_rsg(8, 42)
    mpg = MultiPathGenerator(process, grid, gsg)
    sample = mpg.next()
    assert sample.value.asset_number() == 2  # spot + variance
    assert sample.value.path_size() == 5
    # Spot path starts at 100.
    assert sample.value[0].front() == 100.0
    # Variance path starts at v0 = 0.04.
    assert sample.value[1].front() == 0.04


def test_multi_path_generator_dim_mismatch_rejected() -> None:
    process = _make_heston()
    grid = TimeGrid.regular(1.0, 4)
    gsg = make_pseudo_random_rsg(7, 42)  # wrong: should be 8
    with pytest.raises(Exception, match="dimension"):
        MultiPathGenerator(process, grid, gsg)


def test_brownian_bridge_not_supported_for_multi_path() -> None:
    process = _make_heston()
    grid = TimeGrid.regular(1.0, 4)
    gsg = make_pseudo_random_rsg(8, 42)
    mpg = MultiPathGenerator(process, grid, gsg, brownian_bridge=True)
    with pytest.raises(LibraryException, match="Brownian bridge not supported"):
        mpg.next()


def test_multi_path_antithetic_negates_increments() -> None:
    """Antithetic re-uses the last sequence and negates each variate.

    NOTE: ``MultiPath`` is mutated in place per call (matching C++
    ``mutable Sample<MultiPath>``). The test snapshots immutable
    floats out of the sample before reading the next one.
    """
    process = _make_heston()
    grid = TimeGrid.regular(1.0, 4)
    gsg = make_pseudo_random_rsg(8, 42)
    mpg = MultiPathGenerator(process, grid, gsg)
    fwd_sample = mpg.next().value
    fwd_spot_front = fwd_sample[0].front()
    fwd_var_front = fwd_sample[1].front()
    fwd_spot_back = fwd_sample[0].back()
    anti_sample = mpg.antithetic().value
    anti_spot_front = anti_sample[0].front()
    anti_var_front = anti_sample[1].front()
    anti_spot_back = anti_sample[0].back()
    # Both paths start at the same initial values.
    assert fwd_spot_front == anti_spot_front
    assert fwd_var_front == anti_var_front
    # And the antithetic spot path is generally different from the forward.
    assert fwd_spot_back != anti_spot_back

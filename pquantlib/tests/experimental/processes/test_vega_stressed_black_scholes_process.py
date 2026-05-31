"""Tests for VegaStressedBlackScholesProcess.

# C++ parity: ql/experimental/processes/vegastressedblackscholesprocess.hpp.

The vega-stress bump is a pure Python-side arithmetic shift on top of the
GBSM local vol, so it needs no C++ probe — the contract is verified by
constructing a constant-vol GBSM and checking the additive bump inside /
outside the rectangular stress region.
"""

from __future__ import annotations

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.experimental.processes.vega_stressed_black_scholes_process import (
    VegaStressedBlackScholesProcess,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.tolerance import tight
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month


def _build(
    *,
    lower_time: float = 0.0,
    upper_time: float = 1000000.0,
    lower_asset: float = 0.0,
    upper_asset: float = 1000000.0,
    stress: float = 0.0,
) -> VegaStressedBlackScholesProcess:
    dc = Actual365Fixed()
    cal = NullCalendar()
    ref = Date.from_ymd(15, Month.January, 2024)
    spot = SimpleQuote(100.0)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.02, day_counter=dc)
    vol = BlackConstantVol(reference_date=ref, calendar=cal, day_counter=dc, volatility=0.25)
    return VegaStressedBlackScholesProcess(
        x0=spot,
        dividend_ts=div,
        risk_free_ts=rf,
        black_vol_ts=vol,
        lower_time_border=lower_time,
        upper_time_border=upper_time,
        lower_asset_border=lower_asset,
        upper_asset_border=upper_asset,
        stress_level=stress,
    )


def test_vega_default_zero_stress_matches_base() -> None:
    """With zero stress (default region everywhere), diffusion == base vol.

    TIGHT: constant-vol GBSM gives 0.25 local vol; no bump.
    """
    p = _build()
    tight(p.diffusion_1d(0.5, 100.0), 0.25)


def test_vega_bump_inside_region() -> None:
    """Inside the stress box the diffusion is base + stress.

    TIGHT: 0.25 + 0.05.
    """
    p = _build(
        lower_time=0.0,
        upper_time=1.0,
        lower_asset=90.0,
        upper_asset=110.0,
        stress=0.05,
    )
    tight(p.diffusion_1d(0.5, 100.0), 0.30)


def test_vega_no_bump_outside_time_region() -> None:
    """Outside the time window no bump is applied.

    TIGHT: t=2.0 is past upper_time=1.0.
    """
    p = _build(
        lower_time=0.0,
        upper_time=1.0,
        lower_asset=90.0,
        upper_asset=110.0,
        stress=0.05,
    )
    tight(p.diffusion_1d(2.0, 100.0), 0.25)


def test_vega_no_bump_outside_asset_region() -> None:
    """Outside the asset window no bump is applied.

    TIGHT: x=200 is past upper_asset=110.
    """
    p = _build(
        lower_time=0.0,
        upper_time=1.0,
        lower_asset=90.0,
        upper_asset=110.0,
        stress=0.05,
    )
    tight(p.diffusion_1d(0.5, 200.0), 0.25)


def test_vega_setters_roundtrip_and_notify() -> None:
    """Setters update borders + stress level and re-bump accordingly.

    TIGHT: after widening the asset window, the previously-excluded
    x=200 now gets the bump.
    """
    p = _build(
        lower_time=0.0,
        upper_time=1.0,
        lower_asset=90.0,
        upper_asset=110.0,
        stress=0.05,
    )
    tight(p.diffusion_1d(0.5, 200.0), 0.25)
    p.set_upper_asset_border_for_stress_test(300.0)
    assert p.get_upper_asset_border_for_stress_test() == 300.0
    tight(p.diffusion_1d(0.5, 200.0), 0.30)
    p.set_stress_level(0.10)
    assert p.get_stress_level() == 0.10
    tight(p.diffusion_1d(0.5, 200.0), 0.35)


def test_vega_accessors_roundtrip() -> None:
    """All border accessors return their constructed values."""
    p = _build(
        lower_time=0.1,
        upper_time=0.9,
        lower_asset=80.0,
        upper_asset=120.0,
        stress=0.03,
    )
    assert p.get_lower_time_border_for_stress_test() == 0.1
    assert p.get_upper_time_border_for_stress_test() == 0.9
    assert p.get_lower_asset_border_for_stress_test() == 80.0
    assert p.get_upper_asset_border_for_stress_test() == 120.0
    assert p.get_stress_level() == 0.03

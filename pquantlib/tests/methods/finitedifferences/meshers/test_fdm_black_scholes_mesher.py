"""Tests for FdmBlackScholesMesher (1-D log-spot mesher).

# C++ parity: ql/methods/finitedifferences/meshers/fdmblackscholesmesher.{hpp,cpp}
# @ v1.42.1.

Cross-validates the uniform-mode mesh (no concentrating cPoint —
the Python port defers ``Concentrating1dMesher`` to Phase 6).
Reference JSON: ``fdm_bs_mesher_uniform`` in ``cluster/l5d.json``.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.methods.finitedifferences.meshers.fdm_black_scholes_mesher import (
    FdmBlackScholesMesher,
)
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import tight
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture
def reference_data() -> dict[str, Any]:
    return load_reference("cluster/l5d")


def _build_process() -> tuple[GeneralizedBlackScholesProcess, Date]:
    dc = Actual365Fixed()
    cal = NullCalendar()
    ref = Date.from_ymd(15, Month.June, 2026)
    expiry = ref + 365
    spot_q = SimpleQuote(100.0)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.0, day_counter=dc)
    vol = BlackConstantVol(reference_date=ref, calendar=cal, day_counter=dc, volatility=0.20)
    process = GeneralizedBlackScholesProcess(x0=spot_q, dividend_ts=div, risk_free_ts=rf, black_vol_ts=vol)
    return process, expiry


def test_size(reference_data: dict[str, Any]) -> None:
    process, expiry = _build_process()
    maturity = process.time(expiry)
    m = FdmBlackScholesMesher(11, process, maturity, 100.0)
    assert m.size() == reference_data["fdm_bs_mesher_uniform"]["size"]


def test_locations_match_uniform_reference(reference_data: dict[str, Any]) -> None:
    """Python port uses Uniform1dMesher only — Concentrating1dMesher
    deferred to Phase 6. Reference value here is from the
    ``fdm_bs_mesher_uniform`` C++ probe with ``cPoint = (Null, Null)``.

    TIGHT-tier: the boundary-derivation uses ``norminv(1-eps) *
    sigma * sqrt(T) * scale`` — C++ uses its own InverseCumulativeNormal
    (Acklam's approximation), Python ports the identical routine, so
    the two pipelines agree to TIGHT.
    """
    process, expiry = _build_process()
    maturity = process.time(expiry)
    m = FdmBlackScholesMesher(11, process, maturity, 100.0)
    locs = m.locations()
    expected = reference_data["fdm_bs_mesher_uniform"]["locations"]
    for actual_v, expected_v in zip(locs, expected, strict=True):
        tight(float(actual_v), float(expected_v))


def test_dplus_is_uniform() -> None:
    """All interior + first dplus values are identical (uniform mesh)."""
    process, expiry = _build_process()
    maturity = process.time(expiry)
    m = FdmBlackScholesMesher(11, process, maturity, 100.0)
    # The 0-th and 1-st node have the same dplus (uniform spacing).
    tight(m.dplus(0), m.dplus(1))
    # Last dplus is NaN (boundary sentinel).
    assert math.isnan(m.dplus(10))


def test_negative_spot_raises() -> None:
    """The C++ mesher asserts spot > 0. Python port mirrors this."""
    dc = Actual365Fixed()
    cal = NullCalendar()
    ref = Date.from_ymd(15, Month.June, 2026)
    spot_q = SimpleQuote(-1.0)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.0, day_counter=dc)
    vol = BlackConstantVol(reference_date=ref, calendar=cal, day_counter=dc, volatility=0.20)
    process = GeneralizedBlackScholesProcess(x0=spot_q, dividend_ts=div, risk_free_ts=rf, black_vol_ts=vol)
    with pytest.raises(LibraryException):
        FdmBlackScholesMesher(11, process, 1.0, 100.0)

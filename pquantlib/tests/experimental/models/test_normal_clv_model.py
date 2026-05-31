"""Tests for NormalCLVModel.

Cross-validates against ``migration-harness/references/cluster/w8d.json``.

C++ parity: ql/experimental/models/normalclvmodel.{hpp,cpp} @ v1.42.1.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.experimental.models.normal_clv_model import NormalCLVModel
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.processes.ornstein_uhlenbeck_process import OrnsteinUhlenbeckProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import loose, tight
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_REF = Date.from_ymd(15, Month.January, 2024)


@pytest.fixture
def ref() -> dict[str, Any]:
    return load_reference("cluster/w8d")


def _model() -> tuple[NormalCLVModel, GeneralizedBlackScholesProcess]:
    dc = Actual365Fixed()
    bs = GeneralizedBlackScholesProcess(
        x0=SimpleQuote(100.0),
        dividend_ts=FlatForward(_REF, SimpleQuote(0.02), dc),
        risk_free_ts=FlatForward(_REF, SimpleQuote(0.05), dc),
        black_vol_ts=BlackConstantVol(
            reference_date=_REF, calendar=TARGET(), day_counter=dc, volatility=0.25
        ),
    )
    ou = OrnsteinUhlenbeckProcess(0.5, 0.2, 0.1, 0.0)
    mats = [
        _REF + Period(6, TimeUnit.Months),
        _REF + Period(1, TimeUnit.Years),
        _REF + Period(2, TimeUnit.Years),
    ]
    return NormalCLVModel(bs, ou, mats, 5), bs


def _d1y() -> Date:
    return _REF + Period(1, TimeUnit.Years)


def test_cdf_invcdf(ref: dict[str, Any]) -> None:
    model, _ = _model()
    tight(model.cdf(_d1y(), 100.0), ref["nclv_cdf_1y_k100"])
    tight(model.inv_cdf(_d1y(), 0.5), ref["nclv_invcdf_1y_q05"])


def test_collocation_points_x(ref: dict[str, Any]) -> None:
    model, _ = _model()
    cx = model.collocation_points_x(_d1y())
    for i in range(5):
        tight(float(cx[i]), ref[f"nclv_cx_1y_{i}"])


def test_collocation_points_y(ref: dict[str, Any]) -> None:
    model, _ = _model()
    cy = model.collocation_points_y(_d1y())
    for i in range(5):
        tight(float(cy[i]), ref[f"nclv_cy_1y_{i}"])


def test_mapping_function(ref: dict[str, Any]) -> None:
    model, bs = _model()
    g = model.g()
    t1 = bs.time(_d1y())
    tight(g(t1, 0.1), ref["nclv_g_1y_x0"])
    tight(g(t1, 0.3), ref["nclv_g_1y_xp"])
    tight(g(t1, -0.1), ref["nclv_g_1y_xn"])


def test_mapping_function_interpolated_maturity(ref: dict[str, Any]) -> None:
    """g(t, x) linearly interpolates collocation y-points between maturities."""
    model, bs = _model()
    g = model.g()
    tmid = bs.time(_REF + Period(9, TimeUnit.Months))
    # linear-in-t interpolation across maturities -> LOOSE
    loose(g(tmid, 0.05), ref["nclv_g_9m_x0"])


def test_sorted_maturity_guard() -> None:
    """Out-of-order maturities raise."""
    dc = Actual365Fixed()
    bs = GeneralizedBlackScholesProcess(
        x0=SimpleQuote(100.0),
        dividend_ts=FlatForward(_REF, SimpleQuote(0.02), dc),
        risk_free_ts=FlatForward(_REF, SimpleQuote(0.05), dc),
        black_vol_ts=BlackConstantVol(
            reference_date=_REF, calendar=TARGET(), day_counter=dc, volatility=0.25
        ),
    )
    ou = OrnsteinUhlenbeckProcess(0.5, 0.2, 0.1, 0.0)
    with pytest.raises(LibraryException):
        NormalCLVModel(
            bs, ou, [_REF + Period(2, TimeUnit.Years), _REF + Period(1, TimeUnit.Years)], 5
        )

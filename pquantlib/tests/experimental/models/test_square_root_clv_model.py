"""Tests for SquareRootCLVModel.

Cross-validates against ``migration-harness/references/cluster/w8d.json``.

C++ parity: ql/experimental/models/squarerootclvmodel.{hpp,cpp} @ v1.42.1.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.experimental.models.square_root_clv_model import SquareRootCLVModel
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.processes.square_root_process import SquareRootProcess
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


def _model() -> tuple[SquareRootCLVModel, GeneralizedBlackScholesProcess]:
    dc = Actual365Fixed()
    bs = GeneralizedBlackScholesProcess(
        x0=SimpleQuote(100.0),
        dividend_ts=FlatForward(_REF, SimpleQuote(0.02), dc),
        risk_free_ts=FlatForward(_REF, SimpleQuote(0.05), dc),
        black_vol_ts=BlackConstantVol(
            reference_date=_REF, calendar=TARGET(), day_counter=dc, volatility=0.25
        ),
    )
    # SquareRootProcess(b=theta, a=kappa, sigma, x0)
    sqp = SquareRootProcess(0.09, 1.0, 0.2, 0.09)
    mats = [_REF + Period(1, TimeUnit.Years), _REF + Period(2, TimeUnit.Years)]
    return SquareRootCLVModel(bs, sqp, mats, 5), bs


def _d1y() -> Date:
    return _REF + Period(1, TimeUnit.Years)


def test_cdf_invcdf(ref: dict[str, Any]) -> None:
    model, _ = _model()
    tight(model.cdf(_d1y(), 100.0), ref["sclv_cdf_1y_k100"])
    tight(model.inv_cdf(_d1y(), 0.5), ref["sclv_invcdf_1y_q05"])


def test_collocation_points_x(ref: dict[str, Any]) -> None:
    """Golub-Welsch nodes via scipy eigh_tridiagonal vs C++ TQR -> LOOSE."""
    model, _ = _model()
    cx = model.collocation_points_x(_d1y())
    for i in range(5):
        loose(float(cx[i]), ref[f"sclv_cx_1y_{i}"])


def test_collocation_points_y(ref: dict[str, Any]) -> None:
    model, _ = _model()
    cy = model.collocation_points_y(_d1y())
    for i in range(5):
        loose(float(cy[i]), ref[f"sclv_cy_1y_{i}"])


def test_mapping_function(ref: dict[str, Any]) -> None:
    model, bs = _model()
    g = model.g()
    t1 = bs.time(_d1y())
    cx = model.collocation_points_x(_d1y())
    # exact maturity hit -> Lagrange interpolation through the collocation pts
    loose(g(t1, float(cx[2])), ref["sclv_g_1y_x"])


def test_mapping_function_extrapolation_guard() -> None:
    """g(t, x) refuses t below / above the maturity span."""
    model, bs = _model()
    g = model.g()
    # t before the first maturity (1Y) -> raise
    t_early = bs.time(_REF + Period(3, TimeUnit.Months))
    with pytest.raises(LibraryException):
        g(t_early, 100.0)

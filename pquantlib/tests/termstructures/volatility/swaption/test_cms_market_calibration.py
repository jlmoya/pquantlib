"""Tests for CmsMarketCalibration — SABR cube calibration loop.

# C++ parity: ql/termstructures/volatility/swaption/cmsmarketcalibration.{hpp,cpp}
# (v1.42.1).
"""

from __future__ import annotations

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.math.interpolations.sabr_formula import sabr_volatility
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.swaption.cms_market import (
    CmsMarket,
    SwapIndexLike,
)
from pquantlib.termstructures.volatility.swaption.cms_market_calibration import (
    CalibrationType,
    CmsMarketCalibration,
)
from pquantlib.termstructures.volatility.swaption.sabr_swaption_volatility_cube import (
    SabrSwaptionVolatilityCube,
)
from pquantlib.termstructures.volatility.swaption.swaption_volatility_matrix import (
    SwaptionVolatilityMatrix,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_REF_DATE = Date.from_ymd(15, Month.January, 2024)
_FORWARD = 0.04


class _FakeSwapIndex:
    """SwapIndexLike test double."""

    def __init__(self, tenor: Period) -> None:
        self._tenor = tenor

    def tenor(self) -> Period:
        return self._tenor

    def fixing(self, fixing_date: Date, forecast_todays_fixing: bool = False) -> float:
        _ = fixing_date, forecast_todays_fixing
        return _FORWARD


def _make_cube() -> SabrSwaptionVolatilityCube:
    """Reuse the L9-C cube fixture in compact form."""
    option_tenors = [Period(1, TimeUnit.Years), Period(3, TimeUnit.Years)]
    swap_tenors = [Period(2, TimeUnit.Years), Period(5, TimeUnit.Years)]
    strike_spreads = [-0.01, -0.005, 0.0, 0.005, 0.01]
    expiry = 1.0
    alpha, beta, nu, rho = 0.04, 0.5, 0.4, -0.2
    atm_vol = sabr_volatility(_FORWARD, _FORWARD, expiry, alpha, beta, nu, rho)
    atm = SwaptionVolatilityMatrix(
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        option_tenors=option_tenors,
        swap_tenors=swap_tenors,
        volatilities=np.full((2, 2), atm_vol),
        calendar=TARGET(),
        day_counter=Actual365Fixed(),
        reference_date=_REF_DATE,
    )
    rows: list[list[SimpleQuote]] = []
    for _j in range(len(option_tenors)):
        for _k in range(len(swap_tenors)):
            row: list[SimpleQuote] = []
            for s in strike_spreads:
                strike = _FORWARD + s
                v = sabr_volatility(strike, _FORWARD, expiry, alpha, beta, nu, rho)
                row.append(SimpleQuote(v - atm_vol))
            rows.append(row)
    return SabrSwaptionVolatilityCube(
        atm_vol_structure=atm,
        option_tenors=option_tenors,
        swap_tenors=swap_tenors,
        strike_spreads=strike_spreads,
        vol_spreads=rows,
        swap_index_base=_FakeSwapIndex(Period(5, TimeUnit.Years)),
        short_swap_index_base=_FakeSwapIndex(Period(1, TimeUnit.Years)),
        is_parameter_fixed=(False, True, False, False),
    )


def _make_market() -> CmsMarket:
    swap_lengths = [Period(5, TimeUnit.Years), Period(10, TimeUnit.Years)]
    swap_indexes: list[SwapIndexLike] = [
        _FakeSwapIndex(Period(2, TimeUnit.Years)),
        _FakeSwapIndex(Period(5, TimeUnit.Years)),
    ]
    bid_ask = [
        [
            (SimpleQuote(0.0010), SimpleQuote(0.0020)),
            (SimpleQuote(0.0015), SimpleQuote(0.0025)),
        ],
        [
            (SimpleQuote(0.0020), SimpleQuote(0.0030)),
            (SimpleQuote(0.0025), SimpleQuote(0.0035)),
        ],
    ]
    pricers: list[object] = [object(), object()]
    discount = FlatForward(
        reference_date=_REF_DATE, forward=SimpleQuote(0.03),
        day_counter=Actual365Fixed(),
    )
    return CmsMarket(
        swap_lengths=swap_lengths,
        swap_indexes=swap_indexes,
        bid_ask_spreads=bid_ask,
        pricers=pricers,
        discount_curve=discount,
    )


def _weights() -> list[list[float]]:
    return [[1.0, 1.0], [1.0, 1.0]]


# --- construction + invariants ----------------------------------------------


def test_construction_succeeds() -> None:
    cube = _make_cube()
    market = _make_market()
    calib = CmsMarketCalibration(cube, market, _weights())
    assert calib.calibration_type == CalibrationType.OnSpread
    assert calib.weights.shape == (2, 2)


def test_construction_rejects_weights_shape_mismatch() -> None:
    cube = _make_cube()
    market = _make_market()
    with pytest.raises(LibraryException, match="weights"):
        CmsMarketCalibration(cube, market, [[1.0]])  # 1x1, expected 2x2


def test_calibration_type_default_is_on_spread() -> None:
    cube = _make_cube()
    market = _make_market()
    calib = CmsMarketCalibration(cube, market, _weights())
    assert calib.calibration_type == CalibrationType.OnSpread


def test_calibration_type_enum_values() -> None:
    """Sanity check — the enum members match the C++ enum constants."""
    assert CalibrationType.OnSpread.value == 0
    assert CalibrationType.OnPrice.value == 1
    assert CalibrationType.OnForwardCmsPrice.value == 2


# --- parameter transforms (pure math) ---------------------------------------


def test_beta_transform_roundtrip() -> None:
    """beta_transform_direct(inverse(b)) ≈ b for b in (0, 1)."""
    for b in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
        y = CmsMarketCalibration.beta_transform_inverse(b)
        b2 = CmsMarketCalibration.beta_transform_direct(y)
        assert abs(b - b2) < 1e-10, f"roundtrip mismatch: {b} → {y} → {b2}"


def test_beta_transform_direct_clamps_to_bounds() -> None:
    """For very large |y|, beta clamps to the documented range."""
    # y = 0 → beta = 1, then clamp to 0.999999.
    assert CmsMarketCalibration.beta_transform_direct(0.0) == 0.999999
    # |y| >= 10 → 0 then clamp to 0.000001.
    assert CmsMarketCalibration.beta_transform_direct(10.0) == 0.000001
    assert CmsMarketCalibration.beta_transform_direct(-10.0) == 0.000001


def test_reversion_transform_roundtrip() -> None:
    """reversion_transform_direct(inverse(r)) == r for r >= 0."""
    for r in [0.001, 0.01, 0.1, 0.5, 1.0, 2.0]:
        y = CmsMarketCalibration.reversion_transform_inverse(r)
        r2 = CmsMarketCalibration.reversion_transform_direct(y)
        assert abs(r - r2) < 1e-12, f"roundtrip mismatch: {r} → {y} → {r2}"


# --- compute() ---------------------------------------------------------------


def test_compute_runs_against_synthetic_residual() -> None:
    """compute() returns the calibrated parameter vector for a simple residual.

    We give it a quadratic residual ``r(x) = x - target`` and verify the
    minimiser finds ``x = target``.
    """
    cube = _make_cube()
    market = _make_market()
    calib = CmsMarketCalibration(cube, market, _weights())
    target = np.array([0.04, 0.5, 0.4, -0.2], dtype=np.float64)

    def residual(x: np.ndarray) -> np.ndarray:
        return x - target

    sol = calib.compute(
        guess=[0.05, 0.4, 0.3, -0.1],
        residual_fn=residual,
    )
    assert np.allclose(sol, target, atol=1e-6)
    # Diagnostics populated.
    assert calib.error < 1e-4
    assert isinstance(calib.end_criteria, str)


def test_compute_rejects_empty_guess() -> None:
    cube = _make_cube()
    market = _make_market()
    calib = CmsMarketCalibration(cube, market, _weights())

    def residual(x: np.ndarray) -> np.ndarray:
        return x

    with pytest.raises(LibraryException, match="empty guess"):
        calib.compute(guess=[], residual_fn=residual)


def test_inspectors_return_input_state() -> None:
    cube = _make_cube()
    market = _make_market()
    calib = CmsMarketCalibration(cube, market, _weights(),
                                  calibration_type=CalibrationType.OnPrice)
    assert calib.volatility_cube is cube
    assert calib.cms_market is market
    assert calib.calibration_type == CalibrationType.OnPrice

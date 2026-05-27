"""CalibrationHelper + BlackCalibrationHelper behavioral tests.

The concrete subclasses (SwaptionHelper / CapHelper / HestonModelHelper)
land in L4-C/E. Here we test the abstract surface and the calibration-
error dispatch via a fake_for_testing subclass.

No new C++ probe is needed for the bases — the C++ ``CalibrationHelper``
is abstract (one pure-virtual function); ``BlackCalibrationHelper``'s
``calibrationError()`` dispatch is a 3-arm switch tested here via direct
calls.
"""

from __future__ import annotations

from typing import override

import pytest

from pquantlib.models.calibration_helper import (
    BlackCalibrationHelper,
    CalibrationErrorType,
    CalibrationHelper,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.testing import tolerance


class _FakeBlackHelper(BlackCalibrationHelper):
    """Minimal BlackCalibrationHelper for behavioral tests.

    ``black_price(v) = strike * v`` (linear in vol — used to exercise the
    implied-vol inversion and the three calibration-error arms with
    closed-form values).
    """

    _strike: float
    _model_value: float

    def __init__(
        self,
        volatility: SimpleQuote,
        strike: float = 100.0,
        model_value: float = 5.0,
        calibration_error_type: CalibrationErrorType = CalibrationErrorType.RelativePriceError,
        volatility_type: VolatilityType = VolatilityType.ShiftedLognormal,
    ) -> None:
        super().__init__(volatility, calibration_error_type, volatility_type)
        self._strike = strike
        self._model_value = model_value

    @override
    def model_value(self) -> float:
        return self._model_value

    @override
    def add_times_to(self, times: list[float]) -> None:
        # No schedule in this fake.
        pass

    @override
    def black_price(self, volatility: float) -> float:
        # Linear price for invertibility.
        return self._strike * volatility


# ---------------------------------------------------------------------
# CalibrationHelper abstract surface
# ---------------------------------------------------------------------


def test_calibration_helper_is_abstract() -> None:
    with pytest.raises(TypeError, match="Can't instantiate"):
        CalibrationHelper()  # type: ignore[abstract]


def test_black_calibration_helper_is_abstract() -> None:
    with pytest.raises(TypeError, match="Can't instantiate"):
        BlackCalibrationHelper(SimpleQuote(0.20))  # type: ignore[abstract]


# ---------------------------------------------------------------------
# Lazy-evaluation contract
# ---------------------------------------------------------------------


def test_market_value_uses_black_price_at_vol() -> None:
    # strike=100, vol=0.20 => market_value = 20.0.
    h = _FakeBlackHelper(SimpleQuote(0.20), strike=100.0, model_value=5.0)
    tolerance.tight(h.market_value(), 20.0)


def test_market_value_recomputes_after_vol_update() -> None:
    vol = SimpleQuote(0.20)
    h = _FakeBlackHelper(vol, strike=100.0, model_value=5.0)
    tolerance.tight(h.market_value(), 20.0)
    vol.set_value(0.30)
    # LazyObject.update invalidates the cache; next call re-evaluates.
    tolerance.tight(h.market_value(), 30.0)


# ---------------------------------------------------------------------
# CalibrationErrorType: each arm
# ---------------------------------------------------------------------


def test_relative_price_error() -> None:
    # market = strike * vol = 100 * 0.20 = 20; model = 25.
    # relative error = |20 - 25| / 20 = 0.25.
    h = _FakeBlackHelper(
        SimpleQuote(0.20),
        strike=100.0,
        model_value=25.0,
        calibration_error_type=CalibrationErrorType.RelativePriceError,
    )
    tolerance.tight(h.calibration_error(), 0.25)


def test_price_error() -> None:
    # market - model = 20 - 25 = -5.
    h = _FakeBlackHelper(
        SimpleQuote(0.20),
        strike=100.0,
        model_value=25.0,
        calibration_error_type=CalibrationErrorType.PriceError,
    )
    tolerance.tight(h.calibration_error(), -5.0)


def test_implied_vol_error_lognormal() -> None:
    # market vol = 0.20; model price = 30 => implied vol = 30/100 = 0.30.
    # error = implied - market = 0.30 - 0.20 = 0.10.
    h = _FakeBlackHelper(
        SimpleQuote(0.20),
        strike=100.0,
        model_value=30.0,
        calibration_error_type=CalibrationErrorType.ImpliedVolError,
        volatility_type=VolatilityType.ShiftedLognormal,
    )
    tolerance.loose(h.calibration_error(), 0.10)


def test_implied_vol_error_caps_below_min_vol() -> None:
    # If model price <= price at min_vol, implied = min_vol = 0.0010.
    # error = 0.0010 - 0.20 = -0.199 (ShiftedLognormal min_vol=0.0010).
    h = _FakeBlackHelper(
        SimpleQuote(0.20),
        strike=100.0,
        model_value=0.0,
        calibration_error_type=CalibrationErrorType.ImpliedVolError,
        volatility_type=VolatilityType.ShiftedLognormal,
    )
    tolerance.tight(h.calibration_error(), 0.0010 - 0.20)


# ---------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------


def test_volatility_quote_accessor() -> None:
    vol = SimpleQuote(0.15)
    h = _FakeBlackHelper(vol)
    assert h.volatility is vol


def test_volatility_type_default_is_shifted_lognormal() -> None:
    h = _FakeBlackHelper(SimpleQuote(0.20))
    assert h.volatility_type == VolatilityType.ShiftedLognormal


def test_shift_default_is_zero() -> None:
    h = _FakeBlackHelper(SimpleQuote(0.20))
    tolerance.exact(h.shift, 0.0)

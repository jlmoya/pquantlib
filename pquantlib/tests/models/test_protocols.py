"""Cross-cluster Protocol behavioral tests.

The Protocols themselves carry no logic — they exist for type-checking.
Tests verify that the canonical pquantlib classes (CalibratedModel,
BlackCalibrationHelper subclass) structurally satisfy the corresponding
Protocol, and that arbitrary impostors fail the check.
"""

from __future__ import annotations

from typing import override

import numpy as np
import numpy.typing as npt

from pquantlib.math.optimization.constraint import NoConstraint
from pquantlib.models.calibration_helper import (
    BlackCalibrationHelper,
    CalibrationErrorType,
)
from pquantlib.models.model import CalibratedModel
from pquantlib.models.parameter import ConstantParameter
from pquantlib.models.protocols import (
    CalibrationHelperProtocol,
    ModelProtocol,
    ShortRateModelProtocol,
)
from pquantlib.quotes.simple_quote import SimpleQuote

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


class _MinimalModel(CalibratedModel):
    def __init__(self) -> None:
        super().__init__(1)
        self.arguments[0] = ConstantParameter(0.05, NoConstraint())


class _MinimalHelper(BlackCalibrationHelper):
    @override
    def model_value(self) -> float:
        return 1.0

    @override
    def add_times_to(self, times: list[float]) -> None:
        pass

    @override
    def black_price(self, volatility: float) -> float:
        return volatility * 10.0


class _MinimalShortRate:
    """Hand-rolled class that satisfies ShortRateModelProtocol structurally."""

    def discount(self, t: float) -> float:
        return float(np.exp(-0.05 * t))

    def discount_bond(self, now: float, maturity: float, x: float) -> float:
        return float(np.exp(-0.05 * (maturity - now)))

    def discount_bond_option(
        self,
        option_type: int,
        strike: float,
        maturity: float,
        bond_maturity: float,
    ) -> float:
        return max(0.0, self.discount_bond(0.0, bond_maturity, 0.0) - strike)


# ---------------------------------------------------------------------
# ModelProtocol
# ---------------------------------------------------------------------


def test_calibrated_model_satisfies_model_protocol() -> None:
    m = _MinimalModel()
    assert isinstance(m, ModelProtocol)


def test_arbitrary_object_does_not_satisfy_model_protocol() -> None:
    assert not isinstance(object(), ModelProtocol)


def test_partial_model_does_not_satisfy_protocol() -> None:
    # An object with params() but no set_params should NOT satisfy.
    class _PartialModel:
        def params(self) -> npt.NDArray[np.float64]:
            return np.array([], dtype=np.float64)

    assert not isinstance(_PartialModel(), ModelProtocol)


# ---------------------------------------------------------------------
# CalibrationHelperProtocol
# ---------------------------------------------------------------------


def test_black_calibration_helper_satisfies_protocol() -> None:
    h = _MinimalHelper(SimpleQuote(0.10), CalibrationErrorType.RelativePriceError)
    assert isinstance(h, CalibrationHelperProtocol)


def test_arbitrary_object_does_not_satisfy_helper_protocol() -> None:
    assert not isinstance(object(), CalibrationHelperProtocol)


# ---------------------------------------------------------------------
# ShortRateModelProtocol
# ---------------------------------------------------------------------


def test_minimal_short_rate_satisfies_protocol() -> None:
    m = _MinimalShortRate()
    assert isinstance(m, ShortRateModelProtocol)


def test_arbitrary_object_does_not_satisfy_short_rate_protocol() -> None:
    assert not isinstance(object(), ShortRateModelProtocol)


def test_partial_short_rate_does_not_satisfy_protocol() -> None:
    # Has discount, missing discount_bond + discount_bond_option.
    class _Partial:
        def discount(self, t: float) -> float:
            return 1.0

    assert not isinstance(_Partial(), ShortRateModelProtocol)

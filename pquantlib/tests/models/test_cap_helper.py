"""CapHelper cross-validation vs the C++ probe.

Mirrors the C++ ``cap_helper_roundtrip`` block: builds a 5y CapHelper
on Euribor3M with a 20% Black vol, attaches a BlackCapFloorEngine
with the same vol, verifies market_value == model_value (calibration
identity).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.models.calibration_helper import CalibrationErrorType
from pquantlib.models.cap_helper import CapHelper
from pquantlib.pricingengines.capfloor.black_capfloor_engine import BlackCapFloorEngine
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import tolerance
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_REF_PATH = (
    Path(__file__).resolve().parents[3] / "migration-harness/references/cluster/l4e.json"
)


@pytest.fixture(scope="module")
def cluster_refs() -> dict[str, dict[str, float]]:
    return json.loads(_REF_PATH.read_text())


def _curve() -> YieldTermStructureProtocol:
    eval_date = Date.from_ymd(17, Month.January, 2024)
    return cast(
        YieldTermStructureProtocol,
        FlatForward.from_rate(
            eval_date, 0.05, Actual360(), Compounding.Continuous, Frequency.Annual
        ),
    )


def _make_helper(
    vol_quote: SimpleQuote, curve: YieldTermStructureProtocol
) -> CapHelper:
    idx = Euribor(Period(3, TimeUnit.Months), curve)
    return CapHelper(
        length=Period(5, TimeUnit.Years),
        volatility=vol_quote,
        index=idx,
        fixed_leg_frequency=Frequency.Quarterly,
        fixed_leg_day_counter=Actual360(),
        include_first_swaplet=True,
        term_structure=curve,
        error_type=CalibrationErrorType.PriceError,
    )


def test_cap_helper_market_value_matches_probe(
    cluster_refs: dict[str, dict[str, float]],
) -> None:
    expected = cluster_refs["cap_helper_roundtrip"]
    curve = _curve()
    vol_quote = SimpleQuote(expected["vol"])
    helper = _make_helper(vol_quote, curve)
    tolerance.loose(helper.market_value(), expected["market_value"])


def test_cap_helper_round_trip_zero_calibration_error(
    cluster_refs: dict[str, dict[str, float]],
) -> None:
    """When the helper's engine equals the black engine, calibration_error == 0."""
    expected = cluster_refs["cap_helper_roundtrip"]
    curve = _curve()
    vol_quote = SimpleQuote(expected["vol"])
    helper = _make_helper(vol_quote, curve)
    helper.set_pricing_engine(BlackCapFloorEngine(curve, vol_quote))
    tolerance.loose(helper.model_value(), helper.market_value())
    tolerance.loose(helper.calibration_error(), 0.0)


def test_cap_helper_underlying_cap_lazy_built() -> None:
    curve = _curve()
    vol_quote = SimpleQuote(0.20)
    helper = _make_helper(vol_quote, curve)
    helper.market_value()  # triggers calculate()
    cap = helper.cap()
    # ATM Cap has at least one strike (the fair rate at construction).
    assert len(cap.cap_rates()) >= 1

"""SwaptionHelper cross-validation vs the C++ probe.

Builds the same SwaptionHelper used by the C++ probe
(``swaption_helper_roundtrip`` block) and verifies the market_value
round-trip identity: when the helper's engine and ``black_price``'s
internal engine use the same Black vol, ``market_value()`` ==
``model_value()`` (modulo numerical drift).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.thirty_360 import Convention, Thirty360
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.models.calibration_helper import CalibrationErrorType
from pquantlib.models.swaption_helper import SwaptionHelper
from pquantlib.pricingengines.swaption.black_swaption_engine import BlackSwaptionEngine
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


def _make_helper(
    vol_quote: SimpleQuote, curve: YieldTermStructureProtocol
) -> SwaptionHelper:
    idx = Euribor(Period(3, TimeUnit.Months), curve)
    return SwaptionHelper(
        maturity=Period(5, TimeUnit.Years),
        length=Period(10, TimeUnit.Years),
        volatility=vol_quote,
        index=idx,
        fixed_leg_tenor=Period(6, TimeUnit.Months),
        fixed_leg_day_counter=Thirty360(Convention.BondBasis),
        floating_leg_day_counter=Actual360(),
        term_structure=curve,
        error_type=CalibrationErrorType.PriceError,
        strike=0.03,
        nominal=1.0,
    )


def _curve() -> YieldTermStructureProtocol:
    eval_date = Date.from_ymd(17, Month.January, 2024)
    return cast(
        YieldTermStructureProtocol,
        FlatForward.from_rate(
            eval_date, 0.05, Actual360(), Compounding.Continuous, Frequency.Annual
        ),
    )


def test_swaption_helper_market_value_matches_probe(
    cluster_refs: dict[str, dict[str, float]],
) -> None:
    expected = cluster_refs["swaption_helper_roundtrip"]
    curve = _curve()
    vol_quote = SimpleQuote(expected["vol"])
    helper = _make_helper(vol_quote, curve)
    # No model engine attached — market_value alone is enough.
    tolerance.loose(helper.market_value(), expected["market_value"])


def test_swaption_helper_round_trip_zero_calibration_error(
    cluster_refs: dict[str, dict[str, float]],
) -> None:
    """C++ identity: when the model engine equals the helper's black engine,
    market_value == model_value and calibration_error == 0."""
    expected = cluster_refs["swaption_helper_roundtrip"]
    curve = _curve()
    vol_quote = SimpleQuote(expected["vol"])
    helper = _make_helper(vol_quote, curve)
    helper.set_pricing_engine(BlackSwaptionEngine(curve, vol_quote))
    # Round-trip identity per C++ probe: calibration_error == 0.
    tolerance.loose(helper.model_value(), helper.market_value())
    tolerance.loose(helper.calibration_error(), 0.0)


def test_swaption_helper_underlying_swap_and_swaption_lazy_built() -> None:
    """Until ``calculate()`` runs (via market_value), swap + swaption are None."""
    curve = _curve()
    vol_quote = SimpleQuote(0.20)
    helper = _make_helper(vol_quote, curve)
    helper.market_value()  # triggers calculate()
    swap = helper.underlying_swap()
    swaption = helper.swaption()
    assert swap.fixed_rate() == 0.03
    assert swaption.underlying_swap() is swap

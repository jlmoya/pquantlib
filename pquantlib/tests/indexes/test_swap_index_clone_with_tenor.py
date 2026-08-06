"""SwapIndex.clone_with_tenor, cross-validated against C++ v1.43.

Reference: migration-harness/references/v143/ts/swapindexclone.json
Probe:     migration-harness/cpp/probes/v143_ts_swapindexclone/probe.cpp

C++'s ``SwapIndex::clone(const Period&)`` is what
``Gaussian1dSwaptionVolatility::smileSectionImpl`` uses to honour the requested
swap tenor. The clone keeps every other field and carries the discount curve
across only when the original had an exogenous one.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.indexes.swap.euribor_swap_isda_fix_a import EuriborSwapIsdaFixA
from pquantlib.indexes.swap_index import SwapIndex
from pquantlib.patterns.observable_settings import ObservableSettings
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
    Path(__file__).resolve().parents[3]
    / "migration-harness/references/v143/ts/swapindexclone.json"
)

_EVAL = Date.from_ymd(17, Month.January, 2024)
_FIXING = Date.from_ymd(19, Month.January, 2024)
_Y = TimeUnit.Years
_M = TimeUnit.Months


@pytest.fixture(autouse=True)
def _evaluation_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    settings = ObservableSettings()
    previous = settings.evaluation_date
    settings.evaluation_date = _EVAL
    try:
        yield
    finally:
        settings.evaluation_date = previous


def _curve(rate: float) -> YieldTermStructureProtocol:
    curve = FlatForward.from_rate(
        _EVAL, rate, Actual365Fixed(), Compounding.Continuous, Frequency.Annual
    )
    curve.enable_extrapolation()
    return cast("YieldTermStructureProtocol", curve)


def _indexes() -> dict[str, SwapIndex]:
    base = EuriborSwapIsdaFixA(Period(5, _Y), _curve(0.032))
    exo = EuriborSwapIsdaFixA(Period(5, _Y), _curve(0.032), _curve(0.028))
    return {
        "base_5y": base,
        "clone_10y": base.clone_with_tenor(Period(10, _Y)),
        "clone_1y": base.clone_with_tenor(Period(1, _Y)),
        "clone_18m": base.clone_with_tenor(Period(18, _M)),
        "exo_base_5y": exo,
        "exo_clone_10y": exo.clone_with_tenor(Period(10, _Y)),
    }


@pytest.fixture(scope="module")
def refs() -> dict[str, dict[str, object]]:
    raw = cast("dict[str, object]", json.loads(_REF_PATH.read_text()))
    return {k: cast("dict[str, object]", v) for k, v in raw.items()}


def test_every_probe_case_is_covered(refs: dict[str, dict[str, object]]) -> None:
    assert set(_indexes()) == set(refs)


@pytest.mark.parametrize(
    "key",
    ["base_5y", "clone_10y", "clone_1y", "clone_18m", "exo_base_5y", "exo_clone_10y"],
)
def test_clone_metadata_matches_cpp(
    refs: dict[str, dict[str, object]], key: str
) -> None:
    idx = _indexes()[key]
    block = refs[key]
    assert idx.name() == block["name"], key
    assert idx.tenor().length == block["tenor_length"], key
    assert int(idx.tenor().units) == block["tenor_units"], key
    assert idx.fixing_days() == block["fixing_days"], key
    assert idx.currency().code == block["currency_code"], key
    assert idx.fixing_calendar().name() == block["fixing_calendar"], key
    assert idx.fixed_leg_tenor().length == block["fixed_leg_tenor_length"], key
    assert int(idx.fixed_leg_convention()) == block["fixed_leg_convention"], key
    assert idx.fixed_leg_day_counter().name() == block["day_counter"], key
    assert idx.exogenous_discount() == block["exogenous_discount"], key


@pytest.mark.parametrize(
    "key",
    ["base_5y", "clone_10y", "clone_1y", "clone_18m", "exo_base_5y", "exo_clone_10y"],
)
def test_clone_underlying_swap_matches_cpp(
    refs: dict[str, dict[str, object]], key: str
) -> None:
    swap = _indexes()[key].underlying_swap(_FIXING)
    block = refs[key]
    assert swap.start_date().serial == block["swap_start"], key
    assert swap.maturity_date().serial == block["swap_maturity"], key
    tolerance.tight(swap.fair_rate(), cast("float", block["swap_fair_rate"]), reason=key)


def test_non_exogenous_clone_does_not_acquire_a_discount_curve(
    refs: dict[str, dict[str, object]],
) -> None:
    """The branch a naive clone gets wrong.

    Carrying ``self._discount`` across unconditionally would give the clone of
    a non-exogenous index an exogenous discount curve, changing how the
    underlying swap is priced.
    """
    assert refs["clone_10y"]["exogenous_discount"] is False
    assert refs["exo_clone_10y"]["exogenous_discount"] is True
    indexes = _indexes()
    assert indexes["clone_10y"].exogenous_discount() is False
    assert indexes["clone_10y"].discounting_term_structure() is None
    assert indexes["exo_clone_10y"].exogenous_discount() is True
    assert indexes["exo_clone_10y"].discounting_term_structure() is not None

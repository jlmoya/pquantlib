"""Tests for OneAssetOption + OneAssetOptionResults."""

from __future__ import annotations

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.one_asset_option import (
    OneAssetOption,
    OneAssetOptionResults,
)
from pquantlib.option import OptionArguments
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.time.date import Date
from pquantlib.time.month import Month


class _Engine(GenericEngine[OptionArguments, OneAssetOptionResults]):
    """Engine that fills NPV + delta + gamma + vega + theta + rho."""

    def __init__(self) -> None:
        super().__init__(OptionArguments(), OneAssetOptionResults())

    def calculate(self) -> None:
        self._results.value = 7.5
        self._results.error_estimate = 0.0
        self._results.valuation_date = Date.from_ymd(15, Month.June, 2026)
        self._results.delta = 0.55
        self._results.gamma = 0.02
        self._results.theta = -0.04
        self._results.vega = 0.40
        self._results.rho = 0.50
        self._results.dividend_rho = -0.25
        self._results.itm_cash_probability = 0.45


class _Option(OneAssetOption):
    """Concrete OneAssetOption: never expired."""

    def is_expired(self) -> bool:
        return False


def _payoff() -> PlainVanillaPayoff:
    return PlainVanillaPayoff(OptionType.Call, 100.0)


def _exercise() -> EuropeanExercise:
    return EuropeanExercise(Date.from_ymd(15, Month.June, 2027))


def test_cannot_instantiate_one_asset_option_directly() -> None:
    with pytest.raises(TypeError):
        OneAssetOption(_payoff(), _exercise())  # type: ignore[abstract]


def test_npv_and_greeks_after_engine_calculate() -> None:
    opt = _Option(_payoff(), _exercise())
    opt.set_pricing_engine(_Engine())
    assert opt.npv() == 7.5
    assert opt.delta() == 0.55
    assert opt.gamma() == 0.02
    assert opt.theta() == -0.04
    assert opt.vega() == 0.40
    assert opt.rho() == 0.50
    assert opt.dividend_rho() == -0.25
    assert opt.itm_cash_probability() == 0.45


def test_greek_not_provided_raises() -> None:
    """An engine that doesn't fill rho should raise on accessor call."""

    class _PartialEngine(GenericEngine[OptionArguments, OneAssetOptionResults]):
        def __init__(self) -> None:
            super().__init__(OptionArguments(), OneAssetOptionResults())

        def calculate(self) -> None:
            self._results.value = 7.5
            self._results.error_estimate = 0.0
            self._results.valuation_date = Date.from_ymd(15, Month.June, 2026)
            self._results.delta = 0.55
            # rho intentionally left None

    opt = _Option(_payoff(), _exercise())
    opt.set_pricing_engine(_PartialEngine())
    assert opt.npv() == 7.5
    assert opt.delta() == 0.55
    with pytest.raises(LibraryException, match="rho not provided"):
        opt.rho()


def test_one_asset_option_results_reset_clears_fields() -> None:
    r = OneAssetOptionResults()
    r.value = 10.0
    r.delta = 0.5
    r.itm_cash_probability = 0.3
    r.reset()
    assert r.value is None
    assert r.delta is None
    assert r.itm_cash_probability is None

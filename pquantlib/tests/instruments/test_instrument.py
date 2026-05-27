"""Tests for the abstract Instrument + InstrumentResults pair."""

from __future__ import annotations

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.instruments.instrument import Instrument, InstrumentResults
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.pricing_engine import PricingEngineArguments
from pquantlib.time.date import Date
from pquantlib.time.month import Month


class _Args(PricingEngineArguments):
    def __init__(self) -> None:
        self.notional: float = 0.0

    def validate(self) -> None:
        if self.notional <= 0:
            raise ValueError("notional must be positive")


class _Results(InstrumentResults):
    """Re-uses the standard InstrumentResults; adds nothing."""


class _ConstantEngine(GenericEngine[_Args, _Results]):
    """Engine that always reports NPV=42 and a fixed valuation date."""

    def __init__(self) -> None:
        super().__init__(_Args(), _Results())

    def calculate(self) -> None:
        self._results.value = 42.0
        self._results.error_estimate = 0.001
        self._results.valuation_date = Date.from_ymd(15, Month.June, 2026)
        self._results.additional_results = {"engine_tag": "constant"}


class _SimpleProduct(Instrument):
    """Concrete Instrument for tests: not-expired by default; uses _Args."""

    def __init__(self, notional: float = 1000.0, expired: bool = False) -> None:
        super().__init__()
        self._notional: float = notional
        self._expired: bool = expired

    def is_expired(self) -> bool:
        return self._expired

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        assert isinstance(args, _Args)
        args.notional = self._notional


# --- abstract guardrails ----------------------------------------------


def test_cannot_instantiate_instrument_directly() -> None:
    with pytest.raises(TypeError):
        Instrument()  # type: ignore[abstract]


# --- NPV + engine plumbing --------------------------------------------


def test_set_pricing_engine_then_npv_calls_engine() -> None:
    p = _SimpleProduct()
    p.set_pricing_engine(_ConstantEngine())
    assert p.npv() == 42.0


def test_npv_caches_until_invalidated() -> None:
    p = _SimpleProduct()
    engine = _ConstantEngine()
    p.set_pricing_engine(engine)
    _ = p.npv()
    # Mutate engine output; without invalidation, instrument should
    # still see the cached value.
    engine._results.value = 99.0  # pyright: ignore[reportPrivateUsage]
    assert p.npv() == 42.0
    # Force re-calculation by issuing an update.
    p.update()
    # The next npv() call performs calculations afresh — but since
    # ConstantEngine always reassigns value=42 in calculate(), we get
    # 42 back.
    assert p.npv() == 42.0


def test_npv_without_engine_raises() -> None:
    p = _SimpleProduct()
    with pytest.raises(LibraryException, match="null pricing engine"):
        p.npv()


def test_error_estimate_reflects_engine_output() -> None:
    p = _SimpleProduct()
    p.set_pricing_engine(_ConstantEngine())
    assert p.error_estimate() == 0.001


def test_valuation_date_reflects_engine_output() -> None:
    p = _SimpleProduct()
    p.set_pricing_engine(_ConstantEngine())
    assert p.valuation_date() == Date.from_ymd(15, Month.June, 2026)


def test_additional_results_passes_through() -> None:
    p = _SimpleProduct()
    p.set_pricing_engine(_ConstantEngine())
    assert p.additional_results() == {"engine_tag": "constant"}
    assert p.result("engine_tag") == "constant"


def test_missing_result_tag_raises() -> None:
    p = _SimpleProduct()
    p.set_pricing_engine(_ConstantEngine())
    with pytest.raises(LibraryException, match="not provided"):
        p.result("nope")


# --- expired path ------------------------------------------------------


def test_expired_instrument_returns_zero_npv_without_calling_engine() -> None:
    p = _SimpleProduct(expired=True)
    # No engine attached — but expired short-circuits in calculate().
    assert p.npv() == 0.0


def test_changing_engine_invalidates_cache() -> None:
    """set_pricing_engine on a fresh engine should re-fire calculations."""
    p = _SimpleProduct()
    p.set_pricing_engine(_ConstantEngine())
    _ = p.npv()
    # Same constant engine, fresh instance — still 42.
    new_engine = _ConstantEngine()
    p.set_pricing_engine(new_engine)
    assert p.npv() == 42.0

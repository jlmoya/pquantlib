"""Tests for PricingEngine / PricingEngineArguments / PricingEngineResults +
GenericEngine[ArgsT, ResultsT]."""

from __future__ import annotations

import pytest

from pquantlib.patterns.observer import Observable
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.pricing_engine import (
    PricingEngine,
    PricingEngineArguments,
    PricingEngineResults,
)


class _Args(PricingEngineArguments):
    """Simple args carrier: one number, validates positivity."""

    def __init__(self) -> None:
        self.x: float | None = None

    def validate(self) -> None:
        if self.x is None or self.x <= 0:
            raise ValueError("x must be positive")


class _Results(PricingEngineResults):
    """Simple results carrier: one number."""

    def __init__(self) -> None:
        self.y: float | None = None

    def reset(self) -> None:
        self.y = None


class _DoublingEngine(GenericEngine[_Args, _Results]):
    """Trivial engine: results.y = 2 * arguments.x."""

    def __init__(self) -> None:
        super().__init__(_Args(), _Results())

    def calculate(self) -> None:
        assert self._arguments.x is not None
        self._results.y = 2.0 * self._arguments.x


# --- abstract guardrails ----------------------------------------------


def test_cannot_instantiate_pricing_engine_directly() -> None:
    with pytest.raises(TypeError):
        PricingEngine()  # type: ignore[abstract]


def test_cannot_instantiate_arguments_directly() -> None:
    with pytest.raises(TypeError):
        PricingEngineArguments()  # type: ignore[abstract]


def test_cannot_instantiate_results_directly() -> None:
    with pytest.raises(TypeError):
        PricingEngineResults()  # type: ignore[abstract]


def test_cannot_instantiate_generic_engine_directly() -> None:
    """GenericEngine has abstract ``calculate()`` so direct instantiation fails."""
    with pytest.raises(TypeError):
        GenericEngine(_Args(), _Results())  # type: ignore[abstract]


# --- GenericEngine plumbing -------------------------------------------


def test_generic_engine_owns_arguments_and_results() -> None:
    e = _DoublingEngine()
    assert isinstance(e.get_arguments(), _Args)
    assert isinstance(e.get_results(), _Results)


def test_generic_engine_calculate_runs() -> None:
    e = _DoublingEngine()
    e.get_arguments().x = 5.0
    e.calculate()
    assert e.get_results().y == 10.0


def test_generic_engine_reset_clears_results() -> None:
    e = _DoublingEngine()
    e.get_arguments().x = 5.0
    e.calculate()
    e.reset()
    assert e.get_results().y is None


def test_generic_engine_is_observable() -> None:
    e = _DoublingEngine()
    assert isinstance(e, Observable)


def test_generic_engine_update_notifies_observers() -> None:
    e = _DoublingEngine()
    counts = [0]

    class _Counter:
        def update(self) -> None:
            counts[0] += 1

    obs = _Counter()
    e.register_with(obs)
    e.update()
    assert counts[0] == 1

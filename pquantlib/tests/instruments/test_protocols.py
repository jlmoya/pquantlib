"""Tests for the cross-cluster runtime-checkable Protocols."""

from __future__ import annotations

from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.protocols import (
    InstrumentProtocol,
    PricingEngineProtocol,
    StochasticProcessProtocol,
)
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.pricing_engine import (
    PricingEngineArguments,
    PricingEngineResults,
)
from pquantlib.time.date import Date
from pquantlib.time.month import Month


class _Args(PricingEngineArguments):
    def validate(self) -> None:
        pass


class _Results(PricingEngineResults):
    def reset(self) -> None:
        pass


class _Engine(GenericEngine[_Args, _Results]):
    def __init__(self) -> None:
        super().__init__(_Args(), _Results())

    def calculate(self) -> None:
        pass


def test_pricing_engine_protocol_recognizes_generic_engine() -> None:
    """A concrete GenericEngine instance must satisfy PricingEngineProtocol."""
    e = _Engine()
    assert isinstance(e, PricingEngineProtocol)


# --- StochasticProcessProtocol -----------------------------------------


class _Process:
    """Duck-typed StochasticProcess for the Protocol test.

    Concrete process classes will land in L3-D under pquantlib.processes.*;
    we only need to confirm the Protocol matches structurally.
    """

    def initial_values(self) -> list[float]:
        return [100.0]

    def drift(self, t: float, x: list[float]) -> list[float]:
        return [0.0]

    def diffusion(self, t: float, x: list[float]) -> list[float]:
        return [0.2]

    def evolve(
        self, t0: float, x0: list[float], dt: float, dw: list[float]
    ) -> list[float]:
        return [x0[0] + dw[0]]


def test_stochastic_process_protocol_matches_duck_typed_class() -> None:
    p = _Process()
    assert isinstance(p, StochasticProcessProtocol)


# --- InstrumentProtocol -----------------------------------------------


class _Trade:
    """Duck-typed instrument."""

    def npv(self) -> float:
        return 100.0

    def is_expired(self) -> bool:
        return False

    def valuation_date(self) -> Date:
        return Date.from_ymd(15, Month.June, 2026)


def test_instrument_protocol_matches_duck_typed_class() -> None:
    t = _Trade()
    assert isinstance(t, InstrumentProtocol)


def test_instrument_protocol_rejects_class_without_npv() -> None:
    class _BadTrade:
        def is_expired(self) -> bool:
            return False

        def valuation_date(self) -> Date:
            return Date.from_ymd(15, Month.June, 2026)

    bt = _BadTrade()
    assert not isinstance(bt, InstrumentProtocol)


# --- Payoff/Exercise duck-typed accessors --------------------------


def test_payoff_and_exercise_are_usable_without_an_engine() -> None:
    """Smoke: the L3-A Stage 6 surface lets you build an option's data
    without setting a pricing engine first."""
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(Date.from_ymd(15, Month.June, 2027))
    # Both objects expose their data; this is the L3-D / L3-B entry point.
    assert payoff(120.0) == 20.0
    assert exercise.last_date() == Date.from_ymd(15, Month.June, 2027)

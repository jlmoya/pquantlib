"""Cross-validate EverestOption + MCEverestEngine against C++.

Probe source: migration-harness/cpp/probes/cluster_w4a/probe.cpp
Reference:    migration-harness/references/cluster/w4a.json
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.exercise import EuropeanExercise
from pquantlib.experimental.exoticoptions.everest_option import (
    EverestOption,
    EverestOptionArguments,
)
from pquantlib.payoffs import NullPayoff
from pquantlib.pricingengines.exoticoptions.mc_everest_engine import (
    MCEverestEngine,
)
from pquantlib.processes.stochastic_process_array import StochasticProcessArray
from pquantlib.testing import reference_reader
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.time_unit import TimeUnit


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/w4a")


@pytest.fixture(scope="module")
def exercise(today: Date, calendar: Calendar) -> EuropeanExercise:
    return EuropeanExercise(calendar.advance(today, 1, TimeUnit.Years))


def test_everest_construction_and_inspectors(exercise: EuropeanExercise) -> None:
    """Construction round-trip."""
    opt = EverestOption(notional=1.0e6, guarantee=0.03, exercise=exercise)
    assert opt.notional() == 1.0e6
    assert opt.guarantee() == 0.03
    assert opt.exercise() is exercise
    # The payoff is a NullPayoff (engine bypasses ``args.payoff``).
    assert isinstance(opt.payoff(), NullPayoff)


def test_everest_arguments_validate(exercise: EuropeanExercise) -> None:
    """``validate()`` requires notional + guarantee + nonzero notional."""
    args = EverestOptionArguments()
    args.payoff = NullPayoff()
    args.exercise = exercise
    args.notional = None
    args.guarantee = 0.03
    with pytest.raises(LibraryException):
        args.validate()
    args.notional = 0.0
    with pytest.raises(LibraryException):
        args.validate()
    args.notional = 1.0e6
    args.guarantee = None
    with pytest.raises(LibraryException):
        args.validate()
    args.guarantee = 0.03
    args.validate()  # no raise


def test_everest_mc_npv_within_three_sigma(
    cpp_ref: dict[str, Any],
    basket3: StochasticProcessArray,
    exercise: EuropeanExercise,
) -> None:
    """MC NPV agrees with C++ within 3 sigma.

    The Everest payoff is bounded above (``min_yield + guarantee``
    can be very negative for any one bad asset), so variance is high.
    Cross-impl divergence from the spectral-square-root path
    redirection of correlated dW is captured at 3 sigma — see
    ``test_himalaya_option`` docstring for the full justification.
    """
    ref = cpp_ref["everest"]
    cpp_npv = ref["npv"]

    opt = EverestOption(
        notional=ref["notional"],
        guarantee=ref["guarantee"],
        exercise=exercise,
    )
    eng = MCEverestEngine(
        basket3,
        time_steps_per_year=12,
        required_samples=1023,
        seed=42,
    )
    opt.set_pricing_engine(eng)
    py_npv = opt.npv()
    sigma = eng.error_estimate()
    assert abs(py_npv - cpp_npv) < 3.0 * sigma, (
        f"Everest NPV mismatch: py={py_npv}, cpp={cpp_npv}, sigma={sigma}"
    )

    # Yield should also be set on the option's results.
    py_yield = opt.yield_()
    cpp_yield = ref["yield"]
    # yield = npv / (notional * discount) - 1 — same relative error.
    yield_sigma = sigma / (ref["notional"])  # discount factor ≈ 1, conservative
    assert abs(py_yield - cpp_yield) < 3.0 * yield_sigma + 1e-3, (
        f"Everest yield mismatch: py={py_yield}, cpp={cpp_yield}"
    )

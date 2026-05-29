"""Cross-validate PagodaOption + MCPagodaEngine against C++.

Probe source: migration-harness/cpp/probes/cluster_w4a/probe.cpp
Reference:    migration-harness/references/cluster/w4a.json
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.exercise import EuropeanExercise
from pquantlib.experimental.exoticoptions.pagoda_option import (
    PagodaOption,
    PagodaOptionArguments,
)
from pquantlib.payoffs import NullPayoff
from pquantlib.pricingengines.exoticoptions.mc_pagoda_engine import (
    MCPagodaEngine,
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
def fixings(today: Date, calendar: Calendar) -> list[Date]:
    return [
        calendar.advance(today, 3, TimeUnit.Months),
        calendar.advance(today, 6, TimeUnit.Months),
        calendar.advance(today, 9, TimeUnit.Months),
        calendar.advance(today, 12, TimeUnit.Months),
    ]


def test_pagoda_construction_and_inspectors(fixings: list[Date]) -> None:
    """Construction round-trip: roof, fraction, fixing dates."""
    opt = PagodaOption(fixings, roof=0.20, fraction=0.50)
    assert opt.roof() == 0.20
    assert opt.fraction() == 0.50
    assert opt.fixing_dates() == fixings
    # NullPayoff (engine bypasses).
    assert isinstance(opt.payoff(), NullPayoff)
    # European exercise at last fixing.
    assert opt.exercise().last_date() == fixings[-1]


def test_pagoda_empty_fixings_raises() -> None:
    with pytest.raises(LibraryException):
        PagodaOption([], roof=0.20, fraction=0.50)


def test_pagoda_arguments_validate(fixings: list[Date]) -> None:
    """``validate()`` requires fixing dates + roof + fraction."""
    args = PagodaOptionArguments()
    args.payoff = NullPayoff()
    args.exercise = EuropeanExercise(fixings[-1])
    args.fixing_dates = []
    args.roof = 0.20
    args.fraction = 0.50
    with pytest.raises(LibraryException):
        args.validate()
    args.fixing_dates = list(fixings)
    args.roof = None
    with pytest.raises(LibraryException):
        args.validate()
    args.roof = 0.20
    args.fraction = None
    with pytest.raises(LibraryException):
        args.validate()
    args.fraction = 0.50
    args.validate()  # no raise


def test_pagoda_mc_npv_within_three_sigma(
    cpp_ref: dict[str, Any],
    basket3: StochasticProcessArray,
    fixings: list[Date],
) -> None:
    """MC NPV within 3 sigma of cross-impl noise — see test_himalaya
    docstring for justification.
    """
    ref = cpp_ref["pagoda"]
    cpp_npv = ref["npv"]

    opt = PagodaOption(fixings, roof=ref["roof"], fraction=ref["fraction"])
    eng = MCPagodaEngine(basket3, required_samples=1023, seed=42)
    opt.set_pricing_engine(eng)
    py_npv = opt.npv()
    sigma = eng.error_estimate()
    assert abs(py_npv - cpp_npv) < 3.0 * sigma, (
        f"Pagoda NPV mismatch: py={py_npv}, cpp={cpp_npv}, sigma={sigma}"
    )

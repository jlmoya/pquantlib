"""Cross-validate HimalayaOption + MCHimalayaEngine against C++.

Probe source: migration-harness/cpp/probes/cluster_w4a/probe.cpp
Reference:    migration-harness/references/cluster/w4a.json
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.exercise import EuropeanExercise
from pquantlib.experimental.exoticoptions.himalaya_option import (
    HimalayaOption,
    HimalayaOptionArguments,
)
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.exoticoptions.mc_himalaya_engine import (
    MCHimalayaEngine,
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
        calendar.advance(today, 4, TimeUnit.Months),
        calendar.advance(today, 8, TimeUnit.Months),
        calendar.advance(today, 12, TimeUnit.Months),
    ]


def test_himalaya_construction_and_inspectors(fixings: list[Date]) -> None:
    """Construction round-trip: strike, fixing dates, exercise mapping."""
    opt = HimalayaOption(fixings, 100.0)
    # Strike is carried inside the PlainVanillaPayoff; check via payoff.
    payoff = opt.payoff()
    assert isinstance(payoff, PlainVanillaPayoff)
    assert payoff.strike() == 100.0
    # Exercise is European @ last fixing.
    assert opt.exercise().last_date() == fixings[-1]
    assert opt.fixing_dates() == fixings


def test_himalaya_empty_fixings_raises() -> None:
    """Empty ``fixing_dates`` raises (matches C++ validate)."""
    with pytest.raises(LibraryException):
        HimalayaOption([], 100.0)


def test_himalaya_arguments_validate(fixings: list[Date]) -> None:
    """Arguments validate require fixing dates."""
    args = HimalayaOptionArguments()
    args.payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    args.exercise = EuropeanExercise(fixings[-1])
    args.fixing_dates = []
    with pytest.raises(LibraryException):
        args.validate()
    # Non-empty list passes.
    args.fixing_dates = list(fixings)
    args.validate()  # no raise


def test_himalaya_mc_npv_within_three_sigma(
    cpp_ref: dict[str, Any],
    basket3: StochasticProcessArray,
    fixings: list[Date],
) -> None:
    """MC NPV agrees with C++ within 3 sigma of the engine's error estimate.

    The C++ reference uses ``PseudoRandom`` (MT19937 + Acklam
    InvCumulativeNormal); the Python port shares the same building
    blocks, so the marginal Gaussian sequence matches bit-for-bit.

    However ``StochasticProcessArray`` premultiplies by the spectral
    square root of the correlation matrix. C++ uses Jacobi
    ``SymmetricSchurDecomposition`` whose eigenvector basis differs
    in column ordering / signs from numpy's ``np.linalg.eigh`` —
    paths are statistically equivalent but pathwise distinct.

    The cross-impl test therefore compares NPVs at the engine's
    own 1-sigma error scale, with a 3-sigma allowance (P(|X| > 3 sigma)
    ~= 0.27% — well below the noise floor of MC tests).
    """
    ref = cpp_ref["himalaya"]
    cpp_npv = ref["npv"]

    opt = HimalayaOption(fixings, ref["strike"])
    eng = MCHimalayaEngine(
        basket3,
        required_samples=1023,
        seed=42,
    )
    opt.set_pricing_engine(eng)
    py_npv = opt.npv()
    sigma = eng.error_estimate()
    assert abs(py_npv - cpp_npv) < 3.0 * sigma, (
        f"Himalaya NPV mismatch: py={py_npv}, cpp={cpp_npv}, sigma={sigma}"
    )

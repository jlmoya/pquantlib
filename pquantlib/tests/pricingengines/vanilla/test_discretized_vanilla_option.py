"""Cross-validate :class:`DiscretizedVanillaOption` against C++ v1.43.

Reference: ``migration-harness/references/v143/pe/american`` (produced by
``migration-harness/cpp/probes/v143_pe_american/probe.cpp``), cases prefixed
``dvo_``.

:class:`DiscretizedVanillaOption` is a lattice *asset*, not an engine, so it
is driven here exactly the way ``BinomialVanillaEngine<T>`` drives it in C++:
build a :class:`CoxRossRubinstein` tree over a flattened constant-coefficient
market, wrap it in a :class:`BlackScholesLattice`, ``initialize`` at the
maturity, then perform Hull's three-step rollback — to ``grid[2]``
(3 nodes), to ``grid[1]`` (2 nodes), and finally to 0.

The node arrays at each stop are asserted, not just the final present value.
A port can arrive at the right price with a wrong intermediate slice — for
example by applying the exercise condition one slice early — and pinning
only the PV would let that through.  The underlying prices at those slices
are pinned too, so the tree geometry is fixed independently of the values.

All three arms of ``postAdjustValuesImpl`` are exercised, and they are
genuinely different tests, not three spellings of one:

* **American** uses a *range* test, ``stopping[0] <= now <= stopping[1]``,
  and indexes ``stopping[1]`` — so the exercise must carry two dates and
  every slice inside the window is exercisable;
* **European** uses ``is_on_time(stopping[0])`` — only the maturity slice,
  which is why ``reset``'s trailing ``adjust_values()`` is what makes a
  European option worth its payoff at maturity rather than zero;
* **Bermudan** loops ``is_on_time`` over every stopping time.

The Bermudan case deliberately uses exercise dates at 97 / 201 / 365 days on
a 50-step grid, none of which land on a grid point.  The constructor's
``grid.closest_time(...)`` snapping is therefore observable: the reference
pins ``mandatory_times`` as ``[0.26, 0.56, 1.0]`` rather than the raw
``[0.2658, 0.5507, 1.0]``, so a port that skipped the snap fails on the
mandatory times *and* on the values.

Tolerance is TIGHT.  A 50-step binomial rollback is ~1300 fused
multiply-adds evaluated in the same order on both sides; the whole table
comes out to <= 4e-16 relative, which is round-off and nothing else.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.exercise import AmericanExercise, BermudanExercise, EuropeanExercise, Exercise
from pquantlib.methods.lattices.binomial_tree import CoxRossRubinstein
from pquantlib.methods.lattices.bsm_lattice import BlackScholesLattice
from pquantlib.option import OptionArguments
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import PlainVanillaPayoff
from pquantlib.pricingengines.vanilla.discretized_vanilla_option import (
    DiscretizedVanillaOption,
)
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import BlackConstantVol
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency
from pquantlib.time.time_grid import TimeGrid

from ._american_v143 import CAL, DC, TODAY, market, option_type

CPP: dict[str, Any] = reference_reader.load("v143/pe/american")

_DVO_CASES = [name for name in CPP if name.startswith("dvo_")]


@pytest.fixture(autouse=True)
def _eval_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    settings = ObservableSettings()
    saved = settings.evaluation_date
    # probe.cpp — ``Settings::instance().evaluationDate() = TODAY;`` in main(),
    # with ``const Date TODAY(1, March, 2025);``.
    settings.evaluation_date = TODAY
    yield
    settings.evaluation_date = saved


def _build_exercise(inputs: dict[str, Any]) -> Exercise:
    ex_date = TODAY + int(inputs["maturity_days"])
    kind = inputs["exercise"]
    if kind == "European":
        return EuropeanExercise(ex_date)
    if kind == "Bermudan":
        return BermudanExercise([TODAY + int(d) for d in inputs["bermudan_offset_days"]])
    return AmericanExercise(TODAY, ex_date)


def _assert_array(actual: Sequence[float], expected: list[Any], label: str) -> None:
    assert len(actual) == len(expected), f"{label}: length {len(actual)} != {len(expected)}"
    for j, (got, want) in enumerate(zip(actual, expected, strict=True)):
        tolerance.tight(float(got), float(want), reason=f"{label}[{j}]")


def test_case_table_is_populated() -> None:
    """Guard against a reference that silently lost its cases."""
    assert len(_DVO_CASES) >= 6


@pytest.mark.tight
@pytest.mark.parametrize("case", _DVO_CASES)
def test_discretized_vanilla_option(case: str) -> None:
    """Reproduce the Hull three-step rollback slice by slice."""
    record = CPP[case]
    inputs = record["inputs"]
    expected = record["expected"]
    assert expected["throws"] is False

    spot = float(inputs["spot"])
    strike = float(inputs["strike"])
    steps = int(inputs["time_steps"])
    process = market(spot, float(inputs["q"]), float(inputs["r"]), float(inputs["vol"]))
    ex_date = TODAY + int(inputs["maturity_days"])

    args = OptionArguments()
    args.payoff = PlainVanillaPayoff(option_type(inputs["type"]), strike)
    args.exercise = _build_exercise(inputs)

    maturity = DC.year_fraction(TODAY, ex_date)
    tolerance.tight(maturity, float(expected["maturity"]), reason=f"{case} maturity")

    # Same flat-coefficient reconstruction the C++ binomial engine performs.
    flat_r = process.risk_free_rate().zero_rate(
        ex_date,
        compounding=Compounding.Continuous,
        frequency=Frequency.NoFrequency,
        result_day_counter=DC,
    ).rate()
    flat_q = process.dividend_yield().zero_rate(
        ex_date,
        compounding=Compounding.Continuous,
        frequency=Frequency.NoFrequency,
        result_day_counter=DC,
    ).rate()
    flat_vol = process.black_volatility().black_vol(ex_date, spot)
    tolerance.tight(flat_r, float(expected["flat_r"]), reason=f"{case} flat r")
    tolerance.tight(flat_q, float(expected["flat_q"]), reason=f"{case} flat q")
    tolerance.tight(flat_vol, float(expected["flat_vol"]), reason=f"{case} flat vol")

    flat_process = GeneralizedBlackScholesProcess(
        x0=SimpleQuote(spot),
        dividend_ts=FlatForward.from_rate(
            reference_date=TODAY, forward_rate=flat_q, day_counter=DC
        ),
        risk_free_ts=FlatForward.from_rate(
            reference_date=TODAY, forward_rate=flat_r, day_counter=DC
        ),
        black_vol_ts=BlackConstantVol(
            reference_date=TODAY,
            calendar=NullCalendar(),
            day_counter=DC,
            volatility=flat_vol,
        ),
    )

    grid = TimeGrid.regular(end=maturity, steps=steps)
    tree = CoxRossRubinstein(flat_process, maturity, steps, strike)
    lattice = BlackScholesLattice(tree, flat_r, maturity, steps)

    asset = DiscretizedVanillaOption(args, process, grid)
    asset.initialize(lattice, maturity)

    _assert_array(
        asset.mandatory_times(), expected["mandatory_times"], f"{case} mandatory_times"
    )

    asset.rollback(grid[2])
    _assert_array(list(asset.values), expected["values_at_step2"], f"{case} values@2")
    _assert_array(
        [lattice.underlying(2, j) for j in range(len(asset.values))],
        expected["underlying_at_step2"],
        f"{case} underlying@2",
    )

    asset.rollback(grid[1])
    _assert_array(list(asset.values), expected["values_at_step1"], f"{case} values@1")
    _assert_array(
        [lattice.underlying(1, j) for j in range(len(asset.values))],
        expected["underlying_at_step1"],
        f"{case} underlying@1",
    )

    asset.rollback(0.0)
    tolerance.tight(
        asset.present_value(), float(expected["present_value"]), reason=f"{case} PV"
    )


def test_bermudan_stopping_times_are_snapped_to_the_grid() -> None:
    """The ctor's ``grid.closest_time`` is observable, and pinned.

    97 / 201 / 365 days on a 50-step one-year grid are 0.26575 / 0.55068 /
    1.0 in year fractions; the grid points are multiples of 0.02, so the
    first two snap to 0.26 and 0.56.
    """
    expected = CPP["dvo_bermudan_put"]["expected"]["mandatory_times"]
    assert [round(t, 10) for t in expected] == [0.26, 0.56, 1.0]

    inputs = CPP["dvo_bermudan_put"]["inputs"]
    raw = [d / 365.0 for d in inputs["bermudan_offset_days"]]
    assert raw != expected, "the exercise dates must NOT already sit on the grid"


def test_american_is_worth_more_than_european() -> None:
    """The range test really does add early-exercise value for a put."""
    american = CPP["dvo_american_put"]["expected"]["present_value"]
    european = CPP["dvo_european_put"]["expected"]["present_value"]
    assert american > european


def test_bermudan_sits_between_european_and_american() -> None:
    """Three exercise dates buy less optionality than continuous exercise."""
    american = CPP["dvo_american_put"]["expected"]["present_value"]
    bermudan = CPP["dvo_bermudan_put"]["expected"]["present_value"]
    european = CPP["dvo_european_put"]["expected"]["present_value"]
    assert european < bermudan < american


def test_constructor_requires_payoff_and_exercise() -> None:
    """``args.exercise`` is dereferenced immediately; a bare bundle must fail."""
    process = market(100.0, 0.02, 0.05, 0.25)
    empty = OptionArguments()
    with pytest.raises(LibraryException):
        DiscretizedVanillaOption(empty, process)

    payoff_only = OptionArguments()
    payoff_only.payoff = PlainVanillaPayoff(option_type("Call"), 100.0)
    with pytest.raises(LibraryException):
        DiscretizedVanillaOption(payoff_only, process)


def test_no_grid_means_no_snapping() -> None:
    """C++'s default ``TimeGrid()`` is the empty grid: stopping times pass through."""
    process = market(100.0, 0.02, 0.05, 0.25)
    args = OptionArguments()
    args.payoff = PlainVanillaPayoff(option_type("Put"), 100.0)
    args.exercise = BermudanExercise([TODAY + 97, TODAY + 201, TODAY + 365])
    asset = DiscretizedVanillaOption(args, process)
    raw = [process.time(d) for d in args.exercise.dates()]
    assert asset.mandatory_times() == raw


def test_calendar_and_daycount_conventions_match_the_probe() -> None:
    """Belt and braces: the probe's conventions are what this module rebuilds."""
    assert isinstance(CAL, NullCalendar)
    assert DC.name() == "Actual/365 (Fixed)"

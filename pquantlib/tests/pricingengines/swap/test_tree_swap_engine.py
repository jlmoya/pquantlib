"""TreeVanillaSwapEngine + its two template base classes, against C++ v1.43.

Reference: ``migration-harness/references/v143/pe/bondswap.json`` (probe
``migration-harness/cpp/probes/v143_pe_bondswap/probe.cpp``, "PART 5").

``TreeVanillaSwapEngine`` is the only concrete engine deriving straight from
``LatticeShortRateModelEngine``, which derives from ``GenericModelEngine``, so
this module doubles as the behaviour test for both bases: it pins the two
constructor flavours separately (a step budget vs an explicit ``TimeGrid``),
proves the ``TimeGrid`` flavour builds its lattice eagerly and rebuilds it on
notification, and pins the ``timeSteps > 0`` guard that lives in the base.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.thirty_360 import Convention, Thirty360
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.ibor.euribor import Euribor6M
from pquantlib.instruments.fixed_vs_floating_swap import FixedVsFloatingSwapArguments
from pquantlib.instruments.swap import SwapType
from pquantlib.instruments.vanilla_swap import VanillaSwap
from pquantlib.methods.lattices.discretized_swap import DiscretizedSwap
from pquantlib.models.shortrate.onefactor.hull_white import HullWhite
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.generic_model_engine import GenericModelEngine
from pquantlib.pricingengines.lattice_short_rate_model_engine import (
    LatticeShortRateModelEngine,
)
from pquantlib.pricingengines.swap.discounting_swap_engine import DiscountingSwapEngine
from pquantlib.pricingengines.swap.tree_swap_engine import TreeVanillaSwapEngine
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_grid import TimeGrid
from pquantlib.time.time_unit import TimeUnit

CPP: dict[str, Any] = reference_reader.load("v143/pe/bondswap")

# probe.cpp — `const Date kToday(15, May, 2025);`
TODAY = Date.from_ymd(15, Month.May, 2025)
DC_365 = Actual365Fixed()
DC_360 = Actual360()
DC_30_360 = Thirty360(Convention.BondBasis)

# LOOSE, not TIGHT: the lattice sums ~40 slices of up to ~80 nodes, each a
# probability-weighted three-term sum times a per-node discount, on a notional
# of 1e6 whose two legs cancel to an NPV of ~76. The cancellation alone costs
# ~1e6/76 ~ 1.3e4 in condition number, so a different summation order in
# NumPy vs C++ shows up around 1e-12 relative — which is exactly what the
# measured agreement is (2.9e-12 at 40 steps, 3.8e-13 at 200).
_TREE_REASON = (
    "lattice rollback of a 1e6-notional swap whose legs cancel to ~76: the "
    "~1.3e4 cancellation condition number turns double-rounding differences "
    "into ~1e-12 relative disagreement with C++"
)


@pytest.fixture(autouse=True)
def _eval_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    settings = ObservableSettings()
    saved = settings.evaluation_date
    # probe.cpp:main() — Settings::instance().evaluationDate() = Date(15, May, 2025)
    settings.evaluation_date = TODAY
    yield
    settings.evaluation_date = saved


def _curve() -> FlatForward:
    return FlatForward.from_rate(TODAY, 0.03, DC_365, Compounding.Compounded, Frequency.Annual)


def _schedule(tenor: Period) -> Schedule:
    return Schedule.from_rule(
        effective_date=Date.from_ymd(19, Month.May, 2025),
        termination_date=Date.from_ymd(19, Month.May, 2030),
        tenor=tenor,
        calendar=TARGET(),
        convention=BusinessDayConvention.ModifiedFollowing,
        termination_date_convention=BusinessDayConvention.ModifiedFollowing,
        rule=DateGeneration.Forward,
        end_of_month=False,
    )


def _swap(curve: FlatForward, swap_type: SwapType) -> VanillaSwap:
    """probe.cpp `runTreeSwapEngine` — 5y annual-vs-6M, 3% fixed, 1e6 notional."""
    return VanillaSwap(
        swap_type,
        1_000_000.0,
        _schedule(Period(1, TimeUnit.Years)),
        0.03,
        DC_30_360,
        _schedule(Period(6, TimeUnit.Months)),
        Euribor6M(curve),
        0.0,
        DC_360,
    )


def _mandatory_times(curve: FlatForward, swap_type: SwapType) -> list[float]:
    """``DiscretizedSwap::mandatoryTimes()`` — the grid the engine needs."""
    args = FixedVsFloatingSwapArguments()
    _swap(curve, swap_type).setup_arguments(args)
    return list(
        DiscretizedSwap(args, curve.reference_date(), curve.day_counter()).mandatory_times()
    )


def _model(curve: FlatForward) -> HullWhite:
    return HullWhite(curve, 0.05, 0.0075)


_CASES = sorted(k for k in CPP if k.startswith("tree_swap_") and k.endswith(("40", "80", "200", "throws")))


@pytest.mark.parametrize("case", _CASES)
def test_tree_swap_npv(case: str) -> None:
    inputs = CPP[case]["inputs"]
    expected = CPP[case]["expected"]
    curve = _curve()
    model = _model(curve)
    swap_type = SwapType.Payer if inputs["swap_type"] == "Payer" else SwapType.Receiver
    steps = int(inputs["time_steps"])
    swap = _swap(curve, swap_type)

    if expected.get("throws") is True:
        with pytest.raises(LibraryException, match="timeSteps must be positive"):
            TreeVanillaSwapEngine(model, steps, curve)
        return

    if inputs["use_time_grid"]:
        grid = TimeGrid.with_mandatory_and_steps(_mandatory_times(curve, swap_type), steps)
        assert grid.size() == int(expected["grid_size"])
        tolerance.tight(grid.back(), expected["grid_back"])
        engine = TreeVanillaSwapEngine(model, grid, curve)
        # The TimeGrid ctor builds the lattice eagerly (hpp:78-86).
        assert engine.lattice() is not None
        assert engine.time_steps() == 0
    else:
        engine = TreeVanillaSwapEngine(model, steps, curve)
        # The timeSteps ctor leaves it unbuilt (hpp:57-76).
        assert engine.lattice() is None
        assert engine.time_grid() is None
        assert engine.time_steps() == steps

    swap.set_pricing_engine(engine)
    tolerance.custom(
        swap.npv(), expected["npv"], abs_tol=1e-9, rel_tol=1e-11, reason=_TREE_REASON
    )

    # C++ TreeVanillaSwapEngine::calculate assigns ONLY results_.value.
    assert expected["fixed_leg_npv_throws"] is True
    assert expected["floating_leg_npv_throws"] is True
    assert expected["fair_rate_throws"] is True
    assert expected["fair_spread_throws"] is True
    for accessor in ("fixed_leg_npv", "floating_leg_npv", "fair_rate", "fair_spread"):
        fresh = _swap(curve, swap_type)
        fresh.set_pricing_engine(TreeVanillaSwapEngine(_model(curve), max(steps, 1), curve))
        with pytest.raises(LibraryException):
            getattr(fresh, accessor)()


def test_time_grid_flavour_matches_the_step_flavour_on_the_same_grid() -> None:
    """Same grid, two constructors, identical number — in C++ and in Python."""
    tolerance.exact(
        CPP["tree_swap_payer_timegrid40"]["expected"]["npv"],
        CPP["tree_swap_payer_steps40"]["expected"]["npv"],
    )
    # ... and a denser grid gives a different number, so a port that ignored the
    # TimeGrid constructor could not pass both.
    assert (
        CPP["tree_swap_payer_timegrid200"]["expected"]["npv"]
        != CPP["tree_swap_payer_timegrid40"]["expected"]["npv"]
    )

    curve = _curve()
    model = _model(curve)
    times = _mandatory_times(curve, SwapType.Payer)
    a = _swap(curve, SwapType.Payer)
    a.set_pricing_engine(TreeVanillaSwapEngine(model, 40, curve))
    b = _swap(curve, SwapType.Payer)
    b.set_pricing_engine(
        TreeVanillaSwapEngine(model, TimeGrid.with_mandatory_and_steps(times, 40), curve)
    )
    tolerance.exact(a.npv(), b.npv())


def test_tree_converges_to_the_discounting_engine() -> None:
    reference = CPP["tree_swap_discounting_reference"]["expected"]
    curve = _curve()
    swap = _swap(curve, SwapType.Payer)
    swap.set_pricing_engine(DiscountingSwapEngine(curve))
    tolerance.tight(swap.npv(), reference["npv"])
    tolerance.tight(swap.fair_rate(), reference["fair_rate"])
    # A Hull-White tree fitted to the curve reprices the swap essentially
    # exactly even at 40 steps — that is the whole point of HullWhite::tree's
    # slice-by-slice fit, and it is why C++'s 40-step number agrees with the
    # analytic one to 3e-12.
    assert abs(CPP["tree_swap_payer_steps40"]["expected"]["npv"] / reference["npv"] - 1) < 1e-11


# ---------------------------------------------------------------------------
# the two template base classes carry behaviour
# ---------------------------------------------------------------------------


def test_lattice_engine_rejects_zero_time_steps() -> None:
    """C++ ``QL_REQUIRE(timeSteps>0)`` lives in LatticeShortRateModelEngine."""
    curve = _curve()
    assert CPP["tree_swap_zero_steps_throws"]["expected"]["throws"] is True
    with pytest.raises(LibraryException, match="timeSteps must be positive"):
        TreeVanillaSwapEngine(_model(curve), 0, curve)


def test_generic_model_engine_holds_and_observes_the_model() -> None:
    """``GenericModelEngine`` is not a tag type: it holds the model and registers."""
    curve = _curve()
    model = _model(curve)
    engine = TreeVanillaSwapEngine(model, 40, curve)
    assert isinstance(engine, LatticeShortRateModelEngine)
    assert isinstance(engine, GenericModelEngine)
    assert engine.model() is model
    # The engine is registered as an observer of the model, so a model
    # notification reaches the engine's own observers.
    notified: list[int] = []

    class _Spy:
        def update(self) -> None:
            notified.append(1)

    spy = _Spy()
    engine.register_with(spy)
    model.notify_observers()
    assert notified


def test_time_grid_flavour_rebuilds_its_lattice_on_update() -> None:
    """C++ ``LatticeShortRateModelEngine::update`` (hpp:88-95)."""
    curve = _curve()
    model = _model(curve)
    grid = TimeGrid.with_mandatory_and_steps(_mandatory_times(curve, SwapType.Payer), 40)
    engine = TreeVanillaSwapEngine(model, grid, curve)
    first = engine.lattice()
    assert first is not None
    engine.update()
    second = engine.lattice()
    assert second is not None
    assert second is not first

    # The step flavour has an empty grid, so update() must NOT build one.
    step_engine = TreeVanillaSwapEngine(model, 40, curve)
    step_engine.update()
    assert step_engine.lattice() is None

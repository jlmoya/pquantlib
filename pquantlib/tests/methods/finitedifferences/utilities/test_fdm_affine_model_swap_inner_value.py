"""Cross-validation of ``FdmAffineModelSwapInnerValue`` against C++ v1.43.

# C++ parity: ql/methods/finitedifferences/utilities/fdmaffinemodelswapinnervalue.{hpp,cpp}
# @ v1.43 (6b57206e0).

The probe builds a 5y payer/receiver swap (annual 5% fixed vs semi-annual
IBOR, 1000 notional, NullCalendar / Act-365F throughout so the schedules are
convention-free), a Hull-White pair on 4.5% / 5.0% flat curves and a 5-node
short-rate mesh, then reads the exercise value at every node.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from pquantlib.currencies.europe import EURCurrency
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.instruments.swap import SwapType
from pquantlib.instruments.vanilla_swap import VanillaSwap
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.meshers.uniform_1d_mesher import Uniform1dMesher
from pquantlib.methods.finitedifferences.utilities.fdm_affine_model_swap_inner_value import (
    FdmAffineModelSwapInnerValue,
)
from pquantlib.models.shortrate.onefactor.hull_white import HullWhite
from pquantlib.models.shortrate.onefactor.vasicek import Vasicek
from pquantlib.models.shortrate.twofactor.g2 import G2
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.testing.tolerance import tight
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit

from .conftest import as_floats

_START = Date.from_ymd(15, Month.January, 2025)
_END = Date.from_ymd(15, Month.January, 2030)


@pytest.fixture(autouse=True)
def pin_evaluation_date(today: Date) -> Iterator[None]:
    """The probe pins ``Settings::evaluationDate()`` to 15 Jan 2024."""
    settings = ObservableSettings()
    previous = settings.evaluation_date
    settings.evaluation_date = today
    yield
    settings.evaluation_date = previous


@pytest.fixture
def discount_curve(today: Date, day_counter: DayCounter) -> YieldTermStructure:
    return FlatForward.from_rate(today, 0.045, day_counter)


@pytest.fixture
def forward_curve(today: Date, day_counter: DayCounter) -> YieldTermStructure:
    return FlatForward.from_rate(today, 0.050, day_counter)


def _make_swap(
    swap_type: SwapType,
    forward_curve: YieldTermStructure,
    calendar: Calendar,
    day_counter: DayCounter,
) -> VanillaSwap:
    index = IborIndex(
        "dummy",
        Period(6, TimeUnit.Months),
        0,
        EURCurrency(),
        calendar,
        BusinessDayConvention.Following,
        False,
        day_counter,
        forward_curve,
    )
    fixed_schedule = Schedule.from_rule(
        _START,
        _END,
        Period(1, TimeUnit.Years),
        calendar,
        BusinessDayConvention.Following,
        BusinessDayConvention.Following,
        DateGeneration.Forward,
        False,
    )
    float_schedule = Schedule.from_rule(
        _START,
        _END,
        Period(6, TimeUnit.Months),
        calendar,
        BusinessDayConvention.Following,
        BusinessDayConvention.Following,
        DateGeneration.Forward,
        False,
    )
    return VanillaSwap(
        swap_type,
        1000.0,
        fixed_schedule,
        0.05,
        day_counter,
        float_schedule,
        index,
        0.0,
        day_counter,
    )


def test_exercise_time(reference_data: dict[str, Any], today: Date, day_counter: DayCounter) -> None:
    tight(
        day_counter.year_fraction(today, _START),
        float(reference_data["affine_swap_exercise_time"]),
    )


def test_hull_white_payer_inner_value(
    reference_data: dict[str, Any],
    discount_curve: YieldTermStructure,
    forward_curve: YieldTermStructure,
    calendar: Calendar,
    day_counter: DayCounter,
    today: Date,
) -> None:
    swap = _make_swap(SwapType.Payer, forward_curve, calendar, day_counter)
    t_ex = day_counter.year_fraction(today, _START)
    mesher = FdmMesherComposite(Uniform1dMesher(-0.04, 0.04, 5))
    calc = FdmAffineModelSwapInnerValue(
        HullWhite(discount_curve, 0.05, 0.0075),
        HullWhite(forward_curve, 0.05, 0.0075),
        swap,
        {t_ex: _START},
        mesher,
        0,
    )
    actual = [calc.inner_value(it, t_ex) for it in mesher.layout().iter()]
    expected = as_floats(reference_data["affine_swap_hw_payer_inner"])
    for a, e in zip(actual, expected, strict=True):
        tight(a, e)


def test_hull_white_avg_equals_inner(
    reference_data: dict[str, Any],
    discount_curve: YieldTermStructure,
    forward_curve: YieldTermStructure,
    calendar: Calendar,
    day_counter: DayCounter,
    today: Date,
) -> None:
    """# C++ parity: ``avgInnerValue`` forwards to ``innerValue``."""
    swap = _make_swap(SwapType.Payer, forward_curve, calendar, day_counter)
    t_ex = day_counter.year_fraction(today, _START)
    mesher = FdmMesherComposite(Uniform1dMesher(-0.04, 0.04, 5))
    calc = FdmAffineModelSwapInnerValue(
        HullWhite(discount_curve, 0.05, 0.0075),
        HullWhite(forward_curve, 0.05, 0.0075),
        swap,
        {t_ex: _START},
        mesher,
        0,
    )
    actual = [calc.avg_inner_value(it, t_ex) for it in mesher.layout().iter()]
    expected = as_floats(reference_data["affine_swap_hw_payer_avg"])
    for a, e in zip(actual, expected, strict=True):
        tight(a, e)


def test_hull_white_receiver_inner_value(
    reference_data: dict[str, Any],
    discount_curve: YieldTermStructure,
    forward_curve: YieldTermStructure,
    calendar: Calendar,
    day_counter: DayCounter,
    today: Date,
) -> None:
    """The receiver flips the sign, so the payoff lives on the opposite nodes."""
    swap = _make_swap(SwapType.Receiver, forward_curve, calendar, day_counter)
    t_ex = day_counter.year_fraction(today, _START)
    mesher = FdmMesherComposite(Uniform1dMesher(-0.04, 0.04, 5))
    calc = FdmAffineModelSwapInnerValue(
        HullWhite(discount_curve, 0.05, 0.0075),
        HullWhite(forward_curve, 0.05, 0.0075),
        swap,
        {t_ex: _START},
        mesher,
        0,
    )
    actual = [calc.inner_value(it, t_ex) for it in mesher.layout().iter()]
    expected = as_floats(reference_data["affine_swap_hw_receiver_inner"])
    for a, e in zip(actual, expected, strict=True):
        tight(a, e)


def test_g2_payer_inner_value(
    reference_data: dict[str, Any],
    discount_curve: YieldTermStructure,
    forward_curve: YieldTermStructure,
    calendar: Calendar,
    day_counter: DayCounter,
    today: Date,
) -> None:
    """G2 reads the state off two consecutive mesher directions verbatim."""
    swap = _make_swap(SwapType.Payer, forward_curve, calendar, day_counter)
    t_ex = day_counter.year_fraction(today, _START)
    mesher = FdmMesherComposite(
        Uniform1dMesher(-0.02, 0.02, 3), Uniform1dMesher(-0.03, 0.03, 3)
    )
    calc = FdmAffineModelSwapInnerValue(
        G2(discount_curve, 0.1, 0.01, 0.1, 0.012, -0.75),
        G2(forward_curve, 0.1, 0.01, 0.1, 0.012, -0.75),
        swap,
        {t_ex: _START},
        mesher,
        0,
    )
    actual = [calc.inner_value(it, t_ex) for it in mesher.layout().iter()]
    expected = as_floats(reference_data["affine_swap_g2_payer_inner"])
    for a, e in zip(actual, expected, strict=True):
        tight(a, e)


def test_unsupported_model_rejected(
    discount_curve: YieldTermStructure,
    forward_curve: YieldTermStructure,
    calendar: Calendar,
    day_counter: DayCounter,
    today: Date,
) -> None:
    """C++ only instantiates ``getState`` for HullWhite and G2; anything else
    fails to link. Python raises instead."""
    swap = _make_swap(SwapType.Payer, forward_curve, calendar, day_counter)
    t_ex = day_counter.year_fraction(today, _START)
    mesher = FdmMesherComposite(Uniform1dMesher(-0.04, 0.04, 5))
    calc = FdmAffineModelSwapInnerValue(
        Vasicek(),  # pyright: ignore[reportArgumentType]
        HullWhite(forward_curve, 0.05, 0.0075),
        swap,
        {t_ex: _START},
        mesher,
        0,
    )
    it = next(iter(mesher.layout().iter()))
    with pytest.raises(LibraryException, match="HullWhite and G2"):
        calc.inner_value(it, t_ex)

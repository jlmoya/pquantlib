"""Cross-validate PerpetualFutures + DiscountingPerpetualFuturesEngine against C++ v1.43.

Probe: ``v143/inst/swaps`` (key ``perpetual_futures``).

``PerpetualFutures`` carries no calculation of its own — it forwards its five
constructor arguments to the engine — so the only way to prove an argument is
not silently dropped is to price it.  Every optional argument therefore gets a
case at a NON-default value:

instrument
    ``payoff_type``       — Linear and Inverse (Quanto is rejected by the engine)
    ``funding_type``      — FundingWithPreviousSpot and FundingWithCurrentSpot
    ``funding_frequency`` — Years / Months / Weeks / Days / Hours and the
                            zero-length (continuous) case
    ``cal`` / ``dc``      — TARGET + Actual/365F against the NullCalendar +
                            Actual/Actual(ISDA) defaults; both only bite in the
                            Weeks/Days branch, which advances through the
                            calendar, so that is the case they are swept in

engine
    ``funding_interp_type`` — PiecewiseConstant / Linear / CubicSpline over a
                              three-pillar funding structure
    ``max_t``               — 60 (default), 30, 20, 5 and 1
"""

from __future__ import annotations

from typing import Any, NamedTuple

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.actual_actual import ActualActual, Convention
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.exceptions import LibraryException
from pquantlib.instruments.perpetual_futures import (
    PerpetualFutures,
    PerpetualFuturesArguments,
    PerpetualFuturesFundingType,
    PerpetualFuturesPayoffType,
)
from pquantlib.instruments.swap import SwapArguments
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.futures.discounting_perpetual_futures_engine import (
    DiscountingPerpetualFuturesEngine,
    FundingInterpolationType,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import tight
from pquantlib.time.calendar import Calendar
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_EVAL = Date.from_ymd(17, Month.January, 2024)
_SPOT = 10000.0
_DOM_RATE = 0.04
_FOR_RATE = 0.02

# Single-pillar funding structure (the C++ test-suite case).
_T1 = [0.0]
_K1 = [0.01]
_I1 = [0.005]
# Three-pillar structure — makes the interpolation type observable.
_T3 = [0.0, 1.0, 5.0]
_K3 = [0.010, 0.020, 0.035]
_I3 = [0.005, 0.008, 0.012]

_Linear = PerpetualFuturesPayoffType.Linear
_Inverse = PerpetualFuturesPayoffType.Inverse
_Prev = PerpetualFuturesFundingType.FundingWithPreviousSpot
_Curr = PerpetualFuturesFundingType.FundingWithCurrentSpot
_PWC = FundingInterpolationType.PiecewiseConstant


@pytest.fixture(autouse=True)
def _set_eval_date() -> None:  # pyright: ignore[reportUnusedFunction]
    ObservableSettings().evaluation_date = _EVAL


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/inst/swaps")


def _isda() -> DayCounter:
    return ActualActual(Convention.ISDA)


def _flat(rate: float) -> YieldTermStructure:
    return FlatForward.from_rate(_EVAL, rate, _isda(), Compounding.Continuous, Frequency.Annual)


class _Case(NamedTuple):
    payoff: PerpetualFuturesPayoffType
    funding: PerpetualFuturesFundingType
    freq: Period
    cal: Calendar | None
    dc: DayCounter | None
    multi_pillar: bool
    interp: FundingInterpolationType
    max_t: float


_MONTHS = TimeUnit.Months
_YEARS = TimeUnit.Years
_WEEKS = TimeUnit.Weeks
_DAYS = TimeUnit.Days
_HOURS = TimeUnit.Hours

_CASES: dict[str, _Case] = {
    "lin_prev_3m": _Case(_Linear, _Prev, Period(3, _MONTHS), None, None, False, _PWC, 60.0),
    "lin_curr_3m": _Case(_Linear, _Curr, Period(3, _MONTHS), None, None, False, _PWC, 60.0),
    "inv_prev_3m": _Case(_Inverse, _Prev, Period(3, _MONTHS), None, None, False, _PWC, 60.0),
    "inv_curr_3m": _Case(_Inverse, _Curr, Period(3, _MONTHS), None, None, False, _PWC, 60.0),
    "lin_prev_continuous": _Case(_Linear, _Prev, Period(0, _MONTHS), None, None, False, _PWC, 60.0),
    "inv_prev_continuous": _Case(_Inverse, _Prev, Period(0, _MONTHS), None, None, False, _PWC, 60.0),
    "lin_prev_1y": _Case(_Linear, _Prev, Period(1, _YEARS), None, None, False, _PWC, 60.0),
    "lin_prev_1w": _Case(_Linear, _Prev, Period(1, _WEEKS), None, None, False, _PWC, 20.0),
    "lin_prev_1d_maxt5": _Case(_Linear, _Prev, Period(1, _DAYS), None, None, False, _PWC, 5.0),
    "lin_curr_8h_maxt1": _Case(_Linear, _Curr, Period(8, _HOURS), None, None, False, _PWC, 1.0),
    "lin_prev_1w_target_a365": _Case(
        _Linear, _Prev, Period(1, _WEEKS), TARGET(), Actual365Fixed(), False, _PWC, 20.0
    ),
    "lin_prev_3m_pwc3": _Case(_Linear, _Prev, Period(3, _MONTHS), None, None, True, _PWC, 60.0),
    "lin_prev_3m_linear3": _Case(
        _Linear, _Prev, Period(3, _MONTHS), None, None, True, FundingInterpolationType.Linear, 60.0
    ),
    "lin_prev_3m_cubic3": _Case(
        _Linear,
        _Prev,
        Period(3, _MONTHS),
        None,
        None,
        True,
        FundingInterpolationType.CubicSpline,
        60.0,
    ),
    "lin_prev_3m_maxt30": _Case(_Linear, _Prev, Period(3, _MONTHS), None, None, False, _PWC, 30.0),
    "inv_curr_continuous_maxt30": _Case(
        _Inverse,
        _Curr,
        Period(0, _MONTHS),
        None,
        None,
        True,
        FundingInterpolationType.Linear,
        30.0,
    ),
}


def _build(case: _Case) -> PerpetualFutures:
    trade = PerpetualFutures(case.payoff, case.funding, case.freq, case.cal, case.dc)
    engine = DiscountingPerpetualFuturesEngine(
        _flat(_DOM_RATE),
        _flat(_FOR_RATE),
        SimpleQuote(_SPOT),
        _T3 if case.multi_pillar else _T1,
        _K3 if case.multi_pillar else _K1,
        _I3 if case.multi_pillar else _I1,
        case.interp,
        case.max_t,
    )
    trade.set_pricing_engine(engine)
    return trade


@pytest.mark.parametrize("key", list(_CASES))
def test_npv(cpp: dict[str, Any], key: str) -> None:
    ref = cpp["perpetual_futures"]
    tight(ref["spot"], _SPOT)
    tight(ref["dom_rate"], _DOM_RATE)
    tight(ref["for_rate"], _FOR_RATE)
    tight(_build(_CASES[key]).npv(), ref["values"][key], reason=f"{key} NPV")


def test_payoff_and_funding_type_are_observable(cpp: dict[str, Any]) -> None:
    values = cpp["perpetual_futures"]["values"]
    assert values["lin_prev_3m"] != values["lin_curr_3m"]  # funding type
    assert values["inv_prev_3m"] != values["inv_curr_3m"]  # funding type
    assert values["lin_prev_3m"] != values["inv_prev_3m"]  # payoff type
    assert _build(_CASES["lin_prev_3m"]).npv() != _build(_CASES["lin_curr_3m"]).npv()
    assert _build(_CASES["lin_prev_3m"]).npv() != _build(_CASES["inv_prev_3m"]).npv()


def test_funding_frequency_is_observable(cpp: dict[str, Any]) -> None:
    values = cpp["perpetual_futures"]["values"]
    keys = [
        "lin_prev_3m",
        "lin_prev_continuous",
        "lin_prev_1y",
        "lin_prev_1w",
        "lin_prev_1d_maxt5",
        "lin_curr_8h_maxt1",
    ]
    assert len({values[k] for k in keys}) == len(keys)


def test_calendar_and_day_counter_are_observable(cpp: dict[str, Any]) -> None:
    """The non-default calendar/day-counter pair must change the grid."""
    values = cpp["perpetual_futures"]["values"]
    assert values["lin_prev_1w"] != values["lin_prev_1w_target_a365"]
    assert _build(_CASES["lin_prev_1w"]).npv() != _build(_CASES["lin_prev_1w_target_a365"]).npv()


def test_interpolation_type_is_observable(cpp: dict[str, Any]) -> None:
    values = cpp["perpetual_futures"]["values"]
    keys = ["lin_prev_3m_pwc3", "lin_prev_3m_linear3", "lin_prev_3m_cubic3"]
    assert len({values[k] for k in keys}) == len(keys)
    assert len({_build(_CASES[k]).npv() for k in keys}) == len(keys)


def test_max_t_is_observable(cpp: dict[str, Any]) -> None:
    """``max_t`` only shifts the flat-extrapolated tail, but it must shift it."""
    values = cpp["perpetual_futures"]["values"]
    assert values["lin_prev_3m"] != values["lin_prev_3m_maxt30"]
    assert _build(_CASES["lin_prev_3m"]).npv() != _build(_CASES["lin_prev_3m_maxt30"]).npv()


def test_instrument_inspectors(cpp: dict[str, Any]) -> None:
    """The C++ defaults are FundingWithCurrentSpot / 8 Hours / NullCalendar / AA(ISDA)."""
    default = PerpetualFutures(_Linear)
    assert default.payoff_type() == _Linear
    assert default.funding_type() == _Curr
    assert default.funding_frequency() == Period(8, TimeUnit.Hours)
    assert default.calendar().name() == NullCalendar().name()
    assert default.day_counter().name() == "Actual/Actual (ISDA)"
    # ``PerpetualFutures::isExpired`` is hard-coded false (perpetualfutures.hpp:78).
    assert default.is_expired() is cpp["perpetual_futures"]["is_expired"]

    custom = PerpetualFutures(_Inverse, _Prev, Period(1, TimeUnit.Days), TARGET(), Actual365Fixed())
    assert custom.payoff_type() == _Inverse
    assert custom.funding_type() == _Prev
    assert custom.funding_frequency() == Period(1, TimeUnit.Days)
    assert custom.calendar().name() == TARGET().name()
    assert custom.day_counter().name() == "Actual/365 (Fixed)"


def test_enum_str_round_trip() -> None:
    """# C++ parity: the ``operator<<`` overloads (perpetualfutures.cpp:69-91)."""
    assert str(_Linear) == "Linear"
    assert str(_Inverse) == "Inverse"
    assert str(PerpetualFuturesPayoffType.Quanto) == "Quanto"
    assert str(_Prev) == "FundingWithPreviousSpot"
    assert str(_Curr) == "FundingWithCurrentSpot"


def test_quanto_is_rejected_by_the_engine(cpp: dict[str, Any]) -> None:
    assert cpp["perpetual_futures_raises"]["quanto_not_supported"]["raises"] is True
    trade = PerpetualFutures(PerpetualFuturesPayoffType.Quanto)
    trade.set_pricing_engine(
        DiscountingPerpetualFuturesEngine(
            _flat(_DOM_RATE), _flat(_FOR_RATE), SimpleQuote(_SPOT), _T1, _K1, _I1
        )
    )
    with pytest.raises(LibraryException, match="Only Linear and Inverse payoffs"):
        trade.npv()


def test_empty_funding_times_raises(cpp: dict[str, Any]) -> None:
    assert cpp["perpetual_futures_raises"]["empty_funding_times"]["raises"] is True
    with pytest.raises(LibraryException, match="fundingTimes is empty"):
        DiscountingPerpetualFuturesEngine(
            _flat(_DOM_RATE), _flat(_FOR_RATE), SimpleQuote(_SPOT), [], _K1, _I1
        )


def test_size_mismatch_raises(cpp: dict[str, Any]) -> None:
    raises = cpp["perpetual_futures_raises"]
    assert raises["size_mismatch_rates"]["raises"] is True
    assert raises["size_mismatch_diffs"]["raises"] is True
    with pytest.raises(LibraryException, match="fundingTimes and fundingRates"):
        DiscountingPerpetualFuturesEngine(
            _flat(_DOM_RATE), _flat(_FOR_RATE), SimpleQuote(_SPOT), _T1, _K3, _I1
        )
    with pytest.raises(LibraryException, match="fundingTimes and interestRateDiffs"):
        DiscountingPerpetualFuturesEngine(
            _flat(_DOM_RATE), _flat(_FOR_RATE), SimpleQuote(_SPOT), _T1, _K1, _I3
        )


def test_negative_terminal_funding_rate_raises(cpp: dict[str, Any]) -> None:
    assert cpp["perpetual_futures_raises"]["negative_terminal_funding_rate"]["raises"] is True
    trade = PerpetualFutures(_Linear)
    trade.set_pricing_engine(
        DiscountingPerpetualFuturesEngine(
            _flat(_DOM_RATE), _flat(_FOR_RATE), SimpleQuote(_SPOT), _T1, [-0.01], _I1
        )
    )
    with pytest.raises(LibraryException, match="fundingRate at max time is negative"):
        trade.npv()


def test_unset_argument_types_are_rejected(cpp: dict[str, Any]) -> None:
    """C++ seeds ``arguments`` with ``PayoffType(-1)`` / ``FundingType(-1)``.

    Python enums are closed, so the port's sentinel is ``None``; a
    default-constructed argument set must fail ``validate()`` with the same two
    messages the C++ probe recorded.
    """
    raises = cpp["perpetual_futures_raises"]
    assert raises["unknown_payoff_type"]["raises"] is True
    assert raises["unknown_funding_type"]["raises"] is True

    args = PerpetualFuturesArguments()
    with pytest.raises(LibraryException, match="unknown payoff type"):
        args.validate()
    args.payoff_type = _Linear
    with pytest.raises(LibraryException, match="unknown funding type"):
        args.validate()
    args.funding_type = _Curr
    args.validate()


def test_setup_arguments_forwards_every_field() -> None:
    """A dropped field here is exactly the defect this suite exists to catch."""
    trade = PerpetualFutures(_Inverse, _Prev, Period(2, TimeUnit.Weeks), TARGET(), Actual365Fixed())
    args = PerpetualFuturesArguments()
    trade.setup_arguments(args)
    assert args.payoff_type == _Inverse
    assert args.funding_type == _Prev
    assert args.funding_frequency == Period(2, TimeUnit.Weeks)
    assert args.cal.name() == TARGET().name()
    assert args.dc.name() == "Actual/365 (Fixed)"
    args.validate()


def test_setup_arguments_rejects_the_wrong_carrier() -> None:
    """# C++ parity: ``QL_REQUIRE(moreArgs != nullptr, "wrong argument type")``."""
    trade = PerpetualFutures(_Linear)
    with pytest.raises(LibraryException, match="wrong argument type"):
        trade.setup_arguments(SwapArguments())

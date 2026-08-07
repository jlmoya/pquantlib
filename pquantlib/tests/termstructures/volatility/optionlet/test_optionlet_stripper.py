"""Cross-validation of the abstract ``OptionletStripper`` base against v1.43.

# C++ parity: ql/termstructures/volatility/optionlet/optionletstripper.{hpp,cpp}
# (v1.43).

Reference: ``migration-harness/references/v143/ts/optionletstripper.json``,
emitted verbatim by
``migration-harness/cpp/probes/v143_ts_optionletstripper/probe.cpp`` (three
consecutive runs byte-identical).

What the base actually owns, and therefore what is pinned here:

* the optionlet-tenor / cap-floor-length walk and ``optionletMaturities()``
  (optionletstripper.cpp:58-73) — closed-form, EXACT;
* the ``ext::optional<Period>`` ``optionletFrequency()`` accessor, with the
  ``nullopt`` and the engaged branch pinned separately (…:173-175);
* the metadata forwards to the term-vol surface — dayCounter / calendar /
  settlementDays / businessDayConvention (…:140-154) — EXACT;
* the constructor guard rails (…:43-51, 64-66);
* the lazy-recalculation contract: reading strikes / vols / dates / times /
  payment dates / accrual periods / ATM rates runs ``performCalculations``
  first, while ``optionletFixingTenors()`` and ``optionletMaturities()`` do
  NOT (…:87-137 vs :106-122).

The date grid itself (fixing dates, payment dates, accrual periods, fixing
times, ATM rates) is filled by ``OptionletStripper1::performCalculations``
(optionletstripper1.cpp:61-84) and copied from stripper1 by
``OptionletStripper2::performCalculations`` (optionletstripper2.cpp:60-68) —
the base only sizes it. It is pinned here anyway because it is what the base's
accessors return, and because a date bug is invisible without it. Dates are
compared as serial number AND ISO string.

Tolerance tiers:

* dates, serials, tenors, day counter / calendar / convention, settlement
  days, displacement, volatility type, optionlet frequency — EXACT (integers,
  strings and enums; compared with ``==``);
* fixing times, accrual periods, ATM forward rates, switch strike — TIGHT
  (they came out bit-identical, but TIGHT leaves room for a libm ULP);
* stripped optionlet volatilities — LOOSE. Derivation: each cell is the output
  of ``blackFormulaImpliedStdDev``'s Newton iteration with ``accuracy_ =
  1e-6`` on a cap-price DIFFERENCE, so the two implementations agree only to
  their shared residual, not to their arithmetic. Observed worst relative
  deviation across the pinned scenarios is 4.1e-11 — comfortably inside LOOSE
  and two orders outside TIGHT.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.ibor.eonia import Eonia
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.capfloor.cap_floor_term_vol_surface import (
    CapFloorTermVolSurface,
)
from pquantlib.termstructures.volatility.optionlet.optionlet_stripper import (
    OptionletStripper,
)
from pquantlib.termstructures.volatility.optionlet.optionlet_stripper_1 import (
    OptionletStripper1,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from pquantlib.time.date import Date as _Date

_REF = reference_reader.load("v143/ts/optionletstripper")

_Y = TimeUnit.Years
_M = TimeUnit.Months
_D = TimeUnit.Days

# 1Y / 2Y / 3Y / 5Y — probe.cpp:200, 283, 315, 349, 376.
_TENORS_4 = [Period(1, _Y), Period(2, _Y), Period(3, _Y), Period(5, _Y)]
_STRIKES_3 = [0.02, 0.04, 0.06]

# C++ ``operator<<(ostream&, Period)`` short form, as emitted by probe.cpp:62-66.
_UNIT_SUFFIX = {_D: "D", TimeUnit.Weeks: "W", _M: "M", _Y: "Y"}


def _period_str(p: Period) -> str:
    return f"{p.length}{_UNIT_SUFFIX[p.units]}"


@pytest.fixture(autouse=True)
def _restore_evaluation_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """Save/restore the global evaluation date around every test.

    Each scenario builder below sets it to its own probe-pinned value (see
    ``eval_date_serial`` in the reference, written by probe.cpp:175-176), so
    the fixture only has to put back what was there.
    """
    settings = ObservableSettings()
    previous = settings.evaluation_date
    try:
        yield
    finally:
        settings.evaluation_date = previous


# --------------------------------------------------------------------------
# Scenario builders — one per top-level key in the reference.
# --------------------------------------------------------------------------


def _euribor3m_flat18() -> tuple[dict[str, Any], OptionletStripper1]:
    """probe.cpp:191-221 — Euribor3M, fixed-reference surface, flat 18%."""
    s = _REF["euribor3m_flat18"]
    eval_date = Date(s["eval_date_serial"])
    ObservableSettings().evaluation_date = eval_date
    curve = FlatForward(eval_date, SimpleQuote(0.03), Actual365Fixed())
    index = Euribor(Period(3, _M), curve)
    surface = CapFloorTermVolSurface(
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        option_tenors=_TENORS_4,
        strikes=_STRIKES_3,
        volatilities=np.full((4, 3), 0.18),
        calendar=TARGET(),
        day_counter=Actual365Fixed(),
        reference_date=eval_date,
    )
    return s, OptionletStripper1(surface, index)


def _euribor6m_holiday() -> tuple[dict[str, Any], OptionletStripper1]:
    """probe.cpp:222-272 — Euribor6M, Actual360, eval date on a TARGET holiday."""
    s = _REF["euribor6m_holiday_evaldate"]
    eval_date = Date(s["eval_date_serial"])
    ObservableSettings().evaluation_date = eval_date
    curve = FlatForward(eval_date, SimpleQuote(0.025), Actual365Fixed())
    index = Euribor(Period(6, _M), curve)
    vols = np.array([[0.22 - 0.006 * i + 0.011 * j for j in range(3)] for i in range(6)])
    surface = CapFloorTermVolSurface(
        business_day_convention=BusinessDayConvention.Following,
        option_tenors=[
            Period(1, _Y),
            Period(2, _Y),
            Period(3, _Y),
            Period(5, _Y),
            Period(7, _Y),
            Period(10, _Y),
        ],
        strikes=[0.01, 0.03, 0.05],
        volatilities=vols,
        calendar=TARGET(),
        day_counter=Actual360(),
        reference_date=eval_date,
    )
    return s, OptionletStripper1(surface, index)


def _euribor3m_freq6m() -> tuple[dict[str, Any], OptionletStripper1]:
    """probe.cpp:274-304 — Euribor3M with an explicit 6M optionlet frequency."""
    s = _REF["euribor3m_freq6m"]
    eval_date = Date(s["eval_date_serial"])
    ObservableSettings().evaluation_date = eval_date
    curve = FlatForward(eval_date, SimpleQuote(0.03), Actual365Fixed())
    index = Euribor(Period(3, _M), curve)
    surface = CapFloorTermVolSurface(
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        option_tenors=_TENORS_4,
        strikes=_STRIKES_3,
        volatilities=np.full((4, 3), 0.18),
        calendar=TARGET(),
        day_counter=Actual365Fixed(),
        reference_date=eval_date,
    )
    return s, OptionletStripper1(surface, index, optionlet_frequency=Period(6, _M))


def _euribor3m_moving() -> tuple[dict[str, Any], OptionletStripper1]:
    """probe.cpp:306-339 — moving-reference surface (settlementDays = 2).

    The only configuration in which C++ ``settlementDays()`` returns instead
    of throwing (ql/termstructure.hpp:127-131).
    """
    s = _REF["euribor3m_moving_settlement"]
    eval_date = Date(s["eval_date_serial"])
    ObservableSettings().evaluation_date = eval_date
    # PQuantLib's FlatForward has no moving-reference ctor; pin it to the same
    # date C++'s FlatForward(2, TARGET(), ...) resolves to.
    curve = FlatForward(
        TARGET().advance(eval_date, 2, _D), SimpleQuote(0.03), Actual365Fixed()
    )
    index = Euribor(Period(3, _M), curve)
    surface = CapFloorTermVolSurface(
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        option_tenors=_TENORS_4,
        strikes=_STRIKES_3,
        volatilities=np.full((4, 3), 0.18),
        calendar=TARGET(),
        day_counter=Actual365Fixed(),
        settlement_days=2,
    )
    return s, OptionletStripper1(surface, index)


def _euribor3m_normal() -> tuple[dict[str, Any], OptionletStripper1]:
    """probe.cpp:341-366 — Normal volatility type, displacement 0."""
    s = _REF["euribor3m_normal"]
    eval_date = Date(s["eval_date_serial"])
    ObservableSettings().evaluation_date = eval_date
    curve = FlatForward(eval_date, SimpleQuote(0.03), Actual365Fixed())
    index = Euribor(Period(3, _M), curve)
    surface = CapFloorTermVolSurface(
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        option_tenors=_TENORS_4,
        strikes=_STRIKES_3,
        volatilities=np.full((4, 3), 0.006),
        calendar=TARGET(),
        day_counter=Actual365Fixed(),
        reference_date=eval_date,
    )
    return s, OptionletStripper1(
        surface, index, volatility_type=VolatilityType.Normal
    )


def _euribor3m_shifted() -> tuple[dict[str, Any], OptionletStripper1]:
    """probe.cpp:368-392 — shifted lognormal with displacement 0.01."""
    s = _REF["euribor3m_shifted"]
    eval_date = Date(s["eval_date_serial"])
    ObservableSettings().evaluation_date = eval_date
    curve = FlatForward(eval_date, SimpleQuote(0.03), Actual365Fixed())
    index = Euribor(Period(3, _M), curve)
    surface = CapFloorTermVolSurface(
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        option_tenors=_TENORS_4,
        strikes=_STRIKES_3,
        volatilities=np.full((4, 3), 0.18),
        calendar=TARGET(),
        day_counter=Actual365Fixed(),
        reference_date=eval_date,
    )
    return s, OptionletStripper1(surface, index, displacement=0.01)


# Scenarios whose whole date grid PQuantLib reproduces from C++.
_GRID_SCENARIOS = {
    "euribor3m_flat18": _euribor3m_flat18,
    "euribor6m_holiday_evaldate": _euribor6m_holiday,
    "euribor3m_freq6m": _euribor3m_freq6m,
    "euribor3m_moving_settlement": _euribor3m_moving,
    "euribor3m_normal": _euribor3m_normal,
    "euribor3m_shifted": _euribor3m_shifted,
}

# Scenarios whose stripped ShiftedLognormal vols are also cross-validated.
# All three use a FLAT term-vol surface, so the term vols the stripper reads
# back are exact by construction and the comparison isolates the stripping
# itself. The three excluded scenarios each have a recorded divergence that is
# NOT in OptionletStripper — see the "known divergences" section at the end.
_VOL_SCENARIOS = ["euribor3m_flat18", "euribor3m_freq6m", "euribor3m_shifted"]


def _iso(dates: Sequence[_Date]) -> list[str]:
    """``Date.__str__`` is the ISO form, matching probe.cpp's ``io::iso_date``."""
    return [str(d) for d in dates]


# --------------------------------------------------------------------------
# The class exists and is where the shared state lives.
# --------------------------------------------------------------------------


def test_optionlet_stripper_is_a_real_abstract_base() -> None:
    """# C++ parity: ``class OptionletStripper : public StrippedOptionletBase``."""
    from pquantlib.termstructures.volatility.optionlet.stripped_optionlet_base import (  # noqa: PLC0415
        StrippedOptionletBase,
    )

    assert issubclass(OptionletStripper, StrippedOptionletBase)
    # performCalculations is left to the subclasses (optionletstripper.hpp:38-40).
    assert "_perform_calculations" in OptionletStripper.__abstractmethods__


def test_both_strippers_derive_from_the_base() -> None:
    from pquantlib.termstructures.volatility.optionlet.optionlet_stripper_2 import (  # noqa: PLC0415
        OptionletStripper2,
    )

    assert issubclass(OptionletStripper1, OptionletStripper)
    assert issubclass(OptionletStripper2, OptionletStripper)


def test_base_owns_the_whole_read_interface() -> None:
    """Neither subclass re-implements a base accessor — as in C++."""
    accessors = [
        "optionlet_strikes",
        "optionlet_volatilities",
        "optionlet_fixing_dates",
        "optionlet_fixing_times",
        "optionlet_maturities",
        "atm_optionlet_rates",
        "day_counter",
        "calendar",
        "settlement_days",
        "business_day_convention",
        "displacement",
        "volatility_type",
        "optionlet_fixing_tenors",
        "optionlet_payment_dates",
        "optionlet_accrual_periods",
        "optionlet_frequency",
    ]
    from pquantlib.termstructures.volatility.optionlet.optionlet_stripper_2 import (  # noqa: PLC0415
        OptionletStripper2,
    )

    for name in accessors:
        assert name not in OptionletStripper1.__dict__, f"{name} shadowed on stripper1"
        assert name not in OptionletStripper2.__dict__, f"{name} shadowed on stripper2"
        assert name in OptionletStripper.__dict__, f"{name} missing on the base"


# --------------------------------------------------------------------------
# Constructor-time state: the tenor walk (EXACT).
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(_GRID_SCENARIOS))
def test_tenor_walk_matches_cpp(key: str) -> None:
    """# C++ parity: optionletstripper.cpp:58-73 — EXACT (integer arithmetic)."""
    expected, stripper = _GRID_SCENARIOS[key]()
    assert stripper.optionlet_maturities() == expected["optionlet_maturities"]
    assert [
        _period_str(p) for p in stripper.optionlet_fixing_tenors()
    ] == expected["optionlet_fixing_tenors"]


@pytest.mark.parametrize("key", sorted(_GRID_SCENARIOS))
def test_metadata_forwards_match_cpp(key: str) -> None:
    """# C++ parity: optionletstripper.cpp:140-154 — EXACT."""
    expected, stripper = _GRID_SCENARIOS[key]()
    assert stripper.day_counter().name() == expected["day_counter"]
    assert stripper.calendar().name() == expected["calendar"]
    assert int(stripper.business_day_convention()) == expected["business_day_convention"]
    assert int(stripper.volatility_type()) == expected["volatility_type"]
    tolerance.exact(stripper.displacement(), float(expected["displacement"]))


def test_optionlet_frequency_is_none_when_not_supplied() -> None:
    """``ext::nullopt`` is NOT a zero Period — optionletstripper.cpp:173-175."""
    expected, stripper = _euribor3m_flat18()
    assert expected["optionlet_frequency"] is None
    assert stripper.optionlet_frequency() is None
    assert stripper.optionlet_frequency() != Period(0, _D)


def test_optionlet_frequency_round_trips_the_supplied_period() -> None:
    expected, stripper = _euribor3m_freq6m()
    frequency = stripper.optionlet_frequency()
    assert frequency is not None
    assert _period_str(frequency) == expected["optionlet_frequency"] == "6M"
    # ...and the walk stepped by 6M, not by the index's 3M tenor.
    assert stripper.optionlet_maturities() == expected["optionlet_maturities"] == 9


def test_settlement_days_forwards_from_a_moving_reference_surface() -> None:
    """# C++ parity: optionletstripper.cpp:148-150."""
    expected, stripper = _euribor3m_moving()
    assert expected["settlement_days_throws"] is False
    assert stripper.settlement_days() == expected["settlement_days"] == 2
    assert str(stripper.term_vol_surface().reference_date()) == (
        expected["reference_date_iso"]
    )


def test_fixed_reference_surface_settlement_days_is_a_known_divergence() -> None:
    """C++ THROWS here; PQuantLib returns 0.

    ql/termstructure.hpp:127-131 QL_REQUIREs ``settlementDays_ !=
    Null<Natural>()``, so ``settlementDays()`` on a fixed-reference-date
    surface is an error in C++ — pinned as
    ``settlement_days_throws: true``. ``OptionletStripper.settlement_days``
    swallows it because ``StrippedOptionletAdapter.__init__`` reads it, and
    OptionletStripper2's existing tests are built on fixed-reference
    surfaces. Recorded, not silently accepted; flagged for align().
    """
    expected, stripper = _euribor3m_flat18()
    assert expected["settlement_days_throws"] is True
    assert expected["settlement_days"] is None
    assert stripper.settlement_days() == 0  # DIVERGENCE, see docstring


# --------------------------------------------------------------------------
# The date grid the base's accessors return.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(_GRID_SCENARIOS))
def test_optionlet_fixing_dates_match_cpp(key: str) -> None:
    """EXACT, as serial number AND ISO string. # C++: optionletstripper.cpp:110-113."""
    expected, stripper = _GRID_SCENARIOS[key]()
    dates = stripper.optionlet_fixing_dates()
    assert [d.serial_number() for d in dates] == expected["optionlet_fixing_dates_serial"]
    assert _iso(dates) == expected["optionlet_fixing_dates_iso"]


@pytest.mark.parametrize("key", sorted(_GRID_SCENARIOS))
def test_optionlet_payment_dates_match_cpp(key: str) -> None:
    """EXACT, serial AND ISO. # C++: optionletstripper.cpp:124-127."""
    expected, stripper = _GRID_SCENARIOS[key]()
    dates = stripper.optionlet_payment_dates()
    assert [d.serial_number() for d in dates] == (
        expected["optionlet_payment_dates_serial"]
    )
    assert _iso(dates) == expected["optionlet_payment_dates_iso"]


@pytest.mark.parametrize("key", sorted(_GRID_SCENARIOS))
def test_optionlet_fixing_times_match_cpp(key: str) -> None:
    """TIGHT. # C++: optionletstripper.cpp:115-118."""
    expected, stripper = _GRID_SCENARIOS[key]()
    actual = stripper.optionlet_fixing_times()
    assert len(actual) == len(expected["optionlet_fixing_times"])
    for got, want in zip(actual, expected["optionlet_fixing_times"], strict=True):
        tolerance.tight(got, want)


@pytest.mark.parametrize("key", sorted(_GRID_SCENARIOS))
def test_optionlet_accrual_periods_match_cpp(key: str) -> None:
    """TIGHT. # C++: optionletstripper.cpp:129-132."""
    expected, stripper = _GRID_SCENARIOS[key]()
    actual = stripper.optionlet_accrual_periods()
    for got, want in zip(actual, expected["optionlet_accrual_periods"], strict=True):
        tolerance.tight(got, want)


@pytest.mark.parametrize("key", sorted(_GRID_SCENARIOS))
def test_atm_optionlet_rates_match_cpp(key: str) -> None:
    """TIGHT. # C++: optionletstripper.cpp:134-137."""
    expected, stripper = _GRID_SCENARIOS[key]()
    actual = stripper.atm_optionlet_rates()
    for got, want in zip(actual, expected["atm_optionlet_rates"], strict=True):
        tolerance.tight(got, want)


@pytest.mark.parametrize("key", sorted(_GRID_SCENARIOS))
def test_optionlet_strikes_rows_match_cpp(key: str) -> None:
    """Every row is a copy of the surface's strike vector — EXACT.

    # C++: ``optionletStrikes_ = vector<vector<Rate>>(nOptionletTenors_,
    # termVolSurface->strikes())`` (optionletstripper.cpp:78-79).
    """
    expected, stripper = _GRID_SCENARIOS[key]()
    for i in range(stripper.optionlet_maturities()):
        row = stripper.optionlet_strikes(i)
        for got, want in zip(row, expected["optionlet_strikes"][i], strict=True):
            tolerance.exact(got, want)


@pytest.mark.parametrize("key", _VOL_SCENARIOS)
def test_stripped_optionlet_volatilities_match_cpp(key: str) -> None:
    """LOOSE — the cells are Newton-solver outputs (see the module docstring)."""
    expected, stripper = _GRID_SCENARIOS[key]()
    for i in range(stripper.optionlet_maturities()):
        row = stripper.optionlet_volatilities(i)
        for got, want in zip(row, expected["optionlet_volatilities"][i], strict=True):
            tolerance.loose(got, want)


@pytest.mark.parametrize(
    "key", ["euribor3m_flat18", "euribor6m_holiday_evaldate", "euribor3m_moving_settlement"]
)
def test_switch_strike_matches_cpp(key: str) -> None:
    """TIGHT — the mean of the ATM forwards. # C++: optionletstripper1.cpp:86-92."""
    expected, stripper = _GRID_SCENARIOS[key]()
    tolerance.tight(stripper.switch_strike(), expected["switch_strike"])


def test_holiday_evaluation_date_does_not_shift_the_grid() -> None:
    """1 May 2024 is a TARGET holiday; C++ rolls it forward before spotting.

    ``MakeVanillaSwap`` does ``refDate =
    iborIndex_->fixingCalendar().adjust(refDate)`` BEFORE
    ``iborIndex_->valueDate(refDate)`` (makevanillaswap.cpp:71-77). Without
    that roll the whole schedule lands two business days early and every
    serial in the grid is still perfectly plausible — which is exactly why the
    probe emits ISO strings too.
    """
    expected, stripper = _euribor6m_holiday()
    assert expected["eval_date_is_target_holiday"] is True
    assert not TARGET().is_business_day(Date(expected["eval_date_serial"]))
    assert _iso(stripper.optionlet_fixing_dates())[0] == "2024-11-04"
    assert _iso(stripper.optionlet_payment_dates())[0] == "2025-05-06"


# --------------------------------------------------------------------------
# Constructor guard rails (optionletstripper.cpp:43-51, 64-66).
# --------------------------------------------------------------------------


def _guard_surface(eval_date: Date, tenors: list[Period]) -> CapFloorTermVolSurface:
    return CapFloorTermVolSurface(
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        option_tenors=tenors,
        strikes=_STRIKES_3,
        volatilities=np.full((len(tenors), 3), 0.18),
        calendar=TARGET(),
        day_counter=Actual365Fixed(),
        reference_date=eval_date,
    )


def test_normal_with_non_zero_displacement_is_rejected() -> None:
    """# C++ parity: optionletstripper.cpp:43-46."""
    assert _REF["guards"]["normal_with_displacement_throws"] is True
    eval_date = Date(_REF["euribor3m_flat18"]["eval_date_serial"])
    ObservableSettings().evaluation_date = eval_date
    curve = FlatForward(eval_date, SimpleQuote(0.03), Actual365Fixed())
    index = Euribor(Period(3, _M), curve)
    with pytest.raises(LibraryException, match="displacement is not allowed with Normal"):
        OptionletStripper1(
            _guard_surface(eval_date, _TENORS_4),
            index,
            volatility_type=VolatilityType.Normal,
            displacement=0.01,
        )


def test_overnight_index_without_optionlet_frequency_is_rejected() -> None:
    """# C++ parity: optionletstripper.cpp:48-51."""
    assert _REF["guards"]["overnight_without_frequency_throws"] is True
    eval_date = Date(_REF["euribor3m_flat18"]["eval_date_serial"])
    ObservableSettings().evaluation_date = eval_date
    curve = FlatForward(eval_date, SimpleQuote(0.03), Actual365Fixed())
    with pytest.raises(LibraryException, match="optionlet frequency is required"):
        OptionletStripper1(_guard_surface(eval_date, _TENORS_4), Eonia(curve))


def test_overnight_index_with_optionlet_frequency_walks_by_the_frequency() -> None:
    """# C++ parity: optionletstripper.cpp:58 — the 1-day index tenor is bypassed."""
    guards = _REF["guards"]
    assert guards["overnight_with_frequency_throws"] is False
    eval_date = Date(_REF["euribor3m_flat18"]["eval_date_serial"])
    ObservableSettings().evaluation_date = eval_date
    curve = FlatForward(eval_date, SimpleQuote(0.03), Actual365Fixed())
    stripper = OptionletStripper1(
        _guard_surface(eval_date, _TENORS_4),
        Eonia(curve),
        optionlet_frequency=Period(1, _Y),
    )
    assert stripper.optionlet_maturities() == (
        guards["overnight_with_frequency_maturities"]
    )
    assert [_period_str(p) for p in stripper.optionlet_fixing_tenors()] == (
        guards["overnight_with_frequency_tenors"]
    )


def test_surface_too_short_for_the_first_cap_is_rejected() -> None:
    """# C++ parity: optionletstripper.cpp:64-66."""
    assert _REF["guards"]["too_short_surface_throws"] is True
    eval_date = Date(_REF["euribor3m_flat18"]["eval_date_serial"])
    ObservableSettings().evaluation_date = eval_date
    curve = FlatForward(eval_date, SimpleQuote(0.03), Actual365Fixed())
    index = Euribor(Period(3, _M), curve)
    with pytest.raises(LibraryException, match="too short"):
        OptionletStripper1(
            _guard_surface(eval_date, [Period(6, _M), Period(1, _Y)]),
            index,
            optionlet_frequency=Period(1, _Y),
        )


# --------------------------------------------------------------------------
# The lazy-recalculation contract (optionletstripper.cpp:87-137 vs :106-122).
# --------------------------------------------------------------------------


def _fresh() -> OptionletStripper1:
    return _euribor3m_flat18()[1]


def test_constructor_time_accessors_do_not_calculate() -> None:
    """``optionletFixingTenors``/``optionletMaturities`` skip ``calculate()``.

    # C++ parity: optionletstripper.cpp:106-108 and :120-122 — neither calls
    # calculate(), because both read constructor-time state.
    """
    stripper = _fresh()
    assert stripper._calculated is False  # pyright: ignore[reportPrivateUsage]
    _ = stripper.optionlet_maturities()
    _ = stripper.optionlet_fixing_tenors()
    _ = stripper.optionlet_frequency()
    _ = stripper.day_counter()
    _ = stripper.calendar()
    _ = stripper.business_day_convention()
    _ = stripper.displacement()
    _ = stripper.volatility_type()
    assert stripper._calculated is False  # pyright: ignore[reportPrivateUsage]


# Every accessor below opens with ``calculate();`` in
# optionletstripper.cpp:87-137.
_LAZY_READERS: list[tuple[str, Callable[[OptionletStripper], object]]] = [
    ("atm_optionlet_rates", lambda s: s.atm_optionlet_rates()),
    ("optionlet_strikes", lambda s: s.optionlet_strikes(0)),
    ("optionlet_volatilities", lambda s: s.optionlet_volatilities(0)),
    ("optionlet_fixing_dates", lambda s: s.optionlet_fixing_dates()),
    ("optionlet_fixing_times", lambda s: s.optionlet_fixing_times()),
    ("optionlet_payment_dates", lambda s: s.optionlet_payment_dates()),
    ("optionlet_accrual_periods", lambda s: s.optionlet_accrual_periods()),
]


@pytest.mark.parametrize(
    ("name", "reader"), _LAZY_READERS, ids=[n for n, _ in _LAZY_READERS]
)
def test_reading_a_mutable_vector_triggers_performcalculations(
    name: str,
    reader: Callable[[OptionletStripper], object],
) -> None:
    """# C++ parity: every one of these opens with ``calculate();``."""
    stripper = _fresh()
    assert stripper._calculated is False, name  # pyright: ignore[reportPrivateUsage]
    reader(stripper)
    assert stripper._calculated is True, name  # pyright: ignore[reportPrivateUsage]


def test_pre_calculation_vectors_are_sized_and_zeroed_not_garbage() -> None:
    """The ctor SIZES the mutable vectors; it does not fill them.

    # C++ parity: optionletstripper.cpp:75-84 — value-initialised, so a
    # pre-calculation read yields null Dates and zeros, never uninitialised
    # storage. The public accessors never expose this state (they all
    # calculate first); the invariant is asserted on the members directly.
    """
    stripper = _fresh()
    n = stripper.optionlet_maturities()
    assert len(stripper._optionlet_dates) == n  # pyright: ignore[reportPrivateUsage]
    assert all(d == Date() for d in stripper._optionlet_dates)  # pyright: ignore[reportPrivateUsage]
    assert stripper._optionlet_times == [0.0] * n  # pyright: ignore[reportPrivateUsage]
    assert stripper._optionlet_accrual_periods == [0.0] * n  # pyright: ignore[reportPrivateUsage]
    assert stripper._atm_optionlet_rate == [0.0] * n  # pyright: ignore[reportPrivateUsage]
    assert all(d == Date() for d in stripper._optionlet_payment_dates)  # pyright: ignore[reportPrivateUsage]
    # optionletVolatilities_ is vector<Volatility>(nStrikes_) => zeros.
    assert stripper._optionlet_volatilities == [[0.0] * 3 for _ in range(n)]  # pyright: ignore[reportPrivateUsage]


def test_switch_strike_only_calculates_when_it_floats() -> None:
    """# C++ parity: optionletstripper1.cpp:199-203."""
    pinned = _euribor3m_flat18()[1]
    # A pinned switch strike is final; no calculation needed.
    expected, _ = _euribor3m_flat18()
    eval_date = Date(expected["eval_date_serial"])
    ObservableSettings().evaluation_date = eval_date
    curve = FlatForward(eval_date, SimpleQuote(0.03), Actual365Fixed())
    fixed = OptionletStripper1(
        _guard_surface(eval_date, _TENORS_4),
        Euribor(Period(3, _M), curve),
        switch_strike=0.035,
    )
    assert fixed.switch_strike() == 0.035
    assert fixed._calculated is False  # pyright: ignore[reportPrivateUsage]
    # A floating one must strip first.
    assert pinned._calculated is False  # pyright: ignore[reportPrivateUsage]
    _ = pinned.switch_strike()
    assert pinned._calculated is True  # pyright: ignore[reportPrivateUsage]


# --------------------------------------------------------------------------
# OptionletStripper2 shares the base, and rebuilds its own grid.
# --------------------------------------------------------------------------


def test_stripper2_rebuilds_the_same_base_grid_as_its_stripper1() -> None:
    """# C++ parity: optionletstripper2.cpp:39-44 — the base is re-derived."""
    from pquantlib.termstructures.volatility.capfloor.cap_floor_term_vol_curve import (  # noqa: PLC0415
        CapFloorTermVolCurve,
    )
    from pquantlib.termstructures.volatility.optionlet.optionlet_stripper_2 import (  # noqa: PLC0415
        OptionletStripper2,
    )

    expected, stripper1 = _euribor3m_flat18()
    eval_date = Date(expected["eval_date_serial"])
    curve = CapFloorTermVolCurve(
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        option_tenors=[Period(1, _Y), Period(2, _Y), Period(3, _Y)],
        vols=[0.18, 0.18, 0.18],
        calendar=TARGET(),
        day_counter=Actual365Fixed(),
        reference_date=eval_date,
    )
    stripper2 = OptionletStripper2(
        optionlet_stripper_1=stripper1, atm_cap_floor_term_vol_curve=curve
    )
    # Constructor-time state, so available without any calculation.
    assert stripper2._calculated is False  # pyright: ignore[reportPrivateUsage]
    assert stripper2.optionlet_maturities() == expected["optionlet_maturities"]
    assert [_period_str(p) for p in stripper2.optionlet_fixing_tenors()] == (
        expected["optionlet_fixing_tenors"]
    )
    assert stripper2._calculated is False  # pyright: ignore[reportPrivateUsage]
    # ...and the copied date grid is stripper1's, per optionletstripper2.cpp:60-64.
    assert stripper2.optionlet_fixing_dates() == stripper1.optionlet_fixing_dates()
    assert stripper2.optionlet_payment_dates() == stripper1.optionlet_payment_dates()
    assert stripper2.atm_optionlet_rates() == stripper1.atm_optionlet_rates()
    assert stripper2._calculated is True  # pyright: ignore[reportPrivateUsage]


# --------------------------------------------------------------------------
# Known, recorded divergences. These assert the CURRENT PQuantLib answer AND
# the C++ answer so the gap is visible and cannot silently widen or vanish.
# --------------------------------------------------------------------------


def test_normal_branch_divergence_is_known() -> None:
    """The shortest Normal optionlet at the far strike comes back 0 here.

    C++ ``bachelierBlackFormulaImpliedVol`` recovers the 0.006 input
    (optionletstripper1.cpp:148-154); PQuantLib's
    ``bachelier_black_formula_implied_vol`` returns 0 for the same inputs.
    Every other cell in the Normal grid agrees to 5.2e-8 or better. Flagged
    for align(pricingengines/black_formula); NOT a base-class issue.
    """
    expected, stripper = _euribor3m_normal()
    assert abs(float(expected["optionlet_volatilities"][0][2]) - 0.006) < 1e-12
    assert stripper.optionlet_volatilities(0)[2] == 0.0  # DIVERGENCE


def test_cap_floor_term_vol_lookup_divergence_is_known() -> None:
    """``CapFloorTermVolSurface`` interpolates the TIME axis differently.

    The probe emits the very term vols the stripper reads back
    (optionletstripper1.cpp:128-129) for every ``capFloorLengths_[i]`` of the
    ``euribor6m_holiday_evaldate`` scenario. The pattern is unambiguous:

    * at a pillar tenor (12M/24M/36M/60M/84M, i.e. i in {0, 2, 4, 8, 12}) the
      two agree bit-for-bit — TIGHT below;
    * between pillars they differ by up to 3.0e-3 relative (worst at
      48M: 0.20500000000000002 here vs 0.2043870577586211 in C++).

    So the abscissa PQuantLib interpolates on is not C++'s. That is upstream
    of every stripper: it is
    ``pquantlib/src/pquantlib/termstructures/volatility/capfloor/
    cap_floor_term_vol_surface.py``, and it is what makes the stripped vols of
    this scenario diverge by up to 1.6e-2 relative. Flagged for
    align(termstructures/volatility/capfloor); NOT a base-class issue, which
    is why the whole date grid of this same scenario still matches EXACTLY.
    """
    expected, _ = _euribor6m_holiday()
    eval_date = Date(expected["eval_date_serial"])
    ObservableSettings().evaluation_date = eval_date
    vols = np.array([[0.22 - 0.006 * i + 0.011 * j for j in range(3)] for i in range(6)])
    surface = CapFloorTermVolSurface(
        business_day_convention=BusinessDayConvention.Following,
        option_tenors=[
            Period(1, _Y),
            Period(2, _Y),
            Period(3, _Y),
            Period(5, _Y),
            Period(7, _Y),
            Period(10, _Y),
        ],
        strikes=[0.01, 0.03, 0.05],
        volatilities=vols,
        calendar=TARGET(),
        day_counter=Actual360(),
        reference_date=eval_date,
    )
    strikes = [0.01, 0.03, 0.05]
    at_pillar = {0, 2, 4, 8, 12}  # 12M, 24M, 36M, 60M, 84M
    off_pillar_deviations: list[float] = []
    for i in range(19):
        length = Period((i + 2) * 6, _M)
        row = [surface.volatility(length, k, True) for k in strikes]
        want_row = expected["cap_floor_term_vols"][i]
        if i in at_pillar:
            for got, want in zip(row, want_row, strict=True):
                tolerance.tight(got, want)
        else:
            off_pillar_deviations.extend(
                abs(g - w) / abs(w) for g, w in zip(row, want_row, strict=True)
            )
    worst_off_pillar = max(off_pillar_deviations)
    # DIVERGENCE: pinned so it cannot silently widen (or be "fixed" unnoticed).
    assert 1.0e-5 < worst_off_pillar < 4.0e-3


def test_moving_reference_vol_divergence_is_known() -> None:
    """On a moving-reference surface the stripped vols do not agree.

    Flat 18% inputs: PQuantLib strips 0.18 for every row, C++ ranges from
    0.181968 (row 0) down to 0.180104 (row 18). The date grid, ATM forwards
    and switch strike of this same scenario all match EXACTLY (see the tests
    above), so the base class is not implicated — the moving-reference
    surface's time axis is, which is the same root cause as
    ``test_cap_floor_term_vol_lookup_divergence_is_known``. Recorded with both
    numbers; flagged for align(termstructures/volatility/capfloor).
    """
    expected, stripper = _euribor3m_moving()
    tolerance.exact(float(expected["optionlet_volatilities"][0][0]), 0.18196777241015452)
    tolerance.exact(float(expected["optionlet_volatilities"][18][0]), 0.18010390074698665)
    for i in range(stripper.optionlet_maturities()):
        for got in stripper.optionlet_volatilities(i):
            assert abs(got - 0.18) < 1e-5  # PQuantLib — DIVERGENCE

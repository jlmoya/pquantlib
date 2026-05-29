"""CPICapFloor smoke + structural tests.

CPICapFloor's pricing engine is deferred to Phase 8+; here we exercise the
instrument shell + ``setup_arguments`` + lag-consistency checks.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from pquantlib.indexes.inflation.cpi import InterpolationType
from pquantlib.indexes.inflation.uk_rpi import UKRPI
from pquantlib.instruments.cpi_cap_floor import CPICapFloor, CPICapFloorArguments
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import OptionType
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_EVAL_DATE = Date.from_ymd(17, Month.January, 2024)


@pytest.fixture(autouse=True)
def _pin_eval_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    s = ObservableSettings()
    old = s.evaluation_date
    s.evaluation_date = _EVAL_DATE
    yield
    s.evaluation_date = old


def _build_cap(type_: OptionType = OptionType.Call) -> CPICapFloor:
    cal = TARGET()
    return CPICapFloor(
        type_=type_,
        nominal=1_000_000.0,
        start_date=_EVAL_DATE,
        base_cpi=100.0,
        maturity=Date.from_ymd(17, Month.January, 2029),
        fix_calendar=cal,
        fix_convention=BusinessDayConvention.ModifiedFollowing,
        pay_calendar=cal,
        pay_convention=BusinessDayConvention.ModifiedFollowing,
        strike=0.025,
        inflation_index=UKRPI(),
        observation_lag=Period(3, TimeUnit.Months),
        observation_interpolation=InterpolationType.AsIndex,
    )


def test_cpi_cap_floor_setup_arguments_populates_fields() -> None:
    cap = _build_cap(OptionType.Call)
    args = CPICapFloorArguments()
    cap.setup_arguments(args)
    args.validate()  # no-op per C++
    assert args.type == OptionType.Call
    assert args.nominal == 1_000_000.0
    assert args.base_cpi == 100.0
    assert args.strike == 0.025
    assert args.observation_lag == Period(3, TimeUnit.Months)


def test_cpi_cap_floor_is_expired_only_after_maturity() -> None:
    cap = _build_cap()
    s = ObservableSettings()
    # Eval date < maturity → not expired.
    assert not cap.is_expired()
    # Eval date > maturity → expired.
    s.evaluation_date = Date.from_ymd(18, Month.January, 2029)
    assert cap.is_expired()


def test_cpi_cap_floor_fixing_and_payment_dates() -> None:
    cap = _build_cap()
    # Probe: fixing = fix_calendar.adjust(maturity - obsLag, fix_conv)
    # maturity = 2029-01-17, obsLag = 3M → 2028-10-17.
    expected_fixing = TARGET().adjust(
        Date.from_ymd(17, Month.October, 2028),
        BusinessDayConvention.ModifiedFollowing,
    )
    assert cap.fixing_date() == expected_fixing
    expected_pay = TARGET().adjust(
        Date.from_ymd(17, Month.January, 2029),
        BusinessDayConvention.ModifiedFollowing,
    )
    assert cap.pay_date() == expected_pay


def test_cpi_cap_floor_lag_consistency_rejects_underflag() -> None:
    """observation_lag must be >= index availability lag (effectively flat)."""
    cal = TARGET()
    with pytest.raises(Exception, match="observation lag"):
        CPICapFloor(
            type_=OptionType.Call,
            nominal=1.0,
            start_date=_EVAL_DATE,
            base_cpi=100.0,
            maturity=Date.from_ymd(17, Month.January, 2029),
            fix_calendar=cal,
            fix_convention=BusinessDayConvention.ModifiedFollowing,
            pay_calendar=cal,
            pay_convention=BusinessDayConvention.ModifiedFollowing,
            strike=0.025,
            inflation_index=UKRPI(),
            observation_lag=Period(0, TimeUnit.Days),  # 0 < 1M (UKRPI lag)
            observation_interpolation=InterpolationType.AsIndex,
        )


def test_cpi_cap_floor_inspectors() -> None:
    cap = _build_cap()
    assert cap.type() == OptionType.Call
    assert cap.nominal() == 1_000_000.0
    assert cap.strike() == 0.025
    assert cap.observation_lag() == Period(3, TimeUnit.Months)

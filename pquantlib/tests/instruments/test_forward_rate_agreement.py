"""Tests for ForwardRateAgreement.

Reference values come from ``cluster/l3e.json`` emitted by
``migration-harness/cpp/probes/cluster_l3e/probe.cpp``.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.instruments.forward_rate_agreement import ForwardRateAgreement
from pquantlib.position import PositionType
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import tight
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit
from tests.indexes._mock_curves import FlatForwardMock


@pytest.fixture(scope="module")
def ref() -> dict[str, Any]:
    return reference_reader.load("cluster/l3e")


# --- ForwardRateAgreement construction validation ------------------------


def test_fra_negative_notional_rejected() -> None:
    eval_date = Date.from_ymd(17, Month.January, 2024)
    ts = FlatForwardMock(eval_date, 0.05, Actual360())
    idx = Euribor.three_months(ts)
    with pytest.raises(LibraryException, match="notionalAmount must be positive"):
        ForwardRateAgreement(
            index=idx,
            value_date=Date.from_ymd(19, Month.April, 2024),
            position_type=PositionType.Long,
            strike_forward_rate=0.05,
            notional_amount=-1.0,
        )


def test_fra_value_date_after_maturity_rejected() -> None:
    eval_date = Date.from_ymd(17, Month.January, 2024)
    ts = FlatForwardMock(eval_date, 0.05, Actual360())
    idx = Euribor.three_months(ts)
    with pytest.raises(LibraryException, match="earlier than maturityDate"):
        ForwardRateAgreement(
            index=idx,
            value_date=Date.from_ymd(17, Month.July, 2024),
            position_type=PositionType.Long,
            strike_forward_rate=0.05,
            notional_amount=1_000_000.0,
            # Explicit maturity earlier than value
            maturity_date=Date.from_ymd(17, Month.April, 2024),
        )


# --- ForwardRateAgreement par-coupon branch ------------------------------


def test_fra_par_coupon_branch_matches_probe(ref: dict[str, Any]) -> None:
    """useIndexedCoupon=False / par-coupon branch.

    3M Euribor, 3M into the future, strike 5%, notional 1M EUR.
    FlatForward(5%) Actual360 continuous (matching the C++ probe).
    """
    eval_date = Date.from_ymd(17, Month.January, 2024)
    ts = FlatForward.from_rate(
        eval_date, 0.05, Actual360(),
        Compounding.Continuous, Frequency.Annual,
    )
    idx = Euribor.three_months(ts)

    value_date = TARGET().advance(
        eval_date, 3, TimeUnit.Months,
        idx.business_day_convention(),
        idx.end_of_month(),
    )
    maturity_date = idx.maturity_date(value_date)

    # Sanity-check the probe used the same dates.
    expected_fra = ref["forward_rate_agreement"]
    assert value_date.serial_number() == expected_fra["value_date_serial"]
    assert maturity_date.serial_number() == expected_fra["maturity_date_serial"]

    fra = ForwardRateAgreement(
        index=idx,
        value_date=value_date,
        maturity_date=maturity_date,  # par-coupon branch
        position_type=PositionType.Long,
        strike_forward_rate=0.05,
        notional_amount=1_000_000.0,
    )

    expected_par = expected_fra["par"]
    tight(fra.forward_rate().rate(), float(expected_par["forward_rate"]))
    tight(fra.amount(), float(expected_par["amount"]))
    tight(fra.npv(), float(expected_par["npv"]))


def test_fra_indexed_coupon_branch_matches_probe(ref: dict[str, Any]) -> None:
    """useIndexedCoupon=True / indexed-coupon branch.

    Same setup as par-coupon test; the index's natural-tenor fixing
    happens to coincide because the FlatForward curve is rate-uniform.
    """
    eval_date = Date.from_ymd(17, Month.January, 2024)
    ts = FlatForward.from_rate(
        eval_date, 0.05, Actual360(),
        Compounding.Continuous, Frequency.Annual,
    )
    idx = Euribor.three_months(ts)

    value_date = TARGET().advance(
        eval_date, 3, TimeUnit.Months,
        idx.business_day_convention(),
        idx.end_of_month(),
    )

    fra = ForwardRateAgreement(
        index=idx,
        value_date=value_date,
        # No explicit maturity_date → useIndexedCoupon=True branch
        position_type=PositionType.Long,
        strike_forward_rate=0.05,
        notional_amount=1_000_000.0,
    )

    expected_indexed = ref["forward_rate_agreement"]["indexed"]
    tight(fra.forward_rate().rate(), float(expected_indexed["forward_rate"]))
    tight(fra.amount(), float(expected_indexed["amount"]))
    tight(fra.npv(), float(expected_indexed["npv"]))


def test_fra_short_position_flips_amount_sign() -> None:
    """Short FRA: amount is negated relative to Long."""
    eval_date = Date.from_ymd(17, Month.January, 2024)
    ts = FlatForwardMock(eval_date, 0.05, Actual360())
    idx = Euribor.three_months(ts)
    value_date = TARGET().advance(
        eval_date, Period(3, TimeUnit.Months).length, TimeUnit.Months,
        idx.business_day_convention(),
        idx.end_of_month(),
    )
    maturity_date = idx.maturity_date(value_date)

    long_fra = ForwardRateAgreement(
        index=idx,
        value_date=value_date,
        maturity_date=maturity_date,
        position_type=PositionType.Long,
        strike_forward_rate=0.04,
        notional_amount=1_000_000.0,
    )
    short_fra = ForwardRateAgreement(
        index=idx,
        value_date=value_date,
        maturity_date=maturity_date,
        position_type=PositionType.Short,
        strike_forward_rate=0.04,
        notional_amount=1_000_000.0,
    )
    tight(long_fra.amount(), -short_fra.amount())


def test_fra_inspectors_match_index() -> None:
    eval_date = Date.from_ymd(17, Month.January, 2024)
    ts = FlatForwardMock(eval_date, 0.05, Actual360())
    idx = Euribor.three_months(ts)
    value_date = TARGET().advance(
        eval_date, 3, TimeUnit.Months,
        idx.business_day_convention(),
        idx.end_of_month(),
    )
    fra = ForwardRateAgreement(
        index=idx,
        value_date=value_date,
        position_type=PositionType.Long,
        strike_forward_rate=0.05,
        notional_amount=1_000_000.0,
    )
    assert fra.day_counter().name() == idx.day_counter().name()
    assert fra.business_day_convention() == idx.business_day_convention()
    assert fra.value_date() == value_date
    # fixing_date precedes value_date by fixing_days BD
    assert fra.fixing_date().serial < value_date.serial


def test_fra_is_not_expired_before_settlement() -> None:
    eval_date = Date.from_ymd(17, Month.January, 2024)
    ts = FlatForwardMock(eval_date, 0.05, Actual360())
    idx = Euribor.three_months(ts)
    value_date = TARGET().advance(
        eval_date, 3, TimeUnit.Months,
        idx.business_day_convention(),
        idx.end_of_month(),
    )
    fra = ForwardRateAgreement(
        index=idx,
        value_date=value_date,
        position_type=PositionType.Long,
        strike_forward_rate=0.05,
        notional_amount=1_000_000.0,
    )
    # PQuantLib divergence: is_expired() always False without Settings.
    assert fra.is_expired() is False

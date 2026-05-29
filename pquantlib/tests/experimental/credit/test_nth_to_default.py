"""Cross-validate NthToDefault inspectors against the C++ reference.

Probe source: migration-harness/cpp/probes/cluster_w3d/probe.cpp
Reference:    migration-harness/references/cluster/w3d.json
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.experimental.credit.basket_protocol import BasketProtocol
from pquantlib.experimental.credit.nth_to_default import (
    NthToDefault,
    NthToDefaultArguments,
)
from pquantlib.instruments.claim import Claim, FaceValueClaim
from pquantlib.instruments.credit_default_swap import ProtectionSide
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import MakeSchedule
from pquantlib.time.time_unit import TimeUnit


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/w3d")


class _StubBasket:
    """Minimal in-test stub satisfying ``BasketProtocol``.

    Honest probabilities are not provided here (any non-trivial
    integration is in ``test_integral_ntd_engine``); this stub is
    sufficient for inspector / construction round-trips.
    """

    def __init__(
        self,
        ref_date: Date,
        names: list[str],
        notional_per_name: float = 1.0e6,
        recovery: float = 0.4,
    ) -> None:
        self._ref_date = ref_date
        self._names = list(names)
        self._notional_per_name = notional_per_name
        self._recovery = recovery
        self._claim: Claim = FaceValueClaim()

    def size(self) -> int:
        return len(self._names)

    def names(self) -> list[str]:
        return list(self._names)

    def ref_date(self) -> Date:
        return self._ref_date

    def claim(self) -> Claim:
        return self._claim

    def remaining_size(self) -> int:
        return self.size()

    def remaining_notional(self) -> float:
        return self.size() * self._notional_per_name

    def recovery_rate(self, d: Date, i: int) -> float:
        del d, i
        return self._recovery

    def prob_at_least_n_events(self, n: int, d: Date) -> float:
        del n, d
        return 0.0


def test_nth_to_default_inspectors_match_cpp(cpp_ref: dict[str, Any]) -> None:
    """C++ reference: ``cluster_w3d.probe.cpp`` section 1 (NTD inspectors).

    All inspector round-trips are TIGHT (no floating-point math, just
    field copies).
    """
    today = Date.from_ymd(15, Month.January, 2024)
    cal = TARGET()
    start = cal.advance(today, 1, TimeUnit.Days)
    end = start + Period(1, TimeUnit.Years)
    sched = (
        MakeSchedule()
        .from_date(start)
        .to(end)
        .with_calendar(cal)
        .with_tenor(Period(3, TimeUnit.Months))
        .with_convention(BusinessDayConvention.Unadjusted)
        .backwards()
        .build()
    )
    basket: BasketProtocol = _StubBasket(today, ["Name0", "Name1", "Name2"])
    ntd = NthToDefault(
        basket=basket,
        n=2,
        side=ProtectionSide.Buyer,
        premium_schedule=sched,
        upfront_rate=0.0,
        premium_rate=0.0050,
        day_counter=Actual365Fixed(),
        nominal=1.0e6,
        settle_premium_accrual=True,
    )

    ref = cpp_ref["ntd"]
    tolerance.tight(ntd.premium(), ref["premium"])
    tolerance.tight(ntd.nominal(), ref["nominal"])
    assert ntd.rank() == ref["rank"]
    assert ntd.basket_size() == ref["basket_size"]
    assert int(ntd.side()) == ref["side_idx"]
    # Maturity matches the end-of-schedule serial (TIGHT — both come from
    # the same calendar adjustment).
    assert ntd.maturity().serial_number() == ref["maturity_serial"]


def test_nth_to_default_premium_leg_built() -> None:
    """Construction builds the fixed-rate premium leg eagerly.

    # C++ parity: nthtodefault.cpp:56-59. The leg must have len ==
    # number of coupon periods (quarterly 1y schedule → 4 coupons).
    """
    today = Date.from_ymd(15, Month.January, 2024)
    cal = TARGET()
    start = cal.advance(today, 1, TimeUnit.Days)
    sched = (
        MakeSchedule()
        .from_date(start)
        .to(start + Period(1, TimeUnit.Years))
        .with_calendar(cal)
        .with_tenor(Period(3, TimeUnit.Months))
        .with_convention(BusinessDayConvention.Unadjusted)
        .backwards()
        .build()
    )
    basket: BasketProtocol = _StubBasket(today, ["A", "B", "C"])
    ntd = NthToDefault(
        basket=basket,
        n=1,
        side=ProtectionSide.Seller,
        premium_schedule=sched,
        upfront_rate=0.0,
        premium_rate=0.01,
        day_counter=Actual365Fixed(),
        nominal=1.0e6,
        settle_premium_accrual=False,
    )
    # Quarterly schedule over 1y = 4 coupons (the schedule has 5 dates).
    assert len(ntd.premium_leg()) == 4


def test_nth_to_default_rank_validation() -> None:
    """Constructing with rank > basket size must raise.

    # C++ parity: nthtodefault.cpp:48-49.
    """
    today = Date.from_ymd(15, Month.January, 2024)
    cal = TARGET()
    start = cal.advance(today, 1, TimeUnit.Days)
    sched = (
        MakeSchedule()
        .from_date(start)
        .to(start + Period(1, TimeUnit.Years))
        .with_calendar(cal)
        .with_tenor(Period(3, TimeUnit.Months))
        .with_convention(BusinessDayConvention.Unadjusted)
        .backwards()
        .build()
    )
    basket = _StubBasket(today, ["A", "B"])
    with pytest.raises(LibraryException):
        NthToDefault(
            basket=basket,
            n=3,
            side=ProtectionSide.Buyer,
            premium_schedule=sched,
            upfront_rate=0.0,
            premium_rate=0.005,
            day_counter=Actual365Fixed(),
            nominal=1.0e6,
            settle_premium_accrual=True,
        )


def test_nth_to_default_args_validate() -> None:
    """``NthToDefaultArguments.validate`` rejects missing fields."""
    args = NthToDefaultArguments()
    with pytest.raises(LibraryException):
        args.validate()

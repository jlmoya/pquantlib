"""RiskyBondEngine cross-validation against C++ QuantLib v1.43.

Reference: ``migration-harness/references/v143/pe/bondswap.json`` (probe
``migration-harness/cpp/probes/v143_pe_bondswap/probe.cpp``, "PART 3").

The grid brackets the engine: a zero hazard rate (where the engine must
collapse onto ``DiscountingBondEngine`` exactly), a high hazard rate, and
recovery rates of 0, 0.4 and 1.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.thirty_360 import Convention, Thirty360
from pquantlib.instruments.bonds.fixed_rate_bond import FixedRateBond
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.bond.discounting_bond_engine import DiscountingBondEngine
from pquantlib.pricingengines.bond.risky_bond_engine import RiskyBondEngine
from pquantlib.termstructures.credit.flat_hazard_rate import FlatHazardRate
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
from pquantlib.time.time_unit import TimeUnit

CPP: dict[str, Any] = reference_reader.load("v143/pe/bondswap")

# probe.cpp — `const Date kToday(15, May, 2025);`
TODAY = Date.from_ymd(15, Month.May, 2025)
DC_365 = Actual365Fixed()
DC_30_360 = Thirty360(Convention.BondBasis)


@pytest.fixture(autouse=True)
def _eval_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    settings = ObservableSettings()
    saved = settings.evaluation_date
    # probe.cpp:main() — Settings::instance().evaluationDate() = Date(15, May, 2025)
    settings.evaluation_date = TODAY
    yield
    settings.evaluation_date = saved


def _bond() -> FixedRateBond:
    """probe.cpp `makeFixedBond()` — 10y semiannual 4%, 30/360 BondBasis."""
    schedule = Schedule.from_rule(
        effective_date=Date.from_ymd(15, Month.January, 2020),
        termination_date=Date.from_ymd(15, Month.January, 2030),
        tenor=Period(6, TimeUnit.Months),
        calendar=TARGET(),
        convention=BusinessDayConvention.Unadjusted,
        termination_date_convention=BusinessDayConvention.Unadjusted,
        rule=DateGeneration.Backward,
        end_of_month=False,
    )
    return FixedRateBond(
        settlement_days=3,
        face_amount=100.0,
        schedule=schedule,
        coupons=[0.04],
        accrual_day_counter=DC_30_360,
        payment_convention=BusinessDayConvention.Following,
        redemption=100.0,
        issue_date=Date.from_ymd(15, Month.January, 2020),
    )


def _yield_curve() -> FlatForward:
    """probe.cpp `runRiskyBondEngine` — flat 3.5%, Actual365Fixed, Compounded/Annual."""
    return FlatForward.from_rate(TODAY, 0.035, DC_365, Compounding.Compounded, Frequency.Annual)


_CASES = sorted(
    k
    for k in CPP
    if k.startswith("risky_bond_") and k != "risky_bond_riskfree_reference"
)


@pytest.mark.parametrize("case", _CASES)
def test_risky_bond_prices(case: str) -> None:
    inputs = CPP[case]["inputs"]
    expected = CPP[case]["expected"]
    bond = _bond()
    default_ts = FlatHazardRate.from_rate(TODAY, float(inputs["hazard_rate"]), DC_365)
    engine = RiskyBondEngine(default_ts, float(inputs["recovery_rate"]), _yield_curve())
    bond.set_pricing_engine(engine)

    tolerance.tight(bond.npv(), expected["npv"])
    tolerance.tight(bond.settlement_value(), expected["settlement_value"])
    tolerance.tight(bond.clean_price(), expected["clean_price"])
    tolerance.tight(bond.dirty_price(), expected["dirty_price"])
    assert bond.settlement_date() == Date(int(expected["settlement_date"]))
    assert bond.valuation_date() == TODAY

    # Inspectors round-trip the constructor arguments.
    assert engine.default_ts() is default_ts
    tolerance.exact(engine.recovery_rate(), float(inputs["recovery_rate"]))


def test_zero_hazard_equals_the_risk_free_price() -> None:
    """Survival == 1 and default probability == 0 => the risk-free value."""
    riskless = CPP["risky_bond_riskfree_reference"]["expected"]
    for case in ("risky_bond_zero_hazard_rec40", "risky_bond_zero_hazard_rec0"):
        # C++ itself agrees to the last bit — the recovery rate is multiplied
        # by a zero default probability, so it cannot matter.
        tolerance.exact(CPP[case]["expected"]["npv"], riskless["npv"])
        tolerance.exact(CPP[case]["expected"]["settlement_value"], riskless["settlement_value"])

    bond = _bond()
    bond.set_pricing_engine(RiskyBondEngine(FlatHazardRate.from_rate(TODAY, 0.0, DC_365), 0.4,
                                            _yield_curve()))
    risky_npv = bond.npv()

    reference = _bond()
    reference.set_pricing_engine(DiscountingBondEngine(_yield_curve()))
    tolerance.tight(risky_npv, reference.npv())
    tolerance.tight(reference.npv(), riskless["npv"])


def test_higher_hazard_lowers_the_price() -> None:
    ordered = [
        CPP["risky_bond_zero_hazard_rec40"]["expected"]["npv"],
        CPP["risky_bond_low_hazard"]["expected"]["npv"],
        CPP["risky_bond_high_hazard"]["expected"]["npv"],
    ]
    assert ordered == sorted(ordered, reverse=True)


def test_recovery_rate_is_wired() -> None:
    """Three recovery rates at the same hazard rate must give three prices."""
    values = {
        CPP[case]["expected"]["npv"]
        for case in (
            "risky_bond_high_hazard_zero_recovery",
            "risky_bond_high_hazard",
            "risky_bond_high_hazard_full_recovery",
        )
    }
    assert len(values) == 3

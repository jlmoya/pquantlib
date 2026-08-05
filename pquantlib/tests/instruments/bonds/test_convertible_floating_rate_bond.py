"""Cross-validate ConvertibleFloatingRateBond against C++ v1.43.

Probe: ``v143/inst/bondsbtp`` (section ``convertible_frn``).

The binomial Tsiveriotis-Fernandes engine has no analytic counterpart, so the
reference IS the C++ same-engine value at the SAME step count — tree against
identical tree, hence TIGHT.

The grid varies the conversion ratio, the credit spread, the spread on the
floating leg and the call/put schedule, and additionally moves EVERY optional
constructor argument off its default in turn (``redemption``,
``ex_coupon_period``, ``ex_coupon_calendar``, ``ex_coupon_convention``,
``ex_coupon_end_of_month``) plus the non-default ``fixing_days`` /
``settlement_days`` values, each with a C++-pinned observable consequence.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, NamedTuple, cast

import pytest

from pquantlib.cashflows.coupon import Coupon
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exercise import AmericanExercise, EuropeanExercise, Exercise
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.instruments.bond import BondPrice, BondPriceType
from pquantlib.instruments.bonds.convertible_bonds import ConvertibleFloatingRateBond
from pquantlib.instruments.callability import Callability, CallabilityType
from pquantlib.instruments.soft_callability import SoftCallability
from pquantlib.methods.lattices.binomial_tree import CoxRossRubinstein
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.bond.binomial_convertible_engine import (
    BinomialConvertibleEngine,
)
from pquantlib.processes.black_scholes_merton_process import BlackScholesMertonProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.termstructures.yield_.interpolated_zero_curve import InterpolatedZeroCurve
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit

# Per-variant deviations from the base case; the empty dict IS the base
# (every optional argument at its C++ default).
VARIANTS: dict[str, dict[str, Any]] = {
    "base": {},
    "american": {"american": True},
    "conv_ratio_3": {"conversion_ratio": 3.0},
    "credit_spread_2pc": {"credit_spread": 0.020},
    "leg_spread_2pc": {"leg_spread": 0.020},
    "call_put": {"american": True, "call_put": True},
    "redemption_102": {"redemption": 102.0},
    "fixing_days_5": {"fixing_days": 5},
    "settlement_days_10": {"settlement_days": 10},
    "ex_coupon_1m": {"ex": True},
    "ex_coupon_1m_preceding": {"ex": True, "ex_conv": BusinessDayConvention.Preceding},
    "ex_coupon_1m_nullcal_preceding": {
        "ex": True,
        "ex_conv": BusinessDayConvention.Preceding,
        "ex_null_calendar": True,
    },
    "ex_coupon_1m_eom": {"ex": True, "ex_eom": True},
}


class Setup(NamedTuple):
    """Everything the variants share."""

    ref: dict[str, Any]
    euribor: Euribor
    schedule: Schedule
    issue: Date
    maturity: Date
    today: Date


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/inst/bondsbtp")


@pytest.fixture(scope="module")
def setup(cpp: dict[str, Any]) -> Iterator[Setup]:
    mk = cpp["market"]
    ref = cpp["convertible_frn"]
    settings = ObservableSettings()
    saved = settings.evaluation_date
    today = Date(int(mk["evaluation_date"]))
    settings.evaluation_date = today

    dates = [Date(int(s)) for s in mk["curve_dates"]]
    forecast = InterpolatedZeroCurve(dates, list(mk["forecast_rates"]), Actual365Fixed())
    index = Euribor.six_months(cast("YieldTermStructureProtocol", forecast))
    index.clear_fixings()
    for f in mk["euribor6m_fixings"]:
        index.add_fixing(Date(int(f["date"])), float(f["value"]))

    issue = Date(int(ref["schedule_start"]))
    maturity = Date(int(ref["schedule_end"]))
    schedule = Schedule.from_rule(
        issue,
        maturity,
        Period(6, TimeUnit.Months),
        TARGET(),
        BusinessDayConvention.Following,
        BusinessDayConvention.Following,
        DateGeneration.Backward,
        True,
    )

    yield Setup(ref, index, schedule, issue, maturity, today)

    index.clear_fixings()
    settings.evaluation_date = saved


def _process(setup: Setup) -> BlackScholesMertonProcess:
    dc = Actual360()
    ref = setup.ref
    return BlackScholesMertonProcess(
        x0=SimpleQuote(ref["spot"]),
        dividend_ts=FlatForward.from_rate(setup.today, ref["dividend_yield"], dc),
        risk_free_ts=FlatForward.from_rate(setup.today, ref["risk_free_rate"], dc),
        black_vol_ts=BlackConstantVol(
            reference_date=setup.today,
            calendar=TARGET(),
            volatility=ref["volatility"],
            day_counter=dc,
        ),
    )


def _build(setup: Setup, spec: dict[str, Any]) -> ConvertibleFloatingRateBond:
    exercise: Exercise = (
        AmericanExercise(setup.issue, setup.maturity)
        if spec.get("american")
        else EuropeanExercise(setup.maturity)
    )
    callability: list[Callability] = []
    if spec.get("call_put"):
        callability = [
            SoftCallability(
                BondPrice(108.0, BondPriceType.Clean),
                Date(int(setup.ref["call_date"])),
                1.10,
            ),
            Callability(
                BondPrice(101.0, BondPriceType.Clean),
                CallabilityType.Put,
                Date(int(setup.ref["put_date"])),
            ),
        ]

    ex_kwargs: dict[str, Any] = {}
    if spec.get("ex"):
        ex_kwargs = {
            "ex_coupon_period": Period(1, TimeUnit.Months),
            "ex_coupon_calendar": (NullCalendar() if spec.get("ex_null_calendar") else TARGET()),
            "ex_coupon_convention": spec.get("ex_conv", BusinessDayConvention.Unadjusted),
            "ex_coupon_end_of_month": bool(spec.get("ex_eom", False)),
        }

    bond = ConvertibleFloatingRateBond(
        exercise=exercise,
        conversion_ratio=spec.get("conversion_ratio", 2.0),
        callability=callability,
        issue_date=setup.issue,
        settlement_days=spec.get("settlement_days", 3),
        index=setup.euribor,
        fixing_days=spec.get("fixing_days", 2),
        spreads=[spec.get("leg_spread", 0.005)],
        day_counter=Actual360(),
        schedule=setup.schedule,
        redemption=spec.get("redemption", 100.0),
        **ex_kwargs,
    )
    bond.set_pricing_engine(
        BinomialConvertibleEngine(
            CoxRossRubinstein,
            _process(setup),
            int(setup.ref["time_steps"]),
            SimpleQuote(spec.get("credit_spread", 0.005)),
        )
    )
    return bond


def _variant_ref(setup: Setup, key: str) -> dict[str, Any]:
    for v in setup.ref["variants"]:
        if v["key"] == key:
            return cast("dict[str, Any]", v)
    msg = f"no reference variant named {key!r}"
    raise AssertionError(msg)


@pytest.mark.parametrize("key", list(VARIANTS))
def test_convertible_floating_rate_bond(setup: Setup, key: str) -> None:
    """NPV + the complete cashflow listing, per grid point."""
    ref = _variant_ref(setup, key)
    bond = _build(setup, VARIANTS[key])

    tolerance.tight(bond.npv(), ref["npv"])
    tolerance.tight(bond.conversion_ratio(), ref["conversion_ratio"])
    assert len(bond.callability()) == ref["n_callability"]
    assert bond.settlement_days() == ref["settlement_days"]
    assert bond.settlement_date().serial_number() == ref["settlement_date"]
    assert bond.maturity_date().serial_number() == ref["maturity_date"]
    tolerance.tight(bond.accrued_amount(), ref["accrued"])

    flows = bond.cashflows()
    assert len(flows) == ref["n_cashflows"]
    for cf, r in zip(flows, ref["cashflows"], strict=True):
        assert cf.date().serial_number() == r["date"]
        tolerance.tight(cf.amount(), r["amount"])
        assert cf.ex_coupon_date().serial_number() == r["ex_coupon"]
        assert isinstance(cf, Coupon) == r["is_coupon"]
        if isinstance(cf, Coupon):
            tolerance.tight(cf.nominal(), r["nominal"])
            assert cf.accrual_start_date().serial_number() == r["accrual_start"]
            assert cf.accrual_end_date().serial_number() == r["accrual_end"]
            tolerance.tight(cf.accrual_period(), r["accrual_period"])
            tolerance.tight(cf.rate(), r["rate"])


# --- one assertion per optional argument: it must CHANGE something -------


def _npv(setup: Setup, key: str) -> float:
    return cast("float", _variant_ref(setup, key)["npv"])


def _ex_dates(setup: Setup, key: str) -> list[int]:
    return [cast("int", f["ex_coupon"]) for f in _variant_ref(setup, key)["cashflows"]]


def test_conversion_ratio_is_not_dropped(setup: Setup) -> None:
    assert _npv(setup, "conv_ratio_3") != _npv(setup, "base")


def test_credit_spread_is_not_dropped(setup: Setup) -> None:
    assert _npv(setup, "credit_spread_2pc") != _npv(setup, "base")


def test_leg_spread_is_not_dropped(setup: Setup) -> None:
    base = _variant_ref(setup, "base")
    bumped = _variant_ref(setup, "leg_spread_2pc")
    assert bumped["npv"] != base["npv"]
    # Every coupon rate moves by exactly the spread difference.
    for a, b in zip(base["cashflows"], bumped["cashflows"], strict=True):
        if a["is_coupon"]:
            tolerance.tight(b["rate"] - a["rate"], 0.015)


def test_callability_is_not_dropped(setup: Setup) -> None:
    assert _npv(setup, "call_put") != _npv(setup, "american")
    assert _variant_ref(setup, "call_put")["n_callability"] == 2


def test_exercise_is_not_dropped(setup: Setup) -> None:
    assert _npv(setup, "american") != _npv(setup, "base")


def test_redemption_is_not_dropped(setup: Setup) -> None:
    base = _variant_ref(setup, "base")
    bumped = _variant_ref(setup, "redemption_102")
    assert bumped["npv"] != base["npv"]
    tolerance.exact(base["cashflows"][-1]["amount"], 100.0)
    tolerance.exact(bumped["cashflows"][-1]["amount"], 102.0)


def test_fixing_days_is_not_dropped(setup: Setup) -> None:
    assert _npv(setup, "fixing_days_5") != _npv(setup, "base")


def test_settlement_days_is_not_dropped(setup: Setup) -> None:
    base = _variant_ref(setup, "base")
    bumped = _variant_ref(setup, "settlement_days_10")
    assert bumped["settlement_date"] != base["settlement_date"]
    assert bumped["npv"] != base["npv"]


def test_ex_coupon_period_is_not_dropped(setup: Setup) -> None:
    assert all(d == 0 for d in _ex_dates(setup, "base"))
    assert any(d != 0 for d in _ex_dates(setup, "ex_coupon_1m"))
    # Settlement falls inside the ex-coupon window: the accrual goes negative.
    assert _variant_ref(setup, "ex_coupon_1m")["accrued"] < 0.0


def test_ex_coupon_convention_is_not_dropped(setup: Setup) -> None:
    assert _ex_dates(setup, "ex_coupon_1m_preceding") != _ex_dates(setup, "ex_coupon_1m")


def test_ex_coupon_calendar_is_not_dropped(setup: Setup) -> None:
    assert _ex_dates(setup, "ex_coupon_1m_nullcal_preceding") != _ex_dates(setup, "ex_coupon_1m_preceding")


def test_ex_coupon_end_of_month_is_not_dropped(setup: Setup) -> None:
    assert _ex_dates(setup, "ex_coupon_1m_eom") != _ex_dates(setup, "ex_coupon_1m")

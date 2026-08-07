"""BondFunctions cross-validation against C++ QuantLib v1.43.

Reference: ``migration-harness/references/v143/pe/bondswap.json``, produced by
``migration-harness/cpp/probes/v143_pe_bondswap/probe.cpp``. Every number below
comes from running C++ v1.43; nothing is hand-computed.

The probe pins four bonds (fixed-rate, floating-rate, amortising, zero-coupon)
against a dense grid of settlement dates — before the issue date, mid-period, on
a coupon date and one day either side of it, on an amortisation date and one day
before, on the maturity date and one day either side. That grid is where the
off-by-one accrual bugs live, which is why it is dense rather than
representative.

Everything is rebuilt here from the same market the probe describes in the
``market`` case; the reference is only consulted for expected values.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import pytest

from pquantlib.cashflows.duration import Duration
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.thirty_360 import Convention, Thirty360
from pquantlib.indexes.ibor.euribor import Euribor6M
from pquantlib.indexes.index_manager import IndexManager
from pquantlib.instruments.bond import Bond, BondPrice, BondPriceType
from pquantlib.instruments.bonds.amortizing_fixed_rate_bond import AmortizingFixedRateBond
from pquantlib.instruments.bonds.fixed_rate_bond import FixedRateBond
from pquantlib.instruments.bonds.floating_rate_bond import FloatingRateBond
from pquantlib.instruments.bonds.zero_coupon_bond import ZeroCouponBond
from pquantlib.interest_rate import InterestRate
from pquantlib.math.solvers1d.bisection import Bisection
from pquantlib.math.solvers1d.brent import Brent
from pquantlib.math.solvers1d.finite_difference_newton_safe import FiniteDifferenceNewtonSafe
from pquantlib.math.solvers1d.newton_safe import NewtonSafe
from pquantlib.math.solvers1d.ridder import Ridder
from pquantlib.math.solvers1d.solver_1d import Solver1D
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.bond.bond_functions import BondFunctions
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

# probe.cpp — `const Date kToday(15, May, 2025);` / main():
# `Settings::instance().evaluationDate() = kToday;`
TODAY = Date.from_ymd(15, Month.May, 2025)

DC_30_360 = Thirty360(Convention.BondBasis)
DC_365 = Actual365Fixed()
DC_360 = Actual360()

# probe.cpp `floaterFixings()` — deliberately stops before the evaluation date so
# the 2025-07-15 coupon has no fixing and must be forecast.
_FLOATER_FIXINGS: list[tuple[Date, float]] = [
    (Date.from_ymd(11, Month.January, 2023), 0.0250),
    (Date.from_ymd(13, Month.July, 2023), 0.0275),
    (Date.from_ymd(11, Month.January, 2024), 0.0300),
    (Date.from_ymd(11, Month.July, 2024), 0.0325),
    (Date.from_ymd(13, Month.January, 2025), 0.0280),
]


@pytest.fixture(autouse=True)
def _eval_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    settings = ObservableSettings()
    saved = settings.evaluation_date
    # probe.cpp:main() — Settings::instance().evaluationDate() = Date(15, May, 2025)
    settings.evaluation_date = TODAY
    yield
    settings.evaluation_date = saved


def _schedule(
    start: Date, end: Date, tenor: Period, convention: BusinessDayConvention
) -> Schedule:
    return Schedule.from_rule(
        effective_date=start,
        termination_date=end,
        tenor=tenor,
        calendar=TARGET(),
        convention=convention,
        termination_date_convention=convention,
        rule=DateGeneration.Backward,
        end_of_month=False,
    )


def _make_fixed() -> FixedRateBond:
    """probe.cpp `makeFixedBond()`."""
    return FixedRateBond(
        settlement_days=3,
        face_amount=100.0,
        schedule=_schedule(
            Date.from_ymd(15, Month.January, 2020),
            Date.from_ymd(15, Month.January, 2030),
            Period(6, TimeUnit.Months),
            BusinessDayConvention.Unadjusted,
        ),
        coupons=[0.04],
        accrual_day_counter=DC_30_360,
        payment_convention=BusinessDayConvention.Following,
        redemption=100.0,
        issue_date=Date.from_ymd(15, Month.January, 2020),
    )


def _make_amortising() -> AmortizingFixedRateBond:
    """probe.cpp `makeAmortisingBond()`."""
    return AmortizingFixedRateBond(
        settlement_days=3,
        notionals=[100.0, 80.0, 60.0, 40.0, 20.0],
        schedule=_schedule(
            Date.from_ymd(15, Month.January, 2021),
            Date.from_ymd(15, Month.January, 2026),
            Period(1, TimeUnit.Years),
            BusinessDayConvention.Unadjusted,
        ),
        coupons=[0.05],
        accrual_day_counter=DC_30_360,
        payment_convention=BusinessDayConvention.Following,
        issue_date=Date.from_ymd(15, Month.January, 2021),
        redemptions=[100.0],
    )


def _make_zero() -> ZeroCouponBond:
    """probe.cpp `makeZeroBond()`."""
    return ZeroCouponBond(
        settlement_days=3,
        calendar=TARGET(),
        face_amount=100.0,
        maturity_date=Date.from_ymd(15, Month.January, 2030),
        payment_convention=BusinessDayConvention.Following,
        redemption=100.0,
        issue_date=Date.from_ymd(15, Month.January, 2020),
    )


def _make_floating() -> FloatingRateBond:
    """probe.cpp `makeFloatingBond()` — Euribor6M + 50bp on a flat 3% curve."""
    index = Euribor6M(FlatForward.from_rate(TODAY, 0.03, DC_365))
    return FloatingRateBond(
        settlement_days=3,
        face_amount=100.0,
        schedule=_schedule(
            Date.from_ymd(15, Month.January, 2023),
            Date.from_ymd(15, Month.January, 2028),
            Period(6, TimeUnit.Months),
            BusinessDayConvention.ModifiedFollowing,
        ),
        ibor_index=index,
        accrual_day_counter=DC_360,
        payment_convention=BusinessDayConvention.Following,
        fixing_days=2,
        gearings=[1.0],
        spreads=[0.005],
    )


def _discount_curve() -> FlatForward:
    """probe.cpp `discountCurve()` — flat 3.5%, Actual365Fixed, Compounded/Annual."""
    return FlatForward.from_rate(TODAY, 0.035, DC_365, Compounding.Compounded, Frequency.Annual)


@pytest.fixture
def bonds() -> Iterator[dict[str, Bond]]:
    """The four probe bonds, with the floater's historical fixings seeded."""
    manager = IndexManager()
    name = Euribor6M().name()
    saved = manager.get_history(name) if manager.has_history(name) else None
    manager.clear_history(name)
    for date, value in _FLOATER_FIXINGS:
        manager.add_fixing(name, date, value, True)
    try:
        yield {
            "fixed": _make_fixed(),
            "amortising": _make_amortising(),
            "zero": _make_zero(),
            "floating": _make_floating(),
        }
    finally:
        manager.clear_history(name)
        if saved is not None:
            manager.set_history(name, saved)


def _date(serial: int) -> Date:
    """The probe emits a null ``Date`` as serial 0, which is ``Date()``."""
    return Date(serial)


def _check(expected: dict[str, Any], key: str, compute: Callable[[], float | int]) -> None:
    """Assert ``compute()`` matches ``expected[key]``, or raises if ``key_throws``."""
    if expected.get(f"{key}_throws") is True:
        with pytest.raises(Exception, match=r".*"):
            compute()
        return
    assert key in expected, f"reference has neither {key!r} nor {key}_throws"
    tolerance.tight(float(compute()), float(expected[key]))


def _check_date(expected: dict[str, Any], key: str, compute: Callable[[], Date]) -> None:
    if expected.get(f"{key}_throws") is True:
        with pytest.raises(Exception, match=r".*"):
            compute()
        return
    assert key in expected, f"reference has neither {key!r} nor {key}_throws"
    assert compute() == _date(int(expected[key]))


# ---------------------------------------------------------------------------
# market
# ---------------------------------------------------------------------------


def test_market_matches_probe(bonds: dict[str, Bond]) -> None:
    expected = CPP["market"]["expected"]
    for name, bond in bonds.items():
        assert bond.settlement_date() == _date(expected[f"{name}_settlement_date"])
        assert BondFunctions.start_date(bond) == _date(expected[f"{name}_start_date"])
        assert BondFunctions.maturity_date(bond) == _date(expected[f"{name}_maturity_date"])
        assert len(bond.cashflows()) == expected[f"{name}_n_cashflows"]


def test_default_settlement_date_path(bonds: dict[str, Bond]) -> None:
    """``settlement_date=None`` must be ``bond.settlement_date()`` (C++ ``Date()``)."""
    expected = CPP["fixed_default_settlement_date"]["expected"]
    bond = bonds["fixed"]
    assert bond.settlement_date() == _date(expected["default_settlement_date"])
    assert BondFunctions.is_tradable(bond) is bool(expected["is_tradable_default"])
    tolerance.tight(BondFunctions.accrued_amount(bond), expected["accrued_amount_default"])
    tolerance.tight(
        BondFunctions.accrued_amount(bond, bond.settlement_date()),
        expected["accrued_amount_explicit"],
    )
    assert BondFunctions.next_cash_flow_date(bond) == _date(
        expected["next_cash_flow_date_default"]
    )
    tolerance.tight(BondFunctions.next_coupon_rate(bond), expected["next_coupon_rate_default"])


# ---------------------------------------------------------------------------
# the date / cashflow / coupon inspector surface
# ---------------------------------------------------------------------------

_INSPECTOR_CASES = sorted(k for k in CPP if "_inspect_" in k)


@pytest.mark.parametrize("case", _INSPECTOR_CASES)
def test_inspectors(case: str, bonds: dict[str, Bond]) -> None:
    inputs = CPP[case]["inputs"]
    expected = CPP[case]["expected"]
    bond = bonds[inputs["bond"]]
    settle = _date(inputs["settlement_date"])

    assert BondFunctions.start_date(bond) == _date(expected["start_date"])
    assert BondFunctions.maturity_date(bond) == _date(expected["maturity_date"])
    assert BondFunctions.is_tradable(bond, settle) is bool(expected["is_tradable"])
    tolerance.tight(bond.notional(settle), expected["notional"])

    assert BondFunctions.previous_cash_flow_date(bond, settle) == _date(
        expected["previous_cash_flow_date"]
    )
    assert BondFunctions.next_cash_flow_date(bond, settle) == _date(
        expected["next_cash_flow_date"]
    )
    tolerance.tight(
        BondFunctions.previous_cash_flow_amount(bond, settle),
        expected["previous_cash_flow_amount"],
    )
    tolerance.tight(
        BondFunctions.next_cash_flow_amount(bond, settle), expected["next_cash_flow_amount"]
    )

    _check(expected, "previous_coupon_rate", lambda: BondFunctions.previous_coupon_rate(bond, settle))
    _check(expected, "next_coupon_rate", lambda: BondFunctions.next_coupon_rate(bond, settle))

    _check_date(expected, "accrual_start_date", lambda: BondFunctions.accrual_start_date(bond, settle))
    _check_date(expected, "accrual_end_date", lambda: BondFunctions.accrual_end_date(bond, settle))
    _check_date(
        expected, "reference_period_start", lambda: BondFunctions.reference_period_start(bond, settle)
    )
    _check_date(
        expected, "reference_period_end", lambda: BondFunctions.reference_period_end(bond, settle)
    )
    _check(expected, "accrual_period", lambda: BondFunctions.accrual_period(bond, settle))
    _check(expected, "accrual_days", lambda: BondFunctions.accrual_days(bond, settle))
    _check(expected, "accrued_period", lambda: BondFunctions.accrued_period(bond, settle))
    _check(expected, "accrued_days", lambda: BondFunctions.accrued_days(bond, settle))
    tolerance.tight(BondFunctions.accrued_amount(bond, settle), expected["accrued_amount"])


def test_previous_cash_flow_and_next_cash_flow_objects(bonds: dict[str, Bond]) -> None:
    """The two iterator-returning C++ overloads, mapped to ``CashFlow | None``."""
    bond = bonds["fixed"]
    expected = CPP["fixed_inspect_mid_period"]["expected"]
    settle = _date(CPP["fixed_inspect_mid_period"]["inputs"]["settlement_date"])
    previous = BondFunctions.previous_cash_flow(bond, settle)
    following = BondFunctions.next_cash_flow(bond, settle)
    assert previous is not None
    assert following is not None
    assert previous.date() == _date(expected["previous_cash_flow_date"])
    assert following.date() == _date(expected["next_cash_flow_date"])

    # Before the first cashflow C++ returns ``leg.rend()``; Python returns None.
    before = _date(CPP["fixed_inspect_before_issue"]["inputs"]["settlement_date"])
    assert BondFunctions.previous_cash_flow(bond, before) is None
    # After the last one C++ returns ``leg.end()``.
    after = _date(CPP["fixed_inspect_after_maturity"]["inputs"]["settlement_date"])
    assert BondFunctions.next_cash_flow(bond, after) is None


# ---------------------------------------------------------------------------
# pricing surface: YieldTermStructure / InterestRate / Rate / z-spread groups
# ---------------------------------------------------------------------------

_PRICING_CASES = sorted(k for k in CPP if "_pricing_" in k)


@pytest.mark.parametrize("case", _PRICING_CASES)
def test_pricing_surface(case: str, bonds: dict[str, Bond]) -> None:
    inputs = CPP[case]["inputs"]
    expected = CPP[case]["expected"]
    bond = bonds[inputs["bond"]]
    settle = _date(inputs["settlement_date"])
    curve = _discount_curve()
    comp = Compounding(inputs["yield_compounding"])
    freq = Frequency(inputs["yield_frequency"])
    rate = float(inputs["yield"])
    y = InterestRate(rate, DC_365, comp, freq)
    quote = float(inputs["clean_quote"])
    spread = float(inputs["z_spread"])

    # --- YieldTermStructure group ---
    _check(expected, "clean_price_curve", lambda: BondFunctions.clean_price(bond, curve, settle))
    _check(expected, "dirty_price_curve", lambda: BondFunctions.dirty_price(bond, curve, settle))
    _check(expected, "bps_curve", lambda: BondFunctions.bps(bond, curve, settle))
    _check(expected, "atm_rate_no_price", lambda: BondFunctions.atm_rate(bond, curve, settle))
    _check(
        expected,
        "atm_rate_clean_price",
        lambda: BondFunctions.atm_rate(bond, curve, settle, BondPrice(quote, BondPriceType.Clean)),
    )
    _check(
        expected,
        "atm_rate_dirty_price",
        lambda: BondFunctions.atm_rate(bond, curve, settle, BondPrice(quote, BondPriceType.Dirty)),
    )

    # --- InterestRate group ---
    _check(expected, "clean_price_ir", lambda: BondFunctions.clean_price_from_yield(bond, y, settle))
    _check(expected, "dirty_price_ir", lambda: BondFunctions.dirty_price_from_yield(bond, y, settle))
    _check(expected, "bps_ir", lambda: BondFunctions.bps_from_yield(bond, y, settle))
    _check(
        expected,
        "duration_simple_ir",
        lambda: BondFunctions.duration_from_yield(bond, y, Duration.Simple, settle),
    )
    _check(
        expected,
        "duration_modified_ir",
        lambda: BondFunctions.duration_from_yield(bond, y, Duration.Modified, settle),
    )
    _check(
        expected,
        "duration_macaulay_ir",
        lambda: BondFunctions.duration_from_yield(bond, y, Duration.Macaulay, settle),
    )
    _check(expected, "convexity_ir", lambda: BondFunctions.convexity_from_yield(bond, y, settle))
    _check(
        expected,
        "basis_point_value_ir",
        lambda: BondFunctions.basis_point_value_from_yield(bond, y, settle),
    )
    _check(
        expected,
        "yield_value_basis_point_ir",
        lambda: BondFunctions.yield_value_basis_point_from_yield(bond, y, settle),
    )

    # --- Rate + DayCounter + Compounding + Frequency group ---
    _check(
        expected,
        "clean_price_rate",
        lambda: BondFunctions.clean_price_from_rate(bond, rate, DC_365, comp, freq, settle),
    )
    _check(
        expected,
        "dirty_price_rate",
        lambda: BondFunctions.dirty_price_from_rate(bond, rate, DC_365, comp, freq, settle),
    )
    _check(
        expected,
        "bps_rate",
        lambda: BondFunctions.bps_from_rate(bond, rate, DC_365, comp, freq, settle),
    )
    _check(
        expected,
        "duration_modified_rate",
        lambda: BondFunctions.duration_from_rate(
            bond, rate, DC_365, comp, freq, Duration.Modified, settle
        ),
    )
    _check(
        expected,
        "convexity_rate",
        lambda: BondFunctions.convexity_from_rate(bond, rate, DC_365, comp, freq, settle),
    )
    _check(
        expected,
        "basis_point_value_rate",
        lambda: BondFunctions.basis_point_value_from_rate(bond, rate, DC_365, comp, freq, settle),
    )
    _check(
        expected,
        "yield_value_basis_point_rate",
        lambda: BondFunctions.yield_value_basis_point_from_rate(
            bond, rate, DC_365, comp, freq, settle
        ),
    )

    # --- yield (NewtonSafe overload) ---
    _check(
        expected,
        "yield_clean",
        lambda: BondFunctions.bond_yield(
            bond, BondPrice(quote, BondPriceType.Clean), DC_365, comp, freq, settle
        ),
    )
    _check(
        expected,
        "yield_dirty",
        lambda: BondFunctions.bond_yield(
            bond, BondPrice(quote, BondPriceType.Dirty), DC_365, comp, freq, settle
        ),
    )

    # --- z-spread group ---
    _check(
        expected,
        "clean_price_zspread",
        lambda: BondFunctions.clean_price_from_z_spread(bond, curve, spread, comp, freq, settle),
    )
    _check(
        expected,
        "dirty_price_zspread",
        lambda: BondFunctions.dirty_price_from_z_spread(bond, curve, spread, comp, freq, settle),
    )
    _check(
        expected,
        "z_spread_from_clean",
        lambda: BondFunctions.z_spread(
            bond, BondPrice(quote, BondPriceType.Clean), curve, comp, freq, settle
        ),
    )
    _check(
        expected,
        "z_spread_from_dirty",
        lambda: BondFunctions.z_spread(
            bond, BondPrice(quote, BondPriceType.Dirty), curve, comp, freq, settle
        ),
    )


def test_rate_overloads_agree_with_interest_rate_overloads(bonds: dict[str, Bond]) -> None:
    """The ``*_from_rate`` C++ overloads only build an InterestRate and forward."""
    expected = CPP["fixed_pricing_mid_period"]["expected"]
    for ir_key, rate_key in [
        ("clean_price_ir", "clean_price_rate"),
        ("dirty_price_ir", "dirty_price_rate"),
        ("bps_ir", "bps_rate"),
        ("duration_modified_ir", "duration_modified_rate"),
        ("convexity_ir", "convexity_rate"),
        ("basis_point_value_ir", "basis_point_value_rate"),
        ("yield_value_basis_point_ir", "yield_value_basis_point_rate"),
    ]:
        tolerance.exact(expected[ir_key], expected[rate_key])
    del bonds


# ---------------------------------------------------------------------------
# yield() templated on the solver
# ---------------------------------------------------------------------------

_SOLVERS: dict[str, tuple[Callable[[], Solver1D], int]] = {
    "yield_newton_safe": (NewtonSafe, 100),
    "yield_brent": (Brent, 100),
    "yield_bisection": (Bisection, 200),
    "yield_ridder": (Ridder, 100),
    "yield_fd_newton_safe": (FiniteDifferenceNewtonSafe, 100),
}


@pytest.mark.parametrize("key", sorted(_SOLVERS))
def test_yield_with_solver(key: str, bonds: dict[str, Bond]) -> None:
    """Each solver converges to a different double; all five are pinned separately."""
    inputs = CPP["fixed_yield_solvers"]["inputs"]
    expected = CPP["fixed_yield_solvers"]["expected"]
    bond = bonds["fixed"]
    settle = _date(inputs["settlement_date"])
    price = BondPrice(float(inputs["clean_quote"]), BondPriceType.Clean)
    factory, max_evaluations = _SOLVERS[key]
    solver = factory()
    solver.set_max_evaluations(max_evaluations)
    got = BondFunctions.bond_yield_with_solver(
        solver, bond, price, DC_365, Compounding.Compounded, Frequency.Semiannual, settle
    )
    # LOOSE: a root found to 1e-10 on the PRICE residual, not on the yield. The
    # price/yield slope here is dP/dy ~ -4.1 * 99.3 ~ -410 per unit yield, so a
    # 1e-10 price accuracy pins the yield only to ~2.4e-13 — and the five
    # solvers genuinely disagree in the 12th decimal (see the reference).
    # Reproducing each solver's own double bit-for-bit is the point of the
    # per-solver parametrisation, but the tier has to allow for the last ulps
    # of a different iteration path on a different libm.
    tolerance.loose(got, expected[key])


def test_yield_default_overload_is_newton_safe(bonds: dict[str, Bond]) -> None:
    inputs = CPP["fixed_yield_solvers"]["inputs"]
    expected = CPP["fixed_yield_solvers"]["expected"]
    bond = bonds["fixed"]
    settle = _date(inputs["settlement_date"])
    price = BondPrice(float(inputs["clean_quote"]), BondPriceType.Clean)
    got = BondFunctions.bond_yield(
        bond, price, DC_365, Compounding.Compounded, Frequency.Semiannual, settle
    )
    tolerance.loose(got, expected["yield_default_overload"])
    # C++ bondfunctions.cpp:382-386 — the non-template overload IS NewtonSafe.
    tolerance.exact(expected["yield_default_overload"], expected["yield_newton_safe"])


def test_yield_accuracy_and_guess_are_wired(bonds: dict[str, Bond]) -> None:
    """A 50% guess and 1e-12 accuracy must still land on the same root."""
    inputs = CPP["fixed_yield_solvers"]["inputs"]
    expected = CPP["fixed_yield_solvers"]["expected"]
    bond = bonds["fixed"]
    settle = _date(inputs["settlement_date"])
    price = BondPrice(float(inputs["clean_quote"]), BondPriceType.Clean)
    solver = NewtonSafe()
    solver.set_max_evaluations(100)
    got = BondFunctions.bond_yield_with_solver(
        solver,
        bond,
        price,
        DC_365,
        Compounding.Compounded,
        Frequency.Semiannual,
        settle,
        1.0e-12,
        0.50,
    )
    tolerance.loose(got, expected["yield_newton_safe_guess_50pc"])


# ---------------------------------------------------------------------------
# z-spread: the deprecated day-counter overloads, and the round trip
# ---------------------------------------------------------------------------


def test_zspread_deprecated_day_counter_is_discarded(bonds: dict[str, Bond]) -> None:
    """C++ bondfunctions.cpp:500-508 / 530-538 / 570-582 ignore the DayCounter."""
    inputs = CPP["fixed_zspread_deprecated_daycounter_ignored"]["inputs"]
    expected = CPP["fixed_zspread_deprecated_daycounter_ignored"]["expected"]
    bond = bonds["fixed"]
    settle = _date(inputs["settlement_date"])
    spread = float(inputs["z_spread"])
    curve = _discount_curve()
    comp, freq = Compounding.Compounded, Frequency.Semiannual

    tolerance.tight(
        BondFunctions.clean_price_from_z_spread(bond, curve, spread, comp, freq, settle),
        expected["clean_price_no_day_counter"],
    )
    for day_counter, key in [
        (DC_360, "clean_price_day_counter_a360"),
        (DC_30_360, "clean_price_day_counter_30360"),
    ]:
        tolerance.tight(
            BondFunctions.clean_price_from_z_spread(
                bond, curve, spread, comp, freq, settle, day_counter
            ),
            expected[key],
        )
    tolerance.tight(
        BondFunctions.dirty_price_from_z_spread(
            bond, curve, spread, comp, freq, settle, DC_360
        ),
        expected["dirty_price_day_counter_a360"],
    )
    tolerance.tight(
        BondFunctions.z_spread(
            bond,
            BondPrice(101.5, BondPriceType.Clean),
            curve,
            comp,
            freq,
            settle,
            day_counter=DC_360,
        ),
        expected["z_spread_day_counter_a360"],
    )
    # The whole point: the deprecated answers are the non-deprecated answers.
    tolerance.exact(
        expected["clean_price_day_counter_a360"], expected["clean_price_no_day_counter"]
    )
    tolerance.exact(
        expected["clean_price_day_counter_30360"], expected["clean_price_no_day_counter"]
    )
    tolerance.exact(
        expected["dirty_price_day_counter_a360"], expected["dirty_price_no_day_counter"]
    )
    tolerance.exact(expected["z_spread_day_counter_a360"], expected["z_spread_no_day_counter"])


def test_zspread_round_trip(bonds: dict[str, Bond]) -> None:
    inputs = CPP["fixed_zspread_round_trip"]["inputs"]
    expected = CPP["fixed_zspread_round_trip"]["expected"]
    bond = bonds["fixed"]
    settle = _date(inputs["settlement_date"])
    spread = float(inputs["z_spread_in"])
    curve = _discount_curve()
    clean = BondFunctions.clean_price_from_z_spread(
        bond, curve, spread, Compounding.Compounded, Frequency.Semiannual, settle
    )
    tolerance.tight(clean, expected["clean_price"])
    got = BondFunctions.z_spread(
        bond,
        BondPrice(clean, BondPriceType.Clean),
        curve,
        Compounding.Compounded,
        Frequency.Semiannual,
        settle,
    )
    # LOOSE: a Brent root found to 1e-10 on an NPV of ~1e2, i.e. ~1e-12 relative
    # on the price; divided by the ~4-year duration that is ~2.5e-13 on the
    # spread itself, and the reference itself only recovers 1.25e-2 to 2.7e-14.
    tolerance.loose(got, expected["z_spread_out"])


def test_bond_functions_is_a_namespace() -> None:
    """C++ ``struct BondFunctions`` has only statics; Python refuses construction."""
    with pytest.raises(TypeError, match="namespace"):
        BondFunctions()

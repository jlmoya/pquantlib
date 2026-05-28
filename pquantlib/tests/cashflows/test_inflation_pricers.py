"""Tests for inflation coupon pricers (L7-C).

Cross-validates the Black / UnitDisplacedBlack / Bachelier YoY pricers
against the C++ ``blackFormula`` / ``bachelierBlackFormula`` family
emitted by ``cluster_l7c_probe``.

The pricers are exercised via their ``_optionlet_price_imp`` hook
directly. This isolates the Black-formula plumbing from the
optionlet-volatility-surface wiring (which is L7-D's responsibility).
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.cashflows.cpi_coupon_pricer import (
    BlackCPICouponPricer,
    CPICouponPricer,
)
from pquantlib.cashflows.inflation_coupon_pricer import (
    InflationCouponPricer,
    set_coupon_pricer,
)
from pquantlib.cashflows.yoy_inflation_coupon import YoYInflationCoupon
from pquantlib.cashflows.yoy_inflation_coupon_pricer import (
    BachelierYoYInflationCouponPricer,
    BlackYoYInflationCouponPricer,
    UnitDisplacedBlackYoYInflationCouponPricer,
    YoYInflationCouponPricer,
)
from pquantlib.daycounters.thirty_360 import Convention, Thirty360
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.inflation.cpi import InterpolationType
from pquantlib.indexes.inflation.eu_hicp import YoYEUHICP
from pquantlib.payoffs import OptionType
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


@pytest.fixture(scope="module")
def reference() -> dict[str, Any]:
    return reference_reader.load("cluster/l7c")


# ---- Black / Bachelier / UnitDisplaced optionlet-price impls ---------------


def test_black_yoy_pricer_call_matches_probe(reference: dict[str, Any]) -> None:
    """BlackYoY ``optionlet_price_imp(Call)`` == ``blackFormula(Call, K, F, stdDev)``.

    # Justification (TIGHT): pricer delegates directly to black_formula
    # which we already validated bit-equal vs the C++ blackFormula in
    # L3-A — TIGHT here re-pins the link through the new pricer surface.
    """
    expected = reference["black_pricer"]
    pricer = BlackYoYInflationCouponPricer()
    result = pricer._optionlet_price_imp(  # type: ignore[reportPrivateUsage]
        OptionType.Call,
        expected["strike"],
        expected["forward"],
        expected["std_dev"],
    )
    tolerance.tight(result, expected["black_call"])


def test_black_yoy_pricer_put_matches_probe(reference: dict[str, Any]) -> None:
    """BlackYoY ``optionlet_price_imp(Put)`` == ``blackFormula(Put, K, F, stdDev)``.

    # Justification (LOOSE): C++ probe emits ``-0`` (signed-zero IEEE) for
    # a deep-OTM put; Python's black_formula returns ``+0``. LOOSE tier
    # is appropriate for negative-zero parity at the deep-OTM boundary.
    """
    expected = reference["black_pricer"]
    pricer = BlackYoYInflationCouponPricer()
    result = pricer._optionlet_price_imp(  # type: ignore[reportPrivateUsage]
        OptionType.Put,
        expected["strike"],
        expected["forward"],
        expected["std_dev"],
    )
    tolerance.loose(result, expected["black_put"])


def test_bachelier_yoy_pricer_call_matches_probe(reference: dict[str, Any]) -> None:
    """BachelierYoY ``optionlet_price_imp(Call)``.

    # Justification (TIGHT): delegates to bachelier_black_formula.
    """
    expected = reference["black_pricer"]
    pricer = BachelierYoYInflationCouponPricer()
    result = pricer._optionlet_price_imp(  # type: ignore[reportPrivateUsage]
        OptionType.Call,
        expected["strike"],
        expected["forward"],
        expected["std_dev"],
    )
    tolerance.tight(result, expected["bachelier_call"])


def test_bachelier_yoy_pricer_put_matches_probe(reference: dict[str, Any]) -> None:
    """BachelierYoY ``optionlet_price_imp(Put)``."""
    expected = reference["black_pricer"]
    pricer = BachelierYoYInflationCouponPricer()
    result = pricer._optionlet_price_imp(  # type: ignore[reportPrivateUsage]
        OptionType.Put,
        expected["strike"],
        expected["forward"],
        expected["std_dev"],
    )
    tolerance.tight(result, expected["bachelier_put"])


def test_unit_displaced_black_yoy_pricer_call_matches_probe(
    reference: dict[str, Any],
) -> None:
    """UnitDisplacedBlackYoY ``optionlet_price_imp(Call)``.

    # Justification (TIGHT): forwards through ``blackFormula(Call, K+1,
    # F+1, stdDev)`` — the displaced Black mapping.
    """
    expected = reference["black_pricer"]
    pricer = UnitDisplacedBlackYoYInflationCouponPricer()
    result = pricer._optionlet_price_imp(  # type: ignore[reportPrivateUsage]
        OptionType.Call,
        expected["strike"],
        expected["forward"],
        expected["std_dev"],
    )
    tolerance.tight(result, expected["unit_displaced_call"])


def test_unit_displaced_black_yoy_pricer_put_matches_probe(
    reference: dict[str, Any],
) -> None:
    """UnitDisplacedBlackYoY ``optionlet_price_imp(Put)``."""
    expected = reference["black_pricer"]
    pricer = UnitDisplacedBlackYoYInflationCouponPricer()
    result = pricer._optionlet_price_imp(  # type: ignore[reportPrivateUsage]
        OptionType.Put,
        expected["strike"],
        expected["forward"],
        expected["std_dev"],
    )
    tolerance.tight(result, expected["unit_displaced_put"])


# ---- Base-class error paths -----------------------------------------------


def test_base_pricer_optionlet_price_imp_raises() -> None:
    """The base YoY pricer raises ``LibraryException`` if asked for a
    vol-dependent price.

    # C++ parity: inflationcouponpricer.cpp:80-86 — QL_FAIL("you must
    # implement this to get a vol-dependent price").
    """
    pricer = YoYInflationCouponPricer()
    with pytest.raises(LibraryException):
        pricer._optionlet_price_imp(  # type: ignore[reportPrivateUsage]
            OptionType.Call, 0.02, 0.025, 0.005
        )


def test_yoy_pricer_caplet_without_vol_raises() -> None:
    """Calling ``caplet_rate`` without a vol surface raises.

    # C++ parity: optionletRate's QL_REQUIRE(!capletVolatility().empty()).
    """
    # We need to wire a YoY coupon + pricer pair.
    yoy = YoYEUHICP()
    yoy.clear_fixings()
    yoy.add_fixing(Date.from_ymd(1, Month.March, 2022), 0.025, True)
    cpn = YoYInflationCoupon(
        payment_date=Date.from_ymd(15, Month.June, 2022),
        nominal=100000.0,
        accrual_start_date=Date.from_ymd(1, Month.June, 2021),
        accrual_end_date=Date.from_ymd(1, Month.June, 2022),
        fixing_days=0,
        index=yoy,
        observation_lag=Period(3, TimeUnit.Months),
        interpolation=InterpolationType.Flat,
        day_counter=Thirty360(Convention.BondBasis),
    )
    pricer = BlackYoYInflationCouponPricer()
    cpn.set_pricer(pricer)
    pricer.initialize(cpn)
    with pytest.raises(LibraryException):
        pricer.caplet_rate(0.03)


def test_cpi_pricer_caplet_without_vol_raises() -> None:
    """CPICouponPricer cap/floor without vol raises ``LibraryException``.

    # C++ parity: cpicouponpricer.cpp:74-78.
    """
    pricer = CPICouponPricer()
    with pytest.raises(LibraryException):
        pricer.caplet_rate(0.03)
    with pytest.raises(LibraryException):
        pricer.floorlet_rate(0.01)


# ---- Pricer wiring API ----------------------------------------------------


def test_set_caplet_volatility_rejects_none() -> None:
    """``set_caplet_volatility(None)`` raises.

    # C++ parity: cpicouponpricer.cpp:38-43 / inflationcouponpricer.cpp:52-57.
    """
    yp = YoYInflationCouponPricer()
    cp = CPICouponPricer()
    with pytest.raises(LibraryException):
        yp.set_caplet_volatility(None)  # type: ignore[arg-type]
    with pytest.raises(LibraryException):
        cp.set_caplet_volatility(None)  # type: ignore[arg-type]


def test_set_caplet_volatility_accepts_object() -> None:
    """Plugging in an arbitrary object stores it (full type check is L7-D).

    # C++ parity: registration handshake is omitted in this port.
    """
    yp = YoYInflationCouponPricer()
    sentinel = object()
    yp.set_caplet_volatility(sentinel)
    assert yp.caplet_volatility() is sentinel


def test_set_coupon_pricer_attaches_to_inflation_coupons() -> None:
    """The free function applies a pricer to every InflationCoupon in a leg.

    # C++ parity: inflationcouponpricer.cpp:27-34.
    """
    yoy = YoYEUHICP()
    yoy.clear_fixings()
    yoy.add_fixing(Date.from_ymd(1, Month.March, 2022), 0.025, True)
    cpn = YoYInflationCoupon(
        payment_date=Date.from_ymd(15, Month.June, 2022),
        nominal=100000.0,
        accrual_start_date=Date.from_ymd(1, Month.June, 2021),
        accrual_end_date=Date.from_ymd(1, Month.June, 2022),
        fixing_days=0,
        index=yoy,
        observation_lag=Period(3, TimeUnit.Months),
        interpolation=InterpolationType.Flat,
        day_counter=Thirty360(Convention.BondBasis),
    )
    pricer = YoYInflationCouponPricer()
    set_coupon_pricer([cpn], pricer)
    assert cpn.pricer() is pricer


def test_inflation_coupon_pricer_is_abstract() -> None:
    """``InflationCouponPricer`` cannot be instantiated directly."""
    with pytest.raises(TypeError):
        InflationCouponPricer()  # type: ignore[abstract]


def test_black_cpi_pricer_black_price_closed_form(reference: dict[str, Any]) -> None:
    """``BlackCPICouponPricer._black_price`` delegates to black_formula.

    # Justification (TIGHT): exposed as a helper for L7-D's CPI cap/floor
    # engine; verify it reproduces the L3-A Black path bit-for-bit.
    """
    expected = reference["black_pricer"]
    pricer = BlackCPICouponPricer()
    result_call = pricer._black_price(  # type: ignore[reportPrivateUsage]
        OptionType.Call,
        expected["strike"],
        expected["forward"],
        expected["std_dev"],
    )
    tolerance.tight(result_call, expected["black_call"])

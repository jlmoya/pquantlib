"""Cross-validate the ql/indexes/swap/* families closed against C++ v1.43.

Probe: ``v143/indexes/swap``.

A swap index carries more wiring than an ibor index: on top of name, fixing
days, currency, fixing calendar and day counter it fixes the fixed leg's tenor
and roll convention and picks an underlying ibor index whose tenor is itself a
function of the swap tenor. Every family is checked at a tenor at or below 1Y
and one above it, so a missing 3M/6M ternary cannot pass.

The test walks the reference JSON by key rather than restating values, and
asserts the key set matches ``_INDEXES`` so no probe entry can be skipped.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from pquantlib.currencies.america import USDCurrency
from pquantlib.currencies.europe import EURCurrency
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.indexes.ibor.eonia import Eonia
from pquantlib.indexes.ibor.sofr import Sofr
from pquantlib.indexes.overnight_indexed_swap_index import OvernightIndexedSwapIndex
from pquantlib.indexes.swap.chf_libor_swap_isda_fix import ChfLiborSwapIsdaFix
from pquantlib.indexes.swap.eur_libor_swap_ifr_fix import EurLiborSwapIfrFix
from pquantlib.indexes.swap.eur_libor_swap_isda_fix_a import EurLiborSwapIsdaFixA
from pquantlib.indexes.swap.eur_libor_swap_isda_fix_b import EurLiborSwapIsdaFixB
from pquantlib.indexes.swap.euribor_swap_ifr_fix import EuriborSwapIfrFix
from pquantlib.indexes.swap.euribor_swap_isda_fix_a import EuriborSwapIsdaFixA
from pquantlib.indexes.swap.euribor_swap_isda_fix_b import EuriborSwapIsdaFixB
from pquantlib.indexes.swap.gbp_libor_swap_isda_fix import GbpLiborSwapIsdaFix
from pquantlib.indexes.swap.jpy_libor_swap_isda_fix_am import JpyLiborSwapIsdaFixAm
from pquantlib.indexes.swap.jpy_libor_swap_isda_fix_pm import JpyLiborSwapIsdaFixPm
from pquantlib.indexes.swap.usd_libor_swap_isda_fix_am import UsdLiborSwapIsdaFixAm
from pquantlib.indexes.swap.usd_libor_swap_isda_fix_pm import UsdLiborSwapIsdaFixPm
from pquantlib.indexes.swap_index import SwapIndex
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_1Y = Period(1, TimeUnit.Years)
_5Y = Period(5, TimeUnit.Years)
_10Y = Period(10, TimeUnit.Years)

# Every ISDA/IFR-fix family the probe emits, keyed as ``<family>_<tenor>``.
_FAMILIES: dict[str, Callable[[Period], SwapIndex]] = {
    "euribor_swap_isda_fix_a": EuriborSwapIsdaFixA,
    "euribor_swap_isda_fix_b": EuriborSwapIsdaFixB,
    "euribor_swap_ifr_fix": EuriborSwapIfrFix,
    "eur_libor_swap_isda_fix_a": EurLiborSwapIsdaFixA,
    "eur_libor_swap_isda_fix_b": EurLiborSwapIsdaFixB,
    "eur_libor_swap_ifr_fix": EurLiborSwapIfrFix,
    "chf_libor_swap_isda_fix": ChfLiborSwapIsdaFix,
    "gbp_libor_swap_isda_fix": GbpLiborSwapIsdaFix,
    "jpy_libor_swap_isda_fix_am": JpyLiborSwapIsdaFixAm,
    "jpy_libor_swap_isda_fix_pm": JpyLiborSwapIsdaFixPm,
    "usd_libor_swap_isda_fix_am": UsdLiborSwapIsdaFixAm,
    "usd_libor_swap_isda_fix_pm": UsdLiborSwapIsdaFixPm,
}

_INDEXES: dict[str, Callable[[], SwapIndex]] = {
    **{
        f"{key}_{suffix}": (lambda f=factory, t=tenor: f(t))
        for key, factory in _FAMILIES.items()
        for suffix, tenor in (("1y", _1Y), ("10y", _10Y))
    },
    "ois_index_eonia_5y": lambda: OvernightIndexedSwapIndex(
        "EoniaSwapIsdaFix", _5Y, 2, EURCurrency(), Eonia(),
    ),
    "ois_index_sofr_10y": lambda: OvernightIndexedSwapIndex(
        "SofrSwap", _10Y, 2, USDCurrency(), Sofr(),
    ),
}


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/indexes/swap")


def test_every_reference_key_is_covered(cpp: dict[str, Any]) -> None:
    """No probe entry may be silently skipped by the walk below."""
    assert set(cpp) == set(_INDEXES)


@pytest.mark.parametrize("key", sorted(_INDEXES))
def test_swap_index_matches_cpp(key: str, cpp: dict[str, Any]) -> None:
    ref = cpp[key]
    idx = _INDEXES[key]()

    assert idx.name() == ref["name"]
    assert idx.fixing_days() == ref["fixing_days"]
    assert idx.currency().code == ref["currency_code"]
    assert idx.fixing_calendar().name() == ref["fixing_calendar"]
    assert idx.day_counter().name() == ref["day_counter"]
    assert idx.tenor().length == ref["tenor_length"]
    assert int(idx.tenor().units) == ref["tenor_units"]
    assert idx.fixed_leg_tenor().length == ref["fixed_leg_tenor_length"]
    assert int(idx.fixed_leg_tenor().units) == ref["fixed_leg_tenor_units"]
    assert int(idx.fixed_leg_convention()) == ref["fixed_leg_convention"]
    assert idx.ibor_index().name() == ref["ibor_index_name"]
    assert idx.exogenous_discount() == ref["exogenous_discount"]

    fixing = Date(int(ref["fixing_serial"]))
    value = idx.value_date(fixing)
    assert value.serial_number() == ref["value_date_serial"]
    assert idx.fixing_date(value).serial_number() == ref["fixing_date_serial"]
    assert idx.maturity_date(value).serial_number() == ref["maturity_date_serial"]


def test_exogenous_discount_is_set_when_a_discount_curve_is_given(
    cpp: dict[str, Any],
) -> None:
    """The probe only covers the single-curve ctor; pin the two-curve branch too.

    C++ has two constructors per family and only the second sets
    ``exogenousDiscount_``; PQuantLib merges them into one signature, so the
    flag has to key off the discounting argument being supplied.
    """
    assert cpp["euribor_swap_isda_fix_a_10y"]["exogenous_discount"] is False
    curve = FlatForward.from_rate(Date.from_ymd(15, Month.June, 2026), 0.03, Actual365Fixed())
    with_discount = EuriborSwapIsdaFixA(_10Y, None, curve)
    assert with_discount.exogenous_discount() is True
    assert with_discount.discounting_term_structure() is curve


def test_ois_index_inherits_calendar_and_day_counter_from_overnight_index() -> None:
    """Neither is declared at the OIS-index call site — both come from the index."""
    on = Eonia()
    idx = OvernightIndexedSwapIndex("EoniaSwapIsdaFix", _5Y, 2, EURCurrency(), on)
    assert idx.overnight_index() is on
    assert idx.fixing_calendar().name() == on.fixing_calendar().name()
    assert idx.day_counter().name() == on.day_counter().name()

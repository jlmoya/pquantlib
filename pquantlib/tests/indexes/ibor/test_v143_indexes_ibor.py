"""Cross-validate the ql/indexes/ibor/* families closed against C++ v1.43.

Probe: ``v143/indexes/ibor``.

These indexes are (almost all) thin constructor wrappers, so the meaningful
content is the wiring — name, fixing days, currency, fixing calendar, day
counter, business-day convention, end-of-month — plus the
fixing→value→maturity roll, which is where a wrong calendar or convention
actually shows up.

The test walks the reference JSON rather than restating its values: each
constructed index is looked up by key and every field the probe emitted is
compared. A key present in the reference but missing from ``_INDEXES`` fails
``test_every_reference_key_is_covered``, so a probe entry cannot be silently
skipped.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from pquantlib.currencies.europe import EURCurrency
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.business_252 import Business252
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.ibor.aonia import Aonia
from pquantlib.indexes.ibor.aud_libor import AUDLibor
from pquantlib.indexes.ibor.bbsw import (
    Bbsw,
    Bbsw1M,
    Bbsw2M,
    Bbsw3M,
    Bbsw4M,
    Bbsw5M,
    Bbsw6M,
)
from pquantlib.indexes.ibor.bibor import (
    Bibor,
    Bibor1M,
    Bibor1Y,
    Bibor2M,
    Bibor3M,
    Bibor6M,
    BiborSW,
)
from pquantlib.indexes.ibor.bkbm import (
    Bkbm,
    Bkbm1M,
    Bkbm2M,
    Bkbm3M,
    Bkbm4M,
    Bkbm5M,
    Bkbm6M,
)
from pquantlib.indexes.ibor.cad_libor import CADLibor, CADLiborON
from pquantlib.indexes.ibor.cdi import Cdi
from pquantlib.indexes.ibor.cdor import Cdor
from pquantlib.indexes.ibor.chf_libor import CHFLibor, DailyTenorCHFLibor
from pquantlib.indexes.ibor.corra import Corra
from pquantlib.indexes.ibor.custom import CustomIborIndex
from pquantlib.indexes.ibor.destr import Destr
from pquantlib.indexes.ibor.dkk_libor import DKKLibor
from pquantlib.indexes.ibor.eur_libor import (
    DailyTenorEURLibor,
    EURLibor,
    EURLibor1M,
    EURLibor1Y,
    EURLibor3M,
    EURLibor6M,
    EURLiborON,
)
from pquantlib.indexes.ibor.euribor import (
    Euribor1M,
    Euribor1W,
    Euribor1Y,
    Euribor3M,
    Euribor6M,
    Euribor365,
)
from pquantlib.indexes.ibor.jibar import Jibar
from pquantlib.indexes.ibor.jpy_libor import DailyTenorJPYLibor, JPYLibor
from pquantlib.indexes.ibor.kofr import Kofr
from pquantlib.indexes.ibor.mosprime import Mosprime
from pquantlib.indexes.ibor.nzd_libor import NZDLibor
from pquantlib.indexes.ibor.nzocr import Nzocr
from pquantlib.indexes.ibor.pribor import Pribor
from pquantlib.indexes.ibor.robor import Robor
from pquantlib.indexes.ibor.saron import Saron
from pquantlib.indexes.ibor.sek_libor import SEKLibor
from pquantlib.indexes.ibor.shibor import Shibor
from pquantlib.indexes.ibor.swestr import Swestr
from pquantlib.indexes.ibor.thbfix import THBFIX
from pquantlib.indexes.ibor.tibor import Tibor
from pquantlib.indexes.ibor.tonar import Tonar
from pquantlib.indexes.ibor.tr_libor import TRLibor
from pquantlib.indexes.ibor.wibor import Wibor
from pquantlib.indexes.ibor.zibor import Zibor
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.indexes.overnight_index import OvernightIndex
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import tight
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.brazil import Brazil
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.calendars.united_kingdom import UnitedKingdom
from pquantlib.time.calendars.united_states import UnitedStates
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_1W = Period(1, TimeUnit.Weeks)
_1M = Period(1, TimeUnit.Months)
_3M = Period(3, TimeUnit.Months)
_6M = Period(6, TimeUnit.Months)
_1D = Period(1, TimeUnit.Days)

# key in references/v143/indexes/ibor.json → factory for the Python index.
_INDEXES: dict[str, Callable[[], IborIndex]] = {
    # --- Oceania -----------------------------------------------------------
    "aonia": Aonia,
    "nzocr": Nzocr,
    "audlibor_3m": lambda: AUDLibor(_3M),
    "audlibor_1w": lambda: AUDLibor(_1W),
    "nzdlibor_3m": lambda: NZDLibor(_3M),
    "bbsw_1m": Bbsw1M,
    "bbsw_2m": Bbsw2M,
    "bbsw_3m": Bbsw3M,
    "bbsw_4m": Bbsw4M,
    "bbsw_5m": Bbsw5M,
    "bbsw_6m": Bbsw6M,
    "bkbm_1m": Bkbm1M,
    "bkbm_2m": Bkbm2M,
    "bkbm_3m": Bkbm3M,
    "bkbm_4m": Bkbm4M,
    "bkbm_5m": Bkbm5M,
    "bkbm_6m": Bkbm6M,
    # --- Asia --------------------------------------------------------------
    "bibor_sw": BiborSW,
    "bibor_1m": Bibor1M,
    "bibor_2m": Bibor2M,
    "bibor_3m": Bibor3M,
    "bibor_6m": Bibor6M,
    "bibor_1y": Bibor1Y,
    "thbfix_6m": lambda: THBFIX(_6M),
    "tibor_3m": lambda: Tibor(_3M),
    "tonar": Tonar,
    "jpylibor_3m": lambda: JPYLibor(_3M),
    "daily_tenor_jpylibor_0": lambda: DailyTenorJPYLibor(0),
    "daily_tenor_jpylibor_2": lambda: DailyTenorJPYLibor(2),
    "kofr": Kofr,
    "shibor_on": lambda: Shibor(_1D),
    "shibor_1w": lambda: Shibor(_1W),
    "shibor_3m": lambda: Shibor(_3M),
    # --- Americas ----------------------------------------------------------
    "cadlibor_3m": lambda: CADLibor(_3M),
    "cadlibor_on": CADLiborON,
    "cdor_3m": lambda: Cdor(_3M),
    "corra": Corra,
    "cdi": Cdi,
    # --- Europe ------------------------------------------------------------
    "chflibor_3m": lambda: CHFLibor(_3M),
    "daily_tenor_chflibor_0": lambda: DailyTenorCHFLibor(0),
    "daily_tenor_chflibor_2": lambda: DailyTenorCHFLibor(2),
    "saron": Saron,
    "zibor_3m": lambda: Zibor(_3M),
    "dkklibor_3m": lambda: DKKLibor(_3M),
    "destr": Destr,
    "seklibor_3m": lambda: SEKLibor(_3M),
    "swestr": Swestr,
    "eurlibor_on": EURLiborON,
    "daily_tenor_eurlibor_2": lambda: DailyTenorEURLibor(2),
    "eurlibor_1m": EURLibor1M,
    "eurlibor_3m": EURLibor3M,
    "eurlibor_6m": EURLibor6M,
    "eurlibor_1y": EURLibor1Y,
    "eurlibor_1w": lambda: EURLibor(_1W),
    "euribor_1w": Euribor1W,
    "euribor_1m": Euribor1M,
    "euribor_3m": Euribor3M,
    "euribor_6m": Euribor6M,
    "euribor_1y": Euribor1Y,
    "euribor365_3m": lambda: Euribor365(_3M),
    "euribor365_1w": lambda: Euribor365(_1W),
    "mosprime_on": lambda: Mosprime(_1D),
    "mosprime_3m": lambda: Mosprime(_3M),
    "pribor_on": lambda: Pribor(_1D),
    "pribor_3m": lambda: Pribor(_3M),
    "robor_on": lambda: Robor(_1D),
    "robor_3m": lambda: Robor(_3M),
    "wibor_on": lambda: Wibor(_1D),
    "wibor_3m": lambda: Wibor(_3M),
    "trlibor_3m": lambda: TRLibor(_3M),
    # --- Africa ------------------------------------------------------------
    "jibar_3m": lambda: Jibar(_3M),
    # --- CustomIborIndex: three deliberately different calendars ------------
    "custom": lambda: CustomIborIndex(
        "CustomIndex",
        _3M,
        2,
        EURCurrency(),
        UnitedKingdom(UnitedKingdom.Market.Exchange),
        TARGET(),
        UnitedStates(UnitedStates.Market.GovernmentBond),
        BusinessDayConvention.ModifiedFollowing,
        True,
        Actual365Fixed(),
    ),
}

# Reference keys that are not plain index-wiring entries.
_NON_WIRING_KEYS = frozenset({"cdi_forecast"})


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/indexes/ibor")


def test_every_reference_key_is_covered(cpp: dict[str, Any]) -> None:
    """No probe entry may be silently skipped by the walk below."""
    assert set(cpp) - _NON_WIRING_KEYS == set(_INDEXES)


@pytest.mark.parametrize("key", sorted(_INDEXES))
def test_index_matches_cpp(key: str, cpp: dict[str, Any]) -> None:
    ref = cpp[key]
    idx = _INDEXES[key]()

    assert idx.name() == ref["name"]
    assert idx.fixing_days() == ref["fixing_days"]
    assert idx.currency().code == ref["currency_code"]
    assert idx.fixing_calendar().name() == ref["fixing_calendar"]
    assert idx.day_counter().name() == ref["day_counter"]
    assert idx.tenor().length == ref["tenor_length"]
    assert int(idx.tenor().units) == ref["tenor_units"]
    assert int(idx.business_day_convention()) == ref["business_day_convention"]
    assert idx.end_of_month() == ref["end_of_month"]

    fixing = Date(int(ref["fixing_serial"]))
    value = idx.value_date(fixing)
    assert value.serial_number() == ref["value_date_serial"]
    assert idx.fixing_date(value).serial_number() == ref["fixing_date_serial"]
    assert idx.maturity_date(value).serial_number() == ref["maturity_date_serial"]


def test_cdi_forecast_fixing_is_compounded(cpp: dict[str, Any]) -> None:
    """Cdi overrides forecast_fixing with a compounded forward.

    Pinned against the same flat 5% curve the probe used, alongside the simple
    forward a plain ``OvernightIndex`` on identical wiring produces — so a port
    that inherited ``IborIndex.forecast_fixing`` could not pass.
    """
    ref = cpp["cdi_forecast"]
    today = Date.from_ymd(15, Month.June, 2026)
    curve = FlatForward.from_rate(today, 0.05, Actual365Fixed())

    fixing = Date(int(ref["fixing_serial"]))
    cdi = Cdi(curve)
    tight(cdi.forecast_fixing(fixing), ref["compounded"])

    plain = OvernightIndex(
        "CDI-plain", 0, cdi.currency(), Brazil(Brazil.Market.Settlement),
        Business252(Brazil()), curve,
    )
    tight(plain.forecast_fixing(fixing), ref["simple"])
    assert ref["compounded"] != ref["simple"]


def test_custom_ibor_index_clone_preserves_calendars() -> None:
    """``clone`` must carry all three calendars, not collapse to the fixing one."""
    idx = _INDEXES["custom"]()
    assert isinstance(idx, CustomIborIndex)
    clone = idx.clone(None)
    assert clone.fixing_calendar().name() == idx.fixing_calendar().name()
    assert clone.value_calendar().name() == idx.value_calendar().name()
    assert clone.maturity_calendar().name() == idx.maturity_calendar().name()


@pytest.mark.parametrize("family", [Bbsw, Bkbm, Bibor, EURLibor, Euribor365])
def test_daily_tenor_is_rejected(family: Callable[[Period], IborIndex]) -> None:
    """C++ forbids a daily tenor on these families; the port must too."""
    with pytest.raises(LibraryException, match="daily tenors"):
        family(_1D)

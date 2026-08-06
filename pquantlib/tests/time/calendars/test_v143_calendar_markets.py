"""Cross-validate EVERY market variant of every multi-market calendar.

Probe source: migration-harness/cpp/probes/v143_time_calmarkets/probe.cpp
Reference:    migration-harness/references/v143/time/calmarkets.json

WHY
---
C++ QuantLib implements each market of a calendar as a nested
``Calendar::Impl`` subclass — ``UnitedStates::NyseImpl``,
``Germany::XetraImpl``, ``Canada::TsxImpl``, ... — picked by the ``Market``
enum in the constructor. pquantlib collapses that hierarchy into one class
per calendar dispatching on a ``Market`` enum, so those C++ class names have
no Python counterpart *by name* and are allowlisted in
``migration-harness/check_coverage.py``.

That allowlisting is only honest if the *behaviour* each nested Impl carries
is implemented and pinned. ``time/calendars/all.json`` covers each calendar's
DEFAULT market for 2020-2030 only — it would not catch a market variant that
exists in name but has the wrong holiday rules. This module is the missing
half: every (calendar, market) pair, every non-weekend holiday over
1901-2099, plus the weekend mask, diffed against C++ v1.43.

Tolerance tier: EXACT. Calendars are pure integer/date logic; there is no
floating point anywhere in the comparison, so set equality is the only
acceptable bar.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from pquantlib.testing import reference_reader
from pquantlib.time.calendar import Calendar
from pquantlib.time.calendars.argentina import Argentina
from pquantlib.time.calendars.australia import Australia
from pquantlib.time.calendars.austria import Austria, AustriaMarket
from pquantlib.time.calendars.brazil import Brazil
from pquantlib.time.calendars.canada import Canada
from pquantlib.time.calendars.chile import Chile
from pquantlib.time.calendars.china import China
from pquantlib.time.calendars.croatia import Croatia, CroatiaMarket
from pquantlib.time.calendars.czech_republic import CzechRepublic
from pquantlib.time.calendars.czech_republic import Market as CzechMarket
from pquantlib.time.calendars.france import France, FranceMarket
from pquantlib.time.calendars.germany import Germany, GermanyMarket
from pquantlib.time.calendars.hong_kong import HongKong
from pquantlib.time.calendars.iceland import Iceland
from pquantlib.time.calendars.iceland import Market as IcelandMarket
from pquantlib.time.calendars.india import India
from pquantlib.time.calendars.indonesia import Indonesia
from pquantlib.time.calendars.italy import Italy, ItalyMarket
from pquantlib.time.calendars.malta import Malta, MaltaMarket
from pquantlib.time.calendars.mexico import Mexico, MexicoMarket
from pquantlib.time.calendars.montenegro import Montenegro, MontenegroMarket
from pquantlib.time.calendars.new_zealand import NewZealand
from pquantlib.time.calendars.north_macedonia import NorthMacedonia, NorthMacedoniaMarket
from pquantlib.time.calendars.poland import Poland, PolandMarket
from pquantlib.time.calendars.romania import Romania, RomaniaMarket
from pquantlib.time.calendars.russia import Russia, RussiaMarket
from pquantlib.time.calendars.saudi_arabia import SaudiArabia, SaudiArabiaMarket
from pquantlib.time.calendars.serbia import Serbia, SerbiaMarket
from pquantlib.time.calendars.singapore import Singapore, SingaporeMarket
from pquantlib.time.calendars.slovakia import Slovakia, SlovakiaMarket
from pquantlib.time.calendars.slovenia import Slovenia, SloveniaMarket
from pquantlib.time.calendars.south_korea import SouthKorea
from pquantlib.time.calendars.taiwan import Taiwan, TaiwanMarket
from pquantlib.time.calendars.thailand import Thailand
from pquantlib.time.calendars.ukraine import Market as UkraineMarket
from pquantlib.time.calendars.ukraine import Ukraine
from pquantlib.time.calendars.united_kingdom import UnitedKingdom
from pquantlib.time.calendars.united_states import UnitedStates
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.weekday import Weekday

# Probe section key -> factory for the equivalent pquantlib calendar.
# The comment on each line names the C++ nested Impl the pair pins down.
CASES: dict[str, Callable[[], Calendar]] = {
    # Argentina::MervalImpl
    "argentina_merval": lambda: Argentina(Argentina.Market.Merval),
    # Australia::SettlementImpl / Australia::AsxImpl
    "australia_settlement": lambda: Australia(Australia.Market.Settlement),
    "australia_asx": lambda: Australia(Australia.Market.ASX),
    # Austria::SettlementImpl / Austria::ExchangeImpl
    "austria_settlement": lambda: Austria(AustriaMarket.Settlement),
    "austria_exchange": lambda: Austria(AustriaMarket.Exchange),
    # Brazil::SettlementImpl / Brazil::ExchangeImpl
    "brazil_settlement": lambda: Brazil(Brazil.Market.Settlement),
    "brazil_exchange": lambda: Brazil(Brazil.Market.Exchange),
    # Canada::SettlementImpl / Canada::TsxImpl
    "canada_settlement": lambda: Canada(Canada.Market.Settlement),
    "canada_tsx": lambda: Canada(Canada.Market.TSX),
    # Chile::SseImpl
    "chile_sse": lambda: Chile(Chile.Market.SSE),
    # China::SseImpl / China::IbImpl
    "china_sse": lambda: China(China.Market.SSE),
    "china_ib": lambda: China(China.Market.IB),
    # Croatia::ZseImpl
    "croatia_zse": lambda: Croatia(CroatiaMarket.ZSE),
    # CzechRepublic::PseImpl
    "czech_republic_pse": lambda: CzechRepublic(CzechMarket.PSE),
    # France::SettlementImpl / France::ExchangeImpl
    "france_settlement": lambda: France(FranceMarket.Settlement),
    "france_exchange": lambda: France(FranceMarket.Exchange),
    # Germany::SettlementImpl / FrankfurtStockExchangeImpl / XetraImpl /
    # EurexImpl / EuwaxImpl
    "germany_settlement": lambda: Germany(GermanyMarket.Settlement),
    "germany_frankfurt_stock_exchange": lambda: Germany(GermanyMarket.FrankfurtStockExchange),
    "germany_xetra": lambda: Germany(GermanyMarket.Xetra),
    "germany_eurex": lambda: Germany(GermanyMarket.Eurex),
    "germany_euwax": lambda: Germany(GermanyMarket.Euwax),
    # HongKong::HkexImpl
    "hong_kong_hkex": lambda: HongKong(HongKong.Market.HKEx),
    # Iceland::IcexImpl
    "iceland_icex": lambda: Iceland(IcelandMarket.ICEX),
    # India::NseImpl
    "india_nse": lambda: India(India.Market.NSE),
    # Indonesia::BejImpl (BEJ == JSX == IDX all select BejImpl)
    "indonesia_bej": lambda: Indonesia(Indonesia.Market.BEJ),
    "indonesia_jsx": lambda: Indonesia(Indonesia.Market.JSX),
    "indonesia_idx": lambda: Indonesia(Indonesia.Market.IDX),
    # Italy::SettlementImpl / Italy::ExchangeImpl
    "italy_settlement": lambda: Italy(ItalyMarket.Settlement),
    "italy_exchange": lambda: Italy(ItalyMarket.Exchange),
    # Malta::MseImpl
    "malta_mse": lambda: Malta(MaltaMarket.MSE),
    # Mexico::BmvImpl
    "mexico_bmv": lambda: Mexico(MexicoMarket.BMV),
    # Montenegro::MnseImpl
    "montenegro_mnse": lambda: Montenegro(MontenegroMarket.MNSE),
    # NewZealand::CommonImpl -> WellingtonImpl / AucklandImpl
    "new_zealand_wellington": lambda: NewZealand(NewZealand.Market.Wellington),
    "new_zealand_auckland": lambda: NewZealand(NewZealand.Market.Auckland),
    # NorthMacedonia::MseImpl
    "north_macedonia_mse": lambda: NorthMacedonia(NorthMacedoniaMarket.MSE),
    # Poland::SettlementImpl / Poland::WseImpl
    "poland_settlement": lambda: Poland(PolandMarket.Settlement),
    "poland_wse": lambda: Poland(PolandMarket.WSE),
    # Romania::PublicImpl / Romania::BVBImpl
    "romania_public": lambda: Romania(RomaniaMarket.Public),
    "romania_bvb": lambda: Romania(RomaniaMarket.BVB),
    # Russia::SettlementImpl / Russia::ExchangeImpl
    "russia_settlement": lambda: Russia(RussiaMarket.Settlement),
    "russia_moex": lambda: Russia(RussiaMarket.MOEX),
    # SaudiArabia::TadawulImpl
    "saudi_arabia_tadawul": lambda: SaudiArabia(SaudiArabiaMarket.Tadawul),
    # Serbia::BseImpl
    "serbia_bse": lambda: Serbia(SerbiaMarket.BSE),
    # Singapore::SgxImpl
    "singapore_sgx": lambda: Singapore(SingaporeMarket.SGX),
    # Slovakia::BsseImpl
    "slovakia_bsse": lambda: Slovakia(SlovakiaMarket.BSSE),
    # Slovenia::LseImpl
    "slovenia_lse": lambda: Slovenia(SloveniaMarket.LSE),
    # SouthKorea::SettlementImpl / SouthKorea::KrxImpl
    "south_korea_settlement": lambda: SouthKorea(SouthKorea.Market.Settlement),
    "south_korea_krx": lambda: SouthKorea(SouthKorea.Market.KRX),
    # Taiwan::TsecImpl
    "taiwan_tsec": lambda: Taiwan(TaiwanMarket.TSEC),
    # Thailand::SetImpl
    "thailand_set": Thailand,
    # Ukraine::UseImpl
    "ukraine_use": lambda: Ukraine(UkraineMarket.USE),
    # UnitedKingdom::SettlementImpl / ExchangeImpl / MetalsImpl
    "united_kingdom_settlement": lambda: UnitedKingdom(UnitedKingdom.Market.Settlement),
    "united_kingdom_exchange": lambda: UnitedKingdom(UnitedKingdom.Market.Exchange),
    "united_kingdom_metals": lambda: UnitedKingdom(UnitedKingdom.Market.Metals),
    # UnitedStates::SettlementImpl / NyseImpl / GovernmentBondImpl / NercImpl /
    # LiborImpactImpl / FederalReserveImpl / SofrImpl
    "united_states_settlement": lambda: UnitedStates(UnitedStates.Market.Settlement),
    "united_states_nyse": lambda: UnitedStates(UnitedStates.Market.NYSE),
    "united_states_government_bond": lambda: UnitedStates(UnitedStates.Market.GovernmentBond),
    "united_states_nerc": lambda: UnitedStates(UnitedStates.Market.NERC),
    "united_states_libor_impact": lambda: UnitedStates(UnitedStates.Market.LiborImpact),
    "united_states_federal_reserve": lambda: UnitedStates(UnitedStates.Market.FederalReserve),
    "united_states_sofr": lambda: UnitedStates(UnitedStates.Market.SOFR),
}

_WEEKDAYS: tuple[Weekday, ...] = (
    Weekday.Sunday,
    Weekday.Monday,
    Weekday.Tuesday,
    Weekday.Wednesday,
    Weekday.Thursday,
    Weekday.Friday,
    Weekday.Saturday,
)


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/time/calmarkets")


def test_every_probe_section_has_a_python_case(cpp: dict[str, Any]) -> None:
    """The probe and this module must enumerate exactly the same pairs.

    Guards against a market silently dropping out of either side.
    """
    assert set(cpp) == set(CASES)


@pytest.mark.exact
@pytest.mark.parametrize("key", sorted(CASES))
def test_market_matches_cpp(key: str, cpp: dict[str, Any]) -> None:
    section = cpp[key]
    cal = CASES[key]()

    assert cal.name() == section["name"]

    weekend = [cal.is_weekend(w) for w in _WEEKDAYS]
    assert weekend == section["weekend"]

    frm = Date.from_ymd(1, Month.January, int(section["from_year"]))
    to = Date.from_ymd(31, Month.December, int(section["to_year"]))
    actual = {d.serial for d in cal.holiday_list(frm, to, include_weekends=False)}
    expected = {int(s) for s in section["holidays"]}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    assert actual == expected, (
        f"{key}: {len(missing)} C++ holidays not in Python "
        f"{[str(Date(s)) for s in missing[:10]]}; "
        f"{len(extra)} Python holidays not in C++ "
        f"{[str(Date(s)) for s in extra[:10]]}"
    )

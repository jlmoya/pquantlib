// v1.43 reference: every market variant of every multi-market calendar.
//
// WHY THIS PROBE EXISTS
// ---------------------
// C++ QuantLib implements each market of a calendar as a *nested*
// `Calendar::Impl` subclass — `UnitedStates::NyseImpl`, `Germany::XetraImpl`,
// `Canada::TsxImpl`, ... — selected by the `Market` enum in the constructor.
// The pquantlib port collapses that hierarchy into one Python class per
// calendar that dispatches on a nested `Market` IntEnum, so those ~50 C++
// class NAMES have no Python counterpart by name.
//
// Allowlisting them in migration-harness/check_coverage.py is only honest if
// the BEHAVIOUR each one carries is actually implemented and pinned. The
// existing time/calendars/all.json reference covers each calendar's DEFAULT
// market for 2020-2030 only, which would not catch a market variant that
// exists in name but has the wrong holiday rules.
//
// This probe closes that hole: for every (calendar, market) pair it emits
//   * the C++ `name()` string,
//   * the weekend mask (isWeekend for Sunday..Saturday),
//   * every non-weekend holiday over 1901-2099, as Date serial numbers.
//
// Weekend days are excluded from the holiday list because they are fully
// determined by the weekend mask and would multiply the reference size by
// ~15x. Together, `weekend` + `holidays` determine isBusinessDay on every
// date in range, which is the whole observable surface of a Calendar::Impl.
//
// Range rationale: 1901 is the first year of QuantLib's easterMonday table
// (Calendar::easterMonday indexes easterMondayTable[y - 1901]), so no
// Western/Orthodox calendar is defined before it. 2099 keeps the reference
// around 1 MB while still covering ~80 years past the last hard-coded
// holiday table. Russia's MOEX impl is the one exception: russia.cpp:231
// QL_FAILs for years before 2012, so its range starts there.
//
// Output shape:
//   {
//     "<calendar>_<market>": {
//       "name": "...",
//       "from_year": 1901, "to_year": 2099,
//       "weekend": [bool x7],            // index 0 = Sunday .. 6 = Saturday
//       "holidays": [<Date serial>, ...] // ascending, non-weekend only
//     },
//     ...
//   }

#include <ql/time/calendars/argentina.hpp>
#include <ql/time/calendars/australia.hpp>
#include <ql/time/calendars/austria.hpp>
#include <ql/time/calendars/brazil.hpp>
#include <ql/time/calendars/canada.hpp>
#include <ql/time/calendars/chile.hpp>
#include <ql/time/calendars/china.hpp>
#include <ql/time/calendars/croatia.hpp>
#include <ql/time/calendars/czechrepublic.hpp>
#include <ql/time/calendars/france.hpp>
#include <ql/time/calendars/germany.hpp>
#include <ql/time/calendars/hongkong.hpp>
#include <ql/time/calendars/iceland.hpp>
#include <ql/time/calendars/india.hpp>
#include <ql/time/calendars/indonesia.hpp>
#include <ql/time/calendars/italy.hpp>
#include <ql/time/calendars/malta.hpp>
#include <ql/time/calendars/mexico.hpp>
#include <ql/time/calendars/montenegro.hpp>
#include <ql/time/calendars/newzealand.hpp>
#include <ql/time/calendars/northmacedonia.hpp>
#include <ql/time/calendars/poland.hpp>
#include <ql/time/calendars/romania.hpp>
#include <ql/time/calendars/russia.hpp>
#include <ql/time/calendars/saudiarabia.hpp>
#include <ql/time/calendars/serbia.hpp>
#include <ql/time/calendars/singapore.hpp>
#include <ql/time/calendars/slovakia.hpp>
#include <ql/time/calendars/slovenia.hpp>
#include <ql/time/calendars/southkorea.hpp>
#include <ql/time/calendars/taiwan.hpp>
#include <ql/time/calendars/thailand.hpp>
#include <ql/time/calendars/ukraine.hpp>
#include <ql/time/calendars/unitedkingdom.hpp>
#include <ql/time/calendars/unitedstates.hpp>
#include <ql/time/calendar.hpp>
#include <ql/time/date.hpp>

#include <iostream>
#include <string>

using namespace QuantLib;

namespace {

constexpr Year kDefaultFromYear = 1901;
constexpr Year kDefaultToYear = 2099;

bool g_first_section = true;

void emit(const std::string& key,
          const Calendar& cal,
          Year from_year = kDefaultFromYear,
          Year to_year = kDefaultToYear) {
    if (!g_first_section)
        std::cout << ",\n";
    g_first_section = false;

    std::cout << "  \"" << key << "\": {\n";
    std::cout << "    \"name\": \"" << cal.name() << "\",\n";
    std::cout << "    \"from_year\": " << from_year << ",\n";
    std::cout << "    \"to_year\": " << to_year << ",\n";
    std::cout << "    \"weekend\": [";
    for (int w = Sunday; w <= Saturday; ++w) {
        if (w != Sunday) std::cout << ", ";
        std::cout << (cal.isWeekend(static_cast<Weekday>(w)) ? "true" : "false");
    }
    std::cout << "],\n";
    std::cout << "    \"holidays\": [";
    const Date from(1, January, from_year);
    const Date to(31, December, to_year);
    const auto hs = cal.holidayList(from, to, /*includeWeekEnds=*/false);
    bool first = true;
    for (const auto& d : hs) {
        if (!first) std::cout << ",";
        std::cout << d.serialNumber();
        first = false;
    }
    std::cout << "]\n  }";
}

}  // namespace

int main() {
    std::cout << "{\n";

    // --- Argentina::MervalImpl -------------------------------------------
    emit("argentina_merval", Argentina(Argentina::Merval));

    // --- Australia::SettlementImpl / AsxImpl ------------------------------
    emit("australia_settlement", Australia(Australia::Settlement));
    emit("australia_asx", Australia(Australia::ASX));

    // --- Austria::SettlementImpl / ExchangeImpl ---------------------------
    emit("austria_settlement", Austria(Austria::Settlement));
    emit("austria_exchange", Austria(Austria::Exchange));

    // --- Brazil::SettlementImpl / ExchangeImpl ----------------------------
    emit("brazil_settlement", Brazil(Brazil::Settlement));
    emit("brazil_exchange", Brazil(Brazil::Exchange));

    // --- Canada::SettlementImpl / TsxImpl ---------------------------------
    emit("canada_settlement", Canada(Canada::Settlement));
    emit("canada_tsx", Canada(Canada::TSX));

    // --- Chile::SseImpl ---------------------------------------------------
    emit("chile_sse", Chile(Chile::SSE));

    // --- China::SseImpl / IbImpl ------------------------------------------
    emit("china_sse", China(China::SSE));
    emit("china_ib", China(China::IB));

    // --- Croatia::ZseImpl -------------------------------------------------
    emit("croatia_zse", Croatia(Croatia::ZSE));

    // --- CzechRepublic::PseImpl -------------------------------------------
    emit("czech_republic_pse", CzechRepublic(CzechRepublic::PSE));

    // --- France::SettlementImpl / ExchangeImpl ----------------------------
    emit("france_settlement", France(France::Settlement));
    emit("france_exchange", France(France::Exchange));

    // --- Germany: SettlementImpl / FrankfurtStockExchangeImpl / XetraImpl /
    //     EurexImpl / EuwaxImpl ---------------------------------------------
    emit("germany_settlement", Germany(Germany::Settlement));
    emit("germany_frankfurt_stock_exchange", Germany(Germany::FrankfurtStockExchange));
    emit("germany_xetra", Germany(Germany::Xetra));
    emit("germany_eurex", Germany(Germany::Eurex));
    emit("germany_euwax", Germany(Germany::Euwax));

    // --- HongKong::HkexImpl -----------------------------------------------
    emit("hong_kong_hkex", HongKong(HongKong::HKEx));

    // --- Iceland::IcexImpl ------------------------------------------------
    emit("iceland_icex", Iceland(Iceland::ICEX));

    // --- India::NseImpl ---------------------------------------------------
    emit("india_nse", India(India::NSE));

    // --- Indonesia::BejImpl (BEJ == JSX == IDX all map to BejImpl) --------
    emit("indonesia_bej", Indonesia(Indonesia::BEJ));
    emit("indonesia_jsx", Indonesia(Indonesia::JSX));
    emit("indonesia_idx", Indonesia(Indonesia::IDX));

    // --- Italy::SettlementImpl / ExchangeImpl -----------------------------
    emit("italy_settlement", Italy(Italy::Settlement));
    emit("italy_exchange", Italy(Italy::Exchange));

    // --- Malta::MseImpl ---------------------------------------------------
    emit("malta_mse", Malta(Malta::MSE));

    // --- Mexico::BmvImpl --------------------------------------------------
    emit("mexico_bmv", Mexico(Mexico::BMV));

    // --- Montenegro::MnseImpl ---------------------------------------------
    emit("montenegro_mnse", Montenegro(Montenegro::MNSE));

    // --- NewZealand::CommonImpl -> WellingtonImpl / AucklandImpl ----------
    emit("new_zealand_wellington", NewZealand(NewZealand::Wellington));
    emit("new_zealand_auckland", NewZealand(NewZealand::Auckland));

    // --- NorthMacedonia::MseImpl ------------------------------------------
    emit("north_macedonia_mse", NorthMacedonia(NorthMacedonia::MSE));

    // --- Poland::SettlementImpl / WseImpl ---------------------------------
    emit("poland_settlement", Poland(Poland::Settlement));
    emit("poland_wse", Poland(Poland::WSE));

    // --- Romania::PublicImpl / BVBImpl ------------------------------------
    emit("romania_public", Romania(Romania::Public));
    emit("romania_bvb", Romania(Romania::BVB));

    // --- Russia::SettlementImpl / ExchangeImpl ----------------------------
    // ExchangeImpl QL_FAILs before 2012 (russia.cpp:231).
    emit("russia_settlement", Russia(Russia::Settlement));
    emit("russia_moex", Russia(Russia::MOEX), 2012, kDefaultToYear);

    // --- SaudiArabia::TadawulImpl -----------------------------------------
    emit("saudi_arabia_tadawul", SaudiArabia(SaudiArabia::Tadawul));

    // --- Serbia::BseImpl --------------------------------------------------
    emit("serbia_bse", Serbia(Serbia::BSE));

    // --- Singapore::SgxImpl -----------------------------------------------
    emit("singapore_sgx", Singapore(Singapore::SGX));

    // --- Slovakia::BsseImpl -----------------------------------------------
    emit("slovakia_bsse", Slovakia(Slovakia::BSSE));

    // --- Slovenia::LseImpl ------------------------------------------------
    emit("slovenia_lse", Slovenia(Slovenia::LSE));

    // --- SouthKorea::SettlementImpl / KrxImpl -----------------------------
    emit("south_korea_settlement", SouthKorea(SouthKorea::Settlement));
    emit("south_korea_krx", SouthKorea(SouthKorea::KRX));

    // --- Taiwan::TsecImpl -------------------------------------------------
    emit("taiwan_tsec", Taiwan(Taiwan::TSEC));

    // --- Thailand::SetImpl (single-market calendar) -----------------------
    emit("thailand_set", Thailand());

    // --- Ukraine::UseImpl -------------------------------------------------
    emit("ukraine_use", Ukraine(Ukraine::USE));

    // --- UnitedKingdom::SettlementImpl / ExchangeImpl / MetalsImpl --------
    emit("united_kingdom_settlement", UnitedKingdom(UnitedKingdom::Settlement));
    emit("united_kingdom_exchange", UnitedKingdom(UnitedKingdom::Exchange));
    emit("united_kingdom_metals", UnitedKingdom(UnitedKingdom::Metals));

    // --- UnitedStates: SettlementImpl / NyseImpl / GovernmentBondImpl /
    //     NercImpl / LiborImpactImpl / FederalReserveImpl / SofrImpl -------
    emit("united_states_settlement", UnitedStates(UnitedStates::Settlement));
    emit("united_states_nyse", UnitedStates(UnitedStates::NYSE));
    emit("united_states_government_bond", UnitedStates(UnitedStates::GovernmentBond));
    emit("united_states_nerc", UnitedStates(UnitedStates::NERC));
    emit("united_states_libor_impact", UnitedStates(UnitedStates::LiborImpact));
    emit("united_states_federal_reserve", UnitedStates(UnitedStates::FederalReserve));
    emit("united_states_sofr", UnitedStates(UnitedStates::SOFR));

    std::cout << "\n}\n";
    return 0;
}

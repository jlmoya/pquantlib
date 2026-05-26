// Emit holiday-set reference values for all 41 sovereign/exchange calendars,
// each default-constructed, for years 2020..2030. Output JSON shape:
//
//   {
//     "<canonical_key>": {
//       "name": "<C++ name()>",
//       "holidays": [{"y":..., "m":..., "d":...}, ...]
//     },
//     ...
//   }
//
// The canonical key is the snake_case form of the C++ class (e.g.
// "united_states", "czech_republic", "south_korea", "saudi_arabia").
// Holidays are the union of weekend + non-weekend holidays in the
// inclusive [Jan 1 2020, Dec 31 2030] range — exactly what the
// Python test will assert via set equality.

#include <ql/time/calendars/argentina.hpp>
#include <ql/time/calendars/australia.hpp>
#include <ql/time/calendars/austria.hpp>
#include <ql/time/calendars/botswana.hpp>
#include <ql/time/calendars/brazil.hpp>
#include <ql/time/calendars/canada.hpp>
#include <ql/time/calendars/chile.hpp>
#include <ql/time/calendars/china.hpp>
#include <ql/time/calendars/czechrepublic.hpp>
#include <ql/time/calendars/denmark.hpp>
#include <ql/time/calendars/finland.hpp>
#include <ql/time/calendars/france.hpp>
#include <ql/time/calendars/germany.hpp>
#include <ql/time/calendars/hongkong.hpp>
#include <ql/time/calendars/hungary.hpp>
#include <ql/time/calendars/iceland.hpp>
#include <ql/time/calendars/india.hpp>
#include <ql/time/calendars/indonesia.hpp>
#include <ql/time/calendars/israel.hpp>
#include <ql/time/calendars/italy.hpp>
#include <ql/time/calendars/japan.hpp>
#include <ql/time/calendars/mexico.hpp>
#include <ql/time/calendars/newzealand.hpp>
#include <ql/time/calendars/norway.hpp>
#include <ql/time/calendars/poland.hpp>
#include <ql/time/calendars/romania.hpp>
#include <ql/time/calendars/russia.hpp>
#include <ql/time/calendars/saudiarabia.hpp>
#include <ql/time/calendars/singapore.hpp>
#include <ql/time/calendars/slovakia.hpp>
#include <ql/time/calendars/southafrica.hpp>
#include <ql/time/calendars/southkorea.hpp>
#include <ql/time/calendars/sweden.hpp>
#include <ql/time/calendars/switzerland.hpp>
#include <ql/time/calendars/taiwan.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/calendars/thailand.hpp>
#include <ql/time/calendars/turkey.hpp>
#include <ql/time/calendars/ukraine.hpp>
#include <ql/time/calendars/unitedkingdom.hpp>
#include <ql/time/calendars/unitedstates.hpp>
#include <ql/time/date.hpp>
#include <ql/time/calendar.hpp>

#include <iostream>
#include <string>

using namespace QuantLib;

namespace {

constexpr Year kFromYear = 2020;
constexpr Year kToYear = 2030;

// Emit a calendar section. Holidays are the dates with isHoliday==true AND
// !isWeekend (we exclude weekends because they're identifiable from the
// weekday and would balloon the JSON to ~5000 dates per calendar; the
// weekend rule is verified separately via the per-calendar _is_weekend
// method in the Python test).
void emit(const char* key, const Calendar& cal, bool trailing_comma) {
    std::cout << "  \"" << key << "\": {\n";
    std::cout << "    \"name\": \"" << cal.name() << "\",\n";
    std::cout << "    \"holidays\": [";
    Date from(1, January, kFromYear);
    Date to(31, December, kToYear);
    auto hs = cal.holidayList(from, to, /*includeWeekEnds=*/false);
    bool first = true;
    for (const auto& d : hs) {
        if (!first) std::cout << ", ";
        std::cout << "{\"y\":" << d.year()
                  << ",\"m\":" << static_cast<int>(d.month())
                  << ",\"d\":" << d.dayOfMonth() << "}";
        first = false;
    }
    std::cout << "]\n  }" << (trailing_comma ? "," : "") << "\n";
}

}  // namespace

int main() {
    std::cout << "{\n";

    emit("argentina",      Argentina(),     true);
    emit("australia",      Australia(),     true);
    emit("austria",        Austria(),       true);
    emit("botswana",       Botswana(),      true);
    emit("brazil",         Brazil(),        true);
    emit("canada",         Canada(),        true);
    emit("chile",          Chile(),         true);
    emit("china",          China(),         true);
    emit("czech_republic", CzechRepublic(), true);
    emit("denmark",        Denmark(),       true);
    emit("finland",        Finland(),       true);
    emit("france",         France(),        true);
    emit("germany",        Germany(),       true);
    emit("hong_kong",      HongKong(),      true);
    emit("hungary",        Hungary(),       true);
    emit("iceland",        Iceland(),       true);
    emit("india",          India(),         true);
    emit("indonesia",      Indonesia(),     true);
    emit("israel",         Israel(),        true);
    emit("italy",          Italy(),         true);
    emit("japan",          Japan(),         true);
    emit("mexico",         Mexico(),        true);
    emit("new_zealand",    NewZealand(),    true);
    emit("norway",         Norway(),        true);
    emit("poland",         Poland(),        true);
    emit("romania",        Romania(),       true);
    emit("russia",         Russia(),        true);
    emit("saudi_arabia",   SaudiArabia(),   true);
    emit("singapore",      Singapore(),     true);
    emit("slovakia",       Slovakia(),      true);
    emit("south_africa",   SouthAfrica(),   true);
    emit("south_korea",    SouthKorea(),    true);
    emit("sweden",         Sweden(),        true);
    emit("switzerland",    Switzerland(),   true);
    emit("taiwan",         Taiwan(),        true);
    emit("target",         TARGET(),        true);
    emit("thailand",       Thailand(),      true);
    emit("turkey",         Turkey(),        true);
    emit("ukraine",        Ukraine(),       true);
    emit("united_kingdom", UnitedKingdom(), true);
    emit("united_states",  UnitedStates(UnitedStates::Settlement),  false);

    std::cout << "}\n";
    return 0;
}

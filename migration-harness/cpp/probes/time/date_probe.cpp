// Emit comprehensive Date reference values: serial↔(y,m,d), weekday,
// day-of-year, leap-year, arithmetic, end/start-of-month, nth weekday.
//
// Output JSON sections:
//   {
//     "constants": {"min_serial": 367, "max_serial": 109574},
//     "leap_years": {"1900": true, ...},
//     "from_ymd": [{"y":..,"m":..,"d":..,"serial":..,"weekday":..,
//                    "day_of_year":..}, ...],
//     "from_serial": [{"serial":.., "y":.., "m":.., "d":.., "weekday":..}, ...],
//     "add_days": [{"y":..,"m":..,"d":..,"n":..,"out_serial":..}, ...],
//     "add_period": [{"y":..,"m":..,"d":..,"n":..,"units":"Months",
//                     "out_y":..,"out_m":..,"out_d":..}, ...],
//     "diff": [{"y1":..,"m1":..,"d1":.., ..., "diff":..}, ...],
//     "end_of_month": [{"y":..,"m":..,"d":..,
//                       "eom_d":.., "is_eom": false}, ...],
//     "start_of_month": [{"y":..,"m":..,"d":..,
//                         "som_d":.., "is_som": false}, ...],
//     "next_weekday": [{"y":..,"m":..,"d":..,"target":"Friday",
//                       "out_y":..,"out_m":..,"out_d":..}, ...],
//     "nth_weekday": [{"n":..,"w":"Thursday","m":"March","y":..,
//                      "out_d":..}, ...]
//   }

#include <ql/time/date.hpp>
#include <ql/time/weekday.hpp>
#include <ql/time/period.hpp>
#include <ql/time/timeunit.hpp>

#include <iostream>

using namespace QuantLib;

namespace {

const char* mname(Month m) {
    switch (m) {
      case January:   return "January";
      case February:  return "February";
      case March:     return "March";
      case April:     return "April";
      case May:       return "May";
      case June:      return "June";
      case July:      return "July";
      case August:    return "August";
      case September: return "September";
      case October:   return "October";
      case November:  return "November";
      case December:  return "December";
    }
    return "?";
}

const char* wname(Weekday w) {
    switch (w) {
      case Sunday:    return "Sunday";
      case Monday:    return "Monday";
      case Tuesday:   return "Tuesday";
      case Wednesday: return "Wednesday";
      case Thursday:  return "Thursday";
      case Friday:    return "Friday";
      case Saturday:  return "Saturday";
    }
    return "?";
}

const char* uname(TimeUnit u) {
    switch (u) {
      case Days:   return "Days";
      case Weeks:  return "Weeks";
      case Months: return "Months";
      case Years:  return "Years";
      default:     return "?";
    }
}

}  // namespace

int main() {
    std::cout << "{\n";

    // --- constants ---------------------------------------------------------
    std::cout << "  \"constants\": {\n";
    std::cout << "    \"min_serial\": " << Date::minDate().serialNumber() << ",\n";
    std::cout << "    \"max_serial\": " << Date::maxDate().serialNumber() << ",\n";
    std::cout << "    \"min_y\": " << Date::minDate().year() << ",\n";
    std::cout << "    \"min_m\": " << static_cast<int>(Date::minDate().month()) << ",\n";
    std::cout << "    \"min_d\": " << Date::minDate().dayOfMonth() << ",\n";
    std::cout << "    \"max_y\": " << Date::maxDate().year() << ",\n";
    std::cout << "    \"max_m\": " << static_cast<int>(Date::maxDate().month()) << ",\n";
    std::cout << "    \"max_d\": " << Date::maxDate().dayOfMonth() << "\n";
    std::cout << "  },\n";

    // --- leap years (every 4 years from 1900 to 2200) ----------------------
    std::cout << "  \"leap_years\": {\n";
    bool first = true;
    for (int y = 1900; y <= 2200; ++y) {
        if (!first) std::cout << ",\n";
        std::cout << "    \"" << y << "\": " << (Date::isLeap(y) ? "true" : "false");
        first = false;
    }
    std::cout << "\n  },\n";

    // --- from_ymd: comprehensive (y, m, d) → serial + weekday + day-of-year
    struct YMDCase { Year y; Month m; Day d; };
    const YMDCase ymd_cases[] = {
        {1901,  January,    1},  // min
        {1901,  January,    2},
        {1901,  December,  31},
        {1902,  January,    1},
        {1904,  February,  29},  // leap day
        {2000,  January,    1},
        {2000,  February,  29},  // leap day
        {2000,  December,  31},
        {2024,  February,  29},  // leap day
        {2024,  March,      1},
        {2024,  June,      15},
        {2024, October,    31},
        {2025,  January,    1},
        {2026,  May,       23},  // today
        {2100,  February,  28},  // 2100 is NOT leap
        {2100,  March,      1},
        {2199, December,   31},  // max
    };
    std::cout << "  \"from_ymd\": [\n";
    {
        bool first_local = true;
        for (const auto& c : ymd_cases) {
            if (!first_local) std::cout << ",\n";
            Date dt(c.d, c.m, c.y);
            std::cout << "    {\"y\": " << c.y
                      << ", \"m\": " << static_cast<int>(c.m)
                      << ", \"d\": " << c.d
                      << ", \"serial\": " << dt.serialNumber()
                      << ", \"weekday\": \"" << wname(dt.weekday()) << "\""
                      << ", \"day_of_year\": " << dt.dayOfYear() << "}";
            first_local = false;
        }
        std::cout << "\n  ],\n";
    }

    // --- from_serial: 12 sampled serials → (y, m, d, weekday) -------------
    const Date::serial_type serials[] = {
        367,    // 1901-01-01
        368,    // 1901-01-02
        731,    // 1901-12-31
        732,    // 1902-01-01
        1521,   // 1904-02-29 (leap)
        36526,  // 2000-01-01
        45290,  // 2023-12-31
        46021,  // 2025-12-31
        109574, // 2199-12-31 (max)
    };
    std::cout << "  \"from_serial\": [\n";
    {
        bool first_local = true;
        for (const auto& s : serials) {
            if (!first_local) std::cout << ",\n";
            Date dt(s);
            std::cout << "    {\"serial\": " << s
                      << ", \"y\": " << dt.year()
                      << ", \"m\": " << static_cast<int>(dt.month())
                      << ", \"d\": " << dt.dayOfMonth()
                      << ", \"weekday\": \"" << wname(dt.weekday()) << "\""
                      << ", \"day_of_year\": " << dt.dayOfYear() << "}";
            first_local = false;
        }
        std::cout << "\n  ],\n";
    }

    // --- arithmetic: Date + int days --------------------------------------
    struct AddCase { Year y; Month m; Day d; int n; };
    const AddCase add_cases[] = {
        {2024,  January,   15,   1},
        {2024,  January,   31,   1},   // month boundary
        {2024,  February, 28,    1},   // leap-year Feb→Mar
        {2024,  February, 29,    1},   // Mar 1 next
        {2023,  February, 28,    1},   // non-leap Feb→Mar
        {2023, December,  31,    1},   // year boundary
        {2024, December,  31,    1},   // year boundary into 2025
        {2024,  January,   15, -14},   // negative
        {2024,  January,    1, -1},    // back to 2023-12-31
        {2024,  March,     15,  90},
        {2024,  March,     15,-100},
    };
    std::cout << "  \"add_days\": [\n";
    {
        bool first_local = true;
        for (const auto& c : add_cases) {
            if (!first_local) std::cout << ",\n";
            Date dt(c.d, c.m, c.y);
            Date r = dt + c.n;
            std::cout << "    {\"y\": " << c.y
                      << ", \"m\": " << static_cast<int>(c.m)
                      << ", \"d\": " << c.d
                      << ", \"n\": " << c.n
                      << ", \"out_serial\": " << r.serialNumber()
                      << ", \"out_y\": " << r.year()
                      << ", \"out_m\": " << static_cast<int>(r.month())
                      << ", \"out_d\": " << r.dayOfMonth() << "}";
            first_local = false;
        }
        std::cout << "\n  ],\n";
    }

    // --- arithmetic: Date + Period (Months, Years with EOM clipping) ------
    struct AddPeriodCase { Year y; Month m; Day d; int n; TimeUnit u; };
    const AddPeriodCase add_period_cases[] = {
        {2024,  January,   15,  1, Months},
        {2024,  January,   31,  1, Months}, // Jan 31 + 1mo → Feb 29 (leap)
        {2023,  January,   31,  1, Months}, // Jan 31 + 1mo → Feb 28 (non-leap)
        {2024,  January,   31, 13, Months}, // 14 months
        {2024,  February,  29,  1, Years},  // leap → non-leap: Feb 29 → Feb 28
        {2024,  February,  29,  4, Years},  // → 2028-02-29 (leap again)
        {2024,  March,     15,  1, Weeks},
        {2024,  March,     15, -1, Weeks},
        {2024,  March,     15, 30, Days},
        {2024,  August,    31,  1, Months}, // Aug 31 + 1mo → Sep 30
        {2024,  May,       15,  6, Months},
        {2024,  October,   31,  4, Months}, // Oct 31 + 4mo → Feb 28 (2025 non-leap)
        {2024, December,   31,  1, Years},
    };
    std::cout << "  \"add_period\": [\n";
    {
        bool first_local = true;
        for (const auto& c : add_period_cases) {
            if (!first_local) std::cout << ",\n";
            Date dt(c.d, c.m, c.y);
            Date r = dt + Period(c.n, c.u);
            std::cout << "    {\"y\": " << c.y
                      << ", \"m\": " << static_cast<int>(c.m)
                      << ", \"d\": " << c.d
                      << ", \"n\": " << c.n
                      << ", \"units\": \"" << uname(c.u) << "\""
                      << ", \"out_y\": " << r.year()
                      << ", \"out_m\": " << static_cast<int>(r.month())
                      << ", \"out_d\": " << r.dayOfMonth() << "}";
            first_local = false;
        }
        std::cout << "\n  ],\n";
    }

    // --- Date - Date difference -------------------------------------------
    struct DiffCase { Year y1; Month m1; Day d1; Year y2; Month m2; Day d2; };
    const DiffCase diff_cases[] = {
        {2024, January,  1, 2024, January,  1},   // 0
        {2024, January,  2, 2024, January,  1},   // 1
        {2024, January,  1, 2025, January,  1},   // -366 (leap)
        {2025, January,  1, 2024, January,  1},   // +366
        {2024, March,    1, 2024, February, 1},   // 29 (leap year)
        {2023, March,    1, 2023, February, 1},   // 28
    };
    std::cout << "  \"diff\": [\n";
    {
        bool first_local = true;
        for (const auto& c : diff_cases) {
            if (!first_local) std::cout << ",\n";
            Date d1(c.d1, c.m1, c.y1);
            Date d2(c.d2, c.m2, c.y2);
            std::cout << "    {\"y1\": " << c.y1
                      << ", \"m1\": " << static_cast<int>(c.m1)
                      << ", \"d1\": " << c.d1
                      << ", \"y2\": " << c.y2
                      << ", \"m2\": " << static_cast<int>(c.m2)
                      << ", \"d2\": " << c.d2
                      << ", \"diff\": " << (d1 - d2) << "}";
            first_local = false;
        }
        std::cout << "\n  ],\n";
    }

    // --- end_of_month / is_end_of_month ----------------------------------
    struct EOMCase { Year y; Month m; Day d; };
    const EOMCase eom_cases[] = {
        {2024, January,   15},
        {2024, January,   31},
        {2024, February,   1},
        {2024, February,  29},  // leap
        {2023, February,  28},
        {2024, April,     30},
        {2024, April,     29},
        {2024, December,  31},
        {2024, December,   1},
    };
    std::cout << "  \"end_of_month\": [\n";
    {
        bool first_local = true;
        for (const auto& c : eom_cases) {
            if (!first_local) std::cout << ",\n";
            Date dt(c.d, c.m, c.y);
            Date e = Date::endOfMonth(dt);
            std::cout << "    {\"y\": " << c.y
                      << ", \"m\": " << static_cast<int>(c.m)
                      << ", \"d\": " << c.d
                      << ", \"eom_d\": " << e.dayOfMonth()
                      << ", \"is_eom\": " << (Date::isEndOfMonth(dt) ? "true" : "false") << "}";
            first_local = false;
        }
        std::cout << "\n  ],\n";
    }

    // --- start_of_month / is_start_of_month -------------------------------
    std::cout << "  \"start_of_month\": [\n";
    {
        bool first_local = true;
        for (const auto& c : eom_cases) {
            if (!first_local) std::cout << ",\n";
            Date dt(c.d, c.m, c.y);
            Date s = Date::startOfMonth(dt);
            std::cout << "    {\"y\": " << c.y
                      << ", \"m\": " << static_cast<int>(c.m)
                      << ", \"d\": " << c.d
                      << ", \"som_d\": " << s.dayOfMonth()
                      << ", \"is_som\": " << (Date::isStartOfMonth(dt) ? "true" : "false") << "}";
            first_local = false;
        }
        std::cout << "\n  ],\n";
    }

    // --- nextWeekday: starting date + target weekday → next or equal -----
    struct NWCase { Year y; Month m; Day d; Weekday w; };
    const NWCase nw_cases[] = {
        {2024, January,  15, Friday},      // Mon Jan 15 → Fri Jan 19
        {2024, January,  15, Monday},      // same day → same date
        {2024, January,  15, Sunday},      // Mon Jan 15 → Sun Jan 21
        {2024, December, 30, Tuesday},     // Mon Dec 30 → Tue Dec 31
        {2024, December, 31, Wednesday},   // Tue Dec 31 → Wed Jan 1, 2025
    };
    std::cout << "  \"next_weekday\": [\n";
    {
        bool first_local = true;
        for (const auto& c : nw_cases) {
            if (!first_local) std::cout << ",\n";
            Date dt(c.d, c.m, c.y);
            Date r = Date::nextWeekday(dt, c.w);
            std::cout << "    {\"y\": " << c.y
                      << ", \"m\": " << static_cast<int>(c.m)
                      << ", \"d\": " << c.d
                      << ", \"target\": \"" << wname(c.w) << "\""
                      << ", \"out_y\": " << r.year()
                      << ", \"out_m\": " << static_cast<int>(r.month())
                      << ", \"out_d\": " << r.dayOfMonth() << "}";
            first_local = false;
        }
        std::cout << "\n  ],\n";
    }

    // --- nthWeekday: nth weekday of (m, y) ---------------------------------
    struct NTHCase { int n; Weekday w; Month m; Year y; };
    const NTHCase nth_cases[] = {
        {4, Thursday, November, 2024},  // US Thanksgiving 2024 = Nov 28
        {3, Friday,   January,  2024},  // 3rd Friday of Jan 2024 = Jan 19
        {1, Monday,   May,      2024},  // 1st Monday of May 2024 = May 6
        {5, Wednesday, October, 2024},  // 5th Wednesday of Oct 2024 = Oct 30
        {2, Sunday,   February, 2024},  // 2nd Sunday of Feb 2024 = Feb 11
    };
    std::cout << "  \"nth_weekday\": [\n";
    {
        bool first_local = true;
        for (const auto& c : nth_cases) {
            if (!first_local) std::cout << ",\n";
            Date r = Date::nthWeekday(c.n, c.w, c.m, c.y);
            std::cout << "    {\"n\": " << c.n
                      << ", \"w\": \"" << wname(c.w) << "\""
                      << ", \"m\": " << static_cast<int>(c.m)
                      << ", \"y\": " << c.y
                      << ", \"out_d\": " << r.dayOfMonth() << "}";
            first_local = false;
        }
        std::cout << "\n  ]\n";
    }

    std::cout << "}\n";
    return 0;
}

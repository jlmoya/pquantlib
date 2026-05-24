// Emit reference values for the Calendar abstract behavior plus the four
// trivial concretes: NullCalendar, WeekendsOnly, JointCalendar, BespokeCalendar.
//
// We cover:
//   - is_business_day / is_holiday / is_weekend across a representative span
//   - adjust() under every BusinessDayConvention
//   - advance() under Days / Weeks / Months / Years (with EOM flag)
//   - business_days_between (variants of include-first/include-last)
//   - holiday_list across a date range
//   - JointCalendar under both rules (JoinHolidays / JoinBusinessDays)
//   - BespokeCalendar with custom weekend days
//   - calendar.name() for each variant

#include <ql/time/calendar.hpp>
#include <ql/time/calendars/bespokecalendar.hpp>
#include <ql/time/calendars/jointcalendar.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/calendars/weekendsonly.hpp>
#include <ql/time/date.hpp>
#include <ql/time/period.hpp>

#include <iostream>
#include <vector>

using namespace QuantLib;

namespace {

const char* bdc_name(BusinessDayConvention c) {
    switch (c) {
      case Following:                  return "Following";
      case ModifiedFollowing:          return "ModifiedFollowing";
      case Preceding:                  return "Preceding";
      case ModifiedPreceding:          return "ModifiedPreceding";
      case Unadjusted:                 return "Unadjusted";
      case HalfMonthModifiedFollowing: return "HalfMonthModifiedFollowing";
      case Nearest:                    return "Nearest";
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

    // --- names ------------------------------------------------------------
    std::cout << "  \"names\": {\n";
    std::cout << "    \"Null\": \""           << NullCalendar().name() << "\",\n";
    std::cout << "    \"WeekendsOnly\": \""   << WeekendsOnly().name() << "\",\n";
    {
        JointCalendar jc(NullCalendar(), WeekendsOnly(), JoinHolidays);
        std::cout << "    \"JoinHolidays_Null_WeekendsOnly\": \""
                  << jc.name() << "\",\n";
    }
    {
        JointCalendar jc(NullCalendar(), WeekendsOnly(), JoinBusinessDays);
        std::cout << "    \"JoinBusinessDays_Null_WeekendsOnly\": \""
                  << jc.name() << "\",\n";
    }
    std::cout << "    \"Bespoke_named\": \"" << BespokeCalendar("MyBespoke").name() << "\",\n";
    std::cout << "    \"Bespoke_anon\": \""  << BespokeCalendar().name() << "\"\n";
    std::cout << "  },\n";

    // --- per-day classification (a week + a weekend) ---------------------
    //
    // 2024-03-11 (Mon) through 2024-03-17 (Sun)
    NullCalendar null_cal;
    WeekendsOnly weo;

    auto emit_day_table = [&](const char* key, const Calendar& cal) {
        std::cout << "  \"" << key << "\": [\n";
        bool first = true;
        for (int day = 11; day <= 17; ++day) {
            if (!first) std::cout << ",\n";
            Date d(day, March, 2024);
            std::cout << "    {\"d\": " << day
                      << ", \"is_bd\": " << (cal.isBusinessDay(d) ? "true" : "false")
                      << ", \"is_hol\": " << (cal.isHoliday(d) ? "true" : "false")
                      << ", \"is_we\": " << (cal.isWeekend(d.weekday()) ? "true" : "false") << "}";
            first = false;
        }
        std::cout << "\n  ],\n";
    };
    emit_day_table("null_week", null_cal);
    emit_day_table("weekends_only_week", weo);

    // --- JointCalendar: 2024-03-16 (Sat) ---------------------------------
    {
        JointCalendar jh(NullCalendar(), WeekendsOnly(), JoinHolidays);
        JointCalendar jbd(NullCalendar(), WeekendsOnly(), JoinBusinessDays);
        Date sat(16, March, 2024);
        Date mon(11, March, 2024);
        std::cout << "  \"joint_calendar\": {\n";
        std::cout << "    \"sat_join_holidays_is_bd\": "
                  << (jh.isBusinessDay(sat) ? "true" : "false") << ",\n";
        std::cout << "    \"sat_join_business_days_is_bd\": "
                  << (jbd.isBusinessDay(sat) ? "true" : "false") << ",\n";
        std::cout << "    \"mon_join_holidays_is_bd\": "
                  << (jh.isBusinessDay(mon) ? "true" : "false") << ",\n";
        std::cout << "    \"mon_join_business_days_is_bd\": "
                  << (jbd.isBusinessDay(mon) ? "true" : "false") << "\n";
        std::cout << "  },\n";
    }

    // --- BespokeCalendar with Sun + Fri weekends ------------------------
    {
        BespokeCalendar bc("MidEast");
        bc.addWeekend(Friday);
        bc.addWeekend(Sunday);
        std::cout << "  \"bespoke\": {\n";
        std::cout << "    \"fri_is_bd\": "
                  << (bc.isBusinessDay(Date(15, March, 2024)) ? "true" : "false") << ",\n";
        std::cout << "    \"sat_is_bd\": "
                  << (bc.isBusinessDay(Date(16, March, 2024)) ? "true" : "false") << ",\n";
        std::cout << "    \"sun_is_bd\": "
                  << (bc.isBusinessDay(Date(17, March, 2024)) ? "true" : "false") << "\n";
        std::cout << "  },\n";
    }

    // --- adjust() under every BDC, with WeekendsOnly cal -----------------
    {
        // 2024-03-16 is a Saturday in WeekendsOnly → holiday → tests adjust.
        Date sat(16, March, 2024);
        std::vector<BusinessDayConvention> bdcs = {
            Following, ModifiedFollowing, Preceding, ModifiedPreceding,
            Unadjusted, HalfMonthModifiedFollowing, Nearest
        };
        std::cout << "  \"adjust\": [\n";
        bool first = true;
        for (auto c : bdcs) {
            if (!first) std::cout << ",\n";
            Date r = weo.adjust(sat, c);
            std::cout << "    {\"input\": \"2024-03-16\", \"convention\": \""
                      << bdc_name(c) << "\""
                      << ", \"out_y\": " << r.year()
                      << ", \"out_m\": " << static_cast<int>(r.month())
                      << ", \"out_d\": " << r.dayOfMonth() << "}";
            first = false;
        }
        std::cout << "\n  ],\n";

        // ModifiedFollowing crossing month boundary: 2024-03-30 (Sat) →
        // Following gives Apr 1 (different month) → MF goes back to Mar 29 (Fri).
        std::cout << "  \"adjust_mod_following_month_cross\": {\n";
        Date d(30, March, 2024);
        Date r = weo.adjust(d, ModifiedFollowing);
        std::cout << "    \"out_y\": " << r.year()
                  << ", \"out_m\": " << static_cast<int>(r.month())
                  << ", \"out_d\": " << r.dayOfMonth() << "\n";
        std::cout << "  },\n";
    }

    // --- advance() with various units + BDC + endOfMonth -----------------
    {
        std::cout << "  \"advance\": [\n";
        struct AC { Year y; Month m; Day d; int n; TimeUnit u; BusinessDayConvention c; bool eom; };
        const AC cases[] = {
            {2024, March, 15, 5, Days, Following, false},
            {2024, March, 15, -3, Days, Following, false},
            {2024, March, 11, 1, Weeks, Following, false},   // Mon + 1W = Mon
            {2024, March, 15, 1, Months, Following, false},  // Fri Mar 15 + 1M = Mon Apr 15
            {2024, March, 29, 1, Months, Following, true},   // last business day of Mar (Fri 29) + 1M, EOM → last business day of Apr
            {2024, January, 31, 1, Months, ModifiedFollowing, false}, // Jan 31 + 1M → Feb 29
            {2024, February, 29, 1, Years, Following, false}, // Feb 29 + 1Y → Feb 28 2025
        };
        bool first = true;
        for (const auto& ac : cases) {
            if (!first) std::cout << ",\n";
            Date dt(ac.d, ac.m, ac.y);
            Date r = weo.advance(dt, ac.n, ac.u, ac.c, ac.eom);
            std::cout << "    {\"y\": " << ac.y
                      << ", \"m\": " << static_cast<int>(ac.m)
                      << ", \"d\": " << ac.d
                      << ", \"n\": " << ac.n
                      << ", \"units\": \"" << uname(ac.u) << "\""
                      << ", \"convention\": \"" << bdc_name(ac.c) << "\""
                      << ", \"end_of_month\": " << (ac.eom ? "true" : "false")
                      << ", \"out_y\": " << r.year()
                      << ", \"out_m\": " << static_cast<int>(r.month())
                      << ", \"out_d\": " << r.dayOfMonth() << "}";
            first = false;
        }
        std::cout << "\n  ],\n";
    }

    // --- business_days_between under WeekendsOnly -----------------------
    {
        Date d1(11, March, 2024);  // Mon
        Date d2(18, March, 2024);  // next Mon (7 cal days, 5 bdays exclusive)
        std::cout << "  \"business_days_between\": {\n";
        std::cout << "    \"mon_to_next_mon_incl_first_excl_last\": "
                  << weo.businessDaysBetween(d1, d2, true, false) << ",\n";
        std::cout << "    \"mon_to_next_mon_incl_first_incl_last\": "
                  << weo.businessDaysBetween(d1, d2, true, true) << ",\n";
        std::cout << "    \"mon_to_next_mon_excl_first_excl_last\": "
                  << weo.businessDaysBetween(d1, d2, false, false) << ",\n";
        std::cout << "    \"reversed_direction\": "
                  << weo.businessDaysBetween(d2, d1, true, false) << "\n";
        std::cout << "  },\n";
    }

    // --- holiday_list for WeekendsOnly over 2024-03-11..2024-03-24 ------
    {
        std::cout << "  \"holiday_list_weo\": [";
        Date a(11, March, 2024), b(24, March, 2024);
        std::vector<Date> hs = weo.holidayList(a, b, true);
        bool first = true;
        for (const auto& d : hs) {
            if (!first) std::cout << ", ";
            std::cout << d.dayOfMonth();
            first = false;
        }
        std::cout << "]\n";
    }

    std::cout << "}\n";
    return 0;
}

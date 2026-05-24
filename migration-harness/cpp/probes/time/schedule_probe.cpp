// Emit Schedule reference values across all DateGeneration rules.
//
// Each entry emits the generated date list (as serial numbers and as
// (y, m, d) tuples) plus the is_regular flag list when present. Cross-
// validation in Python checks identical date sequences and equal
// is_regular bitmaps.

#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/calendars/weekendsonly.hpp>
#include <ql/time/dategenerationrule.hpp>
#include <ql/time/date.hpp>
#include <ql/time/period.hpp>
#include <ql/time/schedule.hpp>

#include <iostream>
#include <string>
#include <vector>

using namespace QuantLib;

namespace {

void emit_dates(const char* key, const Schedule& s, bool trailing_comma) {
    std::cout << "    \"" << key << "\": [";
    for (size_t i = 0; i < s.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << s[i].serialNumber();
    }
    std::cout << "]" << (trailing_comma ? "," : "") << "\n";
}

void emit_is_regular(const Schedule& s, bool trailing_comma) {
    std::cout << "    \"is_regular\": [";
    if (s.hasIsRegular()) {
        const auto& v = s.isRegular();
        for (size_t i = 0; i < v.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << (v[i] ? "true" : "false");
        }
    }
    std::cout << "]" << (trailing_comma ? "," : "") << "\n";
}

}  // namespace

int main() {
    std::cout << "{\n";

    // Common inputs
    Date eff(15, March, 2024);
    Date term(15, March, 2026);  // 2 years
    Period tenor(6, Months);     // semi-annual
    WeekendsOnly cal;

    // --- Rule: Zero ------------------------------------------------------
    std::cout << "  \"zero\": {\n";
    {
        Schedule s(eff, term, tenor, cal, Following, Following,
                   DateGeneration::Zero, false);
        emit_dates("dates", s, true);
        emit_is_regular(s, false);
    }
    std::cout << "  },\n";

    // --- Rule: Backward (default for MakeSchedule) -----------------------
    std::cout << "  \"backward\": {\n";
    {
        Schedule s(eff, term, tenor, cal, Following, Following,
                   DateGeneration::Backward, false);
        emit_dates("dates", s, true);
        emit_is_regular(s, false);
    }
    std::cout << "  },\n";

    // --- Rule: Forward ---------------------------------------------------
    std::cout << "  \"forward\": {\n";
    {
        Schedule s(eff, term, tenor, cal, Following, Following,
                   DateGeneration::Forward, false);
        emit_dates("dates", s, true);
        emit_is_regular(s, false);
    }
    std::cout << "  },\n";

    // --- Rule: ThirdWednesday --------------------------------------------
    std::cout << "  \"third_wednesday\": {\n";
    {
        Date e2(20, March, 2024);  // 3rd Wed Mar 2024
        Date t2(17, December, 2025);  // 3rd Wed Dec 2025
        Schedule s(e2, t2, Period(3, Months), cal, Following, Following,
                   DateGeneration::ThirdWednesday, false);
        emit_dates("dates", s, true);
        emit_is_regular(s, false);
    }
    std::cout << "  },\n";

    // --- Rule: Twentieth -------------------------------------------------
    std::cout << "  \"twentieth\": {\n";
    {
        Schedule s(eff, term, Period(3, Months), cal, Following, Following,
                   DateGeneration::Twentieth, false);
        emit_dates("dates", s, true);
        emit_is_regular(s, false);
    }
    std::cout << "  },\n";

    // --- Rule: CDS2015 --------------------------------------------------
    std::cout << "  \"cds2015\": {\n";
    {
        Schedule s(eff, term, Period(3, Months), cal, Following, Following,
                   DateGeneration::CDS2015, false);
        emit_dates("dates", s, true);
        emit_is_regular(s, false);
    }
    std::cout << "  },\n";

    // --- Truncation: after() and until() ---------------------------------
    {
        Schedule full(eff, term, tenor, cal, Following, Following,
                      DateGeneration::Backward, false);
        Date trunc(15, September, 2024);
        Schedule after_t = full.after(trunc);
        Schedule until_t = full.until(trunc);
        std::cout << "  \"after\": {\n";
        emit_dates("dates", after_t, false);
        std::cout << "  },\n";
        std::cout << "  \"until\": {\n";
        emit_dates("dates", until_t, false);
        std::cout << "  },\n";
    }

    // --- previous_date / next_date ---------------------------------------
    {
        Schedule s(eff, term, tenor, cal, Following, Following,
                   DateGeneration::Backward, false);
        Date probe(1, July, 2024);  // between Mar and Sep dates
        Date prev = s.previousDate(probe);
        Date next = s.nextDate(probe);
        std::cout << "  \"prev_next\": {\n";
        std::cout << "    \"probe_serial\": " << probe.serialNumber() << ",\n";
        std::cout << "    \"prev_serial\": " << prev.serialNumber() << ",\n";
        std::cout << "    \"next_serial\": " << next.serialNumber() << "\n";
        std::cout << "  },\n";
    }

    // --- backward with first_date ---------------------------------------
    std::cout << "  \"backward_with_first_date\": {\n";
    {
        Date first(15, April, 2024);  // irregular first period
        Schedule s(eff, term, tenor, cal, Following, Following,
                   DateGeneration::Backward, false, first);
        emit_dates("dates", s, true);
        emit_is_regular(s, false);
    }
    std::cout << "  },\n";

    // --- date-list ctor (no rule, just dates) ----------------------------
    std::cout << "  \"date_list_ctor\": {\n";
    {
        std::vector<Date> dates = {
            Date(15, March,     2024),
            Date(17, September, 2024),  // Mon
            Date(17, March,     2025),  // Mon
            Date(15, September, 2025),  // Mon
            Date(16, March,     2026),  // Mon
        };
        Schedule s(dates);  // no calendar/conv/rule
        emit_dates("dates", s, false);
    }
    std::cout << "  }\n";

    std::cout << "}\n";
    return 0;
}

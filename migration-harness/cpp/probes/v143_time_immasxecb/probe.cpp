// v1.43 reference for the IMM / ASX / ECB API surface that the earlier
// pquantlib port left out.
//
// The pre-existing references (time/imm.json, time/asx/ecb.json) cover the
// Date-argument entry points. They do NOT cover:
//
//   * IMM::nextDate(const std::string&, bool, const Date&)   imm.cpp:189
//   * IMM::nextCode(const std::string&, bool, const Date&)   imm.cpp:202
//   * IMM::nextCode(const Date&, bool)                       imm.cpp:196
//   * ASX::nextDate(const std::string&, bool, const Date&)   asx.cpp:144
//   * ASX::nextCode(const std::string&, bool, const Date&)   asx.cpp:157
//   * ASX::nextCode(const Date&, bool)                       asx.cpp:151
//   * ECB::isECBdate(const Date&)                            ecb.hpp:81
//   * ECB::nextDate(const std::string&, const Date&)         ecb.hpp:65
//   * ECB::nextDates(const std::string&, const Date&)        ecb.hpp:74
//   * ECB::nextCode(const Date&)                             ecb.hpp:87
//   * IMM::Month / ASX::Month enumerators                    imm.hpp:36
//
// Output shape (all dates are Date serial numbers):
//   {
//     "imm_month_enum": {"F": 1, ...},
//     "asx_month_enum": {"F": 1, ...},
//     "imm_next_date_from_code": [{"code":..,"main_cycle":..,"ref":..,"out":..}, ...],
//     "imm_next_code_from_code": [{"code":..,"main_cycle":..,"ref":..,"out":".."}, ...],
//     "imm_next_code_from_date": [{"d":..,"main_cycle":..,"out":".."}, ...],
//     "asx_next_date_from_code": [...], "asx_next_code_from_code": [...],
//     "asx_next_code_from_date": [...],
//     "ecb_is_ecb_date_true":    [<serial>, ...],   // every d in [38300, 45644]
//     "ecb_next_date_from_code": [{"code":"..","ref":..,"out":..}, ...],
//     "ecb_next_dates_from_code":[{"code":"..","ref":..,"out":[..]}, ...],
//     "ecb_next_code_from_date": [{"d":..,"out":".."}, ...]
//   }

#include <ql/time/asx.hpp>
#include <ql/time/date.hpp>
#include <ql/time/ecb.hpp>
#include <ql/time/imm.hpp>

#include <iostream>
#include <string>
#include <vector>

using namespace QuantLib;

namespace {

const std::vector<std::string> kFuturesCodes = {"F", "G", "H", "J", "K", "M",
                                                "N", "Q", "U", "V", "X", "Z"};
const std::vector<int> kFuturesValues = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

// A spread of reference dates: two decades apart, start/middle/end of year,
// and one that straddles the C++ decade-rollover branch in IMM::date.
const std::vector<Date> kRefDates = {
    Date(1, January, 2020),
    Date(15, July, 2015),
    Date(30, December, 2009),
    Date(1, March, 1999),
};

// Two-character futures codes: letter + last digit of the year.
std::vector<std::string> futuresCodes() {
    std::vector<std::string> out;
    for (const auto& letter : kFuturesCodes)
        for (int digit = 0; digit <= 9; digit += 3)
            out.push_back(letter + std::to_string(digit));
    return out;
}

void emitMonthEnum(const char* key) {
    std::cout << "  \"" << key << "\": {";
    for (std::size_t i = 0; i < kFuturesCodes.size(); ++i) {
        if (i != 0) std::cout << ", ";
        std::cout << "\"" << kFuturesCodes[i] << "\": " << kFuturesValues[i];
    }
    std::cout << "},\n";
}

}  // namespace

int main() {
    std::cout << "{\n";

    // --- IMM::Month / ASX::Month -----------------------------------------
    emitMonthEnum("imm_month_enum");
    emitMonthEnum("asx_month_enum");

    const auto codes = futuresCodes();

    // --- IMM::nextDate(code, mainCycle, ref) -----------------------------
    std::cout << "  \"imm_next_date_from_code\": [\n";
    bool first = true;
    for (const auto& c : codes)
        for (const auto& ref : kRefDates)
            for (int mc = 0; mc <= 1; ++mc) {
                if (!first) std::cout << ",\n";
                first = false;
                std::cout << "    {\"code\": \"" << c << "\", \"main_cycle\": "
                          << (mc != 0 ? "true" : "false")
                          << ", \"ref\": " << ref.serialNumber() << ", \"out\": "
                          << IMM::nextDate(c, mc != 0, ref).serialNumber() << "}";
            }
    std::cout << "\n  ],\n";

    // --- IMM::nextCode(code, mainCycle, ref) -----------------------------
    std::cout << "  \"imm_next_code_from_code\": [\n";
    first = true;
    for (const auto& c : codes)
        for (const auto& ref : kRefDates)
            for (int mc = 0; mc <= 1; ++mc) {
                if (!first) std::cout << ",\n";
                first = false;
                std::cout << "    {\"code\": \"" << c << "\", \"main_cycle\": "
                          << (mc != 0 ? "true" : "false")
                          << ", \"ref\": " << ref.serialNumber() << ", \"out\": \""
                          << IMM::nextCode(c, mc != 0, ref) << "\"}";
            }
    std::cout << "\n  ],\n";

    // --- IMM::nextCode(date, mainCycle) ----------------------------------
    std::cout << "  \"imm_next_code_from_date\": [\n";
    first = true;
    for (Date d(1, January, 2015); d <= Date(31, December, 2016); d += 17)
        for (int mc = 0; mc <= 1; ++mc) {
            if (!first) std::cout << ",\n";
            first = false;
            std::cout << "    {\"d\": " << d.serialNumber() << ", \"main_cycle\": "
                      << (mc != 0 ? "true" : "false") << ", \"out\": \""
                      << IMM::nextCode(d, mc != 0) << "\"}";
        }
    std::cout << "\n  ],\n";

    // --- ASX::nextDate(code, mainCycle, ref) -----------------------------
    std::cout << "  \"asx_next_date_from_code\": [\n";
    first = true;
    for (const auto& c : codes)
        for (const auto& ref : kRefDates)
            for (int mc = 0; mc <= 1; ++mc) {
                if (!first) std::cout << ",\n";
                first = false;
                std::cout << "    {\"code\": \"" << c << "\", \"main_cycle\": "
                          << (mc != 0 ? "true" : "false")
                          << ", \"ref\": " << ref.serialNumber() << ", \"out\": "
                          << ASX::nextDate(c, mc != 0, ref).serialNumber() << "}";
            }
    std::cout << "\n  ],\n";

    // --- ASX::nextCode(code, mainCycle, ref) -----------------------------
    std::cout << "  \"asx_next_code_from_code\": [\n";
    first = true;
    for (const auto& c : codes)
        for (const auto& ref : kRefDates)
            for (int mc = 0; mc <= 1; ++mc) {
                if (!first) std::cout << ",\n";
                first = false;
                std::cout << "    {\"code\": \"" << c << "\", \"main_cycle\": "
                          << (mc != 0 ? "true" : "false")
                          << ", \"ref\": " << ref.serialNumber() << ", \"out\": \""
                          << ASX::nextCode(c, mc != 0, ref) << "\"}";
            }
    std::cout << "\n  ],\n";

    // --- ASX::nextCode(date, mainCycle) ----------------------------------
    std::cout << "  \"asx_next_code_from_date\": [\n";
    first = true;
    for (Date d(1, January, 2015); d <= Date(31, December, 2016); d += 17)
        for (int mc = 0; mc <= 1; ++mc) {
            if (!first) std::cout << ",\n";
            first = false;
            std::cout << "    {\"d\": " << d.serialNumber() << ", \"main_cycle\": "
                      << (mc != 0 ? "true" : "false") << ", \"out\": \""
                      << ASX::nextCode(d, mc != 0) << "\"}";
        }
    std::cout << "\n  ],\n";

    // --- ECB::isECBdate ---------------------------------------------------
    // Upper bound 45644 is the last entry of ecbKnownDateSet (ecb.cpp:155);
    // isECBdate(d) evaluates nextDate(d-1), which QL_FAILs once d-1 reaches
    // that last date, so 45644 is the largest queryable date.
    std::cout << "  \"ecb_is_ecb_date_range\": [38300, 45644],\n";
    std::cout << "  \"ecb_is_ecb_date_true\": [";
    first = true;
    for (Date::serial_type s = 38300; s <= 45644; ++s) {
        if (ECB::isECBdate(Date(s))) {
            if (!first) std::cout << ",";
            first = false;
            std::cout << s;
        }
    }
    std::cout << "],\n";

    // --- ECB::nextDate(code, ref) / nextDates(code, ref) / nextCode(date) --
    const std::vector<std::pair<std::string, Date>> ecbCases = {
        {"JAN10", Date(1, January, 2010)},
        {"MAR08", Date(15, June, 2008)},
        {"JUN15", Date(3, March, 2015)},
        {"SEP20", Date(31, December, 2020)},
        {"DEC13", Date(1, July, 2013)},
        {"FEB06", Date(20, November, 2006)},
    };

    std::cout << "  \"ecb_next_date_from_code\": [\n";
    first = true;
    for (const auto& [c, ref] : ecbCases) {
        if (!first) std::cout << ",\n";
        first = false;
        std::cout << "    {\"code\": \"" << c << "\", \"ref\": " << ref.serialNumber()
                  << ", \"out\": " << ECB::nextDate(c, ref).serialNumber() << "}";
    }
    std::cout << "\n  ],\n";

    std::cout << "  \"ecb_next_dates_from_code\": [\n";
    first = true;
    for (const auto& [c, ref] : ecbCases) {
        if (!first) std::cout << ",\n";
        first = false;
        std::cout << "    {\"code\": \"" << c << "\", \"ref\": " << ref.serialNumber()
                  << ", \"out\": [";
        bool inner = true;
        for (const auto& d : ECB::nextDates(c, ref)) {
            if (!inner) std::cout << ",";
            inner = false;
            std::cout << d.serialNumber();
        }
        std::cout << "]}";
    }
    std::cout << "\n  ],\n";

    std::cout << "  \"ecb_next_code_from_date\": [\n";
    first = true;
    for (Date d(1, January, 2010); d <= Date(31, December, 2012); d += 29) {
        if (!first) std::cout << ",\n";
        first = false;
        std::cout << "    {\"d\": " << d.serialNumber() << ", \"out\": \""
                  << ECB::nextCode(d) << "\"}";
    }
    std::cout << "\n  ]\n";

    std::cout << "}\n";
    return 0;
}

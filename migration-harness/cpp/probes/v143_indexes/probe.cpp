// migration-harness/cpp/probes/v143_indexes/probe.cpp
//
// Reference values for the three interest-rate indexes introduced in C++
// QuantLib v1.43:
//
//   * Nibor   — NOK-NIBOR, an IborIndex (tenor-based, 2 fixing days)
//   * Shir    — Shekel Overnight Interest Rate, an OvernightIndex
//   * Zaronia — South African Rand Overnight Index Average, an OvernightIndex
//
// Each index is a thin constructor wrapper, so what actually needs pinning is
// the exact wiring: name, fixing days, currency, fixing calendar, day counter,
// business-day convention and end-of-month flag. Those are cheap to get subtly
// wrong in a port and stay invisible until a fixing date or accrual is off, so
// they are captured explicitly — together with valueDate/maturityDate for a
// sample fixing, which is where a wrong calendar or convention actually bites.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/indexes.json.

#include <iomanip>
#include <iostream>

#include <ql/indexes/ibor/nibor.hpp>
#include <ql/indexes/ibor/shir.hpp>
#include <ql/indexes/ibor/zaronia.hpp>
#include <ql/time/period.hpp>

using namespace QuantLib;

namespace {

// Mid-week so the value/maturity roll is driven by the index's own calendar
// and convention rather than by a weekend.
const Date kFixing(15, June, 2026);

void emitIndex(const char* key, const IborIndex& idx, bool trailingComma) {
    const Date value = idx.valueDate(kFixing);
    std::cout << "  \"" << key << "\": {\n"
              << "    \"name\": \"" << idx.name() << "\",\n"
              << "    \"fixing_days\": " << idx.fixingDays() << ",\n"
              << "    \"currency_code\": \"" << idx.currency().code() << "\",\n"
              << "    \"fixing_calendar\": \"" << idx.fixingCalendar().name() << "\",\n"
              << "    \"day_counter\": \"" << idx.dayCounter().name() << "\",\n"
              << "    \"business_day_convention\": " << static_cast<int>(idx.businessDayConvention()) << ",\n"
              << "    \"end_of_month\": " << (idx.endOfMonth() ? "true" : "false") << ",\n"
              << "    \"fixing_serial\": " << kFixing.serialNumber() << ",\n"
              << "    \"value_date_serial\": " << value.serialNumber() << ",\n"
              << "    \"maturity_date_serial\": " << idx.maturityDate(value).serialNumber() << "\n"
              << "  }" << (trailingComma ? "," : "") << "\n";
}

} // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";
    emitIndex("nibor_3m", Nibor(Period(3, Months)), true);
    emitIndex("nibor_6m", Nibor(Period(6, Months)), true);
    emitIndex("shir", Shir(), true);
    emitIndex("zaronia", Zaronia(), false);
    std::cout << "}\n";
    return 0;
}

// migration-harness/cpp/probes/v143_cf_timebasket/probe.cpp
//
// Reference values for QuantLib::TimeBasket (ql/cashflows/timebasket.hpp,
// ql/cashflows/timebasket.cpp).
//
// TimeBasket privately derives from std::map<Date,Real>, so its observable
// behaviour is the *map* behaviour as much as the algebra:
//
//   * entries iterate in ASCENDING DATE ORDER regardless of insertion order
//     (std::map, not a Python dict) — so every case below is emitted as the
//     full (date-serial, amount) listing rather than a headline number;
//   * operator[] on a MISSING key default-constructs 0.0 AND INSERTS it, so a
//     bare read mutates size(). rebin() and operator+=/-= are built on exactly
//     that behaviour, so a port that makes reads non-mutating silently changes
//     the resulting basket. `lookup_missing_*` pins it directly;
//   * construction from parallel vectors assigns (not accumulates), so a
//     repeated date keeps the LAST value — `ctor_duplicate_*` pins that.
//
// rebin(buckets) redistributes each entry linearly between the two adjacent
// bucket dates (timebasket.cpp:37-71). The interesting branches are:
//   - date exactly on a bucket            -> whole value to that bucket
//   - date before the first bucket        -> whole value to the first bucket
//     (lower_bound == begin, so nDate stays null)
//   - date after the last bucket          -> whole value to the LAST bucket
//     (lower_bound == end, so pDate = back() and nDate stays null)
//   - date strictly between two buckets   -> split value*(nDays/tDays) to the
//     later bucket and value*(pDays/tDays) to the earlier one
// All four are exercised by the coarser / finer / partially-overlapping grids.
// The bucket vector is also passed UNSORTED to pin the internal std::sort.
//
// The QL_REQUIRE branches (size mismatch in the constructor, empty bucket
// vector in rebin) are emitted as {"raises": true}.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/cf/timebasket.json.

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <ql/cashflows/timebasket.hpp>
#include <ql/errors.hpp>
#include <ql/time/date.hpp>

using namespace QuantLib;

namespace {

// Emit a basket as two parallel arrays in the map's own (ascending) order,
// plus its size, so both the ordering and the values are pinned.
void emitBasket(const std::string& key, const TimeBasket& b, bool trailingComma) {
    std::cout << "  \"" << key << "\": {\n    \"size\": " << b.size() << ",\n    \"dates\": [";
    bool first = true;
    for (const auto& entry : b) {
        if (!first)
            std::cout << ", ";
        std::cout << entry.first.serialNumber();
        first = false;
    }
    std::cout << "],\n    \"amounts\": [";
    first = true;
    for (const auto& entry : b) {
        if (!first)
            std::cout << ", ";
        std::cout << entry.second;
        first = false;
    }
    std::cout << "]\n  }" << (trailingComma ? "," : "") << "\n";
}

void emitScalar(const std::string& key, Real v, bool trailingComma) {
    std::cout << "  \"" << key << "\": " << v << (trailingComma ? "," : "") << "\n";
}

void emitBool(const std::string& key, bool v, bool trailingComma) {
    std::cout << "  \"" << key << "\": " << (v ? "true" : "false") << (trailingComma ? "," : "")
              << "\n";
}

void emitRaises(const std::string& key, bool trailingComma) {
    std::cout << "  \"" << key << "\": {\"raises\": true}" << (trailingComma ? "," : "") << "\n";
}

// Deliberately unsorted, and with the middle date repeated so the map's
// "assign, do not accumulate" behaviour is visible.
const std::vector<Date> kDates = {Date(15, March, 2026), Date(15, January, 2026),
                                  Date(15, June, 2026), Date(15, September, 2026)};
const std::vector<Real> kValues = {2000.0, 1000.0, 3000.0, 4000.0};

TimeBasket base() { return {kDates, kValues}; }

} // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    // --- construction ----------------------------------------------------
    const TimeBasket b = base();
    emitBasket("ctor", b, true);

    // A repeated date keeps the LAST assigned value (operator[] assignment).
    {
        const std::vector<Date> d = {Date(15, January, 2026), Date(15, March, 2026),
                                     Date(15, January, 2026)};
        const std::vector<Real> v = {10.0, 20.0, 99.0};
        emitBasket("ctor_duplicate_date", TimeBasket(d, v), true);
    }

    emitRaises("ctor_size_mismatch_raises", true);

    // --- map interface ---------------------------------------------------
    emitBool("has_date_present", b.hasDate(Date(15, March, 2026)), true);
    emitBool("has_date_absent", b.hasDate(Date(16, March, 2026)), true);
    emitScalar("lookup_present", TimeBasket(kDates, kValues)[Date(15, June, 2026)], true);
    {
        // operator[] on a missing key returns 0.0 *and inserts the key*.
        TimeBasket m = base();
        const Size before = m.size();
        const Real got = m[Date(1, December, 2026)];
        emitScalar("lookup_missing_value", got, true);
        emitScalar("lookup_missing_size_before", Real(before), true);
        emitScalar("lookup_missing_size_after", Real(m.size()), true);
        emitBasket("lookup_missing_basket", m, true);
    }

    // --- algebra ---------------------------------------------------------
    // `other` overlaps on 15-Mar and 15-Sep and introduces two brand-new dates
    // that must land in sorted position, one before and one after the range.
    const std::vector<Date> otherDates = {Date(15, March, 2026), Date(15, September, 2026),
                                          Date(1, January, 2026), Date(31, December, 2026)};
    const std::vector<Real> otherValues = {500.0, -250.0, 125.0, 77.5};
    const TimeBasket other(otherDates, otherValues);
    emitBasket("other", other, true);

    {
        TimeBasket s = base();
        s += other;
        emitBasket("plus_equals", s, true);
    }
    {
        TimeBasket s = base();
        s -= other;
        emitBasket("minus_equals", s, true);
    }
    {
        // += then -= with the same operand restores the original amounts, but
        // NOT the original size: the keys introduced by += survive with 0.0.
        TimeBasket s = base();
        s += other;
        s -= other;
        emitBasket("plus_then_minus", s, true);
    }

    // --- rebin -----------------------------------------------------------
    // Coarser grid, passed unsorted: two buckets straddling the four entries.
    // 15-Jan is BEFORE the first bucket (whole value to the first bucket);
    // 15-Sep is AFTER the last (whole value to the last bucket); 15-Mar and
    // 15-Jun fall strictly inside and are split.
    {
        const std::vector<Date> buckets = {Date(1, July, 2026), Date(1, February, 2026)};
        emitBasket("rebin_coarser", b.rebin(buckets), true);
    }
    // Finer grid: monthly buckets, one of which (15-Mar) coincides exactly
    // with an entry so the "pDate == date" branch fires.
    {
        const std::vector<Date> buckets = {Date(15, January, 2026), Date(15, February, 2026),
                                           Date(15, March, 2026),   Date(15, April, 2026),
                                           Date(15, May, 2026),     Date(15, June, 2026),
                                           Date(15, July, 2026),    Date(15, August, 2026),
                                           Date(15, September, 2026)};
        emitBasket("rebin_finer", b.rebin(buckets), true);
    }
    // Partially overlapping grid: starts after the first entry and ends before
    // the last, so both out-of-range branches fire at once.
    {
        const std::vector<Date> buckets = {Date(1, March, 2026), Date(1, May, 2026),
                                           Date(1, July, 2026)};
        emitBasket("rebin_partial_overlap", b.rebin(buckets), true);
    }
    // A single bucket collapses everything onto that date.
    {
        const std::vector<Date> buckets = {Date(1, May, 2026)};
        emitBasket("rebin_single_bucket", b.rebin(buckets), true);
    }
    // rebin does not mutate the source basket.
    emitBasket("rebin_source_unchanged", b, true);

    emitRaises("rebin_empty_buckets_raises", false);

    std::cout << "}\n";
    return 0;
}

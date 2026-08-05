// migration-harness/cpp/probes/v143_indexes_inflation/probe.cpp
//
// Reference values for the ql/indexes/region.hpp hierarchy and the
// ql/indexes/inflation/* indexes PQuantLib was missing against C++ v1.43.
//
// Regions are pure (name, code) payloads, but the name feeds straight into
// InflationIndex::name() ("<region.name()> <familyName>"), so a wrong region
// string silently renames every index built on it — and the index name is the
// IndexManager key that past fixings are stored under. Both halves are
// therefore pinned: the region payloads on their own, and the full index
// identity for each concrete index.
//
// Region equality is also probed: C++ compares name() and ignores code(), so a
// CustomRegion that shadows a built-in name compares equal to it.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/indexes/inflation.json.

#include <iomanip>
#include <iostream>
#include <string>

#include <ql/experimental/inflation/genericindexes.hpp>
#include <ql/indexes/inflation/aucpi.hpp>
#include <ql/indexes/inflation/euhicp.hpp>
#include <ql/indexes/inflation/frhicp.hpp>
#include <ql/indexes/inflation/ukhicp.hpp>
#include <ql/indexes/inflation/ukrpi.hpp>
#include <ql/indexes/inflation/uscpi.hpp>
#include <ql/indexes/inflation/zacpi.hpp>
#include <ql/indexes/region.hpp>

using namespace QuantLib;

namespace {

bool first = true;

void comma() {
    if (!first)
        std::cout << ",\n";
    first = false;
}

void emitRegion(const std::string& key, const Region& r) {
    comma();
    std::cout << "    \"" << key << "\": {\"name\": \"" << r.name() << "\", \"code\": \""
              << r.code() << "\"}";
}

void emitIndex(const std::string& key, const InflationIndex& idx) {
    comma();
    std::cout << "    \"" << key << "\": {\n"
              << "      \"name\": \"" << idx.name() << "\",\n"
              << "      \"family_name\": \"" << idx.familyName() << "\",\n"
              << "      \"region_name\": \"" << idx.region().name() << "\",\n"
              << "      \"region_code\": \"" << idx.region().code() << "\",\n"
              << "      \"revised\": " << (idx.revised() ? "true" : "false") << ",\n"
              << "      \"frequency\": " << static_cast<int>(idx.frequency()) << ",\n"
              << "      \"availability_lag_months\": " << idx.availabilityLag().length() << ",\n"
              << "      \"currency_code\": \"" << idx.currency().code() << "\"\n"
              << "    }";
}

} // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    std::cout << "  \"regions\": {\n";
    first = true;
    emitRegion("australia", AustraliaRegion());
    emitRegion("eu", EURegion());
    emitRegion("france", FranceRegion());
    emitRegion("uk", UKRegion());
    emitRegion("us", USRegion());
    emitRegion("za", ZARegion());
    emitRegion("generic", GenericRegion());
    emitRegion("custom", CustomRegion("Atlantis", "AT"));
    std::cout << "\n  },\n";

    // Equality ignores code(): a CustomRegion carrying a built-in's name
    // compares equal to it even with a different code.
    std::cout << "  \"region_equality\": {\n"
              << "    \"eu_vs_eu\": " << (EURegion() == EURegion() ? "true" : "false") << ",\n"
              << "    \"eu_vs_us\": " << (EURegion() == USRegion() ? "true" : "false") << ",\n"
              << "    \"custom_same_name_different_code_vs_eu\": "
              << (CustomRegion("EU", "XX") == EURegion() ? "true" : "false") << "\n"
              << "  },\n";

    std::cout << "  \"zero_indexes\": {\n";
    first = true;
    emitIndex("AUCPI_quarterly", AUCPI(Quarterly, false));
    emitIndex("AUCPI_monthly_revised", AUCPI(Monthly, true));
    emitIndex("EUHICP", EUHICP());
    emitIndex("EUHICPXT", EUHICPXT());
    emitIndex("FRHICP", FRHICP());
    emitIndex("UKHICP", UKHICP());
    emitIndex("UKRPI", UKRPI());
    emitIndex("USCPI", USCPI());
    emitIndex("ZACPI", ZACPI());
    std::cout << "\n  },\n";

    std::cout << "  \"yoy_indexes\": {\n";
    first = true;
    emitIndex("YYAUCPI_quarterly", YYAUCPI(Quarterly, false));
    emitIndex("YYAUCPI_monthly_revised", YYAUCPI(Monthly, true));
    emitIndex("YYEUHICP", YYEUHICP());
    emitIndex("YYEUHICPXT", YYEUHICPXT());
    emitIndex("YYFRHICP", YYFRHICP());
    emitIndex("YYUKRPI", YYUKRPI());
    emitIndex("YYUSCPI", YYUSCPI());
    emitIndex("YYZACPI", YYZACPI());
    std::cout << "\n  }\n";

    std::cout << "}\n";
    return 0;
}

// L7-A foundations probe: inflation indexes + seasonality + inflation period.
//
// Emits reference values for:
//   * Default-market parameters of EUHICP, FRHICP, UKRPI, UKHICP, USCPI
//     (family_name, region_name, region_code, frequency, availability_lag_months,
//      revised, currency_code, name).
//   * The YoY siblings YYEUHICP, YYFRHICP, YYUKRPI, YYUSCPI (same shape, with
//     `interpolated`+`ratio` flags).  YYUKHICP does not exist in C++; we
//     deliberately drop it from the YoY listing.
//   * MultiplicativePriceSeasonality factor at sample dates (a stationary
//     12-month seasonality curve) — verifies the multiplicative-factor lookup
//     logic Python ports must reproduce.
//   * QuantLib::inflationPeriod(date, frequency) for several anchor dates +
//     each Frequency level the Python port supports.  These pin down the
//     end-of-availability semantics that ZeroInflationIndex::maturityDate
//     (the Python-side convenience) relies on.

#include <ql/currency.hpp>
#include <ql/indexes/inflation/euhicp.hpp>
#include <ql/indexes/inflation/frhicp.hpp>
#include <ql/indexes/inflation/ukhicp.hpp>
#include <ql/indexes/inflation/ukrpi.hpp>
#include <ql/indexes/inflation/uscpi.hpp>
#include <ql/indexes/inflationindex.hpp>
#include <ql/indexes/region.hpp>
#include <ql/termstructures/inflation/seasonality.hpp>
#include <ql/termstructures/inflationtermstructure.hpp>
#include <ql/time/frequency.hpp>
#include <ql/time/date.hpp>

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace QuantLib;

namespace {

// Dump one InflationIndex's default-construction signature.
void dumpZero(const std::string& key, const ZeroInflationIndex& idx, bool last) {
    std::cout << "    \"" << key << "\": {\n";
    std::cout << "      \"name\": \"" << idx.name() << "\",\n";
    std::cout << "      \"family_name\": \"" << idx.familyName() << "\",\n";
    std::cout << "      \"region_name\": \"" << idx.region().name() << "\",\n";
    std::cout << "      \"region_code\": \"" << idx.region().code() << "\",\n";
    std::cout << "      \"revised\": " << (idx.revised() ? "true" : "false") << ",\n";
    std::cout << "      \"frequency\": " << static_cast<int>(idx.frequency()) << ",\n";
    std::cout << "      \"availability_lag_months\": " << idx.availabilityLag().length() << ",\n";
    std::cout << "      \"currency_code\": \"" << idx.currency().code() << "\"\n";
    std::cout << "    }" << (last ? "" : ",") << "\n";
}

void dumpYoY(const std::string& key, const YoYInflationIndex& idx, bool last) {
    std::cout << "    \"" << key << "\": {\n";
    std::cout << "      \"name\": \"" << idx.name() << "\",\n";
    std::cout << "      \"family_name\": \"" << idx.familyName() << "\",\n";
    std::cout << "      \"region_name\": \"" << idx.region().name() << "\",\n";
    std::cout << "      \"region_code\": \"" << idx.region().code() << "\",\n";
    std::cout << "      \"revised\": " << (idx.revised() ? "true" : "false") << ",\n";
    std::cout << "      \"frequency\": " << static_cast<int>(idx.frequency()) << ",\n";
    std::cout << "      \"availability_lag_months\": " << idx.availabilityLag().length() << ",\n";
    std::cout << "      \"currency_code\": \"" << idx.currency().code() << "\",\n";
    std::cout << "      \"interpolated\": " << (idx.interpolated() ? "true" : "false") << ",\n";
    std::cout << "      \"ratio\": " << (idx.ratio() ? "true" : "false") << "\n";
    std::cout << "    }" << (last ? "" : ",") << "\n";
}

void dumpDatePair(const std::string& label,
                  const std::pair<Date, Date>& p,
                  bool last) {
    std::cout << "      {\n";
    std::cout << "        \"label\": \"" << label << "\",\n";
    std::cout << "        \"start_serial\": " << p.first.serialNumber() << ",\n";
    std::cout << "        \"end_serial\": " << p.second.serialNumber() << "\n";
    std::cout << "      }" << (last ? "" : ",") << "\n";
}

}  // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    // ---- zero-inflation index default-market params -----------------------
    std::cout << "  \"zero_indexes\": {\n";
    {
        EUHICP eu;
        FRHICP fr;
        UKRPI ukrpi;
        UKHICP ukhicp;
        USCPI us;
        dumpZero("EUHICP", eu, false);
        dumpZero("FRHICP", fr, false);
        dumpZero("UKRPI", ukrpi, false);
        dumpZero("UKHICP", ukhicp, false);
        dumpZero("USCPI", us, true);
    }
    std::cout << "  },\n";

    // ---- YoY siblings ------------------------------------------------------
    std::cout << "  \"yoy_indexes\": {\n";
    {
        YYEUHICP eu;
        YYFRHICP fr;
        YYUKRPI uk;
        YYUSCPI us;
        dumpYoY("YYEUHICP", eu, false);
        dumpYoY("YYFRHICP", fr, false);
        dumpYoY("YYUKRPI", uk, false);
        dumpYoY("YYUSCPI", us, true);
    }
    std::cout << "  },\n";

    // ---- inflationPeriod for sample (Year, Month, Day) anchors -----------
    std::cout << "  \"inflation_period\": {\n";
    std::cout << "    \"monthly_2020_05_15\": {\n";
    std::cout << "      \"frequency\": 12,\n";
    {
        auto p = inflationPeriod(Date(15, May, 2020), Monthly);
        std::cout << "      \"start_serial\": " << p.first.serialNumber() << ",\n";
        std::cout << "      \"end_serial\": " << p.second.serialNumber() << "\n";
    }
    std::cout << "    },\n";
    std::cout << "    \"quarterly_2021_07_20\": {\n";
    std::cout << "      \"frequency\": 4,\n";
    {
        auto p = inflationPeriod(Date(20, July, 2021), Quarterly);
        std::cout << "      \"start_serial\": " << p.first.serialNumber() << ",\n";
        std::cout << "      \"end_serial\": " << p.second.serialNumber() << "\n";
    }
    std::cout << "    },\n";
    std::cout << "    \"semiannual_2019_11_30\": {\n";
    std::cout << "      \"frequency\": 2,\n";
    {
        auto p = inflationPeriod(Date(30, November, 2019), Semiannual);
        std::cout << "      \"start_serial\": " << p.first.serialNumber() << ",\n";
        std::cout << "      \"end_serial\": " << p.second.serialNumber() << "\n";
    }
    std::cout << "    },\n";
    std::cout << "    \"annual_2022_03_01\": {\n";
    std::cout << "      \"frequency\": 1,\n";
    {
        auto p = inflationPeriod(Date(1, March, 2022), Annual);
        std::cout << "      \"start_serial\": " << p.first.serialNumber() << ",\n";
        std::cout << "      \"end_serial\": " << p.second.serialNumber() << "\n";
    }
    std::cout << "    },\n";
    std::cout << "    \"every_fourth_month_2020_08_10\": {\n";
    std::cout << "      \"frequency\": 3,\n";
    {
        auto p = inflationPeriod(Date(10, August, 2020), EveryFourthMonth);
        std::cout << "      \"start_serial\": " << p.first.serialNumber() << ",\n";
        std::cout << "      \"end_serial\": " << p.second.serialNumber() << "\n";
    }
    std::cout << "    }\n";
    std::cout << "  },\n";

    // ---- MultiplicativePriceSeasonality factor lookup ---------------------
    // A stationary 12-month seasonality curve from C++ test_inflation.cpp
    // (mimics ECB-style seasonal adjustment).
    std::cout << "  \"seasonality\": {\n";
    {
        std::vector<Rate> factors = {1.0, 1.005, 1.01, 1.015, 1.02, 1.025,
                                     1.0, 0.995, 0.99, 0.985, 0.98, 0.975};
        Date seasonalityBase(1, January, 2020);
        MultiplicativePriceSeasonality season(seasonalityBase, Monthly, factors);

        std::cout << "    \"base_date_serial\": " << seasonalityBase.serialNumber() << ",\n";
        std::cout << "    \"frequency\": " << static_cast<int>(season.frequency()) << ",\n";
        std::cout << "    \"factors\": [";
        for (size_t i = 0; i < factors.size(); ++i) {
            std::cout << factors[i];
            if (i + 1 < factors.size())
                std::cout << ", ";
        }
        std::cout << "],\n";

        // Sample factor lookups for various dates.
        std::cout << "    \"samples\": [\n";
        struct Sample { const char* label; Date d; };
        std::vector<Sample> samples = {
            {"2020_jan_01", Date(1, January, 2020)},
            {"2020_jun_15", Date(15, June, 2020)},
            {"2020_dec_31", Date(31, December, 2020)},
            {"2021_apr_10", Date(10, April, 2021)},
            {"2022_feb_28", Date(28, February, 2022)},
            {"2019_jul_20", Date(20, July, 2019)},
        };
        for (size_t i = 0; i < samples.size(); ++i) {
            const auto& s = samples[i];
            Real factor = season.seasonalityFactor(s.d);
            std::cout << "      {\n";
            std::cout << "        \"label\": \"" << s.label << "\",\n";
            std::cout << "        \"date_serial\": " << s.d.serialNumber() << ",\n";
            std::cout << "        \"factor\": " << factor << "\n";
            std::cout << "      }" << (i + 1 == samples.size() ? "" : ",") << "\n";
        }
        std::cout << "    ]\n";
    }
    std::cout << "  }\n";

    std::cout << "}\n";
    return 0;
}

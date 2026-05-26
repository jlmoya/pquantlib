// L1-B mega-probe: copulas (12 distribution-independent), simple
// distributions, simple statistics, currencies (names only).

#include <ql/currencies/africa.hpp>
#include <ql/currencies/america.hpp>
#include <ql/currencies/asia.hpp>
#include <ql/currencies/europe.hpp>
#include <ql/math/copulas/alimikhailhaqcopula.hpp>
#include <ql/math/copulas/claytoncopula.hpp>
#include <ql/math/copulas/farliegumbelmorgensterncopula.hpp>
#include <ql/math/copulas/frankcopula.hpp>
#include <ql/math/copulas/galamboscopula.hpp>
#include <ql/math/copulas/gumbelcopula.hpp>
#include <ql/math/copulas/huslerreisscopula.hpp>
#include <ql/math/copulas/independentcopula.hpp>
#include <ql/math/copulas/marshallolkincopula.hpp>
#include <ql/math/copulas/maxcopula.hpp>
#include <ql/math/copulas/mincopula.hpp>
#include <ql/math/copulas/plackettcopula.hpp>
#include <ql/math/distributions/normaldistribution.hpp>
#include <ql/math/statistics/generalstatistics.hpp>
#include <ql/math/statistics/incrementalstatistics.hpp>

#include <iomanip>
#include <iostream>
#include <vector>

using namespace QuantLib;

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    // --- copulas: emit operator()(x, y) on grid ----------------------------
    std::cout << "  \"copulas\": {\n";

    auto emit_copula = [](const char* key, auto cop, double param) {
        std::cout << "    \"" << key << "\": {\"param\": " << param << ", \"values\": [\n";
        bool first = true;
        for (double x : {0.1, 0.25, 0.5, 0.75, 0.9}) {
            for (double y : {0.1, 0.5, 0.9}) {
                if (!first) std::cout << ",\n";
                std::cout << "      {\"x\":" << x << ",\"y\":" << y
                          << ",\"v\":" << cop(x, y) << "}";
                first = false;
            }
        }
        std::cout << "\n    ]}";
    };

    emit_copula("clayton",   ClaytonCopula(0.5), 0.5); std::cout << ",\n";
    emit_copula("gumbel",    GumbelCopula(2.0),  2.0); std::cout << ",\n";
    emit_copula("frank",     FrankCopula(3.0),   3.0); std::cout << ",\n";
    emit_copula("ali",       AliMikhailHaqCopula(0.5), 0.5); std::cout << ",\n";
    emit_copula("fgm",       FarlieGumbelMorgensternCopula(0.5), 0.5); std::cout << ",\n";
    emit_copula("galambos",  GalambosCopula(2.0), 2.0); std::cout << ",\n";
    emit_copula("husler",    HuslerReissCopula(1.0), 1.0); std::cout << ",\n";
    emit_copula("indep",     IndependentCopula(), 0.0); std::cout << ",\n";
    emit_copula("marshall",  MarshallOlkinCopula(0.3, 0.4), 0.3); std::cout << ",\n";
    emit_copula("maxc",      MaxCopula(), 0.0); std::cout << ",\n";
    emit_copula("minc",      MinCopula(), 0.0); std::cout << ",\n";
    emit_copula("plackett",  PlackettCopula(2.0), 2.0); std::cout << "\n";
    std::cout << "  },\n";

    // --- distributions ----------------------------------------------------
    std::cout << "  \"distributions\": {\n";
    {
        NormalDistribution nd;  // std normal
        CumulativeNormalDistribution cnd;
        std::cout << "    \"normal_pdf\": [\n";
        bool first = true;
        for (double x : {-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0}) {
            if (!first) std::cout << ",\n";
            std::cout << "      {\"x\":" << x << ",\"v\":" << nd(x) << "}";
            first = false;
        }
        std::cout << "\n    ],\n";
        std::cout << "    \"normal_cdf\": [\n";
        first = true;
        for (double x : {-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0}) {
            if (!first) std::cout << ",\n";
            std::cout << "      {\"x\":" << x << ",\"v\":" << cnd(x) << "}";
            first = false;
        }
        std::cout << "\n    ],\n";
        InverseCumulativeNormal icn;
        std::cout << "    \"inverse_normal_cdf\": [\n";
        first = true;
        for (double p : {0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99}) {
            if (!first) std::cout << ",\n";
            std::cout << "      {\"p\":" << p << ",\"v\":" << icn(p) << "}";
            first = false;
        }
        std::cout << "\n    ]\n";
    }
    std::cout << "  },\n";

    // --- statistics -------------------------------------------------------
    std::cout << "  \"statistics\": {\n";
    {
        std::vector<double> samples = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
        GeneralStatistics gs;
        for (auto s : samples) gs.add(s);
        std::cout << "    \"general\": {\n";
        std::cout << "      \"mean\": "        << gs.mean()        << ",\n";
        std::cout << "      \"variance\": "    << gs.variance()    << ",\n";
        std::cout << "      \"standardDeviation\": " << gs.standardDeviation() << ",\n";
        std::cout << "      \"skewness\": "    << gs.skewness()    << ",\n";
        std::cout << "      \"kurtosis\": "    << gs.kurtosis()    << ",\n";
        std::cout << "      \"min\": "         << gs.min()         << ",\n";
        std::cout << "      \"max\": "         << gs.max()         << ",\n";
        std::cout << "      \"samples\": "     << gs.samples()     << "\n";
        std::cout << "    },\n";
        IncrementalStatistics is;
        for (auto s : samples) is.add(s);
        std::cout << "    \"incremental\": {\n";
        std::cout << "      \"mean\": "        << is.mean()        << ",\n";
        std::cout << "      \"variance\": "    << is.variance()    << ",\n";
        std::cout << "      \"standardDeviation\": " << is.standardDeviation() << ",\n";
        std::cout << "      \"min\": "         << is.min()         << ",\n";
        std::cout << "      \"max\": "         << is.max()         << "\n";
        std::cout << "    }\n";
    }
    std::cout << "  },\n";

    // --- currencies: ISO codes only (the rest is static data) -------------
    std::cout << "  \"currencies\": {\n";
    std::cout << "    \"USD\": \"" << USDCurrency().code() << "\",\n";
    std::cout << "    \"EUR\": \"" << EURCurrency().code() << "\",\n";
    std::cout << "    \"GBP\": \"" << GBPCurrency().code() << "\",\n";
    std::cout << "    \"JPY\": \"" << JPYCurrency().code() << "\",\n";
    std::cout << "    \"CHF\": \"" << CHFCurrency().code() << "\",\n";
    std::cout << "    \"USD_name\": \"" << USDCurrency().name() << "\",\n";
    std::cout << "    \"USD_symbol\": \"" << USDCurrency().symbol() << "\",\n";
    std::cout << "    \"USD_numeric\": " << USDCurrency().numericCode() << ",\n";
    std::cout << "    \"USD_fractions\": " << USDCurrency().fractionsPerUnit() << "\n";
    std::cout << "  }\n";

    std::cout << "}\n";
    return 0;
}

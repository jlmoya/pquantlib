// v1.43 methods coverage: risk-neutral density calculators.
//
// ql/methods/finitedifferences/utilities/{riskneutraldensitycalculator,
// bsmrndcalculator, cevrndcalculator, squarerootprocessrndcalculator,
// hestonrndcalculator}.{hpp,cpp} @ v1.43 (6b57206e0).
//
// These are the classes where C++ reaches straight into Boost
// (non_central_chi_squared_distribution, gamma_p, gamma_p_inv) rather than
// through QuantLib's own distributions. The Python port answers with
// scipy.stats.ncx2 / scipy.special.gammainc(inv), so every branch is pinned
// here — including deep-tail quantiles, both sides of the CEV delta<2 / >=2
// split, and the absorbing mass at zero — so the delegation is proven, not
// assumed.
//
// Settings::evaluationDate() is deliberately NOT set (see cluster_w5a's
// note); all term structures are built with an explicit reference date.

#include <ql/methods/finitedifferences/utilities/bsmrndcalculator.hpp>
#include <ql/methods/finitedifferences/utilities/cevrndcalculator.hpp>
#include <ql/methods/finitedifferences/utilities/hestonrndcalculator.hpp>
#include <ql/methods/finitedifferences/utilities/squarerootprocessrndcalculator.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/processes/hestonprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

#include <cstdio>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

using namespace QuantLib;

namespace {

    bool g_first = true;

    std::string fmt(Real x) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.17g", x);
        return buf;
    }

    // A case that throws inside Boost is pinned as the JSON string
    // "raises" rather than dropped: the throw IS the C++ behaviour and the
    // Python port has to reproduce it (or reproduce a finite value, in
    // which case the test fails loudly instead of silently agreeing).
    using Cell = std::string;

    Cell cell(const std::function<Real()>& f) {
        try {
            return fmt(f());
        } catch (...) {
            return "\"raises\"";
        }
    }

    void emit_cells(const std::string& key, const std::vector<Cell>& v) {
        if (!g_first) std::cout << ",\n";
        g_first = false;
        std::cout << "  \"" << key << "\": [";
        for (Size i = 0; i < v.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << v[i];
        }
        std::cout << "]";
    }

    void emit(const std::string& key, const std::vector<Real>& v) {
        std::vector<Cell> cells;
        for (Real x : v) cells.push_back(fmt(x));
        emit_cells(key, cells);
    }

    void emit1(const std::string& key, Real v) {
        if (!g_first) std::cout << ",\n";
        g_first = false;
        std::cout << "  \"" << key << "\": " << fmt(v);
    }

    void emit1c(const std::string& key, const std::function<Real()>& f) {
        if (!g_first) std::cout << ",\n";
        g_first = false;
        std::cout << "  \"" << key << "\": " << cell(f);
    }

    const Date REF_DATE(15, May, 2026);
    const DayCounter DC = Actual365Fixed();

    Handle<YieldTermStructure> flat(Rate r) {
        return Handle<YieldTermStructure>(
            ext::make_shared<FlatForward>(REF_DATE, r, DC));
    }

}

int main() {
    std::cout.precision(17);
    std::cout << "{\n";

    // ------------------------------------------------------------------
    // SquareRootProcessRNDCalculator — the pure Boost-ncx2 case.
    // Two parameter sets: one comfortably Feller-satisfying, one where
    // 2*kappa*theta < sigma^2 so df < 2 and the density blows up at 0.
    // ------------------------------------------------------------------
    {
        struct P { const char* name; Real v0, kappa, theta, sigma; };
        const P sets[] = {
            {"feller",  0.09, 1.0, 0.09, 0.4},
            {"nofeller", 0.04, 0.5, 0.04, 0.9},
        };
        const Real ts[] = {0.05, 0.5, 2.0, 10.0};
        const Real vs[] = {1e-4, 0.005, 0.02, 0.04, 0.09, 0.25, 0.6, 1.5};
        const Real qs[] = {1e-8, 1e-4, 0.01, 0.25, 0.5, 0.75, 0.99, 0.9999, 1 - 1e-8};

        for (const auto& s : sets) {
            SquareRootProcessRNDCalculator c(s.v0, s.kappa, s.theta, s.sigma);
            for (Real t : ts) {
                std::vector<Cell> pdf, cdf;
                for (Real v : vs) {
                    pdf.push_back(cell([&]{ return c.pdf(v, t); }));
                    cdf.push_back(cell([&]{ return c.cdf(v, t); }));
                }
                emit_cells(std::string("sqrt_") + s.name + "_pdf_t" + fmt(t), pdf);
                emit_cells(std::string("sqrt_") + s.name + "_cdf_t" + fmt(t), cdf);

                std::vector<Cell> inv;
                for (Real q : qs) inv.push_back(cell([&]{ return c.invcdf(q, t); }));
                emit_cells(std::string("sqrt_") + s.name + "_invcdf_t" + fmt(t), inv);
            }
            std::vector<Cell> spdf, scdf, sinv;
            for (Real v : vs) {
                spdf.push_back(cell([&]{ return c.stationary_pdf(v); }));
                scdf.push_back(cell([&]{ return c.stationary_cdf(v); }));
            }
            for (Real q : qs) sinv.push_back(cell([&]{ return c.stationary_invcdf(q); }));
            emit_cells(std::string("sqrt_") + s.name + "_stationary_pdf", spdf);
            emit_cells(std::string("sqrt_") + s.name + "_stationary_cdf", scdf);
            emit_cells(std::string("sqrt_") + s.name + "_stationary_invcdf", sinv);
        }
    }

    // ------------------------------------------------------------------
    // CEVRNDCalculator — both sides of the delta split.
    //   beta < 0.5  -> delta = (1-2b)/(1-b) > 0 but < 2  (mass at zero)
    //   beta > 0.5  -> delta < 0 ... also < 2
    //   beta < 0    -> delta > 2
    // ------------------------------------------------------------------
    {
        struct P { const char* name; Real f0, alpha, beta; };
        const P sets[] = {
            {"b03",  100.0, 0.3,  0.3},   // delta = 4/7  (< 2)
            {"b07",  100.0, 0.4,  0.7},   // delta = -4/3 (< 2)
            {"bm05", 100.0, 0.2, -0.5},   // delta = 4/3  (< 2)
            {"bm1",    1.0, 0.5, -1.0},   // delta = 3/2  (< 2), small f0
            {"b15",  100.0, 0.5,  1.5},   // delta = 4    (>= 2 branch)
            {"b20",  100.0, 0.5,  2.0},   // delta = 3    (>= 2 branch)
        };
        const Real ts[] = {0.25, 1.0, 5.0};
        // Grid relative to f0 so every parameter set is probed over the
        // same shape of its own distribution.
        const Real rel[] = {0.01, 0.25, 0.6, 0.9, 1.0, 1.3, 2.0, 4.0};
        const Real qs[] = {0.001, 0.05, 0.25, 0.5, 0.75, 0.95, 0.999};

        for (const auto& s : sets) {
            CEVRNDCalculator c(s.f0, s.alpha, s.beta);
            emit1(std::string("cev_") + s.name + "_delta",
                  (1.0 - 2.0 * s.beta) / (1.0 - s.beta));
            for (Real t : ts) {
                std::vector<Cell> pdf, cdf;
                for (Real rf : rel) {
                    const Real f = rf * s.f0;
                    pdf.push_back(cell([&]{ return c.pdf(f, t); }));
                    cdf.push_back(cell([&]{ return c.cdf(f, t); }));
                }
                emit_cells(std::string("cev_") + s.name + "_pdf_t" + fmt(t), pdf);
                emit_cells(std::string("cev_") + s.name + "_cdf_t" + fmt(t), cdf);
                emit1c(std::string("cev_") + s.name + "_mass0_t" + fmt(t),
                       [&]{ return c.massAtZero(t); });

                std::vector<Cell> inv;
                for (Real q : qs) inv.push_back(cell([&]{ return c.invcdf(q, t); }));
                emit_cells(std::string("cev_") + s.name + "_invcdf_t" + fmt(t), inv);
            }
        }
    }

    // ------------------------------------------------------------------
    // BSMRNDCalculator, on a constant-vol GBSM process.
    // ------------------------------------------------------------------
    {
        const auto spot = ext::make_shared<SimpleQuote>(100.0);
        const auto process = ext::make_shared<BlackScholesMertonProcess>(
            Handle<Quote>(spot), flat(0.02), flat(0.05),
            Handle<BlackVolTermStructure>(ext::make_shared<BlackConstantVol>(
                REF_DATE, NullCalendar(), 0.25, DC)));

        BSMRNDCalculator c(process);
        const Real ts[] = {0.1, 1.0, 3.0};
        const Real xs[] = {3.5, 4.0, 4.4, 4.60517, 4.8, 5.2, 5.8};
        const Real qs[] = {0.01, 0.1, 0.5, 0.9, 0.99};
        for (Real t : ts) {
            std::vector<Cell> pdf, cdf, inv;
            for (Real x : xs) {
                pdf.push_back(cell([&]{ return c.pdf(x, t); }));
                cdf.push_back(cell([&]{ return c.cdf(x, t); }));
            }
            for (Real q : qs) inv.push_back(cell([&]{ return c.invcdf(q, t); }));
            emit_cells(std::string("bsm_pdf_t") + fmt(t), pdf);
            emit_cells(std::string("bsm_cdf_t") + fmt(t), cdf);
            emit_cells(std::string("bsm_invcdf_t") + fmt(t), inv);
        }
    }

    // ------------------------------------------------------------------
    // HestonRNDCalculator (Dragulescu-Yakovenko).
    // ------------------------------------------------------------------
    {
        const auto spot = ext::make_shared<SimpleQuote>(100.0);
        const auto process = ext::make_shared<HestonProcess>(
            flat(0.05), flat(0.02), Handle<Quote>(spot),
            0.09, 1.0, 0.09, 0.4, -0.75);

        HestonRNDCalculator c(process);
        const Real ts[] = {0.5, 2.0};
        const Real xs[] = {4.0, 4.4, 4.60517, 4.8, 5.2};
        const Real qs[] = {0.05, 0.25, 0.5, 0.75, 0.95};
        for (Real t : ts) {
            std::vector<Cell> pdf, cdf, inv;
            for (Real x : xs) {
                pdf.push_back(cell([&]{ return c.pdf(x, t); }));
                cdf.push_back(cell([&]{ return c.cdf(x, t); }));
            }
            for (Real q : qs) inv.push_back(cell([&]{ return c.invcdf(q, t); }));
            emit_cells(std::string("heston_pdf_t") + fmt(t), pdf);
            emit_cells(std::string("heston_cdf_t") + fmt(t), cdf);
            emit_cells(std::string("heston_invcdf_t") + fmt(t), inv);
        }
    }

    std::cout << "\n}" << std::endl;
    return 0;
}

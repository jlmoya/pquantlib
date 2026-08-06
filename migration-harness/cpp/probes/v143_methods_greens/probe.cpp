// v1.43 methods coverage: FdmHestonGreensFct + LocalVolRNDCalculator,
// plus HestonProcess::pdf, which FdmHestonGreensFct's SemiAnalytical
// algorithm depends on and which PQuantLib had listed as a carve-out.
//
// ql/methods/finitedifferences/utilities/{fdmhestongreensfct,
// localvolrndcalculator}.{hpp,cpp} and ql/processes/hestonprocess.cpp
// @ v1.43 (6b57206e0).
//
// HestonProcess::pdf is pinned on its own before the Green's function that
// consumes it, so a discrepancy in the Broadie-Kaya characteristic function
// (modified Bessel of complex argument, 128-point Gauss-Laguerre, the
// Cornish-Fisher bound) is attributed to the right class.
//
// Settings::evaluationDate() is deliberately NOT set (see cluster_w5a's
// note); all term structures carry an explicit reference date.

#include <ql/methods/finitedifferences/meshers/fdmmeshercomposite.hpp>
#include <ql/methods/finitedifferences/meshers/uniform1dmesher.hpp>
#include <ql/methods/finitedifferences/utilities/fdmhestongreensfct.hpp>
#include <ql/methods/finitedifferences/utilities/localvolrndcalculator.hpp>
#include <ql/processes/hestonprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/timegrid.hpp>
#include <ql/termstructures/volatility/equityfx/localconstantvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

#include <cstdio>
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

    void emit(const std::string& key, const std::vector<Real>& v) {
        if (!g_first) std::cout << ",\n";
        g_first = false;
        std::cout << "  \"" << key << "\": [";
        for (Size i = 0; i < v.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << fmt(v[i]);
        }
        std::cout << "]";
    }

    void emit_sizes(const std::string& key, const std::vector<Size>& v) {
        if (!g_first) std::cout << ",\n";
        g_first = false;
        std::cout << "  \"" << key << "\": [";
        for (Size i = 0; i < v.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << v[i];
        }
        std::cout << "]";
    }

    void emit1(const std::string& key, Real v) {
        if (!g_first) std::cout << ",\n";
        g_first = false;
        std::cout << "  \"" << key << "\": " << fmt(v);
    }

    void emit_array(const std::string& key, const Array& a) {
        emit(key, std::vector<Real>(a.begin(), a.end()));
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

    const auto spot = ext::make_shared<SimpleQuote>(100.0);
    const auto hestonProcess = ext::make_shared<HestonProcess>(
        flat(0.05), flat(0.02), Handle<Quote>(spot),
        0.09, 1.0, 0.09, 0.4, -0.75);

    // ------------------------------------------------------------------
    // HestonProcess::pdf on its own.
    // ------------------------------------------------------------------
    {
        const Real xs[] = {4.3, 4.5, 4.60517, 4.75, 4.95};
        const Real vs[] = {0.04, 0.09, 0.16};
        // t = 1 and t = 2 only: at t <= 0.25 QuantLib's own pdf returns NaN
        // (and at t = 0.1 it returns a negative density). Verified by
        // sweeping sigma in {0.4, 0.2, 0.1} x rho in {-0.75, -0.3, 0} — the
        // breakdown is a property of the Broadie-Kaya inversion at short
        // horizons, not of any one parameter set. The port reproduces the
        // usable range; the NaN range is recorded in the test docstring.
        for (Time t : {1.0, 2.0}) {
            for (Real v : vs) {
                std::vector<Real> row;
                for (Real x : xs)
                    row.push_back(hestonProcess->pdf(x, v, t, 1e-4));
                emit("heston_pdf_t" + fmt(t) + "_v" + fmt(v), row);
            }
        }
    }

    // ------------------------------------------------------------------
    // FdmHestonGreensFct on a 2-D mesh, all three algorithms x all three
    // transformation types.
    // ------------------------------------------------------------------
    {
        const auto mx = ext::make_shared<Uniform1dMesher>(4.2, 5.0, 5);
        // Plain / Power read the variance directly; Log reads exp(location),
        // so the second mesher is chosen to be sensible under both readings.
        const auto mv = ext::make_shared<Uniform1dMesher>(-2.6, -1.4, 4);
        const auto mesher = ext::make_shared<FdmMesherComposite>(mx, mv);

        struct T { const char* name; FdmSquareRootFwdOp::TransformationType v; };
        const T trafos[] = {
            {"plain", FdmSquareRootFwdOp::Plain},
            {"log",   FdmSquareRootFwdOp::Log},
            {"power", FdmSquareRootFwdOp::Power},
        };
        struct A { const char* name; FdmHestonGreensFct::Algorithm v; };
        const A algos[] = {
            {"zerocorr", FdmHestonGreensFct::ZeroCorrelation},
            {"gaussian", FdmHestonGreensFct::Gaussian},
            {"semi",     FdmHestonGreensFct::SemiAnalytical},
        };

        for (const auto& tr : trafos) {
            for (Real l0 : {1.0, 1.3}) {
                const FdmHestonGreensFct g(mesher, hestonProcess, tr.v, l0);
                for (const auto& al : algos) {
                    // Plain/Power read the raw location as the variance, which
                    // must be positive; use a positive-variance mesher there.
                    if (tr.v != FdmSquareRootFwdOp::Log)
                        continue;
                    emit_array(std::string("greens_") + tr.name + "_" + al.name +
                                   "_l0" + fmt(l0),
                               g.get(1.0, al.v));
                }
            }
        }

        // Plain and Power on a positive-variance mesher.
        const auto mvPos = ext::make_shared<Uniform1dMesher>(0.02, 0.30, 4);
        const auto mesherPos = ext::make_shared<FdmMesherComposite>(mx, mvPos);
        for (const auto& tr : trafos) {
            if (tr.v == FdmSquareRootFwdOp::Log) continue;
            const FdmHestonGreensFct g(mesherPos, hestonProcess, tr.v, 1.0);
            for (const auto& al : algos)
                emit_array(std::string("greens_pos_") + tr.name + "_" + al.name,
                           g.get(1.0, al.v));
        }
    }

    // ------------------------------------------------------------------
    // LocalVolRNDCalculator on a flat local-vol surface.
    // ------------------------------------------------------------------
    {
        const auto localVol = ext::make_shared<LocalConstantVol>(REF_DATE, 0.25, DC);

        // An explicit TimeGrid: LocalConstantVol has no max date, so
        // localVol->maxTime() is ~174y and the default constructor would put
        // the whole grid past any horizon we care about.
        const auto timeGrid = ext::make_shared<TimeGrid>(1.0, 7);

        LocalVolRNDCalculator calc(
            spot, flat(0.05).currentLink(), flat(0.02).currentLink(),
            localVol, timeGrid, 21);

        emit_sizes("lv_rescale_time_steps", calc.rescaleTimeSteps());
        emit1("lv_time_grid_back", calc.timeGrid()->back());
        emit1("lv_time_grid_size", Real(calc.timeGrid()->size()));

        // The per-step meshers are part of the contract.
        for (Size i : {Size(1), Size(3), Size(7)}) {
            const Time t = calc.timeGrid()->at(i);
            const auto m = calc.mesher(t);
            std::vector<Real> loc;
            for (Size j = 0; j < m->size(); ++j)
                loc.push_back(m->location(j));
            emit("lv_mesher_" + std::to_string(i), loc);
        }

        const Real ts[] = {0.0005, 0.05, 0.4, 1.0};
        for (Real t : ts) {
            const auto m = calc.mesher(calc.timeGrid()->closestTime(t));
            const Real lo = m->locations().front();
            const Real hi = m->locations().back();
            std::vector<Real> pdf;
            for (int k = 1; k <= 7; ++k) {
                const Real x = lo + (hi - lo) * k / 8.0;
                pdf.push_back(calc.pdf(x, t));
            }
            emit("lv_pdf_t" + fmt(t), pdf);
        }

        for (Real t : {0.4, 1.0}) {
            std::vector<Real> cdf, inv;
            const auto m = calc.mesher(calc.timeGrid()->closestTime(t));
            const Real lo = m->locations().front();
            const Real hi = m->locations().back();
            for (int k = 1; k <= 5; ++k) {
                const Real x = lo + (hi - lo) * k / 6.0;
                cdf.push_back(calc.cdf(x, t));
            }
            emit("lv_cdf_t" + fmt(t), cdf);
            for (Real q : {0.1, 0.5, 0.9})
                inv.push_back(calc.invcdf(q, t));
            emit("lv_invcdf_t" + fmt(t), inv);
        }
    }

    std::cout << "\n}" << std::endl;
    return 0;
}

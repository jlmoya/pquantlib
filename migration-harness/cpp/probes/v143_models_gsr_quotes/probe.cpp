// migration-harness/cpp/probes/v143_models_gsr_quotes/probe.cpp
//
// Reference values for Gsr's quote-driven volatility and reversion, i.e. for
// Gsr::VolatilityObserver and Gsr::ReversionObserver (gsr.hpp:176 and :181)
// and the Gsr::updateVolatility / Gsr::updateReversion they route to
// (gsr.cpp:121-135).
//
// What is pinned and why:
//
//   * The model state BEFORE and AFTER a quote is bumped, for the volatility
//     quotes and the reversion quotes separately. The whole point of the two
//     observers is that a bump propagates without the caller touching the
//     model, so a port that stored the numbers instead of the quotes would
//     reproduce the "before" values exactly and every "after" value wrongly.
//     Reading only the post-bump state would not distinguish that from a port
//     that simply rebuilt the model.
//
//   * volatility() and reversion() (the Parameter arrays) AND a zerobond at a
//     non-zero state, because the observers have to push the new numbers into
//     the GsrProcess as well as into the Parameters — flushing the process
//     cache is a separate step (gsr.cpp:125 / :133) that a partial port would
//     miss, and only a state-dependent quantity reads it back.
//
//   * A PIECEWISE reversion (one quote per volstep) as well as the single-quote
//     case, since ReversionObserver is registered against every element.
//
//   * Bumps are applied ONE quote at a time, so a port that refreshed the whole
//     vector on any notification cannot be told apart from one that refreshed
//     only the changed element — both are correct here, but the per-element
//     values catch a port that refreshed the WRONG element.
//
// Emits JSON on stdout; redirect to references/v143/models/gsr_quotes.json.

#include <exception>
#include <iomanip>
#include <iostream>
#include <vector>

#include <ql/models/shortrate/onefactormodels/gsr.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

using namespace QuantLib;

namespace {

    std::ostream& out = std::cout;
    void num(Real x) { out << std::setprecision(17) << x; }

    const Date TODAY(14, May, 2026);

    void emitState(const char* label, const Gsr& gsr) {
        out << "    \"" << label << "\": {\"volatility\": [";
        Array v = gsr.volatility();
        for (Size i = 0; i < v.size(); ++i) {
            if (i)
                out << ", ";
            num(v[i]);
        }
        out << "], \"reversion\": [";
        Array r = gsr.reversion();
        for (Size i = 0; i < r.size(); ++i) {
            if (i)
                out << ", ";
            num(r[i]);
        }
        out << "], \"zerobond_5_1_0p5\": ";
        num(gsr.zerobond(5.0, 1.0, 0.5));
        out << ", \"numeraire_1_0p5\": ";
        num(gsr.numeraire(1.0, 0.5));
        out << "}";
    }

}

int main() {
    try {
        Settings::instance().evaluationDate() = TODAY;
        DayCounter dc = Actual365Fixed();
        Handle<YieldTermStructure> yts(ext::make_shared<FlatForward>(
            TODAY, Handle<Quote>(ext::make_shared<SimpleQuote>(0.03)), dc));
        Calendar cal = TARGET();
        std::vector<Date> volstepdates = {cal.advance(TODAY, 1 * Years),
                                          cal.advance(TODAY, 2 * Years)};

        out << "{\n";
        out << "  \"today_serial\": " << TODAY.serialNumber() << ",\n";
        out << "  \"volstepdate_serials\": [" << volstepdates[0].serialNumber() << ", "
            << volstepdates[1].serialNumber() << "],\n";

        // --- single reversion quote, three volatility quotes ---------------
        {
            std::vector<Handle<Quote> > vols;
            std::vector<ext::shared_ptr<SimpleQuote> > volQ;
            for (Real v : {0.010, 0.012, 0.014}) {
                auto q = ext::make_shared<SimpleQuote>(v);
                volQ.push_back(q);
                vols.emplace_back(q);
            }
            auto revQ = ext::make_shared<SimpleQuote>(0.01);
            Handle<Quote> rev(revQ);

            Gsr gsr(yts, volstepdates, vols, rev, 60.0);

            out << "  \"single_reversion\": {\n";
            emitState("before", gsr);
            out << ",\n";
            volQ[1]->setValue(0.030);
            emitState("after_vol1_bump", gsr);
            out << ",\n";
            revQ->setValue(0.05);
            emitState("after_reversion_bump", gsr);
            out << "\n  },\n";
        }

        // --- piecewise reversion: one quote per volstep --------------------
        {
            std::vector<Handle<Quote> > vols;
            for (Real v : {0.010, 0.012, 0.014})
                vols.emplace_back(ext::make_shared<SimpleQuote>(v));
            std::vector<Handle<Quote> > revs;
            std::vector<ext::shared_ptr<SimpleQuote> > revQ;
            for (Real r : {0.010, 0.020, 0.030}) {
                auto q = ext::make_shared<SimpleQuote>(r);
                revQ.push_back(q);
                revs.emplace_back(q);
            }

            Gsr gsr(yts, volstepdates, vols, revs, 60.0);

            out << "  \"piecewise_reversion\": {\n";
            emitState("before", gsr);
            out << ",\n";
            revQ[2]->setValue(0.075);
            emitState("after_reversion2_bump", gsr);
            out << "\n  }\n";
        }

        out << "}\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "probe failed: " << e.what() << std::endl;
        return 1;
    }
}

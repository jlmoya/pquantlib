// migration-harness/cpp/probes/v143_models_markovfunctional/probe.cpp
//
// Reference values for the two MarkovFunctional calibration paths that the
// existing cluster_w1a probe does NOT reach:
//
//   * ModelSettings::CustomSmile — MarkovFunctional::CustomSmileSection and
//     MarkovFunctional::CustomSmileFactory (markovfunctional.hpp:103 and :108)
//   * a NON-ZERO smile shift, which is what makes
//     `lowerRateBound_ - smileSection_->shift()` differ from `lowerRateBound_`
//     in updateSmiles (markovfunctional.cpp:426-437) and in the y-sweep
//     (markovfunctional.cpp:558-559, :565-567), and what the `shift` argument
//     of marketSwapRate (markovfunctional.cpp:809-823) moves the lower Brent
//     bracket by.
//
// What is pinned and why:
//
//   * The FULL numeraire surface N(t, y) over the calibration times and a
//     spread of y values, for THREE configurations: the plain
//     no-pretreatment baseline, the shifted-lognormal input, and the custom
//     smile. The baseline is emitted from the same setup as cluster_w1a so
//     the three are directly comparable in the reference file itself — a port
//     that accepted `shift` and then discarded it would produce
//     shifted == baseline, which is visible here without running anything.
//
//   * The shift is 0.02 against a lowerRateBound of 0.001, so the shifted
//     configuration's rate floor is NEGATIVE (-0.019). That is the whole
//     point of a shifted-lognormal smile and it is unreachable with shift 0.
//
//   * The custom smile section's inverseDigitalCall is deliberately a CLOSED
//     FORM, not a Brent inversion of the underlying digital. Two independent
//     Brent inversions in two languages would agree only to solver tolerance
//     and would mask a threading error underneath that noise; a closed form
//     both sides compute identically makes the comparison exact, and what is
//     under test here is that the branch is TAKEN and the arguments are
//     threaded, not that some particular smile is right.
//
//     The chosen rule is  atm * exp(-(price/discount - 0.5)),  which is
//     monotone decreasing in price (as a digital-call inverse must be) and
//     lands in roughly [0.6 atm, 1.65 atm] over price/discount in [0, 1].
//
//   * zerobond(T, t, y) at t = 0 for each configuration, because the
//     numeraire and the zerobond are computed from different code paths off
//     the same tabulation.
//
// Emits JSON on stdout; redirect to references/v143/models/markovfunctional.json.

#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <ql/indexes/ibor/euribor.hpp>
#include <ql/indexes/swap/euriborswap.hpp>
#include <ql/models/shortrate/onefactormodels/markovfunctional.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/volatility/swaption/swaptionconstantvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

using namespace QuantLib;

namespace {

    std::ostream& out = std::cout;

    void num(Real x) { out << std::setprecision(17) << x; }

    // A CustomSmileSection whose inverse is closed-form; see the header note.
    class ProbeCustomSmileSection : public MarkovFunctional::CustomSmileSection {
      public:
        ProbeCustomSmileSection(ext::shared_ptr<SmileSection> source, Real atm)
        : MarkovFunctional::CustomSmileSection(), source_(std::move(source)), atm_(atm) {}

        Real minStrike() const override { return source_->minStrike(); }
        Real maxStrike() const override { return source_->maxStrike(); }
        Real atmLevel() const override { return atm_; }
        const Date& exerciseDate() const override { return source_->exerciseDate(); }
        Time exerciseTime() const override { return source_->exerciseTime(); }
        const DayCounter& dayCounter() const override { return source_->dayCounter(); }
        const Date& referenceDate() const override { return source_->referenceDate(); }
        VolatilityType volatilityType() const override { return source_->volatilityType(); }
        Rate shift() const override { return source_->shift(); }

        Real inverseDigitalCall(Real price, Real discount = 1.0) const override {
            return atm_ * std::exp(-(price / discount - 0.5));
        }

      protected:
        Volatility volatilityImpl(Rate strike) const override {
            return source_->volatility(strike);
        }

      private:
        ext::shared_ptr<SmileSection> source_;
        Real atm_;
    };

    class ProbeCustomSmileFactory : public MarkovFunctional::CustomSmileFactory {
      public:
        ext::shared_ptr<MarkovFunctional::CustomSmileSection>
        smileSection(const ext::shared_ptr<SmileSection>& source, Real atm) const override {
            return ext::make_shared<ProbeCustomSmileSection>(source, atm);
        }
    };

    // The cluster_w1a setup, verbatim, so the three configurations differ in
    // exactly one thing each.
    const Date TODAY(14, May, 2026);
    const Real Y_VALUES[] = {-2.0, -1.0, 0.0, 1.0, 2.0};

    void emitConfiguration(const char* name,
                           bool custom,
                           Real shift,
                           VolatilityType volType,
                           Integer tenorYears) {
        Settings::instance().evaluationDate() = TODAY;
        DayCounter dc = Actual365Fixed();
        Handle<YieldTermStructure> yts(
            ext::make_shared<FlatForward>(TODAY, Handle<Quote>(ext::make_shared<SimpleQuote>(0.03)), dc));

        ext::shared_ptr<SwapIndex> swapBase(
            new EuriborSwapIsdaFixA(tenorYears * Years, yts, yts));

        Handle<SwaptionVolatilityStructure> swVol(
            ext::make_shared<ConstantSwaptionVolatility>(
                TODAY, TARGET(), ModifiedFollowing, 0.20, dc, volType, shift));

        Calendar cal = TARGET();
        std::vector<Date> swaptionExpiries = {cal.advance(TODAY, 1 * Years),
                                              cal.advance(TODAY, 2 * Years),
                                              cal.advance(TODAY, 3 * Years)};
        std::vector<Period> swaptionTenors = {tenorYears * Years, tenorYears * Years,
                                              tenorYears * Years};
        std::vector<Date> volstepdates = {cal.advance(TODAY, 1 * Years),
                                          cal.advance(TODAY, 2 * Years)};
        std::vector<Real> volatilities = {0.01, 0.01, 0.01};
        Real reversion = 0.01;

        MarkovFunctional::ModelSettings settings;
        settings.withYGridPoints(32)
            .withYStdDevs(5.0)
            .withGaussHermitePoints(16)
            .withLowerRateBound(0.001)
            .withUpperRateBound(1.5);
        if (custom) {
            settings.withAdjustments(MarkovFunctional::ModelSettings::NoPayoffExtrapolation |
                                     MarkovFunctional::ModelSettings::CustomSmile)
                .withCustomSmileFactory(ext::make_shared<ProbeCustomSmileFactory>());
        } else {
            settings.withAdjustments(MarkovFunctional::ModelSettings::NoPayoffExtrapolation);
        }

        MarkovFunctional mf(yts, reversion, volstepdates, volatilities, swVol, swaptionExpiries,
                            swaptionTenors, swapBase, settings);

        out << "  \"" << name << "\": {\n";
        out << "    \"shift\": ";
        num(shift);
        out << ",\n";
        out << "    \"custom_smile\": " << (custom ? "true" : "false") << ",\n";
        out << "    \"tenor_years\": " << tenorYears << ",\n";
        out << "    \"numeraire_time\": ";
        num(mf.numeraireTime());
        out << ",\n";
        // The back-fill loop of markovfunctional.cpp:169-203 ADDS calibration
        // points at intermediate payment dates until the numeraire date is
        // covered, so both of these are strictly larger than what the three
        // input expiries alone would give.
        out << "    \"numeraire_date_serial\": " << mf.numeraireDate().serialNumber() << ",\n";
        out << "    \"n_calibration_points\": " << mf.modelOutputs().expiries_.size() << ",\n";
        out << "    \"calibration_expiry_serials\": [";
        for (Size q = 0; q < mf.modelOutputs().expiries_.size(); ++q) {
            if (q)
                out << ", ";
            out << mf.modelOutputs().expiries_[q].serialNumber();
        }
        out << "],\n";

        const Real times[] = {0.0, 1.0, 2.0, 3.0};
        out << "    \"numeraire\": [\n";
        bool first = true;
        for (Real t : times) {
            for (Real y : Y_VALUES) {
                if (!first)
                    out << ",\n";
                first = false;
                out << "      {\"t\": ";
                num(t);
                out << ", \"y\": ";
                num(y);
                out << ", \"value\": ";
                num(mf.numeraire(t, y));
                out << "}";
            }
        }
        out << "\n    ],\n";

        out << "    \"zerobond_t0\": [\n";
        first = true;
        for (Real T : {1.0, 5.0, 10.0}) {
            if (!first)
                out << ",\n";
            first = false;
            out << "      {\"T\": ";
            num(T);
            out << ", \"value\": ";
            num(mf.zerobond(T, 0.0, 0.0));
            out << "}";
        }
        out << "\n    ]\n";
        out << "  }";
    }

}

int main() {
    try {
        out << "{\n";
        out << "  \"today_serial\": " << TODAY.serialNumber() << ",\n";
        // 10Y tenor: the back-fill loop fires hard (12 calibration points from
        // 3 inputs). Kept because it is the exhibit that MEASURES the gap.
        emitConfiguration("baseline", false, 0.0, ShiftedLognormal, 10);
        out << ",\n";
        emitConfiguration("shifted", false, 0.02, ShiftedLognormal, 10);
        out << ",\n";
        emitConfiguration("custom_smile", true, 0.0, ShiftedLognormal, 10);
        out << ",\n";
        // 1Y tenor with expiries one year apart: every swaption's single payment
        // date is already a calibration expiry, so the back-fill loop adds
        // nothing and C++ calibrates exactly the 3 input points. That makes this
        // configuration confound-free, which is what the shift and CustomSmile
        // branches are actually asserted against.
        emitConfiguration("baseline_1y", false, 0.0, ShiftedLognormal, 1);
        out << ",\n";
        emitConfiguration("shifted_1y", false, 0.02, ShiftedLognormal, 1);
        out << ",\n";
        emitConfiguration("custom_smile_1y", true, 0.0, ShiftedLognormal, 1);
        out << "\n}\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "probe failed: " << e.what() << std::endl;
        return 1;
    }
}

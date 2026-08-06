// migration-harness/cpp/probes/v143_legacy_lmmswaption/probe.cpp
//
// Reference values for the two remaining LIBOR-Market-Model classes in
// ql/legacy/libormarketmodels/ (v1.43):
//
//   LiborForwardModel   liborforwardmodel.hpp   (S_0, w_0, Rebonato swaption
//                                                vol matrix, exact caplet
//                                                pricing through
//                                                discountBondOption)
//   LfmSwaptionEngine   lfmswaptionengine.hpp   (Black pricing off that matrix)
//
// The volatility / correlation / covariance / process classes they are built
// on are pinned by the sibling probe v143_legacy_lmm.
//
// What is pinned and why:
//
//   * S_0(alpha, beta) over the WHOLE (alpha, beta) triangle, not one cell.
//     S_0 is the forward swap rate implied by the model's own initial forward
//     curve; the C++ test-suite checks it against VanillaSwap::fairRate. The
//     weights w_0 are a nested product over f_ = 1 / (1 + tau_i L_i), so an
//     off-by-one in either loop bound still reproduces the alpha+1 == beta
//     cell. The whole triangle is emitted.
//
//   * The FULL Rebonato swaption volatility matrix returned by
//     getSwaptionVolatilityMatrix(), entry by entry, plus its reference date,
//     its option dates and its swap tenors. The matrix is (size/2) x (size/2)
//     with exercises taken from fixingDates[1 .. size/2] and lengths
//     (i+1) * index->tenor() — a port that mis-sliced either would still get
//     a plausible-looking matrix.
//
//   * Volatilities read back off the returned structure by
//     volatility(optionTime, swapLength, strike, extrapolate) at both pillar
//     and off-pillar coordinates, because that (and not the raw matrix) is
//     what LfmSwaptionEngine actually consults.
//
//   * getSwaptionVolatilityMatrix() is CACHED in a mutable member and the
//     cache is invalidated by setParams(). Both the pre- and post-setParams
//     matrices are emitted, so a port that forgot to drop the cache is caught.
//
//   * setParams() at NON-DEFAULT values: LiborForwardModel::setParams splits
//     the flat parameter array between the volatility model and the
//     correlation model at k = volatilityModel()->params().size(). Getting
//     that split wrong is silent — both halves are still "some numbers" — so
//     the post-setParams S_0 / swaption vol / caplet price are all emitted.
//
//   * discountBondOption for Call AND Put, at several maturities along the
//     process's accrual grid and at strikes on both sides of the money. This
//     is the exact-caplet-pricing path (LiborForwardModel is an AffineModel);
//     it depends on LfmCovarianceProxy::integratedCovariance, on the process
//     initial values, and on the forwarding curve, so a single strike would
//     not separate those.
//
//   * discount(t) and discountBond(now, maturity, factors) — the two
//     "meaningless within this context" AffineModel methods that C++ still
//     implements by delegating to the forwarding curve. They are trivial and
//     therefore exactly the kind of method a port drops.
//
//   * LfmSwaptionEngine NPVs for a grid of co-terminal forward swaptions,
//     payer AND receiver, at a struck rate away from fair so the Black term
//     is not degenerate; plus the engine's rejection of a cash-settled
//     (ParYieldCurve) swaption, and the spread-correction branch
//     (swap->spread() != 0), which is otherwise dead code.
//
// The market setup mirrors test-suite/libormarketmodel.cpp testSwaptionPricing
// (Euribor6M over a two-pillar ZeroCurve 0.04 -> 0.08 to 2011, size 10).
//
// EVALUATION DATE: this probe sets Settings::instance().evaluationDate(). The
// value is emitted as "evaluation_date_serial" and the pytest module MUST pin
// the same date, or every date-derived number below drifts with the wall clock.
//
// Emits JSON on stdout; redirect to references/v143/legacy/lmm_swaption.json.

#include <algorithm>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <ql/exercise.hpp>
#include <ql/indexes/ibor/euribor.hpp>
#include <ql/instruments/swaption.hpp>
#include <ql/instruments/vanillaswap.hpp>
#include <ql/legacy/libormarketmodels/lfmswaptionengine.hpp>
#include <ql/legacy/libormarketmodels/liborforwardmodel.hpp>
#include <ql/legacy/libormarketmodels/lmexpcorrmodel.hpp>
#include <ql/legacy/libormarketmodels/lmlinexpvolmodel.hpp>
#include <ql/math/optimization/constraint.hpp>
#include <ql/models/parameter.hpp>
#include <ql/pricingengines/swap/discountingswapengine.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/yield/zerocurve.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/schedule.hpp>

using namespace QuantLib;

namespace {

void emitArray(const char* key, const std::vector<Real>& v, int indent, bool comma) {
    std::cout << std::string(indent, ' ') << "\"" << key << "\": [";
    for (Size i = 0; i < v.size(); ++i) {
        if (i != 0)
            std::cout << ", ";
        std::cout << v[i];
    }
    std::cout << "]" << (comma ? "," : "") << "\n";
}

void emitArray(const char* key, const Array& a, int indent, bool comma) {
    emitArray(key, std::vector<Real>(a.begin(), a.end()), indent, comma);
}

void emitIntArray(const char* key, const std::vector<long>& v, int indent, bool comma) {
    std::cout << std::string(indent, ' ') << "\"" << key << "\": [";
    for (Size i = 0; i < v.size(); ++i) {
        if (i != 0)
            std::cout << ", ";
        std::cout << v[i];
    }
    std::cout << "]" << (comma ? "," : "") << "\n";
}

void emitMatrix(const char* key, const Matrix& m, int indent, bool comma) {
    std::string pad(indent, ' ');
    std::cout << pad << "\"" << key << "\": [";
    for (Size i = 0; i < m.rows(); ++i) {
        if (i != 0)
            std::cout << ",";
        std::cout << "\n" << pad << "  [";
        for (Size j = 0; j < m.columns(); ++j) {
            if (j != 0)
                std::cout << ", ";
            std::cout << m[i][j];
        }
        std::cout << "]";
    }
    std::cout << "\n" << pad << "]" << (comma ? "," : "") << "\n";
}

void emitScalar(const char* key, Real x, int indent, bool comma) {
    std::cout << std::string(indent, ' ') << "\"" << key << "\": " << x << (comma ? "," : "")
              << "\n";
}

void emitInt(const char* key, long x, int indent, bool comma) {
    std::cout << std::string(indent, ' ') << "\"" << key << "\": " << x << (comma ? "," : "")
              << "\n";
}

const Date kAnchor(4, September, 2005);
const Size kSize = 10;

ext::shared_ptr<IborIndex> makeIndex(std::vector<Date> dates, const std::vector<Rate>& rates) {
    DayCounter dayCounter = Actual360();
    RelinkableHandle<YieldTermStructure> termStructure;
    ext::shared_ptr<IborIndex> index(new Euribor6M(termStructure));

    Date todaysDate = index->fixingCalendar().adjust(kAnchor);
    Settings::instance().evaluationDate() = todaysDate;

    dates[0] = index->fixingCalendar().advance(todaysDate, index->fixingDays(), Days);
    termStructure.linkTo(
        ext::shared_ptr<YieldTermStructure>(new ZeroCurve(dates, rates, dayCounter)));
    return index;
}

// Emit the whole content of a SwaptionVolatilityMatrix: pillars + the
// interpolated read-back the engine actually uses.
void emitVolMatrix(const char* key,
                   const ext::shared_ptr<SwaptionVolatilityMatrix>& vol,
                   bool comma) {
    std::cout << "  \"" << key << "\": {\n";
    emitInt("reference_date_serial", vol->referenceDate().serialNumber(), 4, true);
    {
        std::vector<long> serials;
        for (const auto& d : vol->optionDates())
            serials.push_back(d.serialNumber());
        emitIntArray("option_date_serials", serials, 4, true);
    }
    emitArray("option_times", vol->optionTimes(), 4, true);
    emitArray("swap_lengths", vol->swapLengths(), 4, true);
    {
        std::vector<long> lengths;
        for (const auto& p : vol->swapTenors())
            lengths.push_back(static_cast<long>(p.length()));
        emitIntArray("swap_tenor_lengths_months", lengths, 4, true);
    }
    {
        // v1.43 keeps SwaptionVolatilityMatrix::volatilities_ private, so the
        // pillar grid is recovered by reading the structure at its own
        // (optionTime, swapLength) pillars — bilinear interpolation is exact
        // on the nodes, so this IS the stored matrix.
        Matrix pillars(vol->optionTimes().size(), vol->swapLengths().size());
        for (Size i = 0; i < vol->optionTimes().size(); ++i)
            for (Size j = 0; j < vol->swapLengths().size(); ++j)
                pillars[i][j] =
                    vol->volatility(vol->optionTimes()[i], vol->swapLengths()[j], 0.04, true);
        emitMatrix("volatilities", pillars, 4, true);
    }
    {
        // read-back through the bilinear interpolation, on and off pillar
        const std::vector<Time>& ot = vol->optionTimes();
        const std::vector<Time>& sl = vol->swapLengths();
        std::vector<Real> queriedT, queriedL, read;
        for (Size i = 0; i < ot.size(); ++i) {
            for (Size j = 0; j < sl.size(); ++j) {
                queriedT.push_back(ot[i]);
                queriedL.push_back(sl[j]);
                read.push_back(vol->volatility(ot[i], sl[j], 0.04, true));
            }
        }
        // strictly interior, off-pillar coordinates
        const Time midT = 0.5 * (ot.front() + ot.back());
        const Time midL = 0.5 * (sl.front() + sl.back());
        queriedT.push_back(midT);
        queriedL.push_back(midL);
        read.push_back(vol->volatility(midT, midL, 0.04, true));
        emitArray("query_option_times", queriedT, 4, true);
        emitArray("query_swap_lengths", queriedL, 4, true);
        emitArray("query_volatilities", read, 4, false);
    }
    std::cout << "  }" << (comma ? "," : "") << "\n";
}

} // namespace

int main() {
    try {
        std::cout << std::setprecision(17);

        // test-suite testSwaptionPricing curve: 0.04 -> 0.08 by 2011
        ext::shared_ptr<IborIndex> index =
            makeIndex({{4, September, 2005}, {4, September, 2011}}, {0.04, 0.08});
        const Date evalDate = Settings::instance().evaluationDate();

        ext::shared_ptr<LiborForwardModelProcess> process(
            new LiborForwardModelProcess(kSize, index));

        ext::shared_ptr<LmCorrelationModel> corrModel(
            new LmExponentialCorrelationModel(kSize, 0.5));
        ext::shared_ptr<LmVolatilityModel> volaModel(new LmLinearExponentialVolatilityModel(
            process->fixingTimes(), 0.291, 1.483, 0.116, 0.00001));

        process->setCovarParam(ext::shared_ptr<LfmCovarianceParameterization>(
            new LfmCovarianceProxy(volaModel, corrModel)));

        ext::shared_ptr<LiborForwardModel> model(
            new LiborForwardModel(process, volaModel, corrModel));

        std::cout << "{\n";
        emitInt("evaluation_date_serial", evalDate.serialNumber(), 2, true);

        // ------------------------------------------------------------------
        // setup echo — everything the Python side must rebuild identically
        // ------------------------------------------------------------------
        std::cout << "  \"setup\": {\n";
        emitInt("size", static_cast<long>(process->size()), 4, true);
        emitArray("initial_values", process->initialValues(), 4, true);
        emitArray("fixing_times",
                  std::vector<Real>(process->fixingTimes().begin(), process->fixingTimes().end()),
                  4, true);
        {
            std::vector<long> serials;
            for (const auto& d : process->fixingDates())
                serials.push_back(d.serialNumber());
            emitIntArray("fixing_date_serials", serials, 4, true);
        }
        emitArray("accrual_start_times",
                  std::vector<Real>(process->accrualStartTimes().begin(),
                                    process->accrualStartTimes().end()),
                  4, true);
        emitArray("accrual_end_times",
                  std::vector<Real>(process->accrualEndTimes().begin(),
                                    process->accrualEndTimes().end()),
                  4, true);
        emitInt("n_model_params", static_cast<long>(model->params().size()), 4, true);
        emitArray("model_params", model->params(), 4, false);
        std::cout << "  },\n";

        // ------------------------------------------------------------------
        // LiborForwardModel::S_0 over the whole (alpha, beta) triangle
        // ------------------------------------------------------------------
        {
            std::vector<long> alphas, betas;
            std::vector<Real> s0;
            for (Size alpha = 0; alpha + 1 < kSize; ++alpha) {
                for (Size beta = alpha + 1; beta < kSize; ++beta) {
                    alphas.push_back(static_cast<long>(alpha));
                    betas.push_back(static_cast<long>(beta));
                    s0.push_back(model->S_0(alpha, beta));
                }
            }
            std::cout << "  \"s_0\": {\n";
            emitIntArray("alpha", alphas, 4, true);
            emitIntArray("beta", betas, 4, true);
            emitArray("value", s0, 4, false);
            std::cout << "  },\n";
        }

        // ------------------------------------------------------------------
        // fair forward swap rates the C++ test-suite compares S_0 against
        // ------------------------------------------------------------------
        {
            Calendar calendar = index->fixingCalendar();
            DayCounter dayCounter = index->forwardingTermStructure()->dayCounter();
            BusinessDayConvention convention = index->businessDayConvention();
            Date settlement = index->forwardingTermStructure()->referenceDate();

            std::vector<long> is, js;
            std::vector<Real> fair;
            for (Size i = 1; i < kSize; ++i) {
                for (Size j = 1; j <= kSize - i; ++j) {
                    Date fwdStart = settlement + Period(6 * i, Months);
                    Date fwdMaturity = fwdStart + Period(6 * j, Months);
                    Schedule schedule(fwdStart, fwdMaturity, index->tenor(), calendar, convention,
                                      convention, DateGeneration::Forward, false);
                    VanillaSwap swap(Swap::Receiver, 1.0, schedule, 0.0404, dayCounter, schedule,
                                     index, 0.0, index->dayCounter());
                    swap.setPricingEngine(ext::shared_ptr<PricingEngine>(
                        new DiscountingSwapEngine(index->forwardingTermStructure())));
                    is.push_back(static_cast<long>(i));
                    js.push_back(static_cast<long>(j));
                    fair.push_back(swap.fairRate());
                }
            }
            std::cout << "  \"forward_swap_fair_rates\": {\n";
            emitIntArray("i", is, 4, true);
            emitIntArray("j", js, 4, true);
            emitArray("fair_rate", fair, 4, false);
            std::cout << "  },\n";
        }

        // ------------------------------------------------------------------
        // AffineModel surface: discount / discountBond / discountBondOption
        // ------------------------------------------------------------------
        {
            std::vector<Real> ts = {0.0, 0.5, 1.25, 3.0, 4.75};
            std::vector<Real> disc, discBond;
            Array dummyFactors(2, 0.7);
            for (Real t : ts) {
                disc.push_back(model->discount(t));
                discBond.push_back(model->discountBond(0.3, t, dummyFactors));
            }
            std::cout << "  \"affine\": {\n";
            emitArray("times", ts, 4, true);
            emitArray("discount", disc, 4, true);
            emitArray("discount_bond", discBond, 4, false);
            std::cout << "  },\n";
        }
        {
            // discountBondOption: the exact caplet price. maturity must sit on
            // an accrualStartTime and bondMaturity on the matching
            // accrualEndTime, else C++ QL_REQUIREs.
            const std::vector<Time>& ast = process->accrualStartTimes();
            const std::vector<Time>& aet = process->accrualEndTimes();
            std::vector<long> idx;
            std::vector<Real> strikes, calls, puts;
            for (Size i : {Size(0), Size(3), Size(6), Size(9)}) {
                // strike is a bond-option strike: capRate = (1/K - 1)/tenor
                // must stay non-negative, so K <= 1.
                for (Real k : {0.94, 0.96, 0.98, 0.995}) {
                    idx.push_back(static_cast<long>(i));
                    strikes.push_back(k);
                    calls.push_back(
                        model->discountBondOption(Option::Call, k, ast[i], aet[i]));
                    puts.push_back(model->discountBondOption(Option::Put, k, ast[i], aet[i]));
                }
            }
            std::cout << "  \"discount_bond_option\": {\n";
            emitIntArray("index", idx, 4, true);
            emitArray("strike", strikes, 4, true);
            emitArray("call", calls, 4, true);
            emitArray("put", puts, 4, false);
            std::cout << "  },\n";
        }

        // ------------------------------------------------------------------
        // Rebonato swaption volatility matrix (cached)
        // ------------------------------------------------------------------
        ext::shared_ptr<SwaptionVolatilityMatrix> swaptionVol =
            model->getSwaptionVolatilityMatrix();
        emitVolMatrix("swaption_vol_matrix", swaptionVol, true);
        // second call must return the very same cached object
        emitScalar("swaption_vol_matrix_is_cached",
                   model->getSwaptionVolatilityMatrix() == swaptionVol ? 1.0 : 0.0, 2, true);

        // ------------------------------------------------------------------
        // LfmSwaptionEngine
        // ------------------------------------------------------------------
        {
            Calendar calendar = index->fixingCalendar();
            DayCounter dayCounter = index->forwardingTermStructure()->dayCounter();
            BusinessDayConvention convention = index->businessDayConvention();
            Date settlement = index->forwardingTermStructure()->referenceDate();

            ext::shared_ptr<PricingEngine> engine(
                new LfmSwaptionEngine(model, index->forwardingTermStructure()));

            std::vector<long> is, js;
            std::vector<Real> fairRates, payerNpv, receiverNpv, atmPayerNpv;
            for (Size i = 1; i <= kSize / 2; ++i) {
                for (Size j = 1; j <= kSize / 2; ++j) {
                    Date fwdStart = settlement + Period(6 * i, Months);
                    Date fwdMaturity = fwdStart + Period(6 * j, Months);
                    Schedule schedule(fwdStart, fwdMaturity, index->tenor(), calendar, convention,
                                      convention, DateGeneration::Forward, false);

                    auto probe = ext::make_shared<VanillaSwap>(
                        Swap::Receiver, 1.0, schedule, 0.0404, dayCounter, schedule, index, 0.0,
                        index->dayCounter());
                    probe->setPricingEngine(ext::shared_ptr<PricingEngine>(
                        new DiscountingSwapEngine(index->forwardingTermStructure())));
                    const Rate fair = probe->fairRate();

                    // struck AWAY from fair so the Black term is not degenerate
                    const Rate struck = fair + 0.0035;

                    ext::shared_ptr<Exercise> exercise(
                        new EuropeanExercise(process->fixingDates()[i]));

                    auto payerSwap = ext::make_shared<VanillaSwap>(
                        Swap::Payer, 1.0, schedule, struck, dayCounter, schedule, index, 0.0,
                        index->dayCounter());
                    payerSwap->setPricingEngine(ext::shared_ptr<PricingEngine>(
                        new DiscountingSwapEngine(index->forwardingTermStructure())));
                    auto payerSwaption = ext::make_shared<Swaption>(payerSwap, exercise);
                    payerSwaption->setPricingEngine(engine);

                    auto recSwap = ext::make_shared<VanillaSwap>(
                        Swap::Receiver, 1.0, schedule, struck, dayCounter, schedule, index, 0.0,
                        index->dayCounter());
                    recSwap->setPricingEngine(ext::shared_ptr<PricingEngine>(
                        new DiscountingSwapEngine(index->forwardingTermStructure())));
                    auto recSwaption = ext::make_shared<Swaption>(recSwap, exercise);
                    recSwaption->setPricingEngine(engine);

                    auto atmSwap = ext::make_shared<VanillaSwap>(
                        Swap::Payer, 1.0, schedule, fair, dayCounter, schedule, index, 0.0,
                        index->dayCounter());
                    atmSwap->setPricingEngine(ext::shared_ptr<PricingEngine>(
                        new DiscountingSwapEngine(index->forwardingTermStructure())));
                    auto atmSwaption = ext::make_shared<Swaption>(atmSwap, exercise);
                    atmSwaption->setPricingEngine(engine);

                    is.push_back(static_cast<long>(i));
                    js.push_back(static_cast<long>(j));
                    fairRates.push_back(fair);
                    payerNpv.push_back(payerSwaption->NPV());
                    receiverNpv.push_back(recSwaption->NPV());
                    atmPayerNpv.push_back(atmSwaption->NPV());
                }
            }
            std::cout << "  \"lfm_swaption_engine\": {\n";
            emitIntArray("i", is, 4, true);
            emitIntArray("j", js, 4, true);
            emitArray("fair_rate", fairRates, 4, true);
            emitArray("payer_npv", payerNpv, 4, true);
            emitArray("receiver_npv", receiverNpv, 4, true);
            emitArray("atm_payer_npv", atmPayerNpv, 4, false);
            std::cout << "  },\n";

            // spread-correction branch: swap->spread() != 0 shifts BOTH the
            // struck rate and the fair rate by
            //   spread * |floatingLegBPS / fixedLegBPS|
            {
                Date fwdStart = settlement + Period(6 * 2, Months);
                Date fwdMaturity = fwdStart + Period(6 * 3, Months);
                Schedule schedule(fwdStart, fwdMaturity, index->tenor(), calendar, convention,
                                  convention, DateGeneration::Forward, false);
                auto spreadSwap = ext::make_shared<VanillaSwap>(
                    Swap::Payer, 1.0, schedule, 0.0425, dayCounter, schedule, index, 0.0017,
                    index->dayCounter());
                spreadSwap->setPricingEngine(ext::shared_ptr<PricingEngine>(
                    new DiscountingSwapEngine(index->forwardingTermStructure())));
                ext::shared_ptr<Exercise> exercise(
                    new EuropeanExercise(process->fixingDates()[2]));
                auto sw = ext::make_shared<Swaption>(spreadSwap, exercise);
                sw->setPricingEngine(engine);

                std::cout << "  \"lfm_swaption_with_spread\": {\n";
                emitScalar("spread", spreadSwap->spread(), 4, true);
                emitScalar("fixed_rate", spreadSwap->fixedRate(), 4, true);
                emitScalar("fair_rate", spreadSwap->fairRate(), 4, true);
                emitScalar("fixed_leg_bps", spreadSwap->fixedLegBPS(), 4, true);
                emitScalar("floating_leg_bps", spreadSwap->floatingLegBPS(), 4, true);
                emitScalar("npv", sw->NPV(), 4, false);
                std::cout << "  },\n";
            }

            // cash-settled (ParYieldCurve) must be rejected
            {
                Date fwdStart = settlement + Period(6 * 2, Months);
                Date fwdMaturity = fwdStart + Period(6 * 2, Months);
                Schedule schedule(fwdStart, fwdMaturity, index->tenor(), calendar, convention,
                                  convention, DateGeneration::Forward, false);
                auto swap = ext::make_shared<VanillaSwap>(Swap::Payer, 1.0, schedule, 0.0425,
                                                          dayCounter, schedule, index, 0.0,
                                                          index->dayCounter());
                ext::shared_ptr<Exercise> cashExercise(
                    new EuropeanExercise(process->fixingDates()[2]));
                auto sw = ext::make_shared<Swaption>(swap, cashExercise, Settlement::Cash,
                                                     Settlement::ParYieldCurve);
                sw->setPricingEngine(engine);
                std::cout << "  \"lfm_swaption_par_yield_rejected\": ";
                try {
                    sw->NPV();
                    std::cout << "false,\n";
                } catch (const std::exception&) {
                    std::cout << "true,\n";
                }
            }
        }

        // ------------------------------------------------------------------
        // setParams: splits the flat array between vol and corr models and
        // must invalidate the cached swaption vol matrix
        // ------------------------------------------------------------------
        {
            Array p = model->params();
            // vol model has 4 params (a, b, c, d); corr model has 1 (rho).
            Array np(p.size());
            np[0] = 0.35;   // a
            np[1] = 1.10;   // b
            np[2] = 0.145;  // c
            np[3] = 0.0175; // d
            np[4] = 0.28;   // rho
            model->setParams(np);

            std::cout << "  \"after_set_params\": {\n";
            emitArray("params", model->params(), 4, true);
            emitArray("vol_model_params",
                      std::vector<Real>{volaModel->params()[0](0.0), volaModel->params()[1](0.0),
                                        volaModel->params()[2](0.0), volaModel->params()[3](0.0)},
                      4, true);
            emitArray("corr_model_params", std::vector<Real>{corrModel->params()[0](0.0)}, 4,
                      true);
            {
                std::vector<long> alphas, betas;
                std::vector<Real> s0;
                for (Size alpha = 0; alpha + 1 < kSize; ++alpha) {
                    for (Size beta = alpha + 1; beta < kSize; ++beta) {
                        alphas.push_back(static_cast<long>(alpha));
                        betas.push_back(static_cast<long>(beta));
                        s0.push_back(model->S_0(alpha, beta));
                    }
                }
                // S_0 depends only on the initial forward curve, so it must be
                // UNCHANGED — that is itself worth pinning.
                emitArray("s_0", s0, 4, true);
            }
            ext::shared_ptr<SwaptionVolatilityMatrix> vol2 =
                model->getSwaptionVolatilityMatrix();
            emitScalar("cache_was_invalidated", vol2 == swaptionVol ? 0.0 : 1.0, 4, true);
            {
                Matrix pillars2(vol2->optionTimes().size(), vol2->swapLengths().size());
                for (Size i = 0; i < vol2->optionTimes().size(); ++i)
                    for (Size j = 0; j < vol2->swapLengths().size(); ++j)
                        pillars2[i][j] = vol2->volatility(vol2->optionTimes()[i],
                                                          vol2->swapLengths()[j], 0.04, true);
                emitMatrix("swaption_volatilities", pillars2, 4, false);
            }
            std::cout << "  }\n";
        }

        std::cout << "}\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "probe failed: " << e.what() << std::endl;
        return 1;
    }
}

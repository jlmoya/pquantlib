// migration-harness/cpp/probes/v143_models_volatility/probe.cpp
//
// Reference values for the ql/models/volatility/ family and the two root-level
// abstract bases it implements (ql/volatilitymodel.hpp, v1.43):
//
//   LocalVolatilityEstimator<T>, VolatilityCompositor          (volatilitymodel.hpp)
//   ConstantEstimator                                          (constantestimator.{hpp,cpp})
//   SimpleLocalEstimator                                       (simplelocalestimator.hpp)
//   GarmanKlassAbstract, GarmanKlassSimpleSigma,
//   GarmanKlassOpenClose<T>, GarmanKlassSigma1, ParkinsonSigma,
//   GarmanKlassSigma3, GarmanKlassSigma4, GarmanKlassSigma5,
//   GarmanKlassSigma6                                          (garmanklass.hpp)
//   Garch11                                                    (garch.{hpp,cpp})
//
// NO EVALUATION DATE IS SET. Nothing in this family reads
// Settings::instance().evaluationDate(): the estimators are pure functions of a
// TimeSeries and Garch11 only ever does Date arithmetic on the series' own keys
// (garch.cpp:388 extrapolates one step past the last key). The pytest module
// therefore does NOT need an ObservableSettings fixture, and that absence is a
// deliberate, verified fact rather than an oversight.
//
// What is pinned and why:
//
//   * The FULL output TimeSeries of every estimator — every date serial and
//     every value — not a summary. A summary (first/last/mean) cannot
//     distinguish an off-by-one in which date an estimate is filed under, and
//     ConstantEstimator (constantestimator.cpp:28-41) and
//     GarmanKlassOpenClose::calculate (garmanklass.hpp:86-103) are exactly the
//     two places where that off-by-one lives: the former advances the output
//     cursor by `size_` before the loop, the latter skips the first date.
//
//   * A hand-written OHLC table with all four of open/close/high/low DISTINCT
//     on every bar, and no two bars sharing a shape. Each Garman-Klass sigma
//     reads a different subset of the four fields
//     (SimpleSigma: O,C; Parkinson: O,H,L; Sigma4/Sigma5: O,C,H,L), so a
//     formula that silently dropped one field would still reproduce a table
//     where, say, high == close.
//
//   * A second, DEGENERATE OHLC table with high == low == open != close. There
//     GarmanKlassSigma4 and GarmanKlassSigma5 return a NEGATIVE point value and
//     GarmanKlassAbstract::calculate (garmanklass.hpp:55) takes std::fabs of it
//     before the sqrt. That fabs is unreachable with a consistent bar
//     (H >= max(O,C), L <= min(O,C) forces (u-d)^2 >= c^2), so without this
//     table the port could drop it and stay green. The degenerate table is fed
//     ONLY to the non-OpenClose estimators: GarmanKlassOpenClose::calculate
//     (garmanklass.hpp:100) has NO fabs, so a negative sigma2 there is a NaN,
//     which is not representable in JSON.
//
//   * ConstantEstimator at size 1 AND size 3. Its normalisation is
//     sqrt(sumu2/size - sumu^2/size/(size+1)) — note the (size+1), NOT size or
//     size-1, so it is neither the biased nor the unbiased sample variance.
//     Two window widths pin that denominator; one does not.
//
//   * Garch11: for EACH of the four Mode values, both the initial guess alone
//     (via a DummyOptimizationMethod that evaluates the cost at the guess and
//     stops, mirroring test-suite/garch.cpp:36-42) and the fully Simplex-
//     optimized result. Splitting the two matters because the initial-guess
//     grid (garch.cpp:230-369, initialGuess1/initialGuess2) is deterministic
//     while the Simplex path is not: if the port's optimizer diverges, the
//     guess block still localises the failure.
//
//   * Garch11 intermediates that the optimizer sits on top of: mean_r2 from
//     to_r2 (garch.hpp:157-168 — note the running weight w /= (w+1), which is
//     NOT a plain mean), and the autocovariance vector computed exactly as
//     calibrate_r2 computes it (garch.cpp:416-421: maxLag = (Size)sqrt(n), the
//     series centered by mean_r2 first). Those two are pure arithmetic, so they
//     are pinnable at TIGHT even when the optimizer's answer is not.
//
//   * Garch11::costFunction at three FIXED (alpha, beta, omega) triples. The
//     static cost function is the objective the optimizer minimizes; pinning it
//     independently of any optimization proves the objective before blaming the
//     search.
//
//   * Garch11::calculate on the test-suite's own scenario
//     (test-suite/garch.cpp:169-183: a constant 0.1 series, 10 days from
//     7 July 1962, model (0.2, 0.3, 0.4)) so the emitted numbers can be eyeballed
//     against the upstream expected_calc table, and on the calibrated model.
//     Both include the SYNTHETIC trailing date garch.cpp:388 appends by
//     extrapolating the last date gap — output has one more point than input.
//
//   * Garch11's ctor argument is the long-term VARIANCE vl, and omega() returns
//     vl * (1 - alpha - beta) (garch.hpp:56-70). A port that took the third ctor
//     argument as omega would be wrong; forecast() and calculate() are emitted
//     for the (0.2, 0.3, 0.4) model so the distinction is observable.
//
// The Garch11 input series is generated here with MersenneTwisterUniformRng(48)
// + InverseCumulativeNormal (the upstream test-suite's generator) and is EMITTED
// INTO THE JSON at 17 significant digits, so the Python side reads back the
// identical doubles instead of having to reproduce the RNG. The length is 400,
// not the test-suite's 50000, because the pure-Python cost function is a
// sequential recursion that cannot be vectorised.
//
// Emits JSON on stdout; redirect to references/v143/models/volatility.json.

#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <ql/math/autocovariance.hpp>
#include <ql/math/distributions/normaldistribution.hpp>
#include <ql/math/optimization/endcriteria.hpp>
#include <ql/math/optimization/problem.hpp>
#include <ql/math/randomnumbers/inversecumulativerng.hpp>
#include <ql/math/randomnumbers/mt19937uniformrng.hpp>
#include <ql/models/volatility/constantestimator.hpp>
#include <ql/models/volatility/garch.hpp>
#include <ql/models/volatility/garmanklass.hpp>
#include <ql/models/volatility/simplelocalestimator.hpp>
#include <ql/prices.hpp>
#include <ql/timeseries.hpp>

using namespace QuantLib;

namespace {

// --- shared parameters -------------------------------------------------------

const Real kYearFraction = 1.0 / 252.0;
const Real kMarketOpenFraction = 0.3;

// --- OHLC table: every bar has four DISTINCT fields --------------------------

const Date kOhlcStart(2, January, 2024);

const Real kOpen[] = {100.00, 101.30, 100.20, 103.10, 102.50, 105.00, 104.40, 104.85, 107.10};
const Real kClose[] = {101.25, 100.10, 103.40, 102.60, 105.10, 104.20, 104.90, 107.30, 106.15};
const Real kHigh[] = {102.10, 101.90, 103.80, 104.20, 105.60, 106.10, 105.30, 107.90, 107.55};
const Real kLow[] = {99.40, 99.75, 100.05, 101.90, 102.30, 103.75, 103.20, 104.10, 105.80};
const std::size_t kNBars = sizeof(kOpen) / sizeof(kOpen[0]);

// Degenerate table: high == low == open, close moves. Drives
// GarmanKlassSigma4/Sigma5 calculatePoint NEGATIVE so the fabs in
// GarmanKlassAbstract::calculate is exercised.
const Date kDegenerateStart(5, February, 2024);
const Real kDegOpen[] = {100.00, 100.70, 100.20, 101.40};
const Real kDegClose[] = {100.70, 100.20, 101.40, 100.90};
const Real kDegHigh[] = {100.00, 100.70, 100.20, 101.40};
const Real kDegLow[] = {100.00, 100.70, 100.20, 101.40};
const std::size_t kNDegBars = sizeof(kDegOpen) / sizeof(kDegOpen[0]);

// --- hand-written volatility series for ConstantEstimator --------------------

const Date kVolStart(4, March, 2024);
const Real kVols[] = {0.1250, 0.0930, 0.1610, 0.1075, 0.2040,
                      0.0885, 0.1420, 0.1755, 0.1005, 0.1330};
const std::size_t kNVols = sizeof(kVols) / sizeof(kVols[0]);

// --- Garch11 -----------------------------------------------------------------

typedef InverseCumulativeRng<MersenneTwisterUniformRng, InverseCumulativeNormal> GaussianGenerator;

const std::size_t kGarchN = 400;
const Date kGarchStart(7, July, 1962);
const Real kGarchAlpha = 0.2;
const Real kGarchBeta = 0.3;
const Real kGarchVl = 0.4;

// test-suite/garch.cpp:36-42 — evaluates the cost at the current point and
// stops, so the optimizer leaves the initial guess untouched.
class DummyOptimizationMethod : public OptimizationMethod {
  public:
    EndCriteria::Type minimize(Problem& P, const EndCriteria&) override {
        P.setFunctionValue(P.value(P.currentValue()));
        return EndCriteria::None;
    }
};

// --- emit helpers ------------------------------------------------------------

void emitReals(const char* key, const std::vector<Real>& v, int indent, bool comma) {
    const std::string pad(indent, ' ');
    std::cout << pad << "\"" << key << "\": [";
    for (std::size_t i = 0; i < v.size(); ++i)
        std::cout << (i != 0 ? ", " : "") << v[i];
    std::cout << "]" << (comma ? "," : "") << "\n";
}

void emitSerials(const char* key, const std::vector<BigInteger>& v, int indent, bool comma) {
    const std::string pad(indent, ' ');
    std::cout << pad << "\"" << key << "\": [";
    for (std::size_t i = 0; i < v.size(); ++i)
        std::cout << (i != 0 ? ", " : "") << v[i];
    std::cout << "]" << (comma ? "," : "") << "\n";
}

// Full series: every date serial and every value, in map (i.e. date) order.
void emitSeries(const char* key, const TimeSeries<Volatility>& ts, int indent, bool comma) {
    const std::string pad(indent, ' ');
    std::vector<BigInteger> serials;
    std::vector<Real> values;
    for (auto it = ts.cbegin(); it != ts.cend(); ++it) {
        serials.push_back(it->first.serialNumber());
        values.push_back(it->second);
    }
    std::cout << pad << "\"" << key << "\": {\n";
    emitSerials("date_serials", serials, indent + 2, true);
    emitReals("values", values, indent + 2, false);
    std::cout << pad << "}" << (comma ? "," : "") << "\n";
}

TimeSeries<IntervalPrice> makeOhlc(const Date& start,
                                   const Real* open,
                                   const Real* close,
                                   const Real* high,
                                   const Real* low,
                                   std::size_t n) {
    std::vector<Date> dates;
    Date d = start;
    for (std::size_t i = 0; i < n; ++i, d += 1)
        dates.push_back(d);
    return IntervalPrice::makeSeries(dates,
                                     std::vector<Real>(open, open + n),
                                     std::vector<Real>(close, close + n),
                                     std::vector<Real>(high, high + n),
                                     std::vector<Real>(low, low + n));
}

void emitEstimators() {
    const TimeSeries<IntervalPrice> ohlc =
        makeOhlc(kOhlcStart, kOpen, kClose, kHigh, kLow, kNBars);
    const TimeSeries<IntervalPrice> degenerate =
        makeOhlc(kDegenerateStart, kDegOpen, kDegClose, kDegHigh, kDegLow, kNDegBars);

    std::cout << "  \"ohlc\": {\n";
    {
        std::vector<BigInteger> serials;
        for (auto it = ohlc.cbegin(); it != ohlc.cend(); ++it)
            serials.push_back(it->first.serialNumber());
        emitSerials("date_serials", serials, 4, true);
        emitReals("open", std::vector<Real>(kOpen, kOpen + kNBars), 4, true);
        emitReals("close", std::vector<Real>(kClose, kClose + kNBars), 4, true);
        emitReals("high", std::vector<Real>(kHigh, kHigh + kNBars), 4, true);
        emitReals("low", std::vector<Real>(kLow, kLow + kNBars), 4, false);
    }
    std::cout << "  },\n";

    std::cout << "  \"ohlc_degenerate\": {\n";
    {
        std::vector<BigInteger> serials;
        for (auto it = degenerate.cbegin(); it != degenerate.cend(); ++it)
            serials.push_back(it->first.serialNumber());
        emitSerials("date_serials", serials, 4, true);
        emitReals("open", std::vector<Real>(kDegOpen, kDegOpen + kNDegBars), 4, true);
        emitReals("close", std::vector<Real>(kDegClose, kDegClose + kNDegBars), 4, true);
        emitReals("high", std::vector<Real>(kDegHigh, kDegHigh + kNDegBars), 4, true);
        emitReals("low", std::vector<Real>(kDegLow, kDegLow + kNDegBars), 4, false);
    }
    std::cout << "  },\n";

    std::cout << "  \"year_fraction\": " << kYearFraction << ",\n"
              << "  \"market_open_fraction\": " << kMarketOpenFraction << ",\n";

    // --- ConstantEstimator ---------------------------------------------------
    {
        std::vector<Date> dates;
        std::vector<Real> vols(kVols, kVols + kNVols);
        Date d = kVolStart;
        for (std::size_t i = 0; i < kNVols; ++i, d += 1)
            dates.push_back(d);
        TimeSeries<Volatility> vs(dates.begin(), dates.end(), vols.begin());

        std::vector<BigInteger> serials;
        for (std::size_t i = 0; i < dates.size(); ++i)
            serials.push_back(dates[i].serialNumber());

        std::cout << "  \"vol_series\": {\n";
        emitSerials("date_serials", serials, 4, true);
        emitReals("values", vols, 4, false);
        std::cout << "  },\n";

        ConstantEstimator ce1(1);
        ConstantEstimator ce3(3);
        emitSeries("constant_estimator_size1", ce1.calculate(vs), 2, true);
        emitSeries("constant_estimator_size3", ce3.calculate(vs), 2, true);

        // calibrate() is a documented no-op (constantestimator.hpp:42): the
        // result of calculate() must be unchanged after calling it.
        ce3.calibrate(vs);
        emitSeries("constant_estimator_size3_after_calibrate", ce3.calculate(vs), 2, true);
    }

    // --- SimpleLocalEstimator ------------------------------------------------
    {
        const TimeSeries<Real> closes =
            IntervalPrice::extractComponent(ohlc, IntervalPrice::Close);
        SimpleLocalEstimator sle(kYearFraction);
        emitSeries("simple_local_estimator", sle.calculate(closes), 2, true);

        SimpleLocalEstimator sle2(4.0);
        emitSeries("simple_local_estimator_y4", sle2.calculate(closes), 2, true);
    }

    // --- Garman-Klass family -------------------------------------------------
    {
        GarmanKlassSimpleSigma simple(kYearFraction);
        ParkinsonSigma parkinson(kYearFraction);
        GarmanKlassSigma4 sigma4(kYearFraction);
        GarmanKlassSigma5 sigma5(kYearFraction);
        GarmanKlassSigma1 sigma1(kYearFraction, kMarketOpenFraction);
        GarmanKlassSigma3 sigma3(kYearFraction, kMarketOpenFraction);
        GarmanKlassSigma6 sigma6(kYearFraction, kMarketOpenFraction);

        emitSeries("garman_klass_simple_sigma", simple.calculate(ohlc), 2, true);
        emitSeries("parkinson_sigma", parkinson.calculate(ohlc), 2, true);
        emitSeries("garman_klass_sigma4", sigma4.calculate(ohlc), 2, true);
        emitSeries("garman_klass_sigma5", sigma5.calculate(ohlc), 2, true);
        emitSeries("garman_klass_sigma1", sigma1.calculate(ohlc), 2, true);
        emitSeries("garman_klass_sigma3", sigma3.calculate(ohlc), 2, true);
        emitSeries("garman_klass_sigma6", sigma6.calculate(ohlc), 2, true);

        // Degenerate bars: only the fabs-protected estimators.
        GarmanKlassSimpleSigma dSimple(kYearFraction);
        ParkinsonSigma dParkinson(kYearFraction);
        GarmanKlassSigma4 dSigma4(kYearFraction);
        GarmanKlassSigma5 dSigma5(kYearFraction);
        emitSeries("degenerate_garman_klass_simple_sigma", dSimple.calculate(degenerate), 2, true);
        emitSeries("degenerate_parkinson_sigma", dParkinson.calculate(degenerate), 2, true);
        emitSeries("degenerate_garman_klass_sigma4", dSigma4.calculate(degenerate), 2, true);
        emitSeries("degenerate_garman_klass_sigma5", dSigma5.calculate(degenerate), 2, true);
    }
}

void emitGarchCalibration(const char* key,
                          const Garch11& g,
                          int indent,
                          bool comma) {
    const std::string pad(indent, ' ');
    std::cout << pad << "\"" << key << "\": {\n"
              << pad << "  \"alpha\": " << g.alpha() << ",\n"
              << pad << "  \"beta\": " << g.beta() << ",\n"
              << pad << "  \"omega\": " << g.omega() << ",\n"
              << pad << "  \"lt_vol\": " << g.ltVol() << ",\n"
              << pad << "  \"log_likelihood\": " << g.logLikelihood() << ",\n"
              << pad << "  \"mode\": " << static_cast<int>(g.mode()) << "\n"
              << pad << "}" << (comma ? "," : "") << "\n";
}

void emitGarch() {
    // --- generate the input series -------------------------------------------
    TimeSeries<Volatility> ts;
    std::vector<Real> rs;
    {
        Garch11 generator(kGarchAlpha, kGarchBeta, kGarchVl);
        GaussianGenerator rng((MersenneTwisterUniformRng(48)));
        Volatility r = 0.0, v = 0.0;
        Date d = kGarchStart;
        for (std::size_t i = 0; i < kGarchN; ++i, d += 1) {
            v = generator.forecast(r, v);
            r = rng.next().value * std::sqrt(v);
            ts[d] = r;
            rs.push_back(r);
        }
    }

    std::cout << "  \"garch\": {\n";
    std::cout << "    \"start_serial\": " << kGarchStart.serialNumber() << ",\n";
    emitReals("series", rs, 4, true);

    // --- the generating model (0.2, 0.3, 0.4) --------------------------------
    {
        Garch11 g(kGarchAlpha, kGarchBeta, kGarchVl);
        emitGarchCalibration("generator", g, 4, true);

        // forecast(r, sigma2) at scattered points.
        const Real fr[] = {0.0, 0.1, -0.25, 0.5, 0.0};
        const Real fs[] = {0.0, 0.04, 0.09, 1.0, 0.36};
        std::vector<Real> fw;
        for (std::size_t i = 0; i < 5; ++i)
            fw.push_back(g.forecast(fr[i], fs[i]));
        emitReals("forecast_r", std::vector<Real>(fr, fr + 5), 4, true);
        emitReals("forecast_sigma2", std::vector<Real>(fs, fs + 5), 4, true);
        emitReals("forecast_generator", fw, 4, true);

        // Multi-horizon chain: sigma2_{k+1} = forecast(r_const, sigma2_k).
        std::vector<Real> chain;
        Real sigma2 = 0.0;
        for (std::size_t k = 0; k < 6; ++k) {
            sigma2 = g.forecast(0.15, sigma2);
            chain.push_back(sigma2);
        }
        emitReals("forecast_chain_generator", chain, 4, true);

        // test-suite/garch.cpp:169-183 — constant 0.1 over 10 days.
        TimeSeries<Volatility> flat;
        Date d(7, July, 1962);
        for (std::size_t i = 0; i < 10; ++i, d += 1)
            flat[d] = 0.1;
        emitSeries("calculate_flat", g.calculate(flat), 4, true);
    }

    // --- to_r2 / autocovariances (the optimizer's inputs) --------------------
    std::vector<Volatility> r2;
    Real mean_r2 = Garch11::to_r2(rs.begin(), rs.end(), r2);
    std::cout << "    \"mean_r2\": " << mean_r2 << ",\n";
    {
        // garch.cpp:416-421 verbatim.
        Real dataSize = Real(r2.size());
        Size maxLag = (Size)std::sqrt(dataSize);
        Array acf(maxLag + 1);
        std::vector<Volatility> tmp(r2.size());
        std::transform(r2.begin(), r2.end(), tmp.begin(),
                       [=](Real x) -> Real { return x - mean_r2; });
        autocovariances(tmp.begin(), tmp.end(), acf.begin(), maxLag);
        std::cout << "    \"max_lag\": " << maxLag << ",\n";
        emitReals("acf", std::vector<Real>(acf.begin(), acf.end()), 4, true);
    }

    // --- static costFunction at fixed parameter triples -----------------------
    {
        const Real ca[] = {0.2, 0.1, 0.207592};
        const Real cb[] = {0.3, 0.85, 0.281979};
        const Real co[] = {0.2, 0.05, 0.204647};
        std::vector<Real> cost;
        for (std::size_t i = 0; i < 3; ++i)
            cost.push_back(Garch11::costFunction(rs.begin(), rs.end(), ca[i], cb[i], co[i]));
        emitReals("cost_alpha", std::vector<Real>(ca, ca + 3), 4, true);
        emitReals("cost_beta", std::vector<Real>(cb, cb + 3), 4, true);
        emitReals("cost_omega", std::vector<Real>(co, co + 3), 4, true);
        emitReals("cost_values", cost, 4, true);
    }

    // --- per-Mode calibration -------------------------------------------------
    const Garch11::Mode modes[] = {Garch11::MomentMatchingGuess, Garch11::GammaGuess,
                                   Garch11::BestOfTwo, Garch11::DoubleOptimization};
    const char* guessKeys[] = {"guess_moment_matching", "guess_gamma", "guess_best_of_two",
                               "guess_double_optimization"};
    const char* calibKeys[] = {"calibrated_moment_matching", "calibrated_gamma",
                               "calibrated_best_of_two", "calibrated_double_optimization"};

    for (std::size_t i = 0; i < 4; ++i) {
        // Full Simplex(0.001) + EndCriteria(10000, 500, 1e-8, 1e-8, 1e-8) path
        // that the ctor and the no-method calibrate() overload use.
        Garch11 g(ts, modes[i]);
        emitGarchCalibration(calibKeys[i], g, 4, true);

        // Initial guess alone: no optimization past evaluating the cost there.
        DummyOptimizationMethod dummy;
        Garch11 gg(ts, modes[i]);
        gg.calibrate(ts, dummy, EndCriteria(3, 2, 0.0, 0.0, 0.0));
        emitGarchCalibration(guessKeys[i], gg, 4, true);
    }

    // --- calculate() with a calibrated model ----------------------------------
    {
        Garch11 g(ts, Garch11::BestOfTwo);

        TimeSeries<Volatility> head;
        Date d = kGarchStart;
        for (std::size_t i = 0; i < 12; ++i, d += 1)
            head[d] = rs[i];
        emitSeries("calculate_calibrated_head", g.calculate(head), 4, true);

        std::vector<Real> chain;
        Real sigma2 = 0.0;
        for (std::size_t k = 0; k < 6; ++k) {
            sigma2 = g.forecast(0.15, sigma2);
            chain.push_back(sigma2);
        }
        emitReals("forecast_chain_calibrated", chain, 4, false);
    }

    std::cout << "  }\n";
}

}  // namespace

int main() {
    try {
        std::cout << std::setprecision(17);
        std::cout << "{\n";
        emitEstimators();
        emitGarch();
        std::cout << "}\n";
    } catch (const std::exception& e) {
        std::cerr << "probe failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

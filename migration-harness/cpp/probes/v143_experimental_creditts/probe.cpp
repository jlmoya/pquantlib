// migration-harness/cpp/probes/v143_experimental_creditts/probe.cpp
//
// Reference values for the ql/experimental/credit "term-structure + engine"
// gap cluster @ v1.43:
//
//   A  NorthAmericaCorpDefaultKey          defaultprobabilitykey.hpp:71
//   B  OneFactorGaussianStudentCopula      onefactorstudentcopula.hpp:108
//      OneFactorStudentGaussianCopula      onefactorstudentcopula.hpp:161
//      OneFactorStudentCopula              onefactorstudentcopula.hpp:53
//        (the double-Student one is already "ported" in PQuantLib against a
//         hand-rolled 1-D quadrature; block B exists to falsify that.)
//   C  OneFactorAffineSurvivalStructure    onefactoraffinesurvival.hpp:41
//   D  InterpolatedAffineHazardRateCurve   interpolatedaffinehazardratecurve.hpp:61
//      AffineHazardRate (traits)           interpolatedaffinehazardratecurve.hpp:152
//   E  AssetSwapHelper                     riskyassetswap.hpp:89
//   F  CdsOption / BlackCdsOptionEngine    cdsoption.hpp:43 / blackcdsoptionengine.hpp:35
//   G  IntegralNtdEngine                   integralntdengine.hpp:31
//
// WHY BLOCK G LOOKS THE WAY IT DOES
// ---------------------------------
// IntegralNtdEngine consumes exactly five things from the Basket:
// probAtLeastNEvents(n, d), claim()->amount(d, notional, recoveryRate(d,0)),
// recoveryRate(d, i), remainingNotional() and the premium leg. Everything
// else in the engine is date arithmetic and discounting. Rather than require
// the Python port to first reproduce ConstantLossModel<GaussianCopulaPolicy>
// + Gaussian quadrature (a different gap cluster), block G pins the *engine
// inputs* — the full probAtLeastNEvents(n, d) trace over the exact
// integration grid the engine walks — alongside the engine outputs. The
// Python test replays the pinned probabilities into the Python engine, so the
// engine arithmetic is cross-validated against C++ term by term. The
// underlying basket/loss-model numbers are pinned too, so a later port of the
// latent model can be checked against the same JSON.
//
// Emits JSON on stdout; redirect to
// references/v143/experimental/creditts.json.

#include <ql/currencies/america.hpp>
#include <ql/currencies/europe.hpp>
#include <ql/exercise.hpp>
#include <ql/cashflows/fixedratecoupon.hpp>
#include <ql/experimental/credit/basket.hpp>
#include <ql/experimental/credit/blackcdsoptionengine.hpp>
#include <ql/experimental/credit/cdsoption.hpp>
#include <ql/experimental/credit/constantlosslatentmodel.hpp>
#include <ql/experimental/credit/defaultprobabilitykey.hpp>
#include <ql/experimental/credit/integralntdengine.hpp>
#include <ql/experimental/credit/interpolatedaffinehazardratecurve.hpp>
#include <ql/experimental/credit/nthtodefault.hpp>
#include <ql/experimental/credit/onefactoraffinesurvival.hpp>
#include <ql/experimental/credit/onefactorstudentcopula.hpp>
#include <ql/experimental/credit/pool.hpp>
#include <ql/experimental/credit/riskyassetswap.hpp>
#include <ql/instruments/creditdefaultswap.hpp>
#include <ql/math/interpolations/backwardflatinterpolation.hpp>
#include <ql/models/shortrate/onefactormodels/coxingersollross.hpp>
#include <ql/pricingengines/credit/midpointcdsengine.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/credit/flathazardrate.hpp>
#include <ql/termstructures/credit/piecewisedefaultcurve.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/daycounters/thirty360.hpp>
#include <ql/time/schedule.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace QuantLib;

namespace {

// --------------------------------------------------------------------------
// JSON emission (flat top-level object; the pquantlib harness convention)
// --------------------------------------------------------------------------

bool g_first = true;

void sep() {
    if (!g_first) std::cout << ",\n";
    g_first = false;
}

void emit(const std::string& name, Real v) {
    sep();
    std::cout << "  \"" << name << "\": " << std::setprecision(17) << v;
}

void emit_int(const std::string& name, long long v) {
    sep();
    std::cout << "  \"" << name << "\": " << v;
}

void emit_bool(const std::string& name, bool v) {
    sep();
    std::cout << "  \"" << name << "\": " << (v ? "true" : "false");
}

void emit_str(const std::string& name, const std::string& v) {
    sep();
    std::cout << "  \"" << name << "\": \"" << v << "\"";
}

void emit_arr(const std::string& name, const std::vector<Real>& v) {
    sep();
    std::cout << "  \"" << name << "\": [";
    for (Size i = 0; i < v.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << std::setprecision(17) << v[i];
    }
    std::cout << "]";
}

void emit_iarr(const std::string& name, const std::vector<long long>& v) {
    sep();
    std::cout << "  \"" << name << "\": [";
    for (Size i = 0; i < v.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << v[i];
    }
    std::cout << "]";
}

void emit_sarr(const std::string& name, const std::vector<std::string>& v) {
    sep();
    std::cout << "  \"" << name << "\": [";
    for (Size i = 0; i < v.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << "\"" << v[i] << "\"";
    }
    std::cout << "]";
}

// The single pinned evaluation date for the whole probe.
const Date EVAL_DATE(15, January, 2024);

// --------------------------------------------------------------------------
// A. NorthAmericaCorpDefaultKey
// --------------------------------------------------------------------------

void dumpKey(const std::string& prefix, const DefaultProbKey& k) {
    emit_int(prefix + "_size", static_cast<long long>(k.size()));
    emit_str(prefix + "_currency_code", k.currency().code());
    emit_str(prefix + "_currency_name", k.currency().name());
    emit_int(prefix + "_seniority", static_cast<long long>(k.seniority()));

    std::vector<long long> atomics, restrs;
    std::vector<long long> isRestructuring;
    for (Size i = 0; i < k.eventTypes().size(); ++i) {
        atomics.push_back(static_cast<long long>(k.eventTypes()[i]->defaultType()));
        restrs.push_back(static_cast<long long>(k.eventTypes()[i]->restructuringType()));
        isRestructuring.push_back(k.eventTypes()[i]->isRestructuring() ? 1 : 0);
    }
    emit_iarr(prefix + "_atomic_types", atomics);
    emit_iarr(prefix + "_restructuring_types", restrs);
    emit_iarr(prefix + "_is_restructuring", isRestructuring);

    // The first event type is always the FailureToPay one; pin its extras.
    auto ftp = ext::dynamic_pointer_cast<FailureToPay>(k.eventTypes()[0]);
    emit_bool(prefix + "_first_is_failure_to_pay", static_cast<bool>(ftp));
    if (ftp != nullptr) {
        emit_int(prefix + "_ftp_grace_length",
                 static_cast<long long>(ftp->gracePeriod().length()));
        emit_int(prefix + "_ftp_grace_units",
                 static_cast<long long>(ftp->gracePeriod().units()));
        emit(prefix + "_ftp_amount_required", ftp->amountRequired());
    }
}

void blockA() {
    // Default construction path: grace 30D, amount 1e6, Restructuring::CR.
    NorthAmericaCorpDefaultKey kDefault(USDCurrency(), SeniorSec);
    dumpKey("nacdk_default", kDefault);

    // Fully specified, with a restructuring type.
    NorthAmericaCorpDefaultKey kFull(EURCurrency(), SubLT2, Period(45, Days), 2.5e6,
                                     Restructuring::ModifiedRestructuring);
    dumpKey("nacdk_full", kFull);

    // NoRestructuring drops the third event type entirely.
    NorthAmericaCorpDefaultKey kNoRes(EURCurrency(), SeniorSec, Period(), 1.0,
                                      Restructuring::NoRestructuring);
    dumpKey("nacdk_norestructuring", kNoRes);

    // The test-suite spelling used by nthtodefault.cpp.
    NorthAmericaCorpDefaultKey kTest(EURCurrency(), SeniorSec, Period(), 1.0);
    dumpKey("nacdk_testsuite", kTest);

    // operator== semantics against a hand-built equivalent key.
    std::vector<ext::shared_ptr<DefaultType> > types;
    types.push_back(ext::make_shared<FailureToPay>(Period(30, Days), 1.0e6));
    types.push_back(ext::make_shared<DefaultType>(AtomicDefault::Bankruptcy,
                                                  Restructuring::XR));
    types.push_back(ext::make_shared<DefaultType>(AtomicDefault::Restructuring,
                                                  Restructuring::CR));
    DefaultProbKey handBuilt(types, USDCurrency(), SeniorSec);
    emit_bool("nacdk_default_equals_handbuilt", kDefault == handBuilt);
    emit_bool("nacdk_default_equals_norestructuring", kDefault == kNoRes);
    emit_bool("nacdk_default_equals_testsuite", kDefault == kTest);
}

// --------------------------------------------------------------------------
// B. mixed Student / Gaussian one-factor copulas
// --------------------------------------------------------------------------

// Grids used for every copula: a mix of on-node y values (the table is built
// on -10 + 0.1*i) and deliberately off-node ones (to pin the linear
// interpolation, not just the table).
const std::vector<Real> Y_GRID = {-10.0, -6.0, -3.0, -2.5, -1.0, -0.5, -0.25,
                                  0.0,   0.25, 0.5,  1.0,  2.5,  3.0,  6.0, 10.0};
const std::vector<Real> P_GRID = {0.0001, 0.001, 0.01, 0.05, 0.1, 0.25,
                                  0.5,    0.75,  0.9,  0.99, 0.999};
const std::vector<Real> M_GRID = {-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0};
const std::vector<Real> Z_GRID = {-4.0, -2.0, -1.0, -0.25, 0.0, 0.25, 1.0, 2.0, 4.0};

template <class Copula>
void dumpCopula(const std::string& prefix, Copula& cop, bool withConditional = true) {
    std::vector<Real> cumY, invY, dens, cumZ;
    for (Real y : Y_GRID) cumY.push_back(cop.cumulativeY(y));
    for (Real p : P_GRID) invY.push_back(cop.inverseCumulativeY(p));
    for (Real m : M_GRID) dens.push_back(cop.density(m));
    for (Real z : Z_GRID) cumZ.push_back(cop.cumulativeZ(z));

    emit_arr(prefix + "_cumulative_y", cumY);
    emit_arr(prefix + "_inverse_cumulative_y", invY);
    emit_arr(prefix + "_density", dens);
    emit_arr(prefix + "_cumulative_z", cumZ);
    emit(prefix + "_correlation", cop.correlation());

    // conditionalProbability over the (p, m) product grid, row-major in p.
    // Skipped at rho == 1: sqrt(1-c) == 0 makes the C++ expression 0/0 and
    // QL_REQUIRE fires on the resulting NaN. That is C++ behaviour, not a
    // value to pin.
    if (withConditional) {
        std::vector<Real> integ;
        for (Real p : P_GRID) integ.push_back(cop.integral(p));
        emit_arr(prefix + "_integral", integ);
        std::vector<Real> condp;
        for (Real p : P_GRID)
            for (Real m : M_GRID) condp.push_back(cop.conditionalProbability(p, m));
        emit_arr(prefix + "_conditional_probability", condp);
    }

    // The tabulated cumulative-Y table itself, read back at its own nodes.
    // The table is y = -10 + 20*i/200; evaluating cumulativeY exactly at a
    // node returns that node's value through the interpolation identity.
    std::vector<Real> table;
    for (Size i = 0; i <= 200; ++i) {
        Real y = -10.0 + (10.0 - (-10.0)) * i / 200;
        table.push_back(cop.cumulativeY(y));
    }
    emit_arr(prefix + "_table_cumulative_y", table);
}

void blockB() {
    emit_arr("copula_y_grid", Y_GRID);
    emit_arr("copula_p_grid", P_GRID);
    emit_arr("copula_m_grid", M_GRID);
    emit_arr("copula_z_grid", Z_GRID);

    // Gaussian M, Student Z (nz=5), correlation below and above the 0.5
    // branch switch in cumulativeYintegral.
    for (Real rho : {0.3, 0.7}) {
        ext::shared_ptr<SimpleQuote> q(new SimpleQuote(rho));
        Handle<Quote> h(q);
        OneFactorGaussianStudentCopula cop(h, 5);
        std::string p = "gauss_student_rho" + std::string(rho < 0.5 ? "03" : "07");
        dumpCopula(p, cop);
    }
    // Degenerate correlations take the closed-form early-outs.
    {
        ext::shared_ptr<SimpleQuote> q(new SimpleQuote(0.0));
        Handle<Quote> h(q);
        OneFactorGaussianStudentCopula cop(h, 5);
        dumpCopula("gauss_student_rho00", cop);
    }

    // Student M (nm=5), Gaussian Z.
    for (Real rho : {0.3, 0.7}) {
        ext::shared_ptr<SimpleQuote> q(new SimpleQuote(rho));
        Handle<Quote> h(q);
        OneFactorStudentGaussianCopula cop(h, 5);
        std::string p = "student_gauss_rho" + std::string(rho < 0.5 ? "03" : "07");
        dumpCopula(p, cop);
    }
    {
        ext::shared_ptr<SimpleQuote> q(new SimpleQuote(1.0));
        Handle<Quote> h(q);
        OneFactorStudentGaussianCopula cop(h, 5);
        dumpCopula("student_gauss_rho10", cop, false);
    }

    // Double Student — the class PQuantLib already claims to have.
    {
        ext::shared_ptr<SimpleQuote> q(new SimpleQuote(0.3));
        Handle<Quote> h(q);
        OneFactorStudentCopula cop(h, 5, 5);
        dumpCopula("student_student_rho03", cop);
    }
    {
        ext::shared_ptr<SimpleQuote> q(new SimpleQuote(0.7));
        Handle<Quote> h(q);
        OneFactorStudentCopula cop(h, 7, 5);
        dumpCopula("student_student_rho07", cop);
    }
}

// --------------------------------------------------------------------------
// C. OneFactorAffineSurvivalStructure
// --------------------------------------------------------------------------

const std::vector<Real> T_GRID = {0.0, 0.25, 0.5, 1.0, 2.0, 3.5, 5.0, 7.5, 10.0};

ext::shared_ptr<CoxIngersollRoss> makeCir() {
    // r0, theta, k, sigma; Feller-constrained (sigma^2 < 2*k*theta).
    return ext::make_shared<CoxIngersollRoss>(0.02, 0.04, 0.5, 0.1, true);
}

void blockC() {
    ext::shared_ptr<CoxIngersollRoss> cir = makeCir();
    emit("cir_r0", cir->dynamics()->process()->x0());
    emit("cir_short_rate_at_x0", cir->dynamics()->shortRate(0.0, cir->dynamics()->process()->x0()));
    std::vector<Real> db;
    for (Real t : T_GRID) db.push_back(cir->discountBond(0.0, t, 0.02));
    emit_arr("cir_discount_bond_r002", db);

    OneFactorAffineSurvivalStructure ts(cir, EVAL_DATE, NullCalendar(), Actual365Fixed());
    emit_int("ofas_reference_date", ts.referenceDate().serialNumber());
    emit_int("ofas_max_date", ts.maxDate().serialNumber());

    std::vector<Real> sp, dp, dd, hr;
    for (Real t : T_GRID) {
        sp.push_back(ts.survivalProbability(t, true));
        dp.push_back(ts.defaultProbability(t, true));
        dd.push_back(ts.defaultDensity(t, true));
        hr.push_back(ts.hazardRate(t, true));
    }
    emit_arr("ofas_t_grid", T_GRID);
    emit_arr("ofas_survival_probability", sp);
    emit_arr("ofas_default_probability", dp);
    emit_arr("ofas_default_density", dd);
    emit_arr("ofas_hazard_rate", hr);

    // conditionalSurvivalProbability over (tFwd, tTgt, yVal)
    const std::vector<Real> tFwds = {0.0, 1.0, 2.0, 5.0};
    const std::vector<Real> tTgts = {1.0, 2.0, 5.0, 10.0};
    const std::vector<Real> yVals = {0.005, 0.02, 0.05, 0.10};
    std::vector<Real> csp;
    std::vector<Real> keyFwd, keyTgt, keyY;
    for (Real tf : tFwds)
        for (Real tt : tTgts)
            for (Real yv : yVals) {
                if (tt < tf) continue;
                csp.push_back(ts.conditionalSurvivalProbability(tf, tt, yv, true));
                keyFwd.push_back(tf);
                keyTgt.push_back(tt);
                keyY.push_back(yv);
            }
    emit_arr("ofas_csp_tfwd", keyFwd);
    emit_arr("ofas_csp_ttgt", keyTgt);
    emit_arr("ofas_csp_yval", keyY);
    emit_arr("ofas_csp", csp);

    // Date overload of conditionalSurvivalProbability.
    Date dFwd = EVAL_DATE + Period(1, Years);
    Date dTgt = EVAL_DATE + Period(5, Years);
    emit_int("ofas_csp_date_dfwd", dFwd.serialNumber());
    emit_int("ofas_csp_date_dtgt", dTgt.serialNumber());
    emit("ofas_csp_date", ts.conditionalSurvivalProbability(dFwd, dTgt, 0.02, true));
}

// --------------------------------------------------------------------------
// D. InterpolatedAffineHazardRateCurve<BackwardFlat> + AffineHazardRate traits
// --------------------------------------------------------------------------

typedef InterpolatedAffineHazardRateCurve<BackwardFlat> AffineCurve;

void blockD() {
    ext::shared_ptr<CoxIngersollRoss> cir = makeCir();

    std::vector<Date> dates = {EVAL_DATE,
                               EVAL_DATE + Period(1, Years),
                               EVAL_DATE + Period(3, Years),
                               EVAL_DATE + Period(5, Years),
                               EVAL_DATE + Period(10, Years)};
    std::vector<Rate> hazardRates = {0.008, 0.008, 0.012, 0.016, 0.021};

    AffineCurve curve(dates, hazardRates, Actual365Fixed(), cir, NullCalendar());

    std::vector<long long> dser;
    for (const Date& d : dates) dser.push_back(d.serialNumber());
    emit_iarr("iahrc_dates", dser);
    emit_arr("iahrc_input_hazard_rates", hazardRates);

    std::vector<long long> outDates;
    for (const Date& d : curve.dates()) outDates.push_back(d.serialNumber());
    emit_iarr("iahrc_out_dates", outDates);
    emit_arr("iahrc_times", curve.times());
    emit_arr("iahrc_data", curve.data());
    emit_arr("iahrc_hazard_rates", curve.hazardRates());
    emit_int("iahrc_max_date", curve.maxDate().serialNumber());
    emit_int("iahrc_reference_date", curve.referenceDate().serialNumber());

    std::vector<long long> nodeDates;
    std::vector<Real> nodeVals;
    for (const auto& n : curve.nodes()) {
        nodeDates.push_back(n.first.serialNumber());
        nodeVals.push_back(n.second);
    }
    emit_iarr("iahrc_node_dates", nodeDates);
    emit_arr("iahrc_node_values", nodeVals);

    // On-node and off-node evaluation dates.
    std::vector<Date> probeDates = {EVAL_DATE + Period(6, Months),
                                    EVAL_DATE + Period(1, Years),
                                    EVAL_DATE + Period(2, Years),
                                    EVAL_DATE + Period(3, Years),
                                    EVAL_DATE + Period(4, Years),
                                    EVAL_DATE + Period(5, Years),
                                    EVAL_DATE + Period(7, Years),
                                    EVAL_DATE + Period(10, Years),
                                    EVAL_DATE + Period(12, Years)};
    std::vector<long long> pser;
    std::vector<Real> hz, sv, dpr, den, tms;
    for (const Date& d : probeDates) {
        pser.push_back(d.serialNumber());
        tms.push_back(curve.timeFromReference(d));
        hz.push_back(curve.hazardRate(d, true));
        sv.push_back(curve.survivalProbability(d, true));
        dpr.push_back(curve.defaultProbability(d, true));
        den.push_back(curve.defaultDensity(d, true));
    }
    emit_iarr("iahrc_probe_dates", pser);
    emit_arr("iahrc_probe_times", tms);
    emit_arr("iahrc_probe_hazard_rate", hz);
    emit_arr("iahrc_probe_survival_probability", sv);
    emit_arr("iahrc_probe_default_probability", dpr);
    emit_arr("iahrc_probe_default_density", den);

    // Time overloads (including t == 0, which takes the early-out branch).
    std::vector<Real> tsv, thz;
    for (Real t : T_GRID) {
        tsv.push_back(curve.survivalProbability(t, true));
        thz.push_back(curve.hazardRate(t, true));
    }
    emit_arr("iahrc_t_grid", T_GRID);
    emit_arr("iahrc_time_survival_probability", tsv);
    emit_arr("iahrc_time_hazard_rate", thz);

    // conditionalSurvivalProbability — the interpolated-curve override.
    const std::vector<Real> tFwds = {0.0, 0.5, 2.0, 6.0, 12.0};
    const std::vector<Real> tTgts = {1.0, 3.0, 5.0, 11.0, 15.0};
    const std::vector<Real> yVals = {0.005, 0.02, 0.05};
    std::vector<Real> csp, keyFwd, keyTgt, keyY;
    for (Real tf : tFwds)
        for (Real tt : tTgts)
            for (Real yv : yVals) {
                if (tt < tf) continue;
                csp.push_back(curve.conditionalSurvivalProbability(tf, tt, yv, true));
                keyFwd.push_back(tf);
                keyTgt.push_back(tt);
                keyY.push_back(yv);
            }
    emit_arr("iahrc_csp_tfwd", keyFwd);
    emit_arr("iahrc_csp_ttgt", keyTgt);
    emit_arr("iahrc_csp_yval", keyY);
    emit_arr("iahrc_csp", csp);
    // tFwd == tTgt takes the "return 1" branch only when tFwd - tTgt == 0.
    emit("iahrc_csp_equal_times", curve.conditionalSurvivalProbability(3.0, 3.0, 0.05, true));
    // tFwd == 0 delegates to survivalProbabilityImpl.
    emit("iahrc_csp_zero_fwd", curve.conditionalSurvivalProbability(0.0, 4.0, 0.05, true));

    // --- AffineHazardRate traits ------------------------------------------
    emit_int("aff_traits_max_iterations",
             static_cast<long long>(AffineHazardRate::maxIterations()));
    emit("aff_traits_initial_value", AffineHazardRate::initialValue(&curve));
    emit_int("aff_traits_initial_date",
             AffineHazardRate::initialDate(&curve).serialNumber());

    // guess/min/max with validData == false
    emit("aff_traits_guess_i1_invalid", AffineHazardRate::guess(1, &curve, false, 0));
    emit("aff_traits_guess_i2_invalid", AffineHazardRate::guess(2, &curve, false, 0));
    emit("aff_traits_guess_i4_invalid", AffineHazardRate::guess(4, &curve, false, 0));
    emit("aff_traits_min_after_invalid", AffineHazardRate::minValueAfter(2, &curve, false, 0));
    emit("aff_traits_max_after_invalid", AffineHazardRate::maxValueAfter(2, &curve, false, 0));
    // ... and with validData == true
    emit("aff_traits_guess_i2_valid", AffineHazardRate::guess(2, &curve, true, 0));
    emit("aff_traits_min_after_valid", AffineHazardRate::minValueAfter(2, &curve, true, 0));
    emit("aff_traits_max_after_valid", AffineHazardRate::maxValueAfter(2, &curve, true, 0));

    // updateGuess: i != 1 touches one slot, i == 1 touches two.
    std::vector<Real> d1 = {1.0, 2.0, 3.0, 4.0};
    AffineHazardRate::updateGuess(d1, 9.0, 2);
    emit_arr("aff_traits_update_guess_i2", d1);
    std::vector<Real> d2 = {1.0, 2.0, 3.0, 4.0};
    AffineHazardRate::updateGuess(d2, 9.0, 1);
    emit_arr("aff_traits_update_guess_i1", d2);
}

// --------------------------------------------------------------------------
// E. AssetSwapHelper + RiskyAssetSwap
// --------------------------------------------------------------------------

void blockE() {
    Settings::instance().evaluationDate() = EVAL_DATE;

    DayCounter dc = Actual365Fixed();
    Calendar cal = TARGET();
    RelinkableHandle<YieldTermStructure> yieldTS;
    yieldTS.linkTo(ext::make_shared<FlatForward>(EVAL_DATE, 0.03, dc, Continuous, Annual));

    const Real recovery = 0.4;
    const std::vector<Period> tenors = {Period(2, Years), Period(3, Years), Period(5, Years),
                                        Period(7, Years)};
    const std::vector<Real> spreads = {0.0060, 0.0080, 0.0110, 0.0135};

    std::vector<ext::shared_ptr<SimpleQuote> > quotes;
    std::vector<ext::shared_ptr<DefaultProbabilityHelper> > helpers;
    for (Size i = 0; i < tenors.size(); ++i) {
        ext::shared_ptr<SimpleQuote> q(new SimpleQuote(spreads[i]));
        quotes.push_back(q);
        helpers.push_back(ext::make_shared<AssetSwapHelper>(
            Handle<Quote>(q), tenors[i], 2, cal, Period(1, Years), Following, Thirty360(Thirty360::BondBasis),
            Period(3, Months), Following, Actual360(), recovery, yieldTS, Period(1, Years)));
    }

    std::vector<long long> earliest, latest, maturity, pillar;
    for (const auto& h : helpers) {
        earliest.push_back(h->earliestDate().serialNumber());
        latest.push_back(h->latestDate().serialNumber());
        maturity.push_back(h->maturityDate().serialNumber());
        pillar.push_back(h->pillarDate().serialNumber());
    }
    emit_iarr("asw_helper_earliest_dates", earliest);
    emit_iarr("asw_helper_latest_dates", latest);
    emit_iarr("asw_helper_maturity_dates", maturity);
    emit_iarr("asw_helper_pillar_dates", pillar);
    emit_arr("asw_helper_quotes", spreads);

    ext::shared_ptr<PiecewiseDefaultCurve<HazardRate, BackwardFlat> > curve(
        new PiecewiseDefaultCurve<HazardRate, BackwardFlat>(EVAL_DATE, helpers, dc));
    curve->enableExtrapolation();

    std::vector<long long> curveDates;
    for (const Date& d : curve->dates()) curveDates.push_back(d.serialNumber());
    emit_iarr("asw_curve_dates", curveDates);
    emit_arr("asw_curve_times", curve->times());
    emit_arr("asw_curve_data", curve->data());

    std::vector<Real> impliedQuotes;
    for (const auto& h : helpers) impliedQuotes.push_back(h->impliedQuote());
    emit_arr("asw_helper_implied_quotes", impliedQuotes);

    // The curve itself, sampled.
    std::vector<Date> sample = {EVAL_DATE + Period(1, Years), EVAL_DATE + Period(2, Years),
                                EVAL_DATE + Period(4, Years), EVAL_DATE + Period(5, Years),
                                EVAL_DATE + Period(7, Years)};
    std::vector<long long> sser;
    std::vector<Real> ssurv, shaz;
    for (const Date& d : sample) {
        sser.push_back(d.serialNumber());
        ssurv.push_back(curve->survivalProbability(d, true));
        shaz.push_back(curve->hazardRate(d, true));
    }
    emit_iarr("asw_sample_dates", sser);
    emit_arr("asw_sample_survival", ssurv);
    emit_arr("asw_sample_hazard_rate", shaz);

    // A standalone RiskyAssetSwap on a flat hazard curve — the object the
    // helper builds internally (nominal 100, spread 0.01, fixedPayer true).
    Handle<DefaultProbabilityTermStructure> flatDefault(
        ext::make_shared<FlatHazardRate>(EVAL_DATE, Handle<Quote>(ext::make_shared<SimpleQuote>(0.02)), dc));
    Date settle = cal.advance(EVAL_DATE, 2, Days);
    Date mat = settle + Period(5, Years);
    Schedule fixedSchedule(settle, mat, Period(1, Years), cal, Following, Following,
                           DateGeneration::Forward, false);
    Schedule floatSchedule(settle, mat, Period(3, Months), cal, Following, Following,
                           DateGeneration::Forward, false);
    RiskyAssetSwap asw(true, 100.0, fixedSchedule, floatSchedule,
                       Thirty360(Thirty360::BondBasis), Actual360(), 0.01, recovery,
                       Handle<YieldTermStructure>(*yieldTS), flatDefault);
    emit("asw_standalone_npv", asw.NPV());
    emit("asw_standalone_fair_spread", asw.fairSpread());
    emit("asw_standalone_float_annuity", asw.floatAnnuity());
    emit("asw_standalone_nominal", asw.nominal());
    emit("asw_standalone_spread", asw.spread());
    emit_bool("asw_standalone_fixed_payer", asw.fixedPayer());
    emit_int("asw_standalone_settle", settle.serialNumber());
    emit_int("asw_standalone_maturity", mat.serialNumber());
    std::vector<long long> fsd, lsd;
    for (const Date& d : fixedSchedule.dates()) fsd.push_back(d.serialNumber());
    for (const Date& d : floatSchedule.dates()) lsd.push_back(d.serialNumber());
    emit_iarr("asw_standalone_fixed_schedule", fsd);
    emit_iarr("asw_standalone_float_schedule", lsd);
}

// --------------------------------------------------------------------------
// F. CdsOption + BlackCdsOptionEngine
// --------------------------------------------------------------------------

struct CdsOptionSetup {
    Handle<YieldTermStructure> yts;
    Handle<DefaultProbabilityTermStructure> dts;
    ext::shared_ptr<CreditDefaultSwap> cds;
    Date exerciseDate;
};

CdsOptionSetup makeCdsOption(Protection::Side side) {
    DayCounter dc = Actual365Fixed();
    Calendar cal = TARGET();
    CdsOptionSetup s;
    s.yts = Handle<YieldTermStructure>(
        ext::make_shared<FlatForward>(EVAL_DATE, 0.03, dc, Continuous, Annual));
    s.dts = Handle<DefaultProbabilityTermStructure>(ext::make_shared<FlatHazardRate>(
        EVAL_DATE, Handle<Quote>(ext::make_shared<SimpleQuote>(0.02)), dc));
    s.exerciseDate = cal.advance(EVAL_DATE, 1, Years);
    Date end = cal.advance(s.exerciseDate, 5, Years);
    Schedule sched = MakeSchedule()
                         .from(s.exerciseDate)
                         .to(end)
                         .withCalendar(cal)
                         .withTenor(Period(3, Months))
                         .withConvention(Following)
                         .withRule(DateGeneration::CDS2015);
    s.cds = ext::make_shared<CreditDefaultSwap>(side, 1.0e7, 0.0100, sched, Following, dc,
                                                true, true, s.exerciseDate);
    s.cds->setPricingEngine(ext::make_shared<MidPointCdsEngine>(s.dts, 0.4, s.yts));
    return s;
}

void blockF() {
    Settings::instance().evaluationDate() = EVAL_DATE;

    // --- payer (Buyer-side underlying), knock-out and not ------------------
    for (int knocks = 1; knocks >= 0; --knocks) {
        CdsOptionSetup s = makeCdsOption(Protection::Buyer);
        ext::shared_ptr<SimpleQuote> vol(new SimpleQuote(0.30));
        CdsOption opt(s.cds, ext::make_shared<EuropeanExercise>(s.exerciseDate),
                      knocks != 0);
        opt.setPricingEngine(ext::make_shared<BlackCdsOptionEngine>(
            s.dts, 0.4, s.yts, Handle<Quote>(vol)));
        std::string p = knocks ? "cdso_payer_ko" : "cdso_payer_nko";
        emit(p + "_npv", opt.NPV());
        emit(p + "_risky_annuity", opt.riskyAnnuity());
        emit(p + "_atm_rate", opt.atmRate());
        emit_bool(p + "_is_expired", opt.isExpired());
        // impliedVolatility must round-trip the price back to 0.30.
        emit(p + "_implied_vol_roundtrip",
             opt.impliedVolatility(opt.NPV(), s.yts, s.dts, 0.4));
        // ... and recover a different vol from a different target price.
        vol->setValue(0.55);
        Real target = opt.NPV();
        emit(p + "_npv_vol055", target);
        vol->setValue(0.30);
        emit(p + "_implied_vol_from_055", opt.impliedVolatility(target, s.yts, s.dts, 0.4));
        // Tight accuracy variant, to pin the Brent path rather than the answer.
        emit(p + "_implied_vol_acc1e8",
             opt.impliedVolatility(target, s.yts, s.dts, 0.4, 1e-8, 200, 1e-7, 4.0));
    }

    // --- receiver (Seller-side underlying), must knock out -----------------
    {
        CdsOptionSetup s = makeCdsOption(Protection::Seller);
        ext::shared_ptr<SimpleQuote> vol(new SimpleQuote(0.30));
        CdsOption opt(s.cds, ext::make_shared<EuropeanExercise>(s.exerciseDate), true);
        opt.setPricingEngine(ext::make_shared<BlackCdsOptionEngine>(
            s.dts, 0.4, s.yts, Handle<Quote>(vol)));
        emit("cdso_receiver_ko_npv", opt.NPV());
        emit("cdso_receiver_ko_risky_annuity", opt.riskyAnnuity());
        emit("cdso_receiver_ko_atm_rate", opt.atmRate());
        emit("cdso_receiver_ko_implied_vol_roundtrip",
             opt.impliedVolatility(opt.NPV(), s.yts, s.dts, 0.4));
        vol->setValue(0.15);
        Real target = opt.NPV();
        emit("cdso_receiver_ko_npv_vol015", target);
        vol->setValue(0.30);
        emit("cdso_receiver_ko_implied_vol_from_015",
             opt.impliedVolatility(target, s.yts, s.dts, 0.4));
    }

    // Shared inputs so the Python test can rebuild the same underlying.
    {
        CdsOptionSetup s = makeCdsOption(Protection::Buyer);
        emit_int("cdso_exercise_date", s.exerciseDate.serialNumber());
        std::vector<long long> cf;
        for (const auto& c : s.cds->coupons()) cf.push_back(c->date().serialNumber());
        emit_iarr("cdso_coupon_dates", cf);
        emit("cdso_fair_spread", s.cds->fairSpread());
        emit("cdso_running_spread", s.cds->runningSpread());
        emit("cdso_coupon_leg_npv", s.cds->couponLegNPV());
        emit("cdso_cds_npv", s.cds->NPV());
    }
}

// --------------------------------------------------------------------------
// G. IntegralNtdEngine
// --------------------------------------------------------------------------

void blockG() {
    Settings::instance().evaluationDate() = EVAL_DATE;

    const Size names = 5;
    const Real recovery = 0.4;
    const Real namesNotional = 100.0;   // per-name notional
    const Real lambda = 0.01;
    DayCounter dc = Actual365Fixed();
    Calendar cal = TARGET();

    Handle<YieldTermStructure> yts(
        ext::make_shared<FlatForward>(EVAL_DATE, 0.03, dc, Continuous, Annual));

    std::vector<Handle<DefaultProbabilityTermStructure> > probabilities;
    for (Size i = 0; i < names; ++i)
        probabilities.emplace_back(ext::make_shared<FlatHazardRate>(
            EVAL_DATE, Handle<Quote>(ext::make_shared<SimpleQuote>(lambda)), dc));

    ext::shared_ptr<SimpleQuote> corr(new SimpleQuote(0.3));
    ext::shared_ptr<DefaultLossModel> lossModel(new ConstantLossModel<GaussianCopulaPolicy>(
        Handle<Quote>(corr), std::vector<Real>(names, recovery),
        LatentModelIntegrationType::GaussianQuadrature, names,
        GaussianCopulaPolicy::initTraits()));

    std::vector<std::string> namesIds;
    for (Size i = 0; i < names; ++i) namesIds.push_back("Name" + std::to_string(i));

    ext::shared_ptr<Pool> pool = ext::make_shared<Pool>();
    for (Size i = 0; i < names; ++i) {
        std::vector<Issuer::key_curve_pair> curves(
            1, std::make_pair(NorthAmericaCorpDefaultKey(EURCurrency(), SeniorSec, Period(), 1.0),
                              probabilities[i]));
        pool->add(namesIds[i], Issuer(curves),
                  NorthAmericaCorpDefaultKey(EURCurrency(), SeniorSec, Period(), 1.0));
    }

    ext::shared_ptr<Basket> basket(new Basket(
        EVAL_DATE, namesIds, std::vector<Real>(names, namesNotional), pool, 0.0, 1.0));
    basket->setLossModel(lossModel);

    Date start(20, March, 2024);
    Date end(20, March, 2029);
    Schedule schedule = MakeSchedule().from(start).to(end).withTenor(Period(3, Months)).withCalendar(cal);

    std::vector<long long> schedDates;
    for (const Date& d : schedule.dates()) schedDates.push_back(d.serialNumber());
    emit_iarr("ntd_schedule_dates", schedDates);

    const Period stepSize(1, Months);
    const Size rank = 2;
    const Real notional = namesNotional * names;
    const Real premiumRate = 0.02;
    const Real upfrontRate = 0.0;

    emit_int("ntd_rank", static_cast<long long>(rank));
    emit("ntd_notional", notional);
    emit("ntd_premium_rate", premiumRate);
    emit("ntd_upfront_rate", upfrontRate);
    emit("ntd_recovery_rate", basket->recoveryRate(EVAL_DATE, 0));
    emit("ntd_remaining_notional", basket->remainingNotional());
    emit_int("ntd_remaining_size", static_cast<long long>(basket->remainingSize()));
    emit("ntd_correlation", corr->value());

    // ---- the engine-input trace ------------------------------------------
    // Reproduce exactly the date walk IntegralNtdEngine::calculate performs,
    // and pin probAtLeastNEvents at every date it touches.
    ext::shared_ptr<NthToDefault> ntdForLeg = ext::make_shared<NthToDefault>(
        basket, rank, Protection::Seller, schedule, upfrontRate, premiumRate, Actual360(),
        notional, true);

    NthToDefault::arguments args;
    ntdForLeg->setupArguments(&args);

    std::vector<long long> traceDates;
    std::vector<Real> traceProbs;
    std::vector<long long> payDates;
    std::vector<Real> payAmounts, payProbs, payDiscounts;
    std::vector<Real> accrued;
    for (auto& i : args.premiumLeg) {
        ext::shared_ptr<FixedRateCoupon> coupon = ext::dynamic_pointer_cast<FixedRateCoupon>(i);
        Date d = i->date();
        if (d > yts->referenceDate()) {
            payDates.push_back(d.serialNumber());
            payAmounts.push_back(i->amount());
            payProbs.push_back(basket->probAtLeastNEvents(rank, d));
            payDiscounts.push_back(yts->discount(d));

            if (coupon->accrualStartDate() >= yts->referenceDate())
                d = coupon->accrualStartDate();
            else
                d = yts->referenceDate();
            Date d0 = d;
            Period step = stepSize;
            do {
                traceDates.push_back(d.serialNumber());
                traceProbs.push_back(basket->probAtLeastNEvents(rank, d));
                accrued.push_back(coupon->accruedAmount(d));
                d0 = d;
                d = d0 + step;
                if (step != 1 * Days && d > coupon->accrualEndDate()) {
                    step = 1 * Days;
                    d = d0 + step;
                }
            } while (d <= coupon->accrualEndDate());
        }
    }
    emit_iarr("ntd_pay_dates", payDates);
    emit_arr("ntd_pay_amounts", payAmounts);
    emit_arr("ntd_pay_probs", payProbs);
    emit_arr("ntd_pay_discounts", payDiscounts);
    emit_iarr("ntd_trace_dates", traceDates);
    emit_arr("ntd_trace_probs", traceProbs);
    emit_arr("ntd_trace_accrued", accrued);
    // Discount factors at every traced date, so the Python replay does not
    // depend on the yield curve port.
    std::vector<Real> traceDiscounts;
    for (long long s : traceDates) traceDiscounts.push_back(yts->discount(Date(static_cast<Date::serial_type>(s))));
    emit_arr("ntd_trace_discounts", traceDiscounts);

    emit_int("ntd_first_coupon_accrual_start",
             ext::dynamic_pointer_cast<FixedRateCoupon>(args.premiumLeg[0])
                 ->accrualStartDate()
                 .serialNumber());
    emit_bool("ntd_first_coupon_has_occurred", args.premiumLeg[0]->hasOccurred(EVAL_DATE));
    emit("ntd_claim_amount_unit",
         basket->claim()->amount(EVAL_DATE, notional, basket->recoveryRate(EVAL_DATE, 0)));

    // ---- engine outputs ---------------------------------------------------
    ext::shared_ptr<PricingEngine> engine(new IntegralNtdEngine(stepSize, yts));

    for (int sideIdx = 0; sideIdx < 2; ++sideIdx) {
        Protection::Side side = sideIdx == 0 ? Protection::Seller : Protection::Buyer;
        std::string p = sideIdx == 0 ? "ntd_seller" : "ntd_buyer";
        NthToDefault ntd(basket, rank, side, schedule, upfrontRate, premiumRate, Actual360(),
                         notional, true);
        ntd.setPricingEngine(engine);
        emit(p + "_npv", ntd.NPV());
        emit(p + "_premium_leg_npv", ntd.premiumLegNPV());
        emit(p + "_protection_leg_npv", ntd.protectionLegNPV());
        emit(p + "_fair_premium", ntd.fairPremium());
        emit_int(p + "_rank", static_cast<long long>(ntd.rank()));
        emit_int(p + "_basket_size", static_cast<long long>(ntd.basketSize()));
    }

    // Non-zero upfront, to exercise the upfront branch.
    {
        NthToDefault ntd(basket, rank, Protection::Buyer, schedule, 0.05, premiumRate,
                         Actual360(), notional, true);
        ntd.setPricingEngine(engine);
        emit("ntd_buyer_upfront_npv", ntd.NPV());
        emit("ntd_buyer_upfront_premium_leg_npv", ntd.premiumLegNPV());
        emit("ntd_buyer_upfront_protection_leg_npv", ntd.protectionLegNPV());
        emit("ntd_buyer_upfront_fair_premium", ntd.fairPremium());
    }

    // settlePremiumAccrual == false drops the accrual term.
    {
        NthToDefault ntd(basket, rank, Protection::Seller, schedule, upfrontRate, premiumRate,
                         Actual360(), notional, false);
        ntd.setPricingEngine(engine);
        emit("ntd_noaccrual_npv", ntd.NPV());
        emit("ntd_noaccrual_premium_leg_npv", ntd.premiumLegNPV());
        emit("ntd_noaccrual_protection_leg_npv", ntd.protectionLegNPV());
        emit("ntd_noaccrual_fair_premium", ntd.fairPremium());
    }
}

}  // namespace

int main() {
    std::cout << "{\n";
    Settings::instance().evaluationDate() = EVAL_DATE;
    emit_int("evaluation_date", EVAL_DATE.serialNumber());

    blockA();
    blockB();
    blockC();
    blockD();
    blockE();
    blockF();
    blockG();

    std::cout << "\n}\n";
    return 0;
}

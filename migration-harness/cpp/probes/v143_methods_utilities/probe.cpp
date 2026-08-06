// v1.43 methods/finitedifferences/utilities cluster probe.
//
// Emits references/v143/methods/utilities.json.
//
// Cross-validation reference values for the 17 classes of
// ql/methods/finitedifferences/utilities/ that PQuantLib ports:
//
//   EscrowedDividendAdjustment          escroweddividendadjustment.hpp
//   FdmAffineModelSwapInnerValue        fdmaffinemodelswapinnervalue.hpp
//   FdmAffineModelTermStructure         fdmaffinemodeltermstructure.hpp
//   FdmCellAveragingInnerValue          fdminnervaluecalculator.hpp
//   FdmDirichletBoundary                fdmdirichletboundary.hpp
//   FdmDiscountDirichletBoundary        fdmdiscountdirichletboundary.hpp
//   FdmDividendHandler                  fdmdividendhandler.hpp
//   FdmEscrowedLogInnerValueCalculator  fdmescrowedloginnervaluecalculator.hpp
//   FdmIndicesOnBoundary                fdmindicesonboundary.hpp
//   FdmInnerValueCalculator             fdminnervaluecalculator.hpp
//   FdmLogBasketInnerValue              fdminnervaluecalculator.hpp
//   FdmLogInnerValue                    fdminnervaluecalculator.hpp
//   FdmMesherIntegral                   fdmmesherintegral.hpp
//   FdmQuantoHelper                     fdmquantohelper.hpp
//   FdmShoutLogInnerValueCalculator     fdmshoutloginnervaluecalculator.hpp
//   FdmTimeDepDirichletBoundary         fdmtimedepdirichletboundary.hpp
//   FdmZeroInnerValue                   fdminnervaluecalculator.hpp
//
// @ v1.43 (6b57206e0).
//
// Prints JSON to stdout (never writes a file).

#include <ql/cashflows/dividend.hpp>
#include <ql/currencies/europe.hpp>
#include <ql/handle.hpp>
#include <ql/indexes/iborindex.hpp>
#include <ql/instruments/basketoption.hpp>
#include <ql/instruments/payoffs.hpp>
#include <ql/instruments/vanillaswap.hpp>
#include <ql/math/integrals/discreteintegrals.hpp>
#include <ql/methods/finitedifferences/meshers/fdmmeshercomposite.hpp>
#include <ql/methods/finitedifferences/meshers/uniform1dmesher.hpp>
#include <ql/methods/finitedifferences/operators/fdmlinearoplayout.hpp>
#include <ql/methods/finitedifferences/utilities/escroweddividendadjustment.hpp>
#include <ql/methods/finitedifferences/utilities/fdmaffinemodelswapinnervalue.hpp>
#include <ql/methods/finitedifferences/utilities/fdmaffinemodeltermstructure.hpp>
#include <ql/methods/finitedifferences/utilities/fdmdirichletboundary.hpp>
#include <ql/methods/finitedifferences/utilities/fdmdiscountdirichletboundary.hpp>
#include <ql/methods/finitedifferences/utilities/fdmdividendhandler.hpp>
#include <ql/methods/finitedifferences/utilities/fdmescrowedloginnervaluecalculator.hpp>
#include <ql/methods/finitedifferences/utilities/fdmindicesonboundary.hpp>
#include <ql/methods/finitedifferences/utilities/fdminnervaluecalculator.hpp>
#include <ql/methods/finitedifferences/utilities/fdmmesherintegral.hpp>
#include <ql/methods/finitedifferences/utilities/fdmquantohelper.hpp>
#include <ql/methods/finitedifferences/utilities/fdmshoutloginnervaluecalculator.hpp>
#include <ql/methods/finitedifferences/utilities/fdmtimedepdirichletboundary.hpp>
#include <ql/models/shortrate/onefactormodels/hullwhite.hpp>
#include <ql/models/shortrate/twofactormodels/g2.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/schedule.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace QuantLib;

namespace {

std::vector<std::string> g_entries;

std::string fmt(Real v) {
    std::ostringstream os;
    os << std::setprecision(17) << v;
    return os.str();
}

void put(const std::string& name, Real v) {
    g_entries.push_back("  \"" + name + "\": " + fmt(v));
}

void put_reals(const std::string& name, const std::vector<Real>& v) {
    std::string s = "  \"" + name + "\": [";
    for (Size i = 0; i < v.size(); ++i) {
        if (i != 0) s += ", ";
        s += fmt(v[i]);
    }
    g_entries.push_back(s + "]");
}

void put_array(const std::string& name, const Array& a) {
    put_reals(name, std::vector<Real>(a.begin(), a.end()));
}

void put_sizes(const std::string& name, const std::vector<Size>& v) {
    std::string s = "  \"" + name + "\": [";
    for (Size i = 0; i < v.size(); ++i) {
        if (i != 0) s += ", ";
        s += std::to_string(v[i]);
    }
    g_entries.push_back(s + "]");
}

// Grab the iterator sitting at flat index `idx` of a layout.
FdmLinearOpIterator iter_at(const ext::shared_ptr<FdmLinearOpLayout>& layout, Size idx) {
    for (const auto& i : *layout) {
        if (i.index() == idx) return i;
    }
    QL_FAIL("index not found");
}

Array ramp(Size n) {
    Array a(n);
    for (Size i = 0; i < n; ++i) a[i] = Real(i) + 1.0;
    return a;
}

// ---------------------------------------------------------------- block A
// FdmIndicesOnBoundary
__attribute__((noinline)) void block_indices_on_boundary() {
    auto layout2d = ext::make_shared<FdmLinearOpLayout>(std::vector<Size>{5, 4});
    put_sizes("indices_2d_d0_lower",
              FdmIndicesOnBoundary(layout2d, 0, FdmDirichletBoundary::Lower).getIndices());
    put_sizes("indices_2d_d0_upper",
              FdmIndicesOnBoundary(layout2d, 0, FdmDirichletBoundary::Upper).getIndices());
    put_sizes("indices_2d_d1_lower",
              FdmIndicesOnBoundary(layout2d, 1, FdmDirichletBoundary::Lower).getIndices());
    put_sizes("indices_2d_d1_upper",
              FdmIndicesOnBoundary(layout2d, 1, FdmDirichletBoundary::Upper).getIndices());

    auto layout3d = ext::make_shared<FdmLinearOpLayout>(std::vector<Size>{3, 2, 4});
    put_sizes("indices_3d_d1_lower",
              FdmIndicesOnBoundary(layout3d, 1, FdmDirichletBoundary::Lower).getIndices());
    put_sizes("indices_3d_d1_upper",
              FdmIndicesOnBoundary(layout3d, 1, FdmDirichletBoundary::Upper).getIndices());
    put_sizes("indices_3d_d2_upper",
              FdmIndicesOnBoundary(layout3d, 2, FdmDirichletBoundary::Upper).getIndices());

    auto layout1d = ext::make_shared<FdmLinearOpLayout>(std::vector<Size>{6});
    put_sizes("indices_1d_lower",
              FdmIndicesOnBoundary(layout1d, 0, FdmDirichletBoundary::Lower).getIndices());
    put_sizes("indices_1d_upper",
              FdmIndicesOnBoundary(layout1d, 0, FdmDirichletBoundary::Upper).getIndices());
}

// ---------------------------------------------------------------- block B
// FdmDirichletBoundary
__attribute__((noinline)) void block_dirichlet_boundary() {
    auto m1 = ext::make_shared<Uniform1dMesher>(-2.0, 2.0, 5);
    auto m2 = ext::make_shared<Uniform1dMesher>(-1.0, 1.0, 4);
    auto mesher = ext::make_shared<FdmMesherComposite>(m1, m2);
    const Size n = mesher->layout()->size();

    FdmDirichletBoundary lower(mesher, 3.5, 0, FdmDirichletBoundary::Lower);
    Array a = ramp(n);
    lower.applyAfterApplying(a);
    put_array("dirichlet_d0_lower_after_applying", a);

    Array b = ramp(n);
    lower.applyAfterSolving(b);
    put_array("dirichlet_d0_lower_after_solving", b);

    FdmDirichletBoundary upper(mesher, -1.25, 1, FdmDirichletBoundary::Upper);
    Array c = ramp(n);
    upper.applyAfterApplying(c);
    put_array("dirichlet_d1_upper_after_applying", c);

    // scalar overload: value replaced only strictly outside the extreme node
    put("dirichlet_d0_lower_scalar_below", lower.applyAfterApplying(-3.0, 7.0));
    put("dirichlet_d0_lower_scalar_at", lower.applyAfterApplying(-2.0, 7.0));
    put("dirichlet_d0_lower_scalar_above", lower.applyAfterApplying(0.5, 7.0));
    put("dirichlet_d1_upper_scalar_above", upper.applyAfterApplying(2.0, 7.0));
    put("dirichlet_d1_upper_scalar_at", upper.applyAfterApplying(1.0, 7.0));
    put("dirichlet_d1_upper_scalar_below", upper.applyAfterApplying(0.0, 7.0));
}

// ---------------------------------------------------------------- block C
// FdmTimeDepDirichletBoundary
struct AffineOfTime {
    Real operator()(Real t) const { return 1.0 + 2.0 * t; }
};

struct VectorOfTime {
    Size n;
    Array operator()(Real t) const {
        Array r(n);
        for (Size i = 0; i < n; ++i) r[i] = (Real(i) + 1.0) * t;
        return r;
    }
};

__attribute__((noinline)) void block_time_dep_dirichlet() {
    auto m1 = ext::make_shared<Uniform1dMesher>(-2.0, 2.0, 5);
    auto m2 = ext::make_shared<Uniform1dMesher>(-1.0, 1.0, 4);
    auto mesher = ext::make_shared<FdmMesherComposite>(m1, m2);
    const Size n = mesher->layout()->size();

    FdmTimeDepDirichletBoundary scalarBc(
        mesher, std::function<Real(Real)>(AffineOfTime()), 0, FdmDirichletBoundary::Lower);
    scalarBc.setTime(0.3);
    Array a = ramp(n);
    scalarBc.applyAfterApplying(a);
    put_array("time_dep_scalar_d0_lower_t03", a);

    scalarBc.setTime(0.75);
    Array b = ramp(n);
    scalarBc.applyAfterSolving(b);
    put_array("time_dep_scalar_d0_lower_t075", b);

    // direction 1 / Upper hypersurface has 5 nodes
    FdmTimeDepDirichletBoundary vectorBc(
        mesher, std::function<Array(Real)>(VectorOfTime{5}), 1, FdmDirichletBoundary::Upper);
    vectorBc.setTime(0.4);
    Array c = ramp(n);
    vectorBc.applyAfterApplying(c);
    put_array("time_dep_vector_d1_upper_t04", c);
}

// ---------------------------------------------------------------- block D
// FdmDiscountDirichletBoundary
__attribute__((noinline)) void block_discount_dirichlet(const ext::shared_ptr<YieldTermStructure>& rTS) {
    auto m1 = ext::make_shared<Uniform1dMesher>(-2.0, 2.0, 5);
    auto m2 = ext::make_shared<Uniform1dMesher>(-1.0, 1.0, 4);
    auto mesher = ext::make_shared<FdmMesherComposite>(m1, m2);
    const Size n = mesher->layout()->size();

    FdmDiscountDirichletBoundary bc(mesher, rTS, 1.0, 100.0, 0, FdmDirichletBoundary::Upper);
    bc.setTime(0.4);
    Array a = ramp(n);
    bc.applyAfterApplying(a);
    put_array("discount_dirichlet_d0_upper_t04", a);

    bc.setTime(0.0);
    Array b = ramp(n);
    bc.applyAfterSolving(b);
    put_array("discount_dirichlet_d0_upper_t0", b);

    FdmDiscountDirichletBoundary bcLow(mesher, rTS, 2.0, -5.0, 1, FdmDirichletBoundary::Lower);
    bcLow.setTime(1.25);
    Array c = ramp(n);
    bcLow.applyAfterApplying(c);
    put_array("discount_dirichlet_d1_lower_t125", c);
}

// ---------------------------------------------------------------- block E
// FdmInnerValueCalculator family
__attribute__((noinline)) void block_inner_value_calculators() {
    // --- FdmLogInnerValue: log-spot grid, put payoff
    auto lm = ext::make_shared<Uniform1dMesher>(std::log(50.0), std::log(150.0), 11);
    auto lmesher = ext::make_shared<FdmMesherComposite>(lm);
    auto putPayoff = ext::make_shared<PlainVanillaPayoff>(Option::Put, 100.0);
    FdmLogInnerValue logCalc(putPayoff, lmesher, 0);

    std::vector<Real> logInner, logAvg;
    for (const auto& iter : *lmesher->layout()) {
        logInner.push_back(logCalc.innerValue(iter, 0.0));
    }
    for (const auto& iter : *lmesher->layout()) {
        logAvg.push_back(logCalc.avgInnerValue(iter, 0.0));
    }
    put_reals("log_inner_value_put", logInner);
    put_reals("log_avg_inner_value_put", logAvg);

    // --- FdmLogInnerValue: call payoff (fresh calculator: avg cache)
    auto callPayoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0);
    FdmLogInnerValue logCallCalc(callPayoff, lmesher, 0);
    std::vector<Real> logCallAvg;
    for (const auto& iter : *lmesher->layout()) {
        logCallAvg.push_back(logCallCalc.avgInnerValue(iter, 0.0));
    }
    put_reals("log_avg_inner_value_call", logCallAvg);

    // --- FdmCellAveragingInnerValue with the default identity grid mapping
    auto um = ext::make_shared<Uniform1dMesher>(50.0, 150.0, 11);
    auto umesher = ext::make_shared<FdmMesherComposite>(um);
    FdmCellAveragingInnerValue cellCalc(callPayoff, umesher, 0);
    std::vector<Real> cellInner, cellAvg;
    for (const auto& iter : *umesher->layout()) {
        cellInner.push_back(cellCalc.innerValue(iter, 0.0));
    }
    for (const auto& iter : *umesher->layout()) {
        cellAvg.push_back(cellCalc.avgInnerValue(iter, 0.0));
    }
    put_reals("cell_avg_inner_value_identity_inner", cellInner);
    put_reals("cell_avg_inner_value_identity_avg", cellAvg);

    // --- FdmCellAveragingInnerValue on a 2-D mesh, direction 1
    auto um2 = ext::make_shared<Uniform1dMesher>(0.0, 1.0, 4);
    auto umesher2 = ext::make_shared<FdmMesherComposite>(um2, um);
    FdmCellAveragingInnerValue cellCalc2(callPayoff, umesher2, 1);
    std::vector<Real> cell2Avg;
    for (const auto& iter : *umesher2->layout()) {
        cell2Avg.push_back(cellCalc2.avgInnerValue(iter, 0.0));
    }
    put_reals("cell_avg_inner_value_2d_dir1_avg", cell2Avg);

    // --- FdmLogBasketInnerValue: max-basket on a 2-D log grid
    auto b1 = ext::make_shared<Uniform1dMesher>(std::log(50.0), std::log(150.0), 5);
    auto b2 = ext::make_shared<Uniform1dMesher>(std::log(60.0), std::log(140.0), 4);
    auto bmesher = ext::make_shared<FdmMesherComposite>(b1, b2);
    auto basket = ext::make_shared<MaxBasketPayoff>(callPayoff);
    FdmLogBasketInnerValue basketCalc(basket, bmesher);
    std::vector<Real> basketInner, basketAvg;
    for (const auto& iter : *bmesher->layout()) {
        basketInner.push_back(basketCalc.innerValue(iter, 0.0));
        basketAvg.push_back(basketCalc.avgInnerValue(iter, 0.0));
    }
    put_reals("log_basket_inner_value_max", basketInner);
    put_reals("log_basket_avg_inner_value_max", basketAvg);

    auto minBasket = ext::make_shared<MinBasketPayoff>(callPayoff);
    FdmLogBasketInnerValue minCalc(minBasket, bmesher);
    std::vector<Real> minInner;
    for (const auto& iter : *bmesher->layout()) {
        minInner.push_back(minCalc.innerValue(iter, 0.0));
    }
    put_reals("log_basket_inner_value_min", minInner);

    // --- FdmZeroInnerValue
    FdmZeroInnerValue zero;
    const auto it0 = iter_at(lmesher->layout(), 3);
    put("zero_inner_value", zero.innerValue(it0, 0.75));
    put("zero_avg_inner_value", zero.avgInnerValue(it0, 0.75));
}

// ---------------------------------------------------------------- block F
// FdmMesherIntegral
__attribute__((noinline)) void block_mesher_integral() {
    const std::function<Real(const Array&, const Array&)> simpson = DiscreteSimpsonIntegral();
    const std::function<Real(const Array&, const Array&)> trapezoid = DiscreteTrapezoidIntegral();

    auto i1 = ext::make_shared<Uniform1dMesher>(-2.0, 2.0, 9);
    auto mesher1d = ext::make_shared<FdmMesherComposite>(i1);
    Array f1(mesher1d->layout()->size());
    for (const auto& iter : *mesher1d->layout()) {
        const Real x = mesher1d->location(iter, 0);
        f1[iter.index()] = std::exp(-x * x);
    }
    FdmMesherIntegral mi1(mesher1d, simpson);
    put("mesher_integral_1d_simpson", mi1.integrate(f1));
    FdmMesherIntegral mi1t(mesher1d, trapezoid);
    put("mesher_integral_1d_trapezoid", mi1t.integrate(f1));

    auto i2 = ext::make_shared<Uniform1dMesher>(-1.5, 1.5, 7);
    auto mesher2d = ext::make_shared<FdmMesherComposite>(i1, i2);
    Array f2(mesher2d->layout()->size());
    for (const auto& iter : *mesher2d->layout()) {
        const Real x = mesher2d->location(iter, 0);
        const Real y = mesher2d->location(iter, 1);
        f2[iter.index()] = std::exp(-x * x - 0.5 * y * y) * (1.0 + 0.25 * x * y);
    }
    FdmMesherIntegral mi2(mesher2d, simpson);
    put("mesher_integral_2d_simpson", mi2.integrate(f2));
    FdmMesherIntegral mi2t(mesher2d, trapezoid);
    put("mesher_integral_2d_trapezoid", mi2t.integrate(f2));

    auto i3 = ext::make_shared<Uniform1dMesher>(0.0, 1.0, 5);
    auto mesher3d = ext::make_shared<FdmMesherComposite>(i1, i2, i3);
    Array f3(mesher3d->layout()->size());
    for (const auto& iter : *mesher3d->layout()) {
        const Real x = mesher3d->location(iter, 0);
        const Real y = mesher3d->location(iter, 1);
        const Real z = mesher3d->location(iter, 2);
        f3[iter.index()] = std::exp(-x * x - 0.5 * y * y) * (1.0 + z);
    }
    FdmMesherIntegral mi3(mesher3d, simpson);
    put("mesher_integral_3d_simpson", mi3.integrate(f3));
}

// ---------------------------------------------------------------- block G
// FdmQuantoHelper
__attribute__((noinline)) void block_quanto_helper(const Date& today, const DayCounter& dc) {
    auto rTS = ext::make_shared<FlatForward>(today, 0.05, dc);
    auto fTS = ext::make_shared<FlatForward>(today, 0.03, dc);
    auto fxVolTS = ext::make_shared<BlackConstantVol>(today, NullCalendar(), 0.15, dc);

    FdmQuantoHelper helper(rTS, fTS, fxVolTS, -0.75, 1.25);
    put("quanto_adjustment_scalar_v025", helper.quantoAdjustment(0.25, 0.5, 1.5));
    put("quanto_adjustment_scalar_v040", helper.quantoAdjustment(0.40, 0.0, 1.0));

    Array vols(4);
    vols[0] = 0.10;
    vols[1] = 0.20;
    vols[2] = 0.30;
    vols[3] = 0.45;
    put_array("quanto_adjustment_array", helper.quantoAdjustment(vols, 0.5, 1.5));
}

// ---------------------------------------------------------------- block H
// EscrowedDividendAdjustment
struct ToTime {
    Date reference;
    DayCounter dc;
    Real operator()(Date d) const { return dc.yearFraction(reference, d); }
};

DividendSchedule sample_dividends(const Date& today) {
    std::vector<Date> dates{today + 90, today + 250, today + 500};
    std::vector<Real> amounts{2.5, 3.0, 4.0};
    return DividendVector(dates, amounts);
}

ext::shared_ptr<EscrowedDividendAdjustment>
make_escrowed(const Date& today, const DayCounter& dc) {
    Handle<YieldTermStructure> rTS(ext::make_shared<FlatForward>(today, 0.05, dc));
    Handle<YieldTermStructure> qTS(ext::make_shared<FlatForward>(today, 0.02, dc));
    return ext::make_shared<EscrowedDividendAdjustment>(
        sample_dividends(today), rTS, qTS,
        std::function<Real(Date)>(ToTime{today, dc}), 1.0);
}

__attribute__((noinline)) void block_escrowed_dividend_adjustment(const Date& today,
                                                                  const DayCounter& dc) {
    const auto eda = make_escrowed(today, dc);
    put("escrowed_div_adj_t000", eda->dividendAdjustment(0.0));
    put("escrowed_div_adj_t025", eda->dividendAdjustment(0.25));
    put("escrowed_div_adj_t050", eda->dividendAdjustment(0.50));
    put("escrowed_div_adj_t080", eda->dividendAdjustment(0.80));
    put("escrowed_div_adj_t100", eda->dividendAdjustment(1.00));
    put("escrowed_div_adj_risk_free_discount_1y", eda->riskFreeRate()->discount(1.0));
    put("escrowed_div_adj_dividend_yield_discount_1y", eda->dividendYield()->discount(1.0));
}

// ---------------------------------------------------------------- block I/J
// FdmEscrowedLogInnerValueCalculator + FdmShoutLogInnerValueCalculator
__attribute__((noinline)) void block_escrowed_and_shout(const Date& today, const DayCounter& dc) {
    const auto eda = make_escrowed(today, dc);
    auto lm = ext::make_shared<Uniform1dMesher>(std::log(50.0), std::log(150.0), 11);
    auto mesher = ext::make_shared<FdmMesherComposite>(lm);
    auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0);

    FdmEscrowedLogInnerValueCalculator escrowedCalc(eda, payoff, mesher, 0);
    std::vector<Real> inner025, inner050, avg025;
    for (const auto& iter : *mesher->layout()) {
        inner025.push_back(escrowedCalc.innerValue(iter, 0.25));
        inner050.push_back(escrowedCalc.innerValue(iter, 0.50));
        avg025.push_back(escrowedCalc.avgInnerValue(iter, 0.25));
    }
    put_reals("escrowed_log_inner_value_t025", inner025);
    put_reals("escrowed_log_inner_value_t050", inner050);
    put_reals("escrowed_log_avg_inner_value_t025", avg025);

    auto putPayoff = ext::make_shared<PlainVanillaPayoff>(Option::Put, 100.0);
    FdmEscrowedLogInnerValueCalculator escrowedPut(eda, putPayoff, mesher, 0);
    std::vector<Real> putInner;
    for (const auto& iter : *mesher->layout()) {
        putInner.push_back(escrowedPut.innerValue(iter, 0.25));
    }
    put_reals("escrowed_log_inner_value_put_t025", putInner);

    Handle<BlackVolTermStructure> volTS(
        ext::make_shared<BlackConstantVol>(today, NullCalendar(), 0.25, dc));
    FdmShoutLogInnerValueCalculator shoutCalc(volTS, eda, 1.0, payoff, mesher, 0);
    std::vector<Real> shout025, shout075, shoutAvg;
    for (const auto& iter : *mesher->layout()) {
        shout025.push_back(shoutCalc.innerValue(iter, 0.25));
        shout075.push_back(shoutCalc.innerValue(iter, 0.75));
        shoutAvg.push_back(shoutCalc.avgInnerValue(iter, 0.25));
    }
    put_reals("shout_log_inner_value_call_t025", shout025);
    put_reals("shout_log_inner_value_call_t075", shout075);
    put_reals("shout_log_avg_inner_value_call_t025", shoutAvg);

    FdmShoutLogInnerValueCalculator shoutPut(volTS, eda, 1.0, putPayoff, mesher, 0);
    std::vector<Real> shoutPutVals;
    for (const auto& iter : *mesher->layout()) {
        shoutPutVals.push_back(shoutPut.innerValue(iter, 0.25));
    }
    put_reals("shout_log_inner_value_put_t025", shoutPutVals);
}

// ---------------------------------------------------------------- block K
// FdmDividendHandler
__attribute__((noinline)) void block_dividend_handler(const Date& today, const DayCounter& dc) {
    const DividendSchedule schedule = sample_dividends(today);

    auto lm = ext::make_shared<Uniform1dMesher>(std::log(50.0), std::log(150.0), 11);
    auto mesher1d = ext::make_shared<FdmMesherComposite>(lm);
    FdmDividendHandler handler(schedule, mesher1d, today, dc, 0);

    put_reals("dividend_handler_times", handler.dividendTimes());
    put_reals("dividend_handler_amounts", handler.dividends());
    std::vector<Real> dateSerials;
    for (const auto& d : handler.dividendDates()) dateSerials.push_back(Real(d.serialNumber()));
    put_reals("dividend_handler_date_serials", dateSerials);

    // grid values in physical (spot) units
    Array a(mesher1d->layout()->size());
    for (const auto& iter : *mesher1d->layout()) {
        a[iter.index()] = std::exp(mesher1d->location(iter, 0));
    }
    Array a0 = a;
    handler.applyTo(a, handler.dividendTimes()[0]);
    put_array("dividend_handler_1d_applied_div0", a);

    Array aNoop = a0;
    handler.applyTo(aNoop, 0.123456);
    put_array("dividend_handler_1d_no_dividend", aNoop);

    Array a2 = a0;
    handler.applyTo(a2, handler.dividendTimes()[2]);
    put_array("dividend_handler_1d_applied_div2", a2);

    // 2-D: equity direction 0, second direction is an inert variance-like axis
    auto ym = ext::make_shared<Uniform1dMesher>(0.0, 1.0, 4);
    auto mesher2d = ext::make_shared<FdmMesherComposite>(lm, ym);
    FdmDividendHandler handler2d(schedule, mesher2d, today, dc, 0);
    Array b(mesher2d->layout()->size());
    for (const auto& iter : *mesher2d->layout()) {
        b[iter.index()] =
            std::exp(mesher2d->location(iter, 0)) + 10.0 * mesher2d->location(iter, 1);
    }
    handler2d.applyTo(b, handler2d.dividendTimes()[0]);
    put_array("dividend_handler_2d_applied_div0", b);
}

// ---------------------------------------------------------------- block L
// FdmAffineModelTermStructure
__attribute__((noinline)) void block_affine_model_term_structure(const Date& today,
                                                                 const DayCounter& dc) {
    Handle<YieldTermStructure> ts(ext::make_shared<FlatForward>(today, 0.05, dc));
    auto hw = ext::make_shared<HullWhite>(ts, 0.1, 0.01);

    const Date refDate = today + 180;
    Array r(1, 0.04);
    FdmAffineModelTermStructure hwTs(r, NullCalendar(), dc, refDate, today, hw);
    put("affine_ts_hw_reference_serial", Real(hwTs.referenceDate().serialNumber()));
    put("affine_ts_hw_max_date_serial", Real(hwTs.maxDate().serialNumber()));
    put("affine_ts_hw_discount_05", hwTs.discount(0.5));
    put("affine_ts_hw_discount_10", hwTs.discount(1.0));
    put("affine_ts_hw_discount_30", hwTs.discount(3.0));
    hwTs.setVariable(Array(1, 0.06));
    put("affine_ts_hw_discount_10_after_set", hwTs.discount(1.0));

    auto g2 = ext::make_shared<G2>(ts, 0.1, 0.01, 0.1, 0.012, -0.75);
    Array r2(2);
    r2[0] = 0.01;
    r2[1] = 0.02;
    FdmAffineModelTermStructure g2Ts(r2, NullCalendar(), dc, refDate, today, g2);
    put("affine_ts_g2_discount_05", g2Ts.discount(0.5));
    put("affine_ts_g2_discount_10", g2Ts.discount(1.0));
    Array r2b(2);
    r2b[0] = -0.005;
    r2b[1] = 0.03;
    g2Ts.setVariable(r2b);
    put("affine_ts_g2_discount_10_after_set", g2Ts.discount(1.0));
}

// ---------------------------------------------------------------- block M
// FdmAffineModelSwapInnerValue
__attribute__((noinline)) void block_affine_model_swap_inner_value(const Date& today,
                                                                   const DayCounter& dc) {
    Handle<YieldTermStructure> disTs(ext::make_shared<FlatForward>(today, 0.045, dc));
    Handle<YieldTermStructure> fwdTs(ext::make_shared<FlatForward>(today, 0.050, dc));

    auto index = ext::make_shared<IborIndex>("dummy", Period(6, Months), 0, EURCurrency(),
                                             NullCalendar(), Following, false, dc, fwdTs);

    const Date start(15, January, 2025);
    const Date end(15, January, 2030);
    Schedule fixedSchedule(start, end, Period(1, Years), NullCalendar(), Following, Following,
                           DateGeneration::Forward, false);
    Schedule floatSchedule(start, end, Period(6, Months), NullCalendar(), Following, Following,
                           DateGeneration::Forward, false);

    auto swap = ext::make_shared<VanillaSwap>(Swap::Payer, 1000.0, fixedSchedule, 0.05, dc,
                                              floatSchedule, index, 0.0, dc);

    auto disModel = ext::make_shared<HullWhite>(disTs, 0.05, 0.0075);
    auto fwdModel = ext::make_shared<HullWhite>(fwdTs, 0.05, 0.0075);

    std::map<Time, Date> t2d;
    const Time tEx = dc.yearFraction(today, start);
    t2d[tEx] = start;

    auto mesher = ext::make_shared<FdmMesherComposite>(
        ext::make_shared<Uniform1dMesher>(-0.04, 0.04, 5));

    FdmAffineModelSwapInnerValue<HullWhite> calc(disModel, fwdModel, swap, t2d, mesher, 0);
    std::vector<Real> hwInner, hwAvg;
    for (const auto& iter : *mesher->layout()) {
        hwInner.push_back(calc.innerValue(iter, tEx));
    }
    for (const auto& iter : *mesher->layout()) {
        hwAvg.push_back(calc.avgInnerValue(iter, tEx));
    }
    put("affine_swap_exercise_time", tEx);
    put_reals("affine_swap_hw_payer_inner", hwInner);
    put_reals("affine_swap_hw_payer_avg", hwAvg);

    auto recSwap = ext::make_shared<VanillaSwap>(Swap::Receiver, 1000.0, fixedSchedule, 0.05, dc,
                                                 floatSchedule, index, 0.0, dc);
    FdmAffineModelSwapInnerValue<HullWhite> recCalc(disModel, fwdModel, recSwap, t2d, mesher, 0);
    std::vector<Real> recInner;
    for (const auto& iter : *mesher->layout()) {
        recInner.push_back(recCalc.innerValue(iter, tEx));
    }
    put_reals("affine_swap_hw_receiver_inner", recInner);

    // G2 flavour: the state is read straight off two mesher directions
    auto g2Dis = ext::make_shared<G2>(disTs, 0.1, 0.01, 0.1, 0.012, -0.75);
    auto g2Fwd = ext::make_shared<G2>(fwdTs, 0.1, 0.01, 0.1, 0.012, -0.75);
    auto mesher2d = ext::make_shared<FdmMesherComposite>(
        ext::make_shared<Uniform1dMesher>(-0.02, 0.02, 3),
        ext::make_shared<Uniform1dMesher>(-0.03, 0.03, 3));
    FdmAffineModelSwapInnerValue<G2> g2Calc(g2Dis, g2Fwd, swap, t2d, mesher2d, 0);
    std::vector<Real> g2Inner;
    for (const auto& iter : *mesher2d->layout()) {
        g2Inner.push_back(g2Calc.innerValue(iter, tEx));
    }
    put_reals("affine_swap_g2_payer_inner", g2Inner);
}

}  // namespace

int main() {
    const DayCounter dc = Actual365Fixed();
    const Date today(15, January, 2024);
    Settings::instance().evaluationDate() = today;

    const auto rTS = ext::make_shared<FlatForward>(today, 0.05, dc);

    block_indices_on_boundary();
    block_dirichlet_boundary();
    block_time_dep_dirichlet();
    block_discount_dirichlet(rTS);
    block_inner_value_calculators();
    block_mesher_integral();
    block_quanto_helper(today, dc);
    block_escrowed_dividend_adjustment(today, dc);
    block_escrowed_and_shout(today, dc);
    block_dividend_handler(today, dc);
    block_affine_model_term_structure(today, dc);
    block_affine_model_swap_inner_value(today, dc);

    std::cout << "{\n";
    for (Size i = 0; i < g_entries.size(); ++i) {
        std::cout << g_entries[i];
        if (i + 1 != g_entries.size()) std::cout << ",";
        std::cout << "\n";
    }
    std::cout << "}\n";
    return 0;
}

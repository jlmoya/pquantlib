// migration-harness/cpp/probes/v143_experimental_misc/probe.cpp
//
// Reference values for the "misc" experimental coverage wave @ v1.43:
//
//   A  ql/experimental/commodities/*          — the nested `Data` pimpl /
//      flyweight structs (UnitOfMeasure::Data, CommodityType::Data,
//      PaymentTerm::Data, UnitOfMeasureConversion::Data) and the
//      ExchangeContract non-pimpl control.
//   B  ql/experimental/inflation/polynomial2Dspline.hpp — Polynomial2DSpline,
//      detail::Polynomial2DSplineImpl, and the `Polynomial` traits factory.
//   C  ql/experimental/models/normalclvmodel.hpp   — MappingFunction /
//      MappingFunction::InterpolationData.
//   D  ql/experimental/models/squarerootclvmodel.hpp — MappingFunction.
//   E  ql/experimental/shortrate/generalizedhullwhite.hpp —
//      InterpolationParameter (incl. the params()-aliasing semantics that is
//      the whole reason the class exists).
//   F  ql/experimental/finitedifferences/fdextoujumpvanillaengine.hpp and
//      fdklugeextouspreadengine.hpp — engine NPVs plus the grids the engines
//      build internally.
//
// WHY BLOCK A LOOKS THE WAY IT DOES
// ---------------------------------
// The four commodity `Data` structs are not inert payloads. Three of them are
// *flyweight* payloads held in a file-static std::map, so constructing a
// second object with a key that is already registered SILENTLY IGNORES the
// other constructor arguments and shares the first object's Data. That is
// observable behaviour, and it is what a Python port has to reproduce — so
// each is probed by constructing the same key twice with deliberately
// different secondary arguments and reading back the *first* object's values.
//
// UnitOfMeasureConversion::Data is a plain pimpl (no registry), but its
// two-conversion constructor leaves `commodityType` and `code` untouched,
// so a chained (Derived) conversion has an EMPTY commodity type and an EMPTY
// code. That is pinned explicitly (uomc_chain_*) because it is exactly the
// kind of detail a port "tidies up" by accident.
//
// Emits JSON on stdout; redirect to
// references/v143/experimental/misc.json.

#include <ql/experimental/commodities/commoditytype.hpp>
#include <ql/experimental/commodities/exchangecontract.hpp>
#include <ql/experimental/commodities/paymentterm.hpp>
#include <ql/experimental/commodities/quantity.hpp>
#include <ql/experimental/commodities/unitofmeasure.hpp>
#include <ql/experimental/commodities/unitofmeasureconversion.hpp>

#include <ql/experimental/inflation/polynomial2Dspline.hpp>

#include <ql/experimental/models/normalclvmodel.hpp>
#include <ql/experimental/models/squarerootclvmodel.hpp>

#include <ql/experimental/shortrate/generalizedhullwhite.hpp>

#include <ql/experimental/finitedifferences/fdextoujumpvanillaengine.hpp>
#include <ql/experimental/finitedifferences/fdklugeextouspreadengine.hpp>
#include <ql/experimental/processes/extendedornsteinuhlenbeckprocess.hpp>
#include <ql/experimental/processes/extouwithjumpsprocess.hpp>
#include <ql/experimental/processes/klugeextouprocess.hpp>

#include <ql/instruments/basketoption.hpp>
#include <ql/instruments/vanillaoption.hpp>
#include <ql/exercise.hpp>
#include <ql/methods/finitedifferences/meshers/exponentialjump1dmesher.hpp>
#include <ql/methods/finitedifferences/meshers/fdmsimpleprocess1dmesher.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/processes/ornsteinuhlenbeckprocess.hpp>
#include <ql/processes/squarerootprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/daycounters/actualactual.hpp>

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

// --------------------------------------------------------------------------
// A — commodities: the nested `Data` pimpl / flyweight structs
// --------------------------------------------------------------------------

void blockA() {
    // --- A1: UnitOfMeasure::Data — flyweight keyed on NAME -----------------
    // unitofmeasure.cpp:37-49: the ctor looks `name` up in unitsOfMeasure_ and
    // reuses the stored Data if present, so `code` and `unitType` of the
    // SECOND construction are discarded.
    const UnitOfMeasure bbl("Barrels", "BBL", UnitOfMeasure::Volume);
    const UnitOfMeasure bblAgain("Barrels", "ZZZ", UnitOfMeasure::Mass);

    emit_str("uom_bbl_name", bbl.name());
    emit_str("uom_bbl_code", bbl.code());
    emit_int("uom_bbl_unit_type", (long long)bbl.unitType());
    // Flyweight sharing: the second construction gets the FIRST one's payload.
    emit_str("uom_bbl_again_code", bblAgain.code());
    emit_int("uom_bbl_again_unit_type", (long long)bblAgain.unitType());
    emit_bool("uom_bbl_again_equal", bbl == bblAgain);
    emit_bool("uom_default_empty", UnitOfMeasure().empty());
    emit_bool("uom_bbl_empty", bbl.empty());
    // Data's default rounding argument is `Rounding(0)` (unitofmeasure.hpp:86)
    // — precision 0, type Closest, digit 5 — NOT the no-op `Rounding()`.
    emit_int("uom_bbl_rounding_type", (long long)bbl.rounding().type());
    emit_int("uom_bbl_rounding_precision", (long long)bbl.rounding().precision());
    emit_int("uom_bbl_rounding_digit", (long long)bbl.rounding().roundingDigit());
    emit_bool("uom_bbl_triangulation_empty", bbl.triangulationUnitOfMeasure().empty());

    const LotUnitOfMeasure lot;
    emit_str("uom_lot_name", lot.name());
    emit_str("uom_lot_code", lot.code());
    emit_int("uom_lot_unit_type", (long long)lot.unitType());

    // --- A2: CommodityType::Data — flyweight keyed on the 2nd ctor arg -----
    // commoditytype.cpp:28-38 defines `CommodityType(name, code)` and keys the
    // map on `code`, while the header declares `(code, name)`. Positional
    // binding wins: arg1 -> name(), arg2 -> code().
    const CommodityType ct("Heating Oil", "HO");
    emit_str("ct_name", ct.name());
    emit_str("ct_code", ct.code());
    const CommodityType ctAgain("Something Else", "HO");
    emit_str("ct_again_name", ctAgain.name());  // shared Data -> "Heating Oil"
    emit_bool("ct_again_equal", ct == ctAgain);
    emit_bool("ct_default_empty", CommodityType().empty());
    const NullCommodityType nullCt;
    emit_str("ct_null_name", nullCt.name());
    emit_str("ct_null_code", nullCt.code());

    // --- A3: PaymentTerm::Data — flyweight keyed on NAME -------------------
    const PaymentTerm pt("Pricing end + 5 days", PaymentTerm::PricingDate, 5, TARGET());
    const PaymentTerm ptAgain("Pricing end + 5 days", PaymentTerm::TradeDate, 99,
                              NullCalendar());
    emit_str("pt_name", pt.name());
    emit_int("pt_event_type", (long long)pt.eventType());
    emit_int("pt_offset_days", (long long)pt.offsetDays());
    emit_str("pt_calendar", pt.calendar().name());
    // Shared payload: offset/event/calendar of the SECOND construction lost.
    emit_int("pt_again_offset_days", (long long)ptAgain.offsetDays());
    emit_int("pt_again_event_type", (long long)ptAgain.eventType());
    emit_str("pt_again_calendar", ptAgain.calendar().name());
    emit_bool("pt_again_equal", pt == ptAgain);
    emit_bool("pt_default_empty", PaymentTerm().empty());
    // 30 Dec 2023 is a Saturday; +5 days = 4 Jan 2024 (a TARGET business day).
    emit_int("pt_payment_date_serial",
             (long long)pt.getPaymentDate(Date(30, December, 2023)).serialNumber());
    // 27 Dec 2023 + 5 = 1 Jan 2024, a TARGET holiday -> adjusted forward.
    emit_int("pt_payment_date_adjusted_serial",
             (long long)pt.getPaymentDate(Date(27, December, 2023)).serialNumber());

    // --- A4: UnitOfMeasureConversion::Data — plain pimpl, no registry ------
    const UnitOfMeasure gal("Gallons", "GAL", UnitOfMeasure::Volume);
    const UnitOfMeasure ltr("Litres", "LTR", UnitOfMeasure::Volume);

    const UnitOfMeasureConversion c1(ct, bbl, gal, 42.0);
    emit_str("uomc_c1_code", c1.code());  // commodityType.name() + src + tgt
    emit_int("uomc_c1_type", (long long)c1.type());
    emit("uomc_c1_factor", c1.conversionFactor());
    emit_str("uomc_c1_source", c1.source().code());
    emit_str("uomc_c1_target", c1.target().code());
    emit_str("uomc_c1_commodity_code", c1.commodityType().code());

    // Copy shares the pimpl (ext::shared_ptr<Data> data_).
    const UnitOfMeasureConversion c1copy = c1;
    emit_bool("uomc_c1_copy_same_code", c1copy.code() == c1.code());
    emit("uomc_c1_copy_factor", c1copy.conversionFactor());

    const UnitOfMeasureConversion c2(ct, gal, ltr, 3.785411784);

    // chain(): r1.target == r2.source -> factor multiplies.
    const UnitOfMeasureConversion chained = UnitOfMeasureConversion::chain(c1, c2);
    emit_int("uomc_chain_type", (long long)chained.type());
    emit("uomc_chain_factor", chained.conversionFactor());
    emit_str("uomc_chain_source", chained.source().code());
    emit_str("uomc_chain_target", chained.target().code());
    // C++ quirk: Data(r1,r2) (unitofmeasureconversion.cpp:53-58) sets ONLY
    // conversionFactorChain, so `code` stays default-constructed ("") and
    // `commodityType` stays EMPTY even though chain() fills in everything else.
    emit_str("uomc_chain_code", chained.code());
    emit_bool("uomc_chain_commodity_type_empty", chained.commodityType().empty());

    const Quantity q(ct, bbl, 2.0);
    const Quantity converted = chained.convert(q);
    emit("uomc_chain_convert_amount", converted.amount());
    emit_str("uomc_chain_convert_uom", converted.unitOfMeasure().code());
    const Quantity back = chained.convert(converted);
    emit("uomc_chain_convert_back_amount", back.amount());
    emit_str("uomc_chain_convert_back_uom", back.unitOfMeasure().code());

    // Direct conversion, both directions.
    emit("uomc_c1_convert_fwd", c1.convert(Quantity(ct, bbl, 3.0)).amount());
    emit("uomc_c1_convert_inv", c1.convert(Quantity(ct, gal, 84.0)).amount());

    // Quantity::rounded() (quantity.hpp:140-144) applies the UOM's Data
    // rounding, so it is a direct read-out of the `Rounding(0)` default: NOT a
    // no-op — it rounds to zero decimals, half away from zero.
    emit("uom_quantity_rounded_1p23456789",
         Quantity(NullCommodityType(), bbl, 1.23456789).rounded().amount());
    emit("uom_quantity_rounded_2p5", Quantity(NullCommodityType(), bbl, 2.5).rounded().amount());
    emit("uom_quantity_rounded_m1p5",
         Quantity(NullCommodityType(), bbl, -1.5).rounded().amount());

    // chain() with r1.source == r2.source -> factor divides.
    const UnitOfMeasureConversion c3(ct, bbl, ltr, 158.987294928);
    const UnitOfMeasureConversion chained2 = UnitOfMeasureConversion::chain(c1, c3);
    emit("uomc_chain2_factor", chained2.conversionFactor());
    emit_str("uomc_chain2_source", chained2.source().code());
    emit_str("uomc_chain2_target", chained2.target().code());

    // --- A5: ExchangeContract — the control. It has NO nested `Data`. ------
    // exchangecontract.hpp:33-50 stores four plain protected members.
    const ExchangeContract xc("CLZ24", Date(20, November, 2024), Date(1, December, 2024),
                              Date(31, December, 2024));
    emit_str("xc_code", xc.code());
    emit_int("xc_expiration_serial", (long long)xc.expirationDate().serialNumber());
    emit_int("xc_underlying_start_serial",
             (long long)xc.underlyingStartDate().serialNumber());
    emit_int("xc_underlying_end_serial", (long long)xc.underlyingEndDate().serialNumber());
    emit_bool("xc_default_expiration_null", ExchangeContract().expirationDate() == Date());
}

// --------------------------------------------------------------------------
// B — Polynomial2DSpline / Polynomial2DSplineImpl / Polynomial (factory)
// --------------------------------------------------------------------------

void blockB() {
    const std::vector<Real> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    const std::vector<Real> y = {0.01, 0.02, 0.03, 0.04};
    Matrix z(y.size(), x.size());
    for (Size i = 0; i < y.size(); ++i)
        for (Size k = 0; k < x.size(); ++k)
            z[i][k] = 100.0 * y[i] * y[i] + 5.0 * x[k] + 2.0 * y[i] * x[k];

    Polynomial2DSpline s(x.begin(), x.end(), y.begin(), y.end(), z);
    s.enableExtrapolation();

    // Pillars: both 1-D interpolations are exact at their own nodes.
    emit("poly_pillar_x1_y001", s(1.0, 0.01, true));
    emit("poly_pillar_x3_y002", s(3.0, 0.02, true));
    emit("poly_pillar_x5_y004", s(5.0, 0.04, true));
    emit("poly_pillar_x2_y003", s(2.0, 0.03, true));

    // Interior — this is where the parabolic-in-y slopes and the natural
    // cubic spline in x actually bite.
    emit("poly_interior_x25_y0025", s(2.5, 0.025, true));
    emit("poly_interior_x42_y0015", s(4.2, 0.015, true));
    emit("poly_interior_x15_y0035", s(1.5, 0.035, true));
    emit("poly_interior_x35_y0022", s(3.5, 0.022, true));

    // Extrapolation on both axes (the impl always calls the inner
    // interpolations with allowExtrapolation=true).
    emit("poly_extrap_x0_y0005", s(0.0, 0.005, true));
    emit("poly_extrap_x6_y005", s(6.0, 0.05, true));
    emit("poly_extrap_x3_y0005", s(3.0, 0.005, true));

    // The whole `section` vector at a fixed y — i.e. every column's Parabolic
    // interpolation sampled at y. This isolates Polynomial2DSplineImpl's first
    // half (polynomial2Dspline.hpp:51-61, the `polynomials_` vector) from the
    // natural cubic spline in x that consumes it.
    {
        std::vector<Real> section, sectionExtrap;
        for (Size k = 0; k < x.size(); ++k) {
            std::vector<Real> col(y.size());
            for (Size i = 0; i < y.size(); ++i)
                col[i] = z[i][k];
            const Parabolic poly(y.begin(), y.end(), col.begin());
            section.push_back(poly(0.025, true));
            sectionExtrap.push_back(poly(0.005, true));
        }
        emit_arr("poly_section_y0025", section);
        emit_arr("poly_section_y0005", sectionExtrap);
    }

    // The `Polynomial` traits/factory class (polynomial2Dspline.hpp:95-103).
    const Polynomial factory;
    Interpolation2D viaFactory =
        factory.interpolate(x.begin(), x.end(), y.begin(), y.end(), z);
    viaFactory.enableExtrapolation();
    emit("poly_factory_interior_x25_y0025", viaFactory(2.5, 0.025, true));
    emit("poly_factory_pillar_x3_y002", viaFactory(3.0, 0.02, true));
    emit_bool("poly_factory_matches_direct",
              viaFactory(2.5, 0.025, true) == s(2.5, 0.025, true));

    // Grid metadata the Interpolation2D surface publishes.
    emit("poly_xmin", viaFactory.xMin());
    emit("poly_xmax", viaFactory.xMax());
    emit("poly_ymin", viaFactory.yMin());
    emit("poly_ymax", viaFactory.yMax());
    emit_bool("poly_is_in_range_inside", viaFactory.isInRange(2.5, 0.025));
    emit_bool("poly_is_in_range_outside", viaFactory.isInRange(6.0, 0.025));
}

// --------------------------------------------------------------------------
// C / D — the two CLV models' MappingFunction (and InterpolationData)
// --------------------------------------------------------------------------

Date clvRefDate() { return {15, January, 2024}; }

ext::shared_ptr<GeneralizedBlackScholesProcess> clvBsProcess() {
    const Date ref = clvRefDate();
    const DayCounter dc = Actual365Fixed();
    const Handle<Quote> spot(ext::make_shared<SimpleQuote>(100.0));
    const Handle<YieldTermStructure> qTS(
        ext::make_shared<FlatForward>(ref, Handle<Quote>(ext::make_shared<SimpleQuote>(0.02)), dc));
    const Handle<YieldTermStructure> rTS(
        ext::make_shared<FlatForward>(ref, Handle<Quote>(ext::make_shared<SimpleQuote>(0.05)), dc));
    const Handle<BlackVolTermStructure> volTS(
        ext::make_shared<BlackConstantVol>(ref, TARGET(), 0.25, dc));
    return ext::make_shared<GeneralizedBlackScholesProcess>(spot, qTS, rTS, volTS);
}

void blockC() {
    Settings::instance().evaluationDate() = clvRefDate();
    const Date ref = clvRefDate();

    const ext::shared_ptr<GeneralizedBlackScholesProcess> bs = clvBsProcess();
    const ext::shared_ptr<OrnsteinUhlenbeckProcess> ou =
        ext::make_shared<OrnsteinUhlenbeckProcess>(0.5, 0.2, 0.1, 0.0);

    const std::vector<Date> mats = {ref + Period(6, Months), ref + Period(1, Years),
                                    ref + Period(2, Years)};

    const NormalCLVModel model(bs, ou, mats, 5);

    // Collocation points at every maturity — the columns of
    // MappingFunction::InterpolationData::s_.
    for (Size j = 0; j < mats.size(); ++j) {
        const Array cx = model.collocationPointsX(mats[j]);
        const Array cy = model.collocationPointsY(mats[j]);
        emit_arr("nclv_cx_" + std::to_string(j), std::vector<Real>(cx.begin(), cx.end()));
        emit_arr("nclv_cy_" + std::to_string(j), std::vector<Real>(cy.begin(), cy.end()));
    }
    emit_arr("nclv_maturity_times",
             {bs->time(mats[0]), bs->time(mats[1]), bs->time(mats[2])});

    // g(t, x): exactly at each maturity, between maturities (the
    // LinearInterpolation across t inside InterpolationData::interpl_), and
    // outside the maturity span (interpl_ is queried with extrapolation on).
    const std::function<Real(Time, Real)> g = model.g();
    for (Size j = 0; j < mats.size(); ++j) {
        const Time t = bs->time(mats[j]);
        emit("nclv_g_mat" + std::to_string(j) + "_x0", g(t, 0.1));
        emit("nclv_g_mat" + std::to_string(j) + "_xp", g(t, 0.35));
        emit("nclv_g_mat" + std::to_string(j) + "_xn", g(t, -0.15));
    }
    emit("nclv_g_mid_9m", g(bs->time(ref + Period(9, Months)), 0.05));
    emit("nclv_g_mid_18m", g(bs->time(ref + Period(18, Months)), 0.05));
    // Extrapolation in t: LinearInterpolation(..., true) — no guard in C++.
    emit("nclv_g_extrap_3m", g(bs->time(ref + Period(3, Months)), 0.05));
    emit("nclv_g_extrap_3y", g(bs->time(ref + Period(3, Years)), 0.05));

    // Calling g() twice must return the SAME mapping (LazyObject caches g_,
    // and MappingFunction copies share their InterpolationData shared_ptr).
    const std::function<Real(Time, Real)> g2 = model.g();
    emit_bool("nclv_g_stable", g2(bs->time(mats[1]), 0.1) == g(bs->time(mats[1]), 0.1));

    emit("nclv_cdf_1y_k100", model.cdf(mats[1], 100.0));
    emit("nclv_invcdf_1y_q05", model.invCDF(mats[1], 0.5));
}

void blockD() {
    Settings::instance().evaluationDate() = clvRefDate();
    const Date ref = clvRefDate();

    const ext::shared_ptr<GeneralizedBlackScholesProcess> bs = clvBsProcess();
    // SquareRootProcess(b = theta, a = kappa, sigma, x0)
    const ext::shared_ptr<SquareRootProcess> sqp =
        ext::make_shared<SquareRootProcess>(0.09, 1.0, 0.2, 0.09);

    const std::vector<Date> mats = {ref + Period(1, Years), ref + Period(2, Years)};
    const SquareRootCLVModel model(bs, sqp, mats, 5);

    for (Size j = 0; j < mats.size(); ++j) {
        const Array cx = model.collocationPointsX(mats[j]);
        const Array cy = model.collocationPointsY(mats[j]);
        emit_arr("sclv_cx_" + std::to_string(j), std::vector<Real>(cx.begin(), cx.end()));
        emit_arr("sclv_cy_" + std::to_string(j), std::vector<Real>(cy.begin(), cy.end()));
    }
    emit_arr("sclv_maturity_times", {bs->time(mats[0]), bs->time(mats[1])});

    const std::function<Real(Time, Real)> g = model.g();
    // exact maturity hits (close_enough branch)
    const Array cx0 = model.collocationPointsX(mats[0]);
    const Array cx1 = model.collocationPointsX(mats[1]);
    emit("sclv_g_mat0_xmid", g(bs->time(mats[0]), cx0[2]));
    emit("sclv_g_mat0_xlo", g(bs->time(mats[0]), cx0[0]));
    emit("sclv_g_mat1_xmid", g(bs->time(mats[1]), cx1[2]));
    // in between the two maturities -> linear blend of the two Lagrange curves
    emit("sclv_g_mid_18m", g(bs->time(ref + Period(18, Months)), cx0[2]));
    emit("sclv_g_mid_15m", g(bs->time(ref + Period(15, Months)), cx0[1]));

    emit("sclv_cdf_1y_k100", model.cdf(mats[0], 100.0));
    emit("sclv_invcdf_1y_q05", model.invCDF(mats[0], 0.5));
}

// --------------------------------------------------------------------------
// E — InterpolationParameter
// --------------------------------------------------------------------------

void blockE() {
    // E1: the class in isolation. The point of InterpolationParameter is that
    // the Interpolation it holds is built over `params()`' own storage, so
    // setParam() moves the curve without any rebuild
    // (generalizedhullwhite.hpp:37-61 + :194-212).
    InterpolationParameter p(3, NoConstraint());
    p.setParam(0, 0.05);
    p.setParam(1, 0.10);
    p.setParam(2, 0.20);

    std::vector<Time> pillars = {0.0, 1.0, 3.0};
    Interpolation interp =
        LinearFlat().interpolate(pillars.begin(), pillars.end(), p.params().begin());
    interp.enableExtrapolation();
    p.reset(interp);

    emit_int("ip_size", (long long)p.size());
    emit_arr("ip_params", std::vector<Real>(p.params().begin(), p.params().end()));
    std::vector<Real> before;
    for (Time t : {-1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0})
        before.push_back(p(t));
    emit_arr("ip_values_before", before);

    // Aliasing: change a free parameter, the interpolation follows.
    p.setParam(1, 0.50);
    std::vector<Real> after;
    for (Time t : {-1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0})
        after.push_back(p(t));
    emit_arr("ip_values_after_setparam", after);

    // Constraint plumbing (Parameter base).
    Array feasible(3);
    feasible[0] = 0.01;
    feasible[1] = 0.02;
    feasible[2] = 0.03;
    Array infeasible(3);
    infeasible[0] = -0.01;
    infeasible[1] = 0.02;
    infeasible[2] = 0.03;
    emit_bool("ip_noconstraint_accepts_negative", p.testParams(infeasible));

    InterpolationParameter pos(3, PositiveConstraint());
    emit_bool("ip_positive_accepts_feasible", pos.testParams(feasible));
    emit_bool("ip_positive_rejects_negative", pos.testParams(infeasible));

    // E2: as used by GeneralizedHullWhite — a_ / sigma_ ARE
    // InterpolationParameters bound to arguments_[0] / arguments_[1].
    Settings::instance().evaluationDate() = clvRefDate();
    const Date ref = clvRefDate();
    const DayCounter dc = Actual365Fixed();
    const Handle<YieldTermStructure> yts(
        ext::make_shared<FlatForward>(ref, Handle<Quote>(ext::make_shared<SimpleQuote>(0.04)), dc));

    const std::vector<Date> speedDates = {ref, ref + Period(1, Years), ref + Period(3, Years)};
    const std::vector<Date> volDates = {ref, ref + Period(2, Years), ref + Period(5, Years)};
    const std::vector<Real> speeds = {0.05, 0.10, 0.20};
    const std::vector<Real> vols = {0.010, 0.015, 0.020};

    GeneralizedHullWhite ghw(yts, speedDates, volDates, speeds, vols);

    emit("ghw_a0", ghw.a());
    emit("ghw_sigma0", ghw.sigma());
    std::vector<Real> speedCurve, volCurve;
    for (Time t : {0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0}) {
        speedCurve.push_back(ghw.speed()(t));
        volCurve.push_back(ghw.vol()(t));
    }
    emit_arr("ghw_speed_curve", speedCurve);
    emit_arr("ghw_vol_curve", volCurve);

    const std::vector<bool> fixedRev = ghw.fixedReversion();
    emit_int("ghw_fixed_reversion_size", (long long)fixedRev.size());
    {
        std::vector<Real> mask;
        for (bool b : fixedRev)
            mask.push_back(b ? 1.0 : 0.0);
        emit_arr("ghw_fixed_reversion", mask);
    }

    // The calibration hook: setParams() writes straight into the
    // InterpolationParameters' arrays, and the curves move with them.
    Array newParams(6);
    newParams[0] = 0.07;
    newParams[1] = 0.14;
    newParams[2] = 0.28;
    newParams[3] = 0.011;
    newParams[4] = 0.017;
    newParams[5] = 0.023;
    ghw.setParams(newParams);
    emit("ghw_a0_after", ghw.a());
    emit("ghw_sigma0_after", ghw.sigma());
    std::vector<Real> speedCurve2, volCurve2;
    for (Time t : {0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0}) {
        speedCurve2.push_back(ghw.speed()(t));
        volCurve2.push_back(ghw.vol()(t));
    }
    emit_arr("ghw_speed_curve_after", speedCurve2);
    emit_arr("ghw_vol_curve_after", volCurve2);
    {
        const Array ghwParams = ghw.params();
        emit_arr("ghw_params_after",
                 std::vector<Real>(ghwParams.begin(), ghwParams.end()));
    }
}

// --------------------------------------------------------------------------
// F — the two FD engines
// --------------------------------------------------------------------------

ext::shared_ptr<ExtOUWithJumpsProcess> createKlugeProcess() {
    // Same parameterisation as QuantLib's own test-suite/swingoption.cpp:59-74.
    Array x0(2);
    x0[0] = 3.0;
    x0[1] = 0.0;
    const Real beta = 5.0;
    const Real eta = 2.0;
    const Real jumpIntensity = 1.0;
    const Real speed = 1.0;
    const Real volatility = 2.0;

    const Real level = x0[0];
    const ext::shared_ptr<ExtendedOrnsteinUhlenbeckProcess> ouProcess =
        ext::make_shared<ExtendedOrnsteinUhlenbeckProcess>(
            speed, volatility, x0[0], [level](Real) { return level; });
    return ext::make_shared<ExtOUWithJumpsProcess>(ouProcess, x0[1], beta, jumpIntensity,
                                                   eta);
}

void blockF() {
    const Date today(18, December, 2011);
    Settings::instance().evaluationDate() = today;

    const DayCounter dc = ActualActual(ActualActual::ISDA);
    const Date maturityDate = today + Period(12, Months);
    const Time maturity = dc.yearFraction(today, maturityDate);
    emit("fd_maturity", maturity);
    emit_int("fd_today_serial", (long long)today.serialNumber());
    emit_int("fd_maturity_serial", (long long)maturityDate.serialNumber());

    const ext::shared_ptr<ExtOUWithJumpsProcess> jumpProcess = createKlugeProcess();
    const ext::shared_ptr<YieldTermStructure> rTS =
        ext::make_shared<FlatForward>(today, 0.1, dc);

    // The two 1-D meshers FdExtOUJumpVanillaEngine::calculate builds. Pinned
    // separately so a port can localise a grid mismatch before it shows up as
    // an NPV mismatch.
    const Size tGrid = 25, xGrid = 50, yGrid = 15;
    {
        const FdmSimpleProcess1dMesher xm(
            xGrid, jumpProcess->getExtendedOrnsteinUhlenbeckProcess(), maturity);
        emit_arr("fd_ouj_x_mesher", xm.locations());
        const ExponentialJump1dMesher ym(yGrid, jumpProcess->beta(),
                                         jumpProcess->jumpIntensity(),
                                         jumpProcess->eta());
        emit_arr("fd_ouj_y_mesher", ym.locations());
    }
    {
        // NB: initialValues() returns by value — bind it to a named Array
        // before taking iterators, or begin()/end() come from two different
        // temporaries.
        const Array iv = jumpProcess->initialValues();
        emit_arr("fd_ouj_initial_values", std::vector<Real>(iv.begin(), iv.end()));
    }

    const ext::shared_ptr<Exercise> exercise =
        ext::make_shared<EuropeanExercise>(maturityDate);

    for (Real strike : {20.0, 30.0, 40.0}) {
        for (int typeIdx = 0; typeIdx < 2; ++typeIdx) {
            const Option::Type type = typeIdx == 0 ? Option::Call : Option::Put;
            VanillaOption option(
                ext::make_shared<PlainVanillaPayoff>(type, strike), exercise);
            option.setPricingEngine(ext::make_shared<FdExtOUJumpVanillaEngine>(
                jumpProcess, rTS, tGrid, xGrid, yGrid));
            const std::string key = "fd_ouj_npv_" + std::string(typeIdx == 0 ? "call" : "put") +
                                    "_k" + std::to_string((int)strike);
            emit(key, option.NPV());
        }
    }
    emit_int("fd_ouj_t_grid", (long long)tGrid);
    emit_int("fd_ouj_x_grid", (long long)xGrid);
    emit_int("fd_ouj_y_grid", (long long)yGrid);

    // --- FdKlugeExtOUSpreadEngine (test-suite/vpp.cpp:403-451 setup) -------
    {
        const Real gSpeed = 1.0;
        const Volatility gVol = std::sqrt(1.4);
        const Real betaG = 0.0;
        const Real alphaG = 3.0;
        const Real x0G = 3.0;
        const Real heatRate = 2.0;
        const Real rho = 0.5;

        const ext::shared_ptr<YieldTermStructure> rTS0 =
            ext::make_shared<FlatForward>(today, 0.0, dc);
        const ext::shared_ptr<ExtendedOrnsteinUhlenbeckProcess> extOUProcess =
            ext::make_shared<ExtendedOrnsteinUhlenbeckProcess>(
                gSpeed, gVol, x0G,
                [alphaG, betaG](Real x) { return alphaG + betaG * x; },
                ExtendedOrnsteinUhlenbeckProcess::Trapezodial);
        const ext::shared_ptr<KlugeExtOUProcess> klugeOUProcess =
            ext::make_shared<KlugeExtOUProcess>(rho, createKlugeProcess(), extOUProcess);

        {
            const Array kiv = klugeOUProcess->initialValues();
            emit_arr("fd_kluge_initial_values",
                     std::vector<Real>(kiv.begin(), kiv.end()));
        }

        const Size ktGrid = 5, kxGrid = 25, kyGrid = 8, kuGrid = 10;
        {
            const FdmSimpleProcess1dMesher um(kuGrid, klugeOUProcess->getExtOUProcess(),
                                              maturity);
            emit_arr("fd_kluge_u_mesher", um.locations());
        }

        Array spreadFactors(2);
        spreadFactors[0] = 1.0;
        spreadFactors[1] = -heatRate;

        for (Real strike : {0.0, 2.0}) {
            for (int typeIdx = 0; typeIdx < 2; ++typeIdx) {
                const Option::Type type = typeIdx == 0 ? Option::Call : Option::Put;
                const ext::shared_ptr<BasketPayoff> basketPayoff =
                    ext::make_shared<AverageBasketPayoff>(
                        ext::make_shared<PlainVanillaPayoff>(type, strike), spreadFactors);
                BasketOption option(basketPayoff, exercise);
                option.setPricingEngine(ext::make_shared<FdKlugeExtOUSpreadEngine>(
                    klugeOUProcess, rTS0, ktGrid, kxGrid, kyGrid, kuGrid));
                const std::string key = "fd_kluge_npv_" +
                                        std::string(typeIdx == 0 ? "call" : "put") + "_k" +
                                        std::to_string((int)strike);
                emit(key, option.NPV());
            }
        }
        emit_int("fd_kluge_t_grid", (long long)ktGrid);
        emit_int("fd_kluge_x_grid", (long long)kxGrid);
        emit_int("fd_kluge_y_grid", (long long)kyGrid);
        emit_int("fd_kluge_u_grid", (long long)kuGrid);
        emit("fd_kluge_heat_rate", heatRate);
        emit("fd_kluge_rho", rho);
    }
}

}  // namespace

int main() {
    std::cout << "{\n";
    blockA();
    blockB();
    blockC();
    blockD();
    blockE();
    blockF();
    std::cout << "\n}\n";
    return 0;
}

// Phase 11 W7-B cluster probe: commodity foundation types + units of measure.
//
// Captures reference values for:
//
//   * UnitOfMeasureConversionManager::lookup — known petroleum conversion
//     factors (Barrel<->Litre, Barrel<->Gallon, MB<->Barrel, Kilolitre<->Barrel)
//     and the round-trip convert() on a Quantity.
//   * UnitOfMeasureConversion::convert — direct + inverse application of a
//     factor to a Quantity amount.
//   * Quantity arithmetic — same-UOM add/sub/mul/div and cross-UOM add under
//     AutomatedConversion (routed through the manager).
//   * CommodityType / UnitOfMeasure inspectors — code/name round-trips
//     (the flyweight identity is asserted on the Python side).
//   * PricingPeriod accessors — start/end/payment dates + quantity amount.
//   * CommodityPricingHelper::calculateUomConversionFactor — factor lookup.
//
// C++ parity:
//   ql/experimental/commodities/unitofmeasure.hpp
//   ql/experimental/commodities/petroleumunitsofmeasure.hpp
//   ql/experimental/commodities/unitofmeasureconversion.hpp
//   ql/experimental/commodities/unitofmeasureconversionmanager.hpp
//   ql/experimental/commodities/commoditytype.hpp
//   ql/experimental/commodities/quantity.hpp
//   ql/experimental/commodities/pricingperiod.hpp
//   ql/experimental/commodities/commoditypricinghelpers.hpp
//   @ v1.42.1 (099987f0).

#include <ql/experimental/commodities/commoditypricinghelpers.hpp>
#include <ql/experimental/commodities/commoditytype.hpp>
#include <ql/experimental/commodities/petroleumunitsofmeasure.hpp>
#include <ql/experimental/commodities/pricingperiod.hpp>
#include <ql/experimental/commodities/quantity.hpp>
#include <ql/experimental/commodities/unitofmeasureconversion.hpp>
#include <ql/experimental/commodities/unitofmeasureconversionmanager.hpp>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace QuantLib;

namespace {
    std::string jr(const std::string& key, Real v) {
        std::ostringstream os;
        os << "  \"" << key << "\": " << std::setprecision(17) << v;
        return os.str();
    }
    std::string js(const std::string& key, const std::string& v) {
        std::ostringstream os;
        os << "  \"" << key << "\": \"" << v << "\"";
        return os.str();
    }
}

int main() {
    std::vector<std::string> out;

    NullCommodityType nct;
    BarrelUnitOfMeasure bbl;
    LitreUnitOfMeasure litre;
    GallonUnitOfMeasure gal;
    MBUnitOfMeasure mb;
    KilolitreUnitOfMeasure kl;

    // ---- known conversion factors via the manager (direct lookup) ----
    {
        auto& mgr = UnitOfMeasureConversionManager::instance();

        UnitOfMeasureConversion c_bbl_litre = mgr.lookup(
            nct, bbl, litre, UnitOfMeasureConversion::Direct);
        out.push_back(jr("conv_bbl_litre_factor", c_bbl_litre.conversionFactor()));

        UnitOfMeasureConversion c_bbl_gal = mgr.lookup(
            nct, bbl, gal, UnitOfMeasureConversion::Direct);
        out.push_back(jr("conv_bbl_gallon_factor", c_bbl_gal.conversionFactor()));

        UnitOfMeasureConversion c_mb_bbl = mgr.lookup(
            nct, mb, bbl, UnitOfMeasureConversion::Direct);
        out.push_back(jr("conv_mb_bbl_factor", c_mb_bbl.conversionFactor()));

        UnitOfMeasureConversion c_kl_bbl = mgr.lookup(
            nct, kl, bbl, UnitOfMeasureConversion::Direct);
        out.push_back(jr("conv_kilolitre_bbl_factor", c_kl_bbl.conversionFactor()));

        // code string of the conversion (commodityType.name + src.code + tgt.code)
        out.push_back(js("conv_bbl_litre_code", c_bbl_litre.code()));
    }

    // ---- UnitOfMeasureConversion::convert — direct + inverse ----
    {
        auto& mgr = UnitOfMeasureConversionManager::instance();
        UnitOfMeasureConversion c = mgr.lookup(
            nct, bbl, litre, UnitOfMeasureConversion::Direct);

        // 1 barrel -> litres (forward, source->target)
        Quantity oneBarrel(nct, bbl, 1.0);
        Quantity inLitres = c.convert(oneBarrel);
        out.push_back(jr("convert_1bbl_to_litre_amount", inLitres.amount()));
        out.push_back(js("convert_1bbl_to_litre_uom", inLitres.unitOfMeasure().code()));

        // 158.987 litres -> barrels (inverse, target->source) round-trip
        Quantity someLitres(nct, litre, 158.987);
        Quantity inBarrels = c.convert(someLitres);
        out.push_back(jr("convert_litre_to_bbl_amount", inBarrels.amount()));
        out.push_back(js("convert_litre_to_bbl_uom", inBarrels.unitOfMeasure().code()));
    }

    // ---- Quantity arithmetic: same UOM ----
    {
        Quantity a(nct, bbl, 1.0);
        Quantity b(nct, bbl, 1.0);
        Quantity sum = a + b;
        out.push_back(jr("qty_1bbl_plus_1bbl", sum.amount()));

        Quantity diff = Quantity(nct, bbl, 5.0) - Quantity(nct, bbl, 2.0);
        out.push_back(jr("qty_5bbl_minus_2bbl", diff.amount()));

        Quantity scaled = Quantity(nct, bbl, 3.0) * 4.0;
        out.push_back(jr("qty_3bbl_times_4", scaled.amount()));

        Real ratio = Quantity(nct, bbl, 6.0) / Quantity(nct, bbl, 2.0);
        out.push_back(jr("qty_6bbl_over_2bbl", ratio));

        Quantity divided = Quantity(nct, bbl, 9.0) / 3.0;
        out.push_back(jr("qty_9bbl_over_3", divided.amount()));

        Quantity neg = -Quantity(nct, bbl, 7.0);
        out.push_back(jr("qty_negate_7bbl", neg.amount()));
    }

    // ---- Quantity arithmetic: cross-UOM add under AutomatedConversion ----
    {
        Quantity::ConversionType saved = Quantity::conversionType;
        Quantity::conversionType = Quantity::AutomatedConversion;

        // 1 barrel + 158.987 litres -> result in barrels (first-operand UOM).
        // 158.987 litres == exactly 1 barrel, so the sum is 2 barrels.
        Quantity inBarrels(nct, bbl, 1.0);
        Quantity inLitres(nct, litre, 158.987);
        Quantity total = inBarrels + inLitres;
        out.push_back(jr("qty_1bbl_plus_158_987litre_amount", total.amount()));
        out.push_back(js("qty_1bbl_plus_158_987litre_uom", total.unitOfMeasure().code()));

        Quantity::conversionType = saved;
    }

    // ---- CommodityType / UnitOfMeasure inspectors ----
    {
        CommodityType ho("HO", "Heating Oil");
        out.push_back(js("commodity_ho_code", ho.code()));
        out.push_back(js("commodity_ho_name", ho.name()));
        out.push_back(js("null_commodity_code", nct.code()));

        out.push_back(js("uom_bbl_code", bbl.code()));
        out.push_back(js("uom_bbl_name", bbl.name()));
        out.push_back(js("uom_litre_triangulation_code",
                         litre.triangulationUnitOfMeasure().code()));
    }

    // ---- PricingPeriod accessors ----
    {
        Date start(15, March, 2020);
        Date end(14, April, 2020);
        Date pay(20, April, 2020);
        Quantity q(nct, bbl, 1000.0);
        PricingPeriod pp(start, end, pay, q);
        out.push_back(jr("pricing_period_start_serial",
                         (Real)pp.startDate().serialNumber()));
        out.push_back(jr("pricing_period_end_serial",
                         (Real)pp.endDate().serialNumber()));
        out.push_back(jr("pricing_period_pay_serial",
                         (Real)pp.paymentDate().serialNumber()));
        out.push_back(jr("pricing_period_qty_amount", pp.quantity().amount()));
    }

    // ---- CommodityPricingHelper::calculateUomConversionFactor ----
    {
        Real f = CommodityPricingHelper::calculateUomConversionFactor(
            nct, bbl, litre);
        out.push_back(jr("helper_uom_factor_bbl_litre", f));
        Real same = CommodityPricingHelper::calculateUomConversionFactor(
            nct, bbl, bbl);
        out.push_back(jr("helper_uom_factor_same", same));
    }

    std::cout << "{\n";
    for (Size i = 0; i < out.size(); ++i)
        std::cout << out[i] << (i + 1 < out.size() ? ",\n" : "\n");
    std::cout << "}\n";
    return 0;
}

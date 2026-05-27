// L2-B yield-curve cluster mega-probe.
//
// Captures numerical ground-truth values for all 12 ported concretes:
//   FlatForward, InterpolatedZeroCurve, InterpolatedForwardCurve,
//   InterpolatedDiscountCurve, ZeroCurve, ForwardCurve, DiscountCurve,
//   ForwardSpreadedTermStructure, ZeroSpreadedTermStructure,
//   InterpolatedSpreadDiscountCurve (a.k.a. "DiscountSpreadedTermStructure"
//   per the cluster scope), ImpliedTermStructure.
//
// C++ parity: ql/termstructures/yield/*.hpp @ v1.42.1 (099987f0).

#include <ql/quote.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/handle.hpp>
#include <ql/shared_ptr.hpp>
#include <ql/compounding.hpp>
#include <ql/interestrate.hpp>
#include <ql/termstructures/yieldtermstructure.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/termstructures/yield/zerocurve.hpp>
#include <ql/termstructures/yield/forwardcurve.hpp>
#include <ql/termstructures/yield/discountcurve.hpp>
#include <ql/termstructures/yield/forwardspreadedtermstructure.hpp>
#include <ql/termstructures/yield/zerospreadedtermstructure.hpp>
#include <ql/termstructures/yield/spreaddiscountcurve.hpp>
#include <ql/termstructures/yield/impliedtermstructure.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/calendar.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace QuantLib;

namespace {
// Helper: emit a JSON key+double value pair (no comma).
void emit(const char* key, double v) {
    std::cout << "    \"" << key << "\": " << v;
}
}  // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    // Reference date for ALL curves: 2026-06-15.
    Date ref(15, June, 2026);

    // ============================================================
    // FlatForward: 4 scenarios.
    // ============================================================
    {
        std::cout << "  \"flat_forward\": {\n";

        // Scenario 1: rate-based, Actual360, Continuous.
        // discount(t=1.0) = exp(-0.05) ≈ 0.951229...
        FlatForward ff_cont(ref, 0.05, Actual360(), Continuous, Annual);
        std::cout << "    \"cont_a360_05\": {\n";
        emit("discount_t0",   ff_cont.discount(0.0)); std::cout << ",\n";
        emit("discount_t1",   ff_cont.discount(1.0)); std::cout << ",\n";
        emit("discount_t2",   ff_cont.discount(2.0)); std::cout << ",\n";
        emit("discount_t05",  ff_cont.discount(0.5)); std::cout << ",\n";
        emit("zero_rate_t1",  ff_cont.zeroRate(1.0, Continuous, Annual).rate()); std::cout << ",\n";
        emit("fwd_rate_t1_t2", ff_cont.forwardRate(1.0, 2.0, Continuous, Annual).rate());
        std::cout << "\n    },\n";

        // Scenario 2: rate-based, Actual360, Semiannual Compounded.
        // discount(t=1.0) = (1 + 0.05/2)^(-2) ≈ 0.951814...
        FlatForward ff_semi(ref, 0.05, Actual360(), Compounded, Semiannual);
        std::cout << "    \"semi_a360_05\": {\n";
        emit("discount_t1",   ff_semi.discount(1.0)); std::cout << ",\n";
        emit("discount_t2",   ff_semi.discount(2.0)); std::cout << ",\n";
        emit("zero_rate_t1",  ff_semi.zeroRate(1.0, Compounded, Semiannual).rate());
        std::cout << "\n    },\n";

        // Scenario 3: Handle<Quote>-based, Actual365Fixed, Continuous.
        ext::shared_ptr<SimpleQuote> q = ext::make_shared<SimpleQuote>(0.03);
        Handle<Quote> h(q);
        FlatForward ff_quote(ref, h, Actual365Fixed(), Continuous, Annual);
        std::cout << "    \"quote_a365_03\": {\n";
        emit("discount_t1", ff_quote.discount(1.0)); std::cout << ",\n";
        emit("discount_t3", ff_quote.discount(3.0));
        std::cout << "\n    },\n";

        // Scenario 4: Simple compounding.
        FlatForward ff_simple(ref, 0.05, Actual360(), Simple, Annual);
        std::cout << "    \"simple_a360_05\": {\n";
        emit("discount_t1",  ff_simple.discount(1.0)); std::cout << ",\n";
        emit("discount_t05", ff_simple.discount(0.5));
        std::cout << "\n    },\n";

        // Date-based discount.
        // d = ref + 365 days, Actual365Fixed → t = 1.0.
        Date d365 = ref + 365;
        std::cout << "    \"date_discount\": {\n";
        emit("d365_actual365", ff_quote.discount(d365));
        std::cout << "\n    }\n";

        std::cout << "  },\n";
    }

    // ============================================================
    // InterpolatedZeroCurve (ZeroCurve = Linear).
    // ============================================================
    {
        std::cout << "  \"interpolated_zero_curve\": {\n";

        std::vector<Date> dates = {
            ref,
            ref + 30,    // ~1 month
            ref + 91,    // ~3 months
            ref + 182,   // ~6 months
            ref + 365,   // 1 year
            ref + 730    // 2 years
        };
        std::vector<Rate> yields = {0.020, 0.022, 0.025, 0.028, 0.030, 0.035};
        ZeroCurve zc(dates, yields, Actual365Fixed());  // Linear, Continuous default

        // Nodes
        for (Size i = 0; i < dates.size(); ++i) {
            std::cout << "    \"discount_node" << i << "\": " << zc.discount(dates[i]) << ",\n";
            std::cout << "    \"zero_rate_node" << i << "\": " << zc.zeroRate(dates[i], Actual365Fixed(), Continuous, Annual).rate() << ",\n";
        }
        // Intermediate t at mid-points
        std::cout << "    \"discount_t0_5\": " << zc.discount(0.5) << ",\n";
        std::cout << "    \"discount_t1_0\": " << zc.discount(1.0) << ",\n";
        std::cout << "    \"discount_t1_5\": " << zc.discount(1.5) << ",\n";
        // Extrapolation beyond last (default flat-forward extrapolation, allow=true)
        std::cout << "    \"discount_t3_extrap\": " << zc.discount(3.0, true) << ",\n";
        std::cout << "    \"max_date_serial\": " << zc.maxDate().serialNumber() << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // InterpolatedForwardCurve (ForwardCurve = BackwardFlat).
    // ============================================================
    {
        std::cout << "  \"interpolated_forward_curve\": {\n";

        std::vector<Date> dates = {
            ref,
            ref + 30,
            ref + 91,
            ref + 182,
            ref + 365,
            ref + 730
        };
        std::vector<Rate> fwds = {0.020, 0.022, 0.025, 0.028, 0.030, 0.035};
        ForwardCurve fc(dates, fwds, Actual365Fixed());  // BackwardFlat

        // Discount at boundary times
        std::cout << "    \"discount_t0_2\": " << fc.discount(0.2) << ",\n";
        std::cout << "    \"discount_t0_5\": " << fc.discount(0.5) << ",\n";
        std::cout << "    \"discount_t1_0\": " << fc.discount(1.0) << ",\n";
        std::cout << "    \"discount_t1_5\": " << fc.discount(1.5) << ",\n";
        std::cout << "    \"discount_t2_0_extrap\": " << fc.discount(dates.back()) << ",\n";
        // Extrapolation past last
        std::cout << "    \"discount_t3_extrap\": " << fc.discount(3.0, true) << ",\n";
        std::cout << "    \"max_date_serial\": " << fc.maxDate().serialNumber() << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // InterpolatedDiscountCurve (DiscountCurve = LogLinear).
    // ============================================================
    {
        std::cout << "  \"interpolated_discount_curve\": {\n";

        std::vector<Date> dates = {
            ref,
            ref + 30,
            ref + 91,
            ref + 182,
            ref + 365,
            ref + 730
        };
        // Decreasing discount factors (consistent yield curve).
        std::vector<DiscountFactor> dfs = {1.0, 0.998, 0.993, 0.985, 0.970, 0.930};
        DiscountCurve dc(dates, dfs, Actual365Fixed());  // LogLinear

        // Nodes
        for (Size i = 0; i < dates.size(); ++i) {
            std::cout << "    \"discount_node" << i << "\": " << dc.discount(dates[i]) << ",\n";
        }
        // Intermediate
        std::cout << "    \"discount_t0_2\": " << dc.discount(0.2) << ",\n";
        std::cout << "    \"discount_t1_0\": " << dc.discount(1.0) << ",\n";
        // Extrapolation
        std::cout << "    \"discount_t3_extrap\": " << dc.discount(3.0, true) << ",\n";
        std::cout << "    \"max_date_serial\": " << dc.maxDate().serialNumber() << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // ForwardSpreadedTermStructure.
    // ============================================================
    {
        std::cout << "  \"forward_spreaded\": {\n";

        // Base: 5% continuous on Actual360
        ext::shared_ptr<YieldTermStructure> baseTs =
            ext::make_shared<FlatForward>(ref, 0.05, Actual360(), Continuous, Annual);
        Handle<YieldTermStructure> baseH(baseTs);

        ext::shared_ptr<SimpleQuote> spreadQ = ext::make_shared<SimpleQuote>(0.01);
        Handle<Quote> spreadH(spreadQ);

        ForwardSpreadedTermStructure fst(baseH, spreadH);
        std::cout << "    \"zero_rate_t1\": " << fst.zeroRate(1.0, Continuous, Annual).rate() << ",\n";
        std::cout << "    \"zero_rate_t2\": " << fst.zeroRate(2.0, Continuous, Annual).rate() << ",\n";
        std::cout << "    \"discount_t1\": " << fst.discount(1.0) << ",\n";
        std::cout << "    \"discount_t2\": " << fst.discount(2.0) << ",\n";
        // Spread bumping → reflected in dependent value
        spreadQ->setValue(0.02);
        std::cout << "    \"zero_rate_t1_bumped\": " << fst.zeroRate(1.0, Continuous, Annual).rate() << ",\n";
        std::cout << "    \"discount_t1_bumped\": " << fst.discount(1.0) << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // ZeroSpreadedTermStructure.
    // ============================================================
    {
        std::cout << "  \"zero_spreaded\": {\n";

        ext::shared_ptr<YieldTermStructure> baseTs =
            ext::make_shared<FlatForward>(ref, 0.05, Actual360(), Continuous, Annual);
        Handle<YieldTermStructure> baseH(baseTs);
        ext::shared_ptr<SimpleQuote> spreadQ = ext::make_shared<SimpleQuote>(0.01);
        Handle<Quote> spreadH(spreadQ);

        // Default: Continuous, NoFrequency.
        ZeroSpreadedTermStructure zst(baseH, spreadH);
        std::cout << "    \"continuous_zero_rate_t1\": " << zst.zeroRate(1.0, Continuous, Annual).rate() << ",\n";
        std::cout << "    \"continuous_discount_t1\": " << zst.discount(1.0) << ",\n";
        std::cout << "    \"continuous_discount_t2\": " << zst.discount(2.0) << ",\n";

        // Compounded Semiannual: spreadedRate is constructed in Compounded
        // and converted back to Continuous via equivalentRate.
        ZeroSpreadedTermStructure zst_semi(baseH, spreadH, Compounded, Semiannual);
        std::cout << "    \"semi_zero_rate_t1\": " << zst_semi.zeroRate(1.0, Continuous, Annual).rate() << ",\n";
        std::cout << "    \"semi_discount_t1\": " << zst_semi.discount(1.0) << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // InterpolatedSpreadDiscountCurve (DiscountSpreadedTermStructure
    // per the cluster scope wording).
    // ============================================================
    {
        std::cout << "  \"interpolated_spread_discount\": {\n";

        // Base curve: flat 5% continuous.
        ext::shared_ptr<YieldTermStructure> baseTs =
            ext::make_shared<FlatForward>(ref, 0.05, Actual360(), Continuous, Annual);
        // Note: baseTs day counter is Actual360. The spread curve uses the
        // same day counter (inherited from base in this class).
        Handle<YieldTermStructure> baseH(baseTs);

        // Spread dates/discount factors. First df must == 1.0.
        std::vector<Date> sdates = {ref, ref + 182, ref + 365, ref + 730};
        std::vector<DiscountFactor> sdfs = {1.0, 0.995, 0.985, 0.965};

        SpreadDiscountCurve sdc(baseH, sdates, sdfs);  // LogLinear default

        // baseTs uses Actual360, so curve times come from that DC.
        std::cout << "    \"discount_at_node_ref_plus_182\": " << sdc.discount(sdates[1]) << ",\n";
        std::cout << "    \"discount_at_node_ref_plus_365\": " << sdc.discount(sdates[2]) << ",\n";
        std::cout << "    \"discount_at_node_ref_plus_730\": " << sdc.discount(sdates[3]) << ",\n";
        std::cout << "    \"discount_t0_5\": " << sdc.discount(0.5) << ",\n";
        std::cout << "    \"discount_t1_0\": " << sdc.discount(1.0) << ",\n";
        std::cout << "    \"discount_t2_0_extrap\": " << sdc.discount(2.0, true) << ",\n";
        std::cout << "    \"max_date_serial\": " << sdc.maxDate().serialNumber() << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // ImpliedTermStructure.
    // ============================================================
    {
        std::cout << "  \"implied\": {\n";

        ext::shared_ptr<YieldTermStructure> baseTs =
            ext::make_shared<FlatForward>(ref, 0.05, Actual365Fixed(), Continuous, Annual);
        Handle<YieldTermStructure> baseH(baseTs);

        // Future reference date = ref + 365 days = ~1 year out.
        Date futureRef = ref + 365;
        ImpliedTermStructure its(baseH, futureRef);

        // Time relative to futureRef = 0 → discount should be 1.0
        std::cout << "    \"discount_t0\": " << its.discount(0.0) << ",\n";
        // Time = 1.0 (one year past future ref) → for flat-fwd: exp(-r*t) shape
        std::cout << "    \"discount_t1\": " << its.discount(1.0) << ",\n";
        // Verify ref-date matches futureRef
        std::cout << "    \"reference_date_serial\": " << its.referenceDate().serialNumber() << ",\n";
        std::cout << "    \"max_date_serial\": " << its.maxDate().serialNumber() << "\n";
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}

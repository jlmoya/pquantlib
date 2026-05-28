// L5-B cluster probe: trees + lattices + tree-based engines +
// BlackKarasinski.
//
// Captures reference values for:
//
//   * BinomialTree (CRR + LeisenReimer) on a GBSM-like flat process:
//     S0=100, r=5%, q=0%, sigma=20%, maturity=1.0, steps=4.  Records
//     the underlying at terminal nodes + (pu, pd) for cross-validation
//     of the refactored Python BinomialTree concretes.
//
//   * TrinomialTree over OrnsteinUhlenbeck(a=0.1, sigma=0.01, x0=0,
//     level=0), TimeGrid(end=2.0, 5 steps): dx[i], underlying(i, j),
//     descendant(i, j, branch), probability(i, j, branch) — enough to
//     pin the recombining structure.
//
//   * HullWhite analytic discount(2.0): used as the analytic
//     benchmark against the trinomial-tree state-price summation.
//
//   * TreeSwaptionEngine: 5y10y RECEIVER swaption (3% strike, 1m
//     notional, Euribor3M) under HullWhite(a=0.1, sigma=0.01) with
//     100 timesteps; compare against the Jamshidian closed-form
//     reference (loose tolerance: tree convergence).
//
//   * TreeCapFloorEngine: 5y cap @ 4% on Euribor3M under HW(a=0.1,
//     sigma=0.01) with 100 timesteps; compare against
//     AnalyticCapFloorEngine.
//
//   * BlackKarasinski(a=0.1, sigma=0.1): build a tree(grid=2.0/50) and
//     compute the state-price discount factor at the last grid point.
//     Records `bk_pv_2y` for parity testing (the BK tree numerically
//     fits to the input curve so it reprices its own bond).
//
// C++ parity:
//   ql/methods/lattices/binomialtree.{hpp,cpp},
//   ql/methods/lattices/trinomialtree.{hpp,cpp},
//   ql/methods/lattices/lattice.{hpp},
//   ql/methods/lattices/lattice1d.{hpp},
//   ql/methods/lattices/bsmlattice.{hpp},
//   ql/models/shortrate/onefactormodel.{hpp,cpp},
//   ql/models/shortrate/onefactormodels/blackkarasinski.{hpp,cpp},
//   ql/pricingengines/swaption/treeswaptionengine.{hpp,cpp},
//   ql/pricingengines/capfloor/treecapfloorengine.{hpp,cpp}
//   @ v1.42.1 (099987f0).

#include <ql/instruments/swaption.hpp>
#include <ql/instruments/vanillaswap.hpp>
#include <ql/instruments/capfloor.hpp>
#include <ql/cashflows/iborcoupon.hpp>
#include <ql/cashflows/cashflowvectors.hpp>
#include <ql/exercise.hpp>
#include <ql/indexes/ibor/euribor.hpp>
#include <ql/methods/lattices/binomialtree.hpp>
#include <ql/methods/lattices/trinomialtree.hpp>
#include <ql/methods/lattices/lattice.hpp>
#include <ql/methods/lattices/lattice1d.hpp>
#include <ql/methods/lattices/bsmlattice.hpp>
#include <ql/models/shortrate/onefactormodels/hullwhite.hpp>
#include <ql/models/shortrate/onefactormodels/blackkarasinski.hpp>
#include <ql/pricingengines/capfloor/analyticcapfloorengine.hpp>
#include <ql/pricingengines/capfloor/treecapfloorengine.hpp>
#include <ql/pricingengines/swap/discountingswapengine.hpp>
#include <ql/pricingengines/swaption/jamshidianswaptionengine.hpp>
#include <ql/pricingengines/swaption/treeswaptionengine.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/processes/ornsteinuhlenbeckprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/thirty360.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>

using namespace QuantLib;

int main() {
    std::cout << std::setprecision(17);

    // Evaluation date — Wednesday, no TARGET holiday issues.
    Date evalDate(17, January, 2024);
    Settings::instance().evaluationDate() = evalDate;

    Handle<Quote> rateQuote(ext::make_shared<SimpleQuote>(0.05));
    Handle<YieldTermStructure> curve(
        ext::make_shared<FlatForward>(evalDate, rateQuote, Actual360(),
                                      Continuous, Annual));
    Calendar cal = TARGET();
    DayCounter dc = Actual360();

    auto index = ext::make_shared<Euribor3M>(curve);
    auto swapEngine = ext::make_shared<DiscountingSwapEngine>(curve, false);

    std::cout << "{\n";

    // ------------------------------------------------------------------
    // BinomialTree: CRR + LeisenReimer on a GBSM-like flat process.
    // S0=100, q=0, r=5%, sigma=20%, T=1.0, steps=4.  We pull terminal
    // underlying values + the (pu, pd) probabilities for a sample
    // cross-validation of the Python BinomialTree concretes.
    //
    // The strike value is forwarded to LeisenReimer to drive its
    // PeizerPratt inversion; we use ATM = 100.
    // ------------------------------------------------------------------
    {
        Real s0 = 100.0;
        Rate r = 0.05;
        Rate q = 0.0;
        Volatility sigma = 0.20;
        Time end = 1.0;
        Size steps = 4;
        Real strike = 100.0;

        // Build a constant-coefficient GBSM equivalent.
        Handle<Quote> spot(ext::make_shared<SimpleQuote>(s0));
        Handle<YieldTermStructure> rTS(
            ext::make_shared<FlatForward>(evalDate, r, Actual360(), Continuous, Annual));
        Handle<YieldTermStructure> qTS(
            ext::make_shared<FlatForward>(evalDate, q, Actual360(), Continuous, Annual));
        Handle<BlackVolTermStructure> volTS(
            ext::make_shared<BlackConstantVol>(evalDate, cal, sigma, Actual360()));
        auto process = ext::make_shared<GeneralizedBlackScholesProcess>(
            spot, qTS, rTS, volTS);

        auto crr = ext::make_shared<CoxRossRubinstein>(process, end, steps, strike);
        std::cout << "  \"binomial_crr\": {\n";
        std::cout << "    \"s0\": " << s0 << ",\n";
        std::cout << "    \"steps\": " << steps << ",\n";
        std::cout << "    \"pu\": " << crr->probability(0, 0, 1) << ",\n";
        std::cout << "    \"pd\": " << crr->probability(0, 0, 0) << ",\n";
        std::cout << "    \"underlying_terminal\": [";
        for (Size j = 0; j <= steps; j++) {
            if (j > 0) std::cout << ", ";
            std::cout << crr->underlying(steps, j);
        }
        std::cout << "]\n";
        std::cout << "  },\n";

        auto lr = ext::make_shared<LeisenReimer>(process, end, steps, strike);
        // LeisenReimer forces odd steps internally (steps+1 if even).
        Size lrSteps = (steps % 2 == 0) ? steps + 1 : steps;
        std::cout << "  \"binomial_lr\": {\n";
        std::cout << "    \"s0\": " << s0 << ",\n";
        std::cout << "    \"requested_steps\": " << steps << ",\n";
        std::cout << "    \"actual_steps\": " << lrSteps << ",\n";
        std::cout << "    \"pu\": " << lr->probability(0, 0, 1) << ",\n";
        std::cout << "    \"pd\": " << lr->probability(0, 0, 0) << ",\n";
        std::cout << "    \"underlying_terminal\": [";
        for (Size j = 0; j <= lrSteps; j++) {
            if (j > 0) std::cout << ", ";
            std::cout << lr->underlying(lrSteps, j);
        }
        std::cout << "]\n";
        std::cout << "  },\n";
    }

    // ------------------------------------------------------------------
    // TrinomialTree over a centred OU process.
    // a=0.1, sigma=0.01, x0=0, level=0.  TimeGrid(end=2.0, 5 steps).
    // ------------------------------------------------------------------
    {
        Real a = 0.1;
        Real sigma = 0.01;
        Time end = 2.0;
        Size steps = 5;

        auto process = ext::make_shared<OrnsteinUhlenbeckProcess>(a, sigma);
        TimeGrid grid(end, steps);
        auto tree = ext::make_shared<TrinomialTree>(process, grid);

        std::cout << "  \"trinomial_tree\": {\n";
        std::cout << "    \"a\": " << a << ",\n";
        std::cout << "    \"sigma\": " << sigma << ",\n";
        std::cout << "    \"end\": " << end << ",\n";
        std::cout << "    \"steps\": " << steps << ",\n";
        std::cout << "    \"dx\": [";
        for (Size i = 0; i <= steps; i++) {
            if (i > 0) std::cout << ", ";
            std::cout << tree->dx(i);
        }
        std::cout << "],\n";
        std::cout << "    \"sizes\": [";
        for (Size i = 0; i <= steps; i++) {
            if (i > 0) std::cout << ", ";
            std::cout << tree->size(i);
        }
        std::cout << "],\n";
        // Underlying values per slice.
        std::cout << "    \"underlying\": [";
        for (Size i = 0; i <= steps; i++) {
            if (i > 0) std::cout << ", ";
            std::cout << "[";
            for (Size j = 0; j < tree->size(i); j++) {
                if (j > 0) std::cout << ", ";
                std::cout << tree->underlying(i, j);
            }
            std::cout << "]";
        }
        std::cout << "],\n";
        // Probabilities per descendant for time slice 0 (single node).
        std::cout << "    \"prob_slice0\": [";
        for (Size b = 0; b < 3; b++) {
            if (b > 0) std::cout << ", ";
            std::cout << tree->probability(0, 0, b);
        }
        std::cout << "],\n";
        // Descendants from time slice 0 (single node).
        std::cout << "    \"desc_slice0\": [";
        for (Size b = 0; b < 3; b++) {
            if (b > 0) std::cout << ", ";
            std::cout << tree->descendant(0, 0, b);
        }
        std::cout << "]\n";
        std::cout << "  },\n";
    }

    // ------------------------------------------------------------------
    // Build the same 5y x 10y RECEIVER swaption as the l4e probe.
    // We need the JamshidianSwaptionEngine NPV (reproduced) +
    // TreeSwaptionEngine NPV under HW(a=0.1, sigma=0.01) @ 100 steps.
    // ------------------------------------------------------------------
    Date settle = cal.advance(evalDate, 5 * Years);
    Date end = cal.advance(settle, 10 * Years);
    Schedule fixedSchedule(settle, end, Period(6, Months), cal,
                           ModifiedFollowing, ModifiedFollowing,
                           DateGeneration::Backward, false);
    Schedule floatSchedule(settle, end, Period(3, Months), cal,
                           ModifiedFollowing, ModifiedFollowing,
                           DateGeneration::Backward, false);

    Rate strike = 0.03;
    Real nominal = 1'000'000.0;
    auto swap = ext::make_shared<VanillaSwap>(
        Swap::Receiver, nominal,
        fixedSchedule, strike, Thirty360(Thirty360::BondBasis),
        floatSchedule, index, 0.0, index->dayCounter());
    swap->setPricingEngine(swapEngine);

    auto exercise = ext::make_shared<EuropeanExercise>(settle);
    auto swaption = ext::make_shared<Swaption>(swap, exercise);

    {
        // Jamshidian reference value (closed form).
        auto hw = ext::make_shared<HullWhite>(curve, 0.1, 0.01);
        auto eng = ext::make_shared<JamshidianSwaptionEngine>(hw, curve);
        swaption->setPricingEngine(eng);
        Real npv = swaption->NPV();
        std::cout << "  \"jamshidian_ref\": {\n";
        std::cout << "    \"hw_a\": 0.1, \"hw_sigma\": 0.01,\n";
        std::cout << "    \"npv\": " << npv << "\n";
        std::cout << "  },\n";
    }
    {
        auto hw = ext::make_shared<HullWhite>(curve, 0.1, 0.01);
        Size timeSteps = 100;
        auto eng = ext::make_shared<TreeSwaptionEngine>(hw, timeSteps, curve);
        swaption->setPricingEngine(eng);
        Real npv = swaption->NPV();
        std::cout << "  \"tree_swaption_hw\": {\n";
        std::cout << "    \"hw_a\": 0.1, \"hw_sigma\": 0.01,\n";
        std::cout << "    \"time_steps\": " << timeSteps << ",\n";
        std::cout << "    \"npv\": " << npv << "\n";
        std::cout << "  },\n";
    }

    // ------------------------------------------------------------------
    // 5y cap @ 4% on Euribor3M.
    // We compare TreeCapFloorEngine against AnalyticCapFloorEngine
    // under HW(a=0.1, sigma=0.01).
    // Start the cap two business days in the future to avoid the
    // "missing fixing" path (first reset date else falls on evalDate-2).
    // ------------------------------------------------------------------
    Date capStart = cal.advance(evalDate, 3 * Months);
    Date capEnd = cal.advance(capStart, 5 * Years);
    Schedule capSchedule(capStart, capEnd, Period(3, Months), cal,
                         ModifiedFollowing, ModifiedFollowing,
                         DateGeneration::Backward, false);
    Leg capLeg = IborLeg(capSchedule, index)
        .withNotionals(nominal)
        .withPaymentDayCounter(index->dayCounter())
        .withPaymentAdjustment(ModifiedFollowing);
    auto cap = ext::make_shared<Cap>(capLeg, std::vector<Rate>(1, 0.04));

    {
        auto hw = ext::make_shared<HullWhite>(curve, 0.1, 0.01);
        auto eng = ext::make_shared<AnalyticCapFloorEngine>(hw, curve);
        cap->setPricingEngine(eng);
        Real npv = cap->NPV();
        std::cout << "  \"analytic_cap_ref\": {\n";
        std::cout << "    \"hw_a\": 0.1, \"hw_sigma\": 0.01,\n";
        std::cout << "    \"strike\": 0.04, \"length_years\": 5,\n";
        std::cout << "    \"npv\": " << npv << "\n";
        std::cout << "  },\n";
    }
    {
        auto hw = ext::make_shared<HullWhite>(curve, 0.1, 0.01);
        Size timeSteps = 100;
        auto eng = ext::make_shared<TreeCapFloorEngine>(hw, timeSteps, curve);
        cap->setPricingEngine(eng);
        Real npv = cap->NPV();
        std::cout << "  \"tree_cap_hw\": {\n";
        std::cout << "    \"hw_a\": 0.1, \"hw_sigma\": 0.01,\n";
        std::cout << "    \"strike\": 0.04,\n";
        std::cout << "    \"time_steps\": " << timeSteps << ",\n";
        std::cout << "    \"npv\": " << npv << "\n";
        std::cout << "  },\n";
    }

    // ------------------------------------------------------------------
    // BlackKarasinski sanity: build a tree on [0, 2], 50 steps.
    // The tree numerically fits the curve so the state-price summation
    // at any grid time should reprice the curve discount.
    //
    // We use the lattice's stepback on a DiscretizedDiscountBond to
    // get back PV(2y) — should approximately match curve.discount(2y)
    // = exp(-0.05 * 2 * (Actual360 year fraction over 2y)).
    // ------------------------------------------------------------------
    {
        Real a = 0.1;
        Real sigma = 0.1;
        auto bk = ext::make_shared<BlackKarasinski>(curve, a, sigma);
        Size steps = 50;
        TimeGrid grid(2.0, steps);
        auto lattice = bk->tree(grid);

        // Build a discount bond maturing at the last grid point and
        // roll it back to 0.0 via the lattice.
        DiscretizedDiscountBond bond;
        bond.initialize(lattice, grid.back());
        bond.rollback(0.0);
        Real pv = bond.presentValue();

        std::cout << "  \"bk_zero_bond\": {\n";
        std::cout << "    \"a\": " << a << ", \"sigma\": " << sigma << ",\n";
        std::cout << "    \"end\": 2.0, \"steps\": " << steps << ",\n";
        std::cout << "    \"pv_at_zero\": " << pv << ",\n";
        std::cout << "    \"curve_discount_2y\": " << curve->discount(2.0) << "\n";
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}

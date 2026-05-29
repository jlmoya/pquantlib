// Phase 11 W3-D cluster probe: NTD instrument + CDS option +
// risky asset swap family.
//
// Captures reference values for:
//
//   * NthToDefault construction + inspectors (rank, side, premium,
//     nominal, day-counter, maturity, basket-size).
//
//   * CdsOption construction + inspectors (knocksOut, underlying-side).
//
//   * BlackCdsOptionEngine NPV at known Black inputs against a known
//     CDS configuration (knock-out + non-knock-out variants).
//
//   * RiskyAssetSwap NPV + fairSpread on a simple flat-yield +
//     flat-hazard configuration (Euler-integral recovery; closed-form
//     bond price).
//
//   * RiskyAssetSwapOption NPV on a Bachelier-like spread-option
//     formulation (call/put depending on fixed-payer flag).
//
//   * GaussianRandomDefaultModel.nextSequence statistics: sample-mean
//     of generated default times across a small pool with high
//     correlation; reset/restart determinism check.
//
// C++ parity:
//   ql/experimental/credit/nthtodefault.hpp
//   ql/experimental/credit/cdsoption.hpp
//   ql/experimental/credit/blackcdsoptionengine.hpp
//   ql/experimental/credit/randomdefaultmodel.hpp
//   ql/experimental/credit/riskyassetswap.hpp
//   ql/experimental/credit/riskyassetswapoption.hpp
//   @ v1.42.1 (099987f0).

#include <ql/cashflows/fixedratecoupon.hpp>
#include <ql/currencies/america.hpp>
#include <ql/exercise.hpp>
#include <ql/experimental/credit/basket.hpp>
#include <ql/experimental/credit/blackcdsoptionengine.hpp>
#include <ql/experimental/credit/cdsoption.hpp>
#include <ql/experimental/credit/defaultprobabilitykey.hpp>
#include <ql/experimental/credit/defaulttype.hpp>
#include <ql/experimental/credit/issuer.hpp>
#include <ql/experimental/credit/nthtodefault.hpp>
#include <ql/experimental/credit/onefactorgaussiancopula.hpp>
#include <ql/experimental/credit/pool.hpp>
#include <ql/experimental/credit/randomdefaultmodel.hpp>
#include <ql/experimental/credit/recoveryratemodel.hpp>
#include <ql/experimental/credit/riskyassetswap.hpp>
#include <ql/experimental/credit/riskyassetswapoption.hpp>
#include <ql/handle.hpp>
#include <ql/instruments/claim.hpp>
#include <ql/instruments/creditdefaultswap.hpp>
#include <ql/pricingengines/credit/midpointcdsengine.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/credit/flathazardrate.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/schedule.hpp>

#include <iomanip>
#include <iostream>
#include <vector>

using namespace QuantLib;

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    USDCurrency usd;
    DayCounter dc = Actual365Fixed();
    Calendar cal = TARGET();
    Date today(15, January, 2024);
    Settings::instance().evaluationDate() = today;
    Handle<YieldTermStructure> yts(
        ext::make_shared<FlatForward>(today, 0.03, dc));
    Handle<DefaultProbabilityTermStructure> dts(
        ext::make_shared<FlatHazardRate>(today, Handle<Quote>(
            ext::make_shared<SimpleQuote>(0.02)), dc));

    // Build a small pool with three identical issuers for NTD and
    // RandomDefaultModel coverage. Same default-prob key per name.
    std::vector<ext::shared_ptr<DefaultType>> event_types;
    event_types.push_back(ext::make_shared<DefaultType>(
        AtomicDefault::Bankruptcy, Restructuring::XR));
    DefaultProbKey contract_key(event_types, usd, SeniorUnSec);

    std::vector<std::pair<DefaultProbKey, Handle<DefaultProbabilityTermStructure>>>
        probs{{contract_key, dts}};
    Issuer issuer(probs);

    ext::shared_ptr<Pool> pool = ext::make_shared<Pool>();
    pool->add("Name0", issuer, contract_key);
    pool->add("Name1", issuer, contract_key);
    pool->add("Name2", issuer, contract_key);

    // ============================================================
    // 1) NthToDefault inspectors
    // ============================================================
    {
        // Pool-backed basket. attach=0, detach=1 (whole basket).
        std::vector<Real> notionals{1.0e6, 1.0e6, 1.0e6};
        ext::shared_ptr<Basket> basket =
            ext::make_shared<Basket>(today,
                std::vector<std::string>{"Name0", "Name1", "Name2"},
                notionals, pool, 0.0, 1.0,
                ext::make_shared<FaceValueClaim>());

        Date start = cal.advance(today, 1, Days);
        Schedule prem_sched = MakeSchedule()
            .from(start).to(start + 1 * Years)
            .withCalendar(cal)
            .withTenor(3 * Months)
            .withConvention(Unadjusted)
            .backwards();

        NthToDefault ntd(basket, 2, Protection::Buyer,
                         prem_sched, 0.0, 0.0050, dc, 1.0e6, true);

        std::cout << "  \"ntd\": {\n";
        std::cout << "    \"premium\": " << ntd.premium() << ",\n";
        std::cout << "    \"nominal\": " << ntd.nominal() << ",\n";
        std::cout << "    \"rank\": " << ntd.rank() << ",\n";
        std::cout << "    \"basket_size\": " << ntd.basketSize() << ",\n";
        std::cout << "    \"side_idx\": " << static_cast<int>(ntd.side()) << ",\n";
        std::cout << "    \"maturity_serial\": " << ntd.maturity().serialNumber() << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 2) CdsOption + BlackCdsOptionEngine NPV
    // ============================================================
    {
        // Underlying CDS: protection-buyer, running spread 100bp,
        // 5y term. Use a forward-start schedule from option-exercise
        // date (~1y) to maturity (~6y).
        Date option_exercise = cal.advance(today, 1, Years);
        Date underlying_start = option_exercise;
        Date underlying_end = cal.advance(underlying_start, 5, Years);
        Schedule under_sched = MakeSchedule()
            .from(underlying_start).to(underlying_end)
            .withCalendar(cal)
            .withTenor(3 * Months)
            .withConvention(Following)
            .withRule(DateGeneration::CDS2015);

        ext::shared_ptr<CreditDefaultSwap> cds =
            ext::make_shared<CreditDefaultSwap>(
                Protection::Buyer, 1.0e7, 0.0100, under_sched,
                Following, dc, true, true, underlying_start);
        // Wire engine so that fairSpread + couponLegNPV are valid.
        cds->setPricingEngine(
            ext::make_shared<MidPointCdsEngine>(dts, 0.4, yts));

        ext::shared_ptr<Exercise> ex =
            ext::make_shared<EuropeanExercise>(option_exercise);
        CdsOption opt(cds, ex, true);
        ext::shared_ptr<SimpleQuote> volq =
            ext::make_shared<SimpleQuote>(0.30);
        opt.setPricingEngine(
            ext::make_shared<BlackCdsOptionEngine>(
                dts, 0.4, yts, Handle<Quote>(volq)));

        std::cout << "  \"cds_option\": {\n";
        std::cout << "    \"npv_knockout\": " << opt.NPV() << ",\n";
        std::cout << "    \"risky_annuity\": " << opt.riskyAnnuity() << ",\n";
        std::cout << "    \"atm_rate\": " << opt.atmRate() << ",\n";

        // Non-knock-out payer adds front-end protection.
        CdsOption opt_no_knock(cds, ex, false);
        opt_no_knock.setPricingEngine(
            ext::make_shared<BlackCdsOptionEngine>(
                dts, 0.4, yts, Handle<Quote>(volq)));
        std::cout << "    \"npv_non_knockout\": " << opt_no_knock.NPV() << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 3) RiskyAssetSwap + fairSpread
    // ============================================================
    {
        Date start = cal.advance(today, 2, Days);
        Date end = start + 5 * Years;
        Schedule fixed_sched = MakeSchedule()
            .from(start).to(end)
            .withCalendar(cal)
            .withTenor(1 * Years)
            .withConvention(Unadjusted)
            .backwards();
        Schedule float_sched = MakeSchedule()
            .from(start).to(end)
            .withCalendar(cal)
            .withTenor(6 * Months)
            .withConvention(Unadjusted)
            .backwards();

        Real spread = 0.0150;
        Real recovery = 0.4;
        Real nominal = 100.0;
        Real coupon = 0.05;
        RiskyAssetSwap asw(true, nominal, fixed_sched, float_sched,
                           dc, dc, spread, recovery, yts, dts, coupon);
        std::cout << "  \"risky_asset_swap\": {\n";
        std::cout << "    \"fair_spread\": " << asw.fairSpread() << ",\n";
        std::cout << "    \"npv\": " << asw.NPV() << ",\n";
        std::cout << "    \"float_annuity\": " << asw.floatAnnuity() << ",\n";
        std::cout << "    \"nominal\": " << asw.nominal() << ",\n";
        std::cout << "    \"spread\": " << asw.spread() << "\n";
        std::cout << "  },\n";

        // ============================================================
        // 4) RiskyAssetSwapOption
        // ============================================================
        ext::shared_ptr<RiskyAssetSwap> asw_ptr =
            ext::make_shared<RiskyAssetSwap>(
                true, nominal, fixed_sched, float_sched,
                dc, dc, spread, recovery, yts, dts, coupon);
        Date expiry = cal.advance(today, 1, Years);
        Real market_spread = 0.0200;
        Real spread_vol = 0.40;
        RiskyAssetSwapOption opt(asw_ptr, expiry, market_spread, spread_vol);
        std::cout << "  \"risky_asw_option\": {\n";
        std::cout << "    \"npv\": " << opt.NPV() << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 5) GaussianRandomDefaultModel reset/sample statistics
    // ============================================================
    {
        Handle<Quote> correl(ext::make_shared<SimpleQuote>(0.30));
        Handle<OneFactorCopula> copula(
            ext::make_shared<OneFactorGaussianCopula>(correl));
        std::vector<DefaultProbKey> keys{contract_key, contract_key, contract_key};
        GaussianRandomDefaultModel rdm(pool, keys, copula, 1.0e-6, 42L);

        // Generate a single sequence and read back the three names'
        // default times. Deterministic given the seed.
        rdm.nextSequence(50.0);
        Real t0 = pool->getTime("Name0");
        Real t1 = pool->getTime("Name1");
        Real t2 = pool->getTime("Name2");
        std::cout << "  \"rdm_seq0\": {\n";
        std::cout << "    \"t0\": " << t0 << ",\n";
        std::cout << "    \"t1\": " << t1 << ",\n";
        std::cout << "    \"t2\": " << t2 << "\n";
        std::cout << "  },\n";

        // After reset, the same sequence should reproduce.
        rdm.reset();
        rdm.nextSequence(50.0);
        Real t0r = pool->getTime("Name0");
        std::cout << "  \"rdm_seq0_after_reset_t0\": " << t0r << ",\n";

        // Second sequence after that (different draw).
        rdm.nextSequence(50.0);
        Real t0b = pool->getTime("Name0");
        std::cout << "  \"rdm_seq1_t0\": " << t0b << "\n";
    }

    std::cout << "}\n";
    return 0;
}

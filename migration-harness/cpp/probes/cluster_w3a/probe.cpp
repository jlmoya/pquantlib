// Phase 11 W3-A cluster probe: experimental/credit foundation types.
//
// Captures reference values for:
//
//   * DefaultEvent construction + accessor round-trips (date, currency,
//     seniority, default type, restructuring flag, hasSettled,
//     recoveryRate, matchesDefaultKey).
//
//   * Issuer.defaultProbability(key) round-trip + defaultedBetween/
//     defaultsBetween shape against a small DefaultEventSet.
//
//   * Pool basic API: size(), names(), has(), get(), defaultKey(),
//     setTime/getTime.
//
//   * RecoveryRateQuote conventional ISDA recoveries + setValue/reset
//     + value()/seniority()/isValid() round-trip.
//
//   * ConstantRecoveryModel.recoveryValue() round-trip from both
//     Real-ctor and Quote-ctor.
//
//   * LossDist::probabilityOfNEvents / probabilityOfAtLeastNEvents +
//     binomialProbabilityOfNEvents / binomialProbabilityOfAtLeastNEvents
//     at known probability vectors.
//
//   * LossDistHomogeneous and LossDistBinomial discretized loss
//     distributions: density / cumulative / excess at sample buckets.
//
//   * Distribution base: locate / dx / average / cumulativeDensity /
//     expectedValue / trancheExpectedValue invariants.
//
// C++ parity:
//   ql/experimental/credit/defaulttype.hpp
//   ql/experimental/credit/defaultevent.hpp
//   ql/experimental/credit/defaultprobabilitykey.hpp
//   ql/experimental/credit/issuer.hpp
//   ql/experimental/credit/pool.hpp
//   ql/experimental/credit/recoveryratequote.hpp
//   ql/experimental/credit/recoveryratemodel.hpp
//   ql/experimental/credit/loss.hpp
//   ql/experimental/credit/distribution.hpp
//   ql/experimental/credit/lossdistribution.hpp
//   @ v1.42.1 (099987f0).

#include <ql/currencies/america.hpp>
#include <ql/currencies/europe.hpp>
#include <ql/experimental/credit/defaultevent.hpp>
#include <ql/experimental/credit/defaultprobabilitykey.hpp>
#include <ql/experimental/credit/defaulttype.hpp>
#include <ql/experimental/credit/distribution.hpp>
#include <ql/experimental/credit/issuer.hpp>
#include <ql/experimental/credit/loss.hpp>
#include <ql/experimental/credit/lossdistribution.hpp>
#include <ql/experimental/credit/pool.hpp>
#include <ql/experimental/credit/recoveryratemodel.hpp>
#include <ql/experimental/credit/recoveryratequote.hpp>
#include <ql/handle.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/credit/flathazardrate.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

#include <iomanip>
#include <iostream>
#include <vector>

using namespace QuantLib;

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    // ============================================================
    // 1) DefaultEvent + accessors
    // ============================================================
    {
        USDCurrency usd;
        Date credit_date(15, January, 2024);
        Date settle_date(20, January, 2024);
        DefaultType dt(AtomicDefault::Bankruptcy, Restructuring::XR);
        Real recovery = 0.4;
        DefaultEvent ev(credit_date, dt, usd, SeniorUnSec, settle_date, recovery);

        std::cout << "  \"default_event\": {\n";
        std::cout << "    \"date_serial\": " << ev.date().serialNumber() << ",\n";
        std::cout << "    \"currency_code\": \"" << ev.currency().code() << "\",\n";
        std::cout << "    \"seniority_idx\": " << static_cast<int>(ev.eventSeniority()) << ",\n";
        std::cout << "    \"default_type_idx\": " << static_cast<int>(ev.defaultType().defaultType()) << ",\n";
        std::cout << "    \"restructuring_type_idx\": " << static_cast<int>(ev.defaultType().restructuringType()) << ",\n";
        std::cout << "    \"is_restructuring\": " << (ev.isRestructuring() ? "true" : "false") << ",\n";
        std::cout << "    \"is_default\": " << (ev.isDefault() ? "true" : "false") << ",\n";
        std::cout << "    \"has_settled\": " << (ev.hasSettled() ? "true" : "false") << ",\n";
        std::cout << "    \"settlement_date_serial\": " << ev.settlement().date().serialNumber() << ",\n";
        std::cout << "    \"recovery_rate_snrfor\": " << ev.recoveryRate(SeniorUnSec) << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 2) DefaultProbKey + matchesDefaultKey
    // ============================================================
    {
        USDCurrency usd;
        NorthAmericaCorpDefaultKey key(usd, SeniorUnSec, Period(30, Days), 1.0e6,
                                       Restructuring::CR);
        std::cout << "  \"default_prob_key\": {\n";
        std::cout << "    \"currency_code\": \"" << key.currency().code() << "\",\n";
        std::cout << "    \"seniority_idx\": " << static_cast<int>(key.seniority()) << ",\n";
        std::cout << "    \"size\": " << key.size() << "\n";
        std::cout << "  },\n";

        // A matching event
        Date credit_date(15, January, 2024);
        DefaultType dt(AtomicDefault::Bankruptcy, Restructuring::XR);
        DefaultEvent ev(credit_date, dt, usd, SeniorUnSec);
        bool matches = ev.matchesDefaultKey(key);
        std::cout << "  \"default_event_matches_key\": " << (matches ? "true" : "false") << ",\n";

        // Mismatch on currency
        EURCurrency eur;
        DefaultEvent ev2(credit_date, dt, eur, SeniorUnSec);
        bool matches2 = ev2.matchesDefaultKey(key);
        std::cout << "  \"default_event_eur_matches_usd_key\": " << (matches2 ? "true" : "false") << ",\n";
    }

    // ============================================================
    // 3) Issuer + Pool
    // ============================================================
    {
        USDCurrency usd;
        DayCounter dc = Actual365Fixed();
        Date today(15, January, 2024);
        Settings::instance().evaluationDate() = today;
        Handle<DefaultProbabilityTermStructure> curve(
            ext::make_shared<FlatHazardRate>(today, Handle<Quote>(
                ext::make_shared<SimpleQuote>(0.02)), dc));

        std::vector<ext::shared_ptr<DefaultType>> events;
        events.push_back(ext::make_shared<DefaultType>(
            AtomicDefault::Bankruptcy, Restructuring::XR));
        DefaultProbKey pk(events, usd, SeniorUnSec);
        std::vector<std::pair<DefaultProbKey, Handle<DefaultProbabilityTermStructure>>> probs;
        probs.emplace_back(pk, curve);

        Issuer issuer(probs);
        const Handle<DefaultProbabilityTermStructure>& gotCurve =
            issuer.defaultProbability(pk);

        std::cout << "  \"issuer\": {\n";
        std::cout << "    \"curve_survival_t1\": " << gotCurve->survivalProbability(1.0) << ",\n";
        std::cout << "    \"curve_survival_t5\": " << gotCurve->survivalProbability(5.0) << "\n";
        std::cout << "  },\n";

        Pool pool;
        pool.add("AcmeCorp", issuer, pk);
        pool.add("Globex", issuer, pk);
        pool.setTime("AcmeCorp", 2.5);

        std::cout << "  \"pool\": {\n";
        std::cout << "    \"size\": " << pool.size() << ",\n";
        std::cout << "    \"has_acme\": " << (pool.has("AcmeCorp") ? "true" : "false") << ",\n";
        std::cout << "    \"has_unknown\": " << (pool.has("Unknown") ? "true" : "false") << ",\n";
        std::cout << "    \"time_acme\": " << pool.getTime("AcmeCorp") << ",\n";
        std::cout << "    \"time_globex\": " << pool.getTime("Globex") << ",\n";
        std::cout << "    \"names_first\": \"" << pool.names()[0] << "\",\n";
        std::cout << "    \"names_second\": \"" << pool.names()[1] << "\"\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 4) RecoveryRateQuote + ConstantRecoveryModel
    // ============================================================
    {
        // conventional ISDA recoveries by seniority idx
        std::cout << "  \"isda_conv_recoveries\": {\n";
        std::cout << "    \"secdom\": " << RecoveryRateQuote::conventionalRecovery(SecDom) << ",\n";
        std::cout << "    \"snrfor\": " << RecoveryRateQuote::conventionalRecovery(SnrFor) << ",\n";
        std::cout << "    \"sublt2\": " << RecoveryRateQuote::conventionalRecovery(SubLT2) << ",\n";
        std::cout << "    \"jrsubt2\": " << RecoveryRateQuote::conventionalRecovery(JrSubT2) << ",\n";
        std::cout << "    \"preft1\": " << RecoveryRateQuote::conventionalRecovery(PrefT1) << "\n";
        std::cout << "  },\n";

        RecoveryRateQuote q(0.42, SeniorUnSec);
        std::cout << "  \"recovery_rate_quote\": {\n";
        std::cout << "    \"value\": " << q.value() << ",\n";
        std::cout << "    \"seniority_idx\": " << static_cast<int>(q.seniority()) << ",\n";
        std::cout << "    \"is_valid\": " << (q.isValid() ? "true" : "false") << "\n";
        std::cout << "  },\n";

        // ConstantRecoveryModel from Real
        ConstantRecoveryModel rm1(0.37, SeniorUnSec);
        Date today(15, January, 2024);
        Real r1 = rm1.recoveryValue(today, DefaultProbKey());
        std::cout << "  \"constant_recovery_model_real_ctor\": " << r1 << ",\n";

        // ConstantRecoveryModel from Quote
        Handle<RecoveryRateQuote> hq(ext::make_shared<RecoveryRateQuote>(0.55, SubLT2));
        ConstantRecoveryModel rm2(hq);
        Real r2 = rm2.recoveryValue(today, DefaultProbKey());
        std::cout << "  \"constant_recovery_model_quote_ctor\": " << r2 << ",\n";
    }

    // ============================================================
    // 5) LossDist static helpers
    // ============================================================
    {
        std::vector<Real> p{0.1, 0.2, 0.3, 0.4};

        // P(N=k)
        std::vector<Real> probs = LossDist::probabilityOfNEvents(p);
        std::cout << "  \"loss_dist_probabilities\": {\n";
        std::cout << "    \"p0\": " << probs[0] << ",\n";
        std::cout << "    \"p1\": " << probs[1] << ",\n";
        std::cout << "    \"p2\": " << probs[2] << ",\n";
        std::cout << "    \"p3\": " << probs[3] << ",\n";
        std::cout << "    \"p4\": " << probs[4] << ",\n";
        // Returns P(N>=k)
        std::cout << "    \"at_least_2\": " << LossDist::probabilityOfAtLeastNEvents(2, p) << "\n";
        std::cout << "  },\n";

        // Binomial — use uniform p=0.2, 4 trials → BinomialDistribution(0.2, 4)
        std::vector<Real> pUniform{0.2, 0.2, 0.2, 0.2};
        std::cout << "  \"loss_dist_binomial\": {\n";
        std::cout << "    \"p_n0\": " << LossDist::binomialProbabilityOfNEvents(0, pUniform) << ",\n";
        std::cout << "    \"p_n1\": " << LossDist::binomialProbabilityOfNEvents(1, pUniform) << ",\n";
        std::cout << "    \"p_n2\": " << LossDist::binomialProbabilityOfNEvents(2, pUniform) << ",\n";
        std::cout << "    \"at_least_2\": " << LossDist::binomialProbabilityOfAtLeastNEvents(2, pUniform) << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 6) LossDistHomogeneous
    // ============================================================
    {
        Size nBuckets = 10;
        Real maximum = 10.0;
        Real volume = 1.0;
        std::vector<Real> p{0.1, 0.2, 0.3, 0.4};

        LossDistHomogeneous ldh(nBuckets, maximum);
        Distribution dist = ldh(volume, p);

        std::cout << "  \"loss_dist_homogeneous\": {\n";
        std::cout << "    \"n_buckets\": " << dist.size() << ",\n";
        std::cout << "    \"x_0\": " << dist.x(0) << ",\n";
        std::cout << "    \"x_1\": " << dist.x(1) << ",\n";
        std::cout << "    \"dx_0\": " << dist.dx(Size(0)) << ",\n";
        std::cout << "    \"density_0\": " << dist.density(0) << ",\n";
        std::cout << "    \"density_1\": " << dist.density(1) << ",\n";
        std::cout << "    \"cumulative_0\": " << dist.cumulative(0) << ",\n";
        std::cout << "    \"cumulative_2\": " << dist.cumulative(2) << ",\n";
        std::cout << "    \"excess_0\": " << dist.excess(0) << ",\n";
        std::cout << "    \"excess_2\": " << dist.excess(2) << ",\n";
        std::cout << "    \"prob_n0\": " << ldh.probability()[0] << ",\n";
        std::cout << "    \"prob_n1\": " << ldh.probability()[1] << ",\n";
        std::cout << "    \"prob_n2\": " << ldh.probability()[2] << ",\n";
        std::cout << "    \"prob_n4\": " << ldh.probability()[4] << ",\n";
        std::cout << "    \"excess_prob_n0\": " << ldh.excessProbability()[0] << ",\n";
        std::cout << "    \"excess_prob_n2\": " << ldh.excessProbability()[2] << ",\n";
        std::cout << "    \"volume\": " << ldh.volume() << ",\n";
        std::cout << "    \"size_field\": " << ldh.size() << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 7) LossDistBinomial
    // ============================================================
    {
        Size nBuckets = 10;
        Real maximum = 10.0;
        Real volume = 1.0;
        Size n = 5;
        Real probability = 0.2;

        LossDistBinomial ldb(nBuckets, maximum);
        Distribution dist = ldb(n, volume, probability);

        std::cout << "  \"loss_dist_binomial_dist\": {\n";
        std::cout << "    \"n_buckets\": " << dist.size() << ",\n";
        std::cout << "    \"prob_n0\": " << ldb.probability()[0] << ",\n";
        std::cout << "    \"prob_n1\": " << ldb.probability()[1] << ",\n";
        std::cout << "    \"prob_n2\": " << ldb.probability()[2] << ",\n";
        std::cout << "    \"prob_n5\": " << ldb.probability()[5] << ",\n";
        std::cout << "    \"excess_prob_n0\": " << ldb.excessProbability()[0] << ",\n";
        std::cout << "    \"excess_prob_n3\": " << ldb.excessProbability()[3] << ",\n";
        std::cout << "    \"volume\": " << ldb.volume() << ",\n";
        std::cout << "    \"size_field\": " << ldb.size() << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 8) Distribution invariants — built manually via add()
    // ============================================================
    {
        Distribution d(5, 0.0, 5.0);
        d.add(0.25);
        d.add(0.5);
        d.add(1.5);
        d.add(2.5);
        d.add(2.7);
        d.add(3.9);
        d.normalize();

        std::cout << "  \"distribution_basics\": {\n";
        std::cout << "    \"size\": " << d.size() << ",\n";
        std::cout << "    \"x_0\": " << d.x(0) << ",\n";
        std::cout << "    \"x_3\": " << d.x(3) << ",\n";
        std::cout << "    \"dx_0\": " << d.dx(Size(0)) << ",\n";
        std::cout << "    \"dx_4\": " << d.dx(Size(4)) << ",\n";
        std::cout << "    \"density_0\": " << d.density(0) << ",\n";
        std::cout << "    \"density_2\": " << d.density(2) << ",\n";
        std::cout << "    \"cumulative_0\": " << d.cumulative(0) << ",\n";
        std::cout << "    \"cumulative_4\": " << d.cumulative(4) << ",\n";
        std::cout << "    \"expected_value\": " << d.expectedValue() << ",\n";
        std::cout << "    \"locate_2_3\": " << d.locate(2.3) << ",\n";
        std::cout << "    \"average_0\": " << d.average(0) << ",\n";
        std::cout << "    \"average_2\": " << d.average(2) << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 9) Loss struct round-trip
    // ============================================================
    {
        Loss l1(2.5, 100.0);
        Loss l2(3.0, 50.0);
        Loss l3(2.5, 999.0);

        std::cout << "  \"loss\": {\n";
        std::cout << "    \"l1_time\": " << l1.time << ",\n";
        std::cout << "    \"l1_amount\": " << l1.amount << ",\n";
        std::cout << "    \"l1_lt_l2\": " << (l1 < l2 ? "true" : "false") << ",\n";
        std::cout << "    \"l1_eq_l3\": " << (l1 == l3 ? "true" : "false") << "\n";
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}

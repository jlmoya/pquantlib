// migration-harness/cpp/probes/v143_experimental_creditlm/probe.cpp
//
// Reference values for the ql/experimental/credit latent-model family @ v1.43:
//
//   DefaultLatentModel          defaultprobabilitylatentmodel.hpp
//   ConstantLossLatentmodel     constantlosslatentmodel.hpp   (lower-case 'm')
//   ConstantLossModel           constantlosslatentmodel.hpp
//   SpotRecoveryLatentModel     spotlosslatentmodel.hpp
//   detail::Root                randomdefaultlatentmodel.hpp
//   RandomLM / RandomDefaultLM  randomdefaultlatentmodel.hpp
//   RandomLossLM                randomlosslatentmodel.hpp
//
// WHY THIS PROBE LOOKS THE WAY IT DOES
// ------------------------------------
// The three integrable models (Default / ConstantLoss / SpotRecovery) are
// deterministic given a basket, so they are pinned directly: conditional
// probabilities at scripted market-factor values, the integrated quantities
// (probOfDefault, defaultCorrelation, probAtLeastNEvents, expectedLoss) and
// the recovery machinery. Both copula policies (Gaussian and Student-t) are
// covered where the model is templated on the copula.
//
// The two simulation models are NOT pinned statistically. Their RNG is a
// SobolRsg driven through LatentModel::FactorSampler, which is fully
// reproducible, so:
//
//   block F  pins the FactorSampler stream itself (Sobol draws -> copula
//            inversion), in isolation, for both model dimensionalities.
//            If F disagrees, nothing downstream of it means anything.
//   blocks G/H pin the *simulation buffer* — the (nameIdx, dayFromRef) and,
//            for the loss model, the quantised recovery of every simulated
//            default event in the first sims — and only then the statistics
//            computed off it. A statistic can agree by luck; the event
//            stream cannot.
//
// Reaching simsBuffer_ needs a probe-local subclass because the member is
// protected; the CRTP static_cast inside RandomLM stays valid because the
// subclass still IS-A RandomDefaultLM/RandomLossLM.
//
// Emits JSON on stdout; redirect to
// references/v143/experimental/creditlm.json.

#include <ql/currencies/america.hpp>
#include <ql/experimental/credit/basket.hpp>
#include <ql/experimental/credit/constantlosslatentmodel.hpp>
#include <ql/experimental/credit/defaultprobabilitylatentmodel.hpp>
#include <ql/experimental/credit/pool.hpp>
#include <ql/experimental/credit/randomdefaultlatentmodel.hpp>
#include <ql/experimental/credit/randomlosslatentmodel.hpp>
#include <ql/experimental/credit/spotlosslatentmodel.hpp>
#include <ql/experimental/math/gaussiancopulapolicy.hpp>
#include <ql/experimental/math/latentmodel.hpp>
#include <ql/experimental/math/tcopulapolicy.hpp>
#include <ql/instruments/claim.hpp>
#include <ql/math/randomnumbers/sobolrsg.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/credit/flathazardrate.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
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

// --------------------------------------------------------------------------
// Fixture. One evaluation date, four names on flat hazard-rate curves, one
// tranched basket. Everything below runs off this.
// --------------------------------------------------------------------------

const Date TODAY(15, January, 2024);

const std::vector<Real> HAZARD_RATES = {0.008, 0.015, 0.025, 0.035};
const std::vector<Real> NOTIONALS = {100.0, 100.0, 100.0, 100.0};
const Real ATTACH_RATIO = 0.0;
const Real DETACH_RATIO = 0.5;

// Default-variable factor loadings (one systemic factor).
const std::vector<std::vector<Real> > FCTRS_DEF = {{0.5}, {0.6}, {0.4}, {0.55}};
// Recovery-variable factor loadings, appended after the default ones for the
// spot-recovery model (2N variables).
const std::vector<std::vector<Real> > FCTRS_RR = {{0.3}, {0.35}, {0.25}, {0.4}};

const std::vector<Real> RECOVERIES = {0.4, 0.45, 0.35, 0.5};
const Real MODEL_A = 2.2;

// Student-t orders: numFactors + 1 = 2 entries, odd and > 2.
const std::vector<Integer> T_ORDERS = {5, 5};

// Market-factor realisations the conditional quantities are sampled at.
const std::vector<Real> MKT_GRID = {-2.5, -1.0, -0.25, 0.0, 0.4, 1.75, 3.0};
// Unconditional default probabilities the conditional quantities are sampled at.
const std::vector<Real> PROB_GRID = {0.005, 0.02, 0.1, 0.35, 0.7};

DefaultProbKey makeKey() {
    std::vector<boost::shared_ptr<DefaultType> > dummy;
    return NorthAmericaCorpDefaultKey(USDCurrency(), SeniorSec, Period(0, Weeks),
                                      1);
}

ext::shared_ptr<Basket> makeBasket(std::vector<std::string>& names,
                                   Real attach = ATTACH_RATIO,
                                   Real detach = DETACH_RATIO) {
    DefaultProbKey key = makeKey();
    auto pool = ext::make_shared<Pool>();
    names.clear();
    for (Size i = 0; i < HAZARD_RATES.size(); ++i) {
        Handle<DefaultProbabilityTermStructure> curve(
            ext::make_shared<FlatHazardRate>(
                TODAY, Handle<Quote>(ext::make_shared<SimpleQuote>(HAZARD_RATES[i])),
                Actual365Fixed()));
        std::vector<Issuer::key_curve_pair> probs;
        probs.emplace_back(key, curve);
        Issuer issuer(probs);
        std::string name = "N" + std::to_string(i);
        names.push_back(name);
        pool->add(name, issuer, key);
    }
    return ext::make_shared<Basket>(TODAY, names, NOTIONALS, pool, attach, detach,
                                    ext::make_shared<FaceValueClaim>());
}

// --------------------------------------------------------------------------
// Probe-local subclasses that expose the protected simulation buffer.
// --------------------------------------------------------------------------

class PeekDefaultLM : public RandomDefaultLM<GaussianCopulaPolicy> {
  public:
    PeekDefaultLM(const ext::shared_ptr<ConstantLossLatentmodel<GaussianCopulaPolicy> >& m,
                  Size nSims, Real accuracy, BigNatural seed)
    : RandomDefaultLM<GaussianCopulaPolicy>(m, nSims, accuracy, seed) {}

    // Flattened as [simIndex, nameIdx, dayFromRef] triples for the first
    // `maxSims` simulations.
    std::vector<long long> dumpEvents(Size maxSims) const {
        this->calculate();
        std::vector<long long> out;
        for (Size i = 0; i < std::min(maxSims, this->simsBuffer_.size()); ++i)
            for (const auto& e : this->simsBuffer_[i]) {
                out.push_back(static_cast<long long>(i));
                out.push_back(static_cast<long long>(e.nameIdx));
                out.push_back(static_cast<long long>(e.dayFromRef));
            }
        return out;
    }
    Size totalEvents() const {
        this->calculate();
        Size n = 0;
        for (const auto& s : this->simsBuffer_) n += s.size();
        return n;
    }
    // RandomLM's richer statistics are protected (Basket is the intended
    // caller and only forwards a subset); a derived class may reach them.
    using RandomLM<::QuantLib::RandomDefaultLM, GaussianCopulaPolicy,
                   SobolRsg>::expectedTrancheLossInterval;
    using RandomLM<::QuantLib::RandomDefaultLM, GaussianCopulaPolicy,
                   SobolRsg>::percentileAndInterval;
    using RandomLM<::QuantLib::RandomDefaultLM, GaussianCopulaPolicy,
                   SobolRsg>::computeHistogram;
    using RandomLM<::QuantLib::RandomDefaultLM, GaussianCopulaPolicy,
                   SobolRsg>::splitVaRAndError;
};

class PeekLossLM : public RandomLossLM<GaussianCopulaPolicy> {
  public:
    PeekLossLM(const ext::shared_ptr<SpotRecoveryLatentModel<GaussianCopulaPolicy> >& m,
               Size nSims, Real accuracy, BigNatural seed)
    : RandomLossLM<GaussianCopulaPolicy>(m, nSims, accuracy, seed) {}

    std::vector<long long> dumpEvents(Size maxSims) const {
        this->calculate();
        std::vector<long long> out;
        for (Size i = 0; i < std::min(maxSims, this->simsBuffer_.size()); ++i)
            for (const auto& e : this->simsBuffer_[i]) {
                out.push_back(static_cast<long long>(i));
                out.push_back(static_cast<long long>(e.nameIdx));
                out.push_back(static_cast<long long>(e.dayFromRef));
            }
        return out;
    }
    std::vector<Real> dumpRecoveries(Size maxSims) const {
        this->calculate();
        std::vector<Real> out;
        for (Size i = 0; i < std::min(maxSims, this->simsBuffer_.size()); ++i)
            for (const auto& e : this->simsBuffer_[i]) out.push_back(e.recovery());
        return out;
    }
    Size totalEvents() const {
        this->calculate();
        Size n = 0;
        for (const auto& s : this->simsBuffer_) n += s.size();
        return n;
    }
    using RandomLM<::QuantLib::RandomLossLM, GaussianCopulaPolicy,
                   SobolRsg>::expectedTrancheLossInterval;
    using RandomLM<::QuantLib::RandomLossLM, GaussianCopulaPolicy,
                   SobolRsg>::percentileAndInterval;
    using RandomLM<::QuantLib::RandomLossLM, GaussianCopulaPolicy,
                   SobolRsg>::computeHistogram;
    using RandomLM<::QuantLib::RandomLossLM, GaussianCopulaPolicy,
                   SobolRsg>::splitVaRAndError;
};

// --------------------------------------------------------------------------
// Blocks
// --------------------------------------------------------------------------

void blockSetup(const ext::shared_ptr<Basket>& basket) {
    emit_int("setup_today_serial", TODAY.serialNumber());
    emit_arr("setup_hazard_rates", HAZARD_RATES);
    emit_arr("setup_notionals", NOTIONALS);
    emit("setup_attach_amount", basket->attachmentAmount());
    emit("setup_detach_amount", basket->detachmentAmount());
    emit_int("setup_basket_size", static_cast<long long>(basket->size()));
    emit_arr("setup_recoveries", RECOVERIES);
    emit("setup_model_a", MODEL_A);
    emit_arr("setup_mkt_grid", MKT_GRID);
    emit_arr("setup_prob_grid", PROB_GRID);

    // Unconditional default probabilities of each name at the pinned horizons.
    const std::vector<Integer> years = {1, 3, 5};
    for (Integer y : years) {
        Date d = TODAY + Period(y, Years);
        std::vector<Real> ps;
        const ext::shared_ptr<Pool>& pool = basket->pool();
        for (Size i = 0; i < basket->size(); ++i)
            ps.push_back(pool->get(pool->names()[i])
                             .defaultProbability(basket->defaultKeys()[i])
                             ->defaultProbability(d));
        emit_iarr("setup_horizon_" + std::to_string(y) + "y_serial",
                  {d.serialNumber()});
        emit_arr("setup_uncond_pd_" + std::to_string(y) + "y", ps);
    }
    // The maximum-horizon probabilities RandomLM::initDates caches (4050 days).
    {
        Date maxHorizon = TODAY + Period(4050, Days);
        std::vector<Real> ps;
        const ext::shared_ptr<Pool>& pool = basket->pool();
        for (Size i = 0; i < basket->size(); ++i)
            ps.push_back(pool->get(pool->names()[i])
                             .defaultProbability(basket->defaultKeys()[i])
                             ->defaultProbability(maxHorizon, true));
        emit_int("setup_max_horizon_serial", maxHorizon.serialNumber());
        emit_arr("setup_horizon_default_ps", ps);
    }
}

// ---- A. detail::Root -----------------------------------------------------

void blockRoot(const ext::shared_ptr<Basket>& basket) {
    const ext::shared_ptr<Pool>& pool = basket->pool();
    const Handle<DefaultProbabilityTermStructure>& dts =
        pool->get(pool->names()[1]).defaultProbability(basket->defaultKeys()[1]);
    const Real pd = 0.05;
    detail::Root root(dts, pd);
    std::vector<Real> ts = {0.0, 0.5, 1.0, 30.0, 365.0, 900.0, 1234.5, 4050.0};
    std::vector<Real> vals;
    for (Real t : ts) vals.push_back(root(t));
    emit("root_pd", pd);
    emit_arr("root_ts", ts);
    emit_arr("root_values", vals);

    // The Brent inversion RandomDefaultLM performs on this functor, for a
    // scripted set of simulated default probabilities.
    std::vector<Real> pds = {0.001, 0.01, 0.05, 0.1, 0.19};
    std::vector<Real> strides;
    for (Real p : pds) {
        detail::Root r(dts, p);
        strides.push_back(
            static_cast<Real>(static_cast<Size>(Brent().solve(r, 1.e-6, 0., 1.))));
    }
    emit_arr("root_brent_pds", pds);
    emit_arr("root_brent_day_strides", strides);
}

// ---- B/C. DefaultLatentModel --------------------------------------------

template <class CP>
void blockDefaultLM(const std::string& tag, DefaultLatentModel<CP>& lm,
                    const ext::shared_ptr<Basket>& basket) {
    lm.resetBasket(basket);

    emit_int(tag + "_size", static_cast<long long>(lm.size()));
    emit_int(tag + "_num_factors", static_cast<long long>(lm.numFactors()));
    emit_int(tag + "_num_total_factors",
             static_cast<long long>(lm.numTotalFactors()));
    emit_arr(tag + "_idiosync_fctrs", lm.idiosyncFctrs());

    // Latent-variable correlation matrix.
    {
        std::vector<Real> correls;
        for (Size i = 0; i < lm.size(); ++i)
            for (Size j = 0; j < lm.size(); ++j)
                correls.push_back(lm.latentVariableCorrel(i, j));
        emit_arr(tag + "_latent_correl", correls);
    }
    // latentVarValue on a scripted full-factor sample (systemic then idiosync).
    {
        std::vector<Real> allFactors = {0.7, -1.3, 0.45, 2.1, -0.6};
        std::vector<Real> vals;
        for (Size i = 0; i < lm.size(); ++i)
            vals.push_back(lm.latentVarValue(allFactors, i));
        emit_arr(tag + "_latent_var_sample", allFactors);
        emit_arr(tag + "_latent_var_value", vals);
    }
    // cumulativeY / cumulativeZ / inverseCumulativeY.
    {
        std::vector<Real> xs = {-2.0, -0.5, 0.0, 0.75, 1.5};
        std::vector<Real> cy, cz, invy;
        for (Real x : xs) {
            cy.push_back(lm.cumulativeY(x, 0));
            cz.push_back(lm.cumulativeZ(x));
        }
        for (Real p : PROB_GRID) invy.push_back(lm.inverseCumulativeY(p, 0));
        emit_arr(tag + "_cumulative_xs", xs);
        emit_arr(tag + "_cumulative_y", cy);
        emit_arr(tag + "_cumulative_z", cz);
        emit_arr(tag + "_inverse_cumulative_y", invy);
    }
    // conditionalDefaultProbability over the (prob, m) grid, name by name.
    {
        std::vector<Real> cond, condInv;
        for (Size i = 0; i < lm.size(); ++i)
            for (Real p : PROB_GRID)
                for (Real m : MKT_GRID) {
                    std::vector<Real> mv(1, m);
                    cond.push_back(lm.conditionalDefaultProbability(p, i, mv));
                    condInv.push_back(lm.conditionalDefaultProbabilityInvP(
                        lm.inverseCumulativeY(p, i), i, mv));
                }
        emit_arr(tag + "_cond_def_prob", cond);
        emit_arr(tag + "_cond_def_prob_inv_p", condInv);
    }
    // Integrated quantities at the pinned horizons.
    {
        const std::vector<Integer> years = {1, 3, 5};
        for (Integer y : years) {
            Date d = TODAY + Period(y, Years);
            std::vector<Real> pods, correls, atLeast;
            for (Size i = 0; i < lm.size(); ++i) pods.push_back(lm.probOfDefault(i, d));
            for (Size i = 0; i < lm.size(); ++i)
                for (Size j = 0; j < lm.size(); ++j)
                    correls.push_back(lm.defaultCorrelation(d, i, j));
            for (Size n = 1; n <= lm.size(); ++n)
                atLeast.push_back(lm.probAtLeastNEvents(n, d));
            std::string sfx = "_" + std::to_string(y) + "y";
            emit_arr(tag + "_prob_of_default" + sfx, pods);
            emit_arr(tag + "_default_correlation" + sfx, correls);
            emit_arr(tag + "_prob_at_least_n" + sfx, atLeast);
        }
    }
}

// ---- D. ConstantLossLatentmodel / ConstantLossModel ----------------------

void blockConstantLoss(const ext::shared_ptr<Basket>& basket) {
    ConstantLossLatentmodel<GaussianCopulaPolicy> clm(
        FCTRS_DEF, RECOVERIES, LatentModelIntegrationType::GaussianQuadrature);
    clm.resetBasket(basket);

    emit_arr("cllm_recoveries", clm.recoveries());
    {
        std::vector<Real> byDate, byProb, byInvP, bySample, expRec;
        Date d = TODAY + Period(3, Years);
        std::vector<Real> mv(1, 0.4);
        for (Size i = 0; i < RECOVERIES.size(); ++i) {
            byDate.push_back(clm.conditionalRecovery(d, i, mv));
            byProb.push_back(clm.conditionalRecovery(Probability(0.1), i, mv));
            byInvP.push_back(clm.conditionalRecoveryInvP(-1.28, i, mv));
            bySample.push_back(clm.conditionalRecovery(Real(-0.9), i, d));
            expRec.push_back(clm.expectedRecovery(d, i, basket->defaultKeys()[i]));
        }
        emit_arr("cllm_cond_recovery_date", byDate);
        emit_arr("cllm_cond_recovery_prob", byProb);
        emit_arr("cllm_cond_recovery_inv_p", byInvP);
        emit_arr("cllm_cond_recovery_sample", bySample);
        emit_arr("cllm_expected_recovery", expRec);
    }
    // The inherited DefaultLatentModel behaviour must survive the derivation.
    {
        Date d = TODAY + Period(3, Years);
        std::vector<Real> pods, atLeast;
        for (Size i = 0; i < clm.size(); ++i) pods.push_back(clm.probOfDefault(i, d));
        for (Size n = 1; n <= clm.size(); ++n)
            atLeast.push_back(clm.probAtLeastNEvents(n, d));
        emit_arr("cllm_prob_of_default_3y", pods);
        emit_arr("cllm_prob_at_least_n_3y", atLeast);
    }

    // ConstantLossModel exposes the same numbers through the Basket/loss-model
    // interface (its statistics are protected, Basket is the only caller).
    {
        std::vector<std::string> names;
        ext::shared_ptr<Basket> b2 = makeBasket(names);
        // The `int()` is not decorative: ConstantLossModel's default argument
        // is written `copulaPolicy::initTraits()` without `typename`
        // (constantlosslatentmodel.hpp:127-128, 137-138), which clang rejects
        // the moment the default is used. Passing the traits explicitly is the
        // only way to instantiate the class.
        auto model = ext::make_shared<ConstantLossModel<GaussianCopulaPolicy> >(
            FCTRS_DEF, RECOVERIES, LatentModelIntegrationType::GaussianQuadrature,
            int());
        b2->setLossModel(model);
        Date d = TODAY + Period(3, Years);
        std::vector<Real> correls, atLeast, recRates;
        for (Size i = 0; i < b2->size(); ++i)
            for (Size j = 0; j < b2->size(); ++j)
                correls.push_back(b2->defaultCorrelation(d, i, j));
        for (Size n = 1; n <= b2->size(); ++n)
            atLeast.push_back(b2->probAtLeastNEvents(n, d));
        for (Size i = 0; i < b2->size(); ++i) recRates.push_back(b2->recoveryRate(d, i));
        emit_arr("clmodel_default_correlation_3y", correls);
        emit_arr("clmodel_prob_at_least_n_3y", atLeast);
        emit_arr("clmodel_recovery_rate_3y", recRates);
    }
}

// ---- E. SpotRecoveryLatentModel -----------------------------------------

std::vector<std::vector<Real> > spotFactorWeights() {
    std::vector<std::vector<Real> > w = FCTRS_DEF;
    for (const auto& r : FCTRS_RR) w.push_back(r);
    return w;
}

void blockSpotRecovery(const ext::shared_ptr<Basket>& basket) {
    SpotRecoveryLatentModel<GaussianCopulaPolicy> srm(
        spotFactorWeights(), RECOVERIES, MODEL_A,
        LatentModelIntegrationType::GaussianQuadrature);
    srm.resetBasket(basket);

    emit_int("srlm_size", static_cast<long long>(srm.size()));
    emit_int("srlm_num_factors", static_cast<long long>(srm.numFactors()));
    emit_arr("srlm_idiosync_fctrs", srm.idiosyncFctrs());

    const Size nNames = RECOVERIES.size();
    Date d = TODAY + Period(3, Years);

    // conditionalDefaultProbability(prob) and (date) over the market grid.
    {
        std::vector<Real> condP, condD, condInv;
        for (Size i = 0; i < nNames; ++i)
            for (Real m : MKT_GRID) {
                std::vector<Real> mv(1, m);
                condD.push_back(srm.conditionalDefaultProbability(d, i, mv));
                condInv.push_back(srm.conditionalDefaultProbabilityInvP(-1.5, i, mv));
                for (Real p : PROB_GRID)
                    condP.push_back(srm.conditionalDefaultProbability(p, i, mv));
            }
        emit_arr("srlm_cond_def_prob_p", condP);
        emit_arr("srlm_cond_def_prob_date_3y", condD);
        emit_arr("srlm_cond_def_prob_inv_p", condInv);
    }
    // Expected conditional recovery, three entry points.
    {
        std::vector<Real> expCond, expCondP, expCondInv;
        for (Size i = 0; i < nNames; ++i)
            for (Real m : MKT_GRID) {
                std::vector<Real> mv(1, m);
                expCond.push_back(srm.expCondRecovery(d, i, mv));
                expCondP.push_back(srm.expCondRecoveryP(0.12, i, mv));
                expCondInv.push_back(srm.expCondRecoveryInvPinvRR(-1.2, -0.25, i, mv));
            }
        emit_arr("srlm_exp_cond_recovery_3y", expCond);
        emit_arr("srlm_exp_cond_recovery_p", expCondP);
        emit_arr("srlm_exp_cond_recovery_inv", expCondInv);
    }
    // conditionalRecovery(latentVarSample, iName, d) — eq. 42.
    {
        std::vector<Real> samples = {-3.0, -1.4, -0.2, 0.6, 2.2};
        std::vector<Real> rr;
        for (Size i = 0; i < nNames; ++i)
            for (Real s : samples) rr.push_back(srm.conditionalRecovery(s, i, d));
        emit_arr("srlm_cond_recovery_samples", samples);
        emit_arr("srlm_cond_recovery_3y", rr);
    }
    // latentRRVarValue on a scripted full-factor sample.
    {
        std::vector<Real> allFactors = {0.7,  -1.3, 0.45, 2.1,  -0.6,
                                        1.15, 0.33, -0.8, 1.9};
        std::vector<Real> rrVals, defVals;
        for (Size i = 0; i < nNames; ++i) {
            rrVals.push_back(srm.latentRRVarValue(allFactors, i));
            defVals.push_back(srm.latentVarValue(allFactors, i));
        }
        emit_arr("srlm_all_factors", allFactors);
        emit_arr("srlm_latent_rr_var_value", rrVals);
        emit_arr("srlm_latent_var_value", defVals);
    }
    // conditionalExpLossRR / conditionalExpLossRRInv / expectedLoss are NOT
    // pinned: they cannot be instantiated in v1.43.
    //
    //   spotlosslatentmodel.hpp:307-316
    //     Real SpotRecoveryLatentModel<CP>::conditionalExpLossRRInv(...) const {
    //         return conditionalDefaultProbabilityInvP(invP, iName, mktFactors)
    //             * (1.-this->conditionalRecoveryInvPinvRR(invP, invRR, iName,
    //                                                      mktFactors));
    //     }
    //
    // `conditionalRecoveryInvPinvRR` is declared nowhere in the library (a
    // tree-wide grep finds this call site and nothing else); the member was
    // evidently renamed to `expCondRecoveryInvPinvRR` and this caller was
    // missed. Because these are templates the bodies are only instantiated on
    // use, so QuantLib itself builds — but the first caller gets
    //   error: no member named 'conditionalRecoveryInvPinvRR' in
    //          'QuantLib::SpotRecoveryLatentModel<...>'
    // and `conditionalExpLossRR` and `expectedLoss` both go through
    // `conditionalExpLossRRInv`, so all three are uninstantiable.
    //
    // There is therefore no C++ behaviour to cross-validate against, and the
    // port must not invent one.
    emit_str("srlm_cond_exp_loss_rr_status",
             "uninstantiable in v1.43: conditionalRecoveryInvPinvRR undeclared "
             "(spotlosslatentmodel.hpp:315)");
}

// ---- F. FactorSampler stream --------------------------------------------

void blockFactorSampler() {
    const BigNatural seed = 2863311530UL;
    {
        GaussianCopulaPolicy cop(FCTRS_DEF);
        emit_int("sampler_gauss_num_factors", static_cast<long long>(cop.numFactors()));
        LatentModel<GaussianCopulaPolicy>::FactorSampler<SobolRsg> smp(cop, seed);
        std::vector<Real> flat;
        for (Size k = 0; k < 6; ++k) {
            const std::vector<Real>& v = smp.nextSequence().value;
            flat.insert(flat.end(), v.begin(), v.end());
        }
        emit_arr("sampler_gauss_stream", flat);
    }
    {
        GaussianCopulaPolicy cop(spotFactorWeights());
        emit_int("sampler_spot_num_factors", static_cast<long long>(cop.numFactors()));
        LatentModel<GaussianCopulaPolicy>::FactorSampler<SobolRsg> smp(cop, seed);
        std::vector<Real> flat;
        for (Size k = 0; k < 4; ++k) {
            const std::vector<Real>& v = smp.nextSequence().value;
            flat.insert(flat.end(), v.begin(), v.end());
        }
        emit_arr("sampler_spot_stream", flat);
    }
}

// ---- G. RandomDefaultLM --------------------------------------------------

const Size N_SIMS = 512;
const BigNatural SIM_SEED = 2863311530UL;
const Real SIM_ACCURACY = 1.e-6;

void blockRandomDefault() {
    std::vector<std::string> names;
    ext::shared_ptr<Basket> basket = makeBasket(names);
    auto model = ext::make_shared<ConstantLossLatentmodel<GaussianCopulaPolicy> >(
        FCTRS_DEF, RECOVERIES, LatentModelIntegrationType::GaussianQuadrature);
    auto rdlm = ext::make_shared<PeekDefaultLM>(model, N_SIMS, SIM_ACCURACY, SIM_SEED);
    basket->setLossModel(rdlm);
    // DefaultLossModel::setBasket is only called from Basket::performCalculations,
    // so the model has no basket until the basket is first calculated. Any
    // basket statistic does it; this one is cheap and worth pinning anyway.
    emit("rdlm_remaining_tranche_notional", basket->remainingTrancheNotional());

    emit_int("rdlm_n_sims", static_cast<long long>(N_SIMS));
    emit_int("rdlm_seed", static_cast<long long>(SIM_SEED));
    emit_int("rdlm_total_events", static_cast<long long>(rdlm->totalEvents()));
    emit_iarr("rdlm_events_head", rdlm->dumpEvents(40));

    const std::vector<Integer> years = {1, 3, 5};
    for (Integer y : years) {
        Date d = TODAY + Period(y, Years);
        std::string sfx = "_" + std::to_string(y) + "y";
        std::vector<Real> atLeast;
        for (Size n = 0; n <= basket->size(); ++n)
            atLeast.push_back(basket->probAtLeastNEvents(n, d));
        emit_arr("rdlm_prob_at_least_n" + sfx, atLeast);

        std::vector<Real> nth;
        for (Size n = 1; n <= basket->size(); ++n) {
            std::vector<Probability> v = basket->probsBeingNthEvent(n, d);
            nth.insert(nth.end(), v.begin(), v.end());
        }
        emit_arr("rdlm_probs_being_nth" + sfx, nth);

        std::vector<Real> correls;
        for (Size i = 0; i < basket->size(); ++i)
            for (Size j = 0; j < basket->size(); ++j)
                correls.push_back(basket->defaultCorrelation(d, i, j));
        emit_arr("rdlm_default_correlation" + sfx, correls);

        emit("rdlm_expected_tranche_loss" + sfx, basket->expectedTrancheLoss(d));
    }

    Date d = TODAY + Period(5, Years);
    {
        std::pair<Real, Real> itv = rdlm->expectedTrancheLossInterval(d, 0.95);
        emit("rdlm_etl_interval_mean_5y", itv.first);
        emit("rdlm_etl_interval_width_5y", itv.second);
    }
    {
        std::vector<Real> percs = {0.5, 0.9, 0.95, 0.99};
        std::vector<Real> pv, es;
        for (Real p : percs) {
            pv.push_back(basket->percentile(d, p));
            es.push_back(basket->expectedShortfall(d, p));
        }
        emit_arr("rdlm_percentiles", percs);
        emit_arr("rdlm_percentile_5y", pv);
        emit_arr("rdlm_expected_shortfall_5y", es);
    }
    {
        std::tuple<Real, Real, Real> t = rdlm->percentileAndInterval(d, 0.95);
        emit_arr("rdlm_percentile_and_interval_5y",
                 {std::get<0>(t), std::get<1>(t), std::get<2>(t)});
    }
    {
        std::map<Real, Probability> ld = basket->lossDistribution(d);
        std::vector<Real> keys, vals;
        for (const auto& kv : ld) {
            keys.push_back(kv.first);
            vals.push_back(kv.second);
        }
        emit_int("rdlm_loss_distribution_size", static_cast<long long>(ld.size()));
        emit_arr("rdlm_loss_distribution_keys", keys);
        emit_arr("rdlm_loss_distribution_values", vals);
    }
    {
        Histogram h = rdlm->computeHistogram(d);
        std::vector<Real> freqs, breaks;
        for (Size i = 0; i < h.bins(); ++i) freqs.push_back(h.frequency(i));
        breaks.assign(h.breaks().begin(), h.breaks().end());
        emit_int("rdlm_histogram_bins", static_cast<long long>(h.bins()));
        emit_arr("rdlm_histogram_breaks", breaks);
        emit_arr("rdlm_histogram_frequencies", freqs);
    }
    {
        Real level = basket->percentile(d, 0.9);
        std::vector<Real> split = basket->splitVaRLevel(d, level);
        emit("rdlm_split_var_level_input", level);
        emit_arr("rdlm_split_var_level_5y", split);
        std::vector<std::vector<Real> > se = rdlm->splitVaRAndError(d, level, 0.95);
        std::vector<Real> flat;
        for (const auto& row : se) flat.insert(flat.end(), row.begin(), row.end());
        emit_arr("rdlm_split_var_and_error_5y", flat);
    }
}

// ---- H. RandomLossLM -----------------------------------------------------

void blockRandomLoss() {
    std::vector<std::string> names;
    ext::shared_ptr<Basket> basket = makeBasket(names);
    auto model = ext::make_shared<SpotRecoveryLatentModel<GaussianCopulaPolicy> >(
        spotFactorWeights(), RECOVERIES, MODEL_A,
        LatentModelIntegrationType::GaussianQuadrature);
    auto rllm = ext::make_shared<PeekLossLM>(model, N_SIMS, SIM_ACCURACY, SIM_SEED);
    basket->setLossModel(rllm);
    emit("rllm_remaining_tranche_notional", basket->remainingTrancheNotional());

    emit_int("rllm_n_sims", static_cast<long long>(N_SIMS));
    emit_int("rllm_total_events", static_cast<long long>(rllm->totalEvents()));
    emit_iarr("rllm_events_head", rllm->dumpEvents(40));
    emit_arr("rllm_recoveries_head", rllm->dumpRecoveries(40));

    const std::vector<Integer> years = {1, 3, 5};
    for (Integer y : years) {
        Date d = TODAY + Period(y, Years);
        std::string sfx = "_" + std::to_string(y) + "y";
        std::vector<Real> atLeast;
        for (Size n = 0; n <= basket->size(); ++n)
            atLeast.push_back(basket->probAtLeastNEvents(n, d));
        emit_arr("rllm_prob_at_least_n" + sfx, atLeast);

        std::vector<Real> nth;
        for (Size n = 1; n <= basket->size(); ++n) {
            std::vector<Probability> v = basket->probsBeingNthEvent(n, d);
            nth.insert(nth.end(), v.begin(), v.end());
        }
        emit_arr("rllm_probs_being_nth" + sfx, nth);

        std::vector<Real> correls;
        for (Size i = 0; i < basket->size(); ++i)
            for (Size j = 0; j < basket->size(); ++j)
                correls.push_back(basket->defaultCorrelation(d, i, j));
        emit_arr("rllm_default_correlation" + sfx, correls);

        emit("rllm_expected_tranche_loss" + sfx, basket->expectedTrancheLoss(d));
    }

    Date d = TODAY + Period(5, Years);
    {
        std::pair<Real, Real> itv = rllm->expectedTrancheLossInterval(d, 0.95);
        emit("rllm_etl_interval_mean_5y", itv.first);
        emit("rllm_etl_interval_width_5y", itv.second);
    }
    {
        std::vector<Real> percs = {0.5, 0.9, 0.95, 0.99};
        std::vector<Real> pv, es;
        for (Real p : percs) {
            pv.push_back(basket->percentile(d, p));
            es.push_back(basket->expectedShortfall(d, p));
        }
        emit_arr("rllm_percentile_5y", pv);
        emit_arr("rllm_expected_shortfall_5y", es);
    }
    {
        std::map<Real, Probability> ld = basket->lossDistribution(d);
        std::vector<Real> keys, vals;
        for (const auto& kv : ld) {
            keys.push_back(kv.first);
            vals.push_back(kv.second);
        }
        emit_int("rllm_loss_distribution_size", static_cast<long long>(ld.size()));
        emit_arr("rllm_loss_distribution_keys", keys);
        emit_arr("rllm_loss_distribution_values", vals);
    }
}

}  // namespace

int main() {
    Settings::instance().evaluationDate() = TODAY;

    std::cout << "{\n";

    std::vector<std::string> names;
    ext::shared_ptr<Basket> basket = makeBasket(names);

    blockSetup(basket);
    blockRoot(basket);

    {
        DefaultLatentModel<GaussianCopulaPolicy> lm(
            FCTRS_DEF, LatentModelIntegrationType::GaussianQuadrature);
        blockDefaultLM("dlm_gauss", lm, basket);
    }
    {
        std::vector<std::string> n2;
        ext::shared_ptr<Basket> b2 = makeBasket(n2);
        TCopulaPolicy::initTraits ini;
        ini.tOrders = T_ORDERS;
        DefaultLatentModel<TCopulaPolicy> lm(
            FCTRS_DEF, LatentModelIntegrationType::Trapezoid, ini);
        emit_iarr("dlm_t_orders", {T_ORDERS[0], T_ORDERS[1]});
        blockDefaultLM("dlm_t", lm, b2);
    }

    blockConstantLoss(basket);
    blockSpotRecovery(basket);
    blockFactorSampler();
    blockRandomDefault();
    blockRandomLoss();

    std::cout << "\n}\n";
    return 0;
}

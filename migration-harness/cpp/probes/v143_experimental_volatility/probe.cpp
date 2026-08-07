// migration-harness/cpp/probes/v143_experimental_volatility/probe.cpp
//
// Reference values for the ql/experimental volatility-interpolation gap set
// @ v1.43:
//
//   Svi, detail::SviSpecs             ql/experimental/volatility/sviinterpolation.hpp
//   NoArbSabr, detail::NoArbSabrSpecs ql/experimental/volatility/noarbsabrinterpolation.hpp
//   VannaVolga                        ql/experimental/barrieroption/vannavolgainterpolation.hpp
//   SwaptionVolCubeNoArbSabrModel     ql/experimental/volatility/noarbsabrswaptionvolatilitycube.hpp
//
// WHY THIS PROBE LOOKS THE WAY IT DOES
// ------------------------------------
// Four of these six names are *policy* types consumed by templates, so the
// only way to observe them is to instantiate the policy and call it directly.
// That is what blocks A (SviSpecs) and C (NoArbSabrSpecs) do: every member of
// the struct is exercised in isolation — dimension/eps/defaultValues/guess/
// direct/inverse/weight/instance — including the branches a port is tempted to
// drop (the paramIsFixed arms of `direct`, the sigmaI clamp arms of
// NoArbSabrSpecs::defaultValues and ::direct).
//
// The two *Interpolation classes those policies drive run a
// LevenbergMarquardt + Halton(seed 42) multi-start. PQuantLib delegates that
// optimisation to scipy, so a raw "did we get the same parameter vector"
// comparison is not a meaningful cross-validation. This probe therefore pins
// the calibration at three separable levels:
//
//   1. the ALL-FIXED interpolation (nothing to optimise) — this exercises
//      XABRInterpolationImpl::interpolationError / interpolationMaxError /
//      the vega weight vector with zero optimiser involvement, so it is an
//      exact, optimiser-independent pin of the reported diagnostics.
//      (Block B1/B2, D1/D2.)
//   2. the Halton guess stream itself (block G) plus SviSpecs::guess /
//      NoArbSabrSpecs::guess applied to it — the multi-start restart points
//      are then reproducible in Python without reproducing LM.
//   3. the converged fit on a NOISELESS slice generated from known model
//      parameters (blocks B3, D3). At the global optimum the residual is ~0,
//      so any correct optimiser must land on the same fitted curve; the fitted
//      curve, not the optimiser trajectory, is what is compared.
//
// Block H isolates the QuantLib error metric as pure arithmetic, because it is
// NOT a plain RMS:
//      error_ = sqrt(n * sum_i w_i e_i^2 / (n==1 ? 1 : n-1))
//      maxError_ = max_i |e_i|            (UNWEIGHTED, even when vega weighted)
// PQuantLib currently reports sqrt(mean(r^2)) and max|r| with r already
// weighted; block H makes that divergence checkable from arithmetic alone.
//
// Emits JSON on stdout; redirect to
// references/v143/experimental/volatility.json.

#include <ql/experimental/barrieroption/vannavolgainterpolation.hpp>
#include <ql/experimental/volatility/noarbsabr.hpp>
#include <ql/experimental/volatility/noarbsabrinterpolation.hpp>
#include <ql/experimental/volatility/noarbsabrsmilesection.hpp>
#include <ql/experimental/volatility/noarbsabrswaptionvolatilitycube.hpp>
#include <ql/experimental/volatility/sviinterpolation.hpp>
#include <ql/experimental/volatility/svismilesection.hpp>
#include <ql/math/array.hpp>
#include <ql/math/randomnumbers/haltonrsg.hpp>
#include <ql/pricingengines/blackformula.hpp>
#include <ql/utilities/null.hpp>

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

void emit_arr(const std::string& name, const Array& a) {
    std::vector<Real> v(a.begin(), a.end());
    emit_arr(name, v);
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

const Real NULLR = Null<Real>();

Array mkArray(std::initializer_list<Real> v) {
    Array a(v.size());
    Size i = 0;
    for (Real x : v) a[i++] = x;
    return a;
}

// The shared SVI test slice: raw-SVI parameters used to build a noiseless
// market slice, plus the strike ladder every SVI block reuses.
const Real SVI_T = 1.5;
const Real SVI_FWD = 0.03;
const Real SVI_A = 0.005;
const Real SVI_B = 0.4;
const Real SVI_SIGMA = 0.12;
const Real SVI_RHO = -0.3;
const Real SVI_M = 0.02;

std::vector<Real> sviStrikes() {
    return {0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.060, 0.070};
}

std::vector<Real> sviVols(const std::vector<Real>& strikes) {
    // exactly SviSmileSection::volatilityImpl on the generating parameters
    SviSmileSection sec(SVI_T, SVI_FWD, {SVI_A, SVI_B, SVI_SIGMA, SVI_RHO, SVI_M});
    std::vector<Real> v;
    v.reserve(strikes.size());
    for (Real k : strikes)
        v.push_back(sec.volatility(k));
    return v;
}

// The shared no-arb-SABR test slice.
const Real NAS_T = 1.0;
const Real NAS_FWD = 0.03;
const Real NAS_ALPHA = 0.03;
const Real NAS_BETA = 0.5;
const Real NAS_NU = 0.35;
const Real NAS_RHO = -0.20;

std::vector<Real> nasStrikes() {
    return {0.02, 0.025, 0.03, 0.035, 0.04};
}

std::vector<Real> nasVols(const std::vector<Real>& strikes) {
    NoArbSabrSmileSection sec(NAS_T, NAS_FWD, {NAS_ALPHA, NAS_BETA, NAS_NU, NAS_RHO});
    std::vector<Real> v;
    v.reserve(strikes.size());
    for (Real k : strikes)
        v.push_back(sec.volatility(k));
    return v;
}

// --------------------------------------------------------------------------
// Block A — detail::SviSpecs, member by member.
//
// SviSpecs is NOT an empty tag type: it carries dimension(), eps1(), eps2(),
// defaultValues(), guess(), direct(), inverse(), weight() and instance(), and
// the direct/inverse pair is a non-trivial bijection between the constrained
// SVI box and R^5. Every one of those is pinned here.
// --------------------------------------------------------------------------

void blockA() {
    detail::SviSpecs specs;

    emit_int("A_svi_dimension", static_cast<long long>(specs.dimension()));
    emit("A_svi_eps1", specs.eps1());
    emit("A_svi_eps2", specs.eps2());

    const std::vector<Real> noAdd;

    // --- A1 defaultValues: everything Null -> the full default cascade.
    {
        std::vector<Real> p = {NULLR, NULLR, NULLR, NULLR, NULLR};
        std::vector<bool> fx(5, false);
        specs.defaultValues(p, fx, SVI_FWD, SVI_T, noAdd);
        emit_arr("A_svi_defaultValues_allNull_t1p5", p);
    }
    // same cascade at a different expiry: only `a` moves (it is the only
    // default that reads expiryTime).
    {
        std::vector<Real> p = {NULLR, NULLR, NULLR, NULLR, NULLR};
        std::vector<bool> fx(5, false);
        specs.defaultValues(p, fx, SVI_FWD, 10.0, noAdd);
        emit_arr("A_svi_defaultValues_allNull_t10", p);
    }
    // --- A2 defaultValues with rho supplied: b's default reads rho, and a's
    // default reads b -> the cascade order is observable.
    {
        std::vector<Real> p = {NULLR, NULLR, NULLR, 0.8, NULLR};
        std::vector<bool> fx(5, false);
        specs.defaultValues(p, fx, SVI_FWD, SVI_T, noAdd);
        emit_arr("A_svi_defaultValues_rho0p8", p);
    }
    // --- A3 defaultValues where the `a` default takes the *second* max() arm
    // (the -b*sigma*sqrt(1-rho^2)+eps1 floor) rather than the variance arm.
    {
        std::vector<Real> p = {NULLR, 3.0, 0.9, -0.9, 2.0};
        std::vector<bool> fx(5, false);
        specs.defaultValues(p, fx, SVI_FWD, 0.05, noAdd);
        emit_arr("A_svi_defaultValues_aFloorArm", p);
    }
    // --- A4 nothing Null: defaultValues is a no-op.
    {
        std::vector<Real> p = {SVI_A, SVI_B, SVI_SIGMA, SVI_RHO, SVI_M};
        std::vector<bool> fx(5, false);
        specs.defaultValues(p, fx, SVI_FWD, SVI_T, noAdd);
        emit_arr("A_svi_defaultValues_noop", p);
    }

    // --- A5 guess: r is consumed by ONE running index in the order
    // sigma, rho, m, b, a — not the parameter order — and a fixed parameter
    // consumes nothing. Three masks pin that.
    const std::vector<Real> r = {0.10, 0.30, 0.70, 0.55, 0.90, 0.25};
    {
        Array v(5, 0.0);
        v[0] = SVI_A; v[1] = SVI_B; v[2] = SVI_SIGMA; v[3] = SVI_RHO; v[4] = SVI_M;
        std::vector<bool> fx(5, false);
        specs.guess(v, fx, SVI_FWD, SVI_T, r, noAdd);
        emit_arr("A_svi_guess_allFree", v);
    }
    {
        Array v(5, 0.0);
        v[0] = SVI_A; v[1] = SVI_B; v[2] = SVI_SIGMA; v[3] = SVI_RHO; v[4] = SVI_M;
        std::vector<bool> fx = {false, false, true, false, false};  // sigma fixed
        specs.guess(v, fx, SVI_FWD, SVI_T, r, noAdd);
        emit_arr("A_svi_guess_sigmaFixed", v);
    }
    {
        Array v(5, 0.0);
        v[0] = SVI_A; v[1] = SVI_B; v[2] = SVI_SIGMA; v[3] = SVI_RHO; v[4] = SVI_M;
        std::vector<bool> fx = {true, true, false, false, false};  // a, b fixed
        specs.guess(v, fx, SVI_FWD, SVI_T, r, noAdd);
        emit_arr("A_svi_guess_abFixed", v);
    }

    // --- A6 direct / inverse, all free.
    {
        std::vector<bool> fx(5, false);
        std::vector<Real> params = {SVI_A, SVI_B, SVI_SIGMA, SVI_RHO, SVI_M};
        const std::vector<Array> xs = {
            mkArray({0.10, 0.20, 0.30, 0.40, 0.05}),
            mkArray({-1.50, 2.00, 0.70, -0.90, 1.30}),
            mkArray({3.00, -4.00, -2.00, 5.00, -0.25}),
        };
        for (Size i = 0; i < xs.size(); ++i) {
            Array y = specs.direct(xs[i], fx, params, SVI_FWD);
            emit_arr("A_svi_direct_free_" + std::to_string(i), y);
            Array back = specs.inverse(y, fx, params, SVI_FWD);
            emit_arr("A_svi_inverse_of_direct_" + std::to_string(i), back);
            Array again = specs.direct(back, fx, params, SVI_FWD);
            emit_arr("A_svi_direct_roundtrip_" + std::to_string(i), again);
        }
    }
    // --- A7 direct with a/b FIXED: the two `if (paramIsFixed[k])` arms copy
    // params[k] straight through and short-circuit the transform.
    {
        std::vector<bool> fx = {true, true, false, false, false};
        std::vector<Real> params = {0.0075, 0.55, 0.0, 0.0, 0.0};
        Array x = mkArray({0.10, 0.20, 0.30, 0.40, 0.05});
        emit_arr("A_svi_direct_abFixed", specs.direct(x, fx, params, SVI_FWD));
    }
    // --- A8 direct with only b fixed (a still transformed, and a's formula
    // reads the *fixed* y[1]).
    {
        std::vector<bool> fx = {false, true, false, false, false};
        std::vector<Real> params = {0.0075, 0.55, 0.0, 0.0, 0.0};
        Array x = mkArray({0.10, 0.20, 0.30, 0.40, 0.05});
        emit_arr("A_svi_direct_bFixed", specs.direct(x, fx, params, SVI_FWD));
    }
    // --- A9 inverse on a plain admissible parameter vector.
    {
        std::vector<bool> fx(5, false);
        std::vector<Real> params = {SVI_A, SVI_B, SVI_SIGMA, SVI_RHO, SVI_M};
        Array y = mkArray({SVI_A, SVI_B, SVI_SIGMA, SVI_RHO, SVI_M});
        emit_arr("A_svi_inverse_generating", specs.inverse(y, fx, params, SVI_FWD));
    }

    // --- A10 weight(): blackFormulaStdDevDerivative(strike, forward, stdDev, 1.0)
    // — note there is NO sqrt(T) factor and NO displacement.
    {
        std::vector<Real> ws;
        std::vector<Real> strikes = sviStrikes();
        std::vector<Real> vols = sviVols(strikes);
        for (Size i = 0; i < strikes.size(); ++i) {
            Real stdDev = std::sqrt(vols[i] * vols[i] * SVI_T);
            ws.push_back(specs.weight(strikes[i], SVI_FWD, stdDev, noAdd));
        }
        emit_arr("A_svi_weight_raw", ws);
        Real s = 0.0;
        for (Real w : ws) s += w;
        std::vector<Real> wn;
        for (Real w : ws) wn.push_back(w / s);
        emit_arr("A_svi_weight_normalised", wn);
    }

    // --- A11 instance(): returns a SviSmileSection (typedef SviWrapper).
    {
        auto inst = specs.instance(SVI_T, SVI_FWD,
                                   {SVI_A, SVI_B, SVI_SIGMA, SVI_RHO, SVI_M}, noAdd);
        std::vector<Real> vols;
        for (Real k : sviStrikes())
            vols.push_back(inst->volatility(k));
        emit_arr("A_svi_instance_volatility", vols);
        emit("A_svi_instance_atmLevel", inst->atmLevel());
        emit("A_svi_instance_minStrike", inst->minStrike());
        emit("A_svi_instance_exerciseTime", inst->exerciseTime());
    }
}

// --------------------------------------------------------------------------
// Block B — the Svi factory and SviInterpolation.
// --------------------------------------------------------------------------

void blockB() {
    emit_bool("B_svi_factory_global", Svi::global);

    std::vector<Real> strikes = sviStrikes();
    std::vector<Real> vols = sviVols(strikes);
    emit_arr("B_svi_slice_strikes", strikes);
    emit_arr("B_svi_slice_vols", vols);
    emit("B_svi_slice_t", SVI_T);
    emit("B_svi_slice_forward", SVI_FWD);
    emit_arr("B_svi_slice_params",
             std::vector<Real>{SVI_A, SVI_B, SVI_SIGMA, SVI_RHO, SVI_M});

    // --- B1 ALL PARAMETERS FIXED, no vega weighting.
    // Nothing to optimise -> error_/maxError_ are pure functions of the slice.
    {
        SviInterpolation interp(strikes.begin(), strikes.end(), vols.begin(),
                                SVI_T, SVI_FWD,
                                SVI_A, SVI_B, SVI_SIGMA, SVI_RHO, SVI_M,
                                true, true, true, true, true,
                                /*vegaWeighted*/ false);
        interp.update();
        emit("B1_allfixed_rmsError", interp.rmsError());
        emit("B1_allfixed_maxError", interp.maxError());
        emit_int("B1_allfixed_endCriteria",
                 static_cast<long long>(interp.endCriteria()));
        emit_arr("B1_allfixed_weights", interp.interpolationWeights());
        emit_arr("B1_allfixed_params",
                 std::vector<Real>{interp.a(), interp.b(), interp.sigma(),
                                   interp.rho(), interp.m()});
        std::vector<Real> curve;
        for (Real k : strikes) curve.push_back(interp(k));
        emit_arr("B1_allfixed_curve", curve);
    }
    // --- B1b ALL FIXED but the parameters are DELIBERATELY WRONG, so the
    // residual is non-zero and the error metric is actually observable.
    {
        SviInterpolation interp(strikes.begin(), strikes.end(), vols.begin(),
                                SVI_T, SVI_FWD,
                                0.004, 0.35, 0.10, -0.25, 0.01,
                                true, true, true, true, true,
                                /*vegaWeighted*/ false);
        interp.update();
        emit("B1b_offparams_rmsError", interp.rmsError());
        emit("B1b_offparams_maxError", interp.maxError());
        std::vector<Real> curve;
        std::vector<Real> resid;
        for (Size i = 0; i < strikes.size(); ++i) {
            curve.push_back(interp(strikes[i]));
            resid.push_back(interp(strikes[i]) - vols[i]);
        }
        emit_arr("B1b_offparams_curve", curve);
        emit_arr("B1b_offparams_residuals", resid);
    }
    // --- B2 ALL FIXED with vega weighting: pins weights_ AND the fact that
    // maxError_ stays UNWEIGHTED while error_ uses the weights.
    {
        SviInterpolation interp(strikes.begin(), strikes.end(), vols.begin(),
                                SVI_T, SVI_FWD,
                                0.004, 0.35, 0.10, -0.25, 0.01,
                                true, true, true, true, true,
                                /*vegaWeighted*/ true);
        interp.update();
        emit("B2_vegaweighted_rmsError", interp.rmsError());
        emit("B2_vegaweighted_maxError", interp.maxError());
        emit_arr("B2_vegaweighted_weights", interp.interpolationWeights());
    }

    // --- B3 a real calibration on the noiseless slice, all five free.
    {
        SviInterpolation interp(strikes.begin(), strikes.end(), vols.begin(),
                                SVI_T, SVI_FWD,
                                NULLR, NULLR, NULLR, NULLR, NULLR,
                                false, false, false, false, false,
                                /*vegaWeighted*/ false);
        interp.update();
        emit_arr("B3_fit_params",
                 std::vector<Real>{interp.a(), interp.b(), interp.sigma(),
                                   interp.rho(), interp.m()});
        emit("B3_fit_rmsError", interp.rmsError());
        emit("B3_fit_maxError", interp.maxError());
        emit_int("B3_fit_endCriteria", static_cast<long long>(interp.endCriteria()));
        std::vector<Real> curve;
        for (Real k : strikes) curve.push_back(interp(k));
        emit_arr("B3_fit_curve", curve);
        // off-pillar evaluation: the fitted smile away from the input strikes.
        std::vector<Real> off = {0.012, 0.0225, 0.0325, 0.045, 0.065};
        std::vector<Real> offCurve;
        for (Real k : off) offCurve.push_back(interp(k));
        emit_arr("B3_fit_offpillar_strikes", off);
        emit_arr("B3_fit_offpillar_curve", offCurve);
    }

    // --- B4 the same fit reached through the Svi FACTORY, which is the class
    // under test: Svi::interpolate must produce the identical curve.
    {
        Svi factory(SVI_T, SVI_FWD, NULLR, NULLR, NULLR, NULLR, NULLR,
                    false, false, false, false, false);
        Interpolation interp =
            factory.interpolate(strikes.begin(), strikes.end(), vols.begin());
        interp.update();
        std::vector<Real> curve;
        for (Real k : strikes) curve.push_back(interp(k));
        emit_arr("B4_factory_curve", curve);
        std::vector<Real> off = {0.012, 0.0225, 0.0325, 0.045, 0.065};
        std::vector<Real> offCurve;
        for (Real k : off) offCurve.push_back(interp(k));
        emit_arr("B4_factory_offpillar_curve", offCurve);
    }
    // --- B5 the factory's own defaults differ from SviInterpolation's:
    // vegaWeighted defaults to FALSE on Svi and TRUE on SviInterpolation.
    emit_bool("B5_factory_vegaWeighted_default_is_false", true);
    emit_bool("B5_interpolation_vegaWeighted_default_is_true", true);
    emit("B5_errorAccept_default", 0.0020);
    emit_int("B5_maxGuesses_default", 50);
    emit_bool("B5_useMaxError_default", false);
}

// --------------------------------------------------------------------------
// Block C — detail::NoArbSabrSpecs, member by member.
// --------------------------------------------------------------------------

void blockC() {
    detail::NoArbSabrSpecs specs;

    emit_int("C_nas_dimension", static_cast<long long>(specs.dimension()));
    emit("C_nas_eps", specs.eps());

    // the model bounds the specs are written against
    emit("C_nas_beta_min", detail::NoArbSabrModel::beta_min);
    emit("C_nas_beta_max", detail::NoArbSabrModel::beta_max);
    emit("C_nas_sigmaI_min", detail::NoArbSabrModel::sigmaI_min);
    emit("C_nas_sigmaI_max", detail::NoArbSabrModel::sigmaI_max);
    emit("C_nas_nu_min", detail::NoArbSabrModel::nu_min);
    emit("C_nas_nu_max", detail::NoArbSabrModel::nu_max);
    emit("C_nas_rho_min", detail::NoArbSabrModel::rho_min);
    emit("C_nas_rho_max", detail::NoArbSabrModel::rho_max);

    const std::vector<Real> noAdd;

    // --- C1 defaultValues, everything Null. This runs SABRSpecs::defaultValues
    // first (beta=0.5, alpha=0.2*F^(1-beta), nu=sqrt(0.4), rho=0) then applies
    // the sigmaI clamp. At F=0.03 the plain-SABR alpha gives
    // sigmaI = 0.2 which is inside [0.05, 1.0], so no clamp fires.
    {
        std::vector<Real> p = {NULLR, NULLR, NULLR, NULLR};
        std::vector<bool> fx(4, false);
        specs.defaultValues(p, fx, NAS_FWD, NAS_T, noAdd);
        emit_arr("C_nas_defaultValues_allNull", p);
        emit("C_nas_defaultValues_allNull_sigmaI",
             p[0] * std::pow(NAS_FWD, p[1] - 1.0));
    }
    // --- C2 sigmaI BELOW the band with alpha free -> alpha is rescaled.
    {
        std::vector<Real> p = {0.001, 0.5, NULLR, NULLR};
        std::vector<bool> fx(4, false);
        specs.defaultValues(p, fx, NAS_FWD, NAS_T, noAdd);
        emit_arr("C_nas_defaultValues_sigmaLow_alphaFree", p);
        emit("C_nas_defaultValues_sigmaLow_alphaFree_sigmaI",
             p[0] * std::pow(NAS_FWD, p[1] - 1.0));
    }
    // --- C3 sigmaI BELOW the band with alpha FIXED -> beta is rescaled instead.
    {
        std::vector<Real> p = {0.001, 0.5, NULLR, NULLR};
        std::vector<bool> fx = {true, false, false, false};
        specs.defaultValues(p, fx, NAS_FWD, NAS_T, noAdd);
        emit_arr("C_nas_defaultValues_sigmaLow_alphaFixed", p);
        emit("C_nas_defaultValues_sigmaLow_alphaFixed_sigmaI",
             p[0] * std::pow(NAS_FWD, p[1] - 1.0));
    }
    // --- C4 sigmaI ABOVE the band, alpha free.
    {
        std::vector<Real> p = {0.9, 0.5, NULLR, NULLR};
        std::vector<bool> fx(4, false);
        specs.defaultValues(p, fx, NAS_FWD, NAS_T, noAdd);
        emit_arr("C_nas_defaultValues_sigmaHigh_alphaFree", p);
        emit("C_nas_defaultValues_sigmaHigh_alphaFree_sigmaI",
             p[0] * std::pow(NAS_FWD, p[1] - 1.0));
    }
    // --- C5 sigmaI ABOVE the band, alpha fixed, beta free.
    {
        std::vector<Real> p = {0.9, 0.5, NULLR, NULLR};
        std::vector<bool> fx = {true, false, false, false};
        specs.defaultValues(p, fx, NAS_FWD, NAS_T, noAdd);
        emit_arr("C_nas_defaultValues_sigmaHigh_alphaFixed", p);
    }
    // --- C6 both alpha and beta fixed with sigmaI out of band: C++ silently
    // leaves the inadmissible pair alone (the model ctor throws later).
    {
        std::vector<Real> p = {0.001, 0.5, NULLR, NULLR};
        std::vector<bool> fx = {true, true, false, false};
        specs.defaultValues(p, fx, NAS_FWD, NAS_T, noAdd);
        emit_arr("C_nas_defaultValues_bothFixed_unadjusted", p);
    }

    // --- C7 guess: the running index consumes r in the order beta, alpha,
    // nu, rho, and the alpha draw is made in sigmaI space then divided by
    // F^(beta-1) using the *just drawn* beta.
    const std::vector<Real> r = {0.10, 0.30, 0.70, 0.55, 0.90};
    {
        Array v(4, 0.0);
        v[0] = NAS_ALPHA; v[1] = NAS_BETA; v[2] = NAS_NU; v[3] = NAS_RHO;
        std::vector<bool> fx(4, false);
        specs.guess(v, fx, NAS_FWD, NAS_T, r, noAdd);
        emit_arr("C_nas_guess_allFree", v);
    }
    {
        Array v(4, 0.0);
        v[0] = NAS_ALPHA; v[1] = NAS_BETA; v[2] = NAS_NU; v[3] = NAS_RHO;
        std::vector<bool> fx = {false, true, false, false};  // beta fixed
        specs.guess(v, fx, NAS_FWD, NAS_T, r, noAdd);
        emit_arr("C_nas_guess_betaFixed", v);
    }
    {
        Array v(4, 0.0);
        v[0] = NAS_ALPHA; v[1] = NAS_BETA; v[2] = NAS_NU; v[3] = NAS_RHO;
        std::vector<bool> fx = {true, false, true, false};  // alpha, nu fixed
        specs.guess(v, fx, NAS_FWD, NAS_T, r, noAdd);
        emit_arr("C_nas_guess_alphaNuFixed", v);
    }

    // --- C8 direct / inverse, all free.
    {
        std::vector<bool> fx(4, false);
        std::vector<Real> params = {NAS_ALPHA, NAS_BETA, NAS_NU, NAS_RHO};
        const std::vector<Array> xs = {
            mkArray({0.10, 0.20, 0.30, 0.40}),
            mkArray({-1.50, 2.00, 0.70, -0.90}),
            mkArray({3.00, -4.00, -2.00, 5.00}),
        };
        for (Size i = 0; i < xs.size(); ++i) {
            Array y = specs.direct(xs[i], fx, params, NAS_FWD);
            emit_arr("C_nas_direct_free_" + std::to_string(i), y);
            Array back = specs.inverse(y, fx, params, NAS_FWD);
            emit_arr("C_nas_inverse_of_direct_" + std::to_string(i), back);
            Array again = specs.direct(back, fx, params, NAS_FWD);
            emit_arr("C_nas_direct_roundtrip_" + std::to_string(i), again);
        }
    }
    // --- C9 direct with alpha FIXED and the resulting sigmaI *inside* the band:
    // no beta adjustment.
    {
        std::vector<bool> fx = {true, false, false, false};
        std::vector<Real> params = {NAS_ALPHA, NAS_BETA, NAS_NU, NAS_RHO};
        Array x = mkArray({0.10, 0.20, 0.30, 0.40});
        emit_arr("C_nas_direct_alphaFixed_inBand", specs.direct(x, fx, params, NAS_FWD));
    }
    // --- C10 direct with alpha FIXED and sigmaI BELOW the band -> beta is
    // overwritten by the log-ratio formula (note this happens AFTER beta was
    // already computed from x[1], so the transform result is discarded).
    {
        std::vector<bool> fx = {true, false, false, false};
        std::vector<Real> params = {0.0005, NAS_BETA, NAS_NU, NAS_RHO};
        Array x = mkArray({0.10, 0.20, 0.30, 0.40});
        Array y = specs.direct(x, fx, params, NAS_FWD);
        emit_arr("C_nas_direct_alphaFixed_belowBand", y);
        emit("C_nas_direct_alphaFixed_belowBand_sigmaI",
             y[0] * std::pow(NAS_FWD, y[1] - 1.0));
    }
    // --- C11 direct with alpha FIXED and sigmaI ABOVE the band.
    {
        std::vector<bool> fx = {true, false, false, false};
        std::vector<Real> params = {0.9, NAS_BETA, NAS_NU, NAS_RHO};
        Array x = mkArray({0.10, 0.20, 0.30, 0.40});
        Array y = specs.direct(x, fx, params, NAS_FWD);
        emit_arr("C_nas_direct_alphaFixed_aboveBand", y);
        emit("C_nas_direct_alphaFixed_aboveBand_sigmaI",
             y[0] * std::pow(NAS_FWD, y[1] - 1.0));
    }
    // --- C12 direct with beta/nu/rho fixed (the plain pass-through arms).
    {
        std::vector<bool> fx = {false, true, true, true};
        std::vector<Real> params = {NAS_ALPHA, 0.7, 0.25, 0.15};
        Array x = mkArray({0.10, 0.20, 0.30, 0.40});
        emit_arr("C_nas_direct_betaNuRhoFixed", specs.direct(x, fx, params, NAS_FWD));
    }
    // --- C13 inverse on the generating parameter vector.
    {
        std::vector<bool> fx(4, false);
        std::vector<Real> params = {NAS_ALPHA, NAS_BETA, NAS_NU, NAS_RHO};
        Array y = mkArray({NAS_ALPHA, NAS_BETA, NAS_NU, NAS_RHO});
        emit_arr("C_nas_inverse_generating", specs.inverse(y, fx, params, NAS_FWD));
    }

    // --- C14 weight(): same blackFormulaStdDevDerivative as SviSpecs.
    {
        std::vector<Real> strikes = nasStrikes();
        std::vector<Real> vols = nasVols(strikes);
        std::vector<Real> ws;
        for (Size i = 0; i < strikes.size(); ++i) {
            Real stdDev = std::sqrt(vols[i] * vols[i] * NAS_T);
            ws.push_back(specs.weight(strikes[i], NAS_FWD, stdDev, noAdd));
        }
        emit_arr("C_nas_weight_raw", ws);
        Real s = 0.0;
        for (Real w : ws) s += w;
        std::vector<Real> wn;
        for (Real w : ws) wn.push_back(w / s);
        emit_arr("C_nas_weight_normalised", wn);
    }

    // --- C15 instance(): a NoArbSabrSmileSection (typedef NoArbSabrWrapper).
    {
        auto inst = specs.instance(NAS_T, NAS_FWD,
                                   {NAS_ALPHA, NAS_BETA, NAS_NU, NAS_RHO}, noAdd);
        std::vector<Real> vols;
        for (Real k : nasStrikes())
            vols.push_back(inst->volatility(k));
        emit_arr("C_nas_instance_volatility", vols);
        emit("C_nas_instance_atmLevel", inst->atmLevel());
        emit("C_nas_instance_absorptionProbability",
             inst->model()->absorptionProbability());
        emit("C_nas_instance_numericalForward", inst->model()->numericalForward());
        emit("C_nas_instance_optionPrice_atm", inst->model()->optionPrice(NAS_FWD));
    }
}

// --------------------------------------------------------------------------
// Block D — the NoArbSabr factory and NoArbSabrInterpolation.
// --------------------------------------------------------------------------

void blockD() {
    emit_bool("D_nas_factory_global", NoArbSabr::global);

    std::vector<Real> strikes = nasStrikes();
    std::vector<Real> vols = nasVols(strikes);
    emit_arr("D_nas_slice_strikes", strikes);
    emit_arr("D_nas_slice_vols", vols);
    emit("D_nas_slice_t", NAS_T);
    emit("D_nas_slice_forward", NAS_FWD);
    emit_arr("D_nas_slice_params",
             std::vector<Real>{NAS_ALPHA, NAS_BETA, NAS_NU, NAS_RHO});

    // --- D1 ALL FIXED on the generating parameters (residual == 0).
    {
        NoArbSabrInterpolation interp(strikes.begin(), strikes.end(), vols.begin(),
                                      NAS_T, NAS_FWD,
                                      NAS_ALPHA, NAS_BETA, NAS_NU, NAS_RHO,
                                      true, true, true, true,
                                      /*vegaWeighted*/ false);
        interp.update();
        emit("D1_allfixed_rmsError", interp.rmsError());
        emit("D1_allfixed_maxError", interp.maxError());
        emit_int("D1_allfixed_endCriteria",
                 static_cast<long long>(interp.endCriteria()));
        emit_arr("D1_allfixed_weights", interp.interpolationWeights());
        emit_arr("D1_allfixed_params",
                 std::vector<Real>{interp.alpha(), interp.beta(),
                                   interp.nu(), interp.rho()});
        std::vector<Real> curve;
        for (Real k : strikes) curve.push_back(interp(k));
        emit_arr("D1_allfixed_curve", curve);
    }
    // --- D1b ALL FIXED on deliberately wrong parameters (non-zero residual).
    {
        NoArbSabrInterpolation interp(strikes.begin(), strikes.end(), vols.begin(),
                                      NAS_T, NAS_FWD,
                                      0.028, 0.55, 0.30, -0.10,
                                      true, true, true, true,
                                      /*vegaWeighted*/ false);
        interp.update();
        emit("D1b_offparams_rmsError", interp.rmsError());
        emit("D1b_offparams_maxError", interp.maxError());
        std::vector<Real> curve, resid;
        for (Size i = 0; i < strikes.size(); ++i) {
            curve.push_back(interp(strikes[i]));
            resid.push_back(interp(strikes[i]) - vols[i]);
        }
        emit_arr("D1b_offparams_curve", curve);
        emit_arr("D1b_offparams_residuals", resid);
    }
    // --- D2 ALL FIXED with vega weighting.
    {
        NoArbSabrInterpolation interp(strikes.begin(), strikes.end(), vols.begin(),
                                      NAS_T, NAS_FWD,
                                      0.028, 0.55, 0.30, -0.10,
                                      true, true, true, true,
                                      /*vegaWeighted*/ true);
        interp.update();
        emit("D2_vegaweighted_rmsError", interp.rmsError());
        emit("D2_vegaweighted_maxError", interp.maxError());
        emit_arr("D2_vegaweighted_weights", interp.interpolationWeights());
    }

    // --- D3 a real calibration with only alpha free (beta/nu/rho pinned at the
    // generating values). One free parameter keeps the no-arb density
    // evaluations affordable while still exercising the optimiser arm.
    {
        NoArbSabrInterpolation interp(strikes.begin(), strikes.end(), vols.begin(),
                                      NAS_T, NAS_FWD,
                                      0.025, NAS_BETA, NAS_NU, NAS_RHO,
                                      false, true, true, true,
                                      /*vegaWeighted*/ false,
                                      ext::shared_ptr<EndCriteria>(),
                                      ext::shared_ptr<OptimizationMethod>(),
                                      0.0020, false, /*maxGuesses*/ 1);
        interp.update();
        emit_arr("D3_fit_params",
                 std::vector<Real>{interp.alpha(), interp.beta(),
                                   interp.nu(), interp.rho()});
        emit("D3_fit_rmsError", interp.rmsError());
        emit("D3_fit_maxError", interp.maxError());
        emit_int("D3_fit_endCriteria", static_cast<long long>(interp.endCriteria()));
        std::vector<Real> curve;
        for (Real k : strikes) curve.push_back(interp(k));
        emit_arr("D3_fit_curve", curve);
    }

    // --- D4 the same, reached through the NoArbSabr FACTORY.
    {
        NoArbSabr factory(NAS_T, NAS_FWD, 0.025, NAS_BETA, NAS_NU, NAS_RHO,
                          false, true, true, true, false,
                          ext::shared_ptr<EndCriteria>(),
                          ext::shared_ptr<OptimizationMethod>(),
                          0.0020, false, 1);
        Interpolation interp =
            factory.interpolate(strikes.begin(), strikes.end(), vols.begin());
        interp.update();
        std::vector<Real> curve;
        for (Real k : strikes) curve.push_back(interp(k));
        emit_arr("D4_factory_curve", curve);
    }
    // --- D5 the shift guard: NoArbSabrInterpolation rejects a non-zero shift.
    {
        bool threw = false;
        std::string msg;
        try {
            NoArbSabrInterpolation interp(
                strikes.begin(), strikes.end(), vols.begin(), NAS_T, NAS_FWD,
                NAS_ALPHA, NAS_BETA, NAS_NU, NAS_RHO, true, true, true, true,
                false, ext::shared_ptr<EndCriteria>(),
                ext::shared_ptr<OptimizationMethod>(), 0.0020, false, 50,
                /*shift*/ 0.01);
        } catch (const Error&) {
            threw = true;
        }
        emit_bool("D5_nonzero_shift_throws", threw);
    }
    emit_bool("D5_factory_vegaWeighted_default_is_false", true);
}

// --------------------------------------------------------------------------
// Block E — the VannaVolga factory and VannaVolgaInterpolation.
// --------------------------------------------------------------------------

void blockE() {
    emit_int("E_vv_requiredPoints", static_cast<long long>(VannaVolga::requiredPoints));

    // A textbook FX smile: 25d put / ATM / 25d call.
    const Real spot = 1.30;
    const Real dDiscount = 0.9851119396030626;   // exp(-0.015 * 1.0)
    const Real fDiscount = 0.9704455335485082;   // exp(-0.030 * 1.0)
    const Real T = 1.0;
    std::vector<Real> strikes = {1.20, 1.30, 1.45};
    std::vector<Real> vols = {0.1150, 0.1000, 0.1075};

    emit("E_vv_spot", spot);
    emit("E_vv_dDiscount", dDiscount);
    emit("E_vv_fDiscount", fDiscount);
    emit("E_vv_T", T);
    emit_arr("E_vv_strikes", strikes);
    emit_arr("E_vv_vols", vols);
    emit("E_vv_fwd", spot * fDiscount / dDiscount);

    std::vector<Real> query = {1.10, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45, 1.55};
    emit_arr("E_vv_query_strikes", query);

    {
        VannaVolgaInterpolation interp(strikes.begin(), strikes.end(),
                                       vols.begin(), spot, dDiscount,
                                       fDiscount, T);
        interp.update();
        std::vector<Real> out;
        for (Real k : query) out.push_back(interp(k, true));
        emit_arr("E_vv_interpolation_values", out);
        // the three pillars must be reproduced exactly by construction
        std::vector<Real> pillars;
        for (Real k : strikes) pillars.push_back(interp(k, true));
        emit_arr("E_vv_pillar_values", pillars);
    }
    {
        VannaVolga factory(spot, dDiscount, fDiscount, T);
        Interpolation interp =
            factory.interpolate(strikes.begin(), strikes.end(), vols.begin());
        interp.update();
        std::vector<Real> out;
        for (Real k : query) out.push_back(interp(k, true));
        emit_arr("E_vv_factory_values", out);
    }
    // primitive/derivative/secondDerivative are QL_FAIL stubs.
    {
        VannaVolgaInterpolation interp(strikes.begin(), strikes.end(),
                                       vols.begin(), spot, dDiscount,
                                       fDiscount, T);
        interp.update();
        bool p = false, d = false, s = false;
        try { interp.primitive(1.3, true); } catch (const Error&) { p = true; }
        try { interp.derivative(1.3, true); } catch (const Error&) { d = true; }
        try { interp.secondDerivative(1.3, true); } catch (const Error&) { s = true; }
        emit_bool("E_vv_primitive_throws", p);
        emit_bool("E_vv_derivative_throws", d);
        emit_bool("E_vv_secondDerivative_throws", s);
    }
    // exactly three points are required
    {
        std::vector<Real> k4 = {1.20, 1.30, 1.40, 1.50};
        std::vector<Real> v4 = {0.115, 0.10, 0.105, 0.11};
        bool threw = false;
        try {
            VannaVolgaInterpolation interp(k4.begin(), k4.end(), v4.begin(),
                                           spot, dDiscount, fDiscount, T);
            interp.update();
        } catch (const Error&) { threw = true; }
        emit_bool("E_vv_four_points_throws", threw);
    }
}

// --------------------------------------------------------------------------
// Block F — SwaptionVolCubeNoArbSabrModel + its XabrModelTraits specialisation.
//
// The struct itself is two typedefs; the behaviour lives in the traits
// specialisation keyed on it. Both are pinned.
// --------------------------------------------------------------------------

using NasTraits = XabrModelTraits<SwaptionVolCubeNoArbSabrModel>;

void blockF() {
    emit_int("F_cube_nParams", static_cast<long long>(NasTraits::nParams));

    std::vector<Real> strikes = nasStrikes();
    std::vector<Real> vols = nasVols(strikes);
    const std::vector<Real> params = {NAS_ALPHA, NAS_BETA, NAS_NU, NAS_RHO};
    const std::vector<bool> fixed = {true, true, true, true};

    // --- F1 createSmileSection -> a NoArbSabrSmileSection.
    {
        auto sec = NasTraits::createSmileSection(
            NAS_T, NAS_FWD, params, 0.0, VolatilityType::ShiftedLognormal);
        std::vector<Real> out;
        for (Real k : strikes) out.push_back(sec->volatility(k));
        emit_arr("F1_createSmileSection_volatility", out);
        emit("F1_createSmileSection_atmLevel", sec->atmLevel());
        emit("F1_createSmileSection_minStrike", sec->minStrike());
        emit("F1_createSmileSection_optionPrice_atm", sec->optionPrice(NAS_FWD));
        emit("F1_createSmileSection_digital_atm", sec->digitalOptionPrice(NAS_FWD));
    }

    // --- F2 createInterpolation, and the load-bearing property of this
    // specialisation: volatilityType is DROPPED, so Normal and
    // ShiftedLognormal produce identical interpolations.
    {
        auto lognormal = NasTraits::createInterpolation(
            strikes.begin(), strikes.end(), vols.begin(), NAS_T, NAS_FWD,
            params, fixed, /*vegaWeighted*/ false,
            ext::shared_ptr<EndCriteria>(), ext::shared_ptr<OptimizationMethod>(),
            0.0020, false, 50, /*shift*/ 0.0, VolatilityType::ShiftedLognormal);
        lognormal->update();
        auto normal = NasTraits::createInterpolation(
            strikes.begin(), strikes.end(), vols.begin(), NAS_T, NAS_FWD,
            params, fixed, /*vegaWeighted*/ false,
            ext::shared_ptr<EndCriteria>(), ext::shared_ptr<OptimizationMethod>(),
            0.0020, false, 50, /*shift*/ 0.0, VolatilityType::Normal);
        normal->update();
        std::vector<Real> lnCurve, nCurve;
        for (Real k : strikes) {
            lnCurve.push_back((*lognormal)(k));
            nCurve.push_back((*normal)(k));
        }
        emit_arr("F2_createInterpolation_lognormal_curve", lnCurve);
        emit_arr("F2_createInterpolation_normal_curve", nCurve);
        bool identical = true;
        for (Size i = 0; i < lnCurve.size(); ++i)
            identical = identical && (lnCurve[i] == nCurve[i]);
        emit_bool("F2_volatilityType_is_ignored", identical);
        emit("F2_createInterpolation_rmsError", lognormal->rmsError());
        emit("F2_createInterpolation_maxError", lognormal->maxError());

        // --- F3 extractGamma is hard-wired to 0 for this model (nParams < 5,
        // so the cube never calls it, but the traits member exists).
        emit("F3_extractGamma", NasTraits::extractGamma(lognormal));
    }

    // --- F4 the SABR traits are the primary template with the same nParams,
    // which is what makes the NoArbSabr specialisation necessary at all
    // (its Interpolation ctor has no volatilityType argument).
    emit_int("F4_sabr_traits_nParams",
             static_cast<long long>(XabrModelTraits<SwaptionVolCubeSabrModel>::nParams));
}

// --------------------------------------------------------------------------
// Block G — the multi-start guess stream.
//
// XABRInterpolationImpl::calculate builds `HaltonRsg halton(freeParameters, 42)`
// (randomStart = true, randomShift = false by default) and feeds each drawn
// sequence to Model().guess(). Pinning the raw stream plus the guess applied
// to it makes the restart points reproducible in a port without reproducing
// LevenbergMarquardt.
// --------------------------------------------------------------------------

void blockG() {
    for (Size dim : {4u, 5u}) {
        HaltonRsg halton(dim, 42);
        std::vector<Real> flat;
        for (int draw = 0; draw < 4; ++draw) {
            const auto& s = halton.nextSequence();
            for (Size i = 0; i < dim; ++i)
                flat.push_back(s.value[i]);
        }
        emit_arr("G_halton_seed42_dim" + std::to_string(dim) + "_first4", flat);
    }
    // SviSpecs::guess applied to the first four dim-5 Halton draws, starting
    // from the all-Null defaultValues point (which is what calculate() does).
    {
        detail::SviSpecs specs;
        const std::vector<Real> noAdd;
        std::vector<Real> p = {NULLR, NULLR, NULLR, NULLR, NULLR};
        std::vector<bool> fx(5, false);
        specs.defaultValues(p, fx, SVI_FWD, SVI_T, noAdd);
        Array guess(5);
        for (Size i = 0; i < 5; ++i) guess[i] = p[i];
        HaltonRsg halton(5, 42);
        std::vector<Real> flat;
        for (int draw = 0; draw < 4; ++draw) {
            const auto& s = halton.nextSequence();
            specs.guess(guess, fx, SVI_FWD, SVI_T, s.value, noAdd);
            for (Size i = 0; i < 5; ++i) flat.push_back(guess[i]);
        }
        emit_arr("G_svi_guess_stream_first4", flat);
    }
    // NoArbSabrSpecs::guess applied to the first four dim-4 Halton draws.
    {
        detail::NoArbSabrSpecs specs;
        const std::vector<Real> noAdd;
        std::vector<Real> p = {NULLR, NULLR, NULLR, NULLR};
        std::vector<bool> fx(4, false);
        specs.defaultValues(p, fx, NAS_FWD, NAS_T, noAdd);
        Array guess(4);
        for (Size i = 0; i < 4; ++i) guess[i] = p[i];
        HaltonRsg halton(4, 42);
        std::vector<Real> flat;
        for (int draw = 0; draw < 4; ++draw) {
            const auto& s = halton.nextSequence();
            specs.guess(guess, fx, NAS_FWD, NAS_T, s.value, noAdd);
            for (Size i = 0; i < 4; ++i) flat.push_back(guess[i]);
        }
        emit_arr("G_nas_guess_stream_first4", flat);
    }
}

// --------------------------------------------------------------------------
// Block H — the XABR error metric, isolated as arithmetic.
//
//   interpolationSquaredError = sum_i w_i * e_i^2
//   interpolationError        = sqrt(n * squaredError / (n==1 ? 1 : n-1))
//   interpolationMaxError     = max_i |e_i|         <- UNWEIGHTED
//
// With the default (non-vega) weights w_i = 1/n this collapses to
// sqrt(sum e_i^2 / (n-1)), i.e. an (n-1) denominator, NOT sqrt(mean(e^2)).
// --------------------------------------------------------------------------

Real xabrError(const std::vector<Real>& e, const std::vector<Real>& w) {
    Real total = 0.0;
    for (Size i = 0; i < e.size(); ++i)
        total += e[i] * e[i] * w[i];
    Size n = e.size();
    return std::sqrt(n * total / (n == 1 ? 1 : (n - 1)));
}

void blockH() {
    const std::vector<Real> e = {1e-3, -2e-3, 5e-4, 3e-3, -1.5e-3, 2.5e-4, -8e-4};
    emit_arr("H_errors", e);

    {
        std::vector<Real> w(e.size(), 1.0 / e.size());
        emit_arr("H_flat_weights", w);
        emit("H_flat_interpolationError", xabrError(e, w));
        Real ss = 0.0;
        for (Real x : e) ss += x * x;
        emit("H_plain_rms_for_contrast", std::sqrt(ss / e.size()));
        emit("H_ratio_cpp_over_plain_rms",
             xabrError(e, w) / std::sqrt(ss / e.size()));
    }
    {
        std::vector<Real> w = {0.05, 0.10, 0.20, 0.30, 0.20, 0.10, 0.05};
        emit_arr("H_shaped_weights", w);
        emit("H_shaped_interpolationError", xabrError(e, w));
        Real maxAbs = 0.0;
        for (Real x : e) maxAbs = std::max(maxAbs, std::fabs(x));
        emit("H_maxError_is_unweighted", maxAbs);
        Real maxWeighted = 0.0;
        for (Size i = 0; i < e.size(); ++i)
            maxWeighted = std::max(maxWeighted, std::fabs(e[i] * std::sqrt(w[i])));
        emit("H_max_of_weighted_residual_for_contrast", maxWeighted);
    }
    emit_int("H_n", static_cast<long long>(e.size()));
    emit_str("H_note",
             "XABRInterpolationImpl::interpolationError divides the weighted "
             "squared error by (n-1), not n, and interpolationMaxError is the "
             "max of the UNWEIGHTED residual even when vegaWeighted is on.");
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
    blockG();
    blockH();
    std::cout << "\n}\n";
    return 0;
}

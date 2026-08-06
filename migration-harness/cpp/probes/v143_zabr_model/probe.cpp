// migration-harness/cpp/probes/v143_zabr_model/probe.cpp
//
// Pins QuantLib::ZabrModel (ql/termstructures/volatility/zabr.{hpp,cpp}).
//
// ZabrModel is five closed-form-ish surfaces stacked on one another, and each
// layer has a distinct way of going subtly wrong in a port. The cases below
// are chosen so that each failure mode produces a *visible* number rather than
// a plausible-looking one.
//
//  * nu is TRANSFORMED in the constructor: nu_ = nu * alpha^(1-gamma). Every
//    downstream formula (y, F, fullFdPrice's v0/v1) uses the transformed nu,
//    while the public nu() inspector returns the transformed value too. A port
//    that stores the raw input nu agrees exactly at gamma == 1 (where the
//    factor is 1) and disagrees everywhere else — hence P1 (gamma == 1) is
//    paired with P2/P3 (gamma below/above 1) on otherwise identical params.
//
//  * x(K) has two entirely different code paths. At gamma == 1 it is the
//    closed form log((J + nu y - rho)/(1 - rho))/nu; otherwise it is an
//    AdaptiveRungeKutta<Real>(1e-8, 1e-5, 0.0) integration of du/dy = F(y,u).
//    That integrator is Cash-Karp with the Numerical-Recipes error control
//    (yScale = |y| + |dydx*h| + TINY, errmax /= eps), NOT a generic RK45 with
//    rtol/atol — the second constructor argument is the *initial step size*,
//    not a relative tolerance. Substituting any other adaptive integrator
//    moves the answer at the 1e-5..1e-6 level, which the gamma != 1 blocks
//    expose.
//
//  * x(std::vector) integrates INCREMENTALLY: the strikes are sorted by y,
//    the walk starts at the y closest to zero and marches outward in both
//    directions, carrying (y0, u0) from one strike to the next. So a vector
//    call is NOT the same arithmetic as N scalar calls, and the two disagree
//    at the integrator's tolerance. Both are emitted for identical strike
//    lists (`*_scalar` vs `*_vector`) precisely so a port that implements one
//    and reuses it for the other is caught.
//
//  * y(K) branches on close(beta, 1.0) — the log form vs the power form —
//    and separately on strike < 0. P4 uses beta == 1 to reach the log branch.
//
//  * fdPrice's grid DEPENDS ON THE STRIKE VECTOR: start = min(1e-5, K0*0.5),
//    end = max(0.10, Kn*1.5). So fdPrice({k}) and the k-th entry of
//    fdPrice({...,k,...}) are different numbers whenever the vector's extremes
//    move the grid. `fd_price_vector` includes a 0.12 strike (which pushes
//    `end` from 0.10 to 0.18) while `fd_price_scalar` prices each strike on
//    its own grid; a port that hoists the grid out of the call cannot match
//    both.
//
//  * fullFdPrice is a 2-D (forward, vol) PDE with a Glued1dMesher built from
//    two Concentrating1dMeshers with required points at min/max(F, K), solved
//    with FdmSchemeDesc::Hundsdorfer() (theta = 0.5 + sqrt(3)/6, mu = 0.5)
//    after 5 implicit-Euler damping steps, then read off a BicubicSpline at
//    (forward, alpha). Strikes below/at/above the forward are priced because
//    the mesher split point (x0+x1)/2 and the sizefa/sizefb allocation depend
//    on where the strike sits relative to the forward.
//
// Parameter sets P1..P3 share (alpha, beta, nu, rho, tau, forward) with the
// C++ test-suite's ZabrTests::testConsistency and differ only in gamma, so any
// discrepancy between them is attributable to the gamma machinery alone.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/zabr/model.json.

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <ql/termstructures/volatility/zabr.hpp>

using namespace QuantLib;

namespace {

struct Params {
    const char* key;
    Real tau;
    Real forward;
    Real alpha;
    Real beta;
    Real nu;
    Real rho;
    Real gamma;
    bool withFd;
};

// P1..P3: the test-suite's consistency parameters, gamma swept below / at /
// above 1 so the closed-form and ODE arms of x(K) are both covered on an
// otherwise identical model.
// P4: beta == 1 reaches the logarithmic branch of y(K), on a shorter expiry
// and a fatter vol-of-vol so the RK integration spans a different y range.
const Params kParams[] = {
    {"gamma_1_00", 5.0, 0.03, 0.08, 0.70, 0.20, -0.30, 1.00, true},
    {"gamma_0_75", 5.0, 0.03, 0.08, 0.70, 0.20, -0.30, 0.75, true},
    {"gamma_1_30", 5.0, 0.03, 0.08, 0.70, 0.20, -0.30, 1.30, false},
    {"beta_1_gamma_0_85", 2.0, 0.05, 0.20, 1.00, 0.30, 0.20, 0.85, false},
};

// Deep ITM (0.001 = F/30) through far OTM (0.50 ~ 17x F), with the ATM point
// 0.03 included so the close(strike, forward_) short-circuit in both
// volatility helpers is exercised. Strictly ascending, as x(vector) requires.
const Real kStrikes[] = {0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.06, 0.10, 0.20, 0.50};
const Size kNumStrikes = sizeof(kStrikes) / sizeof(kStrikes[0]);

// P4 has forward 0.05, so its ATM point differs; a separate ladder keeps the
// close(strike, forward_) branch reachable there too.
const Real kStrikesP4[] = {0.005, 0.01, 0.02, 0.035, 0.05, 0.07, 0.10, 0.15, 0.30, 0.60};

// fdPrice grid endpoints are functions of the strike vector's first and last
// entries, so the vector deliberately reaches past 0.10/1.5 = 0.0667 to move
// `end` away from its 0.10 floor.
const Real kFdStrikes[] = {0.01, 0.02, 0.03, 0.04, 0.06, 0.12};
const Size kNumFdStrikes = sizeof(kFdStrikes) / sizeof(kFdStrikes[0]);

// fullFdPrice is priced below, at and above the forward: the f-mesher is glued
// at (min(F,K) + max(F,K))/2 with required points at both, so the grid itself
// changes shape with the strike's side.
const Real kFullFdStrikes[] = {0.02, 0.03, 0.05};
const Size kNumFullFdStrikes = sizeof(kFullFdStrikes) / sizeof(kFullFdStrikes[0]);

void emitArray(const char* key, const std::vector<Real>& v, bool trailingComma) {
    std::cout << "    \"" << key << "\": [";
    for (Size i = 0; i < v.size(); ++i) {
        std::cout << (i != 0U ? ", " : "") << v[i];
    }
    std::cout << "]" << (trailingComma ? "," : "") << "\n";
}

void emitParams(const Params& p) {
    const ZabrModel m(p.tau, p.forward, p.alpha, p.beta, p.nu, p.rho, p.gamma);

    const std::vector<Real> strikes(
        p.beta == 1.0 ? std::vector<Real>(kStrikesP4, kStrikesP4 + kNumStrikes)
                      : std::vector<Real>(kStrikes, kStrikes + kNumStrikes));

    std::cout << "  \"" << p.key << "\": {\n";
    std::cout << "    \"expiry_time\": " << m.expiryTime() << ",\n";
    std::cout << "    \"forward\": " << m.forward() << ",\n";
    std::cout << "    \"alpha\": " << m.alpha() << ",\n";
    std::cout << "    \"beta\": " << m.beta() << ",\n";
    // NOTE: nu() returns the TRANSFORMED nu (nu_input * alpha^(1-gamma)), not
    // the constructor argument. Pinning it is the cheapest way to catch a port
    // that forgot the transform.
    std::cout << "    \"nu_transformed\": " << m.nu() << ",\n";
    std::cout << "    \"nu_input\": " << p.nu << ",\n";
    std::cout << "    \"rho\": " << m.rho() << ",\n";
    std::cout << "    \"gamma\": " << m.gamma() << ",\n";
    emitArray("strikes", strikes, true);

    std::vector<Real> lnScalar, nScalar, lvScalar;
    for (Real k : strikes) {
        lnScalar.push_back(m.lognormalVolatility(k));
        nScalar.push_back(m.normalVolatility(k));
        lvScalar.push_back(m.localVolatility(k));
    }
    emitArray("lognormal_vol_scalar", lnScalar, true);
    emitArray("normal_vol_scalar", nScalar, true);
    emitArray("local_vol_scalar", lvScalar, true);

    emitArray("lognormal_vol_vector", m.lognormalVolatility(strikes), true);
    emitArray("normal_vol_vector", m.normalVolatility(strikes), true);
    emitArray("local_vol_vector", m.localVolatility(strikes), p.withFd);

    if (p.withFd) {
        const std::vector<Real> fdStrikes(kFdStrikes, kFdStrikes + kNumFdStrikes);
        emitArray("fd_strikes", fdStrikes, true);
        emitArray("fd_price_vector", m.fdPrice(fdStrikes), true);

        std::vector<Real> fdScalar;
        for (Real k : fdStrikes)
            fdScalar.push_back(m.fdPrice(k));
        emitArray("fd_price_scalar", fdScalar, true);

        const std::vector<Real> fullFdStrikes(kFullFdStrikes,
                                              kFullFdStrikes + kNumFullFdStrikes);
        emitArray("full_fd_strikes", fullFdStrikes, true);
        std::vector<Real> fullFd;
        for (Real k : fullFdStrikes)
            fullFd.push_back(m.fullFdPrice(k));
        emitArray("full_fd_price", fullFd, false);
    }

    std::cout << "  }";
}

} // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";
    const Size n = sizeof(kParams) / sizeof(kParams[0]);
    for (Size i = 0; i < n; ++i) {
        emitParams(kParams[i]);
        std::cout << (i + 1 < n ? "," : "") << "\n";
    }
    std::cout << "}\n";
    return 0;
}

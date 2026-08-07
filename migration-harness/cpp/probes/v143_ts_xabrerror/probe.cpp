// migration-harness/cpp/probes/v143_ts_xabrerror/probe.cpp
//
// Pins `XabrSwaptionVolatilityCube<Model>::Cube` and
// `XabrSwaptionVolatilityCube<Model>::PrivateObserver`
// (ql/termstructures/volatility/swaption/sabrswaptionvolatilitycube.hpp @ v1.43).
//
// BOTH are PRIVATE nested classes, so a probe cannot name or construct them.
// Everything below therefore reaches them through the public surface of
// `SabrSwaptionVolatilityCube`, in two independent ways:
//
// (A) `cube_kernel` — the Cube's *interpolation composition*, rebuilt here out
//     of the very same PUBLIC building blocks the Cube constructor uses
//     (hpp:1006-1033): `transpose(points[k])`, then
//     `BilinearInterpolation` / `BackwardflatLinearInterpolation` over
//     (optionTimes as x, swapLengths as y), then `FlatExtrapolator2D` with
//     extrapolation enabled. Layer values are hand-chosen integers, so this
//     block is completely optimiser-free and pins the AXIS ORDER: feeding the
//     un-transposed layer, or swapping x/y, changes every number here.
//
// (B) the real private Cube, driven end-to-end through
//     `SabrSwaptionVolatilityCube`. The trick that makes this optimiser-free
//     is `isParameterFixed = {true,true,true,true}`: with every parameter
//     fixed, `XABRInterpolationImpl::calculate` short-circuits at
//     xabrinterpolation.hpp:162-169 — "there is nothing to optimize" — sets
//     `EndCriteria::None` and LEAVES THE PARAMETERS AT THE GUESS. And the
//     guess is `parametersGuess_(optionTimes[j], swapLengths[k])`
//     (hpp:448-449), i.e. a raw `Cube::operator()` evaluation. So:
//
//       sparseSabrParameters() layers 0..3 == parametersGuess_ Cube evaluated
//                                             at the SPARSE pillars
//       denseSabrParameters()  layers 0..3 == parametersGuess_ Cube evaluated
//                                             at the DENSE pillars, which
//                                             include points strictly BETWEEN
//                                             sparse pillars and points OUTSIDE
//                                             the sparse grid in both directions
//
//     That is exactly `Cube::operator()` at pillars, in the interior, and past
//     both ends of the grid — with zero optimiser in the path, hence TIGHT.
//     The dense grid also exists only because `fillVolatilityCube` drove
//     `Cube::setPoint` -> `Cube::expandLayers` (hpp:1132-1197), so the
//     `volCubeAtmCalibrated()` browse geometry is the expandLayers evidence.
//
// `maxErrorTolerance` is passed as 1.0 purely to keep the two QL_ENSURE guards
// at hpp:486-512 quiet: with fixed parameters the "fit" is whatever the guess
// says, and its rms error against the market slice is not meant to be small.
// It does not touch a single emitted number.
//
// Every Matrix is emitted as a nested JSON array in row-major order, at
// setprecision(17). `browse()` output is emitted VERBATIM — the full
// (nSwapLengths*nOptionTimes) x (nLayers+2) matrix, including the
// swapLength/optionTime columns 0 and 1 (hpp:1245-1256).
//
// Emits ONE JSON object on stdout and nothing else.

#include <ql/indexes/ibor/euribor.hpp>
#include <ql/indexes/swap/euriborswap.hpp>
#include <ql/math/interpolations/backwardflatlinearinterpolation.hpp>
#include <ql/math/interpolations/bilinearinterpolation.hpp>
#include <ql/math/interpolations/flatextrapolation2d.hpp>
#include <ql/math/matrix.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/swaption/sabrswaptionvolatilitycube.hpp>
#include <ql/termstructures/volatility/swaption/swaptionvolmatrix.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/utilities/dataformatters.hpp>
#include <ql/version.hpp>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace QuantLib;

namespace {

// --- tiny JSON emitters -----------------------------------------------------

std::ostream& out() { return std::cout; }

std::string periodToString(const Period& p) {
    std::ostringstream os;
    os << p;
    return os.str();
}

std::string isoDate(const Date& d) {
    std::ostringstream os;
    os << io::iso_date(d);
    return os.str();
}

void emitReal(Real x) { out() << std::setprecision(17) << x; }

void emitRealVector(const std::vector<Real>& v) {
    out() << "[";
    for (Size i = 0; i < v.size(); ++i) {
        if (i != 0)
            out() << ", ";
        emitReal(v[i]);
    }
    out() << "]";
}

void emitMatrix(const Matrix& m) {
    out() << "[";
    for (Size i = 0; i < m.rows(); ++i) {
        if (i != 0)
            out() << ", ";
        out() << "[";
        for (Size j = 0; j < m.columns(); ++j) {
            if (j != 0)
                out() << ", ";
            emitReal(m[i][j]);
        }
        out() << "]";
    }
    out() << "]";
}

void emitStringVector(const std::vector<std::string>& v) {
    out() << "[";
    for (Size i = 0; i < v.size(); ++i)
        out() << (i ? ", " : "") << "\"" << v[i] << "\"";
    out() << "]";
}

// --- fixture ----------------------------------------------------------------

// Evaluation date. Pinned; the Python test restores Settings afterwards.
const Date EVAL_DATE = Date(15, January, 2024);

// ATM surface grid. Deliberately a SUPERSET of the cube grid in BOTH
// directions: 6M sits below the cube's [1Y, 3Y] option span, 2Y strictly
// inside it; 3Y sits strictly inside the cube's [2Y, 5Y] swap span. So
// `fillVolatilityCube` is forced to drive `Cube::setPoint` -> `expandLayers`
// in the option direction, in the swap direction, and in both at once, and
// `Cube::operator()` gets sampled below AND between the sparse pillars.
//
// It CANNOT be extended past the top of the cube's option span: with only two
// sparse option pillars, `spreadVolInterpolation` asserts
// `optionTimesPreviousIndex+1 < sparseSmiles_.size()` (hpp:772-775), and for
// an ATM tenor above the last sparse pillar `lower_bound` returns end(), the
// index decrements to 1, and 1+1 >= 2 fires. Above-the-grid extrapolation is
// therefore pinned via `cube_kernel` and via the smile queries below instead.
//
// The ATM swap tenors start BELOW the cube's first swap pillar (1Y < 2Y) so
// that the very first `setPoint` of the fill loop — (6M, 1Y) — needs a new row
// AND a new column at once. Together with the later calls that need only one
// of the two, and the ones that need neither, all four
// (expandOptionTimes, expandSwapLengths) combinations get exercised.
const std::vector<Period> ATM_OPTION_TENORS = {Period(6, Months), Period(1, Years),
                                               Period(2, Years), Period(3, Years)};
const std::vector<Period> ATM_SWAP_TENORS = {Period(1, Years), Period(2, Years),
                                             Period(3, Years), Period(5, Years)};

const std::vector<Period> SWAP_TENORS = {Period(2, Years), Period(5, Years)};
const std::vector<Period> CUBE_OPTION_TENORS = {Period(1, Years), Period(3, Years)};

const std::vector<Real> STRIKE_SPREADS = {-0.02, -0.01, 0.0, 0.01, 0.02};

// ATM vols: deliberately non-constant so that `marketVolCube_`'s browse is a
// non-degenerate matrix.
Matrix atmVolMatrix() {
    Matrix m(ATM_OPTION_TENORS.size(), ATM_SWAP_TENORS.size(), 0.0);
    for (Size i = 0; i < m.rows(); ++i)
        for (Size j = 0; j < m.columns(); ++j)
            m[i][j] = 0.20 + 0.01 * Real(i) + 0.005 * Real(j);
    return m;
}

// Vol spreads, one row per (optionTenor, swapTenor) cell of the CUBE grid.
std::vector<std::vector<Real> > volSpreadValues() {
    std::vector<std::vector<Real> > v;
    for (Size j = 0; j < CUBE_OPTION_TENORS.size(); ++j) {
        for (Size k = 0; k < SWAP_TENORS.size(); ++k) {
            std::vector<Real> row;
            for (Size i = 0; i < STRIKE_SPREADS.size(); ++i)
                row.push_back(0.0100 - 0.0040 * Real(i) + 0.0007 * Real(j) -
                              0.0003 * Real(k));
            v.push_back(row);
        }
    }
    return v;
}

// Parameter guesses (alpha, beta, nu, rho), one 4-vector per cube cell.
// Every one of the four layers varies across BOTH grid directions, so a
// transposed or axis-swapped Cube cannot accidentally agree.
std::vector<std::vector<Real> > parameterGuessValues() {
    std::vector<std::vector<Real> > v;
    for (Size j = 0; j < CUBE_OPTION_TENORS.size(); ++j) {
        for (Size k = 0; k < SWAP_TENORS.size(); ++k) {
            std::vector<Real> row(4);
            row[0] = 0.030 + 0.004 * Real(j) + 0.002 * Real(k);   // alpha
            row[1] = 0.50 + 0.10 * Real(j) - 0.05 * Real(k);      // beta
            row[2] = 0.40 + 0.05 * Real(j) + 0.03 * Real(k);      // nu
            row[3] = -0.20 + 0.07 * Real(j) - 0.04 * Real(k);     // rho
            v.push_back(row);
        }
    }
    return v;
}

struct Fixture {
    Handle<YieldTermStructure> curve;
    ext::shared_ptr<SwapIndex> swapIndexBase;
    ext::shared_ptr<SwapIndex> shortSwapIndexBase;
    Handle<SwaptionVolatilityStructure> atmVol;
    std::vector<std::vector<Handle<Quote> > > volSpreads;
    std::vector<std::vector<Handle<Quote> > > parametersGuess;
    std::vector<ext::shared_ptr<SimpleQuote> > guessQuotes;  // flat, for mutation
};

Fixture makeFixture() {
    Fixture f;

    f.curve = Handle<YieldTermStructure>(ext::make_shared<FlatForward>(
        EVAL_DATE, 0.03, Actual365Fixed(), Continuous, Annual));

    // 5Y is the "long" index, 1Y the "short" one; with SWAP_TENORS = {2Y, 5Y}
    // and shortSwapIndexBase tenor 1Y, BOTH tenors route to swapIndexBase_
    // (2Y > 1Y). The Python fake index therefore only needs one table.
    f.swapIndexBase = ext::make_shared<EuriborSwapIsdaFixA>(Period(5, Years), f.curve);
    f.shortSwapIndexBase = ext::make_shared<EuriborSwapIsdaFixA>(Period(1, Years), f.curve);

    f.atmVol = Handle<SwaptionVolatilityStructure>(ext::make_shared<SwaptionVolatilityMatrix>(
        EVAL_DATE, TARGET(), Following, ATM_OPTION_TENORS, ATM_SWAP_TENORS,
        atmVolMatrix(), Actual365Fixed()));

    const std::vector<std::vector<Real> > vs = volSpreadValues();
    for (const auto& row : vs) {
        std::vector<Handle<Quote> > hrow;
        for (Real x : row)
            hrow.emplace_back(ext::make_shared<SimpleQuote>(x));
        f.volSpreads.push_back(hrow);
    }

    const std::vector<std::vector<Real> > pg = parameterGuessValues();
    for (const auto& row : pg) {
        std::vector<Handle<Quote> > hrow;
        for (Real x : row) {
            auto q = ext::make_shared<SimpleQuote>(x);
            f.guessQuotes.push_back(q);
            hrow.emplace_back(q);
        }
        f.parametersGuess.push_back(hrow);
    }
    return f;
}

ext::shared_ptr<SabrSwaptionVolatilityCube> makeCube(const Fixture& f,
                                                     bool isAtmCalibrated,
                                                     bool backwardFlat) {
    return ext::make_shared<SabrSwaptionVolatilityCube>(
        f.atmVol, CUBE_OPTION_TENORS, SWAP_TENORS, STRIKE_SPREADS, f.volSpreads,
        f.swapIndexBase, f.shortSwapIndexBase,
        /*vegaWeightedSmileFit=*/false, f.parametersGuess,
        std::vector<bool>(4, true),  // EVERY parameter fixed -> no optimiser
        isAtmCalibrated,
        /*endCriteria=*/ext::shared_ptr<EndCriteria>(),
        /*maxErrorTolerance=*/1.0,
        /*optMethod=*/ext::shared_ptr<OptimizationMethod>(),
        /*errorAccept=*/Null<Real>(),
        /*useMaxError=*/false,
        /*maxGuesses=*/50, backwardFlat,
        /*cutoffStrike=*/0.0001);
}

// --- (A) the Cube interpolation composition, from public parts --------------

// Layer laid out exactly as `Cube::points_[k]`: rows index optionTimes,
// columns index swapLengths (hpp:1006-1007, 1105).
Matrix kernelLayer() {
    Matrix m(3, 4, 0.0);
    Real v = 1.0;
    for (Size i = 0; i < 3; ++i)
        for (Size j = 0; j < 4; ++j)
            m[i][j] = v++;  // 1..12, so a transpose is instantly visible
    return m;
}

const std::vector<Real> KERNEL_OPTION_TIMES = {0.5, 1.0, 3.0};
const std::vector<Real> KERNEL_SWAP_LENGTHS = {1.0, 2.0, 5.0, 10.0};

// (optionTime, swapLength) query points: pillars, strictly-between, and
// outside the grid past both ends of both axes.
const std::vector<std::pair<Real, Real> > KERNEL_QUERIES = {
    {0.5, 1.0},    {1.0, 2.0},   {3.0, 10.0},   // pillars
    {0.75, 1.5},   {2.0, 3.5},   {1.5, 7.0},    // strictly between
    {0.5, 3.0},    {2.0, 5.0},                  // on one axis' pillar only
    {0.1, 2.0},    {5.0, 2.0},                  // outside in optionTime
    {1.0, 0.25},   {1.0, 20.0},                 // outside in swapLength
    {0.1, 0.25},   {5.0, 20.0}                  // outside in both
};

void emitCubeKernel() {
    const Matrix layer = kernelLayer();
    // C++ parity: sabrswaptionvolatilitycube.hpp:1010 / 1225 — the
    // Interpolation2D is built over the TRANSPOSED layer.
    const Matrix transposed = transpose(layer);

    std::vector<Real> bilinear, backwardflat;
    for (int mode = 0; mode < 2; ++mode) {
        ext::shared_ptr<Interpolation2D> inner;
        if (mode == 0)
            inner = ext::make_shared<BilinearInterpolation>(
                KERNEL_OPTION_TIMES.begin(), KERNEL_OPTION_TIMES.end(),
                KERNEL_SWAP_LENGTHS.begin(), KERNEL_SWAP_LENGTHS.end(), transposed);
        else
            inner = ext::make_shared<BackwardflatLinearInterpolation>(
                KERNEL_OPTION_TIMES.begin(), KERNEL_OPTION_TIMES.end(),
                KERNEL_SWAP_LENGTHS.begin(), KERNEL_SWAP_LENGTHS.end(), transposed);
        // C++ parity: hpp:1030-1032.
        ext::shared_ptr<Interpolation2D> flat(new FlatExtrapolator2D(inner));
        flat->enableExtrapolation();
        for (const auto& q : KERNEL_QUERIES) {
            const Real v = (*flat)(q.first, q.second);
            if (mode == 0)
                bilinear.push_back(v);
            else
                backwardflat.push_back(v);
        }
    }

    out() << "  \"cube_kernel\": {\n";
    out() << "    \"layer\": ";
    emitMatrix(layer);
    out() << ",\n";
    out() << "    \"transposed\": ";
    emitMatrix(transposed);
    out() << ",\n";
    out() << "    \"option_times\": ";
    emitRealVector(KERNEL_OPTION_TIMES);
    out() << ",\n";
    out() << "    \"swap_lengths\": ";
    emitRealVector(KERNEL_SWAP_LENGTHS);
    out() << ",\n";
    out() << "    \"queries\": [";
    for (Size i = 0; i < KERNEL_QUERIES.size(); ++i) {
        if (i)
            out() << ", ";
        out() << "[";
        emitReal(KERNEL_QUERIES[i].first);
        out() << ", ";
        emitReal(KERNEL_QUERIES[i].second);
        out() << "]";
    }
    out() << "],\n";
    out() << "    \"bilinear\": ";
    emitRealVector(bilinear);
    out() << ",\n";
    out() << "    \"backwardflat\": ";
    emitRealVector(backwardflat);
    out() << "\n  },\n";
}

// --- setup block ------------------------------------------------------------

void emitSetup(const Fixture& f) {
    const ext::shared_ptr<SabrSwaptionVolatilityCube> cube = makeCube(f, false, false);

    out() << "  \"setup\": {\n";
    out() << "    \"eval_date_serial\": " << EVAL_DATE.serialNumber() << ",\n";
    out() << "    \"eval_date_iso\": \"" << isoDate(EVAL_DATE) << "\",\n";

    std::vector<std::string> atmTenors;
    for (const auto& p : ATM_OPTION_TENORS)
        atmTenors.push_back(periodToString(p));
    out() << "    \"atm_option_tenors\": ";
    emitStringVector(atmTenors);
    out() << ",\n";

    std::vector<std::string> cubeTenors;
    for (const auto& p : CUBE_OPTION_TENORS)
        cubeTenors.push_back(periodToString(p));
    out() << "    \"cube_option_tenors\": ";
    emitStringVector(cubeTenors);
    out() << ",\n";

    std::vector<std::string> swapTenors;
    for (const auto& p : SWAP_TENORS)
        swapTenors.push_back(periodToString(p));
    out() << "    \"swap_tenors\": ";
    emitStringVector(swapTenors);
    out() << ",\n";

    std::vector<std::string> atmSwapTenors;
    for (const auto& p : ATM_SWAP_TENORS)
        atmSwapTenors.push_back(periodToString(p));
    out() << "    \"atm_swap_tenors\": ";
    emitStringVector(atmSwapTenors);
    out() << ",\n";

    out() << "    \"strike_spreads\": ";
    emitRealVector(STRIKE_SPREADS);
    out() << ",\n";
    out() << "    \"atm_vols\": ";
    emitMatrix(atmVolMatrix());
    out() << ",\n";

    const std::vector<std::vector<Real> > vs = volSpreadValues();
    out() << "    \"vol_spreads\": [";
    for (Size i = 0; i < vs.size(); ++i) {
        if (i)
            out() << ", ";
        emitRealVector(vs[i]);
    }
    out() << "],\n";

    const std::vector<std::vector<Real> > pg = parameterGuessValues();
    out() << "    \"parameters_guess\": [";
    for (Size i = 0; i < pg.size(); ++i) {
        if (i)
            out() << ", ";
        emitRealVector(pg[i]);
    }
    out() << "],\n";

    out() << "    \"cube_option_times\": ";
    emitRealVector(cube->optionTimes());
    out() << ",\n";
    out() << "    \"cube_swap_lengths\": ";
    emitRealVector(cube->swapLengths());
    out() << ",\n";

    std::vector<Real> cubeDates;
    for (const auto& d : cube->optionDates())
        cubeDates.push_back(Real(d.serialNumber()));
    out() << "    \"cube_option_date_serials\": ";
    emitRealVector(cubeDates);
    out() << ",\n";

    // The ATM surface's own pillars: the dense grid is the sorted union of
    // these with the cube's (hpp:651-681).
    const ext::shared_ptr<SwaptionVolatilityDiscrete> atmDiscrete =
        ext::dynamic_pointer_cast<SwaptionVolatilityDiscrete>(*f.atmVol);
    out() << "    \"atm_option_times\": ";
    emitRealVector(atmDiscrete->optionTimes());
    out() << ",\n";
    std::vector<Real> atmDates;
    for (const auto& d : atmDiscrete->optionDates())
        atmDates.push_back(Real(d.serialNumber()));
    out() << "    \"atm_option_date_serials\": ";
    emitRealVector(atmDates);
    out() << ",\n";
    out() << "    \"atm_swap_lengths\": ";
    emitRealVector(atmDiscrete->swapLengths());
    out() << ",\n";

    // (optionDate, swapTenor) -> atmStrike, over the FULL union grid. The
    // Python fake swap index is driven straight off this table, so the
    // C++ `SwaptionVolatilityCube::atmStrike` swap-rebuild (swaptionvolcube.cpp:89-144)
    // never has to be reproduced on the Python side.
    std::vector<Date> unionDates = atmDiscrete->optionDates();
    for (const auto& d : cube->optionDates())
        unionDates.push_back(d);
    std::sort(unionDates.begin(), unionDates.end());
    unionDates.erase(std::unique(unionDates.begin(), unionDates.end()), unionDates.end());

    out() << "    \"atm_strike_table\": [";
    bool first = true;
    for (const auto& d : unionDates) {
        for (const auto& t : ATM_SWAP_TENORS) {
            if (!first)
                out() << ", ";
            first = false;
            out() << "{\"date_serial\": " << d.serialNumber() << ", \"swap_tenor\": \""
                  << periodToString(t) << "\", \"forward\": ";
            emitReal(cube->atmStrike(d, t));
            out() << "}";
        }
    }
    out() << "]\n";
    out() << "  },\n";
}

// --- (B) the real private Cube, through the public surface ------------------

// (optionTime, swapLength) sample points for the smile/volatility probes.
// 1.0/3.0 and 2.0/5.0 straddle the sparse pillars; the rest are interior or
// outside.
std::vector<std::pair<Real, Real> > smileQueries(
    const ext::shared_ptr<SabrSwaptionVolatilityCube>& cube) {
    const std::vector<Time>& t = cube->optionTimes();
    const std::vector<Time>& s = cube->swapLengths();
    return {
        {t[0], s[0]},                                        // pillar (0,0)
        {t[1], s[1]},                                        // pillar (1,1)
        {0.5 * (t[0] + t[1]), 0.5 * (s[0] + s[1])},          // dead centre
        {0.25 * t[0] + 0.75 * t[1], 0.8 * s[0] + 0.2 * s[1]},// interior, off-centre
        {t[0], 0.5 * (s[0] + s[1])},                         // on an option pillar
        {0.5 * (t[0] + t[1]), s[1]},                         // on a swap pillar
        {0.5 * t[0], s[0]},                                  // below in optionTime
        {t[1] + 3.0, s[1]},                                  // above in optionTime
        {t[0], 0.5 * s[0]},                                  // below in swapLength
        {t[0], s[1] + 5.0}                                   // above in swapLength
    };
}

void emitCubeBlock(const std::string& name, const Fixture& f, bool isAtmCalibrated,
                   bool backwardFlat) {
    const ext::shared_ptr<SabrSwaptionVolatilityCube> cube =
        makeCube(f, isAtmCalibrated, backwardFlat);

    out() << "  \"" << name << "\": {\n";
    out() << "    \"is_atm_calibrated\": " << (isAtmCalibrated ? "true" : "false")
          << ",\n";
    out() << "    \"backward_flat\": " << (backwardFlat ? "true" : "false") << ",\n";

    // Cube::browse of marketVolCube_ — layers are pure quote arithmetic
    // (atmVol + volSpread, hpp:375-385): no optimiser, no atmStrike.
    out() << "    \"market_vol_cube_browse\": ";
    emitMatrix(cube->marketVolCube());
    out() << ",\n";

    // Cube::points()[i] straight through the public accessor (hpp:215-217).
    out() << "    \"market_vol_cube_points\": [";
    for (Size i = 0; i < STRIKE_SPREADS.size(); ++i) {
        if (i)
            out() << ", ";
        emitMatrix(cube->marketVolCube(i));
    }
    out() << "],\n";

    // sparseParameters_.browse(): layers 0..3 are parametersGuess_(t, s) at
    // the sparse pillars; layer 4 is the atmStrike forward; 5/6 are the fit
    // residuals; 7 is EndCriteria::None == 0.
    out() << "    \"sparse_browse\": ";
    emitMatrix(cube->sparseSabrParameters());
    out() << ",\n";

    if (isAtmCalibrated) {
        // volCubeAtmCalibrated_ after fillVolatilityCube: this is the
        // Cube::setPoint / Cube::expandLayers evidence (hpp:646-716).
        out() << "    \"vol_cube_atm_calibrated_browse\": ";
        emitMatrix(cube->volCubeAtmCalibrated());
        out() << ",\n";
        // denseParameters_.browse(): layers 0..3 are parametersGuess_(t, s)
        // evaluated at the EXPANDED grid — the interior and the outside.
        out() << "    \"dense_browse\": ";
        emitMatrix(cube->denseSabrParameters());
        out() << ",\n";
    }

    // smileSection(t, s) -> the Cube::operator() consumer (hpp:860-884).
    const std::vector<std::pair<Real, Real> > qs = smileQueries(cube);
    out() << "    \"smile_queries\": [";
    for (Size i = 0; i < qs.size(); ++i) {
        if (i)
            out() << ", ";
        out() << "[";
        emitReal(qs[i].first);
        out() << ", ";
        emitReal(qs[i].second);
        out() << "]";
    }
    out() << "],\n";

    std::vector<Real> atmLevels, volsAtm, volsLow, volsHigh;
    for (const auto& q : qs) {
        const ext::shared_ptr<SmileSection> sec =
            cube->smileSection(q.first, q.second, /*extrapolate=*/true);
        // atmLevel() is SabrSmileSection's forward_ = allParameters[nParams],
        // i.e. Cube::operator() on the forwards layer (hpp:868).
        atmLevels.push_back(sec->atmLevel());
        volsAtm.push_back(sec->volatility(sec->atmLevel()));
        volsLow.push_back(sec->volatility(sec->atmLevel() - 0.01));
        volsHigh.push_back(sec->volatility(sec->atmLevel() + 0.01));
    }
    out() << "    \"smile_atm_level\": ";
    emitRealVector(atmLevels);
    out() << ",\n";
    out() << "    \"smile_vol_atm\": ";
    emitRealVector(volsAtm);
    out() << ",\n";
    out() << "    \"smile_vol_atm_minus_100bp\": ";
    emitRealVector(volsLow);
    out() << ",\n";
    out() << "    \"smile_vol_atm_plus_100bp\": ";
    emitRealVector(volsHigh);
    out() << "\n";
    out() << "  },\n";
}

// --- PrivateObserver --------------------------------------------------------

void emitObserver() {
    // A dedicated fixture: mutating a guess quote must not leak into the
    // other blocks.
    Fixture f = makeFixture();
    const ext::shared_ptr<SabrSwaptionVolatilityCube> cube = makeCube(f, false, false);
    const std::vector<Time>& t = cube->optionTimes();
    const std::vector<Time>& s = cube->swapLengths();
    const Real qt = 0.5 * (t[0] + t[1]);
    const Real qs = 0.5 * (s[0] + s[1]);

    const ext::shared_ptr<SmileSection> before =
        cube->smileSection(qt, qs, /*extrapolate=*/true);
    const Real strike = before->atmLevel();
    const Real volBefore = before->volatility(strike);
    const Real atmBefore = before->atmLevel();

    // parametersGuessQuotes_[j*nSwapTenors + k][i]; here cell (j=0, k=0),
    // i=0 == alpha. The PrivateObserver registered at hpp:341-347 must fire
    // setParameterGuess() (hpp:349-364) and then update().
    const Size idx = (0 * SWAP_TENORS.size() + 0) * 4 + 0;
    const Real alphaOld = f.guessQuotes[idx]->value();
    const Real alphaNew = alphaOld + 0.010;
    f.guessQuotes[idx]->setValue(alphaNew);

    const ext::shared_ptr<SmileSection> after =
        cube->smileSection(qt, qs, /*extrapolate=*/true);
    const Real volAfter = after->volatility(strike);
    const Real atmAfter = after->atmLevel();

    out() << "  \"observer\": {\n";
    out() << "    \"query_option_time\": ";
    emitReal(qt);
    out() << ",\n";
    out() << "    \"query_swap_length\": ";
    emitReal(qs);
    out() << ",\n";
    out() << "    \"strike\": ";
    emitReal(strike);
    out() << ",\n";
    out() << "    \"alpha_old\": ";
    emitReal(alphaOld);
    out() << ",\n";
    out() << "    \"alpha_new\": ";
    emitReal(alphaNew);
    out() << ",\n";
    out() << "    \"vol_before\": ";
    emitReal(volBefore);
    out() << ",\n";
    out() << "    \"vol_after\": ";
    emitReal(volAfter);
    out() << ",\n";
    out() << "    \"atm_level_before\": ";
    emitReal(atmBefore);
    out() << ",\n";
    out() << "    \"atm_level_after\": ";
    emitReal(atmAfter);
    out() << ",\n";
    out() << "    \"sparse_browse_after\": ";
    emitMatrix(cube->sparseSabrParameters());
    out() << "\n";
    out() << "  },\n";
}

// --- Cube::setPoint at an EXISTING pillar -----------------------------------

void emitRecalibration() {
    Fixture f = makeFixture();
    const ext::shared_ptr<SabrSwaptionVolatilityCube> cube = makeCube(f, false, false);

    const Matrix beforeM = cube->sparseSabrParameters();

    // recalibration(beta, swapTenor) writes parametersGuess_ via
    // Cube::setElement, then drives sabrCalibrationSection, which calls
    // Cube::setPoint at pillars that ARE present -> no expandLayers
    // (hpp:906-945, 638-641).
    const std::vector<Real> betas = {0.65, 0.35};  // one per option tenor
    cube->recalibration(betas, Period(5, Years));

    const Matrix afterM = cube->sparseSabrParameters();

    out() << "  \"recalibration\": {\n";
    out() << "    \"betas\": ";
    emitRealVector(betas);
    out() << ",\n";
    out() << "    \"swap_tenor\": \"" << periodToString(Period(5, Years)) << "\",\n";
    out() << "    \"sparse_browse_before\": ";
    emitMatrix(beforeM);
    out() << ",\n";
    out() << "    \"sparse_browse_after\": ";
    emitMatrix(afterM);
    out() << "\n";
    out() << "  }\n";
}

}  // namespace

int main() {
    Settings::instance().evaluationDate() = EVAL_DATE;

    out() << "{\n";
    out() << "  \"meta\": {\n";
    out() << "    \"quantlib_version\": \"" << QL_VERSION << "\",\n";
    out() << "    \"header\": "
             "\"ql/termstructures/volatility/swaption/sabrswaptionvolatilitycube.hpp\",\n";
    out() << "    \"classes\": [\"XabrSwaptionVolatilityCube<Model>::Cube\", "
             "\"XabrSwaptionVolatilityCube<Model>::PrivateObserver\"]\n";
    out() << "  },\n";

    emitCubeKernel();

    Fixture f = makeFixture();
    emitSetup(f);
    emitCubeBlock("cube_bilinear_sparse", f, /*isAtmCalibrated=*/false,
                  /*backwardFlat=*/false);
    emitCubeBlock("cube_backwardflat_sparse", f, /*isAtmCalibrated=*/false,
                  /*backwardFlat=*/true);
    emitCubeBlock("cube_bilinear_dense", f, /*isAtmCalibrated=*/true,
                  /*backwardFlat=*/false);
    emitCubeBlock("cube_backwardflat_dense", f, /*isAtmCalibrated=*/true,
                  /*backwardFlat=*/true);
    emitObserver();
    emitRecalibration();

    out() << "}\n";
    return 0;
}

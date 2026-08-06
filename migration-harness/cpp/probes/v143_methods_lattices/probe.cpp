// migration-harness/cpp/probes/v143_methods_lattices/probe.cpp
//
// Reference values for the ql/methods/lattices cluster at C++ QuantLib v1.43
// (pinned submodule @ 6b57206e0).  Emits references/v143/methods/lattices.json.
//
// What is pinned here, and why
// ----------------------------
//
// 1. The binomial family (ql/methods/lattices/binomialtree.{hpp,cpp}).
//    Every concrete tree is driven off ONE constant-coefficient
//    GeneralizedBlackScholesProcess so that the differences between the
//    reference blocks are attributable to the tree parameterisation alone:
//
//      * EqualProbabilitiesBinomialTree  -> JarrowRudd, AdditiveEQPBinomialTree
//        (underlying = x0*exp(i*driftPerStep + j*up_), probability == 0.5)
//      * EqualJumpsBinomialTree          -> CoxRossRubinstein, Trigeorgis
//        (underlying = x0*exp(j*dx_), probability = branch==1 ? pu_ : pd_)
//      * BinomialTree (direct)           -> Tian, LeisenReimer, Joshi4
//        (underlying = x0*down^(i-index)*up^index)
//
//    The two intermediate bases have no constructor of their own beyond the
//    BinomialTree one, so they are pinned through their concretes: the
//    "underlying" arrays over a whole slice are exactly the base-class
//    formulas, and `probability` sampled at several (i, index) pins the
//    base-class index-independence.
//
//    Joshi4 and LeisenReimer both round `steps` up to the next odd number, so
//    each is emitted at an even and at an odd request to pin that rounding.
//    Joshi4's up/down come from its own quartic `computeUpProb` expansion,
//    which is what distinguishes it from LeisenReimer's Peizer-Pratt
//    inversion; the two blocks share a fixture so a port that wires the wrong
//    inversion into Joshi4 fails loudly.
//
// 2. TreeLattice (ql/methods/lattices/lattice.hpp) — the CRTP base holding
//    initialize / rollback / partialRollback / presentValue / statePrices /
//    stepback.  Pinned twice, through both of its lattice1d/lattice2d
//    descendants:
//
//      * BlackScholesLattice (lattice1d + bsmlattice) exercises
//        TreeLattice1D::grid plus the *overridden* constant-rate stepback.
//      * TwoFactorModel::ShortRateTree (lattice2d) exercises the *generic*
//        TreeLattice::stepback and the Arrow-Debreu statePrices recursion
//        over a 9-branch lattice.
//
//    A DiscretizedDiscountBond is rolled back on each: full rollback pins
//    presentValue against the state-price dot product, and a partialRollback
//    to an interior grid time pins the per-slice values array (i.e. stepback
//    itself, not just its endpoint).
//
// 3. TreeLattice2D (ql/methods/lattices/lattice2d.hpp) — size / descendant /
//    probability plus the sign-dependent m_ matrix.  BOTH branches of the
//    `correlation < 0.0 && T::branches == 3` test are emitted (rho = +0.5 and
//    rho = -0.5), because the two m_ layouts are transposes of each other
//    about the anti-diagonal and a port that hard-codes one of them agrees
//    with C++ on exactly half of the nodes.  rho_ is fabs(correlation), so
//    the two blocks differ ONLY in m_.
//
// Design notes
// ------------
//
// TreeLattice2D is abstract (discount() comes from Impl), so it is driven
// through the one concrete instantiation in the library,
// TwoFactorModel::ShortRateTree, with a probe-local ShortRateDynamics whose
// shortRate(t, x, y) = x + y + 0.03.  Using a closed-form dynamics rather
// than G2's keeps the reference free of any yield-curve dependence: every
// number below is a function of two OrnsteinUhlenbeckProcess trees and
// elementary arithmetic.
//
// All floating-point output is std::setprecision(17) — round-trip exact for
// IEEE-754 doubles.

#include <ql/discretizedasset.hpp>
#include <ql/methods/lattices/binomialtree.hpp>
#include <ql/methods/lattices/bsmlattice.hpp>
#include <ql/methods/lattices/lattice.hpp>
#include <ql/methods/lattices/lattice1d.hpp>
#include <ql/methods/lattices/lattice2d.hpp>
#include <ql/methods/lattices/trinomialtree.hpp>
#include <ql/models/shortrate/twofactormodel.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/processes/ornsteinuhlenbeckprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/timegrid.hpp>

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

void emit_int(const std::string& name, long v) {
    sep();
    std::cout << "  \"" << name << "\": " << v;
}

void emit_arr(const std::string& name, const std::vector<Real>& v) {
    sep();
    std::cout << "  \"" << name << "\": [";
    for (Size i = 0; i < v.size(); ++i) {
        if (i != 0U) std::cout << ", ";
        std::cout << std::setprecision(17) << v[i];
    }
    std::cout << "]";
}

void emit_iarr(const std::string& name, const std::vector<long>& v) {
    sep();
    std::cout << "  \"" << name << "\": [";
    for (Size i = 0; i < v.size(); ++i) {
        if (i != 0U) std::cout << ", ";
        std::cout << v[i];
    }
    std::cout << "]";
}

std::vector<Real> to_vector(const Array& a) {
    return {a.begin(), a.end()};
}

// --------------------------------------------------------------------------
// Block 1 — the binomial family
// --------------------------------------------------------------------------

ext::shared_ptr<GeneralizedBlackScholesProcess> makeBsmProcess(const Date& evalDate) {
    // Same fixture as the pre-existing cluster/l5b probe: s0 = 100, r = 5%,
    // q = 0, sigma = 20%, Actual360, TARGET.  Constant coefficients, so the
    // process is strike-independent and every quantity used by the trees
    // (x0, drift, variance, stdDeviation) is closed form.
    Handle<Quote> spot(ext::make_shared<SimpleQuote>(100.0));
    Handle<YieldTermStructure> rTS(
        ext::make_shared<FlatForward>(evalDate, 0.05, Actual360(), Continuous, Annual));
    Handle<YieldTermStructure> qTS(
        ext::make_shared<FlatForward>(evalDate, 0.00, Actual360(), Continuous, Annual));
    Handle<BlackVolTermStructure> volTS(
        ext::make_shared<BlackConstantVol>(evalDate, TARGET(), 0.20, Actual360()));
    return ext::make_shared<GeneralizedBlackScholesProcess>(spot, qTS, rTS, volTS);
}

// Emit the full public surface of one binomial tree.  `slices` are the time
// slices whose whole `underlying` row is captured; the terminal slice is
// always included by the caller.
template <class Tree>
void emitBinomial(const std::string& prefix,
                  const Tree& tree,
                  Size columns,
                  const std::vector<Size>& slices) {
    emit_int(prefix + "_columns", static_cast<long>(columns));
    emit(prefix + "_pd", tree.probability(0, 0, 0));
    emit(prefix + "_pu", tree.probability(0, 0, 1));
    // Base-class contract: probability does not depend on (i, index).  Pinned
    // at an interior node so a port that made it node-dependent shows up.
    emit(prefix + "_pd_interior", tree.probability(2, 1, 0));
    emit(prefix + "_pu_interior", tree.probability(2, 1, 1));

    std::vector<long> sizes;
    for (Size i = 0; i < columns; ++i)
        sizes.push_back(static_cast<long>(tree.size(i)));
    emit_iarr(prefix + "_sizes", sizes);

    for (Size i : slices) {
        std::vector<Real> row;
        for (Size j = 0; j < tree.size(i); ++j)
            row.push_back(tree.underlying(i, j));
        emit_arr(prefix + "_underlying_" + std::to_string(i), row);
    }

    // descendant(i, index, branch) == index + branch for every binomial tree.
    std::vector<long> desc;
    for (Size branch = 0; branch < 2; ++branch)
        for (Size index = 0; index < 3; ++index)
            desc.push_back(static_cast<long>(tree.descendant(2, index, branch)));
    emit_iarr(prefix + "_descendant_i2", desc);
}

void block_binomial(const Date& evalDate) {
    auto process = makeBsmProcess(evalDate);
    const Time end = 1.0;
    const Size steps = 4;
    const Real strike = 100.0;

    // --- EqualProbabilitiesBinomialTree concretes -------------------------
    {
        JarrowRudd t(process, end, steps, strike);
        emitBinomial("jarrow_rudd", t, steps + 1, {1, 2, 4});
    }
    {
        AdditiveEQPBinomialTree t(process, end, steps, strike);
        emitBinomial("additive_eqp", t, steps + 1, {1, 2, 4});
    }

    // --- EqualJumpsBinomialTree concretes ---------------------------------
    {
        CoxRossRubinstein t(process, end, steps, strike);
        emitBinomial("crr", t, steps + 1, {1, 2, 4});
    }
    {
        Trigeorgis t(process, end, steps, strike);
        emitBinomial("trigeorgis", t, steps + 1, {1, 2, 4});
    }

    // --- direct BinomialTree concretes ------------------------------------
    {
        Tian t(process, end, steps, strike);
        emitBinomial("tian", t, steps + 1, {1, 2, 4});
    }
    {
        // steps = 4 is even -> forced to 5.
        LeisenReimer t(process, end, steps, strike);
        emitBinomial("leisen_reimer", t, 6, {1, 2, 5});
    }
    {
        // steps = 4 is even -> forced to 5.
        Joshi4 t(process, end, steps, strike);
        emitBinomial("joshi4", t, 6, {1, 2, 5});
    }
    {
        // steps = 7 is already odd -> kept.  Pins the "no rounding" arm and
        // gives Joshi4's k = (oddSteps-1)/2 expansion a second data point.
        Joshi4 t(process, end, 7, strike);
        emitBinomial("joshi4_odd", t, 8, {1, 7});
    }
    {
        LeisenReimer t(process, end, 7, strike);
        emitBinomial("leisen_reimer_odd", t, 8, {1, 7});
    }

    // Strike sensitivity: Joshi4/LeisenReimer are the only trees whose
    // coefficients depend on the strike.  A second strike pins that wiring.
    {
        Joshi4 t(process, end, steps, 120.0);
        emit("joshi4_k120_pu", t.probability(0, 0, 1));
        std::vector<Real> row;
        for (Size j = 0; j < t.size(5); ++j)
            row.push_back(t.underlying(5, j));
        emit_arr("joshi4_k120_underlying_5", row);
    }
}

// --------------------------------------------------------------------------
// Block 2 — BlackScholesLattice: TreeLattice + TreeLattice1D through
//           the 1-D descendant with the overridden stepback.
// --------------------------------------------------------------------------

void block_bsm_lattice(const Date& evalDate) {
    auto process = makeBsmProcess(evalDate);
    const Time end = 1.0;
    const Size steps = 10;
    const Rate riskFreeRate = 0.05;

    auto tree = ext::make_shared<CoxRossRubinstein>(process, end, steps, 100.0);
    auto lattice =
        ext::make_shared<BlackScholesLattice<CoxRossRubinstein>>(tree, riskFreeRate, end, steps);

    emit("bsml_dt", lattice->dt());
    emit("bsml_risk_free_rate", lattice->riskFreeRate());
    emit("bsml_discount", lattice->discount(0, 0));
    emit_arr("bsml_grid_0", to_vector(lattice->grid(0.0)));
    emit_arr("bsml_grid_half", to_vector(lattice->grid(0.5)));
    emit_arr("bsml_grid_terminal", to_vector(lattice->grid(end)));

    // Arrow-Debreu state prices: TreeLattice::computeStatePrices.
    for (Size i = 0; i <= steps; ++i)
        emit_arr("bsml_state_prices_" + std::to_string(i), to_vector(lattice->statePrices(i)));

    // Full rollback -> presentValue (state-price dot product).
    {
        DiscretizedDiscountBond bond;
        bond.initialize(lattice, end);
        bond.rollback(0.0);
        emit("bsml_bond_pv", bond.presentValue());
        emit_arr("bsml_bond_values_0", to_vector(bond.values()));
    }
    // Partial rollback to an interior grid time -> the per-slice values array,
    // which pins BlackScholesLattice::stepback itself rather than a scalar
    // summary of it.
    {
        DiscretizedDiscountBond bond;
        bond.initialize(lattice, end);
        bond.partialRollback(0.5);
        emit("bsml_bond_partial_time", bond.time());
        emit_arr("bsml_bond_partial_values", to_vector(bond.values()));
        emit("bsml_bond_partial_pv", bond.presentValue());
    }
}

// --------------------------------------------------------------------------
// Block 3 — TreeLattice2D through TwoFactorModel::ShortRateTree
// --------------------------------------------------------------------------

class ProbeDynamics : public TwoFactorModel::ShortRateDynamics {
  public:
    ProbeDynamics(const ext::shared_ptr<StochasticProcess1D>& x,
                  const ext::shared_ptr<StochasticProcess1D>& y,
                  Real rho)
    : TwoFactorModel::ShortRateDynamics(x, y, rho) {}

    Rate shortRate(Time, Real x, Real y) const override { return x + y + 0.03; }
};

void emitLattice2D(const std::string& prefix, Real correlation) {
    auto p1 = ext::make_shared<OrnsteinUhlenbeckProcess>(0.1, 0.01);
    auto p2 = ext::make_shared<OrnsteinUhlenbeckProcess>(0.3, 0.02);
    TimeGrid grid(2.0, 5);
    auto tree1 = ext::make_shared<TrinomialTree>(p1, grid);
    auto tree2 = ext::make_shared<TrinomialTree>(p2, grid);
    auto dynamics = ext::make_shared<ProbeDynamics>(p1, p2, correlation);
    auto lattice = ext::make_shared<TwoFactorModel::ShortRateTree>(tree1, tree2, dynamics);

    // size(i) = tree1->size(i) * tree2->size(i)
    std::vector<long> sizes;
    for (Size i = 0; i < grid.size(); ++i)
        sizes.push_back(static_cast<long>(lattice->size(i)));
    emit_iarr(prefix + "_sizes", sizes);

    // descendant + probability over whole slices.  Slice 0 has one node;
    // slice 2 has 25, enough for the modulo/division index split to matter.
    for (Size i : {Size(0), Size(2)}) {
        std::vector<long> desc;
        std::vector<Real> prob;
        for (Size index = 0; index < lattice->size(i); ++index) {
            for (Size branch = 0; branch < 9; ++branch) {
                desc.push_back(static_cast<long>(lattice->descendant(i, index, branch)));
                prob.push_back(lattice->probability(i, index, branch));
            }
        }
        emit_iarr(prefix + "_descendant_" + std::to_string(i), desc);
        emit_arr(prefix + "_probability_" + std::to_string(i), prob);
    }

    // discount(i, index) -- ShortRateTree's own, mirrored by the consuming
    // test's concrete subclass.
    {
        std::vector<Real> disc;
        for (Size index = 0; index < lattice->size(2); ++index)
            disc.push_back(lattice->discount(2, index));
        emit_arr(prefix + "_discount_2", disc);
    }

    // Arrow-Debreu state prices over the 9-branch lattice.
    for (Size i = 0; i < grid.size(); ++i)
        emit_arr(prefix + "_state_prices_" + std::to_string(i),
                 to_vector(lattice->statePrices(i)));

    // Generic TreeLattice::stepback, via a discount bond.
    {
        DiscretizedDiscountBond bond;
        bond.initialize(lattice, 2.0);
        bond.rollback(0.0);
        emit(prefix + "_bond_pv", bond.presentValue());
        emit_arr(prefix + "_bond_values_0", to_vector(bond.values()));
    }
    {
        DiscretizedDiscountBond bond;
        bond.initialize(lattice, 2.0);
        bond.partialRollback(0.8);
        emit(prefix + "_bond_partial_time", bond.time());
        emit_arr(prefix + "_bond_partial_values", to_vector(bond.values()));
        emit(prefix + "_bond_partial_pv", bond.presentValue());
    }
}

void block_lattice2d() {
    // Both arms of `correlation < 0.0 && T::branches == 3`.
    emitLattice2D("l2d_pos", 0.5);
    emitLattice2D("l2d_neg", -0.5);
    // rho_ = fabs(correlation): a third block with |rho| equal to the first
    // pins that the sign is consumed only by the m_ selection.
    emitLattice2D("l2d_zero", 0.0);
}

}  // namespace

int main() {
    Date evalDate(17, January, 2024);
    Settings::instance().evaluationDate() = evalDate;

    std::cout << "{\n";
    block_binomial(evalDate);
    block_bsm_lattice(evalDate);
    block_lattice2d();
    std::cout << "\n}" << std::endl;
    return 0;
}

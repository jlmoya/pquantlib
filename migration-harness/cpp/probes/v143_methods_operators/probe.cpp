// v1.43 methods/finitedifferences/operators probe — model-family operators.
//
// Emits reference values (JSON on stdout) for:
//
//   * FdmHullWhiteOp        (1-D and as direction 2 of a 3-D grid)
//   * FdmG2Op               (2-D)
//   * FdmHestonEquityPart / FdmHestonVariancePart / FdmHestonOp (2-D),
//     plain, mixing-factor != 1, and with a leverage function (both the
//     plain branch and the max(0.01, .) clamp branch)
//   * FdmHestonHullWhiteEquityPart / FdmHestonHullWhiteOp (3-D)
//   * FdmCIREquityPart / FdmCIRRatesPart / FdmCIRMixedPart / FdmCIROp (2-D)
//   * FdmBatesOp + FdmBatesOp::IntegroIntegrand (2-D), with an empty
//     boundary set and with a Lower+Upper Dirichlet boundary set
//
// For every operator we pin, after setTime(t1, t2):
//   apply(v), apply_mixed(v), apply_direction(d, v) for every direction,
//   solve_splitting(d, v, dt) for every direction, preconditioner(v, dt)
//   and the diagonal of every toMatrixDecomp() block.
//
// The synthetic test vectors are built from the mesher locations so the
// Python side can rebuild them bit-identically:
//   const   v[i] = 1
//   lin0    v[i] = location(i, 0)
//   lin1    v[i] = location(i, 1)
//   lin2    v[i] = location(i, 2)          (3-D only)
//   mix     v[i] = location(i,0)*location(i,1)
//   idx     v[i] = 1 + 0.01*i              (catches index permutations)
//
// C++ parity:
//   ql/methods/finitedifferences/operators/fdmhullwhiteop.{hpp,cpp}
//   ql/methods/finitedifferences/operators/fdmg2op.{hpp,cpp}
//   ql/methods/finitedifferences/operators/fdmhestonop.{hpp,cpp}
//   ql/methods/finitedifferences/operators/fdmhestonhullwhiteop.{hpp,cpp}
//   ql/methods/finitedifferences/operators/fdmcirop.{hpp,cpp}
//   ql/methods/finitedifferences/operators/fdmbatesop.{hpp,cpp}
//   @ v1.43 (6b57206e0).
//
// Note: Settings::evaluationDate() is intentionally NOT set (see the
// cluster_w5a probe for the Apple-Clang/dylib rationale); every term
// structure is built with an explicit reference date instead.

#include <ql/handle.hpp>
#include <ql/methods/finitedifferences/meshers/fdmmeshercomposite.hpp>
#include <ql/methods/finitedifferences/meshers/uniform1dmesher.hpp>
#include <ql/methods/finitedifferences/operators/fdmbatesop.hpp>
#include <ql/methods/finitedifferences/operators/fdmcirop.hpp>
#include <ql/methods/finitedifferences/operators/fdmg2op.hpp>
#include <ql/methods/finitedifferences/operators/fdmhestonhullwhiteop.hpp>
#include <ql/methods/finitedifferences/operators/fdmhestonop.hpp>
#include <ql/methods/finitedifferences/operators/fdmhullwhiteop.hpp>
#include <ql/methods/finitedifferences/operators/fdmlinearoplayout.hpp>
#include <ql/methods/finitedifferences/utilities/fdmboundaryconditionset.hpp>
#include <ql/methods/finitedifferences/utilities/fdmdirichletboundary.hpp>
#include <ql/methods/finitedifferences/utilities/fdmquantohelper.hpp>
#include <ql/models/shortrate/onefactormodels/hullwhite.hpp>
#include <ql/models/shortrate/twofactormodels/g2.hpp>
#include <ql/processes/batesprocess.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/processes/coxingersollrossprocess.hpp>
#include <ql/processes/hestonprocess.hpp>
#include <ql/processes/hullwhiteprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/volatility/equityfx/localconstantvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/date.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace QuantLib;

namespace {

bool g_first = true;

void key(const char* name) {
    if (!g_first) std::cout << ",\n";
    g_first = false;
    std::cout << "  \"" << name << "\": ";
}

void emit(const char* name, Real v) {
    key(name);
    std::cout << v;
}

void emit_array(const char* name, const Array& a) {
    key(name);
    std::cout << "[";
    for (Size i = 0; i < a.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << a[i];
    }
    std::cout << "]";
}

void emit_diag(const char* name, const SparseMatrix& m) {
    Array d(m.size1());
    for (Size i = 0; i < m.size1(); ++i) d[i] = m(i, i);
    emit_array(name, d);
}

// --- synthetic test vectors -------------------------------------------

Array vec_const(const ext::shared_ptr<FdmMesher>& mesher) {
    return Array(mesher->layout()->size(), 1.0);
}

Array vec_lin(const ext::shared_ptr<FdmMesher>& mesher, Size dir) {
    Array v(mesher->layout()->size());
    for (const auto& iter : *mesher->layout())
        v[iter.index()] = mesher->location(iter, dir);
    return v;
}

Array vec_mix(const ext::shared_ptr<FdmMesher>& mesher) {
    Array v(mesher->layout()->size());
    for (const auto& iter : *mesher->layout())
        v[iter.index()] = mesher->location(iter, 0) * mesher->location(iter, 1);
    return v;
}

Array vec_idx(const ext::shared_ptr<FdmMesher>& mesher) {
    const Size n = mesher->layout()->size();
    Array v(n);
    for (Size i = 0; i < n; ++i) v[i] = 1.0 + 0.01 * Real(i);
    return v;
}

// Emit the whole composite surface of an operator for one test vector.
template <class OP>
void emit_surface(const std::string& prefix,
                  OP& op,
                  const Array& v,
                  const std::string& tag,
                  Size nDirections,
                  Real dt) {
    const std::string p = prefix + "_" + tag + "_";
    emit_array((p + "apply").c_str(), op.apply(v));
    emit_array((p + "apply_mixed").c_str(), op.apply_mixed(v));
    for (Size d = 0; d < nDirections; ++d) {
        emit_array((p + "apply_dir" + std::to_string(d)).c_str(),
                   op.apply_direction(d, v));
        emit_array((p + "solve_dir" + std::to_string(d)).c_str(),
                   op.solve_splitting(d, v, dt));
    }
    emit_array((p + "precond").c_str(), op.preconditioner(v, dt));
}

// ======================================================================
// Block 1: FdmHullWhiteOp (1-D)
// ======================================================================
__attribute__((noinline))
void block_hull_white_1d(const Handle<YieldTermStructure>& rTS) {
    auto m = ext::make_shared<Uniform1dMesher>(-0.06, 0.08, 9);
    auto mesher = ext::make_shared<FdmMesherComposite>(m);

    auto model = ext::make_shared<HullWhite>(rTS, 0.05, 0.012);
    FdmHullWhiteOp op(mesher, model, 0);
    op.setTime(0.2, 0.5);

    emit("hw1d_size", Real(op.size()));
    emit_array("hw1d_locations", mesher->locations(0));
    // The frozen drift level the operator uses: phi = 0.5*(r(t1,0)+r(t2,0)).
    emit("hw1d_short_rate_t1", model->dynamics()->shortRate(0.2, 0.0));
    emit("hw1d_short_rate_t2", model->dynamics()->shortRate(0.5, 0.0));

    emit_surface("hw1d", op, vec_const(mesher), "const", 1, 0.02);
    emit_surface("hw1d", op, vec_lin(mesher, 0), "lin0", 1, 0.02);
    emit_surface("hw1d", op, vec_idx(mesher), "idx", 1, 0.02);

    // direction 1 does not exist for this op -> zero array
    emit_array("hw1d_const_apply_dir1", op.apply_direction(1, vec_const(mesher)));
    emit_array("hw1d_const_solve_dir1", op.solve_splitting(1, vec_const(mesher), 0.02));

    const std::vector<SparseMatrix> decomp = op.toMatrixDecomp();
    emit("hw1d_decomp_size", Real(decomp.size()));
    emit_diag("hw1d_decomp0_diag", decomp[0]);
}

// ======================================================================
// Block 2: FdmG2Op (2-D)
// ======================================================================
__attribute__((noinline))
void block_g2(const Handle<YieldTermStructure>& rTS) {
    auto mx = ext::make_shared<Uniform1dMesher>(-0.07, 0.09, 6);
    auto my = ext::make_shared<Uniform1dMesher>(-0.05, 0.06, 5);
    auto mesher = ext::make_shared<FdmMesherComposite>(mx, my);

    auto model = ext::make_shared<G2>(rTS, 0.1, 0.011, 0.2, 0.021, -0.45);
    FdmG2Op op(mesher, model, 0, 1);
    op.setTime(0.25, 0.75);

    emit("g2_size", Real(op.size()));
    emit_array("g2_locations0", mesher->locations(0));
    emit_array("g2_locations1", mesher->locations(1));
    emit("g2_short_rate_t1", model->dynamics()->shortRate(0.25, 0.0, 0.0));
    emit("g2_short_rate_t2", model->dynamics()->shortRate(0.75, 0.0, 0.0));

    emit_surface("g2", op, vec_const(mesher), "const", 2, 0.02);
    emit_surface("g2", op, vec_lin(mesher, 0), "lin0", 2, 0.02);
    emit_surface("g2", op, vec_lin(mesher, 1), "lin1", 2, 0.02);
    emit_surface("g2", op, vec_mix(mesher), "mix", 2, 0.02);
    emit_surface("g2", op, vec_idx(mesher), "idx", 2, 0.02);

    emit_array("g2_const_apply_dir2", op.apply_direction(2, vec_const(mesher)));
    emit_array("g2_const_solve_dir2", op.solve_splitting(2, vec_const(mesher), 0.02));

    const std::vector<SparseMatrix> decomp = op.toMatrixDecomp();
    emit("g2_decomp_size", Real(decomp.size()));
    emit_diag("g2_decomp0_diag", decomp[0]);
    emit_diag("g2_decomp1_diag", decomp[1]);
    emit_diag("g2_decomp2_diag", decomp[2]);

    // reversed directions — pins that direction1/direction2 are honoured
    FdmG2Op opRev(mesher, model, 1, 0);
    opRev.setTime(0.25, 0.75);
    emit_array("g2rev_const_apply", opRev.apply(vec_const(mesher)));
    emit_array("g2rev_idx_apply", opRev.apply(vec_idx(mesher)));
    emit_array("g2rev_idx_apply_dir0", opRev.apply_direction(0, vec_idx(mesher)));
    emit_array("g2rev_idx_apply_dir1", opRev.apply_direction(1, vec_idx(mesher)));
    emit_array("g2rev_idx_solve_dir0", opRev.solve_splitting(0, vec_idx(mesher), 0.02));
    emit_array("g2rev_idx_solve_dir1", opRev.solve_splitting(1, vec_idx(mesher), 0.02));
}

// ======================================================================
// Block 3: FdmHestonOp (2-D) + the two parts
// ======================================================================
ext::shared_ptr<HestonProcess> makeHeston(const Handle<YieldTermStructure>& rTS,
                                          const Handle<YieldTermStructure>& qTS) {
    Handle<Quote> s0(ext::make_shared<SimpleQuote>(100.0));
    return ext::make_shared<HestonProcess>(rTS, qTS, s0, 0.09, 1.5, 0.04, 0.4, -0.6);
}

ext::shared_ptr<FdmMesherComposite> makeHestonMesher() {
    auto mx = ext::make_shared<Uniform1dMesher>(std::log(70.0), std::log(140.0), 7);
    auto mv = ext::make_shared<Uniform1dMesher>(0.005, 0.4, 5);
    return ext::make_shared<FdmMesherComposite>(mx, mv);
}

__attribute__((noinline))
void block_heston(const Handle<YieldTermStructure>& rTS,
                  const Handle<YieldTermStructure>& qTS,
                  const Date& today,
                  const DayCounter& dc) {
    auto mesher = makeHestonMesher();
    auto process = makeHeston(rTS, qTS);

    emit_array("heston_locations0", mesher->locations(0));
    emit_array("heston_locations1", mesher->locations(1));

    // --- the two parts on their own -----------------------------------
    FdmHestonEquityPart equityPart(mesher, rTS.currentLink(), qTS.currentLink(),
                                   ext::shared_ptr<FdmQuantoHelper>());
    equityPart.setTime(0.1, 0.35);
    emit_array("heston_equity_part_L", equityPart.getL());
    emit_array("heston_equity_part_apply_const",
               equityPart.getMap().apply(vec_const(mesher)));
    emit_array("heston_equity_part_apply_idx",
               equityPart.getMap().apply(vec_idx(mesher)));
    emit_diag("heston_equity_part_diag", equityPart.getMap().toMatrix());

    FdmHestonVariancePart variancePart(mesher, rTS.currentLink(), 0.4, 1.5, 0.04);
    variancePart.setTime(0.1, 0.35);
    emit_array("heston_variance_part_apply_const",
               variancePart.getMap().apply(vec_const(mesher)));
    emit_array("heston_variance_part_apply_idx",
               variancePart.getMap().apply(vec_idx(mesher)));
    emit_diag("heston_variance_part_diag", variancePart.getMap().toMatrix());

    // --- the composite op ---------------------------------------------
    FdmHestonOp op(mesher, process);
    op.setTime(0.1, 0.35);

    emit("heston_size", Real(op.size()));
    emit_surface("heston", op, vec_const(mesher), "const", 2, 0.02);
    emit_surface("heston", op, vec_lin(mesher, 0), "lin0", 2, 0.02);
    emit_surface("heston", op, vec_lin(mesher, 1), "lin1", 2, 0.02);
    emit_surface("heston", op, vec_mix(mesher), "mix", 2, 0.02);
    emit_surface("heston", op, vec_idx(mesher), "idx", 2, 0.02);

    const std::vector<SparseMatrix> decomp = op.toMatrixDecomp();
    emit("heston_decomp_size", Real(decomp.size()));
    emit_diag("heston_decomp0_diag", decomp[0]);
    emit_diag("heston_decomp1_diag", decomp[1]);
    emit_diag("heston_decomp2_diag", decomp[2]);

    // --- mixing factor != 1 --------------------------------------------
    FdmHestonOp opMix(mesher, process,
                      ext::shared_ptr<FdmQuantoHelper>(),
                      ext::shared_ptr<LocalVolTermStructure>(),
                      0.7);
    opMix.setTime(0.1, 0.35);
    emit_array("heston_mix07_idx_apply", opMix.apply(vec_idx(mesher)));
    emit_array("heston_mix07_idx_apply_mixed", opMix.apply_mixed(vec_idx(mesher)));
    emit_array("heston_mix07_idx_apply_dir1", opMix.apply_direction(1, vec_idx(mesher)));
    emit_array("heston_mix07_idx_solve_dir1", opMix.solve_splitting(1, vec_idx(mesher), 0.02));

    // --- leverage function, plain branch (L = 0.25) ---------------------
    auto lev = ext::make_shared<LocalConstantVol>(today, 0.25, dc);
    FdmHestonOp opLev(mesher, process,
                      ext::shared_ptr<FdmQuantoHelper>(), lev, 1.0);
    opLev.setTime(0.1, 0.35);
    emit_array("heston_lev025_idx_apply", opLev.apply(vec_idx(mesher)));
    emit_array("heston_lev025_idx_apply_mixed", opLev.apply_mixed(vec_idx(mesher)));
    emit_array("heston_lev025_idx_apply_dir0", opLev.apply_direction(0, vec_idx(mesher)));
    emit_array("heston_lev025_idx_solve_dir0", opLev.solve_splitting(0, vec_idx(mesher), 0.02));

    FdmHestonEquityPart equityLev(mesher, rTS.currentLink(), qTS.currentLink(),
                                  ext::shared_ptr<FdmQuantoHelper>(), lev);
    equityLev.setTime(0.1, 0.35);
    emit_array("heston_lev025_L", equityLev.getL());

    // --- leverage function, max(0.01, .) clamp branch --------------------
    auto levTiny = ext::make_shared<LocalConstantVol>(today, 0.005, dc);
    FdmHestonEquityPart equityClamp(mesher, rTS.currentLink(), qTS.currentLink(),
                                    ext::shared_ptr<FdmQuantoHelper>(), levTiny);
    equityClamp.setTime(0.1, 0.35);
    emit_array("heston_levclamp_L", equityClamp.getL());
    emit_array("heston_levclamp_apply_idx",
               equityClamp.getMap().apply(vec_idx(mesher)));

    // --- quanto branch ---------------------------------------------------
    auto fTS = ext::make_shared<FlatForward>(today, 0.025, dc);
    auto fxVolTS = ext::make_shared<BlackConstantVol>(today, NullCalendar(), 0.12, dc);
    auto quanto = ext::make_shared<FdmQuantoHelper>(
        rTS.currentLink(), fTS, fxVolTS, -0.35, 1.25);
    emit("heston_quanto_adjustment_scalar",
         quanto->quantoAdjustment(0.3, 0.1, 0.35));

    FdmHestonEquityPart equityQuanto(mesher, rTS.currentLink(), qTS.currentLink(),
                                     quanto);
    equityQuanto.setTime(0.1, 0.35);
    emit_array("heston_quanto_equity_part_apply_idx",
               equityQuanto.getMap().apply(vec_idx(mesher)));
    emit_diag("heston_quanto_equity_part_diag",
              equityQuanto.getMap().toMatrix());

    FdmHestonOp opQuanto(mesher, process, quanto);
    opQuanto.setTime(0.1, 0.35);
    emit_array("heston_quanto_idx_apply", opQuanto.apply(vec_idx(mesher)));
    emit_array("heston_quanto_idx_apply_dir0",
               opQuanto.apply_direction(0, vec_idx(mesher)));
    emit_array("heston_quanto_idx_solve_dir0",
               opQuanto.solve_splitting(0, vec_idx(mesher), 0.02));
    emit_array("heston_quanto_idx_precond",
               opQuanto.preconditioner(vec_idx(mesher), 0.02));

    // quanto + leverage together — the branch that multiplies the equity
    // vol slice by L before asking for the adjustment
    FdmHestonEquityPart equityQuantoLev(mesher, rTS.currentLink(), qTS.currentLink(),
                                        quanto, lev);
    equityQuantoLev.setTime(0.1, 0.35);
    emit_array("heston_quanto_lev_apply_idx",
               equityQuantoLev.getMap().apply(vec_idx(mesher)));
}

// ======================================================================
// Block 4: FdmHestonHullWhiteOp (3-D)
// ======================================================================
__attribute__((noinline))
void block_heston_hull_white(const Handle<YieldTermStructure>& rTS,
                             const Handle<YieldTermStructure>& qTS) {
    auto mx = ext::make_shared<Uniform1dMesher>(std::log(80.0), std::log(130.0), 5);
    auto mv = ext::make_shared<Uniform1dMesher>(0.01, 0.3, 4);
    // The rate axis deliberately avoids a node at x_r = -phi: there
    // ``x_r + phi`` would cancel to ~5e-6 out of 0.04 and amplify the last
    // bits of phi by ~1e4 in every value that is constant along direction 2.
    auto mr = ext::make_shared<Uniform1dMesher>(-0.05, 0.07, 4);
    auto mesher = ext::make_shared<FdmMesherComposite>(mx, mv, mr);

    auto hestonProcess = makeHeston(rTS, qTS);
    auto hwProcess = ext::make_shared<HullWhiteProcess>(rTS, 0.05, 0.012);

    emit_array("hhw_locations0", mesher->locations(0));
    emit_array("hhw_locations1", mesher->locations(1));
    emit_array("hhw_locations2", mesher->locations(2));

    // the equity part on its own
    auto hwModel = ext::make_shared<HullWhite>(rTS, 0.05, 0.012);
    emit("hhw_short_rate_t1", hwModel->dynamics()->shortRate(0.15, 0.0));
    emit("hhw_short_rate_t2", hwModel->dynamics()->shortRate(0.4, 0.0));
    FdmHestonHullWhiteEquityPart equityPart(mesher, hwModel, qTS.currentLink());
    equityPart.setTime(0.15, 0.4);
    emit_array("hhw_equity_part_apply_const",
               equityPart.getMap().apply(vec_const(mesher)));
    emit_array("hhw_equity_part_apply_idx",
               equityPart.getMap().apply(vec_idx(mesher)));
    emit_diag("hhw_equity_part_diag", equityPart.getMap().toMatrix());

    // FdmHullWhiteOp acting on direction 2 of the 3-D grid
    FdmHullWhiteOp hwOp(mesher, hwModel, 2);
    hwOp.setTime(0.15, 0.4);
    emit_array("hhw_hwop_dir2_apply_idx", hwOp.apply(vec_idx(mesher)));
    emit_array("hhw_hwop_dir2_solve_idx", hwOp.solve_splitting(2, vec_idx(mesher), 0.02));
    emit_array("hhw_hwop_dir2_apply_dir0", hwOp.apply_direction(0, vec_idx(mesher)));

    FdmHestonHullWhiteOp op(mesher, hestonProcess, hwProcess, 0.4);
    op.setTime(0.15, 0.4);

    emit("hhw_size", Real(op.size()));
    emit_surface("hhw", op, vec_const(mesher), "const", 3, 0.02);
    emit_surface("hhw", op, vec_lin(mesher, 0), "lin0", 3, 0.02);
    emit_surface("hhw", op, vec_lin(mesher, 1), "lin1", 3, 0.02);
    emit_surface("hhw", op, vec_lin(mesher, 2), "lin2", 3, 0.02);
    emit_surface("hhw", op, vec_mix(mesher), "mix", 3, 0.02);
    emit_surface("hhw", op, vec_idx(mesher), "idx", 3, 0.02);

    const std::vector<SparseMatrix> decomp = op.toMatrixDecomp();
    emit("hhw_decomp_size", Real(decomp.size()));
    emit_diag("hhw_decomp0_diag", decomp[0]);
    emit_diag("hhw_decomp1_diag", decomp[1]);
    emit_diag("hhw_decomp2_diag", decomp[2]);
    emit_diag("hhw_decomp3_diag", decomp[3]);
}

// ======================================================================
// Block 5: FdmCIROp (2-D)
// ======================================================================
__attribute__((noinline))
void block_cir(const Handle<YieldTermStructure>& rTS,
               const Handle<YieldTermStructure>& qTS,
               const Date& today,
               const DayCounter& dc) {
    auto mx = ext::make_shared<Uniform1dMesher>(std::log(70.0), std::log(140.0), 7);
    auto my = ext::make_shared<Uniform1dMesher>(0.005, 0.09, 5);
    auto mesher = ext::make_shared<FdmMesherComposite>(mx, my);

    Handle<Quote> s0(ext::make_shared<SimpleQuote>(100.0));
    Handle<BlackVolTermStructure> volTS(
        ext::make_shared<BlackConstantVol>(today, NullCalendar(), 0.22, dc));
    auto bsProcess = ext::make_shared<BlackScholesMertonProcess>(s0, qTS, rTS, volTS);
    auto cirProcess = ext::make_shared<CoxIngersollRossProcess>(0.6, 0.08, 0.03, 0.04);

    const Real rho = -0.3;
    const Real strike = 100.0;

    emit_array("cir_locations0", mesher->locations(0));
    emit_array("cir_locations1", mesher->locations(1));

    FdmCIREquityPart equityPart(mesher, bsProcess, strike);
    equityPart.setTime(0.1, 0.35);
    emit_array("cir_equity_part_apply_const",
               equityPart.getMap().apply(vec_const(mesher)));
    emit_array("cir_equity_part_apply_idx",
               equityPart.getMap().apply(vec_idx(mesher)));
    emit_diag("cir_equity_part_diag", equityPart.getMap().toMatrix());

    FdmCIRRatesPart ratesPart(mesher, 0.08, 0.6, 0.04);
    ratesPart.setTime(0.1, 0.35);
    emit_array("cir_rates_part_apply_const",
               ratesPart.getMap().apply(vec_const(mesher)));
    emit_array("cir_rates_part_apply_idx",
               ratesPart.getMap().apply(vec_idx(mesher)));
    emit_diag("cir_rates_part_diag", ratesPart.getMap().toMatrix());

    FdmCIRMixedPart mixedPart(mesher, cirProcess, bsProcess, rho, strike);
    mixedPart.setTime(0.1, 0.35);
    emit_array("cir_mixed_part_apply_const",
               mixedPart.getMap().apply(vec_const(mesher)));
    emit_array("cir_mixed_part_apply_mix",
               mixedPart.getMap().apply(vec_mix(mesher)));
    emit_array("cir_mixed_part_apply_idx",
               mixedPart.getMap().apply(vec_idx(mesher)));

    FdmCIROp op(mesher, cirProcess, bsProcess, rho, strike);
    op.setTime(0.1, 0.35);

    emit("cir_size", Real(op.size()));
    emit_surface("cir", op, vec_const(mesher), "const", 2, 0.02);
    emit_surface("cir", op, vec_lin(mesher, 0), "lin0", 2, 0.02);
    emit_surface("cir", op, vec_lin(mesher, 1), "lin1", 2, 0.02);
    emit_surface("cir", op, vec_mix(mesher), "mix", 2, 0.02);
    emit_surface("cir", op, vec_idx(mesher), "idx", 2, 0.02);

    const std::vector<SparseMatrix> decomp = op.toMatrixDecomp();
    emit("cir_decomp_size", Real(decomp.size()));
    emit_diag("cir_decomp0_diag", decomp[0]);
    emit_diag("cir_decomp1_diag", decomp[1]);
    emit_diag("cir_decomp2_diag", decomp[2]);
}

// ======================================================================
// Block 6: FdmBatesOp (2-D)
// ======================================================================
__attribute__((noinline))
void block_bates(const Handle<YieldTermStructure>& rTS,
                 const Handle<YieldTermStructure>& qTS) {
    auto mx = ext::make_shared<Uniform1dMesher>(std::log(70.0), std::log(140.0), 7);
    auto mv = ext::make_shared<Uniform1dMesher>(0.005, 0.4, 4);
    auto mesher = ext::make_shared<FdmMesherComposite>(mx, mv);

    Handle<Quote> s0(ext::make_shared<SimpleQuote>(100.0));
    auto batesProcess = ext::make_shared<BatesProcess>(
        rTS, qTS, s0, 0.09, 1.5, 0.04, 0.4, -0.6, 0.4, -0.5, 0.3);

    emit_array("bates_locations0", mesher->locations(0));
    emit_array("bates_locations1", mesher->locations(1));

    // --- empty boundary-condition set ----------------------------------
    {
        FdmBoundaryConditionSet bcSet;
        FdmBatesOp op(mesher, batesProcess, bcSet, 8);
        op.setTime(0.1, 0.35);

        emit("bates_size", Real(op.size()));
        emit_surface("bates", op, vec_const(mesher), "const", 2, 0.02);
        emit_surface("bates", op, vec_lin(mesher, 0), "lin0", 2, 0.02);
        emit_surface("bates", op, vec_lin(mesher, 1), "lin1", 2, 0.02);
        emit_surface("bates", op, vec_mix(mesher), "mix", 2, 0.02);
        emit_surface("bates", op, vec_idx(mesher), "idx", 2, 0.02);

        // higher integration order — pins the Gauss-Hermite order plumbing
        FdmBatesOp op16(mesher, batesProcess, bcSet, 16);
        op16.setTime(0.1, 0.35);
        emit_array("bates_ord16_idx_apply", op16.apply(vec_idx(mesher)));
        emit_array("bates_ord16_idx_apply_mixed", op16.apply_mixed(vec_idx(mesher)));
    }

    // --- Dirichlet boundary-condition set -------------------------------
    {
        FdmBoundaryConditionSet bcSet;
        bcSet.push_back(ext::make_shared<FdmDirichletBoundary>(
            mesher, 2.5, 0, FdmDirichletBoundary::Lower));
        bcSet.push_back(ext::make_shared<FdmDirichletBoundary>(
            mesher, -1.5, 0, FdmDirichletBoundary::Upper));

        FdmBatesOp op(mesher, batesProcess, bcSet, 8);
        op.setTime(0.1, 0.35);
        emit_array("bates_bc_const_apply", op.apply(vec_const(mesher)));
        emit_array("bates_bc_const_apply_mixed", op.apply_mixed(vec_const(mesher)));
        emit_array("bates_bc_idx_apply", op.apply(vec_idx(mesher)));
        emit_array("bates_bc_idx_apply_mixed", op.apply_mixed(vec_idx(mesher)));
        emit_array("bates_bc_mix_apply_mixed", op.apply_mixed(vec_mix(mesher)));
        // the Dirichlet extremes the integrand clamps against
        emit("bates_bc_x_lower", mesher->locations(0)[0]);
        emit("bates_bc_x_upper",
             mesher->locations(0)[mesher->layout()->size() - 1]);
    }
}

}  // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    DayCounter dc = Actual365Fixed();
    Date today(15, January, 2024);

    Handle<YieldTermStructure> rTS(ext::make_shared<FlatForward>(today, 0.04, dc));
    Handle<YieldTermStructure> qTS(ext::make_shared<FlatForward>(today, 0.015, dc));

    block_hull_white_1d(rTS);
    block_g2(rTS);
    block_heston(rTS, qTS, today, dc);
    block_heston_hull_white(rTS, qTS);
    block_cir(rTS, qTS, today, dc);
    block_bates(rTS, qTS);

    std::cout << "\n}\n";
    return 0;
}

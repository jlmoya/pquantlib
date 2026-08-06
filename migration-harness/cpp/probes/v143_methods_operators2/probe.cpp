// v1.43 "operators B" cluster probe: forward-equation + miscellaneous
// finite-difference operators.
//
// Emits JSON on stdout (PQuantLib probe convention -- never writes a file).
//
// Covers:
//   * NumericalDifferentiation   (Fornberg weights; Central/Backward/Forward
//                                 schemes + irregular offsets + operator())
//   * NthOrderDerivativeOp       (1-D + 2-D; odd/even stencils; dense matrix)
//   * FdmWienerOp                (2-D, with and without a rTS)
//   * Fdm2dBlackScholesOp        (2-D, constant vol branch)
//   * FdmBlackScholesFwdOp       (1-D const-vol + 1-D local-vol + 2-D dir=1)
//   * FdmLocalVolFwdOp           (1-D)
//   * FdmSquareRootFwdOp         (Plain / Power / Log; 1-D and 2-D dir=1)
//   * FdmHestonFwdOp             (Plain / Power / Log; with + without
//                                 leverage function; mixing factor)
//   * FdmSabrOp                  (2-D)
//   * FdmCEVOp                   (1-D dir=0 and 2-D dir=1)
//
// C++ parity:
//   ql/methods/finitedifferences/operators/numericaldifferentiation.{hpp,cpp}
//   ql/methods/finitedifferences/operators/nthorderderivativeop.{hpp,cpp}
//   ql/methods/finitedifferences/operators/fdmwienerop.{hpp,cpp}
//   ql/methods/finitedifferences/operators/fdm2dblackscholesop.{hpp,cpp}
//   ql/methods/finitedifferences/operators/fdmblackscholesfwdop.{hpp,cpp}
//   ql/methods/finitedifferences/operators/fdmlocalvolfwdop.{hpp,cpp}
//   ql/methods/finitedifferences/operators/fdmsquarerootfwdop.{hpp,cpp}
//   ql/methods/finitedifferences/operators/fdmhestonfwdop.{hpp,cpp}
//   ql/methods/finitedifferences/operators/fdmsabrop.{hpp,cpp}
//   ql/methods/finitedifferences/operators/fdmcevop.{hpp,cpp}
//   @ v1.43 (6b57206e0).
//
// Note: Settings::evaluationDate() is intentionally NOT set in main() --
// every term structure is built with an explicit reference date so the
// probe is independent of the global evaluation date.

#include <ql/handle.hpp>
#include <ql/math/array.hpp>
#include <ql/methods/finitedifferences/meshers/fdmmeshercomposite.hpp>
#include <ql/methods/finitedifferences/meshers/uniform1dmesher.hpp>
#include <ql/methods/finitedifferences/operators/fdm2dblackscholesop.hpp>
#include <ql/methods/finitedifferences/operators/fdmblackscholesfwdop.hpp>
#include <ql/methods/finitedifferences/operators/fdmcevop.hpp>
#include <ql/methods/finitedifferences/operators/fdmhestonfwdop.hpp>
#include <ql/methods/finitedifferences/operators/fdmlinearoplayout.hpp>
#include <ql/methods/finitedifferences/operators/fdmlocalvolfwdop.hpp>
#include <ql/methods/finitedifferences/operators/fdmsabrop.hpp>
#include <ql/methods/finitedifferences/operators/fdmsquarerootfwdop.hpp>
#include <ql/methods/finitedifferences/operators/fdmwienerop.hpp>
#include <ql/methods/finitedifferences/operators/nthorderderivativeop.hpp>
#include <ql/methods/finitedifferences/operators/numericaldifferentiation.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/processes/hestonprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/volatility/equityfx/localvoltermstructure.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/date.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace QuantLib;

namespace {

bool g_first = true;

void key(const std::string& name) {
    if (!g_first) std::cout << ",\n";
    g_first = false;
    std::cout << "  \"" << name << "\": ";
}

void emit(const std::string& name, Real v) {
    key(name);
    std::cout << v;
}

void emitArray(const std::string& name, const Array& a) {
    key(name);
    std::cout << "[";
    for (Size i = 0; i < a.size(); ++i) {
        if (i != 0) std::cout << ", ";
        std::cout << a[i];
    }
    std::cout << "]";
}

void emitVector(const std::string& name, const std::vector<Real>& a) {
    key(name);
    std::cout << "[";
    for (Size i = 0; i < a.size(); ++i) {
        if (i != 0) std::cout << ", ";
        std::cout << a[i];
    }
    std::cout << "]";
}

// Dense row-major dump of a SparseMatrix.
void emitMatrix(const std::string& name, const SparseMatrix& m) {
    std::vector<Real> flat;
    flat.reserve(m.size1() * m.size2());
    for (Size i = 0; i < m.size1(); ++i)
        for (Size j = 0; j < m.size2(); ++j)
            flat.push_back(m(i, j));
    emitVector(name, flat);
}

// ---------------------------------------------------------------------------
// Synthetic test vectors on a mesher: constant, linear along each direction,
// quadratic along direction 0, and a product of the two directions.
// ---------------------------------------------------------------------------
Array constVec(const ext::shared_ptr<FdmMesher>& mesher, Real c) {
    return Array(mesher->layout()->size(), c);
}

Array linVec(const ext::shared_ptr<FdmMesher>& mesher, Size direction) {
    Array v(mesher->layout()->size());
    for (const auto& iter : *mesher->layout())
        v[iter.index()] = mesher->location(iter, direction);
    return v;
}

Array quadVec(const ext::shared_ptr<FdmMesher>& mesher, Size direction) {
    Array v(mesher->layout()->size());
    for (const auto& iter : *mesher->layout()) {
        const Real x = mesher->location(iter, direction);
        v[iter.index()] = x * x;
    }
    return v;
}

Array prodVec(const ext::shared_ptr<FdmMesher>& mesher) {
    Array v(mesher->layout()->size());
    for (const auto& iter : *mesher->layout())
        v[iter.index()] = mesher->location(iter, 0) * mesher->location(iter, 1);
    return v;
}

// "Ramp" vector: strictly increasing in the flat index -- catches operators
// that get direction handling right on symmetric inputs but wrong otherwise.
Array rampVec(const ext::shared_ptr<FdmMesher>& mesher) {
    const Size n = mesher->layout()->size();
    Array v(n);
    for (Size i = 0; i < n; ++i)
        v[i] = 1.0 + 0.25 * Real(i) - 0.03 * Real(i) * Real(i);
    return v;
}

// ---------------------------------------------------------------------------
// A deterministic, closed-form local-volatility surface used for the
// local-vol / leverage-function code paths. Deliberately spatially AND
// temporally varying so a wrong (t, spot) mapping cannot pass.
//
//   sigma_loc(t, S) = 0.20 + 0.05 * log(S / 100) + 0.10 * t
// ---------------------------------------------------------------------------
class ProbeLocalVol : public LocalVolTermStructure {
  public:
    ProbeLocalVol(const Date& refDate, const DayCounter& dc)
    : LocalVolTermStructure(refDate, NullCalendar(), Following, dc) {}

    Date maxDate() const override { return Date::maxDate(); }
    Real minStrike() const override { return 1.0e-8; }
    Real maxStrike() const override { return 1.0e8; }

  protected:
    Volatility localVolImpl(Time t, Real s) const override {
        return 0.20 + 0.05 * std::log(s / 100.0) + 0.10 * t;
    }
};

// Function used to exercise NumericalDifferentiation::operator().
Real probeFunction(Real x) { return std::exp(0.7 * x) * std::sin(1.3 * x) + 0.25 * x * x * x; }

// ===========================================================================
// Block 1: NumericalDifferentiation
// ===========================================================================
void ndCase(const std::string& tag,
            Size order,
            Real h,
            Size steps,
            NumericalDifferentiation::Scheme scheme) {
    const std::function<Real(Real)> empty;
    NumericalDifferentiation nd(empty, order, h, steps, scheme);
    emitArray(tag + "_offsets", nd.offsets());
    emitArray(tag + "_weights", nd.weights());
}

__attribute__((noinline))
void block_numerical_differentiation() {
    // --- Central scheme -----------------------------------------------
    ndCase("nd_central_o1_n3", 1, 0.1, 3, NumericalDifferentiation::Central);
    ndCase("nd_central_o1_n5", 1, 0.05, 5, NumericalDifferentiation::Central);
    ndCase("nd_central_o2_n3", 2, 0.1, 3, NumericalDifferentiation::Central);
    ndCase("nd_central_o2_n5", 2, 0.1, 5, NumericalDifferentiation::Central);
    ndCase("nd_central_o2_n7", 2, 0.25, 7, NumericalDifferentiation::Central);
    ndCase("nd_central_o3_n7", 3, 0.1, 7, NumericalDifferentiation::Central);
    ndCase("nd_central_o4_n9", 4, 0.2, 9, NumericalDifferentiation::Central);

    // --- Backward scheme ----------------------------------------------
    ndCase("nd_backward_o1_n3", 1, 0.1, 3, NumericalDifferentiation::Backward);
    ndCase("nd_backward_o1_n5", 1, 0.02, 5, NumericalDifferentiation::Backward);
    ndCase("nd_backward_o2_n4", 2, 0.1, 4, NumericalDifferentiation::Backward);

    // --- Forward scheme -----------------------------------------------
    ndCase("nd_forward_o1_n3", 1, 0.1, 3, NumericalDifferentiation::Forward);
    ndCase("nd_forward_o2_n4", 2, 0.1, 4, NumericalDifferentiation::Forward);
    ndCase("nd_forward_o3_n5", 3, 0.15, 5, NumericalDifferentiation::Forward);

    // --- Explicit (irregular) offsets ---------------------------------
    const std::function<Real(Real)> empty;
    {
        Array offsets = {-0.3, -0.1, 0.0, 0.15, 0.4};
        for (Size order = 1; order <= 3; ++order) {
            NumericalDifferentiation nd(empty, order, offsets);
            emitArray("nd_irregular_a_o" + std::to_string(order) + "_offsets", nd.offsets());
            emitArray("nd_irregular_a_o" + std::to_string(order) + "_weights", nd.weights());
        }
    }
    {
        Array offsets = {0.0, 0.1, 0.3, 0.7};
        NumericalDifferentiation nd(empty, 1, offsets);
        emitArray("nd_irregular_b_o1_offsets", nd.offsets());
        emitArray("nd_irregular_b_o1_weights", nd.weights());
    }
    {
        // Offsets that are NOT sorted -- Fornberg does not require ordering.
        Array offsets = {0.2, -0.4, 0.0, 0.05};
        NumericalDifferentiation nd(empty, 2, offsets);
        emitArray("nd_unsorted_o2_offsets", nd.offsets());
        emitArray("nd_unsorted_o2_weights", nd.weights());
    }

    // --- operator() -----------------------------------------------------
    const std::function<Real(Real)> f = probeFunction;
    {
        NumericalDifferentiation nd(f, 1, 0.01, 5, NumericalDifferentiation::Central);
        emit("nd_call_central_o1_n5_at_0_5", nd(0.5));
        emit("nd_call_central_o1_n5_at_m1_25", nd(-1.25));
    }
    {
        NumericalDifferentiation nd(f, 2, 0.02, 7, NumericalDifferentiation::Central);
        emit("nd_call_central_o2_n7_at_0_5", nd(0.5));
    }
    {
        NumericalDifferentiation nd(f, 1, 0.005, 4, NumericalDifferentiation::Backward);
        emit("nd_call_backward_o1_n4_at_0_5", nd(0.5));
    }
    {
        Array offsets = {-0.3, -0.1, 0.0, 0.15, 0.4};
        NumericalDifferentiation nd(f, 1, offsets);
        emit("nd_call_irregular_a_o1_at_0_5", nd(0.5));
    }
    emit("nd_probe_function_at_0_5", probeFunction(0.5));
    emit("nd_probe_function_at_m1_25", probeFunction(-1.25));
}

// ===========================================================================
// Block 2: NthOrderDerivativeOp
// ===========================================================================
__attribute__((noinline))
void block_nth_order_derivative_op() {
    // --- 1-D uniform grid, 7 nodes ------------------------------------
    auto m1 = ext::make_shared<Uniform1dMesher>(0.0, 1.2, 7);
    auto mesher1d = ext::make_shared<FdmMesherComposite>(m1);
    emitArray("nth_1d_locations", mesher1d->locations(0));

    struct Case1d {
        const char* tag;
        Size order;
        Integer nPoints;
    };
    const Case1d cases1d[] = {
        {"nth_1d_o1_p3", 1, 3},
        {"nth_1d_o1_p4", 1, 4},
        {"nth_1d_o2_p3", 2, 3},
        {"nth_1d_o2_p5", 2, 5},
        {"nth_1d_o3_p5", 3, 5},
        {"nth_1d_o4_p5", 4, 5},
    };
    for (const auto& c : cases1d) {
        NthOrderDerivativeOp op(0, c.order, c.nPoints, mesher1d);
        emitArray(std::string(c.tag) + "_apply_const", op.apply(constVec(mesher1d, 1.0)));
        emitArray(std::string(c.tag) + "_apply_lin", op.apply(linVec(mesher1d, 0)));
        emitArray(std::string(c.tag) + "_apply_quad", op.apply(quadVec(mesher1d, 0)));
        emitArray(std::string(c.tag) + "_apply_ramp", op.apply(rampVec(mesher1d)));
    }
    // Dense matrices for two representative 1-D cases -- the sharpest pin.
    {
        NthOrderDerivativeOp op(0, 1, 3, mesher1d);
        const SparseMatrix m = op.toMatrix();
        emitMatrix("nth_1d_o1_p3_matrix", m);
    }
    {
        NthOrderDerivativeOp op(0, 2, 5, mesher1d);
        const SparseMatrix m = op.toMatrix();
        emitMatrix("nth_1d_o2_p5_matrix", m);
    }
    {
        // Even stencil -- exercises the isEven offset branch.
        NthOrderDerivativeOp op(0, 1, 4, mesher1d);
        const SparseMatrix m = op.toMatrix();
        emitMatrix("nth_1d_o1_p4_matrix", m);
    }

    // --- non-uniform 1-D grid (via a concatenated mesher is overkill;
    //     use a uniform grid with a different span) ---------------------
    auto m1b = ext::make_shared<Uniform1dMesher>(-2.0, 3.0, 9);
    auto mesher1db = ext::make_shared<FdmMesherComposite>(m1b);
    emitArray("nth_1db_locations", mesher1db->locations(0));
    {
        NthOrderDerivativeOp op(0, 2, 7, mesher1db);
        emitArray("nth_1db_o2_p7_apply_quad", op.apply(quadVec(mesher1db, 0)));
        emitArray("nth_1db_o2_p7_apply_ramp", op.apply(rampVec(mesher1db)));
    }

    // --- 2-D grid: 5 x 4 -----------------------------------------------
    auto mx = ext::make_shared<Uniform1dMesher>(0.0, 1.0, 5);
    auto my = ext::make_shared<Uniform1dMesher>(-1.0, 2.0, 4);
    auto mesher2d = ext::make_shared<FdmMesherComposite>(mx, my);
    emitArray("nth_2d_locations0", mesher2d->locations(0));
    emitArray("nth_2d_locations1", mesher2d->locations(1));

    struct Case2d {
        const char* tag;
        Size direction;
        Size order;
        Integer nPoints;
    };
    const Case2d cases2d[] = {
        {"nth_2d_d0_o1_p3", 0, 1, 3},
        {"nth_2d_d1_o1_p3", 1, 1, 3},
        {"nth_2d_d1_o2_p3", 1, 2, 3},
        {"nth_2d_d0_o2_p4", 0, 2, 4},
    };
    for (const auto& c : cases2d) {
        NthOrderDerivativeOp op(c.direction, c.order, c.nPoints, mesher2d);
        emitArray(std::string(c.tag) + "_apply_const", op.apply(constVec(mesher2d, 1.0)));
        emitArray(std::string(c.tag) + "_apply_lin0", op.apply(linVec(mesher2d, 0)));
        emitArray(std::string(c.tag) + "_apply_lin1", op.apply(linVec(mesher2d, 1)));
        emitArray(std::string(c.tag) + "_apply_prod", op.apply(prodVec(mesher2d)));
        emitArray(std::string(c.tag) + "_apply_ramp", op.apply(rampVec(mesher2d)));
    }
    {
        NthOrderDerivativeOp op(1, 1, 3, mesher2d);
        const SparseMatrix m = op.toMatrix();
        emitMatrix("nth_2d_d1_o1_p3_matrix", m);
    }
}

// ===========================================================================
// Block 3: FdmWienerOp
// ===========================================================================
__attribute__((noinline))
void block_fdm_wiener_op(const ext::shared_ptr<YieldTermStructure>& rTS) {
    auto mx = ext::make_shared<Uniform1dMesher>(-1.0, 1.0, 5);
    auto my = ext::make_shared<Uniform1dMesher>(-0.5, 1.5, 4);
    auto mesher = ext::make_shared<FdmMesherComposite>(mx, my);

    Array lambdas = {0.3, 0.5};
    FdmWienerOp op(mesher, rTS, lambdas);
    emit("wiener_size", Real(op.size()));

    op.setTime(0.1, 0.35);

    emitArray("wiener_apply_const", op.apply(constVec(mesher, 1.0)));
    emitArray("wiener_apply_lin0", op.apply(linVec(mesher, 0)));
    emitArray("wiener_apply_lin1", op.apply(linVec(mesher, 1)));
    emitArray("wiener_apply_quad0", op.apply(quadVec(mesher, 0)));
    emitArray("wiener_apply_ramp", op.apply(rampVec(mesher)));
    emitArray("wiener_apply_mixed_ramp", op.apply_mixed(rampVec(mesher)));
    emitArray("wiener_apply_dir0_ramp", op.apply_direction(0, rampVec(mesher)));
    emitArray("wiener_apply_dir1_ramp", op.apply_direction(1, rampVec(mesher)));
    emitArray("wiener_solve_dir0_ramp", op.solve_splitting(0, rampVec(mesher), 0.1));
    emitArray("wiener_solve_dir1_ramp", op.solve_splitting(1, rampVec(mesher), 0.1));
    emitArray("wiener_precond_ramp", op.preconditioner(rampVec(mesher), 0.1));

    const std::vector<SparseMatrix> decomp = op.toMatrixDecomp();
    emit("wiener_decomp_size", Real(decomp.size()));
    emitMatrix("wiener_decomp0", decomp[0]);
    emitMatrix("wiener_decomp1", decomp[1]);

    // --- null rTS variant (1-D) ----------------------------------------
    auto m1 = ext::make_shared<Uniform1dMesher>(-2.0, 2.0, 6);
    auto mesher1 = ext::make_shared<FdmMesherComposite>(m1);
    Array lambdas1 = {0.7};
    FdmWienerOp opNull(mesher1, ext::shared_ptr<YieldTermStructure>(), lambdas1);
    opNull.setTime(0.1, 0.35);
    emit("wiener_null_size", Real(opNull.size()));
    emitArray("wiener_null_apply_const", opNull.apply(constVec(mesher1, 1.0)));
    emitArray("wiener_null_apply_quad", opNull.apply(quadVec(mesher1, 0)));
    emitArray("wiener_null_apply_ramp", opNull.apply(rampVec(mesher1)));
    emitArray("wiener_null_solve_dir0", opNull.solve_splitting(0, rampVec(mesher1), 0.1));
}

// ===========================================================================
// Block 4: Fdm2dBlackScholesOp
// ===========================================================================
__attribute__((noinline))
void block_fdm_2d_black_scholes_op(const Date& today, const DayCounter& dc) {
    auto cal = NullCalendar();

    Handle<Quote> s1(ext::make_shared<SimpleQuote>(100.0));
    Handle<Quote> s2(ext::make_shared<SimpleQuote>(90.0));
    Handle<YieldTermStructure> r1(ext::make_shared<FlatForward>(today, 0.05, dc));
    Handle<YieldTermStructure> q1(ext::make_shared<FlatForward>(today, 0.02, dc));
    Handle<YieldTermStructure> r2(ext::make_shared<FlatForward>(today, 0.05, dc));
    Handle<YieldTermStructure> q2(ext::make_shared<FlatForward>(today, 0.01, dc));
    Handle<BlackVolTermStructure> v1(
        ext::make_shared<BlackConstantVol>(today, cal, 0.25, dc));
    Handle<BlackVolTermStructure> v2(
        ext::make_shared<BlackConstantVol>(today, cal, 0.30, dc));

    auto p1 = ext::make_shared<GeneralizedBlackScholesProcess>(s1, q1, r1, v1);
    auto p2 = ext::make_shared<GeneralizedBlackScholesProcess>(s2, q2, r2, v2);

    auto mx = ext::make_shared<Uniform1dMesher>(std::log(50.0), std::log(200.0), 5);
    auto my = ext::make_shared<Uniform1dMesher>(std::log(45.0), std::log(180.0), 4);
    auto mesher = ext::make_shared<FdmMesherComposite>(mx, my);

    Fdm2dBlackScholesOp op(mesher, p1, p2, 0.4, 1.0);
    emit("bs2d_size", Real(op.size()));

    op.setTime(0.1, 0.35);

    emitArray("bs2d_apply_const", op.apply(constVec(mesher, 1.0)));
    emitArray("bs2d_apply_lin0", op.apply(linVec(mesher, 0)));
    emitArray("bs2d_apply_lin1", op.apply(linVec(mesher, 1)));
    emitArray("bs2d_apply_prod", op.apply(prodVec(mesher)));
    emitArray("bs2d_apply_ramp", op.apply(rampVec(mesher)));
    emitArray("bs2d_apply_mixed_ramp", op.apply_mixed(rampVec(mesher)));
    emitArray("bs2d_apply_mixed_prod", op.apply_mixed(prodVec(mesher)));
    emitArray("bs2d_apply_dir0_ramp", op.apply_direction(0, rampVec(mesher)));
    emitArray("bs2d_apply_dir1_ramp", op.apply_direction(1, rampVec(mesher)));
    emitArray("bs2d_solve_dir0_ramp", op.solve_splitting(0, rampVec(mesher), 0.1));
    emitArray("bs2d_solve_dir1_ramp", op.solve_splitting(1, rampVec(mesher), 0.1));
    emitArray("bs2d_precond_ramp", op.preconditioner(rampVec(mesher), 0.1));

    const std::vector<SparseMatrix> decomp = op.toMatrixDecomp();
    emit("bs2d_decomp_size", Real(decomp.size()));
    emitMatrix("bs2d_decomp0", decomp[0]);
    emitMatrix("bs2d_decomp1", decomp[1]);
    emitMatrix("bs2d_decomp2", decomp[2]);
}

// ===========================================================================
// Block 5: FdmBlackScholesFwdOp
// ===========================================================================
__attribute__((noinline))
void block_fdm_black_scholes_fwd_op(const Date& today, const DayCounter& dc) {
    auto cal = NullCalendar();
    Handle<Quote> s0(ext::make_shared<SimpleQuote>(100.0));
    Handle<YieldTermStructure> rTS(ext::make_shared<FlatForward>(today, 0.05, dc));
    Handle<YieldTermStructure> qTS(ext::make_shared<FlatForward>(today, 0.02, dc));
    Handle<BlackVolTermStructure> volTS(
        ext::make_shared<BlackConstantVol>(today, cal, 0.25, dc));

    auto process = ext::make_shared<GeneralizedBlackScholesProcess>(s0, qTS, rTS, volTS);

    auto m1 = ext::make_shared<Uniform1dMesher>(std::log(50.0), std::log(200.0), 9);
    auto mesher = ext::make_shared<FdmMesherComposite>(m1);
    emitArray("bsfwd_locations", mesher->locations(0));

    // --- constant-vol branch -------------------------------------------
    {
        FdmBlackScholesFwdOp op(mesher, process, 100.0);
        emit("bsfwd_size", Real(op.size()));
        op.setTime(0.1, 0.35);
        emitArray("bsfwd_apply_const", op.apply(constVec(mesher, 1.0)));
        emitArray("bsfwd_apply_lin", op.apply(linVec(mesher, 0)));
        emitArray("bsfwd_apply_quad", op.apply(quadVec(mesher, 0)));
        emitArray("bsfwd_apply_ramp", op.apply(rampVec(mesher)));
        emitArray("bsfwd_apply_mixed_ramp", op.apply_mixed(rampVec(mesher)));
        emitArray("bsfwd_apply_dir0_ramp", op.apply_direction(0, rampVec(mesher)));
        emitArray("bsfwd_apply_dir1_ramp", op.apply_direction(1, rampVec(mesher)));
        emitArray("bsfwd_solve_dir0_ramp", op.solve_splitting(0, rampVec(mesher), 0.1));
        emitArray("bsfwd_solve_dir1_ramp", op.solve_splitting(1, rampVec(mesher), 0.1));
        emitArray("bsfwd_precond_ramp", op.preconditioner(rampVec(mesher), 0.1));
        const std::vector<SparseMatrix> decomp = op.toMatrixDecomp();
        emit("bsfwd_decomp_size", Real(decomp.size()));
        emitMatrix("bsfwd_decomp0", decomp[0]);
    }

    // --- local-vol branch (external, closed-form surface) ---------------
    {
        Handle<LocalVolTermStructure> lv(ext::make_shared<ProbeLocalVol>(today, dc));
        auto lvProcess = ext::make_shared<GeneralizedBlackScholesProcess>(
            s0, qTS, rTS, volTS, lv);
        FdmBlackScholesFwdOp op(mesher, lvProcess, 100.0, true);
        op.setTime(0.1, 0.35);
        emitArray("bsfwd_lv_apply_const", op.apply(constVec(mesher, 1.0)));
        emitArray("bsfwd_lv_apply_lin", op.apply(linVec(mesher, 0)));
        emitArray("bsfwd_lv_apply_ramp", op.apply(rampVec(mesher)));
        emitArray("bsfwd_lv_solve_dir0_ramp", op.solve_splitting(0, rampVec(mesher), 0.1));
    }

    // --- 2-D mesher, direction = 1 --------------------------------------
    {
        auto ma = ext::make_shared<Uniform1dMesher>(0.0, 1.0, 4);
        auto mb = ext::make_shared<Uniform1dMesher>(std::log(50.0), std::log(200.0), 5);
        auto mesher2d = ext::make_shared<FdmMesherComposite>(ma, mb);
        FdmBlackScholesFwdOp op(mesher2d, process, 100.0, false, -Null<Real>(), 1);
        op.setTime(0.1, 0.35);
        emitArray("bsfwd_d1_apply_ramp", op.apply(rampVec(mesher2d)));
        emitArray("bsfwd_d1_apply_dir0_ramp", op.apply_direction(0, rampVec(mesher2d)));
        emitArray("bsfwd_d1_apply_dir1_ramp", op.apply_direction(1, rampVec(mesher2d)));
        emitArray("bsfwd_d1_solve_dir1_ramp", op.solve_splitting(1, rampVec(mesher2d), 0.1));
        emitArray("bsfwd_d1_solve_dir0_ramp", op.solve_splitting(0, rampVec(mesher2d), 0.1));
        emitArray("bsfwd_d1_precond_ramp", op.preconditioner(rampVec(mesher2d), 0.1));
    }
}

// ===========================================================================
// Block 6: FdmLocalVolFwdOp
// ===========================================================================
__attribute__((noinline))
void block_fdm_local_vol_fwd_op(const Date& today, const DayCounter& dc) {
    auto spot = ext::make_shared<SimpleQuote>(100.0);
    auto rTS = ext::make_shared<FlatForward>(today, 0.05, dc);
    auto qTS = ext::make_shared<FlatForward>(today, 0.02, dc);
    auto localVol = ext::make_shared<ProbeLocalVol>(today, dc);

    auto m1 = ext::make_shared<Uniform1dMesher>(std::log(50.0), std::log(200.0), 9);
    auto mesher = ext::make_shared<FdmMesherComposite>(m1);

    FdmLocalVolFwdOp op(mesher, spot, rTS, qTS, localVol);
    emit("lvfwd_size", Real(op.size()));
    op.setTime(0.1, 0.35);
    emitArray("lvfwd_apply_const", op.apply(constVec(mesher, 1.0)));
    emitArray("lvfwd_apply_lin", op.apply(linVec(mesher, 0)));
    emitArray("lvfwd_apply_quad", op.apply(quadVec(mesher, 0)));
    emitArray("lvfwd_apply_ramp", op.apply(rampVec(mesher)));
    emitArray("lvfwd_apply_mixed_ramp", op.apply_mixed(rampVec(mesher)));
    emitArray("lvfwd_apply_dir0_ramp", op.apply_direction(0, rampVec(mesher)));
    emitArray("lvfwd_apply_dir1_ramp", op.apply_direction(1, rampVec(mesher)));
    emitArray("lvfwd_solve_dir0_ramp", op.solve_splitting(0, rampVec(mesher), 0.1));
    emitArray("lvfwd_solve_dir1_ramp", op.solve_splitting(1, rampVec(mesher), 0.1));
    emitArray("lvfwd_precond_ramp", op.preconditioner(rampVec(mesher), 0.1));
    const std::vector<SparseMatrix> decomp = op.toMatrixDecomp();
    emit("lvfwd_decomp_size", Real(decomp.size()));
    emitMatrix("lvfwd_decomp0", decomp[0]);

    // 2-D, direction = 1.
    auto ma = ext::make_shared<Uniform1dMesher>(0.0, 1.0, 4);
    auto mb = ext::make_shared<Uniform1dMesher>(std::log(50.0), std::log(200.0), 5);
    auto mesher2d = ext::make_shared<FdmMesherComposite>(ma, mb);
    FdmLocalVolFwdOp op2(mesher2d, spot, rTS, qTS, localVol, 1);
    op2.setTime(0.1, 0.35);
    emitArray("lvfwd_d1_apply_ramp", op2.apply(rampVec(mesher2d)));
    emitArray("lvfwd_d1_solve_dir1_ramp", op2.solve_splitting(1, rampVec(mesher2d), 0.1));
}

// ===========================================================================
// Block 7: FdmSquareRootFwdOp
// ===========================================================================
void sqrtFwdCase(const std::string& tag,
                 const ext::shared_ptr<FdmMesher>& mesher,
                 Real kappa,
                 Real theta,
                 Real sigma,
                 Size direction,
                 FdmSquareRootFwdOp::TransformationType type) {
    FdmSquareRootFwdOp op(mesher, kappa, theta, sigma, direction, type);
    emit(tag + "_size", Real(op.size()));

    const Size n = mesher->layout()->dim()[direction];
    std::vector<Real> vs;
    for (Size i = 0; i <= n + 1; ++i)
        vs.push_back(op.v(i));
    emitVector(tag + "_v", vs);

    emit(tag + "_lower_boundary_factor", op.lowerBoundaryFactor(type));
    emit(tag + "_upper_boundary_factor", op.upperBoundaryFactor(type));

    op.setTime(0.1, 0.35);  // no-op in C++ -- pinned to prove that.

    emitArray(tag + "_apply_const", op.apply(constVec(mesher, 1.0)));
    emitArray(tag + "_apply_lin", op.apply(linVec(mesher, direction)));
    emitArray(tag + "_apply_quad", op.apply(quadVec(mesher, direction)));
    emitArray(tag + "_apply_ramp", op.apply(rampVec(mesher)));
    emitArray(tag + "_apply_mixed_ramp", op.apply_mixed(rampVec(mesher)));
    emitArray(tag + "_apply_dir_ramp", op.apply_direction(direction, rampVec(mesher)));
    emitArray(tag + "_apply_other_ramp",
              op.apply_direction(direction == 0 ? 1 : 0, rampVec(mesher)));
    emitArray(tag + "_solve_dir_ramp", op.solve_splitting(direction, rampVec(mesher), 0.05));
    emitArray(tag + "_solve_other_ramp",
              op.solve_splitting(direction == 0 ? 1 : 0, rampVec(mesher), 0.05));
    emitArray(tag + "_precond_ramp", op.preconditioner(rampVec(mesher), 0.05));
    const std::vector<SparseMatrix> decomp = op.toMatrixDecomp();
    emit(tag + "_decomp_size", Real(decomp.size()));
    emitMatrix(tag + "_decomp0", decomp[0]);
}

__attribute__((noinline))
void block_fdm_square_root_fwd_op() {
    const Real kappa = 2.5;
    const Real theta = 0.06;
    const Real sigma = 0.4;

    auto mv = ext::make_shared<Uniform1dMesher>(0.005, 0.4, 8);
    auto mesherPlain = ext::make_shared<FdmMesherComposite>(mv);
    emitArray("sqrtfwd_plain_locations", mesherPlain->locations(0));
    sqrtFwdCase("sqrtfwd_plain", mesherPlain, kappa, theta, sigma, 0,
                FdmSquareRootFwdOp::Plain);
    sqrtFwdCase("sqrtfwd_power", mesherPlain, kappa, theta, sigma, 0,
                FdmSquareRootFwdOp::Power);

    auto mvLog = ext::make_shared<Uniform1dMesher>(std::log(0.005), std::log(0.4), 8);
    auto mesherLog = ext::make_shared<FdmMesherComposite>(mvLog);
    emitArray("sqrtfwd_log_locations", mesherLog->locations(0));
    sqrtFwdCase("sqrtfwd_log", mesherLog, kappa, theta, sigma, 0,
                FdmSquareRootFwdOp::Log);

    // 2-D with direction = 1 (the shape FdmHestonFwdOp uses).
    auto mx = ext::make_shared<Uniform1dMesher>(std::log(50.0), std::log(200.0), 4);
    auto mesher2d = ext::make_shared<FdmMesherComposite>(mx, mv);
    sqrtFwdCase("sqrtfwd_2d_plain", mesher2d, kappa, theta, sigma, 1,
                FdmSquareRootFwdOp::Plain);
}

// ===========================================================================
// Block 8: FdmHestonFwdOp
// ===========================================================================
void hestonFwdCase(const std::string& tag,
                   const ext::shared_ptr<FdmMesher>& mesher,
                   const ext::shared_ptr<HestonProcess>& process,
                   FdmSquareRootFwdOp::TransformationType type,
                   const ext::shared_ptr<LocalVolTermStructure>& leverage,
                   Real mixingFactor) {
    FdmHestonFwdOp op(mesher, process, type, leverage, mixingFactor);
    emit(tag + "_size", Real(op.size()));
    op.setTime(0.1, 0.35);

    emitArray(tag + "_apply_const", op.apply(constVec(mesher, 1.0)));
    emitArray(tag + "_apply_lin0", op.apply(linVec(mesher, 0)));
    emitArray(tag + "_apply_lin1", op.apply(linVec(mesher, 1)));
    emitArray(tag + "_apply_prod", op.apply(prodVec(mesher)));
    emitArray(tag + "_apply_ramp", op.apply(rampVec(mesher)));
    emitArray(tag + "_apply_mixed_ramp", op.apply_mixed(rampVec(mesher)));
    emitArray(tag + "_apply_dir0_ramp", op.apply_direction(0, rampVec(mesher)));
    emitArray(tag + "_apply_dir1_ramp", op.apply_direction(1, rampVec(mesher)));
    emitArray(tag + "_solve_dir0_ramp", op.solve_splitting(0, rampVec(mesher), 0.05));
    emitArray(tag + "_solve_dir1_ramp", op.solve_splitting(1, rampVec(mesher), 0.05));
    emitArray(tag + "_precond_ramp", op.preconditioner(rampVec(mesher), 0.05));
    const std::vector<SparseMatrix> decomp = op.toMatrixDecomp();
    emit(tag + "_decomp_size", Real(decomp.size()));
    emitMatrix(tag + "_decomp0", decomp[0]);
    emitMatrix(tag + "_decomp1", decomp[1]);
    emitMatrix(tag + "_decomp2", decomp[2]);
}

__attribute__((noinline))
void block_fdm_heston_fwd_op(const Date& today, const DayCounter& dc) {
    Handle<Quote> s0(ext::make_shared<SimpleQuote>(100.0));
    Handle<YieldTermStructure> rTS(ext::make_shared<FlatForward>(today, 0.05, dc));
    Handle<YieldTermStructure> qTS(ext::make_shared<FlatForward>(today, 0.02, dc));

    auto process = ext::make_shared<HestonProcess>(
        rTS, qTS, s0, 0.05, 2.5, 0.06, 0.4, -0.6);

    auto mx = ext::make_shared<Uniform1dMesher>(std::log(50.0), std::log(200.0), 6);
    auto mv = ext::make_shared<Uniform1dMesher>(0.005, 0.4, 5);
    auto mesher = ext::make_shared<FdmMesherComposite>(mx, mv);
    emitArray("hestonfwd_locations0", mesher->locations(0));
    emitArray("hestonfwd_locations1", mesher->locations(1));

    const ext::shared_ptr<LocalVolTermStructure> noLeverage;

    hestonFwdCase("hestonfwd_plain", mesher, process, FdmSquareRootFwdOp::Plain,
                  noLeverage, 1.0);
    hestonFwdCase("hestonfwd_power", mesher, process, FdmSquareRootFwdOp::Power,
                  noLeverage, 1.0);
    hestonFwdCase("hestonfwd_plain_mix", mesher, process, FdmSquareRootFwdOp::Plain,
                  noLeverage, 0.8);

    auto mvLog = ext::make_shared<Uniform1dMesher>(std::log(0.005), std::log(0.4), 5);
    auto mesherLog = ext::make_shared<FdmMesherComposite>(mx, mvLog);
    emitArray("hestonfwd_log_locations1", mesherLog->locations(1));
    hestonFwdCase("hestonfwd_log", mesherLog, process, FdmSquareRootFwdOp::Log,
                  noLeverage, 1.0);

    // Leverage-function variants.
    ext::shared_ptr<LocalVolTermStructure> leverage =
        ext::make_shared<ProbeLocalVol>(today, dc);
    hestonFwdCase("hestonfwd_plain_lev", mesher, process, FdmSquareRootFwdOp::Plain,
                  leverage, 1.0);
    hestonFwdCase("hestonfwd_power_lev", mesher, process, FdmSquareRootFwdOp::Power,
                  leverage, 1.0);
    hestonFwdCase("hestonfwd_log_lev", mesherLog, process, FdmSquareRootFwdOp::Log,
                  leverage, 1.0);
}

// ===========================================================================
// Block 9: FdmSabrOp
// ===========================================================================
__attribute__((noinline))
void block_fdm_sabr_op(const Date& today, const DayCounter& dc) {
    auto rTS = ext::make_shared<FlatForward>(today, 0.03, dc);

    auto mf = ext::make_shared<Uniform1dMesher>(0.01, 0.2, 6);
    auto ma = ext::make_shared<Uniform1dMesher>(std::log(0.1), std::log(0.6), 5);
    auto mesher = ext::make_shared<FdmMesherComposite>(mf, ma);
    emitArray("sabr_locations0", mesher->locations(0));
    emitArray("sabr_locations1", mesher->locations(1));

    const Real f0 = 0.05;
    const Real alpha = 0.25;
    const Real beta = 0.6;
    const Real nu = 0.4;
    const Real rho = -0.3;

    FdmSabrOp op(mesher, rTS, f0, alpha, beta, nu, rho);
    emit("sabr_size", Real(op.size()));
    op.setTime(0.1, 0.35);

    emitArray("sabr_apply_const", op.apply(constVec(mesher, 1.0)));
    emitArray("sabr_apply_lin0", op.apply(linVec(mesher, 0)));
    emitArray("sabr_apply_lin1", op.apply(linVec(mesher, 1)));
    emitArray("sabr_apply_prod", op.apply(prodVec(mesher)));
    emitArray("sabr_apply_ramp", op.apply(rampVec(mesher)));
    emitArray("sabr_apply_mixed_ramp", op.apply_mixed(rampVec(mesher)));
    emitArray("sabr_apply_dir0_ramp", op.apply_direction(0, rampVec(mesher)));
    emitArray("sabr_apply_dir1_ramp", op.apply_direction(1, rampVec(mesher)));
    emitArray("sabr_solve_dir0_ramp", op.solve_splitting(0, rampVec(mesher), 0.05));
    emitArray("sabr_solve_dir1_ramp", op.solve_splitting(1, rampVec(mesher), 0.05));
    emitArray("sabr_precond_ramp", op.preconditioner(rampVec(mesher), 0.05));

    const std::vector<SparseMatrix> decomp = op.toMatrixDecomp();
    emit("sabr_decomp_size", Real(decomp.size()));
    emitMatrix("sabr_decomp0", decomp[0]);
    emitMatrix("sabr_decomp1", decomp[1]);
    emitMatrix("sabr_decomp2", decomp[2]);
}

// ===========================================================================
// Block 10: FdmCEVOp
// ===========================================================================
__attribute__((noinline))
void block_fdm_cev_op(const Date& today, const DayCounter& dc) {
    auto rTS = ext::make_shared<FlatForward>(today, 0.03, dc);

    const Real f0 = 0.05;
    const Real alpha = 0.3;
    const Real beta = 0.6;

    auto mf = ext::make_shared<Uniform1dMesher>(0.01, 0.2, 7);
    auto mesher = ext::make_shared<FdmMesherComposite>(mf);
    emitArray("cev_locations", mesher->locations(0));

    FdmCEVOp op(mesher, rTS, f0, alpha, beta, 0);
    emit("cev_size", Real(op.size()));
    op.setTime(0.1, 0.35);

    emitArray("cev_apply_const", op.apply(constVec(mesher, 1.0)));
    emitArray("cev_apply_lin", op.apply(linVec(mesher, 0)));
    emitArray("cev_apply_quad", op.apply(quadVec(mesher, 0)));
    emitArray("cev_apply_ramp", op.apply(rampVec(mesher)));
    emitArray("cev_apply_mixed_ramp", op.apply_mixed(rampVec(mesher)));
    emitArray("cev_apply_dir0_ramp", op.apply_direction(0, rampVec(mesher)));
    emitArray("cev_apply_dir1_ramp", op.apply_direction(1, rampVec(mesher)));
    emitArray("cev_solve_dir0_ramp", op.solve_splitting(0, rampVec(mesher), 0.05));
    emitArray("cev_solve_dir1_ramp", op.solve_splitting(1, rampVec(mesher), 0.05));
    emitArray("cev_precond_ramp", op.preconditioner(rampVec(mesher), 0.05));
    const std::vector<SparseMatrix> decomp = op.toMatrixDecomp();
    emit("cev_decomp_size", Real(decomp.size()));
    emitMatrix("cev_decomp0", decomp[0]);

    // 2-D with direction = 1. NOTE: the C++ ctor builds dxxMap_ from
    // SecondDerivativeOp(0, mesher) -- direction 0, *not* `direction` --
    // while scaling by Pow(mesher->locations(direction), 2*beta) and
    // running mapT_ along `direction`. Pinned verbatim.
    auto mt = ext::make_shared<Uniform1dMesher>(0.0, 1.0, 4);
    auto mesher2d = ext::make_shared<FdmMesherComposite>(mt, mf);
    emitArray("cev_2d_locations0", mesher2d->locations(0));
    emitArray("cev_2d_locations1", mesher2d->locations(1));
    FdmCEVOp op2(mesher2d, rTS, f0, alpha, beta, 1);
    op2.setTime(0.1, 0.35);
    emitArray("cev_2d_apply_const", op2.apply(constVec(mesher2d, 1.0)));
    emitArray("cev_2d_apply_ramp", op2.apply(rampVec(mesher2d)));
    emitArray("cev_2d_apply_dir0_ramp", op2.apply_direction(0, rampVec(mesher2d)));
    emitArray("cev_2d_apply_dir1_ramp", op2.apply_direction(1, rampVec(mesher2d)));
    emitArray("cev_2d_solve_dir1_ramp", op2.solve_splitting(1, rampVec(mesher2d), 0.05));
    emitArray("cev_2d_solve_dir0_ramp", op2.solve_splitting(0, rampVec(mesher2d), 0.05));
    emitArray("cev_2d_precond_ramp", op2.preconditioner(rampVec(mesher2d), 0.05));
    const std::vector<SparseMatrix> decomp2 = op2.toMatrixDecomp();
    emitMatrix("cev_2d_decomp0", decomp2[0]);
}

}  // namespace

int main() {
    // showpoint forces a decimal point on every value, so JSON readers
    // parse them as doubles rather than integers. Without it a literal
    // "-0" round-trips through Python's int() and loses its sign bit --
    // which matters, because the Backward scheme's first offset is
    // -(0*h) == -0.0.
    std::cout << std::setprecision(17) << std::showpoint;
    std::cout << "{\n";

    const DayCounter dc = Actual365Fixed();
    const Date today(15, January, 2024);
    const auto rTS = ext::make_shared<FlatForward>(today, 0.04, dc);

    block_numerical_differentiation();
    block_nth_order_derivative_op();
    block_fdm_wiener_op(rTS);
    block_fdm_2d_black_scholes_op(today, dc);
    block_fdm_black_scholes_fwd_op(today, dc);
    block_fdm_local_vol_fwd_op(today, dc);
    block_fdm_square_root_fwd_op();
    block_fdm_heston_fwd_op(today, dc);
    block_fdm_sabr_op(today, dc);
    block_fdm_cev_op(today, dc);

    std::cout << "\n}\n";
    return 0;
}

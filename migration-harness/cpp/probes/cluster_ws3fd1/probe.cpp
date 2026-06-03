// WS3-FD1 cluster probe: legacy finite-difference linear-algebra core.
//
// Cross-validates the retired pre-1.0 QuantLib FD framework that the
// dividend-option helpers depend on. Three of the ported classes still
// exist in C++ v1.42.1 and are exercised here:
//
//   * TridiagonalOperator (ql/methods/finitedifferences/tridiagonaloperator):
//     applyTo(v), solveFor(rhs), SOR(rhs, tol), identity(n), and the
//     operator algebra (+, -, scalar *) used by MixedScheme.step/setStep.
//   * FiniteDifferenceModel + CrankNicolson<TridiagonalOperator>: a short
//     rollback of a known BSM operator from a payoff vector. NOTE: the
//     v1.42.1 MixedScheme/CrankNicolson/FiniteDifferenceModel are the
//     *deprecated* old-FD-framework headers (still compilable in 1.42.1);
//     we instantiate them under QL_DEPRECATED_DISABLE_WARNING.
//   * BSMOperator: the modern bsmoperator.hpp is an empty deprecated stub
//     in 1.42.1, so the operator no longer ships as a class. We reproduce
//     its documented coefficient formula (identical in old-QuantLib and the
//     JQuantLib port) directly here and BUILD a TridiagonalOperator from
//     those coefficients, then emit applyTo/solveFor of that real C++
//     TridiagonalOperator. This validates both the BSM coefficient stencil
//     AND the tridiagonal arithmetic against genuine C++ v1.42.1.
//
// C++ parity: v1.42.1 (099987f0).

#include <ql/math/array.hpp>
#include <ql/methods/finitedifferences/tridiagonaloperator.hpp>
#include <ql/methods/finitedifferences/cranknicolson.hpp>
#include <ql/methods/finitedifferences/finitedifferencemodel.hpp>
#include <ql/methods/finitedifferences/boundarycondition.hpp>

#include <iomanip>
#include <iostream>
#include <vector>

using namespace QuantLib;

namespace {

// emit a JSON array of doubles at full precision.
void emitArray(const Array& a) {
    std::cout << "[";
    for (Size i = 0; i < a.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << a[i];
    }
    std::cout << "]";
}

// Build the BSMOperator's tridiagonal coefficients from the documented
// old-QuantLib / JQuantLib formula (BSMOperator(size, dx, r, q, sigma)):
//   sigma2 = sigma*sigma
//   nu = r - q - sigma2/2
//   pd = -(sigma2/dx - nu)/(2*dx)
//   pu = -(sigma2/dx + nu)/(2*dx)
//   pm =  sigma2/(dx*dx) + r
//   setMidRows(pd, pm, pu)   // first/last rows left at 0
TridiagonalOperator bsmOperator(Size size, Real dx, Real r, Real q, Real sigma) {
    TridiagonalOperator op(size);
    Real sigma2 = sigma * sigma;
    Real nu = r - q - sigma2 / 2.0;
    Real pd = -(sigma2 / dx - nu) / (2.0 * dx);
    Real pu = -(sigma2 / dx + nu) / (2.0 * dx);
    Real pm = sigma2 / (dx * dx) + r;
    op.setMidRows(pd, pm, pu);
    return op;
}

}  // namespace

QL_DEPRECATED_DISABLE_WARNING

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";
    std::cout << "  \"quantlib_version\": \"1.42.1\",\n";

    // ------------------------------------------------------------------
    // TridiagonalOperator: a generic, hand-chosen tridiagonal matrix.
    //   diagonal = [2, 2, 2, 2, 2]
    //   lower    = [-1, -1, -1, -1]
    //   upper    = [-1, -1, -1, -1]
    // v = [1, 2, 3, 4, 5]
    // ------------------------------------------------------------------
    {
        Array low(4, -1.0), mid(5, 2.0), high(4, -1.0);
        TridiagonalOperator T(low, mid, high);
        Array v(5);
        for (Size i = 0; i < 5; ++i) v[i] = static_cast<Real>(i + 1);

        std::cout << "  \"tridiag_generic\": {\n";
        std::cout << "    \"diagonal\": "; emitArray(T.diagonal()); std::cout << ",\n";
        std::cout << "    \"lower\": "; emitArray(T.lowerDiagonal()); std::cout << ",\n";
        std::cout << "    \"upper\": "; emitArray(T.upperDiagonal()); std::cout << ",\n";
        std::cout << "    \"v\": "; emitArray(v); std::cout << ",\n";
        std::cout << "    \"applyTo\": "; emitArray(T.applyTo(v)); std::cout << ",\n";
        std::cout << "    \"solveFor\": "; emitArray(T.solveFor(v)); std::cout << ",\n";
        std::cout << "    \"SOR\": "; emitArray(T.SOR(v, 1e-13)); std::cout << "\n";
        std::cout << "  },\n";
    }

    // ------------------------------------------------------------------
    // identity + operator algebra (used by MixedScheme.setStep):
    //   I = identity(5); A = I + 0.5*T ; B = I + 0.25*T
    // emit A.applyTo(v) and B.solveFor(v) for the same T/v. (Both A and B
    // keep strictly-positive diagonals so solveFor is well-posed.)
    // ------------------------------------------------------------------
    {
        Array low(4, -1.0), mid(5, 2.0), high(4, -1.0);
        TridiagonalOperator T(low, mid, high);
        Array v(5);
        for (Size i = 0; i < 5; ++i) v[i] = static_cast<Real>(i + 1);

        TridiagonalOperator I = TridiagonalOperator::identity(5);
        TridiagonalOperator A = I + 0.5 * T;
        TridiagonalOperator B = I + 0.25 * T;

        std::cout << "  \"algebra\": {\n";
        std::cout << "    \"A_diagonal\": "; emitArray(A.diagonal()); std::cout << ",\n";
        std::cout << "    \"A_lower\": "; emitArray(A.lowerDiagonal()); std::cout << ",\n";
        std::cout << "    \"A_upper\": "; emitArray(A.upperDiagonal()); std::cout << ",\n";
        std::cout << "    \"A_applyTo\": "; emitArray(A.applyTo(v)); std::cout << ",\n";
        std::cout << "    \"B_solveFor\": "; emitArray(B.solveFor(v)); std::cout << "\n";
        std::cout << "  },\n";
    }

    // ------------------------------------------------------------------
    // BSMOperator coefficients (size=7, dx=0.1, r=0.05, q=0.01,
    // sigma=0.20). Built into a real TridiagonalOperator; emit the
    // coefficient bands plus applyTo / solveFor of a payoff-like vector.
    // ------------------------------------------------------------------
    {
        const Size n = 7;
        const Real dx = 0.1, r = 0.05, q = 0.01, sigma = 0.20;
        TridiagonalOperator B = bsmOperator(n, dx, r, q, sigma);

        // a smooth, strictly-non-degenerate test vector.
        Array v(n);
        for (Size i = 0; i < n; ++i) v[i] = static_cast<Real>(i) * 0.5 + 1.0;

        // NOTE: the bare BSM operator has zero first/last diagonal entries
        // (setMidRows only fills the interior), so solveFor is undefined on
        // it. We emit only its coefficient stencil plus applyTo; solveFor is
        // exercised via the `algebra` (I-0.5T) and `fd_rollback` sections,
        // where the identity term makes the diagonal non-degenerate.
        std::cout << "  \"bsm_operator\": {\n";
        std::cout << "    \"n\": " << n << ",\n";
        std::cout << "    \"dx\": " << dx << ", \"r\": " << r
                  << ", \"q\": " << q << ", \"sigma\": " << sigma << ",\n";
        std::cout << "    \"diagonal\": "; emitArray(B.diagonal()); std::cout << ",\n";
        std::cout << "    \"lower\": "; emitArray(B.lowerDiagonal()); std::cout << ",\n";
        std::cout << "    \"upper\": "; emitArray(B.upperDiagonal()); std::cout << ",\n";
        std::cout << "    \"v\": "; emitArray(v); std::cout << ",\n";
        std::cout << "    \"applyTo\": "; emitArray(B.applyTo(v)); std::cout << "\n";
        std::cout << "  },\n";
    }

    // ------------------------------------------------------------------
    // FiniteDifferenceModel rollback: CrankNicolson<TridiagonalOperator>
    // over a BSM operator (same params as above), no boundary conditions,
    // no stopping times. Roll a payoff vector back from t=1.0 to t=0.0 in
    // N=10 steps. Emit the final grid.
    // ------------------------------------------------------------------
    {
        const Size n = 7;
        const Real dx = 0.1, r = 0.05, q = 0.01, sigma = 0.20;
        TridiagonalOperator L = bsmOperator(n, dx, r, q, sigma);

        // bc_set is std::vector<ext::shared_ptr<bc_type>>; empty here.
        typedef BoundaryCondition<TridiagonalOperator> bc_type;
        std::vector<ext::shared_ptr<bc_type> > bcs;

        FiniteDifferenceModel<CrankNicolson<TridiagonalOperator> > model(L, bcs);

        Array payoff(n);
        for (Size i = 0; i < n; ++i) {
            Real x = static_cast<Real>(i) - 3.0;  // centered
            payoff[i] = std::max(x, 0.0);          // a kinked, ramp payoff
        }

        Array a = payoff;  // rolled in place
        model.rollback(a, 1.0, 0.0, 10);

        std::cout << "  \"fd_rollback\": {\n";
        std::cout << "    \"n\": " << n << ",\n";
        std::cout << "    \"from\": 1.0, \"to\": 0.0, \"steps\": 10,\n";
        std::cout << "    \"payoff\": "; emitArray(payoff); std::cout << ",\n";
        std::cout << "    \"result\": "; emitArray(a); std::cout << "\n";
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}

QL_DEPRECATED_ENABLE_WARNING

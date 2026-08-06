// v1.43 methods coverage: FD framework core primitives.
//
// Pins the pieces every other FD operator is built on, so a divergence
// there cannot hide behind a downstream tolerance:
//
//   * FdmLinearOpLayout::neighbourhood — the boundary rule. C++ REFLECTS
//     (coordinate -1 -> +1, coordinate dim -> dim-2); it does not clamp.
//     Emitted for a 1-D grid and for both directions of a 2-D grid, at
//     offsets -2, -1, +1, +2, over every node.
//
//   * FdmLinearOpLayout::neighbourhood (two-direction overload) and
//     iter_neighbourhood, same grids.
//
//   * TripleBandLinearOp on a 2-D mesh, built as
//     FirstDerivativeOp(direction) so the bands are non-trivial at the
//     boundaries: apply(), solve_splitting() along BOTH directions
//     (direction 1 is the case that needs the reverseIndex permutation),
//     mult(), multR(), add(op), add(array), axpyb().
//
//   * ModTripleBandLinearOp: the mutable band accessors, by patching one
//     boundary row and re-applying.
//
// C++ parity:
//   ql/methods/finitedifferences/operators/fdmlinearoplayout.{hpp,cpp}
//   ql/methods/finitedifferences/operators/triplebandlinearop.{hpp,cpp}
//   ql/methods/finitedifferences/operators/modtriplebandlinearop.hpp
//   @ v1.43 (6b57206e0).

#include <ql/methods/finitedifferences/meshers/fdmmeshercomposite.hpp>
#include <ql/methods/finitedifferences/meshers/uniform1dmesher.hpp>
#include <ql/methods/finitedifferences/operators/firstderivativeop.hpp>
#include <ql/methods/finitedifferences/operators/fdmlinearopiterator.hpp>
#include <ql/methods/finitedifferences/operators/fdmlinearoplayout.hpp>
#include <ql/methods/finitedifferences/operators/modtriplebandlinearop.hpp>
#include <ql/methods/finitedifferences/operators/triplebandlinearop.hpp>

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using namespace QuantLib;

namespace {

    std::string fmt(Real x) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.17g", x);
        return buf;
    }

    void emit_reals(const std::string& key, const std::vector<Real>& v, bool& first) {
        if (!first) std::cout << ",\n";
        first = false;
        std::cout << "  \"" << key << "\": [";
        for (Size i = 0; i < v.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << fmt(v[i]);
        }
        std::cout << "]";
    }

    void emit_array(const std::string& key, const Array& a, bool& first) {
        std::vector<Real> v(a.begin(), a.end());
        emit_reals(key, v, first);
    }

    void emit_sizes(const std::string& key, const std::vector<Size>& v, bool& first) {
        if (!first) std::cout << ",\n";
        first = false;
        std::cout << "  \"" << key << "\": [";
        for (Size i = 0; i < v.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << v[i];
        }
        std::cout << "]";
    }

    // Deterministic synthetic vector: distinct, O(1), no symmetry that
    // could mask an index permutation bug.
    Array probe_vector(Size n, Real seed) {
        Array v(n);
        for (Size i = 0; i < n; ++i)
            v[i] = std::sin(seed + 0.37 * Real(i)) + 1.5 + 0.011 * Real(i * i);
        return v;
    }

}

int main() {
    std::cout.precision(17);
    bool first = true;
    std::cout << "{\n";

    // ---------------------------------------------------------------
    // 1-D layout: neighbourhood at every node for offsets -2..+2.
    // ---------------------------------------------------------------
    {
        FdmLinearOpLayout layout(std::vector<Size>{5});
        for (Integer off : {-2, -1, 1, 2}) {
            std::vector<Size> res;
            for (const auto& iter : layout)
                res.push_back(layout.neighbourhood(iter, 0, off));
            emit_sizes("layout_1d_5_neighbourhood_off" + std::to_string(off), res, first);
        }
    }

    // ---------------------------------------------------------------
    // 2-D layout 4x3: neighbourhood along both directions, the
    // two-direction overload, and iter_neighbourhood's coordinates.
    // ---------------------------------------------------------------
    {
        FdmLinearOpLayout layout(std::vector<Size>{4, 3});
        emit_sizes("layout_2d_4x3_dim", layout.dim(), first);
        emit_sizes("layout_2d_4x3_spacing", layout.spacing(), first);

        for (Size dir = 0; dir < 2; ++dir) {
            for (Integer off : {-1, 1}) {
                std::vector<Size> res;
                for (const auto& iter : layout)
                    res.push_back(layout.neighbourhood(iter, dir, off));
                emit_sizes("layout_2d_4x3_nb_dir" + std::to_string(dir) +
                               "_off" + std::to_string(off),
                           res, first);
            }
        }

        for (Integer o0 : {-1, 1}) {
            for (Integer o1 : {-1, 1}) {
                std::vector<Size> res;
                for (const auto& iter : layout)
                    res.push_back(layout.neighbourhood(iter, 0, o0, 1, o1));
                emit_sizes("layout_2d_4x3_nb2_o0" + std::to_string(o0) +
                               "_o1" + std::to_string(o1),
                           res, first);
            }
        }

        // iter_neighbourhood: flat index plus both coordinates.
        {
            std::vector<Size> idx, c0, c1;
            for (const auto& iter : layout) {
                const FdmLinearOpIterator n = layout.iter_neighbourhood(iter, 0, -1);
                idx.push_back(n.index());
                c0.push_back(n.coordinates()[0]);
                c1.push_back(n.coordinates()[1]);
            }
            emit_sizes("layout_2d_4x3_iternb_dir0_offm1_index", idx, first);
            emit_sizes("layout_2d_4x3_iternb_dir0_offm1_c0", c0, first);
            emit_sizes("layout_2d_4x3_iternb_dir0_offm1_c1", c1, first);
        }
    }

    // ---------------------------------------------------------------
    // TripleBandLinearOp on a 2-D non-uniform-ish mesh.
    // FirstDerivativeOp gives non-trivial bands including one-sided
    // boundary rows.
    // ---------------------------------------------------------------
    {
        const auto m0 = ext::make_shared<Uniform1dMesher>(-1.0, 2.0, 4);
        const auto m1 = ext::make_shared<Uniform1dMesher>(0.5, 3.5, 3);
        const auto mesher = ext::make_shared<FdmMesherComposite>(m0, m1);
        const Size n = mesher->layout()->size();

        const Array v = probe_vector(n, 0.3);
        const Array w = probe_vector(n, 1.7);
        emit_array("tb_v", v, first);
        emit_array("tb_w", w, first);

        for (Size dir = 0; dir < 2; ++dir) {
            const FirstDerivativeOp d(dir, mesher);
            const std::string p = "tb_dir" + std::to_string(dir) + "_";

            emit_array(p + "apply", d.apply(v), first);
            emit_array(p + "solve_splitting_a0.4_b1", d.solve_splitting(v, 0.4, 1.0), first);
            emit_array(p + "solve_splitting_am0.25_b2", d.solve_splitting(v, -0.25, 2.0), first);
            emit_array(p + "mult_apply", d.mult(w).apply(v), first);
            emit_array(p + "multR_apply", d.multR(w).apply(v), first);
            emit_array(p + "add_op_apply", d.add(d.mult(w)).apply(v), first);
            emit_array(p + "add_array_apply", d.add(w).apply(v), first);

            // axpyb: self <- a*x + y + b, with x = d.mult(w), y = d.
            {
                TripleBandLinearOp t(dir, mesher);
                const TripleBandLinearOp x = d.mult(w);
                t.axpyb(Array(1, 0.75), x, d, Array(1, -0.2));
                emit_array(p + "axpyb_scalar_apply", t.apply(v), first);

                TripleBandLinearOp t2(dir, mesher);
                t2.axpyb(w, x, d, v);
                emit_array(p + "axpyb_array_apply", t2.apply(v), first);
            }

            // ModTripleBandLinearOp: mutate one boundary row + one
            // interior row, then re-apply.
            {
                ModTripleBandLinearOp mod((TripleBandLinearOp(d)));
                emit_reals(p + "mod_row0",
                           {mod.lower(0), mod.diag(0), mod.upper(0)}, first);
                emit_reals(p + "mod_row_last",
                           {mod.lower(n - 1), mod.diag(n - 1), mod.upper(n - 1)}, first);
                mod.lower(0) = 0.125;
                mod.diag(0) = -2.5;
                mod.upper(0) = 3.75;
                mod.diag(n / 2) += 1.0;
                emit_array(p + "mod_patched_apply", mod.apply(v), first);
                emit_array(p + "mod_patched_solve",
                           mod.solve_splitting(v, 0.3, 1.0), first);
            }
        }
    }

    std::cout << "\n}" << std::endl;
    return 0;
}

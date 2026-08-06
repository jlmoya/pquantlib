// v1.43 reference for QuantLib::detail::ZabrSpecs<Evaluation>
// (ql/math/interpolations/zabrinterpolation.hpp:36-118) and the traits of the
// Zabr<Evaluation> interpolation factory (zabrinterpolation.hpp:169).
//
// ZabrSpecs is the model policy XABRInterpolation<Model> is instantiated with.
// The substantive part is the direct/inverse bijection between the constrained
// 5-parameter box and unconstrained R^5, which lets an unconstrained optimiser
// drive a constrained fit. This probe sweeps BOTH arms of every branch in those
// maps - the saturating arms (|x0| >= 5, |x1| >= sqrt(-log(eps1)),
// |x3| >= 2.5 pi on both signs) are exactly the ones a port drops as
// unreachable.
//
// `guess` is pinned separately because its running index over the random draws
// goes beta, alpha, nu, rho, gamma - not parameter order - and a fixed
// parameter consumes no draw.
//
// Output: one JSON object, all reals at 17 significant digits.

#include <ql/math/interpolations/zabrinterpolation.hpp>
#include <ql/pricingengines/blackformula.hpp>
#include <ql/utilities/null.hpp>

#include <iomanip>
#include <iostream>
#include <vector>

using namespace QuantLib;

namespace {

using Specs = detail::ZabrSpecs<QuantLib::ZabrShortMaturityLognormal>;

void arr(const std::vector<Real>& v) {
    std::cout << "[";
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i != 0) std::cout << ", ";
        std::cout << v[i];
    }
    std::cout << "]";
}

void arr(const Array& v) {
    std::cout << "[";
    for (Size i = 0; i < v.size(); ++i) {
        if (i != 0) std::cout << ", ";
        std::cout << v[i];
    }
    std::cout << "]";
}

Array mk(std::initializer_list<Real> values) {
    Array a(values.size());
    Size i = 0;
    for (const Real v : values)
        a[i++] = v;
    return a;
}

}  // namespace

int main() {
    std::cout << std::setprecision(17);
    Specs specs;

    std::cout << "{\n";
    std::cout << "  \"dimension\": " << specs.dimension() << ",\n";
    std::cout << "  \"eps\": " << specs.eps() << ",\n";
    std::cout << "  \"eps1\": " << specs.eps1() << ",\n";
    std::cout << "  \"eps2\": " << specs.eps2() << ",\n";
    std::cout << "  \"dilation_factor\": " << specs.dilationFactor() << ",\n";
    std::cout << "  \"null_real\": " << Null<Real>() << ",\n";
    // Zabr<Evaluation>::global (zabrinterpolation.hpp) - the traits flag the
    // interpolated-curve machinery branches on.
    std::cout << "  \"zabr_factory_global\": "
              << (Zabr<QuantLib::ZabrShortMaturityLognormal>::global ? "true" : "false") << ",\n";

    // --- defaultValues -----------------------------------------------------
    {
        const Real n = Null<Real>();
        struct Case { Real a, b, nu, rho, g, forward, expiry; };
        const Case cases[] = {
            {n, n, n, n, n, 0.03, 1.0},
            {n, n, n, n, n, 120.0, 5.0},
            {n, 0.99995, n, n, n, 0.03, 1.0},   // beta >= 0.9999 arm
            {n, 0.7, n, n, n, 0.05, 2.0},
            {0.25, n, 0.4, 0.1, 1.3, 0.03, 1.0},
            {n, 0.0, n, n, n, 0.03, 1.0},       // beta == 0 (pow(f, 1) arm)
        };
        std::cout << "  \"default_values\": [\n";
        bool first = true;
        for (const Case& c : cases) {
            std::vector<Real> params = {c.a, c.b, c.nu, c.rho, c.g};
            std::vector<bool> isFixed(5, false);
            std::vector<Real> addParams;
            specs.defaultValues(params, isFixed, c.forward, c.expiry, addParams);
            if (!first) std::cout << ",\n";
            first = false;
            std::cout << "    {\"in\": ";
            arr(std::vector<Real>{c.a, c.b, c.nu, c.rho, c.g});
            std::cout << ", \"forward\": " << c.forward << ", \"expiry\": " << c.expiry
                      << ", \"out\": ";
            arr(params);
            std::cout << "}";
        }
        std::cout << "\n  ],\n";
    }

    // --- guess -------------------------------------------------------------
    {
        struct GCase { bool fixed[5]; Real r[5]; Real forward; };
        const GCase cases[] = {
            {{false, false, false, false, false}, {0.1, 0.2, 0.3, 0.4, 0.5}, 0.03},
            {{false, false, false, false, false}, {0.9, 0.95, 0.5, 0.05, 0.8}, 120.0},
            {{false, false, false, false, false}, {0.1, 0.9995, 0.3, 0.4, 0.2}, 0.03},
            {{true, false, true, false, true}, {0.25, 0.75, 0.5, 0.5, 0.5}, 0.03},
            {{true, true, true, true, true}, {0.5, 0.5, 0.5, 0.5, 0.5}, 0.03},
            {{false, true, false, true, false}, {0.33, 0.66, 0.11, 0.22, 0.44}, 1.0},
        };
        std::cout << "  \"guess\": [\n";
        bool first = true;
        for (const GCase& c : cases) {
            Array values(5, 0.0);
            // Distinct sentinels so untouched slots are visible in the output.
            values[0] = -1.0; values[1] = -2.0; values[2] = -3.0;
            values[3] = -4.0; values[4] = -5.0;
            std::vector<bool> isFixed(c.fixed, c.fixed + 5);
            std::vector<Real> r(c.r, c.r + 5);
            std::vector<Real> addParams;
            specs.guess(values, isFixed, c.forward, 1.0, r, addParams);
            if (!first) std::cout << ",\n";
            first = false;
            std::cout << "    {\"fixed\": [";
            for (int i = 0; i < 5; ++i) {
                if (i != 0) std::cout << ", ";
                std::cout << (c.fixed[i] ? "true" : "false");
            }
            std::cout << "], \"r\": ";
            arr(r);
            std::cout << ", \"forward\": " << c.forward << ", \"out\": ";
            arr(values);
            std::cout << "}";
        }
        std::cout << "\n  ],\n";
    }

    // --- direct: both arms of the x0, x1, x3 branches -----------------------
    {
        // sqrt(-log(eps1)) with eps1 = 1e-7 is ~4.0147; 2.5*pi is ~7.854.
        const std::vector<Array> xs = {
            mk({0.5, 0.5, 0.5, 0.5, 0.5}),
            mk({-0.5, -0.5, -0.5, -0.5, -0.5}),
            mk({4.9, 4.0, 2.0, 7.8, 3.0}),
            mk({5.1, 4.1, -2.0, 7.9, -3.0}),
            mk({-5.1, -4.1, 0.0, -7.9, 0.0}),
            mk({12.0, 10.0, 50.0, 20.0, 50.0}),
            mk({0.0, 0.0, 0.0, 0.0, 0.0}),
        };
        std::cout << "  \"direct\": [\n";
        bool first = true;
        for (const Array& x : xs) {
            const std::vector<bool> isFixed(5, false);
            const std::vector<Real> params(5, 0.0);
            const Array y = specs.direct(x, isFixed, params, 0.03);
            if (!first) std::cout << ",\n";
            first = false;
            std::cout << "    {\"x\": ";
            arr(x);
            std::cout << ", \"y\": ";
            arr(y);
            std::cout << "}";
        }
        std::cout << "\n  ],\n";
    }

    // --- inverse: both arms of the y0 branch, plus the domain edges ---------
    {
        const std::vector<Array> ys = {
            mk({0.25, 0.5, 2.5, 0.5, 0.95}),
            mk({24.9, 0.9, 0.1, -0.5, 0.1}),
            mk({25.5, 0.1, 4.9, 0.99, 1.85}),
            mk({100.0, 0.01, 1.0, -0.99, 1.0}),
            mk({0.04, 0.99, 0.5, 0.0, 0.5}),
        };
        std::cout << "  \"inverse\": [\n";
        bool first = true;
        for (const Array& y : ys) {
            const std::vector<bool> isFixed(5, false);
            const std::vector<Real> params(5, 0.0);
            const Array x = specs.inverse(y, isFixed, params, 0.03);
            if (!first) std::cout << ",\n";
            first = false;
            std::cout << "    {\"y\": ";
            arr(y);
            std::cout << ", \"x\": ";
            arr(x);
            std::cout << "}";
        }
        std::cout << "\n  ],\n";
    }

    // --- weight (Black vega wrt std dev; ZabrSpecs passes no shift) ---------
    {
        struct WCase { Real strike, forward, stdDev; };
        const WCase cases[] = {
            {0.03, 0.03, 0.20},
            {0.01, 0.03, 0.20},
            {0.06, 0.03, 0.35},
            {120.0, 100.0, 0.50},
            {80.0, 100.0, 0.10},
        };
        std::cout << "  \"weight\": [\n";
        bool first = true;
        for (const WCase& c : cases) {
            const std::vector<Real> addParams;
            const Real w = specs.weight(c.strike, c.forward, c.stdDev, addParams);
            if (!first) std::cout << ",\n";
            first = false;
            std::cout << "    {\"strike\": " << c.strike << ", \"forward\": " << c.forward
                      << ", \"std_dev\": " << c.stdDev << ", \"out\": " << w << "}";
        }
        std::cout << "\n  ]\n";
    }

    std::cout << "}\n";
    return 0;
}

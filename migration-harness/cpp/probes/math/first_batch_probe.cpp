// Emit Stage-5 math reference values: closeness predicates, Rounding (all
// 5 types), Factorial, ErrorFunction, betaFunction + incompleteBetaFunction,
// BernsteinPolynomial.get, PascalTriangle.get.

#include <ql/math/bernsteinpolynomial.hpp>
#include <ql/math/beta.hpp>
#include <ql/math/comparison.hpp>
#include <ql/math/errorfunction.hpp>
#include <ql/math/factorial.hpp>
#include <ql/math/pascaltriangle.hpp>
#include <ql/math/rounding.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

using namespace QuantLib;

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    // --- closeness predicates ---------------------------------------------
    {
        std::cout << "  \"closeness\": [\n";
        struct CC { double x; double y; bool default_close; bool default_close_enough; };
        const CC cases[] = {
            {1.0, 1.0,                                       true,  true},
            {0.1 + 0.2, 0.3,                                  true,  true},
            {1.0, 1.0 + 1e-10,                                false, false},
            {1.0, 1.0 + 1e-13,                                true,  true},
            {0.0, 0.0,                                        true,  true},
            {0.0, 1e-25,                                      true,  true},
            {1.0, -1.0,                                       false, false},
            {1e10, 1e10 + 1.0,                                true,  true},
        };
        bool first = true;
        for (const auto& c : cases) {
            if (!first) std::cout << ",\n";
            std::cout << "    {\"x\": " << c.x << ", \"y\": " << c.y
                      << ", \"close\": " << (close(c.x, c.y) ? "true" : "false")
                      << ", \"close_enough\": " << (close_enough(c.x, c.y) ? "true" : "false")
                      << ", \"close_n7\": " << (close(c.x, c.y, 7) ? "true" : "false") << "}";
            first = false;
        }
        std::cout << "\n  ],\n";
    }

    // --- Rounding (each Type) ---------------------------------------------
    {
        std::cout << "  \"rounding\": [\n";
        struct RC { double value; int precision; const char* type_name; int digit; };
        const RC cases[] = {
            // Format: value, precision, type, digit. precision=2, digit=5 default.
            {1.23456, 2, "None",    5},
            {1.23456, 2, "Up",      5},
            {1.23456, 2, "Down",    5},
            {1.23456, 2, "Closest", 5},
            {1.23456, 2, "Floor",   5},
            {1.23456, 2, "Ceiling", 5},
            // Negative
            {-1.23456, 2, "Up",      5},
            {-1.23456, 2, "Down",    5},
            {-1.23456, 2, "Closest", 5},
            {-1.23456, 2, "Floor",   5},
            {-1.23456, 2, "Ceiling", 5},
            // Boundary at digit
            {1.235, 2, "Closest", 5},
            {1.234, 2, "Closest", 5},
            // Higher precision
            {3.14159265, 4, "Closest", 5},
            {3.14159265, 0, "Closest", 5},
            // Custom digit (round at 7 instead of 5)
            {1.276, 2, "Closest", 7},
            {1.275, 2, "Closest", 7},
        };
        auto type_of = [](const char* s) -> Rounding::Type {
            if (std::string(s) == "None")    return Rounding::None;
            if (std::string(s) == "Up")      return Rounding::Up;
            if (std::string(s) == "Down")    return Rounding::Down;
            if (std::string(s) == "Closest") return Rounding::Closest;
            if (std::string(s) == "Floor")   return Rounding::Floor;
            return Rounding::Ceiling;
        };
        bool first = true;
        for (const auto& c : cases) {
            if (!first) std::cout << ",\n";
            Rounding r(c.precision, type_of(c.type_name), c.digit);
            std::cout << "    {\"value\": " << c.value
                      << ", \"precision\": " << c.precision
                      << ", \"type\": \"" << c.type_name << "\""
                      << ", \"digit\": " << c.digit
                      << ", \"result\": " << r(c.value) << "}";
            first = false;
        }
        std::cout << "\n  ],\n";
    }

    // --- Factorial --------------------------------------------------------
    {
        std::cout << "  \"factorial\": {\n    \"get\": [\n";
        bool first = true;
        // 0..27 tabulated, 28..30 via lgamma branch
        for (unsigned n : {0u, 1u, 5u, 10u, 20u, 27u, 28u, 30u, 50u, 100u, 170u}) {
            if (!first) std::cout << ",\n";
            std::cout << "      {\"n\": " << n
                      << ", \"value\": " << Factorial::get(n) << "}";
            first = false;
        }
        std::cout << "\n    ],\n    \"ln\": [\n";
        first = true;
        for (unsigned n : {0u, 1u, 5u, 10u, 20u, 27u, 28u, 30u, 50u, 100u, 170u, 1000u}) {
            if (!first) std::cout << ",\n";
            std::cout << "      {\"n\": " << n
                      << ", \"value\": " << Factorial::ln(n) << "}";
            first = false;
        }
        std::cout << "\n    ]\n  },\n";
    }

    // --- ErrorFunction ----------------------------------------------------
    {
        std::cout << "  \"error_function\": [\n";
        ErrorFunction ef;
        bool first = true;
        for (double x : {-5.0, -3.0, -1.0, -0.5, -0.25, 0.0, 0.1, 0.25, 0.5, 0.84375, 1.0, 1.5, 2.0, 3.0, 5.0}) {
            if (!first) std::cout << ",\n";
            std::cout << "    {\"x\": " << x << ", \"erf\": " << ef(x) << "}";
            first = false;
        }
        std::cout << "\n  ],\n";
    }

    // --- Beta ------------------------------------------------------------
    {
        std::cout << "  \"beta\": {\n    \"beta_function\": [\n";
        bool first = true;
        struct BC { double z; double w; };
        const BC bcases[] = {{1.0, 1.0}, {2.0, 3.0}, {0.5, 0.5}, {5.0, 5.0}, {10.0, 1.0}, {3.5, 2.5}};
        for (const auto& c : bcases) {
            if (!first) std::cout << ",\n";
            std::cout << "      {\"z\": " << c.z << ", \"w\": " << c.w
                      << ", \"value\": " << betaFunction(c.z, c.w) << "}";
            first = false;
        }
        std::cout << "\n    ],\n    \"incomplete_beta\": [\n";
        first = true;
        struct IBC { double a; double b; double x; };
        const IBC ibc[] = {
            {1.0, 1.0, 0.0}, {1.0, 1.0, 1.0}, {1.0, 1.0, 0.5},
            {2.0, 5.0, 0.3}, {0.5, 0.5, 0.25}, {3.0, 7.0, 0.4},
            {10.0, 5.0, 0.7}, {1.5, 2.5, 0.6},
        };
        for (const auto& c : ibc) {
            if (!first) std::cout << ",\n";
            std::cout << "      {\"a\": " << c.a << ", \"b\": " << c.b << ", \"x\": " << c.x
                      << ", \"value\": " << incompleteBetaFunction(c.a, c.b, c.x) << "}";
            first = false;
        }
        std::cout << "\n    ]\n  },\n";
    }

    // --- BernsteinPolynomial ----------------------------------------------
    {
        std::cout << "  \"bernstein\": [\n";
        bool first = true;
        struct BPC { unsigned i; unsigned n; double x; };
        const BPC bp[] = {
            {0, 0, 0.5},
            {0, 1, 0.5}, {1, 1, 0.5},
            {0, 2, 0.5}, {1, 2, 0.5}, {2, 2, 0.5},
            {0, 5, 0.3}, {2, 5, 0.3}, {5, 5, 0.3},
            {3, 10, 0.7},
            {5, 20, 0.25},
        };
        for (const auto& c : bp) {
            if (!first) std::cout << ",\n";
            std::cout << "    {\"i\": " << c.i << ", \"n\": " << c.n << ", \"x\": " << c.x
                      << ", \"value\": " << BernsteinPolynomial::get(c.i, c.n, c.x) << "}";
            first = false;
        }
        std::cout << "\n  ],\n";
    }

    // --- PascalTriangle ---------------------------------------------------
    {
        std::cout << "  \"pascal\": [\n";
        bool first = true;
        for (unsigned order : {0u, 1u, 2u, 3u, 4u, 5u, 10u, 20u, 30u}) {
            if (!first) std::cout << ",\n";
            std::cout << "    {\"order\": " << order << ", \"row\": [";
            const auto& row = PascalTriangle::get(order);
            for (size_t i = 0; i < row.size(); ++i) {
                if (i) std::cout << ", ";
                std::cout << row[i];
            }
            std::cout << "]}";
            first = false;
        }
        std::cout << "\n  ]\n";
    }

    std::cout << "}\n";
    return 0;
}

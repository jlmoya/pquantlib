// L2-A foundations probe — Quote / SimpleQuote / DerivedQuote / CompositeQuote
//
// Captures numeric ground-truth values + diff-from-setValue semantics
// for the L2-A pilot cluster. Abstract bases (TermStructure, Extrapolator,
// Index, BootstrapHelper, BootstrapError) are tested behaviorally in
// Python via mocks — no probe values needed there.
//
// C++ parity: ql/quote.hpp, ql/quotes/simplequote.hpp,
//             ql/quotes/derivedquote.hpp, ql/quotes/compositequote.hpp
//             @ v1.42.1 (099987f0).

#include <ql/quote.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/quotes/derivedquote.hpp>
#include <ql/quotes/compositequote.hpp>
#include <ql/handle.hpp>
#include <ql/shared_ptr.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>

using namespace QuantLib;

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    // === SimpleQuote ===
    // Constructor with value
    SimpleQuote q1(5.0);
    std::cout << "  \"simple_quote_value\": " << q1.value() << ",\n";
    std::cout << "  \"simple_quote_is_valid\": " << (q1.isValid() ? "true" : "false") << ",\n";

    // Default constructor → invalid (Null<Real>())
    SimpleQuote q_default;
    std::cout << "  \"simple_quote_default_is_valid\": " << (q_default.isValid() ? "true" : "false") << ",\n";

    // setValue returns diff
    SimpleQuote q2(2.0);
    Real diff = q2.setValue(7.0);
    std::cout << "  \"simple_quote_set_value_diff\": " << diff << ",\n";
    std::cout << "  \"simple_quote_after_set\": " << q2.value() << ",\n";

    // setValue with same value returns 0 (no observer notification path)
    SimpleQuote q3(3.0);
    Real diff_zero = q3.setValue(3.0);
    std::cout << "  \"simple_quote_set_same_diff\": " << diff_zero << ",\n";

    // reset → invalid
    SimpleQuote q4(4.0);
    q4.reset();
    std::cout << "  \"simple_quote_reset_is_valid\": " << (q4.isValid() ? "true" : "false") << ",\n";

    // === DerivedQuote ===
    // f(x) = 2x + 1 over SimpleQuote(3.0) → 7.0
    auto sq3 = ext::make_shared<SimpleQuote>(3.0);
    Handle<Quote> h3(sq3);
    auto dq_linear = makeDerivedQuote(h3, [](Real x) { return 2.0 * x + 1.0; });
    std::cout << "  \"derived_quote_linear\": " << dq_linear.value() << ",\n";

    // f(x) = x² over SimpleQuote(4.0) → 16.0
    auto sq4 = ext::make_shared<SimpleQuote>(4.0);
    Handle<Quote> h4(sq4);
    auto dq_square = makeDerivedQuote(h4, [](Real x) { return x * x; });
    std::cout << "  \"derived_quote_square\": " << dq_square.value() << ",\n";

    // Update propagation: change underlying, derived recomputes
    sq3->setValue(10.0);
    std::cout << "  \"derived_quote_after_update\": " << dq_linear.value() << ",\n";

    // === CompositeQuote ===
    // f(x, y) = x + y over (3.0, 4.0) → 7.0
    auto sq3a = ext::make_shared<SimpleQuote>(3.0);
    auto sq4a = ext::make_shared<SimpleQuote>(4.0);
    Handle<Quote> h3a(sq3a);
    Handle<Quote> h4a(sq4a);
    auto cq_sum = makeCompositeQuote(h3a, h4a, [](Real x, Real y) { return x + y; });
    std::cout << "  \"composite_quote_sum\": " << cq_sum.value() << ",\n";

    // f(x, y) = sqrt(x²+y²) over (3.0, 4.0) → 5.0
    auto sq3b = ext::make_shared<SimpleQuote>(3.0);
    auto sq4b = ext::make_shared<SimpleQuote>(4.0);
    Handle<Quote> h3b(sq3b);
    Handle<Quote> h4b(sq4b);
    auto cq_hypot = makeCompositeQuote(h3b, h4b,
        [](Real x, Real y) { return std::sqrt(x * x + y * y); });
    std::cout << "  \"composite_quote_hypot\": " << cq_hypot.value() << ",\n";

    // Inspectors
    std::cout << "  \"composite_quote_value1\": " << cq_hypot.value1() << ",\n";
    std::cout << "  \"composite_quote_value2\": " << cq_hypot.value2() << "\n";

    std::cout << "}\n";
    return 0;
}

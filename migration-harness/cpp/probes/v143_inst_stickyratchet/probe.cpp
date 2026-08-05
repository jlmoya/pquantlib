// migration-harness/cpp/probes/v143_inst_stickyratchet/probe.cpp
//
// Reference values for the sticky/ratchet payoff family
// (ql/instruments/stickyratchet.hpp + .cpp).
//
// DoubleStickyRatchetPayoff takes eleven scalars. The six named subclasses are
// nothing but presets of (type1, type2) plus a re-slotting of the caller's
// gearings and spreads into different constructor positions:
//
//   RatchetPayoff   (-1,  0) g1,0,g2  s1,0,s2  iv,0
//   StickyPayoff    (+1,  0) g1,0,g2  s1,0,s2  iv,0
//   RatchetMaxPayoff(-1, -1) g1,g2,g3 s1,s2,s3 iv1,iv2
//   RatchetMinPayoff(-1, +1) ...
//   StickyMaxPayoff (+1, -1) ...
//   StickyMinPayoff (+1, +1) ...
//
// A port that mis-slots a single argument (say, feeding the second gearing to
// gearing2_ instead of gearing3_ for the single-option variants) produces a
// perfectly plausible number. So every one of the eleven scalars is given a
// distinct non-default value here, and each payoff is evaluated over a
// forward ladder that crosses both effective strikes — driving all branches
// of the nested max() — with each subclass ALSO emitted as the equivalent
// explicit DoubleStickyRatchetPayoff, so the argument slotting itself is
// pinned rather than merely the arithmetic.
//
// name() and description() are pinned too: every subclass overrides name(),
// and description() is defined to return name().
//
// The two QL_REQUIRE branches (illegal type1 / type2) are only reachable via
// the base class, and are pinned as {"raises": true}.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/inst/stickyratchet.json.

#include <exception>
#include <iomanip>
#include <iostream>
#include <vector>

#include <ql/instruments/stickyratchet.hpp>

using namespace QuantLib;

namespace {

// Deliberately all distinct, none equal to a plausible default (0 or 1), so
// that any mis-slotted argument moves the answer.
const Real kGearing1 = 0.83;
const Real kGearing2 = 1.17;
const Real kGearing3 = 0.91;
const Real kSpread1 = 0.0043;
const Real kSpread2 = -0.0027;
const Real kSpread3 = 0.0011;
const Real kInitialValue1 = 0.031;
const Real kInitialValue2 = 0.045;
const Real kAccrualFactor = 0.2528;

// Crosses both effective strikes in both directions.
const std::vector<Real> kForwards = {-0.02, 0.0,  0.01, 0.02, 0.025, 0.03,
                                     0.035, 0.04, 0.05, 0.06, 0.12};

void emitPayoff(const char* key, const Payoff& p, bool trailingComma) {
    std::cout << "  \"" << key << "\": {\n"
              << "    \"name\": \"" << p.name() << "\",\n"
              << "    \"description\": \"" << p.description() << "\",\n"
              << "    \"values\": [";
    for (Size i = 0; i < kForwards.size(); ++i) {
        if (i != 0)
            std::cout << ", ";
        std::cout << p(kForwards[i]);
    }
    std::cout << "]\n  }" << (trailingComma ? "," : "") << "\n";
}

void emitRaises(const char* key, Real type1, Real type2, bool trailingComma) {
    DoubleStickyRatchetPayoff p(type1, type2, kGearing1, kGearing2, kGearing3, kSpread1, kSpread2,
                                kSpread3, kInitialValue1, kInitialValue2, kAccrualFactor);
    std::cout << "  \"" << key << "\": {\n"
              << "    \"type1\": " << type1 << ",\n"
              << "    \"type2\": " << type2 << ",\n";
    try {
        (void)p(0.03);
        std::cout << "    \"raises\": false\n";
    } catch (const std::exception& e) {
        std::cout << "    \"raises\": true,\n    \"error\": \"" << e.what() << "\"\n";
    }
    std::cout << "  }" << (trailingComma ? "," : "") << "\n";
}

} // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    std::cout << "  \"forwards\": [";
    for (Size i = 0; i < kForwards.size(); ++i) {
        if (i != 0)
            std::cout << ", ";
        std::cout << kForwards[i];
    }
    std::cout << "],\n";

    std::cout << "  \"inputs\": {\n"
              << "    \"gearing1\": " << kGearing1 << ",\n"
              << "    \"gearing2\": " << kGearing2 << ",\n"
              << "    \"gearing3\": " << kGearing3 << ",\n"
              << "    \"spread1\": " << kSpread1 << ",\n"
              << "    \"spread2\": " << kSpread2 << ",\n"
              << "    \"spread3\": " << kSpread3 << ",\n"
              << "    \"initial_value1\": " << kInitialValue1 << ",\n"
              << "    \"initial_value2\": " << kInitialValue2 << ",\n"
              << "    \"accrual_factor\": " << kAccrualFactor << "\n"
              << "  },\n";

    // --- the base class over the full (type1, type2) grid -----------------
    emitPayoff("double_m1_m1",
               DoubleStickyRatchetPayoff(-1.0, -1.0, kGearing1, kGearing2, kGearing3, kSpread1,
                                         kSpread2, kSpread3, kInitialValue1, kInitialValue2,
                                         kAccrualFactor),
               true);
    emitPayoff("double_m1_p1",
               DoubleStickyRatchetPayoff(-1.0, +1.0, kGearing1, kGearing2, kGearing3, kSpread1,
                                         kSpread2, kSpread3, kInitialValue1, kInitialValue2,
                                         kAccrualFactor),
               true);
    emitPayoff("double_p1_m1",
               DoubleStickyRatchetPayoff(+1.0, -1.0, kGearing1, kGearing2, kGearing3, kSpread1,
                                         kSpread2, kSpread3, kInitialValue1, kInitialValue2,
                                         kAccrualFactor),
               true);
    emitPayoff("double_p1_p1",
               DoubleStickyRatchetPayoff(+1.0, +1.0, kGearing1, kGearing2, kGearing3, kSpread1,
                                         kSpread2, kSpread3, kInitialValue1, kInitialValue2,
                                         kAccrualFactor),
               true);
    emitPayoff("double_m1_zero",
               DoubleStickyRatchetPayoff(-1.0, 0.0, kGearing1, kGearing2, kGearing3, kSpread1,
                                         kSpread2, kSpread3, kInitialValue1, kInitialValue2,
                                         kAccrualFactor),
               true);
    emitPayoff("double_p1_zero",
               DoubleStickyRatchetPayoff(+1.0, 0.0, kGearing1, kGearing2, kGearing3, kSpread1,
                                         kSpread2, kSpread3, kInitialValue1, kInitialValue2,
                                         kAccrualFactor),
               true);
    emitPayoff("double_zero_zero",
               DoubleStickyRatchetPayoff(0.0, 0.0, kGearing1, kGearing2, kGearing3, kSpread1,
                                         kSpread2, kSpread3, kInitialValue1, kInitialValue2,
                                         kAccrualFactor),
               true);

    // --- the six named single/double presets ------------------------------
    // Single-option variants take (g1, g2, s1, s2, initialValue, accrual):
    // the SECOND gearing/spread lands in slot 3, not slot 2.
    emitPayoff("ratchet",
               RatchetPayoff(kGearing1, kGearing3, kSpread1, kSpread3, kInitialValue1,
                             kAccrualFactor),
               true);
    emitPayoff("sticky",
               StickyPayoff(kGearing1, kGearing3, kSpread1, kSpread3, kInitialValue1,
                            kAccrualFactor),
               true);
    emitPayoff("ratchet_max",
               RatchetMaxPayoff(kGearing1, kGearing2, kGearing3, kSpread1, kSpread2, kSpread3,
                                kInitialValue1, kInitialValue2, kAccrualFactor),
               true);
    emitPayoff("ratchet_min",
               RatchetMinPayoff(kGearing1, kGearing2, kGearing3, kSpread1, kSpread2, kSpread3,
                                kInitialValue1, kInitialValue2, kAccrualFactor),
               true);
    emitPayoff("sticky_max",
               StickyMaxPayoff(kGearing1, kGearing2, kGearing3, kSpread1, kSpread2, kSpread3,
                               kInitialValue1, kInitialValue2, kAccrualFactor),
               true);
    emitPayoff("sticky_min",
               StickyMinPayoff(kGearing1, kGearing2, kGearing3, kSpread1, kSpread2, kSpread3,
                               kInitialValue1, kInitialValue2, kAccrualFactor),
               true);

    // --- the illegal-type guards ------------------------------------------
    emitRaises("raises_type1_two", 2.0, 0.0, true);
    emitRaises("raises_type2_half", 1.0, 0.5, true);
    emitRaises("raises_type1_zero_ok", 0.0, 1.0, false);

    std::cout << "}\n";
    return 0;
}

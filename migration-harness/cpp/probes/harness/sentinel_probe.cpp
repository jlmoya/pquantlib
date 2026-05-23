// Sentinel probe — proves the harness:
//   submodule clone → C++ build → probe binary → JSON emission → reference_reader load
// works end-to-end.
//
// No QuantLib semantics under test here; this is harness self-check only.

#include <ql/version.hpp>
#include <cmath>
#include <iomanip>
#include <iostream>

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";
    std::cout << "  \"quantlib_version\": \"" << QL_VERSION << "\",\n";
    std::cout << "  \"quantlib_hex_version\": " << QL_HEX_VERSION << ",\n";
    std::cout << "  \"sqrt_two\": " << std::sqrt(2.0) << ",\n";
    std::cout << "  \"pi\": " << 4.0 * std::atan(1.0) << "\n";
    std::cout << "}\n";
    return 0;
}

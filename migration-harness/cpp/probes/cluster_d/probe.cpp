// L1-D mega-probe: 5 RNGs + BoxMullerGaussianRng (first N values per seed)
// + simple optimization-scaffold static behaviors.

#include <ql/math/randomnumbers/boxmullergaussianrng.hpp>
#include <ql/math/randomnumbers/knuthuniformrng.hpp>
#include <ql/math/randomnumbers/lecuyeruniformrng.hpp>
#include <ql/math/randomnumbers/mt19937uniformrng.hpp>
#include <ql/math/randomnumbers/ranluxuniformrng.hpp>
#include <ql/math/randomnumbers/xoshiro256starstaruniformrng.hpp>

#include <iomanip>
#include <iostream>

using namespace QuantLib;

namespace {

template <typename Rng>
void emit_rng(const char* key, Rng& rng, int n) {
    std::cout << "  \"" << key << "\": [";
    for (int i = 0; i < n; ++i) {
        if (i) std::cout << ", ";
        std::cout << rng.next().value;
    }
    std::cout << "]";
}

}

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    const unsigned long seed = 42;
    const int n_values = 5;

    {
        MersenneTwisterUniformRng rng(seed);
        emit_rng("mt19937", rng, n_values);
        std::cout << ",\n";
    }
    {
        KnuthUniformRng rng(seed);
        emit_rng("knuth", rng, n_values);
        std::cout << ",\n";
    }
    {
        LecuyerUniformRng rng(seed);
        emit_rng("lecuyer", rng, n_values);
        std::cout << ",\n";
    }
    {
        // Ranlux64UniformRng is a template; use Ranlux3UniformRng (the typedef
        // for <223, 24>) as a representative concrete-typedef instance.
        Ranlux3UniformRng rng(seed);
        emit_rng("ranlux3", rng, n_values);
        std::cout << ",\n";
    }
    {
        Xoshiro256StarStarUniformRng rng(seed);
        emit_rng("xoshiro256ss", rng, n_values);
        std::cout << ",\n";
    }
    {
        // BoxMuller over MT19937: emit Gaussian samples.
        MersenneTwisterUniformRng base(seed);
        BoxMullerGaussianRng<MersenneTwisterUniformRng> rng(base);
        std::cout << "  \"box_muller_mt\": [";
        for (int i = 0; i < n_values; ++i) {
            if (i) std::cout << ", ";
            std::cout << rng.next().value;
        }
        std::cout << "]\n";
    }

    std::cout << "}\n";
    return 0;
}

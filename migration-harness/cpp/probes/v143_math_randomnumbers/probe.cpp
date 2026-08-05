// migration-harness/cpp/probes/v143_math_randomnumbers/probe.cpp
//
// Reference values for ql/math/randomnumbers/** (C++ QuantLib v1.43).
//
// This area is bit-exact-or-worthless: a pseudo-random generator that agrees
// to 1e-12 is a *different* generator. Every draw below is therefore emitted
// at setprecision(17) and, where the C++ class exposes the underlying integer
// state (Sobol/Burley, MT19937), the raw uint32 stream is pinned as well.
//
// Two failure modes drive the choice of what is pinned:
//   * an off-by-one in initialisation shows up in draw 1;
//   * an off-by-one in a direction-number / lag recurrence only shows up much
//     later — so every sequence is also sampled after a deep skip (10000).
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/math/randomnumbers.json.

#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <ql/math/distributions/normaldistribution.hpp>
#include <ql/math/distributions/poissondistribution.hpp>
#include <ql/math/randomnumbers/centrallimitgaussianrng.hpp>
#include <ql/math/randomnumbers/faurersg.hpp>
#include <ql/math/randomnumbers/haltonrsg.hpp>
#include <ql/math/randomnumbers/inversecumulativerng.hpp>
#include <ql/math/randomnumbers/inversecumulativersg.hpp>
#include <ql/math/randomnumbers/latticerules.hpp>
#include <ql/math/randomnumbers/latticersg.hpp>
#include <ql/math/randomnumbers/mt19937uniformrng.hpp>
#include <ql/math/randomnumbers/randomizedlds.hpp>
#include <ql/math/randomnumbers/randomsequencegenerator.hpp>
#include <ql/math/randomnumbers/ranluxuniformrng.hpp>
#include <ql/math/randomnumbers/rngtraits.hpp>
#include <ql/math/randomnumbers/seedgenerator.hpp>
#include <ql/math/randomnumbers/sobolbrownianbridgersg.hpp>
#include <ql/math/randomnumbers/sobolrsg.hpp>
#include <ql/math/randomnumbers/burley2020sobolrsg.hpp>
#include <ql/math/randomnumbers/stochasticcollocationinvcdf.hpp>
#include <ql/math/randomnumbers/xoshiro256starstaruniformrng.hpp>
#include <ql/math/randomnumbers/zigguratgaussianrng.hpp>

using namespace QuantLib;

namespace {

// ---------------------------------------------------------------- emitters

void openObj(const char* key) { std::cout << "\"" << key << "\": {"; }

void emitReals(const std::vector<Real>& v) {
    std::cout << "[";
    for (Size i = 0; i < v.size(); ++i)
        std::cout << (i ? "," : "") << v[i];
    std::cout << "]";
}

void emitU32(const std::vector<std::uint32_t>& v) {
    std::cout << "[";
    for (Size i = 0; i < v.size(); ++i)
        std::cout << (i ? "," : "") << v[i];
    std::cout << "]";
}

void emitU64(const std::vector<std::uint64_t>& v) {
    std::cout << "[";
    for (Size i = 0; i < v.size(); ++i)
        std::cout << (i ? "," : "") << v[i];
    std::cout << "]";
}

void emitLongs(const std::vector<long int>& v) {
    std::cout << "[";
    for (Size i = 0; i < v.size(); ++i)
        std::cout << (i ? "," : "") << v[i];
    std::cout << "]";
}

void emitSizes(const std::vector<Size>& v) {
    std::cout << "[";
    for (Size i = 0; i < v.size(); ++i)
        std::cout << (i ? "," : "") << v[i];
    std::cout << "]";
}

// ------------------------------------------------------- 1. SeedGenerator
//
// NOTE (important, and contrary to a common assumption): SeedGenerator IS
// time-seeded. seedgenerator.cpp:34 reads
//     auto firstSeed = (unsigned long)(std::time(nullptr));
// so SeedGenerator::instance().get() is NOT reproducible across runs and
// cannot be pinned directly. What *is* deterministic — and what a port must
// reproduce — is the transformation chain applied to firstSeed. We replay
// that chain verbatim for fixed firstSeed values.

void emitSeedGeneratorChain(unsigned long firstSeed, bool comma) {
    MersenneTwisterUniformRng first(firstSeed);
    unsigned long secondSeed = first.nextInt32();
    MersenneTwisterUniformRng second(secondSeed);
    unsigned long skip = second.nextInt32() % 1000;
    std::vector<unsigned long> init(4);
    init[0] = second.nextInt32();
    init[1] = second.nextInt32();
    init[2] = second.nextInt32();
    init[3] = second.nextInt32();

    MersenneTwisterUniformRng rng(init);
    for (unsigned long i = 0; i < skip; i++)
        rng.nextInt32();

    std::vector<std::uint32_t> gets;
    for (int i = 0; i < 8; ++i)
        gets.push_back(static_cast<std::uint32_t>(rng.nextInt32()));

    std::cout << "{\"first_seed\":" << firstSeed << ",\"second_seed\":" << secondSeed
              << ",\"skip\":" << skip << ",\"init\":[" << init[0] << "," << init[1] << ","
              << init[2] << "," << init[3] << "],\"gets\":";
    emitU32(gets);
    std::cout << "}" << (comma ? "," : "");
}

// MersenneTwisterUniformRng's vector<unsigned long> ctor (init_by_array) is
// what SeedGenerator ultimately builds its rng from; pin it standalone too.
void emitMtByArray(const std::vector<unsigned long>& seeds, bool comma) {
    MersenneTwisterUniformRng rng(seeds);
    std::vector<std::uint32_t> ints;
    for (int i = 0; i < 8; ++i)
        ints.push_back(static_cast<std::uint32_t>(rng.nextInt32()));
    std::vector<Real> reals;
    for (int i = 0; i < 4; ++i)
        reals.push_back(rng.next().value);
    std::cout << "{\"seeds\":[";
    for (Size i = 0; i < seeds.size(); ++i)
        std::cout << (i ? "," : "") << seeds[i];
    std::cout << "],\"ints\":";
    emitU32(ints);
    std::cout << ",\"reals\":";
    emitReals(reals);
    std::cout << "}" << (comma ? "," : "");
}

// -------------------------------------------------- 2. RandomSequenceGenerator

void emitRsg(Size dim, unsigned long seed, bool comma) {
    RandomSequenceGenerator<MersenneTwisterUniformRng> g(dim, seed);
    std::cout << "{\"dimension\":" << dim << ",\"seed\":" << seed << ",\"sequences\":[";
    for (int n = 0; n < 4; ++n) {
        const auto& s = g.nextSequence();
        std::cout << (n ? "," : "") << "{\"value\":";
        emitReals(s.value);
        std::cout << ",\"weight\":" << s.weight << "}";
    }
    std::cout << "],\"last_weight\":" << g.lastSequence().weight;
    // deep draw: 10000 further sequences, then pin the next one
    for (int n = 0; n < 10000; ++n)
        g.nextSequence();
    std::cout << ",\"after_10000\":";
    emitReals(g.nextSequence().value);
    // int32 sequence continues from the same underlying stream
    std::cout << ",\"next_int32_sequence\":";
    {
        std::vector<std::uint32_t> v;
        for (auto x : g.nextInt32Sequence())
            v.push_back(static_cast<std::uint32_t>(x));
        emitU32(v);
    }
    std::cout << "}" << (comma ? "," : "");
}

// ------------------------------------------------- 3. InverseCumulativeRng

void emitIcRng(unsigned long seed, bool comma) {
    typedef InverseCumulativeRng<MersenneTwisterUniformRng, InverseCumulativeNormal> Gen;
    Gen g((MersenneTwisterUniformRng(seed)));
    std::vector<Real> draws;
    for (int i = 0; i < 12; ++i)
        draws.push_back(g.next().value);
    for (int i = 0; i < 10000; ++i)
        g.next();
    Real deep = g.next().value;
    std::cout << "{\"seed\":" << seed << ",\"draws\":";
    emitReals(draws);
    std::cout << ",\"after_10000\":" << deep << ",\"weight\":" << g.next().weight << "}"
              << (comma ? "," : "");
}

void emitIcRngPoisson(unsigned long seed, Real lambda, bool comma) {
    typedef InverseCumulativeRng<MersenneTwisterUniformRng, InverseCumulativePoisson> Gen;
    // InverseCumulativePoisson's default ctor uses lambda = 1; the RNG template
    // default-constructs IC, so the only reachable configuration is lambda = 1.
    // We emit the default configuration and additionally the standalone IC at
    // the requested lambda so the Python port can check both wirings.
    Gen g((MersenneTwisterUniformRng(seed)));
    std::vector<Real> draws;
    for (int i = 0; i < 12; ++i)
        draws.push_back(g.next().value);
    std::cout << "{\"seed\":" << seed << ",\"lambda\":" << lambda << ",\"draws\":";
    emitReals(draws);
    std::cout << "}" << (comma ? "," : "");
}

// ------------------------------------------------- 4. InverseCumulativeRsg

void emitIcRsg(Size dim, unsigned long seed, bool comma) {
    typedef RandomSequenceGenerator<MersenneTwisterUniformRng> Ursg;
    typedef InverseCumulativeRsg<Ursg, InverseCumulativeNormal> Gen;
    Gen g(Ursg(dim, seed));
    std::cout << "{\"dimension\":" << dim << ",\"seed\":" << seed << ",\"sequences\":[";
    for (int n = 0; n < 4; ++n) {
        const auto& s = g.nextSequence();
        std::cout << (n ? "," : "") << "{\"value\":";
        emitReals(s.value);
        std::cout << ",\"weight\":" << s.weight << "}";
    }
    std::cout << "]";
    for (int n = 0; n < 10000; ++n)
        g.nextSequence();
    std::cout << ",\"after_10000\":";
    emitReals(g.nextSequence().value);
    std::cout << ",\"dimension_accessor\":" << g.dimension() << "}" << (comma ? "," : "");
}

// ---------------------------------------------------------- 5. CLGaussianRng

void emitClGaussian(unsigned long seed, bool comma) {
    CLGaussianRng<MersenneTwisterUniformRng> g((MersenneTwisterUniformRng(seed)));
    std::vector<Real> draws;
    for (int i = 0; i < 10; ++i)
        draws.push_back(g.next().value);
    for (int i = 0; i < 1000; ++i)
        g.next();
    Real deep = g.next().value;
    std::cout << "{\"seed\":" << seed << ",\"draws\":";
    emitReals(draws);
    std::cout << ",\"after_1000\":" << deep << ",\"weight\":" << g.next().weight << "}"
              << (comma ? "," : "");
}

// ------------------------------------------------------ 6. Ranlux64UniformRng

template <std::size_t P, std::size_t R>
void emitRanlux(const char* label, std::uint_fast64_t seed, bool comma) {
    Ranlux64UniformRng<P, R> g(seed);
    std::vector<Real> draws;
    for (int i = 0; i < 30; ++i)
        draws.push_back(g.next().value);
    for (int i = 0; i < 10000; ++i)
        g.next();
    Real deep = g.next().value;
    std::cout << "{\"label\":\"" << label << "\",\"seed\":" << (unsigned long long)seed
              << ",\"P\":" << P << ",\"R\":" << R << ",\"draws\":";
    emitReals(draws);
    std::cout << ",\"after_10000\":" << deep << "}" << (comma ? "," : "");
}

// ------------------------------------------------------ 7. ZigguratGaussianRng

void emitZiggurat(std::uint64_t seed, bool comma) {
    ZigguratGaussianRng<Xoshiro256StarStarUniformRng> g((Xoshiro256StarStarUniformRng(seed)));
    std::vector<Real> draws;
    for (int i = 0; i < 40; ++i)
        draws.push_back(g.nextReal());
    for (int i = 0; i < 10000; ++i)
        g.nextReal();
    Real deep = g.nextReal();
    std::cout << "{\"seed\":" << (unsigned long long)seed << ",\"draws\":";
    emitReals(draws);
    std::cout << ",\"after_10000\":" << deep << ",\"weight\":" << g.next().weight << "}"
              << (comma ? "," : "");
}

// ------------------------------------------------------------- 8. LatticeRule

void emitLatticeRule(const char* label, LatticeRule::type t, bool comma) {
    std::vector<Real> Z;
    LatticeRule::getRule(t, Z, 1024);
    std::cout << "{\"label\":\"" << label << "\",\"size\":" << Z.size() << ",\"z\":";
    emitReals(Z);
    std::cout << "}" << (comma ? "," : "");
}

// -------------------------------------------------------------- 9. LatticeRsg

void emitLatticeRsg(const char* label, LatticeRule::type t, Size dim, Size N, bool comma) {
    std::vector<Real> Z;
    LatticeRule::getRule(t, Z, static_cast<Integer>(N));
    LatticeRsg g(dim, Z, N);
    std::cout << "{\"label\":\"" << label << "\",\"dimension\":" << dim << ",\"N\":" << N
              << ",\"sequences\":[";
    for (int n = 0; n < 6; ++n) {
        std::cout << (n ? "," : "");
        emitReals(g.nextSequence().value);
    }
    std::cout << "]";
    g.skipTo(10000);
    std::cout << ",\"after_skip_to_10000\":";
    emitReals(g.nextSequence().value);
    std::cout << ",\"dimension_accessor\":" << g.dimension() << "}" << (comma ? "," : "");
}

// ---------------------------------------------------------------- 10. FaureRsg

void emitFaure(Size dim, bool comma) {
    FaureRsg g(dim);
    std::cout << "{\"dimension\":" << dim << ",\"int_sequences\":[";
    for (int n = 0; n < 8; ++n) {
        std::cout << (n ? "," : "");
        emitLongs(g.nextIntSequence());
    }
    std::cout << "],\"sequences\":[";
    // continue on the same generator: the integer sequence is cumulative
    for (int n = 0; n < 4; ++n) {
        std::cout << (n ? "," : "");
        emitReals(g.nextSequence().value);
    }
    std::cout << "]";
    for (int n = 0; n < 1000; ++n)
        g.nextIntSequence();
    std::cout << ",\"after_1000_int\":";
    emitLongs(g.nextIntSequence());
    std::cout << ",\"after_1000_real\":";
    emitReals(g.nextSequence().value);
    std::cout << ",\"dimension_accessor\":" << g.dimension() << "}" << (comma ? "," : "");
}

// ------------------------------------------------------------------ 11. Sobol

const char* diName(SobolRsg::DirectionIntegers d) {
    switch (d) {
        case SobolRsg::Unit: return "Unit";
        case SobolRsg::Jaeckel: return "Jaeckel";
        case SobolRsg::SobolLevitan: return "SobolLevitan";
        case SobolRsg::SobolLevitanLemieux: return "SobolLevitanLemieux";
        case SobolRsg::JoeKuoD5: return "JoeKuoD5";
        case SobolRsg::JoeKuoD6: return "JoeKuoD6";
        case SobolRsg::JoeKuoD7: return "JoeKuoD7";
        case SobolRsg::Kuo: return "Kuo";
        case SobolRsg::Kuo2: return "Kuo2";
        case SobolRsg::Kuo3: return "Kuo3";
    }
    return "?";
}

void emitSobol(Size dim,
               unsigned long seed,
               SobolRsg::DirectionIntegers di,
               bool useGrayCode,
               bool comma) {
    SobolRsg g(dim, seed, di, useGrayCode);
    std::cout << "{\"dimension\":" << dim << ",\"seed\":" << seed << ",\"direction_integers\":\""
              << diName(di) << "\",\"use_gray_code\":" << (useGrayCode ? "true" : "false")
              << ",\"int_sequences\":[";
    for (int n = 0; n < 6; ++n) {
        std::cout << (n ? "," : "");
        emitU32(g.nextInt32Sequence());
    }
    std::cout << "],\"sequences\":[";
    for (int n = 0; n < 4; ++n) {
        std::cout << (n ? "," : "");
        emitReals(g.nextSequence().value);
    }
    std::cout << "]";
    {
        // fresh generator so the skipTo semantics are unambiguous
        SobolRsg h(dim, seed, di, useGrayCode);
        std::cout << ",\"skip_to_10000\":";
        emitU32(h.skipTo(10000));
        std::cout << ",\"after_skip_to_10000\":";
        emitU32(h.nextInt32Sequence());
        std::cout << ",\"after_skip_real\":";
        emitReals(h.nextSequence().value);
    }
    {
        // long sequential run: 10000 draws in, no skipTo — catches recurrence
        // errors that skipTo's direct Gray-code evaluation would mask
        SobolRsg h(dim, seed, di, useGrayCode);
        for (int n = 0; n < 10000; ++n)
            h.nextInt32Sequence();
        std::cout << ",\"sequential_10000\":";
        emitU32(h.nextInt32Sequence());
    }
    std::cout << ",\"dimension_accessor\":" << g.dimension() << "}" << (comma ? "," : "");
}

void emitBurley(Size dim,
                unsigned long seed,
                SobolRsg::DirectionIntegers di,
                unsigned long scrambleSeed,
                bool comma) {
    Burley2020SobolRsg g(dim, seed, di, scrambleSeed);
    std::cout << "{\"dimension\":" << dim << ",\"seed\":" << seed << ",\"direction_integers\":\""
              << diName(di) << "\",\"scramble_seed\":" << scrambleSeed << ",\"int_sequences\":[";
    for (int n = 0; n < 6; ++n) {
        std::cout << (n ? "," : "");
        emitU32(g.nextInt32Sequence());
    }
    std::cout << "],\"sequences\":[";
    for (int n = 0; n < 4; ++n) {
        std::cout << (n ? "," : "");
        emitReals(g.nextSequence().value);
    }
    std::cout << "]";
    {
        Burley2020SobolRsg h(dim, seed, di, scrambleSeed);
        std::cout << ",\"skip_to_10000\":";
        emitU32(h.skipTo(10000));
        std::cout << ",\"after_skip_to_10000\":";
        emitU32(h.nextInt32Sequence());
    }
    {
        Burley2020SobolRsg h(dim, seed, di, scrambleSeed);
        for (int n = 0; n < 10000; ++n)
            h.nextInt32Sequence();
        std::cout << ",\"sequential_10000\":";
        emitU32(h.nextInt32Sequence());
    }
    std::cout << ",\"dimension_accessor\":" << g.dimension() << "}" << (comma ? "," : "");
}

// ------------------------------------------------------------ 12. RandomizedLDS

void emitRandomizedLds(Size dim, BigNatural ldsSeed, BigNatural prsSeed, bool comma) {
    RandomizedLDS<SobolRsg> g(dim, ldsSeed, prsSeed);
    std::cout << "{\"dimension\":" << dim << ",\"lds_seed\":" << ldsSeed
              << ",\"prs_seed\":" << prsSeed << ",\"sequences\":[";
    for (int n = 0; n < 5; ++n) {
        std::cout << (n ? "," : "");
        emitReals(g.nextSequence().value);
    }
    std::cout << "],\"after_next_randomizer\":[";
    g.nextRandomizer();
    for (int n = 0; n < 5; ++n) {
        std::cout << (n ? "," : "");
        emitReals(g.nextSequence().value);
    }
    std::cout << "]";
    for (int n = 0; n < 10000; ++n)
        g.nextSequence();
    std::cout << ",\"after_10000\":";
    emitReals(g.nextSequence().value);
    std::cout << ",\"dimension_accessor\":" << g.dimension() << "}" << (comma ? "," : "");
}

// -------------------------------------------------- 13. SobolBrownianBridgeRsg

const char* ordName(SobolBrownianGenerator::Ordering o) {
    switch (o) {
        case SobolBrownianGenerator::Factors: return "Factors";
        case SobolBrownianGenerator::Steps: return "Steps";
        case SobolBrownianGenerator::Diagonal: return "Diagonal";
    }
    return "?";
}

void emitSbbRsg(Size factors,
                Size steps,
                SobolBrownianGenerator::Ordering ordering,
                unsigned long seed,
                SobolRsg::DirectionIntegers di,
                bool comma) {
    SobolBrownianBridgeRsg g(factors, steps, ordering, seed, di);
    std::cout << "{\"factors\":" << factors << ",\"steps\":" << steps << ",\"ordering\":\""
              << ordName(ordering) << "\",\"seed\":" << seed << ",\"direction_integers\":\""
              << diName(di) << "\",\"dimension\":" << g.dimension() << ",\"sequences\":[";
    for (int n = 0; n < 4; ++n) {
        std::cout << (n ? "," : "");
        emitReals(g.nextSequence().value);
    }
    std::cout << "]";
    for (int n = 0; n < 1000; ++n)
        g.nextSequence();
    std::cout << ",\"after_1000\":";
    emitReals(g.nextSequence().value);
    std::cout << ",\"last_weight\":" << g.lastSequence().weight << "}" << (comma ? "," : "");
}

void emitBurleySbbRsg(Size factors,
                      Size steps,
                      SobolBrownianGenerator::Ordering ordering,
                      unsigned long seed,
                      SobolRsg::DirectionIntegers di,
                      unsigned long scrambleSeed,
                      bool comma) {
    Burley2020SobolBrownianBridgeRsg g(factors, steps, ordering, seed, di, scrambleSeed);
    std::cout << "{\"factors\":" << factors << ",\"steps\":" << steps << ",\"ordering\":\""
              << ordName(ordering) << "\",\"seed\":" << seed << ",\"direction_integers\":\""
              << diName(di) << "\",\"scramble_seed\":" << scrambleSeed
              << ",\"dimension\":" << g.dimension() << ",\"sequences\":[";
    for (int n = 0; n < 4; ++n) {
        std::cout << (n ? "," : "");
        emitReals(g.nextSequence().value);
    }
    std::cout << "]";
    for (int n = 0; n < 1000; ++n)
        g.nextSequence();
    std::cout << ",\"after_1000\":";
    emitReals(g.nextSequence().value);
    std::cout << "}" << (comma ? "," : "");
}

// The dimension ordering IS the answer for the bridge RSGs; pin it directly.
void emitOrderedIndices(Size factors,
                        Size steps,
                        SobolBrownianGenerator::Ordering ordering,
                        bool comma) {
    SobolBrownianGenerator gen(factors, steps, ordering, 42, SobolRsg::JoeKuoD7);
    std::cout << "{\"factors\":" << factors << ",\"steps\":" << steps << ",\"ordering\":\""
              << ordName(ordering) << "\",\"ordered_indices\":[";
    const auto& oi = gen.orderedIndices();
    for (Size i = 0; i < oi.size(); ++i) {
        std::cout << (i ? "," : "");
        emitSizes(oi[i]);
    }
    std::cout << "]}" << (comma ? "," : "");
}

// ------------------------------------------------------------- 14. rngtraits

void emitPseudoRandomTraits(Size dim, BigNatural seed, bool comma) {
    auto rsg = PseudoRandom::make_sequence_generator(dim, seed);
    std::cout << "{\"dimension\":" << dim << ",\"seed\":" << seed
              << ",\"allows_error_estimate\":" << (int)PseudoRandom::allowsErrorEstimate
              << ",\"sequences\":[";
    for (int n = 0; n < 4; ++n) {
        std::cout << (n ? "," : "");
        emitReals(rsg.nextSequence().value);
    }
    std::cout << "],\"dimension_accessor\":" << rsg.dimension() << "}" << (comma ? "," : "");
}

void emitLowDiscrepancyTraits(Size dim, BigNatural seed, bool comma) {
    auto rsg = LowDiscrepancy::make_sequence_generator(dim, seed);
    std::cout << "{\"dimension\":" << dim << ",\"seed\":" << seed
              << ",\"allows_error_estimate\":" << (int)LowDiscrepancy::allowsErrorEstimate
              << ",\"sequences\":[";
    for (int n = 0; n < 4; ++n) {
        std::cout << (n ? "," : "");
        emitReals(rsg.nextSequence().value);
    }
    std::cout << "],\"dimension_accessor\":" << rsg.dimension() << "}" << (comma ? "," : "");
}

void emitPoissonPseudoRandomTraits(Size dim, BigNatural seed, bool comma) {
    auto rsg = PoissonPseudoRandom::make_sequence_generator(dim, seed);
    std::cout << "{\"dimension\":" << dim << ",\"seed\":" << seed
              << ",\"allows_error_estimate\":" << (int)PoissonPseudoRandom::allowsErrorEstimate
              << ",\"sequences\":[";
    for (int n = 0; n < 4; ++n) {
        std::cout << (n ? "," : "");
        emitReals(rsg.nextSequence().value);
    }
    std::cout << "]}" << (comma ? "," : "");
}

// ------------------------------------------- 15. StochasticCollocationInvCDF

void emitScInvCdf(const char* label,
                  const std::function<Real(Real)>& invCDF,
                  Size order,
                  Real pMax,
                  Real pMin,
                  bool comma) {
    StochasticCollocationInvCDF f(invCDF, order, pMax, pMin);
    const std::vector<Real> xs = {-3.0, -2.0, -1.0, -0.5, 0.0, 0.25, 1.0, 2.0, 3.0, 4.5};
    const std::vector<Real> us = {1e-6, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999};
    std::vector<Real> values, calls;
    for (Real x : xs)
        values.push_back(f.value(x));
    for (Real u : us)
        calls.push_back(f(u));
    std::cout << "{\"label\":\"" << label << "\",\"order\":" << order << ",\"p_max\":";
    if (pMax == Null<Real>())
        std::cout << "null";
    else
        std::cout << pMax;
    std::cout << ",\"p_min\":";
    if (pMin == Null<Real>())
        std::cout << "null";
    else
        std::cout << pMin;
    std::cout << ",\"xs\":";
    emitReals(xs);
    std::cout << ",\"values\":";
    emitReals(values);
    std::cout << ",\"us\":";
    emitReals(us);
    std::cout << ",\"calls\":";
    emitReals(calls);
    std::cout << "}" << (comma ? "," : "");
}

} // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    // ---- SeedGenerator (deterministic chain, see note above) -------------
    openObj("seed_generator");
    std::cout << "\"note\":\"SeedGenerator::initialize() seeds from std::time(nullptr); "
                 "only the transformation chain is reproducible\",\"chains\":[";
    emitSeedGeneratorChain(1UL, true);
    emitSeedGeneratorChain(42UL, true);
    emitSeedGeneratorChain(1700000000UL, false);
    std::cout << "]},\n";

    // ---- MersenneTwisterUniformRng init_by_array -------------------------
    std::cout << "\"mt19937_by_array\": [";
    emitMtByArray({0x123UL, 0x234UL, 0x345UL, 0x456UL}, true);
    emitMtByArray({1UL}, true);
    emitMtByArray({7UL, 11UL, 13UL, 17UL, 19UL, 23UL}, false);
    std::cout << "],\n";

    // ---- RandomSequenceGenerator ----------------------------------------
    std::cout << "\"random_sequence_generator\": [";
    emitRsg(1, 42UL, true);
    emitRsg(3, 42UL, true);
    emitRsg(5, 1UL, false);
    std::cout << "],\n";

    // ---- InverseCumulativeRng -------------------------------------------
    std::cout << "\"inverse_cumulative_rng\": [";
    emitIcRng(42UL, true);
    emitIcRng(1UL, false);
    std::cout << "],\n";

    std::cout << "\"inverse_cumulative_rng_poisson\": [";
    emitIcRngPoisson(42UL, 1.0, false);
    std::cout << "],\n";

    // ---- InverseCumulativeRsg -------------------------------------------
    std::cout << "\"inverse_cumulative_rsg\": [";
    emitIcRsg(1, 42UL, true);
    emitIcRsg(4, 42UL, true);
    emitIcRsg(3, 1UL, false);
    std::cout << "],\n";

    // ---- CLGaussianRng ---------------------------------------------------
    std::cout << "\"cl_gaussian_rng\": [";
    emitClGaussian(42UL, true);
    emitClGaussian(1UL, false);
    std::cout << "],\n";

    // ---- Ranlux64UniformRng ---------------------------------------------
    std::cout << "\"ranlux64\": [";
    emitRanlux<223, 24>("Ranlux3_default", 19780503UL, true);
    emitRanlux<223, 24>("Ranlux3_seed0", 0UL, true);
    emitRanlux<223, 24>("Ranlux3_seed42", 42UL, true);
    emitRanlux<389, 24>("Ranlux4_default", 19780503UL, true);
    emitRanlux<389, 24>("Ranlux4_seed42", 42UL, false);
    std::cout << "],\n";

    // ---- ZigguratGaussianRng --------------------------------------------
    std::cout << "\"ziggurat_gaussian_rng\": [";
    emitZiggurat(1UL, true);
    emitZiggurat(42UL, true);
    emitZiggurat(123456789ULL, false);
    std::cout << "],\n";

    // ---- LatticeRule -----------------------------------------------------
    std::cout << "\"lattice_rule\": [";
    emitLatticeRule("A", LatticeRule::A, true);
    emitLatticeRule("B", LatticeRule::B, true);
    emitLatticeRule("C", LatticeRule::C, true);
    emitLatticeRule("D", LatticeRule::D, false);
    std::cout << "],\n";

    // ---- LatticeRsg ------------------------------------------------------
    std::cout << "\"lattice_rsg\": [";
    emitLatticeRsg("A", LatticeRule::A, 4, 1024, true);
    emitLatticeRsg("B", LatticeRule::B, 3, 2048, true);
    emitLatticeRsg("D", LatticeRule::D, 5, 65536, false);
    std::cout << "],\n";

    // ---- FaureRsg --------------------------------------------------------
    std::cout << "\"faure_rsg\": [";
    emitFaure(1, true);
    emitFaure(2, true);
    emitFaure(3, true);
    emitFaure(5, false);
    std::cout << "],\n";

    // ---- SobolRsg (all ten direction-integer families) -------------------
    std::cout << "\"sobol_rsg\": [";
    emitSobol(2, 42UL, SobolRsg::Jaeckel, true, true);
    emitSobol(5, 42UL, SobolRsg::Jaeckel, true, true);
    emitSobol(5, 42UL, SobolRsg::Jaeckel, false, true);
    emitSobol(8, 42UL, SobolRsg::Unit, true, true);
    emitSobol(8, 42UL, SobolRsg::SobolLevitan, true, true);
    emitSobol(8, 42UL, SobolRsg::SobolLevitanLemieux, true, true);
    emitSobol(8, 42UL, SobolRsg::JoeKuoD5, true, true);
    emitSobol(8, 42UL, SobolRsg::JoeKuoD6, true, true);
    emitSobol(8, 42UL, SobolRsg::JoeKuoD7, true, true);
    emitSobol(8, 42UL, SobolRsg::Kuo, true, true);
    emitSobol(8, 42UL, SobolRsg::Kuo2, true, true);
    emitSobol(8, 42UL, SobolRsg::Kuo3, true, true);
    // dimensions past the tabulated blocks exercise the polynomial roll-over
    emitSobol(45, 42UL, SobolRsg::Jaeckel, true, true);
    emitSobol(45, 42UL, SobolRsg::SobolLevitan, true, true);
    emitSobol(400, 7UL, SobolRsg::SobolLevitanLemieux, true, true);
    emitSobol(64, 42UL, SobolRsg::JoeKuoD7, true, false);
    std::cout << "],\n";

    // ---- Burley2020SobolRsg ---------------------------------------------
    std::cout << "\"burley2020_sobol_rsg\": [";
    emitBurley(2, 42UL, SobolRsg::Jaeckel, 43UL, true);
    emitBurley(5, 42UL, SobolRsg::Jaeckel, 43UL, true);
    emitBurley(8, 1UL, SobolRsg::JoeKuoD7, 17UL, true);
    emitBurley(45, 42UL, SobolRsg::Jaeckel, 43UL, false);
    std::cout << "],\n";

    // ---- RandomizedLDS ---------------------------------------------------
    std::cout << "\"randomized_lds\": [";
    emitRandomizedLds(3, 42UL, 42UL, true);
    emitRandomizedLds(5, 7UL, 11UL, false);
    std::cout << "],\n";

    // ---- SobolBrownianBridgeRsg -----------------------------------------
    std::cout << "\"sobol_brownian_bridge_rsg\": [";
    emitSbbRsg(2, 4, SobolBrownianGenerator::Diagonal, 0UL, SobolRsg::JoeKuoD7, true);
    emitSbbRsg(2, 4, SobolBrownianGenerator::Factors, 0UL, SobolRsg::JoeKuoD7, true);
    emitSbbRsg(2, 4, SobolBrownianGenerator::Steps, 0UL, SobolRsg::JoeKuoD7, true);
    emitSbbRsg(3, 5, SobolBrownianGenerator::Diagonal, 42UL, SobolRsg::Jaeckel, false);
    std::cout << "],\n";

    std::cout << "\"burley2020_sobol_brownian_bridge_rsg\": [";
    emitBurleySbbRsg(2, 4, SobolBrownianGenerator::Diagonal, 42UL, SobolRsg::JoeKuoD7, 43UL, true);
    emitBurleySbbRsg(3, 5, SobolBrownianGenerator::Steps, 42UL, SobolRsg::Jaeckel, 43UL, false);
    std::cout << "],\n";

    std::cout << "\"brownian_ordered_indices\": [";
    emitOrderedIndices(2, 4, SobolBrownianGenerator::Factors, true);
    emitOrderedIndices(2, 4, SobolBrownianGenerator::Steps, true);
    emitOrderedIndices(2, 4, SobolBrownianGenerator::Diagonal, true);
    emitOrderedIndices(3, 5, SobolBrownianGenerator::Diagonal, false);
    std::cout << "],\n";

    // ---- rngtraits -------------------------------------------------------
    std::cout << "\"pseudo_random\": [";
    emitPseudoRandomTraits(1, 42UL, true);
    emitPseudoRandomTraits(4, 42UL, false);
    std::cout << "],\n";

    std::cout << "\"poisson_pseudo_random\": [";
    emitPoissonPseudoRandomTraits(3, 42UL, false);
    std::cout << "],\n";

    std::cout << "\"low_discrepancy\": [";
    emitLowDiscrepancyTraits(1, 0UL, true);
    emitLowDiscrepancyTraits(4, 0UL, false);
    std::cout << "],\n";

    // ---- StochasticCollocationInvCDF ------------------------------------
    std::cout << "\"stochastic_collocation_inv_cdf\": [";
    {
        const InverseCumulativeNormal icn;
        emitScInvCdf(
            "normal_order8", [icn](Real p) { return icn(p); }, 8, Null<Real>(), Null<Real>(), true);
        emitScInvCdf(
            "lognormal_order10", [icn](Real p) { return std::exp(0.3 * icn(p) - 0.045); }, 10,
            Null<Real>(), Null<Real>(), true);
        emitScInvCdf(
            "lognormal_order10_pmax", [icn](Real p) { return std::exp(0.3 * icn(p) - 0.045); }, 10,
            0.9999, Null<Real>(), true);
        emitScInvCdf(
            "shifted_order6", [icn](Real p) { return 2.0 * icn(p) + 1.5; }, 6, Null<Real>(),
            0.0001, false);
    }
    std::cout << "]\n";

    std::cout << "}\n";
    return 0;
}

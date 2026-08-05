"""Cross-validate the generic RNG wrappers against the v1.43 C++ probe.

Reference: ``migration-harness/references/v143/math/randomnumbers.json`` —
sections ``seed_generator``, ``mt19937_by_array``,
``random_sequence_generator``, ``inverse_cumulative_rng``,
``inverse_cumulative_rng_poisson``, ``inverse_cumulative_rsg``,
``cl_gaussian_rng``, ``pseudo_random``, ``poisson_pseudo_random``,
``low_discrepancy``.

Two tiers appear here, and the split is not arbitrary:

* **EXACT** wherever the value is a uniform deviate or an integer. Those come
  out of shifts, XORs and one exact division by a power of two.
* **TIGHT** wherever the value has been through ``InverseCumulativeNormal``.
  Both sides evaluate the *same* Acklam rational approximation with the same
  constants (C++ leaves its optional Halley refinement ``#ifdef``-ed out), so
  the only difference is that the C++ Release build contracts the Horner
  multiply-adds into FMAs and Python cannot. Measured over these cases the
  discrepancy peaks at ~5e-15 relative; TIGHT's 1e-12 is a 200x margin and is
  not tuned to the data.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.distributions.poisson_distribution import InverseCumulativePoisson
from pquantlib.math.randomnumbers.central_limit_gaussian_rng import CLGaussianRng
from pquantlib.math.randomnumbers.inverse_cumulative_rng import InverseCumulativeRng
from pquantlib.math.randomnumbers.inverse_cumulative_rsg import InverseCumulativeRsg
from pquantlib.math.randomnumbers.mersenne_twister import MersenneTwisterUniformRng
from pquantlib.math.randomnumbers.random_sequence_generator import (
    RandomSequenceGenerator,
)
from pquantlib.math.randomnumbers.rng_traits import (
    LowDiscrepancy,
    PoissonPseudoRandom,
    PseudoRandom,
)
from pquantlib.math.randomnumbers.seed_generator import SeedGenerator
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/randomnumbers")


# -- SeedGenerator --------------------------------------------------------


def test_seed_generator_chain_exact(cpp: dict[str, Any]) -> None:
    """The clock-independent part of ``SeedGenerator::initialize`` is exact.

    ``SeedGenerator`` reads ``std::time(nullptr)`` for its first seed, so its
    output cannot be pinned directly. Everything downstream of that first
    seed is deterministic, and it is what a port gets wrong: the modulo-1000
    skip, the four-word ``init_by_array`` key, and the order the two
    intermediate generators are drawn from. The probe replays those four
    steps for fixed first seeds; ``initialize(first_seed)`` is the seam that
    lets Python do the same.
    """
    for chain in cpp["seed_generator"]["chains"]:
        sg = SeedGenerator.__new__(SeedGenerator)
        sg.initialize(chain["first_seed"])
        assert [sg.get() for _ in range(len(chain["gets"]))] == chain["gets"]


def test_seed_generator_is_a_singleton() -> None:
    """# C++ parity: ``Singleton<SeedGenerator>``."""
    assert SeedGenerator.instance() is SeedGenerator.instance()


def test_seed_generator_output_is_a_uint32() -> None:
    """Whatever the clock says, ``get()`` must stay in the C++ output range."""
    value = SeedGenerator.instance().get()
    assert 0 <= value <= 0xFFFFFFFF


def test_zero_seed_reaches_the_seed_generator() -> None:
    """Seed 0 must not raise and must not be deterministic.

    # C++ parity: mt19937uniformrng.cpp:88. Two generators seeded with 0
    # draw from independent ``SeedGenerator`` seeds, so agreeing on the first
    # draw has probability 2^-32.
    """
    a = MersenneTwisterUniformRng(0).next_int32()
    b = MersenneTwisterUniformRng(0).next_int32()
    assert a != b


# -- MersenneTwisterUniformRng.from_seeds ---------------------------------


def test_mt_init_by_array_exact(cpp: dict[str, Any]) -> None:
    """``init_by_array`` seeding — what ``SeedGenerator`` ultimately uses."""
    for case in cpp["mt19937_by_array"]:
        rng = MersenneTwisterUniformRng.from_seeds(case["seeds"])
        assert [rng.next_int32() for _ in case["ints"]] == case["ints"]
        for expected in case["reals"]:
            tolerance.exact(rng.next().value, expected)


# -- RandomSequenceGenerator ----------------------------------------------


def test_random_sequence_generator_exact(cpp: dict[str, Any]) -> None:
    """Uniform sequences, their weights, and a draw 10 000 deep."""
    for case in cpp["random_sequence_generator"]:
        g = RandomSequenceGenerator.from_seed(case["dimension"], case["seed"])
        for expected in case["sequences"]:
            sample = g.next_sequence()
            for a, e in zip(sample.value, expected["value"], strict=True):
                tolerance.exact(float(a), float(e))
            tolerance.exact(sample.weight, expected["weight"])
        tolerance.exact(g.last_sequence().weight, case["last_weight"])

        for _ in range(10000):
            g.next_sequence()
        for a, e in zip(g.next_sequence().value, case["after_10000"], strict=True):
            tolerance.exact(float(a), float(e))
        assert g.next_int32_sequence() == case["next_int32_sequence"]
        assert g.dimension() == case["dimension"]


def test_random_sequence_generator_rejects_zero_dimension() -> None:
    """# C++ parity: randomsequencegenerator.hpp:59-61."""
    with pytest.raises(LibraryException, match="greater than 0"):
        RandomSequenceGenerator(0, MersenneTwisterUniformRng(42))


# -- InverseCumulativeRng / Rsg -------------------------------------------


def test_inverse_cumulative_rng_tight(cpp: dict[str, Any]) -> None:
    """Gaussian draws — TIGHT because of FMA contraction (see module docstring)."""
    for case in cpp["inverse_cumulative_rng"]:
        g = InverseCumulativeRng(MersenneTwisterUniformRng(case["seed"]))
        for expected in case["draws"]:
            tolerance.tight(g.next().value, expected)
        for _ in range(10000):
            g.next()
        tolerance.tight(g.next().value, case["after_10000"])


def test_inverse_cumulative_rng_weight_passes_through(cpp: dict[str, Any]) -> None:
    """The weight is copied, not recomputed — so it stays exactly 1."""
    for case in cpp["inverse_cumulative_rng"]:
        g = InverseCumulativeRng(MersenneTwisterUniformRng(case["seed"]))
        tolerance.exact(g.next().weight, case["weight"])


def test_inverse_cumulative_rng_poisson_exact(cpp: dict[str, Any]) -> None:
    """Poisson deviates are integers, so EXACT applies even after the mapping."""
    for case in cpp["inverse_cumulative_rng_poisson"]:
        g = InverseCumulativeRng(
            MersenneTwisterUniformRng(case["seed"]), InverseCumulativePoisson()
        )
        for expected in case["draws"]:
            tolerance.exact(g.next().value, expected)


def test_inverse_cumulative_rsg_tight(cpp: dict[str, Any]) -> None:
    """Gaussian sequences over a pseudo-random uniform sequence."""
    for case in cpp["inverse_cumulative_rsg"]:
        g = InverseCumulativeRsg(
            RandomSequenceGenerator.from_seed(case["dimension"], case["seed"])
        )
        assert g.dimension() == case["dimension_accessor"]
        for expected in case["sequences"]:
            sample = g.next_sequence()
            for a, e in zip(sample.value, expected["value"], strict=True):
                tolerance.tight(float(a), float(e))
            tolerance.exact(sample.weight, expected["weight"])
        for _ in range(10000):
            g.next_sequence()
        for a, e in zip(g.next_sequence().value, case["after_10000"], strict=True):
            tolerance.tight(float(a), float(e))


# -- CLGaussianRng --------------------------------------------------------


def test_cl_gaussian_rng_exact(cpp: dict[str, Any]) -> None:
    """Twelve uniforms summed and shifted by -6 — no transcendental, so EXACT.

    The accumulation order is load-bearing: adding the draws in a different
    order, or summing them before subtracting 6, changes the last bits.
    """
    for case in cpp["cl_gaussian_rng"]:
        g = CLGaussianRng(MersenneTwisterUniformRng(case["seed"]))
        for expected in case["draws"]:
            tolerance.exact(g.next().value, expected)
        for _ in range(1000):
            g.next()
        tolerance.exact(g.next().value, case["after_1000"])


def test_cl_gaussian_support_is_bounded() -> None:
    """The method's support is exactly (-6, 6) — a real limitation, pinned."""
    g = CLGaussianRng(MersenneTwisterUniformRng(42))
    for _ in range(2000):
        assert -6.0 < g.next().value < 6.0


# -- rngtraits ------------------------------------------------------------


def test_pseudo_random_traits(cpp: dict[str, Any]) -> None:
    """``PseudoRandom::make_sequence_generator`` and its error-estimate flag."""
    for case in cpp["pseudo_random"]:
        assert PseudoRandom.allows_error_estimate == case["allows_error_estimate"]
        rsg = PseudoRandom.make_sequence_generator(case["dimension"], case["seed"])
        assert rsg.dimension() == case["dimension_accessor"]
        for expected in case["sequences"]:
            for a, e in zip(rsg.next_sequence().value, expected, strict=True):
                tolerance.tight(float(a), float(e))


def test_poisson_pseudo_random_traits(cpp: dict[str, Any]) -> None:
    """The Poisson trait default-constructs its inverse cumulative (lambda 1)."""
    for case in cpp["poisson_pseudo_random"]:
        assert PoissonPseudoRandom.allows_error_estimate == case["allows_error_estimate"]
        rsg = PoissonPseudoRandom.make_sequence_generator(case["dimension"], case["seed"])
        for expected in case["sequences"]:
            for a, e in zip(rsg.next_sequence().value, expected, strict=True):
                tolerance.exact(float(a), float(e))


def test_low_discrepancy_traits(cpp: dict[str, Any]) -> None:
    """``LowDiscrepancy`` wires Sobol into the same inverse-cumulative shell.

    ``allows_error_estimate`` is 0 here and 1 for ``PseudoRandom``: a
    low-discrepancy point set is not i.i.d., so a sample standard error would
    be meaningless. Monte Carlo engines branch on this flag, so it is pinned.
    """
    for case in cpp["low_discrepancy"]:
        assert LowDiscrepancy.allows_error_estimate == case["allows_error_estimate"]
        rsg = LowDiscrepancy.make_sequence_generator(case["dimension"], case["seed"])
        assert rsg.dimension() == case["dimension_accessor"]
        for expected in case["sequences"]:
            for a, e in zip(rsg.next_sequence().value, expected, strict=True):
                tolerance.tight(float(a), float(e))


def test_error_estimate_flags_differ() -> None:
    """The whole point of the two traits families."""
    assert PseudoRandom.allows_error_estimate == 1
    assert LowDiscrepancy.allows_error_estimate == 0

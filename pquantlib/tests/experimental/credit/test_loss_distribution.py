"""Cross-validate LossDist family against C++.

Probe source: migration-harness/cpp/probes/cluster_w3a/probe.cpp
Reference:    migration-harness/references/cluster/w3a.json
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.experimental.credit.distribution import Distribution
from pquantlib.experimental.credit.loss_distribution import (
    BinomialProbabilityOfAtLeastNEvents,
    LossDist,
    LossDistBinomial,
    LossDistBucketing,
    LossDistHomogeneous,
    LossDistMonteCarlo,
    ProbabilityOfAtLeastNEvents,
    ProbabilityOfNEvents,
    binomial_probability_of_at_least_n_events,
    binomial_probability_of_n_events,
    probability_of_at_least_n_events,
    probability_of_n_events,
    probability_of_n_events_vec,
)
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/w3a")


def test_probability_of_n_events_vec_matches_cpp(cpp_ref: dict[str, Any]) -> None:
    ref = cpp_ref["loss_dist_probabilities"]
    p = [0.1, 0.2, 0.3, 0.4]
    probs = probability_of_n_events_vec(p)
    tolerance.tight(probs[0], ref["p0"])
    tolerance.tight(probs[1], ref["p1"])
    tolerance.tight(probs[2], ref["p2"])
    tolerance.tight(probs[3], ref["p3"])
    tolerance.tight(probs[4], ref["p4"])


def test_probability_of_at_least_n_events_matches_cpp(cpp_ref: dict[str, Any]) -> None:
    p = [0.1, 0.2, 0.3, 0.4]
    tolerance.tight(
        probability_of_at_least_n_events(2, p),
        cpp_ref["loss_dist_probabilities"]["at_least_2"],
    )


def test_binomial_probability_helpers_match_cpp(cpp_ref: dict[str, Any]) -> None:
    p = [0.2, 0.2, 0.2, 0.2]
    ref = cpp_ref["loss_dist_binomial"]
    tolerance.tight(binomial_probability_of_n_events(0, p), ref["p_n0"])
    tolerance.tight(binomial_probability_of_n_events(1, p), ref["p_n1"])
    tolerance.tight(binomial_probability_of_n_events(2, p), ref["p_n2"])
    tolerance.tight(binomial_probability_of_at_least_n_events(2, p), ref["at_least_2"])


def test_loss_dist_homogeneous_matches_cpp(cpp_ref: dict[str, Any]) -> None:
    ref = cpp_ref["loss_dist_homogeneous"]
    n_buckets = 10
    maximum = 10.0
    volume = 1.0
    p = [0.1, 0.2, 0.3, 0.4]
    ldh = LossDistHomogeneous(n_buckets, maximum)
    dist = ldh.for_volume(volume, p)

    assert dist.size() == ref["n_buckets"]
    tolerance.tight(dist.x(0), ref["x_0"])
    tolerance.tight(dist.x(1), ref["x_1"])
    tolerance.tight(dist.dx(0), ref["dx_0"])
    tolerance.tight(dist.density(0), ref["density_0"])
    tolerance.tight(dist.density(1), ref["density_1"])
    tolerance.tight(dist.cumulative(0), ref["cumulative_0"])
    tolerance.tight(dist.cumulative(2), ref["cumulative_2"])
    tolerance.tight(dist.excess(0), ref["excess_0"])
    tolerance.tight(dist.excess(2), ref["excess_2"])
    # Probability vector
    tolerance.tight(ldh.probability()[0], ref["prob_n0"])
    tolerance.tight(ldh.probability()[1], ref["prob_n1"])
    tolerance.tight(ldh.probability()[2], ref["prob_n2"])
    tolerance.tight(ldh.probability()[4], ref["prob_n4"])
    tolerance.tight(ldh.excess_probability()[0], ref["excess_prob_n0"])
    tolerance.tight(ldh.excess_probability()[2], ref["excess_prob_n2"])
    tolerance.tight(ldh.volume(), ref["volume"])
    assert ldh.size() == ref["size_field"]


def test_loss_dist_binomial_matches_cpp(cpp_ref: dict[str, Any]) -> None:
    ref = cpp_ref["loss_dist_binomial_dist"]
    n_buckets = 10
    maximum = 10.0
    volume = 1.0
    n = 5
    probability = 0.2

    ldb = LossDistBinomial(n_buckets, maximum)
    dist = ldb.for_uniform(n, volume, probability)

    assert dist.size() == ref["n_buckets"]
    tolerance.tight(ldb.probability()[0], ref["prob_n0"])
    tolerance.tight(ldb.probability()[1], ref["prob_n1"])
    tolerance.tight(ldb.probability()[2], ref["prob_n2"])
    tolerance.tight(ldb.probability()[5], ref["prob_n5"])
    tolerance.tight(ldb.excess_probability()[0], ref["excess_prob_n0"])
    tolerance.tight(ldb.excess_probability()[3], ref["excess_prob_n3"])
    assert ldb.size() == ref["size_field"]


def test_loss_dist_binomial_for_array_overload_leaves_volume_unassigned() -> None:
    """The 2-arg call (volumes, probabilities) never writes ``volume_``.

    # C++ parity note (DEFECT, reproduced verbatim): neither
    # ``LossDistBinomial::operator()`` assigns the member ``volume_``
    # (lossdistribution.cpp:146-179), which is declared without an initialiser
    # (lossdistribution.hpp:111) and read as the loop guard
    # ``if (volume_ * i <= maximum_)`` at cpp:155. C++ therefore branches on
    # indeterminate memory; the v1.43 probe measured a different denormal near
    # 2.14e-314 on each of 25 runs, so the guard never binds. Leaving the
    # Python member at 0.0 lands in the same regime.
    #
    # An earlier revision of this test asserted ``volume() == volumes[0]``,
    # i.e. that the port "plugged the hole". That is a divergence, not a fix:
    # a non-zero ``volume_`` makes the guard bind for large i and silently
    # zeroes binomial terms C++ keeps.
    """
    n_buckets = 10
    maximum = 10.0
    volume = 1.0
    n = 5

    ldb = LossDistBinomial(n_buckets, maximum)
    volumes = [volume] * n
    probabilities = [0.2] * n
    dist = ldb(volumes, probabilities)
    assert dist.size() == n_buckets
    tolerance.exact(ldb.volume(), 0.0)


def test_loss_dist_bucketing_runs_on_arbitrary_inputs() -> None:
    """LossDistBucketing should not crash on a small inhomogeneous basket.

    The C++ probe doesn't capture detailed bucketing-output values
    (cross-validation belongs to the W3-C correlation/basket cluster).
    Here we just exercise the construction + invariants.
    """
    ldb = LossDistBucketing(20, 5.0, epsilon=1e-9)
    volumes = [0.1, 0.2, 0.3]
    probabilities = [0.05, 0.10, 0.15]
    dist = ldb(volumes, probabilities)
    assert dist.size() == 20
    # CDF of last bucket should be ≈ 1 (some over/under-shoot tolerated).
    assert dist.cumulative(dist.size() - 1) > 0.99


def test_loss_dist_monte_carlo_runs_without_error() -> None:
    """LossDistMonteCarlo should sample without numerical pathology."""
    ldmc = LossDistMonteCarlo(10, 5.0, simulations=100, seed=42)
    volumes = [0.5, 0.5]
    probabilities = [0.1, 0.2]
    dist = ldmc(volumes, probabilities)
    assert dist.size() == 10
    assert dist.cumulative(dist.size() - 1) > 0.9


def test_probability_of_n_events_functor() -> None:
    p = [0.1, 0.2, 0.3, 0.4]
    f = ProbabilityOfNEvents(2)
    tolerance.tight(f(p), probability_of_n_events(2, p))


def test_probability_of_at_least_n_events_functor() -> None:
    p = [0.1, 0.2, 0.3, 0.4]
    f = ProbabilityOfAtLeastNEvents(2)
    tolerance.tight(f(p), probability_of_at_least_n_events(2, p))


def test_binomial_probability_of_at_least_n_events_functor() -> None:
    p = [0.2] * 4
    f = BinomialProbabilityOfAtLeastNEvents(2)
    tolerance.tight(f(p), binomial_probability_of_at_least_n_events(2, p))


# =============================================================================
# v1.43 cross-validation — block B of the creditloss probe.
#
# Probe source: migration-harness/cpp/probes/v143_experimental_creditloss/probe.cpp
# Reference:    migration-harness/references/v143/experimental/creditloss.json
#
# All TIGHT. The Monte-Carlo variant is pinned EXACTLY rather than
# statistically: C++ draws from a MersenneTwisterUniformRng, so the stream is
# reproducible and the Python port must walk it identically.
# =============================================================================

#: probe.cpp:376.
_B_P = [0.02, 0.05, 0.10, 0.17, 0.30]
#: probe.cpp:404-408.
_B_BUCKETS = 10
_B_VOLUME = 20.0
_B_MAXIMUM = 100.0
#: probe.cpp:431.
_B_HET_VOLUMES = [10.0, 20.0, 25.0, 30.0, 15.0]


@pytest.fixture(scope="module")
def v143_ref() -> dict[str, Any]:
    return reference_reader.load("v143/experimental/creditloss")


def _assert_distribution(
    dist: Distribution, prefix: str, ref: dict[str, Any]
) -> None:
    n = dist.size()
    assert n == ref[f"{prefix}_size"]
    for suffix, getter in (
        ("_x", dist.x),
        ("_dx", dist.dx),
        ("_density", dist.density),
        ("_cumulative", dist.cumulative),
        ("_excess", dist.excess),
        ("_cumulative_excess", dist.cumulative_excess),
        ("_average", dist.average),
    ):
        expected = ref[prefix + suffix]
        for i in range(n):
            tolerance.tight(getter(i), expected[i])


def test_v143_probability_helpers_match_cpp(v143_ref: dict[str, Any]) -> None:
    for actual, expected in zip(
        probability_of_n_events_vec(_B_P),
        v143_ref["lossdist_prob_n_events_vec"],
        strict=True,
    ):
        tolerance.tight(actual, expected)
    for k in range(6):
        tolerance.tight(
            probability_of_n_events(k, _B_P), v143_ref["lossdist_prob_n_events"][k]
        )
        tolerance.tight(
            probability_of_at_least_n_events(k, _B_P),
            v143_ref["lossdist_prob_at_least_n_events"][k],
        )
        tolerance.tight(
            binomial_probability_of_n_events(k, _B_P),
            v143_ref["lossdist_binom_prob_n_events"][k],
        )
        tolerance.tight(
            binomial_probability_of_at_least_n_events(k, _B_P),
            v143_ref["lossdist_binom_prob_at_least_n_events"][k],
        )


def test_v143_functor_wrappers_agree_with_statics(v143_ref: dict[str, Any]) -> None:
    for k in range(6):
        tolerance.tight(
            ProbabilityOfNEvents(k)(_B_P), v143_ref["lossdist_functor_n_events"][k]
        )
        tolerance.tight(
            ProbabilityOfAtLeastNEvents(k)(_B_P),
            v143_ref["lossdist_functor_at_least_n_events"][k],
        )
        tolerance.tight(
            BinomialProbabilityOfAtLeastNEvents(k)(_B_P),
            v143_ref["lossdist_functor_binom_at_least_n_events"][k],
        )


def test_v143_binomial_at_least_zero_events_reproduces_cpp_defect(
    v143_ref: dict[str, Any],
) -> None:
    """P(N >= 0) comes out 0, not 1.

    # C++ parity note (DEFECT, reproduced verbatim):
    # ``binomialProbabilityOfAtLeastNEvents`` (lossdistribution.cpp:34-46)
    # calls ``CumulativeBinomialDistribution::operator()(BigNatural k)`` with
    # ``k = n-1``. At ``n == 0`` the ``-1`` converts to the largest unsigned
    # value, the ``if (k >= n_) return 1.0;`` guard
    # (binomialdistribution.hpp:71-73) fires, and the result is ``1.0 - 1.0``.
    #
    # The heterogeneous sibling ``probabilityOfAtLeastNEvents`` has no such
    # conversion and correctly returns 1 — asserted below so the two cannot
    # silently converge if the C++ is ever fixed.
    """
    tolerance.exact(binomial_probability_of_at_least_n_events(0, _B_P), 0.0)
    assert v143_ref["lossdist_binom_prob_at_least_n_events"][0] == 0
    tolerance.tight(probability_of_at_least_n_events(0, _B_P), 1.0)


def test_v143_loss_dist_binomial_matches_cpp(v143_ref: dict[str, Any]) -> None:
    ldb = LossDistBinomial(_B_BUCKETS, _B_MAXIMUM)
    dist = ldb([_B_VOLUME] * 5, _B_P)
    assert ldb.size() == v143_ref["lossdist_binomial_n"]
    for actual, expected in zip(
        ldb.probability(), v143_ref["lossdist_binomial_probability"], strict=True
    ):
        tolerance.tight(actual, expected)
    for actual, expected in zip(
        ldb.excess_probability(),
        v143_ref["lossdist_binomial_excess_probability"],
        strict=True,
    ):
        tolerance.tight(actual, expected)
    _assert_distribution(dist, "lossdist_binomial_dist", v143_ref)


def test_v143_loss_dist_binomial_volume_defect(v143_ref: dict[str, Any]) -> None:
    """``volume_`` is never assigned, so the loop guard never binds.

    # C++ parity note (DEFECT, reproduced verbatim): lossdistribution.hpp:111
    # declares ``mutable Real volume_;`` with no initialiser (contrast
    # ``LossDistHomogeneous`` at hpp:153-154, which does initialise), and
    # lossdistribution.cpp:155 reads it as ``if (volume_ * i <= maximum_)``.
    # The value is indeterminate — measured over 25 probe runs on arm64/libc++
    # it is a different denormal near 2.14e-314 each time — so the float itself
    # is unpinnable and the probe pins the guard's *outcome* instead.
    """
    assert v143_ref["lossdist_binomial_defect_guard_binds"] is False
    assert v143_ref["lossdist_binomial_defect_volume_assigned_by_call"] is False
    ldb = LossDistBinomial(_B_BUCKETS, _B_MAXIMUM)
    ldb([_B_VOLUME] * 5, _B_P)
    # Same regime as C++: the guard cannot exclude any i.
    tolerance.exact(ldb.volume(), 0.0)


def test_v143_loss_dist_homogeneous_matches_cpp(v143_ref: dict[str, Any]) -> None:
    ldh = LossDistHomogeneous(_B_BUCKETS, _B_MAXIMUM)
    dist = ldh([_B_VOLUME] * 5, _B_P)
    assert ldh.size() == v143_ref["lossdist_homog_n"]
    tolerance.tight(ldh.volume(), v143_ref["lossdist_homog_volume"])
    for actual, expected in zip(
        ldh.probability(), v143_ref["lossdist_homog_probability"], strict=True
    ):
        tolerance.tight(actual, expected)
    for actual, expected in zip(
        ldh.excess_probability(),
        v143_ref["lossdist_homog_excess_probability"],
        strict=True,
    ):
        tolerance.tight(actual, expected)
    _assert_distribution(dist, "lossdist_homog_dist", v143_ref)


@pytest.mark.parametrize(
    ("epsilon", "prefix"),
    [(1e-6, "lossdist_bucketing_dist"), (1e-9, "lossdist_bucketing_eps9_dist")],
)
def test_v143_loss_dist_bucketing_matches_cpp(
    v143_ref: dict[str, Any], epsilon: float, prefix: str
) -> None:
    ldbk = LossDistBucketing(_B_BUCKETS, _B_MAXIMUM, epsilon)
    _assert_distribution(ldbk(_B_HET_VOLUMES, _B_P), prefix, v143_ref)


@pytest.mark.parametrize(
    ("simulations", "seed", "prefix"),
    [
        (2000, 42, "lossdist_montecarlo_dist"),
        (500, 17, "lossdist_montecarlo_s17_dist"),
    ],
)
def test_v143_loss_dist_montecarlo_matches_cpp(
    v143_ref: dict[str, Any], simulations: int, seed: int, prefix: str
) -> None:
    """Pinned exactly: the C++ Mersenne Twister stream is reproducible."""
    ldmc = LossDistMonteCarlo(_B_BUCKETS, _B_MAXIMUM, simulations, seed)
    _assert_distribution(ldmc(_B_HET_VOLUMES, _B_P), prefix, v143_ref)


def test_v143_loss_dist_shape_accessors(v143_ref: dict[str, Any]) -> None:
    ldbk = LossDistBucketing(_B_BUCKETS, _B_MAXIMUM)
    assert ldbk.n_buckets() == v143_ref["lossdist_buckets"]
    tolerance.tight(ldbk.maximum(), v143_ref["lossdist_maximum"])
    assert isinstance(ldbk, LossDist)

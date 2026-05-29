"""Cross-validate LossDist family against C++.

Probe source: migration-harness/cpp/probes/cluster_w3a/probe.cpp
Reference:    migration-harness/references/cluster/w3a.json
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.experimental.credit.loss_distribution import (
    BinomialProbabilityOfAtLeastNEvents,
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


def test_loss_dist_binomial_for_array_overload_sets_volume() -> None:
    """The 2-arg call (volumes, probabilities) wires ``volume_`` from volumes[0].

    # C++ parity divergence: the C++ version leaves volume_ uninitialised
    # for this codepath (probe shows 1.97e-323 = uninitialised memory). The
    # Python port sets it from volumes[0] to plug the hole.
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
    tolerance.exact(ldb.volume(), volume)


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

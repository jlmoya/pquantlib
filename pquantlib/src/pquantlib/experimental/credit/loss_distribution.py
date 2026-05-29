"""LossDistribution — probability formulas + bucketed convolution algorithms.

# C++ parity: ql/experimental/credit/lossdistribution.{hpp,cpp} (v1.42.1).

Top-level abstractions:

  - ``LossDistribution`` — abstract base; subclasses build a
    ``Distribution`` from per-name (notional, probability) arrays.
  - ``LossDistBinomial`` — binomial loss distribution with constant
    per-name notional + probability (uses p[0]/n).
  - ``LossDistHomogeneous`` — exact loss distribution for equal-volume,
    varying-probability names via Ma 2007 / Hull-White 2004 recursion.
  - ``LossDistBucketing`` — Hull-White bucketing for arbitrary
    notionals + probabilities (no copula).
  - ``LossDistMonteCarlo`` — Monte-Carlo sampling for arbitrary
    notionals + probabilities, independent default events.

Plus the static helpers exposed at module scope (mirrors C++
``LossDist::probabilityOfNEvents`` etc.):

  - ``probability_of_n_events_vec(p)`` — full P(N=k) array.
  - ``probability_of_n_events(n, p)`` — single P(N=n).
  - ``probability_of_at_least_n_events(n, p)`` — P(N>=n).
  - ``binomial_probability_of_n_events(n, p)`` — Binomial(p[0], len(p)).
  - ``binomial_probability_of_at_least_n_events(n, p)`` — Binomial>=.

# C++ parity divergence: the C++ MersenneTwisterUniformRng is replaced by
# pquantlib.math.randomnumbers.mt19937_uniform_rng to match the EXACT-tier
# RNG porting policy in MEMORY.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from scipy.stats import binom  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

from pquantlib import qassert
from pquantlib.experimental.credit.distribution import Distribution
from pquantlib.math.randomnumbers.mersenne_twister import MersenneTwisterUniformRng


# C++ parity: the C++ ``BinomialDistribution`` (PMF) + ``CumulativeBinomialDistribution``
# (CDF) classes live at ql/math/distributions/binomialdistribution.hpp. Python uses
# scipy.stats.binom — algebraically the same since both rely on log-gamma based
# binomial coefficients. Phase 1 left ``BinomialDistribution`` as an L1 carve-out
# (no test path exercised it yet); using scipy here closes that carve-out at
# the call sites that need it without inflating Phase 1 scope.
def _binom_pmf(p: float, n: int, k: int) -> float:
    """PMF of Binomial(n, p) at k.

    # C++ parity: BinomialDistribution(p, n)(k).
    """
    return float(binom.pmf(k, n, p))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]


def _binom_cdf(p: float, n: int, k: int) -> float:
    """CDF of Binomial(n, p) at k (= P(X <= k)).

    # C++ parity: CumulativeBinomialDistribution(p, n)(k).
    """
    return float(binom.cdf(k, n, p))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]

# ----- module-level static helpers --------------------------------------------


def binomial_probability_of_n_events(n: int, p: list[float]) -> float:
    """P(N=n) under Binomial(p[0], len(p)).

    # C++ parity: LossDist::binomialProbabilityOfNEvents at
    # lossdistribution.cpp:28-32.
    """
    return _binom_pmf(p[0], len(p), n)


def binomial_probability_of_at_least_n_events(n: int, p: list[float]) -> float:
    """P(N>=n) under Binomial(p[0], len(p)).

    # C++ parity: LossDist::binomialProbabilityOfAtLeastNEvents at
    # lossdistribution.cpp:35-46.
    """
    return 1.0 - _binom_cdf(p[0], len(p), n - 1)


def probability_of_n_events_vec(p: list[float]) -> list[float]:
    """Full P(N=k) array using the Ma 2007 / Hull-White recursion.

    # C++ parity: LossDist::probabilityOfNEvents(vector<Real>&) at
    # lossdistribution.cpp:49-64.
    """
    n = len(p)
    probability = [0.0] * (n + 1)
    probability[0] = 1.0
    for j in range(n):
        prev = list(probability)
        probability[0] = prev[0] * (1.0 - p[j])
        for i in range(1, j + 1):
            probability[i] = prev[i - 1] * p[j] + prev[i] * (1.0 - p[j])
        probability[j + 1] = prev[j] * p[j]
    return probability


def probability_of_n_events(k: int, p: list[float]) -> float:
    """P(N=k) for a heterogeneous probability vector.

    # C++ parity: LossDist::probabilityOfNEvents(int, vector<Real>&) at
    # lossdistribution.cpp:67-69 — delegates to the vector overload.
    """
    return probability_of_n_events_vec(p)[k]


def probability_of_at_least_n_events(k: int, p: list[float]) -> float:
    """P(N>=k) for a heterogeneous probability vector.

    # C++ parity: LossDist::probabilityOfAtLeastNEvents at
    # lossdistribution.cpp:112-118.
    """
    probability = probability_of_n_events_vec(p)
    sum_below = 1.0
    for j in range(k):
        sum_below -= probability[j]
    return sum_below


# ----- functor-style wrappers (mirror C++ class wrappers) ---------------------


class ProbabilityOfNEvents:
    """Functor wrapping P(N=n).

    # C++ parity: lossdistribution.hpp:67.
    """

    __slots__ = ("_n",)

    def __init__(self, n: int) -> None:
        self._n = n

    def __call__(self, p: list[float]) -> float:
        return probability_of_n_events(self._n, p)


class ProbabilityOfAtLeastNEvents:
    """Functor wrapping P(N>=n).

    # C++ parity: lossdistribution.hpp:76.
    """

    __slots__ = ("_n",)

    def __init__(self, n: int) -> None:
        self._n = n

    def __call__(self, p: list[float]) -> float:
        return probability_of_at_least_n_events(self._n, p)


class BinomialProbabilityOfAtLeastNEvents:
    """Functor wrapping Binomial-P(N>=n).

    # C++ parity: lossdistribution.hpp:85.
    """

    __slots__ = ("_n",)

    def __init__(self, n: int) -> None:
        self._n = n

    def __call__(self, p: list[float]) -> float:
        return binomial_probability_of_at_least_n_events(self._n, p)


# ----- abstract base + concretes ----------------------------------------------


class LossDistribution(ABC):
    """Abstract base for bucketed loss-distribution algorithms.

    Each subclass produces a ``Distribution`` from per-name volume +
    probability arrays. The ``n_buckets`` + ``maximum`` parameters fix
    the loss grid the result is discretised onto.
    """

    @abstractmethod
    def __call__(
        self, volumes: list[float], probabilities: list[float]
    ) -> Distribution:
        """Produce a bucketed loss distribution."""

    @abstractmethod
    def n_buckets(self) -> int: ...

    @abstractmethod
    def maximum(self) -> float: ...


class LossDistBinomial(LossDistribution):
    """Binomial loss distribution with constant volume + probability.

    Treats the input as a Binomial(probabilities[0], len(volumes)) with
    a uniform per-name volume.

    # C++ parity: lossdistribution.hpp:95 + cpp:146-179.
    """

    __slots__ = (
        "_excess_probability",
        "_maximum",
        "_n",
        "_n_buckets",
        "_probability",
        "_volume",
    )

    def __init__(self, n_buckets: int, maximum: float) -> None:
        self._n_buckets = n_buckets
        self._maximum = maximum
        self._volume: float = 0.0
        self._n: int = 0
        self._probability: list[float] = []
        self._excess_probability: list[float] = []

    def n_buckets(self) -> int:
        return self._n_buckets

    def maximum(self) -> float:
        return self._maximum

    def volume(self) -> float:
        return self._volume

    def size(self) -> int:
        return self._n

    def probability(self) -> list[float]:
        return list(self._probability)

    def excess_probability(self) -> list[float]:
        return list(self._excess_probability)

    def for_uniform(
        self, n: int, volume: float, probability: float
    ) -> Distribution:
        """3-arg overload — Binomial(probability, n) with uniform ``volume`` per name.

        # C++ parity: lossdistribution.cpp:146-172.
        """
        self._n = n
        self._probability = [0.0] * (n + 1)
        dist = Distribution(self._n_buckets, 0.0, self._maximum)
        for i in range(n + 1):
            if self._volume * i <= self._maximum:
                self._probability[i] = _binom_pmf(probability, n, i)
                bucket = dist.locate(volume * i)
                dist.add_density(bucket, self._probability[i] / dist.dx(bucket))
                dist.add_average(bucket, volume * i)

        self._excess_probability = [0.0] * (n + 1)
        self._excess_probability[self._n] = self._probability[self._n]
        for k in range(self._n - 1, -1, -1):
            self._excess_probability[k] = (
                self._excess_probability[k + 1] + self._probability[k]
            )

        dist.normalize()
        return dist

    def __call__(
        self, volumes: list[float], probabilities: list[float]
    ) -> Distribution:
        """Variant overload using parallel arrays — picks the head as the binomial pair.

        # C++ parity: lossdistribution.cpp:175-179 — delegates to the
        # 3-arg overload (n, volumes[0], probabilities[0]). Note the C++
        # implementation does NOT set ``volume_`` here, which leaves
        # ``volume()`` returning uninitialised memory if called after
        # this overload. The Python port plugs that hole by writing
        # ``self._volume = volumes[0]``.
        """
        self._volume = volumes[0]
        return self.for_uniform(len(volumes), volumes[0], probabilities[0])


class LossDistHomogeneous(LossDistribution):
    """Exact loss distribution for equal-volume names with varying probabilities.

    # C++ parity: lossdistribution.hpp:138 + cpp:182-224. Implementation
    # follows Xiaofong Ma's PhD thesis recursion (formula 2.1).
    """

    __slots__ = (
        "_excess_probability",
        "_maximum",
        "_n",
        "_n_buckets",
        "_probability",
        "_volume",
    )

    def __init__(self, n_buckets: int, maximum: float) -> None:
        self._n_buckets = n_buckets
        self._maximum = maximum
        self._n: int = 0
        self._volume: float = 0.0
        self._probability: list[float] = []
        self._excess_probability: list[float] = []

    def n_buckets(self) -> int:
        return self._n_buckets

    def maximum(self) -> float:
        return self._maximum

    def size(self) -> int:
        return self._n

    def volume(self) -> float:
        return self._volume

    def probability(self) -> list[float]:
        return list(self._probability)

    def excess_probability(self) -> list[float]:
        return list(self._excess_probability)

    def for_volume(self, volume: float, p: list[float]) -> Distribution:
        """Build the loss distribution given a single ``volume`` + probability vector.

        # C++ parity: lossdistribution.cpp:182-217.
        """
        self._volume = volume
        self._n = len(p)
        self._probability = [0.0] * (self._n + 1)
        self._probability[0] = 1.0
        for k in range(self._n):
            prev = list(self._probability)
            self._probability[0] = prev[0] * (1.0 - p[k])
            for i in range(1, k + 1):
                self._probability[i] = prev[i - 1] * p[k] + prev[i] * (1.0 - p[k])
            self._probability[k + 1] = prev[k] * p[k]

        self._excess_probability = [0.0] * (self._n + 1)
        self._excess_probability[self._n] = self._probability[self._n]
        for k in range(self._n - 1, -1, -1):
            self._excess_probability[k] = (
                self._excess_probability[k + 1] + self._probability[k]
            )

        dist = Distribution(self._n_buckets, 0.0, self._maximum)
        for i in range(self._n + 1):
            if volume * i <= self._maximum:
                bucket = dist.locate(volume * i)
                dist.add_density(bucket, self._probability[i] / dist.dx(bucket))
                dist.add_average(bucket, volume * i)

        dist.normalize()
        return dist

    def __call__(
        self, volumes: list[float], probabilities: list[float]
    ) -> Distribution:
        """Treat ``volumes[0]`` as the shared notional + delegate.

        # C++ parity: lossdistribution.cpp:220-224.
        """
        return self.for_volume(volumes[0], probabilities)


class LossDistBucketing(LossDistribution):
    """Hull-White bucketing for arbitrary notionals + independent probabilities.

    # C++ parity: lossdistribution.hpp:170 + cpp:227-298.

    Independence is assumed (no copula). For correlated names use the
    latent-model loss models from the downstream W3 clusters.
    """

    __slots__ = ("_epsilon", "_maximum", "_n_buckets")

    def __init__(
        self, n_buckets: int, maximum: float, epsilon: float = 1e-6
    ) -> None:
        self._n_buckets = n_buckets
        self._maximum = maximum
        self._epsilon = epsilon

    def n_buckets(self) -> int:
        return self._n_buckets

    def maximum(self) -> float:
        return self._maximum

    def _locate_target_bucket(self, loss: float, i0: int = 0) -> int:
        # # C++ parity: lossdistribution.cpp:292-298.
        qassert.require(loss >= 0.0, f"loss {loss} must be >= 0")
        dx = self._maximum / self._n_buckets
        for i in range(i0, self._n_buckets):
            if dx * i > loss + self._epsilon:
                return i - 1
        return self._n_buckets

    def __call__(
        self, volumes: list[float], probabilities: list[float]
    ) -> Distribution:
        qassert.require(
            len(volumes) == len(probabilities),
            f"sizes differ: {len(volumes)} vs {len(probabilities)}",
        )

        p = [0.0] * self._n_buckets
        a = [0.0] * self._n_buckets

        p[0] = 1.0
        a[0] = 0.0
        dx = self._maximum / self._n_buckets
        for k in range(1, self._n_buckets):
            a[k] = dx * k + dx / 2.0

        for i in range(len(volumes)):
            L = volumes[i]  # noqa: N806
            P = probabilities[i]  # noqa: N806
            for k in range(self._n_buckets - 1, -1, -1):
                if p[k] > 0:
                    u = self._locate_target_bucket(a[k] + L, k)
                    qassert.require(u >= 0, f"u={u} at i={i} k={k}")
                    qassert.require(u >= k, f"u={u}<k={k} at i={i}")
                    dp = p[k] * P
                    if u == k:
                        a[k] += P * L
                    else:
                        if u < self._n_buckets:
                            if dp > 0.0:
                                f = 1.0 / (1.0 + (p[u] / p[k]) / P)
                                a[u] = (1.0 - f) * a[u] + f * (a[k] + L)
                            p[u] += dp
                        p[k] -= dp
                qassert.require(
                    a[k] + self._epsilon >= dx * k and a[k] < dx * (k + 1),
                    f"a out of range at k={k}, contract {i}",
                )

        dist = Distribution(self._n_buckets, 0.0, self._maximum)
        for i in range(self._n_buckets):
            dist.add_density(i, p[i] / dx)
            dist.add_average(i, a[i])
        return dist


class LossDistMonteCarlo(LossDistribution):
    """Monte-Carlo sampling for independent default events.

    # C++ parity: lossdistribution.hpp:193 + cpp:302-322.
    """

    __slots__ = ("_epsilon", "_maximum", "_n_buckets", "_seed", "_simulations")

    def __init__(
        self,
        n_buckets: int,
        maximum: float,
        simulations: int,
        seed: int = 42,
        epsilon: float = 1e-6,
    ) -> None:
        self._n_buckets = n_buckets
        self._maximum = maximum
        self._simulations = simulations
        self._seed = seed
        self._epsilon = epsilon

    def n_buckets(self) -> int:
        return self._n_buckets

    def maximum(self) -> float:
        return self._maximum

    def __call__(
        self, volumes: list[float], probabilities: list[float]
    ) -> Distribution:
        # # C++ parity: lossdistribution.cpp:302-322. Uses
        # MersenneTwisterUniformRng — Python port uses pquantlib's
        # mersenne_twister implementation.
        dist = Distribution(self._n_buckets, 0.0, self._maximum)
        rng = MersenneTwisterUniformRng(self._seed)
        for _ in range(self._simulations):
            e = 0.0
            for j in range(len(volumes)):
                r = rng.next().value
                if r <= probabilities[j]:
                    e += volumes[j]
            dist.add(e + self._epsilon)
        dist.normalize()
        return dist

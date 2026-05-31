"""MTBrownianGenerator — Mersenne-Twister Brownian generator + factory.

# C++ parity: ql/models/marketmodels/browniangenerators/
# mtbrowniangenerator.{hpp,cpp} (v1.42.1).

Incremental Brownian generator using a Mersenne-twister uniform generator
and the inverse-cumulative Gaussian method. The underlying uniform sequence
is generated eagerly per path (``next_path``), and its transformation into
Gaussian variates is lazy per step (``next_step``) — matching the C++ note.

The uniform stream is a length ``factors * steps``
``UniformRandomSequenceGenerator`` over a ``MersenneTwisterUniformRng``; both
are bit-identical ports of the C++ originals, so the Gaussian variates this
generator emits match the C++ reference exactly.
"""

from __future__ import annotations

from pquantlib.math.distributions.inverse_cumulative_normal import (
    InverseCumulativeNormal,
)
from pquantlib.math.randomnumbers.mersenne_twister import MersenneTwisterUniformRng
from pquantlib.methods.montecarlo.gaussian_sequence_generator import (
    UniformRandomSequenceGenerator,
)
from pquantlib.models.marketmodels.brownian_generator import (
    BrownianGenerator,
    BrownianGeneratorFactory,
)


class MTBrownianGenerator(BrownianGenerator):
    """Mersenne-twister Brownian generator for market-model simulations.

    # C++ parity: mtbrowniangenerator.hpp MTBrownianGenerator.
    """

    def __init__(self, factors: int, steps: int, seed: int = 0) -> None:
        self._factors = factors
        self._steps = steps
        self._last_step = 0
        # C++ parity: RandomSequenceGenerator<MersenneTwisterUniformRng>
        # (factors*steps, MersenneTwisterUniformRng(seed)). Note: the C++
        # MersenneTwisterUniformRng falls back to a clock-seeded
        # SeedGenerator when seed == 0; pquantlib's MT raises on seed 0
        # (the SeedGenerator clock fallback is not ported), so the default
        # seed=0 here is retained only for signature parity.
        self._generator = UniformRandomSequenceGenerator(
            factors * steps, MersenneTwisterUniformRng(seed)
        )
        self._inverse_cumulative = InverseCumulativeNormal()

    def next_step(self, output: list[float]) -> float:
        """Fill ``output`` with this step's Gaussian variates; return weight 1.0.

        # C++ parity: mtbrowniangenerator.cpp MTBrownianGenerator::nextStep.
        """
        current_sequence = self._generator.last_sequence().value
        start = self._last_step * self._factors
        for i in range(self._factors):
            output[i] = self._inverse_cumulative(float(current_sequence[start + i]))
        self._last_step += 1
        return 1.0

    def next_path(self) -> float:
        """Draw a new uniform sequence; return its weight.

        # C++ parity: mtbrowniangenerator.cpp MTBrownianGenerator::nextPath.
        """
        sample = self._generator.next_sequence()
        self._last_step = 0
        return sample.weight

    def number_of_factors(self) -> int:
        return self._factors

    def number_of_steps(self) -> int:
        return self._steps


class MTBrownianGeneratorFactory(BrownianGeneratorFactory):
    """Factory building ``MTBrownianGenerator`` instances with a fixed seed.

    # C++ parity: mtbrowniangenerator.hpp MTBrownianGeneratorFactory.
    """

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed

    def create(self, factors: int, steps: int) -> BrownianGenerator:
        return MTBrownianGenerator(factors, steps, self._seed)

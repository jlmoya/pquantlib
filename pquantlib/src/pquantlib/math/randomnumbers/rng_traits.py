"""Random-number-generation policies.

# C++ parity: ql/math/randomnumbers/rngtraits.hpp (v1.43) —
# ``GenericPseudoRandom`` / ``GenericLowDiscrepancy`` and the three concrete
# typedefs ``PseudoRandom``, ``PoissonPseudoRandom``, ``LowDiscrepancy``.

These are C++ traits structs — no instances are ever created — but they are
not *empty* tag types: each carries a static factory
(``make_sequence_generator``), a compile-time flag (``allowsErrorEstimate``)
that Monte Carlo engines branch on, and a mutable static
``icInstance`` that lets a caller swap in a configured inverse cumulative
without touching the type. All three survive the port as real behaviour, so
these are classes with class-level state rather than something to allowlist.

The one thing that does not survive is the *typedef* surface: C++ exposes
``PseudoRandom::rsg_type`` etc. so that templates can name the generator type
they will get. Python has no need to name a type before constructing it, so
the typedefs are represented as class attributes holding the classes
themselves — which is enough for the same job (a caller can do
``PseudoRandom.rsg_type(...)``).
"""

from __future__ import annotations

from typing import ClassVar

from pquantlib.math.distributions.poisson_distribution import InverseCumulativePoisson
from pquantlib.math.randomnumbers.inverse_cumulative_rng import InverseCumulativeRng
from pquantlib.math.randomnumbers.inverse_cumulative_rsg import InverseCumulativeRsg
from pquantlib.math.randomnumbers.random_number_generator import (
    InverseCumulative,
    RandomNumberGenerator,
    UniformSequenceGenerator,
)
from pquantlib.math.randomnumbers.random_sequence_generator import (
    RandomSequenceGenerator,
)
from pquantlib.math.randomnumbers.sobol_rsg import SobolRsg


class GenericPseudoRandom[Urng: RandomNumberGenerator, Ic: InverseCumulative]:
    """Traits for pseudo-random number generation.

    # C++ parity: ``GenericPseudoRandom<URNG, IC>`` (rngtraits.hpp:43-63).

    Subclass and set ``urng_factory`` / ``ic_factory`` to instantiate the
    template; ``PseudoRandom`` and ``PoissonPseudoRandom`` below are the two
    instantiations C++ ships.
    """

    #: # C++ parity: ``enum { allowsErrorEstimate = 1 }``. A pseudo-random
    #: sample is i.i.d., so the sample standard error is meaningful.
    allows_error_estimate: ClassVar[int] = 1

    #: # C++ parity: ``typedef InverseCumulativeRng<urng_type, IC> rng_type``.
    rng_type: ClassVar[type] = InverseCumulativeRng
    #: # C++ parity: ``typedef RandomSequenceGenerator<urng_type> ursg_type``.
    ursg_type: ClassVar[type] = RandomSequenceGenerator
    #: # C++ parity: ``typedef InverseCumulativeRsg<ursg_type, IC> rsg_type``.
    rsg_type: ClassVar[type] = InverseCumulativeRsg

    #: # C++ parity: ``static ext::shared_ptr<IC> icInstance``. Null by
    #: default, in which case the factory lets ``rsg_type`` default-construct
    #: its own inverse cumulative.
    ic_instance: ClassVar[InverseCumulative | None] = None

    @classmethod
    def make_sequence_generator(
        cls, dimension: int, seed: int
    ) -> InverseCumulativeRsg[UniformSequenceGenerator, InverseCumulative]:
        """Build the Gaussian (or otherwise mapped) sequence generator.

        # C++ parity: ``make_sequence_generator`` (rngtraits.hpp:53-58).
        Seed 0 reaches ``SeedGenerator`` through the Mersenne Twister and is
        therefore clock-derived, exactly as in C++.
        """
        g = RandomSequenceGenerator.from_seed(dimension, seed)
        ic = cls.ic_instance
        return InverseCumulativeRsg(g, ic) if ic is not None else InverseCumulativeRsg(g)


class GenericLowDiscrepancy[Ursg: UniformSequenceGenerator, Ic: InverseCumulative]:
    """Traits for low-discrepancy sequence generation.

    # C++ parity: ``GenericLowDiscrepancy<URSG, IC>`` (rngtraits.hpp:83-101).
    """

    #: # C++ parity: ``enum { allowsErrorEstimate = 0 }``. A low-discrepancy
    #: point set is not i.i.d., so the sample standard error would lie.
    allows_error_estimate: ClassVar[int] = 0

    #: # C++ parity: ``typedef InverseCumulativeRsg<ursg_type, IC> rsg_type``.
    rsg_type: ClassVar[type] = InverseCumulativeRsg

    #: # C++ parity: ``static ext::shared_ptr<IC> icInstance``.
    ic_instance: ClassVar[InverseCumulative | None] = None

    @classmethod
    def ursg_factory(cls, dimension: int, seed: int) -> UniformSequenceGenerator:
        """Construct the underlying uniform sequence generator.

        # C++ parity: the ``ursg_type g(dimension, seed)`` line of
        # ``make_sequence_generator`` (rngtraits.hpp:96). Subclasses override
        # this the way C++ supplies the ``URSG`` template argument.
        """
        raise NotImplementedError

    @classmethod
    def make_sequence_generator(
        cls, dimension: int, seed: int
    ) -> InverseCumulativeRsg[UniformSequenceGenerator, InverseCumulative]:
        """Build the mapped low-discrepancy sequence generator.

        # C++ parity: ``make_sequence_generator`` (rngtraits.hpp:94-99).
        """
        g = cls.ursg_factory(dimension, seed)
        ic = cls.ic_instance
        return InverseCumulativeRsg(g, ic) if ic is not None else InverseCumulativeRsg(g)


class PseudoRandom(GenericPseudoRandom[RandomNumberGenerator, InverseCumulative]):
    """Default pseudo-random traits: Mersenne Twister + inverse normal.

    # C++ parity: ``typedef GenericPseudoRandom<MersenneTwisterUniformRng,
    # InverseCumulativeNormal> PseudoRandom`` (rngtraits.hpp:70-71).
    """


class PoissonPseudoRandom(GenericPseudoRandom[RandomNumberGenerator, InverseCumulative]):
    """Poisson-distributed pseudo-random traits.

    # C++ parity: ``typedef GenericPseudoRandom<MersenneTwisterUniformRng,
    # InverseCumulativePoisson> PoissonPseudoRandom`` (rngtraits.hpp:77-78).

    C++ default-constructs the inverse cumulative, so ``lambda`` is 1 unless
    ``icInstance`` is set; ``ic_instance`` below reproduces that default
    rather than leaving it null, because Python cannot default-construct a
    type parameter inside ``InverseCumulativeRsg``.
    """

    @classmethod
    def make_sequence_generator(
        cls, dimension: int, seed: int
    ) -> InverseCumulativeRsg[UniformSequenceGenerator, InverseCumulative]:
        """# C++ parity: rngtraits.hpp:53-58 with ``IC =
        InverseCumulativePoisson``."""
        g = RandomSequenceGenerator.from_seed(dimension, seed)
        ic = cls.ic_instance if cls.ic_instance is not None else InverseCumulativePoisson()
        return InverseCumulativeRsg(g, ic)


class LowDiscrepancy(GenericLowDiscrepancy[UniformSequenceGenerator, InverseCumulative]):
    """Default low-discrepancy traits: Sobol + inverse normal.

    # C++ parity: ``typedef GenericLowDiscrepancy<SobolRsg,
    # InverseCumulativeNormal> LowDiscrepancy`` (rngtraits.hpp:103-104).
    """

    @classmethod
    def ursg_factory(cls, dimension: int, seed: int) -> UniformSequenceGenerator:
        """# C++ parity: ``URSG = SobolRsg`` (rngtraits.hpp:103)."""
        return SobolRsg(dimension, seed)


__all__ = [
    "GenericLowDiscrepancy",
    "GenericPseudoRandom",
    "LowDiscrepancy",
    "PoissonPseudoRandom",
    "PseudoRandom",
]

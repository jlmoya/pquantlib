"""SingleVariate / MultiVariate — Monte Carlo path-generation traits.

# C++ parity: ql/methods/montecarlo/mctraits.hpp (v1.43).

C++ uses these as compile-time policy structs: ``McSimulation<MC, RNG, S>``
picks ``path_type`` / ``path_generator_type`` / ``path_pricer_type`` and the
``allowsErrorEstimate`` flag out of whichever trait it is instantiated with.

Python has no compile-time type selection, so the two structs become classes
whose *class attributes* carry the same choices and whose one class method
builds the matching path generator. That keeps them usable — an engine can
branch on ``traits.allows_error_estimate`` and call
``traits.make_path_generator(...)`` — rather than being pure documentation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from pquantlib.math.randomnumbers.rng_traits import PseudoRandom
from pquantlib.methods.montecarlo.multi_path import MultiPath
from pquantlib.methods.montecarlo.multi_path_generator import MultiPathGenerator
from pquantlib.methods.montecarlo.path import Path
from pquantlib.methods.montecarlo.path_generator import (
    GaussianSequenceGeneratorProtocol,
    PathGenerator,
)

if TYPE_CHECKING:
    from pquantlib.processes.stochastic_process import StochasticProcess
    from pquantlib.processes.stochastic_process_1d import StochasticProcess1D
    from pquantlib.time.time_grid import TimeGrid


class SingleVariate:
    """Monte Carlo traits for single-variate models.

    # C++ parity: ``template <class RNG = PseudoRandom> struct SingleVariate``.

    ``rng_traits`` defaults to ``PseudoRandom``, matching the C++ default
    template argument. Subclass and override it for low-discrepancy runs.
    """

    #: # C++ parity: ``typedef RNG rng_traits``.
    rng_traits: ClassVar[type[PseudoRandom]] = PseudoRandom
    #: # C++ parity: ``typedef Path path_type``.
    path_type: ClassVar[type[Path]] = Path
    #: # C++ parity: ``typedef PathGenerator<rsg_type> path_generator_type``.
    path_generator_type: ClassVar[type[PathGenerator]] = PathGenerator
    #: # C++ parity: ``enum { allowsErrorEstimate = RNG::allowsErrorEstimate }``.
    allows_error_estimate: ClassVar[int] = PseudoRandom.allows_error_estimate

    @classmethod
    def make_path_generator(
        cls,
        process: StochasticProcess1D,
        time_grid: TimeGrid,
        generator: GaussianSequenceGeneratorProtocol,
        brownian_bridge: bool = False,
    ) -> PathGenerator:
        """Build the path generator this trait selects."""
        return PathGenerator.with_time_grid(process, time_grid, generator, brownian_bridge)


class MultiVariate:
    """Monte Carlo traits for multi-variate models.

    # C++ parity: ``template <class RNG = PseudoRandom> struct MultiVariate``.
    """

    #: # C++ parity: ``typedef RNG rng_traits``.
    rng_traits: ClassVar[type[PseudoRandom]] = PseudoRandom
    #: # C++ parity: ``typedef MultiPath path_type``.
    path_type: ClassVar[type[MultiPath]] = MultiPath
    #: # C++ parity: ``typedef MultiPathGenerator<rsg_type> path_generator_type``.
    path_generator_type: ClassVar[type[MultiPathGenerator]] = MultiPathGenerator
    #: # C++ parity: ``enum { allowsErrorEstimate = RNG::allowsErrorEstimate }``.
    allows_error_estimate: ClassVar[int] = PseudoRandom.allows_error_estimate

    @classmethod
    def make_path_generator(
        cls,
        process: StochasticProcess,
        time_grid: TimeGrid,
        generator: GaussianSequenceGeneratorProtocol,
        brownian_bridge: bool = False,
    ) -> MultiPathGenerator:
        """Build the path generator this trait selects."""
        return MultiPathGenerator(process, time_grid, generator, brownian_bridge)


__all__ = ["MultiVariate", "SingleVariate"]

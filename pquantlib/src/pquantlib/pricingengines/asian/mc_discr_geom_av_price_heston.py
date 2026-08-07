"""MCDiscreteGeometricAPHestonEngine — Heston MC for discrete geometric Asians.

# C++ parity: ql/pricingengines/asian/mc_discr_geom_av_price_heston.{hpp,cpp}
# (v1.43) —
# ``template <class RNG = PseudoRandom, class S = Statistics,
#            class P = HestonProcess>
#  class MCDiscreteGeometricAPHestonEngine
#      : public MCDiscreteAveragingAsianEngineBase<MultiVariate,RNG,S>``,
# ``class GeometricAPOHestonPathPricer : public PathPricer<MultiPath>`` and
# ``template <...> class MakeMCDiscreteGeometricAPHestonEngine``.

By default the MC discretization uses one time step per fixing date, but this
can be controlled via ``time_steps`` / ``time_steps_per_year``, which provide
additional steps. The grid tries to space them as evenly as it can and does not
guarantee an exact number of steps; the realised grid is published in
``results.additional_results["TimeGrid"]``.

Because extra steps land *between* fixings, the path pricer must average over
the fixing points only — it is handed the grid indices of the mandatory times
(``TimeGrid.closest_index``), not the whole path.

This is a ``MultiVariate`` engine: paths come from a ``MultiPathGenerator`` over
the 2-factor Heston process, so the Gaussian sequence has dimension
``factors * (grid.size() - 1)`` and is consumed two at a time per step. Brownian
bridge is unavailable (``MultiPathGenerator`` fails on it), which is why the
constructor passes ``brownianBridge = false`` to its base
(mc_discr_geom_av_price_heston.hpp:123-132).
"""

from __future__ import annotations

import sys

from pquantlib import qassert
from pquantlib.exercise import EuropeanExercise
from pquantlib.math.randomnumbers.rng_traits import PseudoRandom
from pquantlib.methods.montecarlo.multi_path import MultiPath
from pquantlib.methods.montecarlo.path_pricer import PathPricer
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.asian.mc_discrete_asian_engine_base import (
    MCDiscreteAveragingAsianEngineBase,
)
from pquantlib.pricingengines.vanilla.mc_vanilla_engine import RngTraits
from pquantlib.processes.heston_process import HestonProcess
from pquantlib.time.time_grid import TimeGrid

#: # C++ parity: ``QL_MAX_REAL`` = ``std::numeric_limits<Real>::max()``.
_QL_MAX_REAL = sys.float_info.max


def fixing_indices(time_grid: TimeGrid) -> list[int]:
    """Grid indices of the mandatory (fixing) times.

    # C++ parity: the identical five-line block at the head of every Heston
    # Asian ``pathPricer`` / ``controlPathPricer``
    # (mc_discr_geom_av_price_heston.hpp:142-149,
    #  mc_discr_arith_av_price_heston.hpp:160-167 and 199-206).
    """
    return [time_grid.closest_index(t) for t in time_grid.mandatory_times]


class GeometricAPOHestonPathPricer(PathPricer[MultiPath]):
    """Geometric-average path pricer reading the spot leg of a ``MultiPath``.

    # C++ parity: ``GeometricAPOHestonPathPricer``
    # (mc_discr_geom_av_price_heston.hpp:91-107,
    #  mc_discr_geom_av_price_heston.cpp:25-60).
    """

    __slots__ = (
        "_discount",
        "_fixing_indices",
        "_past_fixings",
        "_payoff",
        "_running_product",
    )

    def __init__(
        self,
        option_type: OptionType,
        strike: float,
        discount: float,
        fixing_index_list: list[int],
        running_product: float = 1.0,
        past_fixings: int = 0,
    ) -> None:
        # C++ parity: mc_discr_geom_av_price_heston.cpp:33-34.
        qassert.require(strike >= 0.0, "strike less than zero not allowed")
        self._payoff: PlainVanillaPayoff = PlainVanillaPayoff(option_type, strike)
        self._discount: float = discount
        self._fixing_indices: list[int] = fixing_index_list
        self._running_product: float = running_product
        self._past_fixings: int = past_fixings

    def __call__(self, path: MultiPath) -> float:
        """# C++ parity: ``GeometricAPOHestonPathPricer::operator()``
        # (mc_discr_geom_av_price_heston.cpp:37-60).

        Only ``multiPath[0]`` (the spot leg) is read; the variance leg is
        ignored. Unlike the single-variate pricers, the t=0 point enters the
        average iff index 0 is one of the fixing indices — there is no
        ``mandatoryTimes()[0] == 0`` special case here, because the fixing
        indices already encode it.
        """
        spot = path[0]
        qassert.require(path.path_size() > 0, "the path cannot be empty")

        average_price = 1.0
        product = self._running_product
        fixings = self._past_fixings + len(self._fixing_indices)

        # care must be taken not to overflow product
        for index in self._fixing_indices:
            price = spot[index]
            if product < _QL_MAX_REAL / price:
                product *= price
            else:
                average_price *= product ** (1.0 / fixings)
                product = price

        average_price *= product ** (1.0 / fixings)
        return self._discount * self._payoff(average_price)


class MCDiscreteGeometricAPHestonEngine(MCDiscreteAveragingAsianEngineBase[MultiPath]):
    """Heston MC engine for discrete-geometric-average price Asians.

    # C++ parity: ``MCDiscreteGeometricAPHestonEngine<RNG,S,P>``
    # (mc_discr_geom_av_price_heston.hpp:46-65, 112-174).
    """

    def __init__(
        self,
        process: HestonProcess,
        *,
        antithetic_variate: bool = False,
        required_samples: int | None = None,
        required_tolerance: float | None = None,
        max_samples: int | None = None,
        seed: int = 0,
        time_steps: int | None = None,
        time_steps_per_year: int | None = None,
        rng_traits: RngTraits = PseudoRandom,
    ) -> None:
        super().__init__(
            process,
            brownian_bridge=False,
            antithetic_variate=antithetic_variate,
            control_variate=False,
            required_samples=required_samples,
            required_tolerance=required_tolerance,
            max_samples=max_samples,
            seed=seed,
            time_steps=time_steps,
            time_steps_per_year=time_steps_per_year,
            rng_traits=rng_traits,
            multi_variate=True,
        )
        # C++ parity: mc_discr_geom_av_price_heston.hpp:133-134.
        qassert.require(
            time_steps is None or time_steps_per_year is None,
            "both time steps and time steps per year were provided",
        )

    def path_pricer(self) -> PathPricer[MultiPath]:
        """Build the geometric-average Heston path pricer.

        # C++ parity: ``MCDiscreteGeometricAPHestonEngine::pathPricer``
        # (mc_discr_geom_av_price_heston.hpp:137-174).
        """
        # Keep track of the fixing indices, the path pricer will need to prod
        # only these.
        indices = fixing_indices(self.time_grid())

        args = self._arguments
        payoff = args.payoff
        qassert.require(isinstance(payoff, PlainVanillaPayoff), "non-plain payoff given")
        assert isinstance(payoff, PlainVanillaPayoff)

        exercise = args.exercise
        qassert.require(isinstance(exercise, EuropeanExercise), "wrong exercise given")
        assert isinstance(exercise, EuropeanExercise)

        process = self._process
        qassert.require(
            isinstance(process, HestonProcess), "Heston like process required"
        )
        assert isinstance(process, HestonProcess)

        assert args.running_accumulator is not None
        assert args.past_fixings is not None
        return GeometricAPOHestonPathPricer(
            payoff.option_type(),
            payoff.strike(),
            process.risk_free_rate().discount(exercise.last_date()),
            indices,
            args.running_accumulator,
            args.past_fixings,
        )


class MakeMCDiscreteGeometricAPHestonEngine:
    """Fluent builder for :class:`MCDiscreteGeometricAPHestonEngine`.

    # C++ parity: ``MakeMCDiscreteGeometricAPHestonEngine<RNG,S,P>``
    # (mc_discr_geom_av_price_heston.hpp:68-89, 176-254).

    Unlike the Black-Scholes builders this one carries steps knobs, and it
    rejects over-specification EARLY, inside :meth:`with_steps` /
    :meth:`with_steps_per_year`. :meth:`engine` performs NO validation at all:
    leaving both unset is legal and means "one step per fixing".
    """

    __slots__ = (
        "_antithetic",
        "_max_samples",
        "_process",
        "_rng_traits",
        "_samples",
        "_seed",
        "_steps",
        "_steps_per_year",
        "_tolerance",
    )

    def __init__(
        self, process: HestonProcess, rng_traits: RngTraits = PseudoRandom
    ) -> None:
        self._process: HestonProcess = process
        self._rng_traits: RngTraits = rng_traits
        self._antithetic: bool = False
        self._samples: int | None = None
        self._max_samples: int | None = None
        self._steps: int | None = None
        self._steps_per_year: int | None = None
        self._tolerance: float | None = None
        self._seed: int = 0

    def with_samples(self, samples: int) -> MakeMCDiscreteGeometricAPHestonEngine:
        """# C++ parity: ``withSamples``
        # (mc_discr_geom_av_price_heston.hpp:182-189)."""
        qassert.require(self._tolerance is None, "tolerance already set")
        self._samples = samples
        return self

    def with_absolute_tolerance(
        self, tolerance: float
    ) -> MakeMCDiscreteGeometricAPHestonEngine:
        """# C++ parity: ``withAbsoluteTolerance``
        # (mc_discr_geom_av_price_heston.hpp:191-202)."""
        qassert.require(self._samples is None, "number of samples already set")
        qassert.require(
            bool(self._rng_traits.allows_error_estimate),
            "chosen random generator policy does not allow an error estimate",
        )
        self._tolerance = tolerance
        return self

    def with_max_samples(self, samples: int) -> MakeMCDiscreteGeometricAPHestonEngine:
        """# C++ parity: ``withMaxSamples``
        # (mc_discr_geom_av_price_heston.hpp:204-209)."""
        self._max_samples = samples
        return self

    def with_seed(self, seed: int) -> MakeMCDiscreteGeometricAPHestonEngine:
        """# C++ parity: ``withSeed`` (mc_discr_geom_av_price_heston.hpp:211-216)."""
        self._seed = seed
        return self

    def with_antithetic_variate(
        self, b: bool = True
    ) -> MakeMCDiscreteGeometricAPHestonEngine:
        """# C++ parity: ``withAntitheticVariate``
        # (mc_discr_geom_av_price_heston.hpp:218-223)."""
        self._antithetic = b
        return self

    def with_steps(self, steps: int) -> MakeMCDiscreteGeometricAPHestonEngine:
        """# C++ parity: ``withSteps`` (mc_discr_geom_av_price_heston.hpp:225-232)."""
        qassert.require(
            self._steps_per_year is None, "number of steps per year already set"
        )
        self._steps = steps
        return self

    def with_steps_per_year(self, steps: int) -> MakeMCDiscreteGeometricAPHestonEngine:
        """# C++ parity: ``withStepsPerYear``
        # (mc_discr_geom_av_price_heston.hpp:234-241)."""
        qassert.require(self._steps is None, "number of steps already set")
        self._steps_per_year = steps
        return self

    def engine(self) -> MCDiscreteGeometricAPHestonEngine:
        """# C++ parity: ``operator ext::shared_ptr<PricingEngine>() const``
        # (mc_discr_geom_av_price_heston.hpp:243-254)."""
        return MCDiscreteGeometricAPHestonEngine(
            self._process,
            antithetic_variate=self._antithetic,
            required_samples=self._samples,
            required_tolerance=self._tolerance,
            max_samples=self._max_samples,
            seed=self._seed,
            time_steps=self._steps,
            time_steps_per_year=self._steps_per_year,
            rng_traits=self._rng_traits,
        )


__all__ = [
    "GeometricAPOHestonPathPricer",
    "MCDiscreteGeometricAPHestonEngine",
    "MakeMCDiscreteGeometricAPHestonEngine",
    "fixing_indices",
]

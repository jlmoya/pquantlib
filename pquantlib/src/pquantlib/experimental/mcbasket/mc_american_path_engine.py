"""MCAmericanPathEngine — least-squares Monte Carlo for early-exercise baskets.

# C++ parity: ql/experimental/mcbasket/mcamericanpathengine.hpp (v1.43).

Concrete :class:`~pquantlib.experimental.mcbasket.mc_longstaff_schwartz_path_engine.MCLongstaffSchwartzPathEngine`
for a
:class:`~pquantlib.experimental.mcbasket.path_multi_asset_option.PathMultiAssetOption`.
The regression basis is hard-wired in C++ to order-2 monomials, and the LSM
pricer is fed one implied forward curve per fixing.

Warning (from the C++ header): this method is intrinsically weak for
out-of-the-money options.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.experimental.mcbasket.longstaff_schwartz_multi_path_pricer import (
    LongstaffSchwartzMultiPathPricer,
)
from pquantlib.experimental.mcbasket.mc_longstaff_schwartz_path_engine import (
    MCLongstaffSchwartzPathEngine,
)
from pquantlib.math.randomnumbers.rng_traits import PseudoRandom
from pquantlib.methods.montecarlo.lsm_basis_system import PolynomialType
from pquantlib.pricingengines.mc_rng_traits import RngTraits
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.processes.stochastic_process_array import StochasticProcessArray
from pquantlib.termstructures.yield_.implied_term_structure import ImpliedTermStructure
from pquantlib.termstructures.yield_term_structure import YieldTermStructure

_POLYNOMIAL_ORDER = 2
"""# C++ parity: ``const Size polynomialOrder = 2`` (mcamericanpathengine.hpp:147)."""

_POLYNOMIAL_TYPE = PolynomialType.Monomial
"""# C++ parity: ``LsmBasisSystem::Monomial`` (mcamericanpathengine.hpp:148)."""


class MCAmericanPathEngine(MCLongstaffSchwartzPathEngine):
    """Least-squares Monte Carlo engine for path-dependent American baskets.

    # C++ parity: ``MCAmericanPathEngine<RNG>`` (mcamericanpathengine.hpp:40-57).
    """

    def __init__(
        self,
        processes: StochasticProcessArray,
        time_steps: int | None,
        time_steps_per_year: int | None,
        brownian_bridge: bool,
        antithetic_variate: bool,
        control_variate: bool,
        required_samples: int | None,
        required_tolerance: float | None,
        max_samples: int | None,
        seed: int,
        n_calibration_samples: int | None = None,
        rng_traits: RngTraits = PseudoRandom,
    ) -> None:
        super().__init__(
            processes,
            time_steps,
            time_steps_per_year,
            brownian_bridge,
            antithetic_variate,
            control_variate,
            required_samples,
            required_tolerance,
            max_samples,
            seed,
            n_calibration_samples,
            rng_traits,
        )

    def lsm_path_pricer(self) -> LongstaffSchwartzMultiPathPricer:
        """Build the LSM pricer from the fixing grid and the risk-free curve.

        # C++ parity: ``MCAmericanPathEngine<RNG>::lsmPathPricer``
        # (mcamericanpathengine.hpp:114-155).
        """
        process_array = self._process
        qassert.require(
            isinstance(process_array, StochasticProcessArray) and process_array.size() > 0,
            "Stochastic process array required",
        )
        assert isinstance(process_array, StochasticProcessArray)

        process = process_array.process(0)
        qassert.require(
            isinstance(process, GeneralizedBlackScholesProcess),
            "generalized Black-Scholes process required",
        )
        assert isinstance(process, GeneralizedBlackScholesProcess)

        the_time_grid = self.time_grid()
        times = list(the_time_grid.mandatory_times)
        number_of_times = len(times)

        fixings = self._arguments.fixing_dates
        qassert.require(len(fixings) == number_of_times, "Invalid dates/times")

        time_positions = [0] * number_of_times
        discount_factors: npt.NDArray[np.float64] = np.empty(number_of_times, dtype=np.float64)
        forward_term_structures: list[YieldTermStructure] = []

        risk_free_rate = process.risk_free_rate()
        for i in range(number_of_times):
            time_positions[i] = the_time_grid.index(times[i])
            discount_factors[i] = risk_free_rate.discount(times[i])
            forward_term_structures.append(ImpliedTermStructure(risk_free_rate, fixings[i]))

        payoff = self._arguments.path_payoff
        qassert.require(payoff is not None, "no payoff given")
        assert payoff is not None

        return LongstaffSchwartzMultiPathPricer(
            payoff,
            time_positions,
            forward_term_structures,
            discount_factors,
            _POLYNOMIAL_ORDER,
            _POLYNOMIAL_TYPE,
        )


class MakeMCAmericanPathEngine:
    """Fluent factory for :class:`MCAmericanPathEngine`.

    # C++ parity: ``MakeMCAmericanPathEngine<RNG>``
    # (mcamericanpathengine.hpp:61-83, 158-259).

    Args:
        process: the basket process.
        rng_traits: the random-number policy (C++ ``RNG`` template argument).
    """

    __slots__ = (
        "_antithetic",
        "_brownian_bridge",
        "_calibration_samples",
        "_control_variate",
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
        self, process: StochasticProcessArray, rng_traits: RngTraits = PseudoRandom
    ) -> None:
        # C++ parity: mcamericanpathengine.hpp:158-165.
        self._process: StochasticProcessArray = process
        self._rng_traits: RngTraits = rng_traits
        self._brownian_bridge: bool = False
        self._antithetic: bool = False
        self._control_variate: bool = False
        self._steps: int | None = None
        self._steps_per_year: int | None = None
        self._samples: int | None = None
        self._max_samples: int | None = None
        self._calibration_samples: int | None = None
        self._tolerance: float | None = None
        self._seed: int = 0

    # ---- named parameters --------------------------------------------------

    def with_steps(self, steps: int) -> MakeMCAmericanPathEngine:
        self._steps = steps
        return self

    def with_steps_per_year(self, steps: int) -> MakeMCAmericanPathEngine:
        self._steps_per_year = steps
        return self

    def with_brownian_bridge(self, brownian_bridge: bool = True) -> MakeMCAmericanPathEngine:
        self._brownian_bridge = brownian_bridge
        return self

    def with_antithetic_variate(self, b: bool = True) -> MakeMCAmericanPathEngine:
        self._antithetic = b
        return self

    def with_control_variate(self, b: bool = True) -> MakeMCAmericanPathEngine:
        self._control_variate = b
        return self

    def with_samples(self, samples: int) -> MakeMCAmericanPathEngine:
        qassert.require(self._tolerance is None, "tolerance already set")
        self._samples = samples
        return self

    def with_absolute_tolerance(self, tolerance: float) -> MakeMCAmericanPathEngine:
        qassert.require(self._samples is None, "number of samples already set")
        qassert.require(
            bool(self._rng_traits.allows_error_estimate),
            "chosen random generator policy does not allow an error estimate",
        )
        self._tolerance = tolerance
        return self

    def with_max_samples(self, samples: int) -> MakeMCAmericanPathEngine:
        self._max_samples = samples
        return self

    def with_seed(self, seed: int) -> MakeMCAmericanPathEngine:
        self._seed = seed
        return self

    def with_calibration_samples(self, samples: int) -> MakeMCAmericanPathEngine:
        self._calibration_samples = samples
        return self

    # ---- terminal ----------------------------------------------------------

    def build(self) -> MCAmericanPathEngine:
        """Construct the engine.

        # C++ parity: ``operator ext::shared_ptr<PricingEngine>()``
        # (mcamericanpathengine.hpp:236-259).
        """
        qassert.require(
            self._steps is not None or self._steps_per_year is not None,
            "number of steps not given",
        )
        qassert.require(
            self._steps is None or self._steps_per_year is None,
            "number of steps overspecified",
        )
        return MCAmericanPathEngine(
            self._process,
            self._steps,
            self._steps_per_year,
            self._brownian_bridge,
            self._antithetic,
            self._control_variate,
            self._samples,
            self._tolerance,
            self._max_samples,
            self._seed,
            self._calibration_samples,
            self._rng_traits,
        )

    def __call__(self) -> MCAmericanPathEngine:
        """Mirror the C++ conversion operator to ``shared_ptr<PricingEngine>``."""
        return self.build()


__all__ = ["MCAmericanPathEngine", "MakeMCAmericanPathEngine"]

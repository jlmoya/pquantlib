"""MCEuropeanEngine — Monte Carlo pricing for European vanilla options.

# C++ parity: ql/pricingengines/vanilla/mceuropeanengine.hpp (v1.42.1) —
# ``template <class RNG = PseudoRandom, class S = Statistics>
#  class MCEuropeanEngine : public MCVanillaEngine<SingleVariate, RNG, S>``.

Drives a single-asset MC simulation for a European vanilla option
under any 1-D process (typically a ``GeneralizedBlackScholesProcess``)
and prices each path via the terminal payoff times the maturity
discount factor.

For BSM with constant vol the analytic engine gives a closed form;
the MC engine exists to cross-validate variance-reduction techniques
(antithetic, control-variate) and as the on-ramp for path-dependent
exotics (Asian, barrier, ...).

The Python port keeps the canonical no-frills constructor (``process,
time_steps, samples, seed, antithetic_variate=False``) and *omits* the
``MakeMCEuropeanEngine`` builder — Python kwargs make builders
unnecessary.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.methods.montecarlo.path import Path
from pquantlib.methods.montecarlo.path_pricer import PathPricer
from pquantlib.payoffs import PlainVanillaPayoff, StrikedTypePayoff
from pquantlib.pricingengines.vanilla.mc_vanilla_engine import MCVanillaEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)


class EuropeanPathPricer(PathPricer[Path]):
    """Plain-vanilla terminal-payoff pricer.

    # C++ parity: ``EuropeanPathPricer`` (mceuropeanengine.hpp:93-104, 246-257).

    Stores a ``PlainVanillaPayoff`` (which captures option type +
    strike) plus the maturity discount factor.  Each path is priced
    as ``discount * payoff(path.back())``.
    """

    __slots__ = ("_discount", "_payoff")

    def __init__(
        self,
        option_type: object,  # OptionType enum from pquantlib.payoffs
        strike: float,
        discount: float,
    ) -> None:
        qassert.require(strike >= 0.0, "strike less than zero not allowed")
        self._payoff: PlainVanillaPayoff = PlainVanillaPayoff(option_type, strike)  # type: ignore[arg-type]
        self._discount: float = discount

    def __call__(self, path: Path) -> float:
        qassert.require(not path.empty(), "the path cannot be empty")
        return self._payoff(path.back()) * self._discount


class MCEuropeanEngine(MCVanillaEngine):
    """Monte Carlo pricing engine for European vanilla options.

    # C++ parity: ``MCEuropeanEngine<RNG, S>`` (mceuropeanengine.hpp).
    """

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        *,
        time_steps: int | None = None,
        time_steps_per_year: int | None = None,
        brownian_bridge: bool = False,
        antithetic_variate: bool = False,
        required_samples: int | None = None,
        required_tolerance: float | None = None,
        max_samples: int | None = None,
        seed: int = 0,
    ) -> None:
        # MCEuropeanEngine sets control_variate=False (C++ delegates
        # via MCVanillaEngine ctor with controlVariate=false).
        super().__init__(
            process,
            time_steps=time_steps,
            time_steps_per_year=time_steps_per_year,
            brownian_bridge=brownian_bridge,
            antithetic_variate=antithetic_variate,
            control_variate=False,
            required_samples=required_samples,
            required_tolerance=required_tolerance,
            max_samples=max_samples,
            seed=seed,
        )

    def path_pricer(self) -> PathPricer[Path]:
        """Build the terminal-payoff pricer with the maturity discount.

        # C++ parity: ``MCEuropeanEngine::pathPricer`` (mceuropeanengine.hpp:132-153).
        """
        payoff = self._arguments.payoff
        qassert.require(
            isinstance(payoff, PlainVanillaPayoff),
            "non-plain payoff given",
        )
        assert isinstance(payoff, PlainVanillaPayoff)

        # Use the process directly (we know it's BSM by typing).
        # Cast guard isn't strictly needed but matches C++ check.
        process = self._process
        qassert.require(
            isinstance(process, GeneralizedBlackScholesProcess),
            "Black-Scholes process required",
        )
        assert isinstance(process, GeneralizedBlackScholesProcess)

        # Discount factor at the option's last exercise date.
        # C++ parity: ``process->riskFreeRate()->discount(timeGrid().back())``
        # — but the C++ ``timeGrid().back()`` is the year-fraction value
        # built from ``last_exercise_date``, so we go through the date
        # directly (avoids drifting if the year-fraction conversion ever
        # diverges between the curve and the process).
        last_exercise = self._arguments.exercise
        assert last_exercise is not None
        discount = process.risk_free_rate().discount(last_exercise.last_date())

        # Capture strike + option_type from the payoff.
        # PlainVanillaPayoff inherits from StrikedTypePayoff.
        assert isinstance(payoff, StrikedTypePayoff)
        return EuropeanPathPricer(
            option_type=payoff.option_type(),
            strike=payoff.strike(),
            discount=discount,
        )

    # Time grid for European: regular grid 0 → t with N steps (default
    # via MCVanillaEngine.time_grid; we just inherit).
    # last_t = float(self.time_grid().back())  # placeholder for future expiry-time captures
    _ = None


__all__ = ["EuropeanPathPricer", "MCEuropeanEngine"]

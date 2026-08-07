"""Monte Carlo Hull-White engine for cap/floors.

# C++ parity: ql/pricingengines/capfloor/mchullwhiteengine.{hpp,cpp}
# @ v1.43 (6b57206e0).

Simulates the Hull-White short rate under the ``Tb``-forward measure —
``Tb`` being the last caplet's end date — with a
:class:`~pquantlib.processes.hull_white_forward_process.HullWhiteForwardProcess`
whose forward-measure time is set to that horizon, and prices each path
with :class:`HullWhiteCapFloorPricer`.

Two things a port typically gets wrong, both pinned in the probe:

* The time grid contains only *future* fixing times plus the final end
  date; a caplet whose fixing has already happened does NOT get a grid
  point, and :class:`HullWhiteCapFloorPricer` compensates with a running
  ``past_fixings`` offset when indexing the path.
* Discounting is done under the forward measure: each caplet is divided
  by ``P(end, Tb; r)`` along the path and the whole NPV is multiplied
  once by the curve discount to ``Tb``. Discounting on the curve
  per-caplet instead gives a different — and wrong — number that a
  statistical band would not catch.

v1.43 note: this engine has NO control variate. It passes
``controlVariate = false`` to ``McSimulation`` (mchullwhiteengine.hpp:81)
and :class:`MakeMCHullWhiteCapFloorEngine` exposes no
``withControlVariate``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.instruments.cap_floor import (
    CapFloorArguments,
    CapFloorResults,
    CapFloorType,
)
from pquantlib.methods.montecarlo.gaussian_sequence_generator import (
    make_pseudo_random_rsg,
)
from pquantlib.methods.montecarlo.path import Path
from pquantlib.methods.montecarlo.path_generator import PathGenerator
from pquantlib.methods.montecarlo.path_pricer import PathPricer
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.mc_simulation import McSimulation
from pquantlib.processes.hull_white_forward_process import HullWhiteForwardProcess
from pquantlib.time.time_grid import TimeGrid

if TYPE_CHECKING:
    from pquantlib.methods.montecarlo.monte_carlo_model import PathGeneratorTypeProtocol
    from pquantlib.models.shortrate.onefactor.hull_white import HullWhite


class HullWhiteCapFloorPricer(PathPricer[Path]):
    """Per-path cap/floor payoff under the ``Tb``-forward measure.

    # C++ parity: ``detail::HullWhiteCapFloorPricer`` in
    # mchullwhiteengine.hpp:38-51 + mchullwhiteengine.cpp:26-107 (v1.43).
    """

    __slots__ = (
        "_args",
        "_end_discount",
        "_end_times",
        "_fixing_times",
        "_forward_measure_time",
        "_model",
        "_start_times",
    )

    def __init__(
        self,
        args: CapFloorArguments,
        model: HullWhite,
        forward_measure_time: float,
    ) -> None:
        # # C++ parity: mchullwhiteengine.cpp:26-50.
        self._args: CapFloorArguments = args
        self._model: HullWhite = model
        self._forward_measure_time: float = forward_measure_time
        curve = model.term_structure
        self._end_discount: float = curve.discount(forward_measure_time)

        reference_date = curve.reference_date()
        day_counter = curve.day_counter()
        self._start_times: list[float] = [
            day_counter.year_fraction(reference_date, d) for d in args.start_dates
        ]
        self._end_times: list[float] = [
            day_counter.year_fraction(reference_date, d) for d in args.end_dates
        ]
        self._fixing_times: list[float] = [
            day_counter.year_fraction(reference_date, d) for d in args.fixing_dates
        ]

    def __call__(self, path: Path) -> float:
        # # C++ parity: mchullwhiteengine.cpp:52-107.
        args = self._args
        is_cap = args.type == CapFloorType.Cap
        npv = 0.0
        tb = self._forward_measure_time

        past_fixings = 0
        for i in range(len(self._fixing_times)):
            tau = args.accrual_times[i]
            start = self._start_times[i]
            end = self._end_times[i]
            fixing = self._fixing_times[i]
            if end <= 0.0:
                # The fixing is in the past and the caplet has expired.
                past_fixings += 1
                continue

            if fixing <= 0.0:
                # Current caplet: the fixing has happened, so the rate is
                # known, but the short rate at caplet expiry is not.
                past_fixings += 1
                current_libor = args.forwards[i]
                r_i2 = path[i - past_fixings + 2]
            else:
                # Future caplet: everything is forecast. past_fixings is
                # the offset into the (fixing-time) path.
                r_i1 = path[i - past_fixings + 1]
                r_i2 = path[i - past_fixings + 2]
                d1 = self._model.discount_bond_scalar(fixing, start, r_i1)
                d2 = self._model.discount_bond_scalar(fixing, end, r_i1)
                current_libor = (d1 / d2 - 1.0) / tau

            accrual_factor = 1.0 / self._model.discount_bond_scalar(end, tb, r_i2)

            strike = args.cap_rates[i] if is_cap else args.floor_rates[i]
            payoff = (
                max(current_libor - strike, 0.0)
                if is_cap
                else max(strike - current_libor, 0.0)
            )

            npv += payoff * tau * args.gearings[i] * args.nominals[i] * accrual_factor

        return npv * self._end_discount


class MCHullWhiteCapFloorEngine(
    GenericEngine[CapFloorArguments, CapFloorResults], McSimulation[Path]
):
    """Monte Carlo Hull-White engine for cap/floors.

    # C++ parity: ``MCHullWhiteCapFloorEngine<RNG, S>`` in
    # mchullwhiteengine.hpp:58-151 (v1.43).
    """

    def __init__(
        self,
        model: HullWhite,
        brownian_bridge: bool,
        antithetic_variate: bool,
        required_samples: int | None,
        required_tolerance: float | None,
        max_samples: int | None,
        seed: int,
    ) -> None:
        # # C++ parity: mchullwhiteengine.hpp:74-85 — note the
        # # ``McSimulation(antitheticVariate, false)``: control variate is
        # # hard-wired OFF in v1.43.
        GenericEngine.__init__(  # pyright: ignore[reportUnknownMemberType]
            self, CapFloorArguments(), CapFloorResults()
        )
        McSimulation.__init__(  # pyright: ignore[reportUnknownMemberType]
            self, antithetic_variate=antithetic_variate, control_variate=False
        )
        self._model: HullWhite = model
        self._required_samples: int | None = required_samples
        self._max_samples: int | None = max_samples
        self._required_tolerance: float | None = required_tolerance
        self._brownian_bridge: bool = brownian_bridge
        self._seed: int = seed
        model.register_with(self)

    def calculate(self) -> None:
        # # C++ parity: mchullwhiteengine.hpp:87-95.
        self.run_mc(
            required_tolerance=self._required_tolerance,
            required_samples=self._required_samples,
            max_samples=self._max_samples,
        )
        assert self._mc_model is not None
        accumulator = self._mc_model.sample_accumulator()
        self._results.value = accumulator.mean()
        # PseudoRandom::allowsErrorEstimate is 1, so C++ always fills this.
        self._results.error_estimate = accumulator.error_estimate()

    # --- McSimulation hooks -------------------------------------------

    def _forward_measure_time(self) -> float:
        curve = self._model.term_structure
        return curve.day_counter().year_fraction(
            curve.reference_date(), self._arguments.end_dates[-1]
        )

    def path_pricer(self) -> PathPricer[Path]:
        # # C++ parity: mchullwhiteengine.hpp:98-107.
        return HullWhiteCapFloorPricer(
            self._arguments, self._model, self._forward_measure_time()
        )

    def time_grid(self) -> TimeGrid:
        # # C++ parity: mchullwhiteengine.hpp:109-127 — only FUTURE fixing
        # # times are added, then the final end date.
        curve = self._model.term_structure
        reference_date = curve.reference_date()
        day_counter = curve.day_counter()
        times = [
            day_counter.year_fraction(reference_date, d)
            for d in self._arguments.fixing_dates
            if d > reference_date
        ]
        times.append(
            day_counter.year_fraction(reference_date, self._arguments.end_dates[-1])
        )
        return TimeGrid.with_mandatory(times)

    def path_generator(self) -> PathGeneratorTypeProtocol[Path]:
        # # C++ parity: mchullwhiteengine.hpp:129-150.
        curve = self._model.term_structure
        forward_measure_time = self._forward_measure_time()
        params = self._model.params()
        a, sigma = float(params[0]), float(params[1])
        process = HullWhiteForwardProcess(curve, a, sigma)
        process.set_forward_measure_time(forward_measure_time)

        grid = self.time_grid()
        generator = make_pseudo_random_rsg(len(grid) - 1, self._seed)
        return PathGenerator.with_time_grid(
            process, grid, generator, brownian_bridge=self._brownian_bridge
        )


class MakeMCHullWhiteCapFloorEngine:
    """Fluent builder for :class:`MCHullWhiteCapFloorEngine`.

    # C++ parity: ``MakeMCHullWhiteCapFloorEngine<RNG, S>`` in
    # mchullwhiteengine.hpp:156-176 + 181-245 (v1.43).

    C++ ends the chain with ``operator ext::shared_ptr<PricingEngine>()``;
    Python has no implicit conversion operator, so the chain ends with the
    explicit terminal :meth:`engine`.

    Defaults reproduce the C++ member initialisers: ``antithetic_ = false``,
    ``brownianBridge_ = false``, ``seed_ = 0`` and ``samples_ /
    maxSamples_ / tolerance_`` all ``Null`` (``None`` here). There is
    deliberately no ``with_control_variate``: v1.43 has none.
    """

    __slots__ = (
        "_antithetic",
        "_brownian_bridge",
        "_max_samples",
        "_model",
        "_samples",
        "_seed",
        "_tolerance",
    )

    def __init__(self, model: HullWhite) -> None:
        # # C++ parity: mchullwhiteengine.hpp:181-185.
        self._model: HullWhite = model
        self._antithetic: bool = False
        self._samples: int | None = None
        self._max_samples: int | None = None
        self._tolerance: float | None = None
        self._brownian_bridge: bool = False
        self._seed: int = 0

    def with_samples(self, samples: int) -> MakeMCHullWhiteCapFloorEngine:
        # # C++ parity: mchullwhiteengine.hpp:187-194.
        qassert.require(self._tolerance is None, "tolerance already set")
        self._samples = samples
        return self

    def with_absolute_tolerance(self, tolerance: float) -> MakeMCHullWhiteCapFloorEngine:
        # # C++ parity: mchullwhiteengine.hpp:196-207 — the
        # # ``RNG::allowsErrorEstimate`` guard is vacuous for PseudoRandom,
        # # the only generator policy this port wires.
        qassert.require(self._samples is None, "number of samples already set")
        self._tolerance = tolerance
        return self

    def with_max_samples(self, samples: int) -> MakeMCHullWhiteCapFloorEngine:
        # # C++ parity: mchullwhiteengine.hpp:209-214.
        self._max_samples = samples
        return self

    def with_seed(self, seed: int) -> MakeMCHullWhiteCapFloorEngine:
        # # C++ parity: mchullwhiteengine.hpp:216-221.
        self._seed = seed
        return self

    def with_brownian_bridge(self, b: bool = True) -> MakeMCHullWhiteCapFloorEngine:
        # # C++ parity: mchullwhiteengine.hpp:223-228.
        self._brownian_bridge = b
        return self

    def with_antithetic_variate(self, b: bool = True) -> MakeMCHullWhiteCapFloorEngine:
        # # C++ parity: mchullwhiteengine.hpp:230-235.
        self._antithetic = b
        return self

    def engine(self) -> MCHullWhiteCapFloorEngine:
        """# C++ parity: ``operator ext::shared_ptr<PricingEngine>()``
        # (mchullwhiteengine.hpp:237-245).
        """
        return MCHullWhiteCapFloorEngine(
            self._model,
            self._brownian_bridge,
            self._antithetic,
            self._samples,
            self._tolerance,
            self._max_samples,
            self._seed,
        )


__all__ = [
    "HullWhiteCapFloorPricer",
    "MCHullWhiteCapFloorEngine",
    "MakeMCHullWhiteCapFloorEngine",
]

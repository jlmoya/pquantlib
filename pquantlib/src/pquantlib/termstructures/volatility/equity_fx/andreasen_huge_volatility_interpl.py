"""Andreasen-Huge local-volatility calibration and interpolation.

# C++ parity: ql/termstructures/volatility/equityfx/andreasenhugevolatilityinterpl.hpp +
#             andreasenhugevolatilityinterpl.cpp (v1.43).

Reference: Andreasen J., Huge B., 2010. *Volatility Interpolation*,
https://ssrn.com/abstract=1694972.

The idea: instead of fitting a smooth implied-vol surface and differentiating
it (Dupire), fit a piecewise-in-time *local* vol directly, one expiry slice at
a time, by requiring that a single implicit finite-difference step of the
Dupire forward PDE reproduces the market prices at that expiry. One step per
expiry means the model is arbitrage-free by construction and the calibration
decomposes into a sequence of small least-squares problems.

Layout of one step, in undiscounted forward-moneyness coordinates
``x = log(K / F)``:

* ``C(x)`` starts at the payoff, ``max(±(1 - e^x), 0)`` on the grid.
* One implicit step of ``dC/dT = 0.5 sigma(x)^2 (d^2C/dx^2 - dC/dx)`` is
  ``(I - dT * L) C_new = C_old`` with ``L`` the operator above — that is
  ``AndreasenHugeCostFunction.solve_for``, a tridiagonal splitting solve.
* The residual is ``C_new`` interpolated to the market log-strikes minus the
  market prices; the optimiser moves the per-strike ``sigma`` until it
  vanishes.
* The next expiry starts from the previous expiry's ``C_new``, so the
  calibration marches forward in time.

Local vol comes back out of the same step (``get_local_vol_slice``) as
``sqrt(2 dC/dT / d^2C/dK^2)`` — Dupire's formula evaluated on the FDM
operators rather than on a fitted surface.

Two behaviours worth flagging because they look like bugs and are not:

* ``option_price`` and ``local_vol`` cache per *time*, then re-enter
  themselves so the second call hits the cache (cpp:555, 631). The
  re-entrance is C++'s and is reproduced.
* Under ``CalibrationType.CallPut`` the stored cost function is the *put*
  one (cpp:400), and ``get_price_slice`` then drives it with the *call* NPV
  vector. That is correct: ``solve_for`` depends only on the mesher and the
  sigmas, never on which side the cost function was built from.

The optimiser is injectable (C++ defaults to ``LevenbergMarquardt``); see
``pquantlib/tests/.../test_andreasen_huge.py`` for what that costs in
reproducibility.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

from pquantlib import qassert
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.math.array import Array
from pquantlib.math.closeness import close_enough
from pquantlib.math.constants import QL_MAX_REAL
from pquantlib.math.interpolations.backward_flat import BackwardFlatInterpolation
from pquantlib.math.interpolations.cubic_interpolation import (
    CubicNaturalSpline,
    MonotonicCubicNaturalSpline,
)
from pquantlib.math.interpolations.interpolation import Interpolation
from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.math.optimization.constraint import PositiveConstraint
from pquantlib.math.optimization.cost_function import CostFunction
from pquantlib.math.optimization.end_criteria import EndCriteria
from pquantlib.math.optimization.levenberg_marquardt import LevenbergMarquardt
from pquantlib.math.optimization.optimization_method import OptimizationMethod
from pquantlib.math.optimization.problem import Problem
from pquantlib.methods.finitedifferences.meshers.concentrating_1d_mesher import (
    Concentrating1dMesher,
)
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.operators.first_derivative_op import (
    FirstDerivativeOp,
)
from pquantlib.methods.finitedifferences.operators.second_derivative_op import (
    SecondDerivativeOp,
)
from pquantlib.methods.finitedifferences.operators.triple_band_linear_op import (
    TripleBandLinearOp,
)
from pquantlib.patterns.lazy_object import LazyObject
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.black_calculator import BlackCalculator
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.date import Date

if TYPE_CHECKING:
    from pquantlib.exercise import Exercise


class InterpolationType(IntEnum):
    """How the per-strike sigmas are interpolated across log-strike.

    # C++ parity: ``AndreasenHugeVolatilityInterpl::InterpolationType``
    # (andreasenhugevolatilityinterpl.hpp:56).
    """

    PiecewiseConstant = 0
    Linear = 1
    CubicSpline = 2


class CalibrationType(IntEnum):
    """Which option side(s) the slice is calibrated to.

    # C++ parity: ``AndreasenHugeVolatilityInterpl::CalibrationType``
    # (andreasenhugevolatilityinterpl.hpp:57-58) — the enumerators are seeded
    # from ``Option::Call`` (+1) and ``Option::Put`` (-1), so ``CallPut``, the
    # unseeded third enumerator, lands on ``Put + 1 == 0``. The exact integers
    # are preserved because C++ code can and does compare against them.
    """

    Call = 1
    Put = -1
    CallPut = 0


class AndreasenHugeCostFunction(CostFunction):
    """One expiry's calibration residual: FDM step price minus market price.

    # C++ parity: ``class AndreasenHugeCostFunction`` — declared in the header
    # only as a forward declaration (andreasenhugevolatilityinterpl.hpp:43),
    # defined in andreasenhugevolatilityinterpl.cpp:54-166.
    """

    def __init__(
        self,
        market_npvs: Array,
        market_vegas: Array,
        ln_market_strikes: Array,
        previous_npvs: Array,
        mesher: FdmMesherComposite,
        dt: float,
        interpolation_type: InterpolationType,
    ) -> None:
        self._market_npvs: Array = np.asarray(market_npvs, dtype=np.float64)
        self._market_vegas: Array = np.asarray(market_vegas, dtype=np.float64)
        self._ln_market_strikes: Array = np.asarray(ln_market_strikes, dtype=np.float64)
        self._previous_npvs: Array = np.asarray(previous_npvs, dtype=np.float64)
        self._mesher: FdmMesherComposite = mesher
        self._n_grid_points: int = mesher.layout().size()
        self._dt: float = dt
        # C++ parity: cpp:67-69 — a single market strike degenerates to
        # PiecewiseConstant whatever the caller asked for.
        self._interpolation_type: InterpolationType = (
            interpolation_type
            if self._ln_market_strikes.size > 1
            else InterpolationType.PiecewiseConstant
        )

        # C++ parity: cpp:70-72.
        self._dx_map: FirstDerivativeOp = FirstDerivativeOp(0, mesher)
        self._dxx_map: TripleBandLinearOp = SecondDerivativeOp(0, mesher)
        self._d2_cdk2: TripleBandLinearOp = self._dx_map.mult(
            np.full(self._n_grid_points, -1.0)
        ).add(self._dxx_map)
        self._map_t: TripleBandLinearOp = TripleBandLinearOp(0, mesher)
        # The grid's log-strikes; in 1-D ``mesher.location(iter, 0)`` over the
        # layout is exactly the 1-D mesher's own location vector (cpp:107-109).
        self._grid_points: Array = mesher.locations(0)

    # --- the implicit FDM step ---------------------------------------------

    def d2_cdk2(self, c: Array) -> Array:
        """# C++ parity: cpp:74-76."""
        return self._d2_cdk2.apply(c)

    def _sigma_interpolation(self, sig: Array) -> Interpolation:
        """# C++ parity: cpp:81-104 — the sigma interpolation policy."""
        strikes = self._ln_market_strikes
        if self._interpolation_type == InterpolationType.CubicSpline:
            return CubicNaturalSpline(strikes, np.asarray(sig, dtype=np.float64))
        if self._interpolation_type == InterpolationType.Linear:
            return LinearInterpolation(strikes, np.asarray(sig, dtype=np.float64))
        if self._interpolation_type == InterpolationType.PiecewiseConstant:
            # C++ parity: cpp:95-100 — the knots move to the midpoints, the
            # last one stays on the last strike, and BackwardFlat carries each
            # sigma to the LEFT of its knot.
            x = np.empty(strikes.size, dtype=np.float64)
            for i in range(strikes.size - 1):
                x[i] = 0.5 * (strikes[i] + strikes[i + 1])
            x[-1] = strikes[-1]
            return BackwardFlatInterpolation(x, np.asarray(sig, dtype=np.float64))
        qassert.fail("unknown interpolation type")
        raise AssertionError  # unreachable

    def solve_for(self, dt: float, sig: Array, b: Array) -> Array:
        """One implicit step: solve ``(I - dt * L(sig)) c = b``.

        # C++ parity: ``AndreasenHugeCostFunction::solveFor`` (cpp:78-121).
        """
        sig_interpl = self._sigma_interpolation(sig)

        lo = float(self._ln_market_strikes[0])
        hi = float(self._ln_market_strikes[-1])
        z = np.empty(self._n_grid_points, dtype=np.float64)
        for i in range(self._n_grid_points):
            ln_strike = float(self._grid_points[i])
            vol = sig_interpl(min(max(ln_strike, lo), hi), allow_extrapolation=True)
            z[i] = 0.5 * vol * vol

        # C++ parity: cpp:118-119 — mapT_ = z*dxMap + dxxMap*(-z), then
        # solve (I - dt*mapT) via solve_splitting(b, 1.0).
        self._map_t.axpyb(z, self._dx_map, self._dxx_map.mult(-z), None)
        return self._map_t.mult(np.full(z.size, dt)).solve_splitting(b, 1.0)

    def apply(self, c: Array) -> Array:
        """# C++ parity: cpp:123-125 — ``-mapT_.apply(c)``.

        Note this reads ``mapT_`` as left by the LAST ``solve_for`` call: the
        operator is mutable state, and ``getLocalVolSlice`` relies on that
        ordering (cpp:573-578).
        """
        return -self._map_t.apply(c)

    # --- CostFunction interface --------------------------------------------

    def values(self, x: Array) -> Array:
        """Residual at the market strikes.

        # C++ parity: cpp:127-142.
        """
        new_npvs = self.solve_for(self._dt, x, self._previous_npvs)
        grid_points = self._mesher.get_fdm_1d_meshers()[0].locations()
        interpl = MonotonicCubicNaturalSpline(
            np.asarray(grid_points, dtype=np.float64), new_npvs
        )
        return (
            np.array(
                [interpl(float(k)) for k in self._ln_market_strikes],
                dtype=np.float64,
            )
            - self._market_npvs
        )

    def vega_calibration_error(self, sig: Array) -> Array:
        """# C++ parity: cpp:144-146."""
        return self.values(sig) / self._market_vegas

    def initial_values(self) -> Array:
        """# C++ parity: cpp:148-150 — a flat 0.25 seed."""
        return np.full(self._ln_market_strikes.size, 0.25, dtype=np.float64)


class CombinedCostFunction(CostFunction):
    """Stacks the put and call residuals for a ``CallPut`` calibration.

    # C++ parity: ``class CombinedCostFunction``
    # (andreasenhugevolatilityinterpl.cpp:168-207).
    """

    def __init__(
        self,
        put_cost_fct: AndreasenHugeCostFunction | None,
        call_cost_fct: AndreasenHugeCostFunction | None,
    ) -> None:
        self._put: AndreasenHugeCostFunction | None = put_cost_fct
        self._call: AndreasenHugeCostFunction | None = call_cost_fct

    def values(self, x: Array) -> Array:
        """# C++ parity: cpp:174-190.

        C++ writes the call block at ``retVal.begin() + cv.size()``, which is
        the *call* size, not the put size. With put and call blocks of equal
        length — always, since both come from the same expiry's strike set —
        that is the same offset; reproduced as a plain concatenation.
        """
        if self._put is not None and self._call is not None:
            return np.concatenate((self._put.values(x), self._call.values(x)))
        if self._put is not None:
            return self._put.values(x)
        if self._call is not None:
            return self._call.values(x)
        qassert.fail("internal error: cost function not set")
        raise AssertionError  # unreachable

    def initial_values(self) -> Array:
        """# C++ parity: cpp:192-202."""
        if self._put is not None and self._call is not None:
            return 0.5 * (self._put.initial_values() + self._call.initial_values())
        if self._put is not None:
            return self._put.initial_values()
        if self._call is not None:
            return self._call.initial_values()
        qassert.fail("internal error: cost function not set")
        raise AssertionError  # unreachable


@dataclass
class SingleStepCalibrationResult:
    """One expiry's calibration state.

    # C++ parity: ``AndreasenHugeVolatilityInterpl::SingleStepCalibrationResult``
    # (andreasenhugevolatilityinterpl.hpp:104-107) — an aggregate holding the
    # put/call NPV vectors the slice *started* from, the calibrated sigmas, and
    # the cost function that produced them. ``get_price_slice`` and
    # ``get_local_vol_slice`` replay the slice from exactly these four fields.
    """

    put_npvs: Array
    call_npvs: Array
    sigmas: Array
    cost_function: AndreasenHugeCostFunction | None


# (fwd, values, interpolation) keyed by time — C++ ``TimeValueCacheType``
# (andreasenhugevolatilityinterpl.hpp:98-102).
_TimeValueCacheEntry = tuple[float, Array, Interpolation]


class AndreasenHugeVolatilityInterpl(LazyObject):
    """Local-volatility surface calibrated to a sparse grid of options.

    # C++ parity: ``class AndreasenHugeVolatilityInterpl``
    # (andreasenhugevolatilityinterpl.hpp:53).

    Args:
        calibration_set: ``(VanillaOption, vol Quote)`` pairs. Every option
            must be European with a plain-vanilla payoff; only its strike and
            expiry are read, never its option type.
        spot: underlying spot quote.
        r_ts: risk-free curve — also the source of the reference date, day
            counter and calendar the two adapters report.
        q_ts: dividend / foreign curve.
        interpolation_type: sigma interpolation across log-strike.
        calibration_type: which option side(s) the residual measures.
        n_grid_points: FDM grid size (C++ default 500).
        min_strike / max_strike: grid bounds; ``None`` means C++'s
            ``Null<Real>()``, i.e. ``strikes[0] / 8`` and ``8 * strikes[-1]``.
        optimization_method: defaults to :class:`LevenbergMarquardt`.
        end_criteria: defaults to C++'s ``EndCriteria(500, 100, 1e-12, 1e-10,
            1e-10)``.
    """

    def __init__(
        self,
        calibration_set: Sequence[tuple[VanillaOption, Quote]],
        spot: Quote,
        r_ts: YieldTermStructure,
        q_ts: YieldTermStructure,
        interpolation_type: InterpolationType = InterpolationType.CubicSpline,
        calibration_type: CalibrationType = CalibrationType.Call,
        n_grid_points: int = 500,
        min_strike: float | None = None,
        max_strike: float | None = None,
        optimization_method: OptimizationMethod | None = None,
        end_criteria: EndCriteria | None = None,
    ) -> None:
        super().__init__()
        self._spot: Quote = spot
        self._r_ts: YieldTermStructure = r_ts
        self._q_ts: YieldTermStructure = q_ts
        self._interpolation_type: InterpolationType = interpolation_type
        self._calibration_type: CalibrationType = calibration_type
        self._n_grid_points: int = n_grid_points
        self._min_strike_arg: float | None = min_strike
        self._max_strike_arg: float | None = max_strike
        self._optimization_method: OptimizationMethod = (
            optimization_method if optimization_method is not None else LevenbergMarquardt()
        )
        self._end_criteria: EndCriteria = (
            end_criteria
            if end_criteria is not None
            else EndCriteria(500, 100, 1e-12, 1e-10, 1e-10)
        )

        # C++ parity: cpp:226.
        qassert.require(
            n_grid_points > 2 and len(calibration_set) > 0,
            "undefined grid or calibration set",
        )

        strike_set: set[float] = set()
        expiry_set: set[Date] = set()
        self._calibration_set: list[tuple[VanillaOption, Quote]] = []

        # C++ parity: cpp:232-253.
        for option, quote in calibration_set:
            exercise: Exercise = option.exercise()
            qassert.require(exercise.type() == exercise.Type.European, "European option required")
            expiry_set.add(exercise.last_date())
            payoff = option.payoff()
            qassert.require(
                isinstance(payoff, PlainVanillaPayoff), "plain vanilla payoff required"
            )
            assert isinstance(payoff, PlainVanillaPayoff)
            strike_set.add(payoff.strike())
            self._calibration_set.append((VanillaOption(payoff, exercise), quote))
            quote.register_with(self)

        self._strikes: list[float] = sorted(strike_set)
        self._expiries: list[Date] = sorted(expiry_set)

        n_expiries = len(self._expiries)
        self._d_t: list[float] = [0.0] * n_expiries
        self._expiry_times: list[float] = [0.0] * n_expiries

        # C++ parity: cpp:261-279 — expiry x strike -> index into the
        # calibration set, with Null<Size> for the holes.
        self._calibration_matrix: list[list[int | None]] = [
            [None] * len(self._strikes) for _ in range(n_expiries)
        ]
        for i, (option, _) in enumerate(calibration_set):
            expiry = option.exercise().last_date()
            # C++ uses lower_bound on the ordered set, i.e. the position of the
            # first expiry not before this one.
            row = next(
                idx for idx, e in enumerate(self._expiries) if not e < expiry
            )
            payoff = option.payoff()
            assert isinstance(payoff, PlainVanillaPayoff)
            strike = payoff.strike()
            # C++ parity: cpp:274-277 — find_if(close_enough_to(strike)), a
            # 42-ulp comparison, not an exact match.
            col = next(
                idx for idx, k in enumerate(self._strikes) if close_enough(k, strike)
            )
            self._calibration_matrix[row][col] = i

        spot.register_with(self)
        r_ts.register_with(self)
        q_ts.register_with(self)

        # calculation state (C++ ``mutable`` members)
        self._avg_error: float = 0.0
        self._min_error: float = 0.0
        self._max_error: float = 0.0
        self._mesher: FdmMesherComposite | None = None
        self._grid_points: Array = np.empty(0, dtype=np.float64)
        self._grid_in_fwd: Array = np.empty(0, dtype=np.float64)
        self._calibration_results: list[SingleStepCalibrationResult] = []
        self._local_vol_cache: dict[float, _TimeValueCacheEntry] = {}
        self._price_cache: dict[float, _TimeValueCacheEntry] = {}

    # --- inspectors ---------------------------------------------------------

    def max_date(self) -> Date:
        """# C++ parity: cpp:447-449."""
        return self._expiries[-1]

    def min_strike(self) -> float:
        """# C++ parity: cpp:451-454."""
        return self._strikes[0] / 8.0 if self._min_strike_arg is None else self._min_strike_arg

    def max_strike(self) -> float:
        """# C++ parity: cpp:456-459."""
        return 8.0 * self._strikes[-1] if self._max_strike_arg is None else self._max_strike_arg

    def fwd(self, t: float) -> float:
        """# C++ parity: cpp:461-463."""
        return self._spot.value() * self._q_ts.discount(t) / self._r_ts.discount(t)

    def risk_free_rate(self) -> YieldTermStructure:
        """# C++ parity: cpp:465-468."""
        return self._r_ts

    def calibration_error(self) -> tuple[float, float, float]:
        """``(min, max, avg)`` calibration error in volatility units.

        # C++ parity: cpp:470-475.
        """
        self.calculate()
        return (self._min_error, self._max_error, self._avg_error)

    # --- calibration --------------------------------------------------------

    def _build_cost_function(
        self, i_expiry: int, option_type: OptionType, previous_npvs: Array
    ) -> AndreasenHugeCostFunction | None:
        """# C++ parity: ``buildCostFunction`` (cpp:286-339)."""
        # C++ parity: cpp:291-294 — a one-sided calibration returns a null
        # cost function for the other side.
        if self._calibration_type != CalibrationType.CallPut and (
            (self._calibration_type == CalibrationType.Call and option_type == OptionType.Put)
            or (self._calibration_type == CalibrationType.Put and option_type == OptionType.Call)
        ):
            return None

        expiry_time = self._expiry_times[i_expiry]
        discount = self._r_ts.discount(expiry_time)
        fwd = self._spot.value() * self._q_ts.discount(expiry_time) / discount

        row = self._calibration_matrix[i_expiry]
        n_options = sum(1 for n in row if n is not None)

        ln_market_strikes = np.empty(n_options, dtype=np.float64)
        market_npvs = np.empty(n_options, dtype=np.float64)
        market_vegas = np.empty(n_options, dtype=np.float64)

        # C++ parity: cpp:311-329 — undiscounted, forward-normalised prices.
        k = 0
        for j, strike in enumerate(self._strikes):
            idx = row[j]
            if idx is None:
                continue
            vol = self._calibration_set[idx][1].value()
            std_dev = vol * math.sqrt(expiry_time)
            calculator = BlackCalculator.from_type_strike(
                option_type, strike, fwd, std_dev, discount
            )
            market_npvs[k] = calculator.value() / (discount * fwd)
            market_vegas[k] = calculator.vega(expiry_time) / (discount * fwd)
            ln_market_strikes[k] = math.log(strike / fwd)
            k += 1

        assert self._mesher is not None
        return AndreasenHugeCostFunction(
            market_npvs,
            market_vegas,
            ln_market_strikes,
            previous_npvs,
            self._mesher,
            self._d_t[i_expiry],
            self._interpolation_type,
        )

    def _perform_calculations(self) -> None:
        """# C++ parity: ``performCalculations`` (cpp:342-445)."""
        qassert.require(
            self.max_strike() > self.min_strike(),
            "max strike must be greater than min strike",
        )

        dc = self._r_ts.day_counter()
        for i in range(len(self._expiry_times)):
            self._expiry_times[i] = dc.year_fraction(
                self._r_ts.reference_date(), self._expiries[i]
            )
            self._d_t[i] = self._expiry_times[i] - (0.0 if i == 0 else self._expiry_times[i - 1])

        # C++ parity: cpp:353-359 — a single Concentrating1dMesher over
        # log-moneyness, concentrated at the money with density 0.025.
        self._mesher = FdmMesherComposite(
            Concentrating1dMesher(
                math.log(self.min_strike() / self._spot.value()),
                math.log(self.max_strike() / self._spot.value()),
                self._n_grid_points,
                (0.0, 0.025),
            )
        )

        self._grid_points = self._mesher.locations(0)
        self._grid_in_fwd = np.exp(self._grid_points) * self._spot.value()

        self._local_vol_cache.clear()
        # C++ does NOT clear priceCache_ here — see cpp:364-365. Reproduced:
        # clearing it would change what a second calculate() returns.
        self._calibration_results = []

        self._avg_error = 0.0
        self._min_error = QL_MAX_REAL
        self._max_error = 0.0

        # C++ parity: cpp:373-380 — the payoffs on the moneyness grid, with
        # the forward normalised to 1.
        npv_puts = np.empty(self._n_grid_points, dtype=np.float64)
        npv_calls = np.empty(self._n_grid_points, dtype=np.float64)
        for i in range(self._n_grid_points):
            strike = math.exp(float(self._grid_points[i]))
            npv_puts[i] = PlainVanillaPayoff(OptionType.Put, strike)(1.0)
            npv_calls[i] = PlainVanillaPayoff(OptionType.Call, strike)(1.0)

        for i in range(len(self._expiries)):
            put_cost_fct = self._build_cost_function(i, OptionType.Put, npv_puts)
            call_cost_fct = self._build_cost_function(i, OptionType.Call, npv_calls)

            cost_function = CombinedCostFunction(put_cost_fct, call_cost_fct)
            problem = Problem(
                cost_function, PositiveConstraint(), cost_function.initial_values()
            )
            self._optimization_method.minimize(problem, self._end_criteria)
            sig = np.asarray(problem.current_value, dtype=np.float64)

            self._calibration_results.append(
                SingleStepCalibrationResult(
                    put_npvs=npv_puts,
                    call_npvs=npv_calls,
                    sigmas=sig,
                    cost_function=(
                        call_cost_fct
                        if self._calibration_type == CalibrationType.Call
                        else put_cost_fct
                    ),
                )
            )

            # C++ parity: cpp:405-429 — the error is measured in vega units,
            # and under CallPut the side is chosen per grid node by whether the
            # node sits below or above the forward.
            if self._calibration_type == CalibrationType.CallPut:
                assert put_cost_fct is not None
                assert call_cost_fct is not None
                vega_put_diffs = put_cost_fct.vega_calibration_error(sig)
                vega_call_diffs = call_cost_fct.vega_calibration_error(sig)
                fwd = (
                    self._spot.value()
                    * self._q_ts.discount(self._expiry_times[i])
                    / self._r_ts.discount(self._expiry_times[i])
                )
                vega_diffs = np.array(
                    [
                        abs(
                            vega_put_diffs[j]
                            if fwd > self._grid_in_fwd[j]
                            else vega_call_diffs[j]
                        )
                        for j in range(sig.size)
                    ],
                    dtype=np.float64,
                )
            elif self._calibration_type == CalibrationType.Put:
                assert put_cost_fct is not None
                vega_diffs = np.abs(put_cost_fct.vega_calibration_error(sig))
            elif self._calibration_type == CalibrationType.Call:
                assert call_cost_fct is not None
                vega_diffs = np.abs(call_cost_fct.vega_calibration_error(sig))
            else:
                qassert.fail("unknown calibration type")
                raise AssertionError  # unreachable

            self._avg_error += float(np.sum(vega_diffs))
            self._min_error = min(self._min_error, float(np.min(vega_diffs)))
            self._max_error = max(self._max_error, float(np.max(vega_diffs)))

            if put_cost_fct is not None:
                npv_puts = put_cost_fct.solve_for(self._d_t[i], sig, npv_puts)
            if call_cost_fct is not None:
                npv_calls = call_cost_fct.solve_for(self._d_t[i], sig, npv_calls)

        self._avg_error /= len(self._calibration_set)

    # --- evaluation ---------------------------------------------------------

    def _get_exercise_time_idx(self, t: float) -> int:
        """# C++ parity: cpp:477-482 — ``min(n-1, upper_bound(times, t))``."""
        upper = sum(1 for et in self._expiry_times if et <= t)
        return min(len(self._expiry_times) - 1, upper)

    def _get_cache_value(self, strike: float, entry: _TimeValueCacheEntry) -> float:
        """# C++ parity: cpp:484-494.

        The lookup is clamped to ``[gridPoints[1], gridPoints[-2]]`` because
        the cached interpolation is built over the interior nodes only.
        """
        fwd = entry[0]
        k = math.log(strike / fwd)
        s = max(float(self._grid_points[1]), min(float(self._grid_points[-2]), k))
        return entry[2](s)

    def _get_price_slice(self, t: float, option_type: OptionType) -> Array:
        """# C++ parity: cpp:496-506."""
        iu = self._get_exercise_time_idx(t)
        result = self._calibration_results[iu]
        assert result.cost_function is not None
        return result.cost_function.solve_for(
            t if iu == 0 else t - self._expiry_times[iu - 1],
            result.sigmas,
            result.call_npvs if option_type == OptionType.Call else result.put_npvs,
        )

    def option_price(self, t: float, strike: float, option_type: OptionType) -> float:
        """Undiscounted-then-rediscounted vanilla price at ``(t, strike)``.

        # C++ parity: cpp:508-556. Put-call parity converts between the sides
        # when the calibration only carries one of them.
        """
        entry = self._price_cache.get(t)
        df = self._r_ts.discount(t)

        if entry is not None:
            fwd = entry[0]
            price = self._get_cache_value(strike, entry)
            if option_type == OptionType.Put and self._calibration_type in (
                CalibrationType.Call,
                CalibrationType.CallPut,
            ):
                price = price + strike / fwd - 1.0
            elif option_type == OptionType.Call and self._calibration_type == CalibrationType.Put:
                price = 1.0 - strike / fwd + price
            return price * df * fwd

        self.calculate()

        if self._calibration_type == CalibrationType.Put:
            prices = self._get_price_slice(t, OptionType.Put)
        elif self._calibration_type in (CalibrationType.Call, CalibrationType.CallPut):
            prices = self._get_price_slice(t, OptionType.Call)
        else:
            qassert.fail("unknown calibration type")
            raise AssertionError  # unreachable

        fwd = self._spot.value() * self._q_ts.discount(t) / df

        # C++ parity: cpp:549-553 — the cached interpolation skips the two
        # boundary nodes.
        self._price_cache[t] = (
            fwd,
            prices,
            CubicNaturalSpline(self._grid_points[1:-1], prices[1:-1]),
        )
        # C++ re-enters itself so the value comes back through the cache path.
        return self.option_price(t, strike, option_type)

    def _get_local_vol_slice(self, t: float, option_type: OptionType) -> Array:
        """# C++ parity: cpp:558-589 — Dupire on the FDM operators."""
        iu = self._get_exercise_time_idx(t)
        result = self._calibration_results[iu]
        previous_npvs = (
            result.call_npvs if option_type == OptionType.Call else result.put_npvs
        )
        cost_function = result.cost_function
        assert cost_function is not None

        dt = t if iu == 0 else t - self._expiry_times[iu - 1]
        sig = result.sigmas

        c_at_j = cost_function.solve_for(dt, sig, previous_npvs)

        # C++ parity: cpp:575-578 — note the inner solve_for runs first and
        # leaves mapT_ set up for the apply() in the middle.
        dc_dt = cost_function.solve_for(
            dt, sig, cost_function.apply(cost_function.solve_for(dt, sig, previous_npvs))
        )

        d2_cdk2 = cost_function.d2_cdk2(c_at_j)

        with np.errstate(invalid="ignore", divide="ignore"):
            local_vol = np.sqrt(2 * dc_dt / d2_cdk2)

        # C++ parity: cpp:584-586 — the two boundary nodes are left alone,
        # non-finite or negative interior nodes fall back to 0.25.
        for i in range(1, local_vol.size - 1):
            if not math.isfinite(local_vol[i]) or local_vol[i] < 0.0:
                local_vol[i] = 0.25

        return local_vol

    def local_vol(self, t: float, strike: float) -> float:
        """Local volatility at ``(t, strike)``.

        # C++ parity: cpp:591-632.
        """
        entry = self._local_vol_cache.get(t)
        if entry is not None:
            return self._get_cache_value(strike, entry)

        self.calculate()

        if self._calibration_type == CalibrationType.CallPut:
            put_local_vol = self._get_local_vol_slice(t, OptionType.Put)
            call_local_vol = self._get_local_vol_slice(t, OptionType.Call)
            # C++ parity: cpp:608-610 — call side above the forward, put below.
            local_vol = np.where(self._grid_points > 0.0, call_local_vol, put_local_vol)
        elif self._calibration_type == CalibrationType.Put:
            local_vol = self._get_local_vol_slice(t, OptionType.Put)
        elif self._calibration_type == CalibrationType.Call:
            local_vol = self._get_local_vol_slice(t, OptionType.Call)
        else:
            qassert.fail("unknown calibration type")
            raise AssertionError  # unreachable

        fwd = self._spot.value() * self._q_ts.discount(t) / self._r_ts.discount(t)

        self._local_vol_cache[t] = (
            fwd,
            local_vol,
            LinearInterpolation(self._grid_points[1:-1], local_vol[1:-1]),
        )
        return self.local_vol(t, strike)


__all__ = [
    "AndreasenHugeCostFunction",
    "AndreasenHugeVolatilityInterpl",
    "CalibrationType",
    "CombinedCostFunction",
    "InterpolationType",
    "SingleStepCalibrationResult",
]

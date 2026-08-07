"""Discount curve fitted to a set of fixed-coupon bonds.

# C++ parity: ql/termstructures/yield/fittedbonddiscountcurve.{hpp,cpp} (v1.43)

A user-supplied :class:`FittingMethod` parameterises a discount function
``d(t)``; the curve drives that parameter vector with an ``OptimizationMethod``
(``Simplex`` by default) until the bonds, repriced off ``d(t)``, reproduce
their quotes.  Price errors are weighted by the inverse of each bond's modified
duration.

Two things about the shape of this port are worth stating up front.

**The nested class is flattened.** C++ declares ``FittingMethod`` inside
``FittedBondDiscountCurve``, which buys it access to the curve's private
members (``curve_->maxEvaluations_`` and friends, cpp:187-235).  Python has no
such privilege rule, so ``FittingMethod`` is a module-level class — as this
port does for every nested C++ type — and is re-exposed as
``FittedBondDiscountCurve.FittingMethod`` for callers transcribing C++.

**The curve owns a CLONE of the fitting method.** C++ holds
``Clone<FittingMethod>`` (hpp:164), whose converting constructor calls
``x.clone()``.  The caller's instance is therefore never touched by the fit;
``fit_results()`` returns the curve's private copy.  :meth:`FittingMethod.clone`
reproduces that, including deep-copying the parameter arrays — C++ gets the
deep copy for free from ``Array``'s copy constructor, NumPy does not.

Divergences from C++, both of them places where C++ relies on indeterminate
values:

- ``numberOfIterations_`` and ``costValue_`` are declared without an
  initialiser (hpp:273-275), so reading them before a fit is undefined in C++.
  Python seeds them with ``0`` and ``NULL_REAL``.
- ``SpreadFittingMethod::rebase_`` is likewise uninitialised until ``init()``
  runs (nonlinearfittingmethods.hpp:292); the Python field starts at ``1.0``,
  the value ``init()`` assigns when the two reference dates agree.
"""

from __future__ import annotations

import copy
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.cashflows.duration import Duration
from pquantlib.math.constants import QL_MAX_REAL
from pquantlib.math.optimization.constraint import Constraint, NoConstraint
from pquantlib.math.optimization.cost_function import CostFunction
from pquantlib.math.optimization.end_criteria import NULL_REAL, EndCriteria
from pquantlib.math.optimization.end_criteria import Type as EndCriteriaType
from pquantlib.math.optimization.optimization_method import OptimizationMethod
from pquantlib.math.optimization.problem import Problem
from pquantlib.math.optimization.simplex import Simplex
from pquantlib.patterns.lazy_object import LazyObject
from pquantlib.termstructures.term_structure import TermStructure
from pquantlib.termstructures.yield_.bond_helper import BondPriceType as HelperPriceType
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency

if TYPE_CHECKING:
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.termstructures.yield_.bond_helper import BondHelper
    from pquantlib.time.calendar import Calendar

Array = npt.NDArray[np.float64]

# C++ ``Date()`` — the null date, serial 0.
_NULL_DATE: Final[Date] = Date()


def _as_array(values: Sequence[float] | Array | None) -> Array:
    """Normalise an optional parameter vector to a fresh float64 array.

    ``None`` is the Python spelling of C++'s default-constructed ``Array()``,
    which is what ``weights.empty()`` / ``l2_.empty()`` test for.
    """
    if values is None:
        return np.empty(0, dtype=np.float64)
    return np.asarray(values, dtype=np.float64).astype(np.float64, copy=True)


@dataclass(frozen=True, slots=True)
class _CurveState:
    """The enclosing curve's private fields, bundled for the fitting method.

    See :meth:`FittingMethod._curve_state`.  ``bond_helpers`` and
    ``guess_solution`` are the curve's own objects, not copies: the cost
    function re-reads them on every evaluation, exactly as C++ does through
    ``curve_``.
    """

    max_evaluations: int
    accuracy: float
    simplex_lambda: float
    max_stationary_state_iterations: int
    guess_solution: Array
    bond_helpers: list[BondHelper]


def _ordinal(n: int) -> str:
    """C++ parity: ``io::ordinal`` (ql/utilities/dataformatters.cpp).

    Used only to build the diagnostics ``performCalculations`` raises
    (cpp:146-157); pquantlib has no ``dataformatters`` module to host it.
    """
    if n % 100 in (11, 12, 13):
        return f"{n}th"
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


class FittingMethod(ABC):
    """Strategy supplying the parametric discount function and its size.

    # C++ parity: ``FittedBondDiscountCurve::FittingMethod``
    # (fittedbonddiscountcurve.hpp:203-284).

    Subclasses define :meth:`_discount_function`, :meth:`size` and
    :meth:`clone`.  An array of ``weights`` overrides the default
    inverse-duration weighting; an array ``l2`` adds a Gaussian penalty on each
    parameter measured from the guess (equivalently, a Gaussian prior).
    """

    def __init__(
        self,
        constrain_at_zero: bool = True,
        weights: Sequence[float] | Array | None = None,
        optimization_method: OptimizationMethod | None = None,
        l2: Sequence[float] | Array | None = None,
        min_cutoff_time: float = 0.0,
        max_cutoff_time: float = QL_MAX_REAL,
        constraint: Constraint | None = None,
    ) -> None:
        # C++ parity: cpp:169-183.
        self._constrain_at_zero: bool = constrain_at_zero
        self._weights: Array = _as_array(weights)
        self._l2: Array = _as_array(l2)
        # ``calculateWeights_(weights.empty())`` is frozen at construction, so a
        # method built with explicit weights never recomputes them.
        self._calculate_weights: bool = self._weights.size == 0
        self._optimization_method: OptimizationMethod | None = optimization_method
        # C++ tests ``constraint_.empty()`` (the PIMPL sentinel) and substitutes
        # NoConstraint; pquantlib's Constraint is abstract, so ``None`` is the
        # sentinel instead.
        self._constraint: Constraint = constraint if constraint is not None else NoConstraint()
        self._min_cutoff_time: float = min_cutoff_time
        self._max_cutoff_time: float = max_cutoff_time

        self._curve: FittedBondDiscountCurve | None = None
        self._solution: Array = np.empty(0, dtype=np.float64)
        self._guess_solution: Array = np.empty(0, dtype=np.float64)
        self._cost_function: _FittingCost | None = None
        # C++ leaves these two indeterminate (hpp:273-275); see module docstring.
        self._number_of_iterations: int = 0
        self._cost_value: float = NULL_REAL
        # C++ parity: hpp:277 — this one IS initialised.
        self._error_code: EndCriteriaType = EndCriteriaType.None_

    # --- abstract surface --------------------------------------------------

    @abstractmethod
    def size(self) -> int:
        """Total number of coefficients to fit/solve for.

        # C++ parity: hpp:210 — ``virtual Size size() const = 0``.
        """

    @abstractmethod
    def clone(self) -> FittingMethod:
        """Copy of this method, as ``Clone<FittingMethod>`` makes.

        # C++ parity: hpp:220 —
        # ``virtual std::unique_ptr<FittingMethod> clone() const = 0``.
        """

    @abstractmethod
    def _discount_function(self, x: Array, t: float) -> float:
        """Parametric discount factor, before the cutoff extrapolations.

        # C++ parity: hpp:246-247 — the protected pure-virtual
        # ``discountFunction``.
        """

    # --- inspectors --------------------------------------------------------

    def solution(self) -> Array:
        """# C++ parity: hpp:333-335 — returns a COPY of ``solution_``."""
        return self._solution.copy()

    def number_of_iterations(self) -> int:
        """# C++ parity: hpp:318-321."""
        return self._number_of_iterations

    def minimum_cost_value(self) -> float:
        """# C++ parity: hpp:323-326."""
        return self._cost_value

    def error_code(self) -> EndCriteriaType:
        """# C++ parity: hpp:328-331."""
        return self._error_code

    def constrain_at_zero(self) -> bool:
        """# C++ parity: hpp:337-339."""
        return self._constrain_at_zero

    def weights(self) -> Array:
        """# C++ parity: hpp:341-343 — returns a COPY of ``weights_``."""
        return self._weights.copy()

    def l2(self) -> Array:
        """# C++ parity: hpp:345-347 — returns a COPY of ``l2_``."""
        return self._l2.copy()

    def optimization_method(self) -> OptimizationMethod | None:
        """# C++ parity: hpp:349-352 — a null shared_ptr means "use Simplex"."""
        return self._optimization_method

    def constraint(self) -> Constraint:
        """# C++ parity: hpp:354-356 — returns a reference, so no copy here."""
        return self._constraint

    # --- discount ----------------------------------------------------------

    def discount(self, x: Array, t: float) -> float:
        """Discount factor at ``t`` for parameters ``x``, cutoffs applied.

        # C++ parity: hpp:358-371.

        Outside ``[min_cutoff_time, max_cutoff_time]`` it is the INSTANTANEOUS
        FORWARD that is held flat, not the discount factor: below the min
        cutoff by rescaling the log-discount linearly in ``t``, above the max
        cutoff by extending the forward measured across a 1e-4 step at the
        cutoff.
        """
        if t < self._min_cutoff_time:
            # flat fwd extrapolation before min cutoff time
            return math.exp(
                math.log(self._discount_function(x, self._min_cutoff_time))
                / self._min_cutoff_time
                * t
            )
        if t > self._max_cutoff_time:
            # flat fwd extrapolation after max cutoff time
            return self._discount_function(x, self._max_cutoff_time) * math.exp(
                (
                    math.log(self._discount_function(x, self._max_cutoff_time + 1e-4))
                    - math.log(self._discount_function(x, self._max_cutoff_time))
                )
                * 1e4
                * (t - self._max_cutoff_time)
            )
        return self._discount_function(x, t)

    # --- internals ---------------------------------------------------------

    def _cloned(self) -> FittingMethod:
        """Shared implementation of :meth:`clone` for every concrete subclass.

        A shallow copy reproduces C++'s implicit copy constructor — which
        shares the ``shared_ptr`` members (``optimizationMethod_``,
        ``costFunction_``) and copies the raw ``curve_`` pointer — EXCEPT for
        the ``Array`` members, whose C++ copy constructor is deep.  NumPy
        arrays are shared by a shallow copy and ``init()`` mutates ``weights_``
        element-wise, so those four are copied explicitly; otherwise a fit
        would write through into the caller's object.
        """
        other = copy.copy(self)
        other._weights = self._weights.copy()
        other._l2 = self._l2.copy()
        other._solution = self._solution.copy()
        other._guess_solution = self._guess_solution.copy()
        return other

    def _init(self) -> None:
        """Recompute the per-bond weights; rerun whenever the inputs change.

        # C++ parity: cpp:185-237 — the protected virtual ``init()``.
        """
        # Local import: termstructures/ must not depend on pricingengines/ at
        # module-load time (bond_helper.py carves the same hole).
        from pquantlib.instruments.bond import BondPrice, BondPriceType  # noqa: PLC0415
        from pquantlib.pricingengines.bond.bond_functions import (  # noqa: PLC0415
            BondFunctions,
        )

        curve = self._require_curve()
        state = self._curve_state(curve)
        if state.max_evaluations == 0:
            return  # we can skip the rest

        # yield conventions — hard-coded in C++ (cpp:190-193).
        yield_dc = curve.day_counter()
        yield_comp = Compounding.Compounded
        yield_freq = Frequency.Annual

        n = len(state.bond_helpers)
        self._cost_function = _FittingCost(self)

        for bond_helper in state.bond_helpers:
            bond_helper.set_term_structure(curve)

        if self._calculate_weights:
            if self._weights.size == 0:
                self._weights = np.zeros(n, dtype=np.float64)

            squared_sum = 0.0
            for i, helper in enumerate(state.bond_helpers):
                bond = helper.bond()

                amount = helper.quote().value()
                # C++ passes the helper's price type straight into Bond::Price
                # (cpp:210-211) because there is only ONE enum over there.
                # pquantlib has two and they DISAGREE on the integer values:
                # instruments/bond.py:64 follows C++ (Dirty = 0, Clean = 1,
                # bond.hpp:64) while termstructures/yield_/bond_helper.py:43
                # declares Clean = 0, Dirty = 1. Each is internally consistent,
                # so nothing breaks until a value crosses between them — which
                # is precisely what this line does. Mapping BY NAME is correct
                # under either resolution.
                price_type = (
                    BondPriceType.Clean
                    if helper.price_type() == HelperPriceType.Clean
                    else BondPriceType.Dirty
                )
                price = BondPrice(amount, price_type)

                bond_settlement = bond.settlement_date()
                ytm = BondFunctions.bond_yield(
                    bond, price, yield_dc, yield_comp, yield_freq, bond_settlement
                )

                dur = BondFunctions.duration_from_rate(
                    bond,
                    ytm,
                    yield_dc,
                    yield_comp,
                    yield_freq,
                    Duration.Modified,
                    bond_settlement,
                )
                self._weights[i] = 1.0 / dur
                # Sequential accumulation, in the C++ loop order — np.sum would
                # pair-sum and round differently.
                squared_sum += float(self._weights[i]) * float(self._weights[i])
            self._weights /= math.sqrt(squared_sum)

        qassert.require(
            self._weights.size == n, "Given weights do not cover all boostrapping helpers"
        )

        if self._l2.size != 0:
            qassert.require(
                self._l2.size == self.size(),
                "Given penalty factors do not cover all parameters",
            )
            qassert.require(state.guess_solution.size != 0, "L2 penalty requires a guess")

    def _calculate(self) -> None:
        """Run the optimization; store solution, iteration count and cost.

        # C++ parity: cpp:239-296.
        """
        curve = self._require_curve()
        state = self._curve_state(curve)

        if state.max_evaluations == 0:
            # Don't calculate, simply use the given parameters to provide a
            # fitted curve. This turns the instance into an evaluator of the
            # parametric curve — e.g. so a credit-spread curve fitted to bonds
            # in one currency can be coupled to a discount curve in another.
            qassert.require(
                state.guess_solution.size == self.size(), "wrong number of parameters"
            )
            self._solution = state.guess_solution.copy()
            self._number_of_iterations = 0
            self._cost_value = NULL_REAL
            self._error_code = EndCriteriaType.None_
            return

        assert self._cost_function is not None, "_init() must run before _calculate()"
        cost_function = self._cost_function

        # start with the guess solution, if it exists
        x = np.zeros(self.size(), dtype=np.float64)
        if state.guess_solution.size != 0:
            qassert.require(state.guess_solution.size == self.size(), "wrong size for guess")
            x = state.guess_solution.astype(np.float64, copy=True)

        # workaround for backwards compatibility
        optimization = self._optimization_method
        if optimization is None:
            optimization = Simplex(state.simplex_lambda)
        problem = Problem(cost_function, self._constraint, x)

        root_epsilon = state.accuracy
        function_epsilon = state.accuracy
        gradient_norm_epsilon = state.accuracy

        end_criteria = EndCriteria(
            state.max_evaluations,
            state.max_stationary_state_iterations,
            root_epsilon,
            function_epsilon,
            gradient_norm_epsilon,
        )

        self._error_code = optimization.minimize(problem, end_criteria)
        self._solution = problem.current_value.astype(np.float64, copy=True)

        self._number_of_iterations = problem.function_evaluation
        self._cost_value = problem.function_value

        # save the results as the guess solution, in case of recalculation
        self._store_guess_solution(curve, self._solution.copy())

    def _require_curve(self) -> FittedBondDiscountCurve:
        """The curve this method was attached to.

        C++ dereferences ``curve_`` unconditionally — it is a raw pointer the
        curve's constructor always sets (cpp:64).  A method never handed to a
        curve would dereference null there; Python says so instead.
        """
        qassert.require(
            self._curve is not None,
            "fitting method is not attached to a FittedBondDiscountCurve",
        )
        assert self._curve is not None
        return self._curve

    @staticmethod
    def _curve_state(curve: FittedBondDiscountCurve) -> _CurveState:
        """The enclosing curve's private fields, as one named bundle.

        C++ gets this access for free: ``FittingMethod`` is a NESTED class, so
        it may read ``curve_->maxEvaluations_`` and the rest of the curve's
        private state directly (cpp:187-235).  Python has no such rule and the
        type checker is right to object, so every crossing is funnelled through
        this one place rather than sprinkled across ``init`` and ``calculate``.

        Read-only.  The one field the C++ code WRITES back through the pointer
        — ``guessSolution_``, at cpp:295 — goes through
        :meth:`_store_guess_solution` instead.
        """
        return _CurveState(
            max_evaluations=curve._max_evaluations,  # pyright: ignore[reportPrivateUsage]
            accuracy=curve._accuracy,  # pyright: ignore[reportPrivateUsage]
            simplex_lambda=curve._simplex_lambda,  # pyright: ignore[reportPrivateUsage]
            max_stationary_state_iterations=(
                curve._max_stationary_state_iterations  # pyright: ignore[reportPrivateUsage]
            ),
            guess_solution=curve._guess_solution,  # pyright: ignore[reportPrivateUsage]
            bond_helpers=curve._bond_helpers,  # pyright: ignore[reportPrivateUsage]
        )

    @staticmethod
    def _store_guess_solution(curve: FittedBondDiscountCurve, guess: Array) -> None:
        """# C++ parity: cpp:295 — ``curve_->guessSolution_ = solution_``."""
        curve._guess_solution = guess  # pyright: ignore[reportPrivateUsage]


class _FittingCost(CostFunction):
    """Weighted squared quote errors plus the L2 parameter penalty.

    # C++ parity: ``FittedBondDiscountCurve::FittingMethod::FittingCost``
    # (cpp:35-46, 299-336) — a private nested class, so C++ callers can only
    # reach it through ``Problem``. Same here: it is module-private.
    """

    __slots__ = ("_fitting_method",)

    def __init__(self, fitting_method: FittingMethod) -> None:
        # C++ parity: cpp:299-301.
        self._fitting_method: FittingMethod = fitting_method

    def value(self, x: Array) -> float:
        """Sum — NOT the RMS — of :meth:`values`.

        # C++ parity: cpp:304-312. This overrides ``CostFunction``'s default
        # ``sqrt(mean(v^2))``; the residuals are already squared here, so the
        # default would square them twice.
        """
        squared_error = 0.0
        for val in self.values(x):
            # Sequential accumulation, in the C++ loop order.
            squared_error += float(val)
        return squared_error

    def values(self, x: Array) -> Array:
        """# C++ parity: cpp:314-336.

        The private access below is C++'s too: ``FittingCost`` is nested inside
        ``FittingMethod`` AND declares it a friend (cpp:37), and
        ``FittingMethod`` in turn is nested inside the curve.  The state bundle
        is re-read on every call rather than cached because ``guessSolution_``
        is rebound between fits (cpp:295) and the L2 penalty measures from it.
        """
        fm = self._fitting_method
        state = FittingMethod._curve_state(  # pyright: ignore[reportPrivateUsage]
            fm._require_curve()  # pyright: ignore[reportPrivateUsage]
        )
        n = len(state.bond_helpers)
        l2 = fm._l2  # pyright: ignore[reportPrivateUsage]
        n_l2 = int(l2.size)

        # set solution so that the curve represents the current trial; the
        # final solution is written back in FittingMethod::calculate().
        fm._solution = np.asarray(x, dtype=np.float64)  # pyright: ignore[reportPrivateUsage]
        weights = fm._weights  # pyright: ignore[reportPrivateUsage]

        values = np.zeros(n + n_l2, dtype=np.float64)
        for i in range(n):
            helper = state.bond_helpers[i]
            weighted_error = float(weights[i]) * helper.quote_error()
            values[i] = weighted_error * weighted_error

        if n_l2 != 0:
            for i in range(n_l2):
                error = float(x[i]) - float(state.guess_solution[i])
                values[i + n] = float(l2[i]) * error * error
        return values


class FittedBondDiscountCurve(YieldTermStructure, LazyObject):
    """Discount curve fitted to a set of bonds.

    # C++ parity: ``class FittedBondDiscountCurve`` (hpp:81-165).

    C++ declares four constructors, which are two independent binary choices:
    reference-date vs settlement-days anchoring, and fit vs "don't fit, just
    evaluate these parameters".  The second choice is not really a separate
    object in C++ either — the no-fit constructors simply set
    ``maxEvaluations_ = 0`` and stash the parameters in ``guessSolution_``
    (cpp:95-97) — so this port keeps ONE keyword-only constructor covering both
    anchorings, plus :meth:`from_parameters` for the no-fit pair.

    Args:
        day_counter: day counter for the curve's own time measure.  It is also
            the yield convention ``init()`` uses (cpp:191).
        fitting_method: strategy to CLONE and fit.
        reference_date: fixed anchoring.  Mutually exclusive with
            ``settlement_days`` / ``calendar``.
        settlement_days: moving anchoring, relative to the evaluation date.
        calendar: required with ``settlement_days``.
        bonds: the helpers to fit.  Empty in the no-fit mode.
        accuracy: fed to all three ``EndCriteria`` epsilons (cpp:278-280).
        max_evaluations: ``EndCriteria`` iteration cap; ``0`` selects the
            no-fit mode.
        guess: starting parameter vector — and, in the no-fit mode, THE
            parameter vector.
        simplex_lambda: edge length of the default ``Simplex``'s initial
            simplex; ignored when an explicit optimizer was given.
        max_stationary_state_iterations: ``EndCriteria``'s second argument.
        max_date: explicit latest date; required in the no-fit mode when no
            helpers are supplied.
    """

    def __init__(
        self,
        *,
        day_counter: DayCounter,
        fitting_method: FittingMethod,
        reference_date: Date | None = None,
        settlement_days: int | None = None,
        calendar: Calendar | None = None,
        bonds: Sequence[BondHelper] = (),
        accuracy: float = 1.0e-10,
        max_evaluations: int = 10000,
        guess: Sequence[float] | Array | None = None,
        simplex_lambda: float = 1.0,
        max_stationary_state_iterations: int = 100,
        max_date: Date | None = None,
    ) -> None:
        qassert.require(
            (reference_date is None) != (settlement_days is None),
            "give exactly one of reference_date or settlement_days",
        )
        if settlement_days is not None:
            # C++ parity: cpp:60 / cpp:95 —
            # ``YieldTermStructure(settlementDays, calendar, dayCounter)``.
            #
            # pquantlib's YieldTermStructure.__init__ does not forward
            # settlement_days to TermStructure (it accepts only the fixed and
            # delegated modes), so moving mode is reached by calling
            # TermStructure directly and then seeding the five jump-related
            # fields YieldTermStructure would have set for ``jumps=None``.
            # Filed as a divergence against yield_term_structure.py.
            TermStructure.__init__(
                self,
                reference_date=None,
                calendar=calendar,
                day_counter=day_counter,
                settlement_days=settlement_days,
            )
            self._jumps = []
            self._jump_dates = []
            self._jump_times = []
            self._n_jumps = 0
            self._latest_reference = None
        else:
            # C++ parity: cpp:79 / cpp:109 — ``YieldTermStructure(referenceDate,
            # Calendar(), dayCounter)``; the empty calendar is ``None`` here.
            YieldTermStructure.__init__(
                self, reference_date=reference_date, calendar=None, day_counter=day_counter
            )
        LazyObject.__init__(self)

        self._accuracy: float = accuracy
        self._max_evaluations: int = max_evaluations
        self._simplex_lambda: float = simplex_lambda
        self._max_stationary_state_iterations: int = max_stationary_state_iterations
        self._guess_solution: Array = _as_array(guess)
        self._max_date: Date = max_date if max_date is not None else _NULL_DATE
        self._bond_helpers: list[BondHelper] = list(bonds)
        # C++ parity: hpp:164 — ``Clone<FittingMethod> fittingMethod_``, whose
        # converting constructor calls ``clone()``. The caller keeps its own.
        self._fitting_method: FittingMethod = fitting_method.clone()

        # C++ parity: cpp:64 — the curve claims the clone. Writing the clone's
        # private ``curve_`` is C++'s own move; ``FittedBondDiscountCurve`` is
        # declared a friend of ``FittingMethod`` (hpp:204) precisely for this
        # and for the ``init`` / ``calculate`` / ``solution_`` accesses below.
        self._fitting_method._curve = self  # pyright: ignore[reportPrivateUsage]
        self._setup()

    @classmethod
    def from_parameters(
        cls,
        fitting_method: FittingMethod,
        parameters: Sequence[float] | Array,
        max_date: Date,
        day_counter: DayCounter,
        *,
        reference_date: Date | None = None,
        settlement_days: int | None = None,
        calendar: Calendar | None = None,
    ) -> FittedBondDiscountCurve:
        """Don't fit — evaluate the parametric curve at ``parameters``.

        # C++ parity: the two "don't fit, use precalculated parameters"
        # constructors (hpp:111-124, cpp:88-115), which hard-code
        # ``accuracy_ = 1e-10`` and ``maxEvaluations_ = 0`` and move the
        # parameters into ``guessSolution_``.

        C++ leaves ``simplexLambda_`` and ``maxStationaryStateIterations_``
        uninitialised on this path; they are unreachable because
        ``calculate()`` returns before touching them, so the Python defaults
        change nothing.
        """
        return cls(
            day_counter=day_counter,
            fitting_method=fitting_method,
            reference_date=reference_date,
            settlement_days=settlement_days,
            calendar=calendar,
            accuracy=1.0e-10,
            max_evaluations=0,
            guess=parameters,
            max_date=max_date,
        )

    # --- inspectors --------------------------------------------------------

    def number_of_bonds(self) -> int:
        """# C++ parity: hpp:288-290."""
        return len(self._bond_helpers)

    def max_date(self) -> Date:
        """# C++ parity: hpp:292-295 — forces the fit first."""
        self.calculate()
        return self._max_date

    def fit_results(self) -> FittingMethod:
        """The curve's OWN fitting method (a clone of the caller's).

        # C++ parity: hpp:297-301 — ``const FittingMethod& fitResults()``.
        """
        self.calculate()
        return self._fitting_method

    def reset_guess(self, guess: Sequence[float] | Array) -> None:
        """Restart the next fit from ``guess`` instead of the last solution.

        # C++ parity: cpp:118-122. Needed because ``calculate()`` writes its
        # answer back into ``guessSolution_`` (cpp:295), so an ordinary refit
        # continues from where the previous one stopped.
        """
        g = _as_array(guess)
        qassert.require(
            g.size == 0 or g.size == self._fitting_method.size(), "guess is of wrong size"
        )
        self._guess_solution = g
        self.update()

    # --- Observer / LazyObject wiring --------------------------------------

    def update(self) -> None:
        """# C++ parity: hpp:303-306 — YieldTermStructure first, then LazyObject."""
        YieldTermStructure.update(self)
        LazyObject.update(self)

    def _setup(self) -> None:
        """# C++ parity: hpp:308-311 — register with every helper."""
        for bond_helper in self._bond_helpers:
            bond_helper.register_with(self)

    # --- the fit -----------------------------------------------------------

    def _perform_calculations(self) -> None:
        # C++ parity: cpp:125-166.
        from pquantlib.pricingengines.bond.bond_functions import (  # noqa: PLC0415
            BondFunctions,
        )

        if self._max_evaluations != 0:
            # we need to fit, so we require helpers
            qassert.require(len(self._bond_helpers) != 0, "no bond helpers given")

        if self._max_evaluations == 0:
            # no fit, but we need either an explicit max date or helpers from
            # which to deduce it
            qassert.require(
                self._max_date != _NULL_DATE or len(self._bond_helpers) != 0,
                "no bond helpers or max date given",
            )

        if self._bond_helpers:
            self._max_date = Date.min_date()
            ref_date = self.reference_date()

            # double check bond quotes still valid and/or instruments not expired
            for i, helper in enumerate(self._bond_helpers):
                bond = helper.bond()
                qassert.require(
                    helper.quote().is_valid(),
                    f"{_ordinal(i + 1)} bond (maturity: {bond.maturity_date()}) "
                    "has an invalid price quote",
                )
                bond_settlement = bond.settlement_date()
                qassert.require(
                    bond_settlement >= ref_date,
                    f"{_ordinal(i + 1)} bond settlemente date ({bond_settlement}) "
                    f"before curve reference date ({ref_date})",
                )
                qassert.require(
                    BondFunctions.is_tradable(bond, bond_settlement),
                    f"{_ordinal(i + 1)} bond non tradable at {bond_settlement} "
                    f"settlement date (maturity being {bond.maturity_date()})",
                )
                self._max_date = max(self._max_date, helper.pillar_date())
                helper.set_term_structure(self)

        self._fitting_method._init()  # pyright: ignore[reportPrivateUsage]
        self._fitting_method._calculate()  # pyright: ignore[reportPrivateUsage]

    def _discount_impl(self, t: float) -> float:
        """# C++ parity: hpp:313-316."""
        self.calculate()
        return self._fitting_method.discount(
            self._fitting_method._solution,  # pyright: ignore[reportPrivateUsage]
            t,
        )


# C++ parity: ``FittingMethod`` is a nested class of ``FittedBondDiscountCurve``
# (hpp:84, 203). Flattened above, re-attached here so that code transcribed
# from C++ can keep saying ``FittedBondDiscountCurve.FittingMethod``.
FittedBondDiscountCurve.FittingMethod = FittingMethod  # type: ignore[attr-defined]


__all__ = ["FittedBondDiscountCurve", "FittingMethod"]

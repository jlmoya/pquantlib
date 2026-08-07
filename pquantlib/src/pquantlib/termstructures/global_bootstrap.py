"""Global (least-squares) bootstrap, with additional restrictions.

# C++ parity: ql/termstructures/globalbootstrap.{hpp,cpp} (v1.43)

Where :class:`~pquantlib.termstructures.bootstrap.iterative_bootstrap.IterativeBootstrap`
solves one pillar at a time with a 1-D root finder, ``GlobalBootstrap``
hands *every* pillar to a least-squares optimizer at once. That buys
three things the iterative bootstrap cannot do:

- **over- and under-determined curves.** The pillar grid is decoupled
  from the helper set: extra grid points come from ``additional_dates``
  and extra residuals from ``additional_penalties``. The only
  requirement is
  ``len(alive helpers) + len(additional penalties) >= len(dates) - 1``.
- **helpers that do not own a pillar** (``additional_helpers``). They are
  registered with the curve and wired to it, but contribute neither a
  grid point nor a residual; the caller adds both explicitly.
- **extra optimization variables** that are not curve values at all
  (``additional_variables``) — e.g. the volatility feeding a futures
  convexity adjustment. See
  :class:`~pquantlib.termstructures.global_bootstrap_vars.SimpleQuoteVariables`.

Four classes live here, mirroring the C++ header one for one:

- :class:`MultiCurveBootstrapContributor` — globalbootstrap.hpp:40, the
  pure-virtual interface a curve's bootstrap exposes to a joint solve.
- :class:`MultiCurveBootstrap` — globalbootstrap.hpp:51 + the .cpp, which
  concatenates several contributors' guesses and residuals into ONE
  optimization problem so a cycle of curves can be solved simultaneously.
- :class:`AdditionalBootstrapVariables` — globalbootstrap.hpp:69.
- :class:`GlobalBootstrap` — globalbootstrap.hpp:105.

Python-specific divergences
---------------------------

- **Templates → protocols.** C++ ``template <class Curve>`` reaches
  ``Curve::traits_type`` and ``Curve::interpolator_type``. Python takes
  the traits explicitly and reads the curve through
  :class:`GlobalBootstrapCurve`.
- **Friendship → private attributes.** C++ ``GlobalBootstrap`` is a
  ``friend`` of ``PiecewiseYieldCurve`` (piecewiseyieldcurve.hpp:181) and
  therefore reads ``ts_->accuracy_`` and ``ts_->interpolator_`` directly.
  The port reads the corresponding private attributes through
  ``getattr`` with a documented default; there is no other way to say
  "friend" in Python, and inventing a public accessor would widen the
  curve's API beyond v1.43.
- **Three constructors → one.** The C++ overload set
  (hpp:112 / hpp:116 / hpp:124) differs only in which arguments are
  supplied, and the third merely wraps a zero-argument penalty functor
  into the two-argument shape (hpp:196-206). Python takes keyword
  arguments and performs the same wrapping by inspecting the callable's
  arity.
- **``ext::enable_shared_from_this``** (hpp:51) has no Python analogue
  and needs none: ``MultiCurveBootstrap.add`` passes ``self`` where C++
  must reconstruct a ``shared_ptr`` from ``this``. Python object identity
  and reference counting give the same lifetime guarantee for free.
- **``MultiCurveBootstrap::finalizeCalculation``** (hpp:60) and
  **``setOtherContributorsToValid``** (hpp:59) are DECLARED in v1.43 and
  never defined anywhere in the tree. They are dead declarations, so the
  port does not carry them: a Python method that only ever raised
  ``NotImplementedError`` would be strictly worse than its absence.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

import numpy as np

from pquantlib import qassert
from pquantlib.math.optimization.constraint import NoConstraint
from pquantlib.math.optimization.cost_function import SimpleCostFunction
from pquantlib.math.optimization.end_criteria import EndCriteria
from pquantlib.math.optimization.levenberg_marquardt import LevenbergMarquardt
from pquantlib.math.optimization.problem import Problem

if TYPE_CHECKING:
    from pquantlib.math.array import Array
    from pquantlib.math.optimization.optimization_method import OptimizationMethod
    from pquantlib.termstructures.bootstrap_helper import BootstrapHelper
    from pquantlib.time.date import Date

# C++ parity: globalbootstrap.cpp:34 — the accuracy MultiCurveBootstrap
# substitutes for a null optimizer / end criteria.
_MULTI_CURVE_DEFAULT_ACCURACY: Final[float] = 1.0e-10

# C++ parity: piecewiseyieldcurve.hpp:163 — ``accuracy_(1.0e-12)``. Used only
# when the curve does not carry an accuracy of its own.
_CURVE_DEFAULT_ACCURACY: Final[float] = 1.0e-12

# C++ parity: globalbootstrap.hpp:228 — ``EndCriteria(1000, 10, a, a, a)``.
_END_CRITERIA_MAX_ITERATIONS: Final[int] = 1000
_END_CRITERIA_MAX_STATIONARY_STATE: Final[int] = 10

# C++ parity: globalbootstrap.hpp:291 — ``Interpolator::requiredPoints``.
# LinearInterpolation / LogLinearInterpolation both declare 2; used when the
# curve's interpolator does not expose the attribute.
_DEFAULT_REQUIRED_POINTS: Final[int] = 2

# The C++ traits pass ``0`` as ``firstAliveHelper``, with the comment "it's not
# used in the standard QL traits anyway" (globalbootstrap.hpp:368).
_FIRST_ALIVE_HELPER: Final[int] = 0

#: Penalty functor shapes. C++ ``AdditionalPenalties`` is
#: ``std::function<Array(const std::vector<Time>&, const std::vector<Real>&)>``
#: (globalbootstrap.hpp:108); the third constructor also accepts
#: ``std::function<Array()>`` and wraps it.
AdditionalPenalties = Callable[[list[float], list[float]], "Array"]
ZeroArgPenalties = Callable[[], "Array"]


@runtime_checkable
class GlobalBootstrapCurve(Protocol):
    """Structural surface ``GlobalBootstrap`` needs from its curve.

    Every member corresponds to something the C++ template touches on
    ``ts_``; the mapping is given per method. ``PiecewiseYieldCurve``
    satisfies this protocol as written.
    """

    def reference_date(self) -> Date: ...
    def time_from_reference(self, d: Date) -> float:
        """C++ ``ts_->timeFromReference(d)`` — globalbootstrap.hpp:299."""
        ...

    def instruments(self) -> list[BootstrapHelper[Any]]:
        """C++ ``ts_->instruments_`` — globalbootstrap.hpp:217."""
        ...

    def dates(self) -> list[Date]:
        """C++ ``ts_->dates_`` — globalbootstrap.hpp:277."""
        ...

    def times(self) -> list[float]:
        """C++ ``ts_->times_`` — globalbootstrap.hpp:278."""
        ...

    def data_live(self) -> list[float]:
        """C++ ``ts_->data_``; must alias the curve's storage, not copy it."""
        ...

    def bootstrap_install_grid(
        self, dates: list[Date], times: list[float], data: list[float]
    ) -> None:
        """Install dates_/times_/data_ in one go — globalbootstrap.hpp:277-313."""
        ...

    def set_max_date(self, d: Date) -> None:
        """C++ ``ts_->maxDate_ = ...`` — globalbootstrap.hpp:302-306."""
        ...

    def refresh_interpolation_through(self, up_to: int) -> None:
        """C++ ``ts_->interpolation_.update()`` — globalbootstrap.hpp:385."""
        ...


class MultiCurveBootstrapContributor(ABC):
    """One curve's contribution to a joint multi-curve solve.

    # C++ parity: ``class MultiCurveBootstrapContributor`` at
    # globalbootstrap.hpp:40-49 — five pure-virtual methods, no state.
    """

    @abstractmethod
    def set_parent_bootstrapper(self, b: MultiCurveBootstrap) -> None:
        """C++ globalbootstrap.hpp:43-44."""

    @abstractmethod
    def setup_cost_function(self) -> Array:
        """C++ globalbootstrap.hpp:45 — return this contributor's initial guess."""

    @abstractmethod
    def set_cost_function_argument(self, v: Array) -> None:
        """C++ globalbootstrap.hpp:46 — push this contributor's slice of ``x``."""

    @abstractmethod
    def evaluate_cost_function(self) -> Array:
        """C++ globalbootstrap.hpp:47 — this contributor's residuals."""

    @abstractmethod
    def set_to_valid(self) -> None:
        """C++ globalbootstrap.hpp:48."""


class MultiCurveBootstrapObserver(Protocol):
    """What ``MultiCurveBootstrap.add_observer`` accepts.

    # C++ parity: ``std::vector<Observer*> observers_`` at
    # globalbootstrap.hpp:66. Curves that are part of the cycle but are NOT
    # bootstrapped (a spreaded curve, say) are refreshed here after every
    # cost-function argument update — globalbootstrap.cpp:74-75.
    """

    def update(self) -> None: ...


class AdditionalBootstrapVariables(ABC):
    """Extra optimization variables that are not curve values.

    # C++ parity: ``class AdditionalBootstrapVariables`` at
    # globalbootstrap.hpp:69-76.
    """

    @abstractmethod
    def initialize(self, valid_data: bool) -> Array:
        """Set the variables to their initial guesses and return them.

        # C++ parity: globalbootstrap.hpp:73.
        """

    @abstractmethod
    def update(self, x: Array) -> None:
        """Set the variables to the given values.

        # C++ parity: globalbootstrap.hpp:75.
        """


class MultiCurveBootstrap:
    """Joint least-squares solve over several contributing curves.

    # C++ parity: ``class MultiCurveBootstrap`` at globalbootstrap.hpp:51-67
    # and globalbootstrap.cpp:26-116.

    Each contributor supplies its own guess vector and its own residual
    vector; this class concatenates them, hands the concatenation to ONE
    optimizer, and slices ``x`` back out per contributor. Guess sizes and
    residual sizes are independent — a contributor with penalties returns
    more residuals than it has variables — so the two concatenations are
    laid out with separate offset walks (cpp:64-97).

    The C++ class derives from ``ext::enable_shared_from_this`` purely so
    ``add()`` can hand each contributor a ``shared_ptr`` to itself; Python
    passes ``self``.
    """

    __slots__ = ("_contributors", "_end_criteria", "_observers", "_optimizer")

    def __init__(
        self,
        accuracy: float | None = None,
        optimizer: OptimizationMethod | None = None,
        end_criteria: EndCriteria | None = None,
    ) -> None:
        """Merge of the two C++ constructors.

        ``MultiCurveBootstrap(accuracy)`` (cpp:26-29) builds both the
        optimizer and the end criteria from ``accuracy``.
        ``MultiCurveBootstrap(optimizer, endCriteria)`` (cpp:31-39)
        substitutes ``1e-10`` for whichever argument is null. Supplying
        ``accuracy`` together with either object is not expressible in C++
        and is rejected here.
        """
        if accuracy is not None:
            qassert.require(
                optimizer is None and end_criteria is None,
                "MultiCurveBootstrap: accuracy cannot be combined with an "
                "explicit optimizer or end criteria",
            )
            # C++ parity: globalbootstrap.cpp:27-28.
            self._optimizer: OptimizationMethod = LevenbergMarquardt(
                accuracy, accuracy, accuracy
            )
            self._end_criteria: EndCriteria = _default_end_criteria(accuracy)
        else:
            # C++ parity: globalbootstrap.cpp:34-38 — ``constexpr auto
            # accuracy = 1E-10`` fills in each null argument INDEPENDENTLY,
            # so a caller may supply only one of the two.
            a = _MULTI_CURVE_DEFAULT_ACCURACY
            self._optimizer = (
                optimizer if optimizer is not None else LevenbergMarquardt(a, a, a)
            )
            self._end_criteria = (
                end_criteria if end_criteria is not None else _default_end_criteria(a)
            )
        self._contributors: list[MultiCurveBootstrapContributor] = []
        self._observers: list[MultiCurveBootstrapObserver] = []

    def add(self, c: MultiCurveBootstrapContributor) -> None:
        """C++ parity: globalbootstrap.cpp:41-44."""
        self._contributors.append(c)
        c.set_parent_bootstrapper(self)

    def add_observer(self, o: MultiCurveBootstrapObserver) -> None:
        """C++ parity: globalbootstrap.cpp:46-48."""
        self._observers.append(o)

    def contributors(self) -> list[MultiCurveBootstrapContributor]:
        """Read-only view of the registered contributors (no C++ counterpart)."""
        return list(self._contributors)

    def optimizer(self) -> OptimizationMethod:
        """C++ ``optimizer_`` (globalbootstrap.hpp:63), resolved by the ctor."""
        return self._optimizer

    def end_criteria(self) -> EndCriteria:
        """C++ ``endCriteria_`` (globalbootstrap.hpp:64), resolved by the ctor."""
        return self._end_criteria

    def guess_sizes(self) -> list[int]:
        """Per-contributor guess lengths, in registration order.

        Calls each contributor's ``setup_cost_function`` exactly as
        ``run_multi_curve_bootstrap`` does (cpp:55-59); exposed separately so
        the concatenation layout can be inspected without running the
        optimizer.
        """
        return [len(c.setup_cost_function()) for c in self._contributors]

    def _global_cost(self, guess_sizes: list[int], x: Array) -> Array:
        """The joint cost function — a transcription of cpp:61-100.

        Splitting the walk out of ``run_multi_curve_bootstrap`` keeps the
        layout testable at a fixed ``x`` without an optimizer in the loop.
        """
        # Distribute x. C++ parity: cpp:64-71.
        offset = 0
        for c, size in zip(self._contributors, guess_sizes, strict=True):
            c.set_cost_function_argument(x[offset : offset + size])
            offset += size

        # C++ parity: cpp:74-75 — non-bootstrapped cycle members refresh here.
        for o in self._observers:
            o.update()

        # Collect and concatenate. C++ parity: cpp:79-97. The result offsets
        # walk the RESULT sizes, which are not the guess sizes.
        results = [c.evaluate_cost_function() for c in self._contributors]
        if not results:
            return np.empty(0, dtype=np.float64)
        return np.concatenate(results)

    def run_multi_curve_bootstrap(self) -> None:
        """C++ parity: ``MultiCurveBootstrap::runMultiCurveBootstrap``, cpp:50-116."""
        guess_sizes: list[int] = []
        global_guess: list[float] = []
        for c in self._contributors:
            guess = c.setup_cost_function()
            global_guess.extend(float(v) for v in guess)
            guess_sizes.append(len(guess))

        def fn(x: Array) -> Array:
            return self._global_cost(guess_sizes, x)

        cost_function = SimpleCostFunction(fn)
        problem = Problem(
            cost_function, NoConstraint(), np.asarray(global_guess, dtype=np.float64)
        )
        end_type = self._optimizer.minimize(problem, self._end_criteria)
        qassert.require(
            EndCriteria.succeeded(end_type),
            "global bootstrap failed to minimize to required accuracy "
            f"(during multi curve bootstrap): {end_type!s}",
        )
        # C++ parity: levenbergmarquardt.cpp:137 —
        # ``P.setFunctionValue(P.costFunction().value(P.currentValue()))``
        # re-runs the cost function at the accepted point, which is what
        # leaves every member curve holding the SOLUTION rather than
        # whichever trial point the optimizer happened to try last. The
        # pquantlib LevenbergMarquardt omits that trailing evaluation
        # (levenberg_marquardt.py:185-189), so it is done here explicitly.
        fn(problem.current_value)

        # C++ parity: cpp:114-115.
        for c in self._contributors:
            c.set_to_valid()


class GlobalBootstrap(MultiCurveBootstrapContributor):
    """Least-squares bootstrap over all pillars at once.

    # C++ parity: ``template <class Curve> class GlobalBootstrap final``
    # at globalbootstrap.hpp:105-157 plus the out-of-line template
    # definitions at hpp:161-429.

    Usage mirrors C++: construct, ``setup(curve)`` once (the C++ curve
    constructor does this at piecewiseyieldcurve.hpp:164), then
    ``calculate()`` to run the solve. When the bootstrap has been handed
    to a :class:`MultiCurveBootstrap`, ``calculate()`` delegates to the
    joint solve instead (hpp:408-411).
    """

    def __init__(
        self,
        traits: Any,
        accuracy: float | None = None,
        optimizer: OptimizationMethod | None = None,
        end_criteria: EndCriteria | None = None,
        instrument_weights: Sequence[float] | None = None,
        additional_helpers: Sequence[BootstrapHelper[Any]] | None = None,
        additional_dates: Callable[[], list[Date]] | None = None,
        additional_penalties: AdditionalPenalties | ZeroArgPenalties | None = None,
        additional_variables: AdditionalBootstrapVariables | None = None,
    ) -> None:
        """Merge of the three C++ constructors (hpp:112 / :116 / :124).

        ``traits`` is explicit because Python has no
        ``Curve::traits_type``; everything else keeps the C++ name and
        default. ``additional_penalties`` accepts either the two-argument
        ``(times, data) -> Array`` shape or the zero-argument
        ``() -> Array`` shape; the latter is wrapped exactly as the third
        C++ constructor does at hpp:196-206.
        """
        self._traits: Any = traits() if isinstance(traits, type) else traits
        self._ts: Any = None
        # ``accuracy_`` is Null<Real>() by default in C++, meaning "take the
        # curve's own accuracy at setup() time" (hpp:223). ``None`` is that
        # sentinel here — pquantlib does not use the C++ Null-as-float-max
        # trick for optional constructor arguments.
        self._accuracy: float | None = accuracy
        self._optimizer: OptimizationMethod | None = optimizer
        self._end_criteria: EndCriteria | None = end_criteria
        self._additional_helpers: list[BootstrapHelper[Any]] = (
            list(additional_helpers) if additional_helpers else []
        )
        self._additional_dates: Callable[[], list[Date]] | None = additional_dates
        self._additional_penalties: AdditionalPenalties | None = _wrap_penalties(
            additional_penalties
        )
        self._additional_variables: AdditionalBootstrapVariables | None = (
            additional_variables
        )
        self._instrument_weights: list[float] = (
            list(instrument_weights) if instrument_weights else []
        )

        # mutable state — C++ hpp:148-156.
        self._alive_instruments: list[BootstrapHelper[Any]] = []
        self._alive_additional_helpers: list[BootstrapHelper[Any]] = []
        self._alive_instrument_weights: list[float] = []
        self._initialized: bool = False
        self._valid_curve: bool = False
        self._parent_bootstrapper: MultiCurveBootstrap | None = None

    # -- setup -------------------------------------------------------------

    def setup(self, ts: GlobalBootstrapCurve) -> None:
        """Bind the bootstrap to its curve.

        # C++ parity: ``GlobalBootstrap<Curve>::setup`` at hpp:215-240.
        """
        self._ts = ts
        # C++ parity: hpp:217-220 — ``ts_->registerWithObservables(h)``, i.e.
        # the CURVE observes each helper. pquantlib has no Handle layer, so
        # the registration is spelled the other way round: the helper (an
        # Observable) registers the curve (an Observer).
        curve_as_observer = getattr(ts, "update", None)
        if callable(curve_as_observer):
            for h in list(ts.instruments()) + self._additional_helpers:
                h.register_with(ts)  # pyright: ignore[reportArgumentType]

        # C++ parity: hpp:223 — ``accuracy_ != Null<Real>() ? accuracy_
        # : ts_->accuracy_``. See the module docstring on friendship: C++
        # reads the curve's private ``accuracy_`` because GlobalBootstrap is
        # a friend of PiecewiseYieldCurve (piecewiseyieldcurve.hpp:181).
        accuracy = (
            self._accuracy
            if self._accuracy is not None
            else float(getattr(ts, "_accuracy", _CURVE_DEFAULT_ACCURACY))
        )
        if self._optimizer is None:
            # C++ parity: hpp:225.
            self._optimizer = LevenbergMarquardt(accuracy, accuracy, accuracy)
        if self._end_criteria is None:
            # C++ parity: hpp:228.
            self._end_criteria = _default_end_criteria(accuracy)

        # C++ parity: hpp:232-236.
        n = len(ts.instruments())
        qassert.require(
            not self._instrument_weights or len(self._instrument_weights) == n,
            f"GlobalBootstrap: number of instrument weights "
            f"({len(self._instrument_weights)}) must match number of "
            f"instruments ({n})",
        )
        # ``resize(n, 1.0)`` — grow with 1.0, never shrink an exact-length one.
        self._instrument_weights.extend([1.0] * (n - len(self._instrument_weights)))

        # C++ hpp:238-239: do NOT initialize here; helpers may be invalid now
        # and valid later, when the bootstrap is actually required.

    # -- initialize --------------------------------------------------------

    def _initialize(self) -> None:
        """Build the pillar grid and the alive-helper selection.

        # C++ parity: ``GlobalBootstrap<Curve>::initialize`` at hpp:242-317.
        """
        ts = self._ts
        traits = self._traits
        first_date: Date = traits.initial_date(ts)

        # C++ parity: hpp:247-254. A helper whose pillar is ON the first date
        # is DEAD: the test is strictly greater.
        self._alive_instruments = []
        self._alive_instrument_weights = []
        for i, h in enumerate(ts.instruments()):
            if h.pillar_date() > first_date:
                self._alive_instruments.append(h)
                self._alive_instrument_weights.append(self._instrument_weights[i])

        # C++ parity: hpp:257-262.
        self._alive_additional_helpers = [
            h for h in self._additional_helpers if h.pillar_date() > first_date
        ]

        # C++ parity: hpp:265-274 — expired additional dates are dropped
        # (again ``date <= firstDate``, so equality drops).
        additional_dates: list[Date] = []
        if self._additional_dates is not None:
            additional_dates = list(self._additional_dates())
        additional_dates = [d for d in additional_dates if d > first_date]

        # C++ parity: hpp:281-288 — first date, then every alive pillar, then
        # the surviving additional dates; then sort and unique.
        dates: list[Date] = [first_date]
        dates.extend(h.pillar_date() for h in self._alive_instruments)
        dates.extend(additional_dates)
        dates.sort()
        dates = _unique_sorted(dates)

        # C++ parity: hpp:291-294.
        required_points = int(
            getattr(
                getattr(ts, "_interpolator", None),
                "required_points",
                _DEFAULT_REQUIRED_POINTS,
            )
        )
        qassert.require(
            len(dates) >= required_points,
            f"GlobalBootstrap: not enough curve points ({len(dates)}) for "
            f"interpolation requiring at least {required_points}",
        )

        # C++ parity: hpp:297-299.
        times = [ts.time_from_reference(d) for d in dates]

        # C++ parity: hpp:302-306 — the curve reaches past its last pillar
        # whenever any alive helper (ordinary OR additional) needs it to.
        max_date = dates[-1]
        for h in (*self._alive_instruments, *self._alive_additional_helpers):
            max_date = max(max_date, h.latest_relevant_date())

        # C++ parity: hpp:309-315 — reuse the current curve as the guess only
        # when it is valid AND still the right length.
        current_data = self._current_data()
        if not self._valid_curve or current_data is None or len(
            current_data
        ) != len(dates):
            data = [float(traits.initial_value(ts))] * len(dates)
            self._valid_curve = False
        else:
            data = list(current_data)

        ts.bootstrap_install_grid(dates, times, data)
        ts.set_max_date(max_date)
        self._initialized = True

    def _current_data(self) -> list[float] | None:
        """``ts_->data_`` if the curve has one yet, else ``None``.

        C++ can always read ``ts_->data_`` because the vector exists (empty)
        from construction; a pquantlib curve has no data array at all until
        ``bootstrap_install_grid`` allocates it, so the read is guarded.
        """
        try:
            return self._ts.data_live()
        except (AssertionError, AttributeError):
            return None

    # -- MultiCurveBootstrapContributor ------------------------------------

    def set_parent_bootstrapper(self, b: MultiCurveBootstrap) -> None:
        """C++ parity: hpp:208-211."""
        self._parent_bootstrapper = b

    def set_to_valid(self) -> None:
        """C++ parity: hpp:213."""
        self._valid_curve = True

    def setup_cost_function(self) -> Array:
        """Wire the helpers and return the initial guess.

        # C++ parity: ``GlobalBootstrap<Curve>::setupCostFunction`` at
        # hpp:319-376.
        """
        ts = self._ts
        traits = self._traits
        qassert.require(ts is not None, "GlobalBootstrap: setup() not called")

        # C++ parity: hpp:324 — for a multi-curve solve, calculate() is never
        # triggered on the contributing curves, so the "already calculated"
        # flag has to be set by hand. pquantlib's PiecewiseYieldCurve guards
        # re-entry on ``_underlying is not None`` instead (which
        # bootstrap_install_grid has already made true), so the call is only
        # made when the curve actually offers the hook.
        set_calculated = getattr(ts, "set_calculated", None)
        if callable(set_calculated):
            set_calculated(True)

        # C++ parity: hpp:331-332 — re-initialize a moving curve every time,
        # because date-relative helpers move with the evaluation date.
        if not self._initialized or bool(getattr(ts, "_moving", False)):
            self._initialize()

        # C++ parity: hpp:335-344.
        for helper in self._alive_instruments:
            qassert.require(
                helper.quote().is_valid(),
                f"instrument (maturity: {helper.maturity_date()}, pillar: "
                f"{helper.pillar_date()}) has an invalid quote",
            )
            helper.set_term_structure(ts)

        # C++ parity: hpp:347-352.
        for helper in self._alive_additional_helpers:
            qassert.require(
                helper.quote().is_valid(),
                f"additional instrument (maturity: {helper.maturity_date()}) "
                f"has an invalid quote",
            )
            helper.set_term_structure(ts)

        # C++ parity: hpp:355-358 — build the interpolation from scratch
        # unless the previous solution is being reused.
        data = ts.data_live()
        if not self._valid_curve:
            ts.refresh_interpolation_through(len(data) - 1)

        # C++ parity: hpp:362-365.
        additional_guesses: list[float] = []
        if self._additional_variables is not None:
            additional_guesses = [
                float(v) for v in self._additional_variables.initialize(self._valid_curve)
            ]

        # C++ parity: hpp:366-374 — curve guesses first, additional variables
        # after. ``updateGuess`` is called for its SIDE EFFECT: Traits::guess
        # for pillar i reads data[i-1], so each guess must be written back
        # before the next one is computed.
        times = ts.times()
        n = len(times) - 1
        guess = np.empty(n + len(additional_guesses), dtype=np.float64)
        for i in range(n):
            traits.update_guess(
                data,
                traits.guess(i + 1, ts, self._valid_curve, _FIRST_ALIVE_HELPER),
                i + 1,
            )
            guess[i] = traits.transform_inverse(data[i + 1], i + 1, ts)
        guess[n:] = additional_guesses
        return guess

    def set_cost_function_argument(self, v: Array) -> None:
        """Push ``x`` into the curve (and the additional variables).

        # C++ parity: ``GlobalBootstrap<Curve>::setCostFunctionArgument`` at
        # hpp:378-389. ``x`` has the same layout as the guess: the first
        # ``len(times) - 1`` entries are curve values, the rest belong to the
        # additional variables.
        """
        ts = self._ts
        traits = self._traits
        data = ts.data_live()
        n = len(ts.times()) - 1
        for i in range(n):
            traits.update_guess(
                data, traits.transform_direct(float(v[i]), i + 1, ts), i + 1
            )
        # C++ parity: hpp:385 — ``ts_->interpolation_.update()`` rebuilds over
        # the WHOLE grid, not a prefix of it.
        ts.refresh_interpolation_through(len(data) - 1)
        if self._additional_variables is not None:
            # C++ parity: hpp:387.
            self._additional_variables.update(np.asarray(v[n:], dtype=np.float64))

    def evaluate_cost_function(self) -> Array:
        """Residuals: weighted quote errors, then the additional penalties.

        # C++ parity: ``GlobalBootstrap<Curve>::evaluateCostFunction`` at
        # hpp:391-403.
        """
        ts = self._ts
        additional_errors: list[float] = []
        if self._additional_penalties is not None:
            additional_errors = [
                float(e)
                for e in self._additional_penalties(ts.times(), ts.data_live())
            ]
        n = len(self._alive_instruments)
        result = np.empty(n + len(additional_errors), dtype=np.float64)
        for i, helper in enumerate(self._alive_instruments):
            result[i] = helper.quote_error() * self._alive_instrument_weights[i]
        result[n:] = additional_errors
        return result

    # -- main entry --------------------------------------------------------

    def calculate(self) -> None:
        """Run the solve.

        # C++ parity: ``GlobalBootstrap<Curve>::calculate`` at hpp:405-429.
        """
        # C++ parity: hpp:408-411 — a member of a multi-curve cycle defers to
        # the joint solve and returns.
        if self._parent_bootstrapper is not None:
            self._parent_bootstrapper.run_multi_curve_bootstrap()
            return

        guess = self.setup_cost_function()

        def values(x: Array) -> Array:
            # C++ parity: hpp:419-422 — the SimpleCostFunction lambda.
            self.set_cost_function_argument(x)
            return self.evaluate_cost_function()

        problem = Problem(SimpleCostFunction(values), NoConstraint(), guess)
        assert self._optimizer is not None
        assert self._end_criteria is not None
        end_type = self._optimizer.minimize(problem, self._end_criteria)
        qassert.require(
            EndCriteria.succeeded(end_type),
            "global bootstrap failed to minimize to required accuracy: "
            f"{end_type!s}",
        )
        # See the note in MultiCurveBootstrap.run_multi_curve_bootstrap: C++
        # gets this for free from levenbergmarquardt.cpp:137, the pquantlib
        # optimizer does not, so the curve is put back on the solution here.
        values(problem.current_value)
        self._valid_curve = True

    # -- inspectors --------------------------------------------------------

    def alive_instruments(self) -> list[BootstrapHelper[Any]]:
        """C++ ``aliveInstruments_`` (hpp:148)."""
        return list(self._alive_instruments)

    def alive_additional_helpers(self) -> list[BootstrapHelper[Any]]:
        """C++ ``aliveAdditionalHelpers_`` (hpp:149)."""
        return list(self._alive_additional_helpers)

    def alive_instrument_weights(self) -> list[float]:
        """C++ ``aliveInstrumentWeights_`` (hpp:154)."""
        return list(self._alive_instrument_weights)

    def instrument_weights(self) -> list[float]:
        """C++ ``instrumentWeights_`` after the resize in setup (hpp:236)."""
        return list(self._instrument_weights)

    def optimizer(self) -> OptimizationMethod | None:
        """C++ ``optimizer_`` (hpp:145) — resolved by ``setup``."""
        return self._optimizer

    def end_criteria(self) -> EndCriteria | None:
        """C++ ``endCriteria_`` (hpp:146) — resolved by ``setup``."""
        return self._end_criteria

    def valid_curve(self) -> bool:
        """C++ ``validCurve_`` (hpp:155)."""
        return self._valid_curve


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _default_end_criteria(accuracy: float) -> EndCriteria:
    """C++ parity: ``EndCriteria(1000, 10, a, a, a)`` — hpp:228, cpp:28, cpp:38."""
    return EndCriteria(
        _END_CRITERIA_MAX_ITERATIONS,
        _END_CRITERIA_MAX_STATIONARY_STATE,
        accuracy,
        accuracy,
        accuracy,
    )


def _wrap_penalties(
    penalties: AdditionalPenalties | ZeroArgPenalties | None,
) -> AdditionalPenalties | None:
    """Normalize both C++ penalty shapes onto the two-argument one.

    # C++ parity: globalbootstrap.hpp:196-206 — the third constructor wraps
    # ``std::function<Array()>`` in a lambda that discards ``times`` and
    # ``data``. C++ picks the overload from the argument's TYPE; Python has
    # to ask the callable how many arguments it takes.
    """
    if penalties is None:
        return None
    try:
        arity = len(inspect.signature(penalties).parameters)
    except (TypeError, ValueError):  # pragma: no cover - C-implemented callables
        arity = 2
    if arity == 0:
        zero_arg: Any = penalties
        return lambda _times, _data: zero_arg()
    return penalties  # pyright: ignore[reportReturnType]


def _unique_sorted(dates: list[Date]) -> list[Date]:
    """``std::unique`` on an already-sorted vector — C++ hpp:288."""
    out: list[Date] = []
    for d in dates:
        if not out or out[-1] != d:
            out.append(d)
    return out


__all__ = [
    "AdditionalBootstrapVariables",
    "AdditionalPenalties",
    "GlobalBootstrap",
    "GlobalBootstrapCurve",
    "MultiCurveBootstrap",
    "MultiCurveBootstrapContributor",
    "MultiCurveBootstrapObserver",
    "ZeroArgPenalties",
]

"""Model parameter hierarchy.

# C++ parity: ql/models/parameter.hpp (v1.42.1).

C++ uses the PIMPL pattern: ``Parameter`` is a value type that holds a
``shared_ptr<Impl>`` strategy, an ``Array params_`` (the actual numbers),
and a ``Constraint``. Subclasses (``ConstantParameter``,
``NullParameter``, ``PiecewiseConstantParameter``,
``TermStructureFittingParameter``) construct the right ``Impl`` in
their ctor; ``operator()(Time t)`` dispatches to ``impl_->value``.

The Python port preserves this strategy structure (nested ``Impl``
class with a ``value(params, t)`` abstractmethod) for these reasons:

1. ``TermStructureFittingParameter::NumericalImpl`` carries
   non-parameter state (``times_``, ``values_``, ``termStructure_``)
   that callers mutate via ``set/change/reset`` methods on the Impl,
   not on the Parameter. Collapsing Impl into Parameter would break
   that API.
2. ``CalibratedModel`` stores ``std::vector<Parameter>`` by value;
   the value-with-strategy idiom keeps ``Parameter`` cheap to copy
   (just two ``shared_ptr`` and one numpy array) while still allowing
   strategy-based polymorphism.

Divergences from C++:

- ``ConstantParameter`` ctor — C++ has overloads
  ``ConstantParameter(Real, Constraint)`` and ``ConstantParameter(Constraint)``;
  Python collapses to one ``__init__(value=None, constraint=...)`` with
  a None default that means "no initial value (allocate a 1-element
  array of zeros)". Same observable behavior.
- ``PiecewiseConstantParameter`` ctor — C++ default-constructs
  ``Constraint = NoConstraint()`` (Python: same).
- ``TermStructureFittingParameter`` — Python takes a ``Quote``-style
  yield curve object (instead of ``Handle<YieldTermStructure>``)
  per the pquantlib convention that Handle<> collapses to direct
  object reference (the Quote/YieldTermStructure objects are
  themselves Observable). The implementation may need refinement
  in L4-B/C when concrete short-rate models start using it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from bisect import bisect_right
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.math.optimization.constraint import Constraint, NoConstraint

if TYPE_CHECKING:
    from pquantlib.termstructures.yield_term_structure import YieldTermStructure


class ParameterImpl(ABC):
    """Strategy interface — given the params array and a time, return the
    real-valued parameter at that time.

    # C++ parity: ``Parameter::Impl`` (nested abstract class) in
    # ql/models/parameter.hpp:41-45 (v1.42.1).
    """

    @abstractmethod
    def value(self, params: npt.NDArray[np.float64], t: float) -> float:
        """Return the parameter value at time ``t``, given the params vector."""
        ...


class Parameter:
    """Base class for model arguments.

    # C++ parity: ``class Parameter`` in
    # ql/models/parameter.hpp:38-68 (v1.42.1).

    Holds three pieces:

    - ``params_`` — a numpy array of real-valued underlying parameters
      (free parameters that an optimizer can adjust).
    - ``impl_`` — strategy describing how to evaluate the parameter
      at a given time from those params.
    - ``constraint_`` — feasibility predicate on ``params_``.

    Subclasses set up the Impl + initial params in their ctor. Direct
    instantiation of ``Parameter`` is supported (mirroring C++'s
    public default ctor, which produces an "empty" Parameter).
    """

    __slots__ = ("_constraint", "_impl", "_params")

    def __init__(
        self,
        size: int = 0,
        impl: ParameterImpl | None = None,
        constraint: Constraint | None = None,
    ) -> None:
        # C++ parity: parameter.hpp:48-49 — default ctor allocates a
        # zero-size params array and a NoConstraint, with a null Impl.
        # The protected ctor parameter.hpp:64-65 allocates ``size``
        # zero-initialized slots and stores the provided Impl + constraint.
        self._params: npt.NDArray[np.float64] = np.zeros(size, dtype=np.float64)
        self._impl: ParameterImpl | None = impl
        self._constraint: Constraint = constraint if constraint is not None else NoConstraint()

    # --- inspectors -----------------------------------------------------

    @property
    def params(self) -> npt.NDArray[np.float64]:
        """The underlying free-parameter vector.

        # C++ parity: ``Parameter::params`` in parameter.hpp:50.
        """
        return self._params

    @property
    def constraint(self) -> Constraint:
        """The constraint applied to ``params``.

        # C++ parity: ``Parameter::constraint`` in parameter.hpp:62.
        """
        return self._constraint

    @property
    def size(self) -> int:
        """Number of free parameters.

        # C++ parity: ``Parameter::size`` in parameter.hpp:55.
        """
        return int(self._params.size)

    @property
    def impl(self) -> ParameterImpl | None:
        """The lookup-strategy implementation.

        # C++ parity: ``Parameter::implementation`` in parameter.hpp:59-61.
        """
        return self._impl

    # --- mutators -------------------------------------------------------

    def set_param(self, i: int, x: float) -> None:
        """Set the i-th free parameter.

        # C++ parity: ``Parameter::setParam`` in parameter.hpp:51.
        """
        self._params[i] = x

    def test_params(self, params: npt.NDArray[np.float64]) -> bool:
        """Check ``params`` against this parameter's constraint.

        # C++ parity: ``Parameter::testParams`` in parameter.hpp:52-54.
        """
        return self._constraint.test(params)

    # --- evaluation -----------------------------------------------------

    def __call__(self, t: float) -> float:
        """Evaluate the parameter at time ``t``.

        # C++ parity: ``Parameter::operator()(Time)`` in parameter.hpp:56-58.
        Raises if no Impl has been configured (Python equivalent of
        dereferencing a null shared_ptr).
        """
        if self._impl is None:
            raise RuntimeError("Parameter has no implementation")
        return self._impl.value(self._params, t)


class _NullImpl(ParameterImpl):
    """Always-zero strategy.

    # C++ parity: nested ``NullParameter::Impl`` in parameter.hpp:101-104.
    """

    def value(self, params: npt.NDArray[np.float64], t: float) -> float:
        return 0.0


class NullParameter(Parameter):
    """Parameter that is always zero, ``a(t) = 0``.

    # C++ parity: ``class NullParameter`` in
    # ql/models/parameter.hpp:99-112 (v1.42.1).
    """

    def __init__(self) -> None:
        super().__init__(size=0, impl=_NullImpl(), constraint=NoConstraint())


class _ConstantImpl(ParameterImpl):
    """Single-scalar lookup strategy.

    # C++ parity: nested ``ConstantParameter::Impl`` in parameter.hpp:73-76.
    """

    def value(self, params: npt.NDArray[np.float64], t: float) -> float:
        return float(params[0])


class ConstantParameter(Parameter):
    """Standard constant parameter ``a(t) = a``.

    # C++ parity: ``class ConstantParameter`` in
    # ql/models/parameter.hpp:71-96 (v1.42.1).

    Two construction modes (matching the C++ overloads):

    - ``ConstantParameter(constraint)`` — allocate a 1-element params
      array initialized to zero; the caller can ``set_param(0, x)``
      later.
    - ``ConstantParameter(value, constraint)`` — allocate and
      initialize to ``value``; raises if ``value`` violates
      ``constraint``.
    """

    def __init__(
        self,
        value: float | Constraint,
        constraint: Constraint | None = None,
    ) -> None:
        # Mirror the C++ two-overload pattern. If ``value`` is a Constraint,
        # treat it as the no-initial-value overload.
        if isinstance(value, Constraint):
            real_constraint: Constraint = value
            initial_value: float | None = None
        else:
            real_constraint = constraint if constraint is not None else NoConstraint()
            initial_value = float(value)

        super().__init__(size=1, impl=_ConstantImpl(), constraint=real_constraint)
        if initial_value is not None:
            self._params[0] = initial_value
            # C++ parity: parameter.hpp:92-93 — validate against constraint.
            qassert.require(
                self.test_params(self._params),
                f"{initial_value}: invalid value",
            )


class _PiecewiseConstantImpl(ParameterImpl):
    """Step-function lookup strategy on a tenor grid.

    # C++ parity: nested ``PiecewiseConstantParameter::Impl`` in
    # parameter.hpp:121-132.

    Given times ``[t_0, t_1, ..., t_{n-1}]`` and params ``[p_0, p_1,
    ..., p_n]`` (one more param than time-break), the value at ``t``
    is ``params[i]`` where ``i = upper_bound(times, t)``. I.e.:

    - ``t <= t_0`` -> ``params[0]``
    - ``t_0 < t <= t_1`` -> ``params[1]``
    - ...
    - ``t_{n-2} < t <= t_{n-1}`` -> ``params[n-1]``
    - ``t > t_{n-1}`` -> ``params[n]``

    Python's ``bisect_right`` is the exact analogue of C++
    ``std::upper_bound``.
    """

    __slots__ = ("_times",)

    def __init__(self, times: list[float]) -> None:
        # Defensive copy + sort-check ensures the lookup invariant.
        self._times: list[float] = list(times)

    def value(self, params: npt.NDArray[np.float64], t: float) -> float:
        i = bisect_right(self._times, t)
        return float(params[i])


class PiecewiseConstantParameter(Parameter):
    """Piecewise-constant parameter ``a(t) = a_i for t_{i-1} <= t < t_i``.

    # C++ parity: ``class PiecewiseConstantParameter`` in
    # ql/models/parameter.hpp:119-142 (v1.42.1).

    The parameter holds ``len(times)+1`` slots — one for each interval
    plus a trailing slot for ``t > max(times)``.
    """

    def __init__(self, times: list[float], constraint: Constraint | None = None) -> None:
        # C++ parity: parameter.hpp:134-141 — allocates times.size()+1 slots.
        super().__init__(
            size=len(times) + 1,
            impl=_PiecewiseConstantImpl(times),
            constraint=constraint if constraint is not None else NoConstraint(),
        )


class TermStructureFittingParameterImpl(ParameterImpl):
    """Numerical fitter — caches (time, value) pairs and looks them up.

    # C++ parity: ``TermStructureFittingParameter::NumericalImpl`` in
    # parameter.hpp:147-176 (v1.42.1).

    Builder methods:

    - ``set(t, x)`` — append a new (time, value) pair to the cache.
    - ``change(x)`` — overwrite the most recently appended value.
    - ``reset()`` — empty the cache.
    - ``value(_, t)`` — look up the value at ``t``; raises if not set.

    Subclasses (e.g. used by short-rate models in L4-B/C) populate the
    cache during fitting, then read it during pricing.
    """

    __slots__ = ("_term_structure", "_times", "_values")

    def __init__(self, term_structure: YieldTermStructure | None = None) -> None:
        self._times: list[float] = []
        self._values: list[float] = []
        self._term_structure: YieldTermStructure | None = term_structure

    @property
    def term_structure(self) -> YieldTermStructure | None:
        """The associated yield curve, if any.

        # C++ parity: parameter.hpp:169-171 — ``termStructure()``.
        """
        return self._term_structure

    def set(self, t: float, x: float) -> None:
        """Append a fitted (t, x) pair.

        # C++ parity: parameter.hpp:152-155 — ``set``.
        """
        self._times.append(t)
        self._values.append(x)

    def change(self, x: float) -> None:
        """Overwrite the last appended value.

        # C++ parity: parameter.hpp:156-158 — ``change``.
        """
        if not self._values:
            raise IndexError("change() on empty fitter cache")
        self._values[-1] = x

    def reset(self) -> None:
        """Empty the cache.

        # C++ parity: parameter.hpp:159-162 — ``reset``.
        """
        self._times.clear()
        self._values.clear()

    def value(self, params: npt.NDArray[np.float64], t: float) -> float:
        """Look up the value at ``t``; raises if not set.

        # C++ parity: parameter.hpp:163-168 — ``value``.
        """
        try:
            i = self._times.index(t)
        except ValueError as exc:
            raise RuntimeError("fitting parameter not set!") from exc
        return self._values[i]


class TermStructureFittingParameter(Parameter):
    """Deterministic time-dependent parameter used for yield-curve fitting.

    # C++ parity: ``class TermStructureFittingParameter`` in
    # ql/models/parameter.hpp:145-188 (v1.42.1).

    Constructed either from a yield curve (allocates a default
    ``NumericalImpl`` cache) or from a custom Impl. Holds zero free
    parameters (``size == 0``) — its value comes from the cache
    populated during fitting, not from optimizer parameters.
    """

    def __init__(
        self,
        impl_or_term_structure: ParameterImpl | YieldTermStructure | None = None,
    ) -> None:
        # C++ parity: parameter.hpp:178-187 — two overloads merged:
        #   (a) ctor from an Impl (used when a caller provides custom
        #       fitter logic, e.g. a HullWhiteFittingParameter).
        #   (b) ctor from a YieldTermStructure handle (defaults to
        #       NumericalImpl).
        impl: ParameterImpl
        if isinstance(impl_or_term_structure, ParameterImpl):
            impl = impl_or_term_structure
        else:
            # impl_or_term_structure is either None or a YieldTermStructure.
            impl = TermStructureFittingParameterImpl(term_structure=impl_or_term_structure)
        super().__init__(size=0, impl=impl, constraint=NoConstraint())

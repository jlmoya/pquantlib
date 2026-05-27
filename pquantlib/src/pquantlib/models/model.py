"""Model abstract bases — AffineModel / TermStructureConsistentModel / CalibratedModel.

# C++ parity: ql/models/model.{hpp,cpp} (v1.42.1).

C++ defines four bases:

- ``AffineModel`` (analytic discount + discountBond + discountBondOption)
- ``TermStructureConsistentModel`` (holds a YieldTermStructure handle)
- ``CalibratedModel`` (holds Parameter list, runs the optimizer on a
  CalibrationFunction over CalibrationHelper instruments)
- ``ShortRateModel`` (CalibratedModel + tree(TimeGrid))

The pquantlib port mirrors the same hierarchy. ``AffineModel`` was
originally sketched only as a ``Protocol`` in ``protocols.py`` (L4-A
stage 5); L4-B introduces a proper abstract base because OneFactorAffineModel
needs to MRO-multi-inherit from both ``OneFactorModel`` (CalibratedModel
subclass) and ``AffineModel``, which Python only supports cleanly with
a concrete (abstract) class.

``ShortRateModel`` is added here in L4-B — it's the CalibratedModel
subclass with the abstract ``tree(TimeGrid)`` method. The tree() impl
itself is deferred per the cluster spec (TrinomialTree + lattice
plumbing carry-overs).

L4-A scope (this module): the three abstract bases required to define
``calibrate()`` semantics. Concrete models land in L4-B/C/D/E.

Calibrate() orchestration:

1. Build a ``CalibrationFunction`` (a ``CostFunction`` whose
   ``values(x)`` is the residual vector ``[helper.calibration_error()
   * sqrt(weight) for helper in instruments]``).
2. Apply the constraint (model's own PrivateConstraint composed with
   any user-supplied extra constraint).
3. Build the ``Problem(cost_function, constraint, current_params)``.
4. Project away any fixed parameters via the inline ``Projection``
   helper (so the optimizer only sees free parameters).
5. Call ``method.minimize(problem, end_criteria)``.
6. Reconstruct the full param vector via ``Projection.include`` and
   set it on the model.

Divergences from C++:

- ``ProjectedConstraint`` is not separately ported in L4-A — its only
  consumer was ``CalibratedModel::calibrate``, so its logic is inlined
  here. The L4-B/C/D/E clusters won't need it independently.
- ``CompositeConstraint`` is also inlined as ``_AndConstraint`` —
  this matches the C++ ``CompositeConstraint`` semantics exactly (and
  was carried over from L1-D).
- ``null_deleter`` (used in C++ ``CalibrationFunction`` to break a
  shared_ptr cycle between model and cost function) is unneeded in
  Python — we just hold a regular reference, since Python's GC
  handles cycles.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.math.optimization.constraint import Constraint
from pquantlib.math.optimization.cost_function import CostFunction
from pquantlib.math.optimization.end_criteria import Type
from pquantlib.math.optimization.problem import Problem
from pquantlib.models.parameter import Parameter
from pquantlib.patterns.observer import Observable

if TYPE_CHECKING:
    from pquantlib.math.optimization.end_criteria import EndCriteria
    from pquantlib.math.optimization.optimization_method import OptimizationMethod
    from pquantlib.payoffs import OptionType
    from pquantlib.termstructures.yield_term_structure import YieldTermStructure


@runtime_checkable
class _CalibrationHelperLike(Protocol):
    """Structural type for the ``calibrate()`` orchestration.

    The Stage 4 ``CalibrationHelper`` ABC implements this Protocol;
    Stage 5 ``CalibrationHelperProtocol`` widens it. Keeping a
    minimal Protocol here keeps ``model.py`` self-contained without
    forcing a circular import on ``calibration_helper.py``.
    """

    def calibration_error(self) -> float: ...


# ---------------------------------------------------------------------
# Internal helpers (Projection + composite constraints)
# ---------------------------------------------------------------------


class _Projection:
    """Map between full-parameter and free-parameter vectors.

    # C++ parity: ``class Projection`` in
    # ql/math/optimization/projection.{hpp,cpp} (v1.42.1).

    Given a full parameter vector ``p`` and a list of "fix" flags
    indicating which entries are held constant, ``project(p)``
    returns the subset of free entries; ``include(q)`` reconstructs
    the full vector by re-injecting the fixed entries.

    Inlined here rather than in ``math/optimization`` because L4-A
    is its only consumer. If a later cluster needs it independently,
    promote to ``pquantlib.math.optimization.projection``.
    """

    __slots__ = ("_fix_parameters", "_fixed_parameters", "_num_free_parameters")

    def __init__(
        self,
        parameter_values: npt.NDArray[np.float64],
        fix_parameters: list[bool] | None = None,
    ) -> None:
        n = int(parameter_values.size)
        self._fixed_parameters: npt.NDArray[np.float64] = parameter_values.astype(np.float64, copy=True)
        self._fix_parameters: list[bool] = (
            list(fix_parameters) if fix_parameters is not None and len(fix_parameters) > 0 else [False] * n
        )
        qassert.require(
            len(self._fix_parameters) == n,
            f"fix_parameters.size ({len(self._fix_parameters)}) != parameters.size ({n})",
        )
        self._num_free_parameters: int = sum(1 for f in self._fix_parameters if not f)
        qassert.require(self._num_free_parameters > 0, "numberOfFreeParameters==0")

    def project(self, parameters: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return the free entries of ``parameters``.

        # C++ parity: ``Projection::project`` in projection.cpp:54-65.
        """
        qassert.require(
            int(parameters.size) == len(self._fix_parameters),
            f"parameters.size ({parameters.size}) != fix.size ({len(self._fix_parameters)})",
        )
        return np.array(
            [parameters[j] for j in range(len(self._fix_parameters)) if not self._fix_parameters[j]],
            dtype=np.float64,
        )

    def include(self, projected_parameters: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Reconstruct the full parameter vector from the free entries.

        # C++ parity: ``Projection::include`` in projection.cpp:67-78.
        """
        qassert.require(
            int(projected_parameters.size) == self._num_free_parameters,
            f"projected.size ({projected_parameters.size}) != free count ({self._num_free_parameters})",
        )
        y = self._fixed_parameters.copy()
        i = 0
        for j in range(y.size):
            if not self._fix_parameters[j]:
                y[j] = projected_parameters[i]
                i += 1
        return y


class _AndConstraint(Constraint):
    """Conjunction of two constraints.

    # C++ parity: ``class CompositeConstraint`` in
    # ql/math/optimization/constraint.hpp:140-174 (v1.42.1).

    Inlined here per the module docstring's divergence note.
    """

    __slots__ = ("_c1", "_c2")

    def __init__(self, c1: Constraint, c2: Constraint) -> None:
        self._c1: Constraint = c1
        self._c2: Constraint = c2

    def test(self, params: npt.NDArray[np.float64]) -> bool:
        return self._c1.test(params) and self._c2.test(params)

    def upper_bound(self, params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.minimum(self._c1.upper_bound(params), self._c2.upper_bound(params))

    def lower_bound(self, params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.maximum(self._c1.lower_bound(params), self._c2.lower_bound(params))


class _PrivateConstraint(Constraint):
    """Compose per-Parameter constraints into a single composite.

    # C++ parity: ``CalibratedModel::PrivateConstraint`` in
    # ql/models/model.hpp:164-229 (v1.42.1).

    Tests the constraint of each Parameter against the corresponding
    slice of the full params array.
    """

    __slots__ = ("_arguments",)

    def __init__(self, arguments: list[Parameter]) -> None:
        self._arguments: list[Parameter] = arguments

    def test(self, params: npt.NDArray[np.float64]) -> bool:
        k = 0
        for argument in self._arguments:
            size = argument.size
            if size == 0:
                continue
            slice_ = params[k : k + size]
            if not argument.test_params(slice_):
                return False
            k += size
        return True

    def upper_bound(self, params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        total_size = sum(a.size for a in self._arguments)
        result = np.zeros(total_size, dtype=np.float64)
        k = 0
        out = 0
        for argument in self._arguments:
            size = argument.size
            if size == 0:
                continue
            slice_ = params[k : k + size]
            ub = argument.constraint.upper_bound(slice_)
            result[out : out + size] = ub
            k += size
            out += size
        return result

    def lower_bound(self, params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        total_size = sum(a.size for a in self._arguments)
        result = np.zeros(total_size, dtype=np.float64)
        k = 0
        out = 0
        for argument in self._arguments:
            size = argument.size
            if size == 0:
                continue
            slice_ = params[k : k + size]
            lb = argument.constraint.lower_bound(slice_)
            result[out : out + size] = lb
            k += size
            out += size
        return result


# ---------------------------------------------------------------------
# Abstract bases
# ---------------------------------------------------------------------


class Model(Observable, ABC):
    """Abstract interest-rate model.

    # C++ parity: There is no standalone ``Model`` in C++ v1.42.1 —
    # ``CalibratedModel`` is the most general base. pquantlib introduces
    # a thin ``Model`` ABC that captures the params/setParams/update
    # surface so that callers can type against an abstract Model
    # without forcing them to depend on ``CalibratedModel``.

    Subclasses (CalibratedModel, TermStructureConsistentModel) provide
    the param storage and the calibrate() machinery.
    """

    @abstractmethod
    def params(self) -> npt.NDArray[np.float64]:
        """Return the flat free-parameter vector."""
        ...

    @abstractmethod
    def set_params(self, params: npt.NDArray[np.float64]) -> None:
        """Update all free parameters; notify observers."""
        ...

    def generate_arguments(self) -> None:
        """Hook: refresh any derived arguments after ``set_params``.

        # C++ parity: ``CalibratedModel::generateArguments`` in
        # ql/models/model.hpp:125 — default no-op.
        """

    def update(self) -> None:
        # C++ parity: ``CalibratedModel::update`` in model.hpp:90-93.
        self.generate_arguments()
        self.notify_observers()


class TermStructureConsistentModel(Observable):
    """Base for models that exactly reprice any discount bond.

    # C++ parity: ``class TermStructureConsistentModel`` in
    # ql/models/model.hpp:73-82 (v1.42.1).

    Holds a yield term structure; concrete subclasses (G2++,
    HullWhiteForward, etc.) layer the model dynamics on top.
    """

    __slots__ = ("_term_structure",)

    def __init__(self, term_structure: YieldTermStructure) -> None:
        super().__init__()
        self._term_structure: YieldTermStructure = term_structure

    @property
    def term_structure(self) -> YieldTermStructure:
        """The yield term structure the model is consistent with.

        # C++ parity: ``TermStructureConsistentModel::termStructure`` in
        # model.hpp:77-79.
        """
        return self._term_structure


class _CalibrationFunction(CostFunction):
    """Residual cost function for a calibration.

    # C++ parity: ``CalibratedModel::CalibrationFunction`` (nested) in
    # ql/models/model.cpp:35-73 (v1.42.1).

    For each instrument, the residual is
    ``helper.calibration_error() * sqrt(weight)``. The scalar
    ``value(x)`` is ``sqrt(sum(residual^2))`` (matches the C++
    override that returns ``sqrt(value)`` because instrument errors
    are signed deviations).
    """

    __slots__ = ("_instruments", "_model", "_projection", "_weights")

    def __init__(
        self,
        model: CalibratedModel,
        instruments: Sequence[_CalibrationHelperLike],
        weights: list[float],
        projection: _Projection,
    ) -> None:
        self._model: CalibratedModel = model
        self._instruments: Sequence[_CalibrationHelperLike] = instruments
        self._weights: list[float] = weights
        self._projection: _Projection = projection

    def value(self, x: npt.NDArray[np.float64]) -> float:
        # C++ parity: model.cpp:46-54.
        self._model.set_params(self._projection.include(x))
        total = 0.0
        for i, instrument in enumerate(self._instruments):
            diff = instrument.calibration_error()
            total += diff * diff * self._weights[i]
        return float(np.sqrt(total))

    def values(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # C++ parity: model.cpp:56-64.
        self._model.set_params(self._projection.include(x))
        out = np.zeros(len(self._instruments), dtype=np.float64)
        for i, instrument in enumerate(self._instruments):
            out[i] = instrument.calibration_error() * np.sqrt(self._weights[i])
        return out

    def finite_difference_epsilon(self) -> float:
        # C++ parity: model.cpp:66.
        return 1e-6


class CalibratedModel(Model):
    """Calibratable model — runs an optimizer over a CalibrationHelper list.

    # C++ parity: ``class CalibratedModel`` in
    # ql/models/model.hpp:86-137 (v1.42.1).

    Stores a list of ``Parameter`` objects (the "arguments"); a
    flattened view ``params()`` is the optimizer's free-variable
    vector. ``calibrate()`` builds a CalibrationFunction over the
    user-provided instruments and runs the user-provided
    OptimizationMethod.
    """

    __slots__ = (
        "_arguments",
        "_constraint",
        "_end_criteria",
        "_function_evaluation",
        "_problem_values",
    )

    def __init__(self, n_arguments: int) -> None:
        super().__init__()
        # C++ parity: model.cpp:32-33 — default-initialize n_arguments
        # empty Parameters; the PrivateConstraint wraps them.
        self._arguments: list[Parameter] = [Parameter() for _ in range(n_arguments)]
        self._constraint: Constraint = _PrivateConstraint(self._arguments)
        self._end_criteria: Type = Type.None_
        self._problem_values: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self._function_evaluation: int = 0

    # --- inspectors -----------------------------------------------------

    @property
    def arguments(self) -> list[Parameter]:
        """The model's Parameter list (mutable).

        # C++ parity: ``CalibratedModel::arguments_`` (protected) in
        # model.hpp:126.
        """
        return self._arguments

    @property
    def constraint(self) -> Constraint:
        """The model's composite (per-Parameter) constraint.

        # C++ parity: ``CalibratedModel::constraint`` in model.hpp:159-162.
        """
        return self._constraint

    @property
    def end_criteria(self) -> Type:
        """The optimizer outcome from the last ``calibrate()`` call.

        # C++ parity: ``CalibratedModel::endCriteria`` in model.hpp:113.
        """
        return self._end_criteria

    @property
    def problem_values(self) -> npt.NDArray[np.float64]:
        """The residual vector at the calibration optimum.

        # C++ parity: ``CalibratedModel::problemValues`` in model.hpp:116.
        """
        return self._problem_values

    @property
    def function_evaluation(self) -> int:
        """Number of cost-function evaluations during the last calibrate().

        # C++ parity: ``CalibratedModel::functionEvaluation`` in model.hpp:122.
        """
        return self._function_evaluation

    # --- params -------------------------------------------------------

    def params(self) -> npt.NDArray[np.float64]:
        """Flatten the per-Parameter arrays into a single params vector.

        # C++ parity: ``CalibratedModel::params`` in model.cpp:126-136.
        """
        total_size = sum(a.size for a in self._arguments)
        out = np.zeros(total_size, dtype=np.float64)
        k = 0
        for argument in self._arguments:
            for j in range(argument.size):
                out[k] = argument.params[j]
                k += 1
        return out

    def set_params(self, params: npt.NDArray[np.float64]) -> None:
        """Write the params vector back into the per-Parameter arrays.

        # C++ parity: ``CalibratedModel::setParams`` in model.cpp:138-149.
        """
        n = int(params.size)
        idx = 0
        for argument in self._arguments:
            for j in range(argument.size):
                qassert.require(idx < n, "parameter array too small")
                argument.set_param(j, float(params[idx]))
                idx += 1
        qassert.require(idx == n, "parameter array too big!")
        self.generate_arguments()
        self.notify_observers()

    # --- calibration ---------------------------------------------------

    def value(
        self,
        params: npt.NDArray[np.float64],
        instruments: Sequence[_CalibrationHelperLike],
    ) -> float:
        """Evaluate the RMS calibration error at ``params``.

        # C++ parity: ``CalibratedModel::value`` in model.cpp:117-124.
        """
        weights = [1.0] * len(instruments)
        projection = _Projection(params)
        cf = _CalibrationFunction(self, instruments, weights, projection)
        return cf.value(params)

    def calibrate(
        self,
        instruments: Sequence[_CalibrationHelperLike],
        method: OptimizationMethod,
        end_criteria: EndCriteria,
        constraint: Constraint | None = None,
        weights: list[float] | None = None,
        fix_parameters: list[bool] | None = None,
    ) -> None:
        """Calibrate ``arguments`` to ``instruments`` via ``method``.

        # C++ parity: ``CalibratedModel::calibrate`` in model.cpp:75-115.

        Builds a CalibrationFunction (residuals = per-helper
        calibration_error scaled by sqrt(weight)), composes the
        model's private constraint with any user-supplied extra
        constraint, projects away any fixed parameters, and runs
        ``method.minimize`` on the resulting Problem. The optimum is
        written back via ``set_params``.
        """
        qassert.require(len(instruments) > 0, "no instruments provided")

        # Compose model's private constraint with any extra constraint.
        c: Constraint = self._constraint if constraint is None else _AndConstraint(self._constraint, constraint)

        # Resolve weights (default = 1.0 per instrument).
        if weights is None or len(weights) == 0:
            w = [1.0] * len(instruments)
        else:
            qassert.require(
                len(weights) == len(instruments),
                f"mismatch between number of instruments ({len(instruments)}) and weights ({len(weights)})",
            )
            w = list(weights)

        # Project away fixed parameters.
        prms = self.params()
        if fix_parameters is not None and len(fix_parameters) > 0:
            qassert.require(
                len(fix_parameters) == int(prms.size),
                f"mismatch between number of parameters ({prms.size}) and fixed-parameter specs ({len(fix_parameters)})",
            )

        projection = _Projection(prms, fix_parameters)
        cf = _CalibrationFunction(self, instruments, w, projection)
        # The C++ code wraps ``c`` in a ProjectedConstraint here so the
        # optimizer only tests free parameters. pquantlib inlines that
        # by giving the optimizer a "projected" view of the constraint:
        # a constraint whose ``test`` round-trips through include() so
        # the per-Parameter constraint logic still operates on the full
        # param vector.
        projected_constraint = _ProjectedConstraintAdapter(c, projection)
        problem = Problem(cf, projected_constraint, projection.project(prms))
        self._end_criteria = method.minimize(problem, end_criteria)
        result = problem.current_value
        self.set_params(projection.include(result))
        self._problem_values = problem.values(result)
        self._function_evaluation = problem.function_evaluation
        self.notify_observers()


class _ProjectedConstraintAdapter(Constraint):
    """Constraint adapter that lifts a free-param vector to full-param.

    # C++ parity: ``class ProjectedConstraint`` in
    # ql/math/optimization/projectedconstraint.hpp (v1.42.1).

    The optimizer sees the free parameters and tests this adapter,
    which uses Projection.include to lift the free vector to the
    full vector and forwards the test to the underlying constraint.
    """

    __slots__ = ("_constraint", "_projection")

    def __init__(self, constraint: Constraint, projection: _Projection) -> None:
        self._constraint: Constraint = constraint
        self._projection: _Projection = projection

    def test(self, params: npt.NDArray[np.float64]) -> bool:
        return self._constraint.test(self._projection.include(params))

    def upper_bound(self, params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        full_ub = self._constraint.upper_bound(self._projection.include(params))
        # Subset the upper bound by the projection mask.
        return self._projection.project(full_ub)

    def lower_bound(self, params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        full_lb = self._constraint.lower_bound(self._projection.include(params))
        return self._projection.project(full_lb)


# ---------------------------------------------------------------------
# L4-B abstract bases — AffineModel + ShortRateModel
# ---------------------------------------------------------------------


class AffineModel(Observable, ABC):
    """Analytically tractable interest-rate model.

    # C++ parity: ``class AffineModel : public virtual Observable`` in
    # ql/models/model.hpp:45-64 (v1.42.1).

    A pure abstract base: subclasses must provide closed-form
    ``discount(t)``, ``discount_bond(now, maturity, factors)`` (vector
    factors), and ``discount_bond_option(type, strike, maturity,
    bond_maturity)``. The default 5-arg ``discount_bond_option``
    forwards to the 4-arg form (mirroring the C++ inline default).

    Multi-inherits in OneFactorAffineModel together with OneFactorModel
    (which is a CalibratedModel); MRO resolves with Observable as the
    common base.
    """

    @abstractmethod
    def discount(self, t: float) -> float:
        """Implied discount factor at time ``t``.

        # C++ parity: ``AffineModel::discount(Time)``.
        """
        ...

    @abstractmethod
    def discount_bond(
        self,
        now: float,
        maturity: float,
        factors: npt.NDArray[np.float64],
    ) -> float:
        """Price of a discount bond ``P(now, maturity)`` given factors.

        # C++ parity: ``AffineModel::discountBond(Time, Time, Array)``.
        """
        ...

    @abstractmethod
    def discount_bond_option(
        self,
        option_type: OptionType,
        strike: float,
        maturity: float,
        bond_maturity: float,
    ) -> float:
        """Closed-form discount-bond European option.

        # C++ parity: ``AffineModel::discountBondOption(Type, Real,
        # Time, Time)``.
        """
        ...

    def discount_bond_option_3args(
        self,
        option_type: OptionType,
        strike: float,
        maturity: float,
        bond_start: float,
        bond_maturity: float,
    ) -> float:
        """5-arg overload that defaults to ignoring ``bond_start``.

        # C++ parity: inline ``AffineModel::discountBondOption(Type,
        # Real, Time, Time, Time)`` at model.hpp:151-157 — by default
        # delegates to the 4-arg overload. Subclasses (e.g. HullWhite)
        # override this when the bond_start carries information.
        """
        return self.discount_bond_option(option_type, strike, maturity, bond_maturity)


class ShortRateModel(CalibratedModel):
    """Abstract one-/multi-factor short-rate model.

    # C++ parity: ``class ShortRateModel : public CalibratedModel`` in
    # ql/models/model.hpp:141-145 (v1.42.1).

    Has ``tree(TimeGrid) -> Lattice`` as the only added abstract — its
    implementation depends on TrinomialTree / Lattice lattice machinery
    that is deferred per the L1 carve-outs. Concrete short-rate models
    in L4-B (Vasicek/HullWhite/CIR) declare the method but raise
    ``LibraryException("tree not yet implemented")`` when called.
    """

    def __init__(self, n_arguments: int) -> None:
        # C++ parity: model.cpp ShortRateModel ctor — forwards to
        # CalibratedModel(n_arguments).
        super().__init__(n_arguments)

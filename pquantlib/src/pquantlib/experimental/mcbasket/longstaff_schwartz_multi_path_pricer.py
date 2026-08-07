"""LongstaffSchwartzMultiPathPricer — early-exercise pricer over multi-asset paths.

# C++ parity: ql/experimental/mcbasket/longstaffschwartzmultipathpricer.{hpp,cpp}
#             (v1.43).

References:
    Francis Longstaff, Eduardo Schwartz, 2001. *Valuing American Options by
    Simulation: A Simple Least-Squares Approach*, The Review of Financial
    Studies, Volume 14, No. 1, 113-147.

This is the basket sibling of
:class:`~pquantlib.methods.montecarlo.longstaff_schwartz_path_pricer.LongstaffSchwartzPathPricer`,
and it is a different algorithm rather than a re-parameterisation of it:

* the payoff is a :class:`~pquantlib.experimental.mcbasket.path_payoff.PathPayoff`,
  so every fixing carries a *payment*, an *exercise value* and a *regression
  state* at once, and exercising at ``t`` cancels payments strictly after
  ``t`` — ``payments[t]`` itself is still made;
* a path may be non-exercisable at some fixings (empty state), and those paths
  sit out the regression at those dates;
* on top of the regression, ``calibrate`` compares the fitted policy against
  "always exercise" and "never exercise" and *overrides* the regression when
  either dominates, encoding the winner in the size of ``coeff_[i]``:

  =====================  ===============================================
  ``len(coeff_[i])``     meaning
  =====================  ===============================================
  ``0``                  never exercise at fixing ``i``
  ``len(v_)``            use the fitted continuation value
  ``len(v_) + 1``        always exercise at fixing ``i`` (values unused)
  =====================  ===============================================
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.experimental.mcbasket.path_payoff import PathPayoff
from pquantlib.math.general_linear_least_squares import GeneralLinearLeastSquares
from pquantlib.methods.montecarlo.lsm_basis_system import LsmBasisSystem, PolynomialType
from pquantlib.methods.montecarlo.multi_path import MultiPath
from pquantlib.methods.montecarlo.path_pricer import PathPricer
from pquantlib.termstructures.yield_term_structure import YieldTermStructure

_BasisFunction = Callable[[npt.NDArray[np.float64]], float]

_SUPPORTED_POLYNOMIAL_TYPES: frozenset[PolynomialType] = frozenset(
    {
        PolynomialType.Monomial,
        PolynomialType.Laguerre,
        PolynomialType.Hermite,
        PolynomialType.Hyperbolic,
        PolynomialType.Chebyshev2nd,
    }
)


class PathInfo:
    """The per-path state the Longstaff-Schwartz regression is built from.

    # C++ parity: ``LongstaffSchwartzMultiPathPricer::PathInfo``
    # (longstaffschwartzmultipathpricer.hpp:59-67,
    #  longstaffschwartzmultipathpricer.cpp:28-37).

    Three parallel per-fixing sequences, all of length ``number_of_times``:

    * ``payments`` — the cashflow paid at that fixing;
    * ``exercises`` — what exercising at that fixing would pay;
    * ``states`` — the regression state at that fixing, or an *empty* array,
      which is how the payoff signals "exercise is impossible here".

    The arrays are handed straight to
    :meth:`~pquantlib.experimental.mcbasket.path_payoff.PathPayoff.value` as
    output parameters, exactly as C++ does.
    """

    __slots__ = ("exercises", "payments", "states")

    def __init__(self, number_of_times: int) -> None:
        # C++ parity: ``payments(numberOfTimes, 0.0), exercises(numberOfTimes,
        # 0.0), states(numberOfTimes)`` — the states vector holds
        # ``numberOfTimes`` *empty* Arrays, not zeros.
        self.payments: npt.NDArray[np.float64] = np.zeros(number_of_times, dtype=np.float64)
        self.exercises: npt.NDArray[np.float64] = np.zeros(number_of_times, dtype=np.float64)
        self.states: list[npt.NDArray[np.float64]] = [
            np.empty(0, dtype=np.float64) for _ in range(number_of_times)
        ]

    def path_length(self) -> int:
        """Number of fixings.

        # C++ parity: ``PathInfo::pathLength`` — ``states.size()``, not
        # ``payments.size()``. They are the same by construction.
        """
        return len(self.states)


class LongstaffSchwartzMultiPathPricer(PathPricer[MultiPath]):
    """Longstaff-Schwartz path pricer for early-exercise basket options.

    # C++ parity: ``LongstaffSchwartzMultiPathPricer`` (v1.43).

    Args:
        payoff: the multi-asset path payoff.
        time_positions: index into the sampled path for each fixing.
        forward_term_structures: per-fixing forward yield curves, handed to
            the payoff.
        discounts: discount factor at each fixing (``dF`` in C++).
        polynomial_order: order of the regression basis.
        polynomial_type: family of the regression basis.
    """

    __slots__ = (
        "_calibration_phase",
        "_coeff",
        "_df",
        "_forward_term_structures",
        "_lower_bounds",
        "_paths",
        "_payoff",
        "_time_positions",
        "_v",
    )

    def __init__(
        self,
        payoff: PathPayoff,
        time_positions: Sequence[int],
        forward_term_structures: Sequence[YieldTermStructure],
        discounts: npt.NDArray[np.float64],
        polynomial_order: int,
        polynomial_type: PolynomialType,
    ) -> None:
        # C++ parity: longstaffschwartzmultipathpricer.cpp:39-56.
        qassert.require(
            polynomial_type in _SUPPORTED_POLYNOMIAL_TYPES,
            "insufficient polynomial type",
        )
        self._payoff: PathPayoff = payoff
        # C++ ``coeff_(new Array[timePositions.size() - 1])``: one coefficient
        # vector per *interior* exercise date; the last fixing has no
        # continuation value to regress against.
        self._coeff: list[npt.NDArray[np.float64]] = [
            np.empty(0, dtype=np.float64) for _ in range(len(time_positions) - 1)
        ]
        self._lower_bounds: npt.NDArray[np.float64] = np.zeros(
            len(time_positions), dtype=np.float64
        )
        self._time_positions: tuple[int, ...] = tuple(time_positions)
        self._forward_term_structures: Sequence[YieldTermStructure] = forward_term_structures
        self._df: npt.NDArray[np.float64] = np.asarray(discounts, dtype=np.float64)
        self._v: list[_BasisFunction] = LsmBasisSystem.multi_path_basis_system(
            payoff.basis_system_dimension(), polynomial_order, polynomial_type
        )
        self._calibration_phase: bool = True
        self._paths: list[PathInfo] = []

    # --- inspectors ---------------------------------------------------------

    def basis_system(self) -> list[_BasisFunction]:
        """The regression basis (C++ ``v_``)."""
        return self._v

    def coefficients(self) -> list[npt.NDArray[np.float64]]:
        """Regression coefficients per interior exercise date (C++ ``coeff_``).

        Empty at index ``i`` means "never exercise"; length ``len(v_) + 1``
        means "always exercise" (see the class docstring).
        """
        return self._coeff

    def lower_bounds(self) -> npt.NDArray[np.float64]:
        """Per-fixing lower bound on the continuation value (C++ ``lowerBounds_``)."""
        return self._lower_bounds

    def calibration_phase(self) -> bool:
        """True until :meth:`calibrate` has run (C++ ``calibrationPhase_``)."""
        return self._calibration_phase

    # --- path extraction ----------------------------------------------------

    def transform_path(self, path: MultiPath) -> PathInfo:
        """Slice a sampled path at the fixings and run the payoff over it.

        # C++ parity: ``transformPath`` (longstaffschwartzmultipathpricer.cpp:61-81).
        """
        number_of_assets = path.asset_number()
        number_of_times = len(self._time_positions)

        # C++ fills the Matrix with Null<Real>() first; every cell is then
        # written by the loop below, so the sentinel is never observable.
        path_matrix = np.empty((number_of_assets, number_of_times), dtype=np.float64)
        for i in range(number_of_times):
            pos = self._time_positions[i]
            for j in range(number_of_assets):
                path_matrix[j, i] = path[j].values[pos]

        info = PathInfo(number_of_times)
        self._payoff.value(
            path_matrix,
            self._forward_term_structures,
            info.payments,
            info.exercises,
            info.states,
        )
        return info

    # --- PathPricer contract ------------------------------------------------

    def __call__(self, path: MultiPath) -> float:
        """Record the path (calibration phase) or price it (pricing phase).

        # C++ parity: ``operator()(const MultiPath&)``
        # (longstaffschwartzmultipathpricer.cpp:83-146).
        """
        info = self.transform_path(path)

        if self._calibration_phase:
            # Store the extracted state for calibration — only the relevant
            # part, so unlike the 1-D pricer there is no path to deep-copy.
            self._paths.append(info)
            # Result doesn't matter.
            return 0.0

        # Exercise at time t cancels all payments AFTER t.
        length = info.path_length()
        price = 0.0

        # This is the last event date.
        payment = float(info.payments[length - 1])
        exercise = float(info.exercises[length - 1])
        can_exercise = info.states[length - 1].size != 0
        # At the end the continuation value is 0.0.
        if can_exercise and exercise > 0.0:
            price += exercise
        price += payment

        for i in range(length - 2, -1, -1):
            price *= float(self._df[i + 1]) / float(self._df[i])

            exercise = float(info.exercises[i])
            can_exercise = info.states[i].size != 0

            if can_exercise:
                if self._coeff[i].size == len(self._v) + 1:
                    # Special value: always exercise.
                    price = exercise
                elif self._coeff[i].size != 0 and exercise > float(self._lower_bounds[i]):
                    continuation_value = 0.0
                    for level in range(len(self._v)):
                        continuation_value += float(self._coeff[i][level]) * self._v[level](
                            info.states[i]
                        )
                    if continuation_value < exercise:
                        price = exercise

            price += float(info.payments[i])

        return price * float(self._df[0])

    # --- calibration --------------------------------------------------------

    def calibrate(self) -> None:  # noqa: PLR0915
        """Fit the exercise policy by backward induction over the stored paths.

        # C++ parity: ``calibrate()`` (longstaffschwartzmultipathpricer.cpp:148-274).
        """
        n = len(self._paths)  # number of paths
        prices = np.zeros(n, dtype=np.float64)
        exercise = np.zeros(n, dtype=np.float64)

        basis_dimension = self._payoff.basis_system_dimension()
        length = self._paths[0].path_length()

        # We try to estimate the lower bound of the continuation value, so
        # that only ITM paths contribute to the regression.
        for j in range(n):
            payment = float(self._paths[j].payments[length - 1])
            ex = float(self._paths[j].exercises[length - 1])
            can_exercise = self._paths[j].states[length - 1].size != 0
            # At the end the continuation value is 0.0.
            if can_exercise and ex > 0.0:
                prices[j] += ex
            prices[j] += payment

        self._lower_bounds[length - 1] = float(np.min(prices))

        ls_exercise = [False] * n

        for i in range(length - 2, -1, -1):
            y: list[float] = []
            x: list[npt.NDArray[np.float64]] = []

            # Prices are discounted up to time i.
            discount_ratio = float(self._df[i + 1]) / float(self._df[i])
            prices *= discount_ratio
            self._lower_bounds[i + 1] *= discount_ratio

            # Roll-back step.
            for j in range(n):
                exercise[j] = self._paths[j].exercises[i]

                # If states is empty, no exercise in this path and the path
                # will not participate in the least-squares regression.
                states = self._paths[j].states[i]
                # C++ parity: ``QL_REQUIRE(states.empty() || states.size() ==
                # basisDimension, ...)`` (longstaffschwartzmultipathpricer.cpp:201).
                qassert.require(
                    states.size in (0, basis_dimension),
                    "Invalid size of basis system",
                )

                # Only paths that could potentially create exercise
                # opportunities participate in the regression: if exercise is
                # lower than the minimum continuation value, no point in
                # considering it.
                if states.size != 0 and float(exercise[j]) > float(self._lower_bounds[i + 1]):
                    x.append(states)
                    y.append(float(prices[j]))

            if len(self._v) <= len(x):
                self._coeff[i] = GeneralLinearLeastSquares(
                    cast("Sequence[float | Sequence[float]]", x),
                    y,
                    cast("Sequence[Callable[..., float]]", self._v),
                ).coefficients()
            else:
                # If the number of ITM paths is smaller than the number of
                # calibration functions -> never exercise.
                self._coeff[i] = np.empty(0, dtype=np.float64)

            # Attempt to avoid static arbitrage given by always or never
            # exercising. "always" is absolute: regardless of the lower bound
            # on the continuation value (this could be changed) but it still
            # honours "canExercise".
            sum_optimized = 0.0
            sum_no_exercise = 0.0
            sum_always_exercise = 0.0  # always, if allowed

            k = 0
            for j in range(n):
                sum_no_exercise += float(prices[j])
                ls_exercise[j] = False

                can_exercise = self._paths[j].states[i].size != 0
                if can_exercise:
                    sum_always_exercise += float(exercise[j])
                    if self._coeff[i].size != 0 and float(exercise[j]) > float(
                        self._lower_bounds[i + 1]
                    ):
                        continuation_value = 0.0
                        for level in range(len(self._v)):
                            continuation_value += float(self._coeff[i][level]) * self._v[level](
                                x[k]
                            )

                        if continuation_value < float(exercise[j]):
                            ls_exercise[j] = True
                        k += 1
                else:
                    sum_always_exercise += float(prices[j])

                sum_optimized += float(exercise[j]) if ls_exercise[j] else float(prices[j])

            sum_optimized /= n
            sum_no_exercise /= n
            sum_always_exercise /= n

            if sum_optimized >= sum_no_exercise and sum_optimized >= sum_always_exercise:
                # Accepted LS decision.
                for j in range(n):
                    # ls_exercise already contains "can_exercise".
                    prices[j] = exercise[j] if ls_exercise[j] else prices[j]
            elif sum_always_exercise > sum_no_exercise:
                # Overridden bad LS decision: ALWAYS.
                for j in range(n):
                    can_exercise = self._paths[j].states[i].size != 0
                    prices[j] = exercise[j] if can_exercise else prices[j]
                # Special value to indicate always exercise. C++ writes
                # ``Array(v_.size() + 1)``, which is *uninitialised* — only
                # the size is ever read, so the contents are arbitrary.
                self._coeff[i] = np.zeros(len(self._v) + 1, dtype=np.float64)
            else:
                # Overridden bad LS decision: NEVER. Prices already contain
                # the continuation value; special value to indicate never
                # exercise.
                self._coeff[i] = np.empty(0, dtype=np.float64)

            # Then we add in any case the payment at time t, which is made
            # even if cancellation happens at t.
            for j in range(n):
                prices[j] += float(self._paths[j].payments[i])

            self._lower_bounds[i] = float(np.min(prices))

        # Remove calibration paths.
        self._paths.clear()
        # Entering the calculation phase.
        self._calibration_phase = False


__all__ = ["LongstaffSchwartzMultiPathPricer", "PathInfo"]

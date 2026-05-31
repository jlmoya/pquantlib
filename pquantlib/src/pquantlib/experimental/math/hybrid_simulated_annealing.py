"""Hybrid Simulated Annealing (SA + optional local optimizer).

# C++ parity: ql/experimental/math/hybridsimulatedannealing.hpp
# (v1.42.1) (Copyright 2015 Andres Hernandez).

Implementation follows Ingber (1989), "Very Fast Simulated
Re-Annealing", *Mathl. Comput. Modelling*, 967-973.

The algorithm:

1. ``Sampler`` proposes a new point from the current one (temperature-
   scaled).
2. ``Probability`` (Metropolis) decides whether to accept it.
3. ``Temperature`` is the cooling schedule ``T(k)``.
4. ``Reannealing`` optionally rescales the per-dimension step counters
   to concentrate the search on sensitive dimensions.

The *hybrid* aspect: an optional local optimizer (default Levenberg-
Marquardt) is run whenever a new best — or every accepted — point is
found (selectable via ``optimize_scheme``).

This is a **stochastic** optimizer with a LOOSE convergence contract.
The four functor families are supplied as plain callables (see
``hybrid_simulated_annealing_functors``).
"""

from __future__ import annotations

import sys
from enum import IntEnum
from typing import TYPE_CHECKING, Protocol

import numpy as np
import numpy.typing as npt

from pquantlib.experimental.math.hybrid_simulated_annealing_functors import ReannealingTrivial
from pquantlib.math.optimization.end_criteria import Type
from pquantlib.math.optimization.levenberg_marquardt import LevenbergMarquardt
from pquantlib.math.optimization.optimization_method import OptimizationMethod

if TYPE_CHECKING:
    from pquantlib.math.optimization.end_criteria import EndCriteria
    from pquantlib.math.optimization.problem import Problem

_REAL_MAX: float = sys.float_info.max
_MAX_INTEGER: int = 2**63 - 1


class _Sampler(Protocol):
    def __call__(
        self,
        new_point: npt.NDArray[np.float64],
        current_point: npt.NDArray[np.float64],
        temp: npt.NDArray[np.float64],
    ) -> None: ...


class _Probability(Protocol):
    def __call__(
        self, current_value: float, new_value: float, temp: npt.NDArray[np.float64]
    ) -> bool: ...


class _Temperature(Protocol):
    def __call__(
        self,
        new_temp: npt.NDArray[np.float64],
        curr_temp: npt.NDArray[np.float64],
        steps: npt.NDArray[np.float64],
    ) -> None: ...


class _Reannealing(Protocol):
    def set_problem(self, problem: Problem) -> None: ...
    def __call__(
        self,
        steps: npt.NDArray[np.float64],
        current_point: npt.NDArray[np.float64],
        current_value: float,
        curr_temp: npt.NDArray[np.float64],
    ) -> None: ...


class LocalOptimizeScheme(IntEnum):
    """When to invoke the local optimizer.

    # C++ parity: ``HybridSimulatedAnnealing::LocalOptimizeScheme``
    # hybridsimulatedannealing.hpp:73-77.
    """

    NoLocalOptimize = 0
    EveryNewPoint = 1
    EveryBestPoint = 2


class ResetScheme(IntEnum):
    """How to reset the search state when ``reset_steps`` elapses.

    # C++ parity: ``HybridSimulatedAnnealing::ResetScheme``
    # hybridsimulatedannealing.hpp:78-82.
    """

    NoResetScheme = 0
    ResetToBestPoint = 1
    ResetToOrigin = 2


class HybridSimulatedAnnealing(OptimizationMethod):
    """Hybrid SA + optional local optimizer.

    # C++ parity: ``template <Sampler, Probability, Temperature,
    # Reannealing> class HybridSimulatedAnnealing`` in
    # hybridsimulatedannealing.hpp:70-260 (v1.42.1). The C++ template
    # parameters become constructor arguments (the functors are plain
    # callables).

    Parameters
    ----------
    sampler, probability, temperature:
        The three required functor families.
    reannealing:
        Optional reannealing functor (default trivial / no-op). Must
        expose ``set_problem`` + ``__call__``.
    start_temperature, end_temperature:
        Initial / final annealing temperatures.
    re_anneal_steps:
        Reanneal every this-many iterations (0 -> never).
    reset_scheme, reset_steps:
        Reset policy + period (0 -> never).
    local_optimizer:
        Local optimizer to run per ``optimize_scheme`` (default a fresh
        ``LevenbergMarquardt``; pass ``None`` to disable).
    optimize_scheme:
        When to run the local optimizer.
    """

    def __init__(
        self,
        sampler: _Sampler,
        probability: _Probability,
        temperature: _Temperature,
        reannealing: _Reannealing | None = None,
        start_temperature: float = 200.0,
        end_temperature: float = 0.01,
        re_anneal_steps: int = 50,
        reset_scheme: ResetScheme = ResetScheme.ResetToBestPoint,
        reset_steps: int = 150,
        local_optimizer: OptimizationMethod | None | object = _REAL_MAX,
        optimize_scheme: LocalOptimizeScheme = LocalOptimizeScheme.EveryBestPoint,
    ) -> None:
        self._sampler = sampler
        self._probability = probability
        self._temperature = temperature
        if reannealing is None:
            reannealing = ReannealingTrivial()
        self._reannealing = reannealing
        self._start_temperature = start_temperature
        self._end_temperature = end_temperature
        self._re_anneal_steps = _MAX_INTEGER if re_anneal_steps == 0 else re_anneal_steps
        self._reset_scheme = reset_scheme
        self._reset_steps = _MAX_INTEGER if reset_steps == 0 else reset_steps
        # The sentinel default ``_REAL_MAX`` means "construct a default
        # LevenbergMarquardt"; ``None`` means "no local optimizer".
        if local_optimizer is _REAL_MAX:
            self._local_optimizer: OptimizationMethod | None = LevenbergMarquardt()
        else:
            assert local_optimizer is None or isinstance(local_optimizer, OptimizationMethod)
            self._local_optimizer = local_optimizer
        self._optimize_scheme = (
            optimize_scheme
            if self._local_optimizer is not None
            else LocalOptimizeScheme.NoLocalOptimize
        )

    def minimize(self, problem: Problem, end_criteria: EndCriteria) -> Type:  # noqa: PLR0915
        """Run hybrid SA; return the termination outcome.

        # C++ parity: ``HybridSimulatedAnnealing<...>::minimize``
        # hybridsimulatedannealing.hpp:121-238. The driver is a single
        # faithful translation of one C++ function; PLR0915 (statement
        # count) is waived to preserve a 1:1 structural correspondence.
        """
        ec_type = Type.None_
        problem.reset()
        self._reannealing.set_problem(problem)
        x = problem.current_value
        n = x.size
        k = 1
        k_stationary = 1
        k_re_anneal = 1
        k_reset = 1
        max_k = end_criteria.max_iterations
        max_k_stationary = end_criteria.max_stationary_state
        temperature_breached = False
        current_temperature = np.full(n, self._start_temperature, dtype=np.float64)
        anneal_step = np.ones(n, dtype=np.float64)
        best_point = x.astype(np.float64, copy=True)
        current_point = x.astype(np.float64, copy=True)
        starting_point = x.astype(np.float64, copy=True)
        new_point = x.astype(np.float64, copy=True)
        best_value = problem.value(best_point)
        current_value = best_value
        starting_value = best_value

        while k <= max_k and k_stationary <= max_k_stationary and not temperature_breached:
            # Draw a new sample point.
            self._sampler(new_point, current_point, current_temperature)
            try:
                new_value = problem.value(new_point)

                if self._probability(current_value, new_value, current_temperature):
                    if (
                        self._optimize_scheme == LocalOptimizeScheme.EveryNewPoint
                        and self._local_optimizer is not None
                    ):
                        problem.set_current_value(new_point)
                        problem.set_function_value(new_value)
                        self._local_optimizer.minimize(problem, end_criteria)
                        new_point = problem.current_value.astype(np.float64, copy=True)
                        new_value = problem.function_value
                    current_point = new_point.astype(np.float64, copy=True)
                    current_value = new_value

                if new_value < best_value:
                    if (
                        self._optimize_scheme == LocalOptimizeScheme.EveryBestPoint
                        and self._local_optimizer is not None
                    ):
                        problem.set_current_value(new_point)
                        problem.set_function_value(new_value)
                        self._local_optimizer.minimize(problem, end_criteria)
                        new_point = problem.current_value.astype(np.float64, copy=True)
                        new_value = problem.function_value
                    k_stationary = 0
                    best_value = new_value
                    best_point = new_point.astype(np.float64, copy=True)
            except Exception:
                # C++ parity: ``catch(...) { /* move on to new draw */ }``.
                pass

            # Increase steps.
            k += 1
            k_stationary += 1
            anneal_step += 1.0

            # Reanneal if necessary.
            if k_re_anneal == self._re_anneal_steps:
                k_re_anneal = 0
                self._reannealing(anneal_step, current_point, current_value, current_temperature)
            k_re_anneal += 1

            # Reset if necessary.
            if k_reset == self._reset_steps:
                k_reset = 0
                if self._reset_scheme == ResetScheme.NoResetScheme:
                    pass
                elif self._reset_scheme == ResetScheme.ResetToOrigin:
                    current_point = starting_point.astype(np.float64, copy=True)
                    current_value = starting_value
                elif self._reset_scheme == ResetScheme.ResetToBestPoint:
                    current_point = best_point.astype(np.float64, copy=True)
                    current_value = best_value
            k_reset += 1

            # Update the current temperature according to current step.
            self._temperature(current_temperature, current_temperature, anneal_step)

            # Check if temperature condition is breached.
            # C++ parity note: the C++ loop initialises ``temperature
            # Breached`` once (false) and updates it with
            # ``temperatureBreached = temperatureBreached && ...``, so it
            # can never flip to true (latent bug) — the loop terminates on
            # max-iterations / stationary instead. Reproduced verbatim.
            for i in range(n):
                temperature_breached = (
                    temperature_breached and current_temperature[i] < self._end_temperature
                )

        if k > max_k:
            ec_type = Type.MaxIterations
        elif k_stationary > max_k_stationary:
            ec_type = Type.StationaryPoint

        problem.set_current_value(best_point)
        problem.set_function_value(best_value)
        return ec_type

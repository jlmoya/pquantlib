"""FdndimBlackScholesVanillaEngine — n-dimensional FD Black-Scholes basket engine.

# C++ parity: ql/pricingengines/basket/fdndimblackscholesvanillaengine.{hpp,cpp}
# (v1.43) — ``class FdndimBlackScholesVanillaEngine : public BasketOption::engine``.

The engine does **not** discretise the n correlated log-spots directly. It
diagonalises the covariance matrix (``SymmetricSchurDecomposition`` of
``getCovariance(vols, rho)``) and solves an *uncorrelated* n-dimensional
Wiener PDE (``FdmWienerOp``) in principal-component space; the payoff is
mapped back through ``S = exp(Q x - v t / 2 + log S0) * q_i(t) / r(t)`` by
``detail::FdmPCABasketInnerValue``.

Consequences a port must not "simplify" away:

* Each axis is a ``Predefined1dMesher`` on ``1.3 sqrt(l_i T) N^{-1}(eps + j h)``
  — not a Black-Scholes mesher.
* The solution is read at the **origin** of every axis, ``interpolateAt([0]*n)``.
* For a European exercise the operator gets a *null* discount curve and the
  final value is multiplied by the interest-rate discount factor afterwards;
  for early exercise the curve goes into the operator instead. That asymmetry
  is the whole reason the European branch discounts at the end.
* The C++ ``PDE_MAX_SUPPORTED_DIM`` preprocessor constant is 4, and the engine
  ``QL_REQUIRE``s that the number of processes does not exceed it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import final

import numpy as np

from pquantlib import qassert
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.basket_option import BasketOptionResults, BasketPayoff
from pquantlib.math.array import Array
from pquantlib.math.closeness import close_enough
from pquantlib.math.distributions.inverse_cumulative_normal import (
    InverseCumulativeNormal,
)
from pquantlib.math.matrix import Matrix
from pquantlib.math.matrixutilities.get_covariance import get_covariance
from pquantlib.math.matrixutilities.symmetric_schur_decomposition import (
    SymmetricSchurDecomposition,
)
from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.meshers.predefined_1d_mesher import (
    Predefined1dMesher,
)
from pquantlib.methods.finitedifferences.operators.fdm_linear_op_layout import (
    FdmLinearOpIterator,
)
from pquantlib.methods.finitedifferences.operators.fdm_wiener_op import FdmWienerOp
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import FdmSchemeDesc
from pquantlib.methods.finitedifferences.solvers.fdm_ndim_solver import FdmNdimSolver
from pquantlib.methods.finitedifferences.solvers.fdm_solver_desc import FdmSolverDesc
from pquantlib.methods.finitedifferences.step_conditions.fdm_step_condition_composite import (
    FdmStepConditionComposite,
)
from pquantlib.methods.finitedifferences.utilities.fdm_inner_value_calculator import (
    FdmInnerValueCalculator,
)
from pquantlib.option import OptionArguments
from pquantlib.pricingengines.basket.vector_bsm_process_extractor import (
    VectorBsmProcessExtractor,
)
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.termstructures.yield_term_structure import YieldTermStructure

#: # C++ parity: ``#define PDE_MAX_SUPPORTED_DIM 4``
#: (fdndimblackscholesvanillaengine.cpp:125-127).
PDE_MAX_SUPPORTED_DIM: int = 4

#: # C++ parity: ``const Real eps = 1e-4;`` in ``calculate()``.
_EPS: float = 1e-4


@final
class FdmPCABasketInnerValue(FdmInnerValueCalculator):
    """Basket payoff seen from principal-component space.

    # C++ parity: ``class detail::FdmPCABasketInnerValue``
    # (fdndimblackscholesvanillaengine.cpp:38-90).
    """

    def __init__(
        self,
        payoff: BasketPayoff,
        mesher: FdmMesher,
        log_s0: Array,
        vols: Array,
        q_ts: Sequence[YieldTermStructure],
        r_ts: YieldTermStructure,
        q: Matrix,
        lambdas: Array,
    ) -> None:
        self._n: int = log_s0.shape[0]
        self._payoff: BasketPayoff = payoff
        self._mesher: FdmMesher = mesher
        self._log_s0: Array = log_s0
        self._v: Array = vols * vols
        self._q_ts: list[YieldTermStructure] = list(q_ts)
        self._r_ts: YieldTermStructure = r_ts
        self._q: Matrix = q
        self._l: Array = lambdas
        # C++ parity: ``cachedT_(Null<Real>())``. C++ never *assigns*
        # ``cachedT_`` anywhere, so the guard below is always true and the
        # discount factors are re-read on every call — the cache is dead code
        # in v1.43. The port reproduces the behaviour, not the intent.
        self._cached_t: float | None = None
        self._qf: Array = np.zeros(self._n, dtype=np.float64)
        self._rf: float = 0.0

    def inner_value(self, iterator: FdmLinearOpIterator, t: float) -> float:
        """# C++ parity: ``FdmPCABasketInnerValue::innerValue``."""
        if self._cached_t is None or not close_enough(t, self._cached_t):
            self._rf = self._r_ts.discount(t)
            for i in range(self._n):
                self._qf[i] = self._q_ts[i].discount(t)
        x = np.array(
            [self._mesher.location(iterator, i) for i in range(self._n)],
            dtype=np.float64,
        )
        s = np.exp(self._q @ x - 0.5 * self._v * t + self._log_s0) * self._qf / self._rf
        return self._payoff.evaluate([float(v) for v in s])

    def avg_inner_value(self, iterator: FdmLinearOpIterator, t: float) -> float:
        """# C++ parity: ``FdmPCABasketInnerValue::avgInnerValue`` — same value."""
        return self.inner_value(iterator, t)


@final
class FdndimBlackScholesVanillaEngine(GenericEngine[OptionArguments, BasketOptionResults]):
    """n-dimensional finite-differences Black-Scholes basket engine.

    # C++ parity: ``class FdndimBlackScholesVanillaEngine``.

    C++ has two constructors: one taking a per-asset ``std::vector<Size>``
    of grid sizes, one taking a single ``Size`` that the others are scaled
    from by ``xGrid * (l_i/l_0)^0.1``. Python takes ``x_grids`` as either an
    ``int`` or a sequence, exactly reproducing both.
    """

    def __init__(
        self,
        processes: Sequence[GeneralizedBlackScholesProcess],
        rho: Matrix,
        x_grids: int | Sequence[int],
        t_grid: int = 50,
        damping_steps: int = 0,
        scheme_desc: FdmSchemeDesc | None = None,
    ) -> None:
        super().__init__(OptionArguments(), BasketOptionResults())
        self._processes: list[GeneralizedBlackScholesProcess] = list(processes)
        self._rho: Matrix = np.asarray(rho, dtype=np.float64)
        self._x_grids: list[int] = (
            [int(x_grids)] if isinstance(x_grids, int) else [int(g) for g in x_grids]
        )
        self._t_grid: int = t_grid
        self._damping_steps: int = damping_steps
        # C++ parity: ``schemeDesc = FdmSchemeDesc::Douglas()``.
        self._scheme_desc: FdmSchemeDesc = (
            scheme_desc if scheme_desc is not None else FdmSchemeDesc.douglas()
        )

        qassert.require(len(self._processes) > 0, "no Black-Scholes process is given.")
        qassert.require(
            self._rho.shape[0] == self._rho.shape[1]
            and self._rho.shape[0] == len(self._processes),
            "correlation matrix has the wrong size.",
        )
        qassert.require(
            len(self._x_grids) == 1 or len(self._x_grids) == len(self._processes),
            "wrong number of xGrids is given.",
        )

        # C++ parity: ``for_each(... registerWith(p))``.
        for p in self._processes:
            p.register_with(self)

    def calculate(self) -> None:
        """# C++ parity: ``FdndimBlackScholesVanillaEngine::calculate`` (cpp:124-224)."""
        args = self._arguments
        results = self._results

        n_proc = len(self._processes)
        qassert.require(
            n_proc <= PDE_MAX_SUPPORTED_DIM,
            f"This engine does not support {n_proc} underlyings. "
            f"Max number of underlyings is {PDE_MAX_SUPPORTED_DIM}. "
            "Please change preprocessor constant PDE_MAX_SUPPORTED_DIM and recompile "
            "if a larger number of underlyings is needed.",
        )

        qassert.require(args.exercise is not None, "no exercise given")
        assert args.exercise is not None
        exercise = args.exercise

        maturity_date = exercise.last_date()
        maturity = self._processes[0].time(maturity_date)
        sqrt_t = np.sqrt(maturity)

        p_extractor = VectorBsmProcessExtractor(self._processes)
        s = p_extractor.get_spot()
        std_dev = np.sqrt(p_extractor.get_black_variance(maturity_date))
        vols = std_dev / sqrt_t

        schur = SymmetricSchurDecomposition(
            get_covariance([float(v) for v in vols], self._rho)
        )
        q = schur.eigenvectors()
        lambdas = schur.eigenvalues()

        inv_cum_normal = InverseCumulativeNormal()
        meshers: list[Predefined1dMesher] = []
        for i in range(n_proc):
            x_grid = (
                self._x_grids[i]
                if len(self._x_grids) > 1
                else max(4, int(self._x_grids[0] * (lambdas[i] / lambdas[0]) ** 0.1))
            )
            qassert.require(x_grid >= 4, "minimum grid size is four")

            x_step_size = (1.0 - 2 * _EPS) / (x_grid - 1)
            x = [
                1.3 * float(np.sqrt(lambdas[i])) * sqrt_t * inv_cum_normal(_EPS + j * x_step_size)
                for j in range(x_grid)
            ]
            meshers.append(Predefined1dMesher(x))

        mesher = FdmMesherComposite(*meshers)

        qassert.require(isinstance(args.payoff, BasketPayoff), "basket payoff expected")
        assert isinstance(args.payoff, BasketPayoff)
        payoff: BasketPayoff = args.payoff

        r_ts = self._processes[0].risk_free_rate()
        q_ts = [p.dividend_yield() for p in self._processes]

        calculator = FdmPCABasketInnerValue(
            payoff, mesher, np.log(s), std_dev / sqrt_t, q_ts, r_ts, q, lambdas
        )

        conditions = FdmStepConditionComposite.vanilla_composite(
            (), exercise, mesher, calculator, r_ts.reference_date(), r_ts.day_counter()
        )

        solver_desc = FdmSolverDesc(
            mesher=mesher,
            condition=conditions,
            calculator=calculator.avg_inner_value,
            maturity=maturity,
            time_steps=self._t_grid,
            damping_steps=self._damping_steps,
        )

        # C++ parity: ``dynamic_pointer_cast<EuropeanExercise>`` — the *type*
        # of the exercise object, not its ``type()`` tag.
        is_european = isinstance(exercise, EuropeanExercise)
        op = FdmWienerOp(mesher, None if is_european else r_ts, lambdas)

        value = FdmNdimSolver(solver_desc, self._scheme_desc, op).interpolate_at(
            [0.0] * n_proc
        )

        if is_european:
            value *= p_extractor.get_interest_rate_df(maturity_date)

        results.value = value


__all__ = [
    "PDE_MAX_SUPPORTED_DIM",
    "FdmPCABasketInnerValue",
    "FdndimBlackScholesVanillaEngine",
]

"""LiborForwardModelProcess — the LFM stochastic process (rolling forward measure).

# C++ parity: ql/legacy/libormarketmodels/lfmprocess.{hpp,cpp} (v1.43).

References:

- Glasserman, Paul, 2004, *Monte Carlo Methods in Financial Engineering*, §3.7
- Antoon Pelsser, 2000, *Efficient Methods for Valuing Interest Rate
  Derivatives*, ch. 8
- Hull, John, White, Alan, 1999, *Forward Rate Volatilities, Swap Rate
  Volatilities and the Implementation of the Libor Market Model*

The state vector is the set of forward rates fixing on the index's own
schedule. Drift and evolution are taken under the rolling ("spot") forward
measure, and :meth:`evolve` implements the predictor-corrector step rather
than the plain Euler step of the base class.

Handle indirection: C++ reads the forwarding curve through
``index->forwardingTermStructure()``, an ``Handle<YieldTermStructure>``. This
port does not implement ``Handle``/``RelinkableHandle``; the index holds the
term structure directly (``IborIndex.forecast_term_structure()``).
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.ibor_coupon import IborCoupon, IborLeg
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.legacy.libormarketmodels.lfm_covar_param import (
    LfmCovarianceParameterization,
)
from pquantlib.math.array import Array
from pquantlib.math.matrix import Matrix
from pquantlib.processes.euler_discretization import EulerDiscretization
from pquantlib.processes.stochastic_process import StochasticProcess
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule


class LiborForwardModelProcess(StochasticProcess):
    """LIBOR-forward-model process.

    # C++ parity: ``class LiborForwardModelProcess : public StochasticProcess``
    # (lfmprocess.hpp:58-107).
    """

    def __init__(self, size: int, index: IborIndex) -> None:
        # C++ parity: lfmprocess.cpp:33-66.
        super().__init__(EulerDiscretization())
        self._size: int = size
        self._index: IborIndex = index
        self._lfm_param: LfmCovarianceParameterization | None = None

        day_counter = index.day_counter()
        flows = self.cash_flows()

        qassert.require(size == len(flows), "wrong number of cashflows")

        ts = index.forecast_term_structure()
        qassert.require(ts is not None, "null term structure set to this instance of the index")
        assert ts is not None
        settlement = ts.reference_date()

        first = flows[0]
        qassert.require(isinstance(first, IborCoupon), "irregular coupon types are not suppported")
        assert isinstance(first, IborCoupon)
        start_date = first.fixing_date()

        self._initial_values: Array = np.zeros(size, dtype=np.float64)
        self._fixing_times: list[float] = [0.0] * size
        self._fixing_dates: list[Date] = [Date()] * size
        self._accrual_start_times: list[float] = [0.0] * size
        self._accrual_end_times: list[float] = [0.0] * size
        self._accrual_period: list[float] = [0.0] * size

        for i in range(size):
            coupon = flows[i]
            qassert.require(
                isinstance(coupon, IborCoupon), "irregular coupon types are not suppported"
            )
            assert isinstance(coupon, IborCoupon)
            qassert.require(
                coupon.date() == coupon.accrual_end_date(),
                "irregular coupon types are not suppported",
            )

            self._initial_values[i] = coupon.rate()
            self._accrual_period[i] = coupon.accrual_period()

            self._fixing_dates[i] = coupon.fixing_date()
            self._fixing_times[i] = day_counter.year_fraction(start_date, coupon.fixing_date())
            self._accrual_start_times[i] = day_counter.year_fraction(
                settlement, coupon.accrual_start_date()
            )
            self._accrual_end_times[i] = day_counter.year_fraction(
                settlement, coupon.accrual_end_date()
            )

        # C++ keeps ``m1`` / ``m2`` as mutable scratch Arrays on the process.
        self._m1: Array = np.zeros(size, dtype=np.float64)
        self._m2: Array = np.zeros(size, dtype=np.float64)

    # --- StochasticProcess interface --------------------------------------

    def size(self) -> int:
        """# C++ parity: lfmprocess.cpp:183-185."""
        return self._size

    def factors(self) -> int:
        """# C++ parity: lfmprocess.cpp:187-189 — delegates to the covariance
        parameterization, so it is only defined once one is set.
        """
        return self._covar_param_or_fail().factors()

    def initial_values(self) -> Array:
        """# C++ parity: lfmprocess.cpp:147-149."""
        return self._initial_values.copy()

    def drift(self, t: float, x: npt.NDArray[np.float64]) -> Array:
        """Drift under the rolling forward measure.

        # C++ parity: lfmprocess.cpp:68-83. Forwards that have already fixed
        (index below ``next_index_reset(t)``) keep a zero drift.
        """
        param = self._covar_param_or_fail()
        f = np.zeros(self._size, dtype=np.float64)
        covariance = param.covariance(t, x)

        m = self.next_index_reset(t)

        for k in range(m, self._size):
            self._m1[k] = self._accrual_period[k] * x[k] / (1 + self._accrual_period[k] * x[k])
            f[k] = (
                float(np.dot(self._m1[m : k + 1], covariance[m : k + 1, k]))
                - 0.5 * covariance[k, k]
            )

        return f

    def diffusion(self, t: float, x: npt.NDArray[np.float64]) -> Matrix:
        """# C++ parity: lfmprocess.cpp:85-87."""
        return self._covar_param_or_fail().diffusion(t, x)

    def covariance(
        self, t0: float, x0: npt.NDArray[np.float64], dt: float
    ) -> Matrix:
        """# C++ parity: lfmprocess.cpp:89-91 — the parameterization's
        instantaneous covariance SCALED by ``dt`` (no discretization object is
        consulted).
        """
        return self._covar_param_or_fail().covariance(t0, x0) * dt

    def apply(
        self, x0: npt.NDArray[np.float64], dx: npt.NDArray[np.float64]
    ) -> Array:
        """# C++ parity: lfmprocess.cpp:93-101 — multiplicative, ``x0 exp(dx)``."""
        tmp = np.zeros(self._size, dtype=np.float64)
        for k in range(self._size):
            tmp[k] = x0[k] * math.exp(dx[k])
        return tmp

    def evolve(
        self,
        t0: float,
        x0: npt.NDArray[np.float64],
        dt: float,
        dw: npt.NDArray[np.float64],
    ) -> Array:
        """Predictor-corrector step.

        # C++ parity: lfmprocess.cpp:103-145. The C++ comment spells out the
        short-but-slow equivalent; this is the fused version, term for term.
        Forwards below ``next_index_reset(t0)`` are copied over unchanged
        (``Array f(x0)``).
        """
        param = self._covar_param_or_fail()
        m = self.next_index_reset(t0)
        sdt = math.sqrt(dt)

        f = np.array(x0, dtype=np.float64, copy=True)
        diff = param.diffusion(t0, x0)
        covariance = param.covariance(t0, x0)

        for k in range(m, self._size):
            y = self._accrual_period[k] * x0[k]
            self._m1[k] = y / (1 + y)
            d = (
                float(np.dot(self._m1[m : k + 1], covariance[m : k + 1, k]))
                - 0.5 * covariance[k, k]
            ) * dt

            r = float(np.dot(diff[k], dw)) * sdt

            x = y * math.exp(d + r)
            self._m2[k] = x / (1 + x)
            f[k] = x0[k] * math.exp(
                0.5
                * (
                    d
                    + (
                        float(np.dot(self._m2[m : k + 1], covariance[m : k + 1, k]))
                        - 0.5 * covariance[k, k]
                    )
                    * dt
                )
                + r
            )

        return f

    # --- covariance parameterization --------------------------------------

    def set_covar_param(self, param: LfmCovarianceParameterization) -> None:
        """# C++ parity: lfmprocess.cpp:151-154."""
        self._lfm_param = param

    def covar_param(self) -> LfmCovarianceParameterization | None:
        """# C++ parity: lfmprocess.cpp:156-159."""
        return self._lfm_param

    def _covar_param_or_fail(self) -> LfmCovarianceParameterization:
        """C++ dereferences a null ``shared_ptr`` here; raise instead."""
        qassert.require(
            self._lfm_param is not None,
            "no covariance parameterization given (call set_covar_param first)",
        )
        assert self._lfm_param is not None
        return self._lfm_param

    # --- convenience support methods --------------------------------------

    def index(self) -> IborIndex:
        """# C++ parity: lfmprocess.cpp:161-164."""
        return self._index

    def cash_flows(self, amount: float = 1.0) -> list[CashFlow]:
        """The Ibor leg whose coupons define the model's forward rates.

        # C++ parity: lfmprocess.cpp:166-181.
        """
        ts = self._index.forecast_term_structure()
        qassert.require(ts is not None, "null term structure set to this instance of the index")
        assert ts is not None
        ref_date = ts.reference_date()
        tenor = self._index.tenor()
        schedule = Schedule.from_rule(
            ref_date,
            ref_date + Period(tenor.length * self._size, tenor.units),
            tenor,
            self._index.fixing_calendar(),
            self._index.business_day_convention(),
            self._index.business_day_convention(),
            DateGeneration.Forward,
            False,
        )
        return (
            IborLeg(schedule, self._index)
            .with_notionals(amount)
            .with_payment_day_counter(self._index.day_counter())
            .with_payment_adjustment(self._index.business_day_convention())
            .with_fixing_days(self._index.fixing_days())
            .build()
        )

    def next_index_reset(self, t: float) -> int:
        """Index of the first forward rate that has NOT yet fixed at ``t``.

        # C++ parity: lfmprocess.cpp:209-212 —
        # ``upper_bound(fixingTimes_.begin(), fixingTimes_.end(), t) - begin``.
        # Strict: at ``t`` exactly on a fixing time the answer is the NEXT
        # index, which is what the C++ test-suite testInitialisation asserts.
        """
        lo, hi = 0, len(self._fixing_times)
        while lo < hi:
            mid = (lo + hi) // 2
            if t < self._fixing_times[mid]:
                hi = mid
            else:
                lo = mid + 1
        return lo

    def fixing_times(self) -> list[float]:
        """# C++ parity: lfmprocess.cpp:191-193."""
        return list(self._fixing_times)

    def fixing_dates(self) -> list[Date]:
        """# C++ parity: lfmprocess.cpp:195-197."""
        return list(self._fixing_dates)

    def accrual_start_times(self) -> list[float]:
        """# C++ parity: lfmprocess.cpp:199-202."""
        return list(self._accrual_start_times)

    def accrual_end_times(self) -> list[float]:
        """# C++ parity: lfmprocess.cpp:204-207."""
        return list(self._accrual_end_times)

    def discount_bond(self, rates: list[float]) -> list[float]:
        """Discount factors implied by a realisation of the forward rates.

        # C++ parity: lfmprocess.cpp:214-226.
        """
        discount_factors = [0.0] * self._size
        discount_factors[0] = 1.0 / (1.0 + rates[0] * self._accrual_period[0])
        for i in range(1, self._size):
            discount_factors[i] = discount_factors[i - 1] / (
                1.0 + rates[i] * self._accrual_period[i]
            )
        return discount_factors


__all__ = ["LiborForwardModelProcess"]

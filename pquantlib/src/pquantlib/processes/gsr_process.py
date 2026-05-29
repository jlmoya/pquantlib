"""GsrProcess — Gaussian short-rate (GSR) state process under the forward measure.

# C++ parity: ql/processes/gsrprocess.{hpp,cpp} + ql/processes/gsrprocesscore.{hpp,cpp}
# @ v1.42.1 (099987f0).

The GSR process is a piecewise-constant-vol/-reversion 1-factor short-rate
state process. The dynamics under the T-forward measure are

    dx_t = (y(t) - G(t,T) sigma(t)^2 - kappa(t) x_t) dt + sigma(t) dW_t

with x_0 = 0, ``sigma(t)`` piecewise constant, ``kappa(t)`` piecewise
constant (or strictly constant), and the curve-fitting term carried
externally by the GSR model. For a derivation of all closed-form
quantities (y, G, expectation, variance) see Caspers 2013
http://ssrn.com/abstract=2246013.

Implementation notes:

- The C++ separates ``GsrProcessCore`` (cache-heavy math) from
  ``GsrProcess`` (StochasticProcess1D + ForwardMeasureProcess1D
  surface). PQuantLib collapses both into this module:
  ``_GsrProcessCore`` is the cache + closed-form arithmetic, and
  ``GsrProcess`` is the public 1-D forward-measure process.

- Reversions with ``|kappa| < 1e-4`` are treated as "approximately
  zero" to avoid 1/kappa numerical blowup. The C++ uses a fixed
  ``1e-4`` threshold; we mirror it.

- All cached quantities (``y``, ``G``, ``variance``, ``expectation_*``)
  use plain Python dicts as memo tables (matching the C++
  ``std::map`` use; we don't need ordering).

- The ``yGrid`` helper is on the model side (Gaussian1dModel), not here.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.math.constants import QL_EPSILON, QL_MIN_POSITIVE_REAL
from pquantlib.processes.forward_measure_process import ForwardMeasureProcess1D

if TYPE_CHECKING:
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.time.date import Date


# C++ parity: gsrprocesscore.cpp:71 — small-reversion threshold below
# which 1/kappa terms switch to the algebraic limit.
_REV_ZERO_THRESHOLD: float = 1.0e-4


class _GsrProcessCore:
    """Cached closed-form arithmetic for the GSR process.

    # C++ parity: ``QuantLib::detail::GsrProcessCore`` in
    # gsrprocesscore.{hpp,cpp} (v1.42.1).

    Holds the piecewise volatility / reversion grids plus five memo
    tables (``cache1`` through ``cache5``). Mutators flush the caches.

    The closed-form formulae assume:
    - ``times_`` has size ``N - 1`` (step times)
    - ``vols_`` has size ``N`` (one per piece)
    - ``reversions_`` has size ``N`` (one per piece) or 1 (constant)
    """

    __slots__ = (
        "_T",
        "_cache1",
        "_cache2a",
        "_cache2b",
        "_cache3",
        "_cache4",
        "_cache5",
        "_rev_zero",
        "_reversions",
        "_times",
        "_vols",
    )

    def __init__(
        self,
        times: npt.NDArray[np.float64],
        vols: npt.NDArray[np.float64],
        reversions: npt.NDArray[np.float64],
        T: float = 60.0,  # noqa: N803 — math symbol
    ) -> None:
        self._times: npt.NDArray[np.float64] = np.asarray(times, dtype=np.float64)
        self._vols: npt.NDArray[np.float64] = np.asarray(vols, dtype=np.float64)
        self._reversions: npt.NDArray[np.float64] = np.asarray(reversions, dtype=np.float64)
        # Math-symbol slot name. C++ parity: ``T_`` in gsrprocesscore.hpp:96.
        self._T: float = float(T)  # pyright: ignore[reportConstantRedefinition]
        self._rev_zero: list[bool] = [False] * len(self._reversions)
        self._cache1: dict[tuple[float, float], float] = {}
        self._cache2a: dict[tuple[float, float], float] = {}
        self._cache2b: dict[tuple[float, float], float] = {}
        self._cache3: dict[tuple[float, float], float] = {}
        self._cache4: dict[float, float] = {}
        self._cache5: dict[tuple[float, float], float] = {}
        self.flush_cache()
        self._check_times_vols_reversions()

    # --- mutators (flush cache after changing the grids) ----------------

    def set_times(self, times: npt.NDArray[np.float64]) -> None:
        # C++ parity: gsrprocesscore.cpp:36-39.
        self._times = np.asarray(times, dtype=np.float64)
        self._check_times_vols_reversions()

    def set_vols(self, vols: npt.NDArray[np.float64]) -> None:
        # C++ parity: gsrprocesscore.cpp:41-44.
        self._vols = np.asarray(vols, dtype=np.float64)
        self._check_times_vols_reversions()

    def set_reversions(self, reversions: npt.NDArray[np.float64]) -> None:
        # C++ parity: gsrprocesscore.cpp:46-49.
        self._reversions = np.asarray(reversions, dtype=np.float64)
        self._rev_zero = [False] * len(self._reversions)
        self._check_times_vols_reversions()

    def flush_cache(self) -> None:
        """Reset memoization caches and recompute the rev-zero mask.

        # C++ parity: gsrprocesscore.cpp:68-82.
        """
        for i in range(len(self._reversions)):
            self._rev_zero[i] = abs(self._reversions[i]) < _REV_ZERO_THRESHOLD
        self._cache1.clear()
        self._cache2a.clear()
        self._cache2b.clear()
        self._cache3.clear()
        self._cache4.clear()
        self._cache5.clear()

    def _check_times_vols_reversions(self) -> None:
        # C++ parity: gsrprocesscore.cpp:51-66.
        qassert.require(
            self._times.size == self._vols.size - 1,
            f"number of volatilities ({self._vols.size}) compared to number of "
            f"times ({self._times.size}) must be bigger by one",
        )
        qassert.require(
            self._times.size == self._reversions.size - 1
            or self._reversions.size == 1,
            f"number of reversions ({self._reversions.size}) compared to number of "
            f"times ({self._times.size}) must be bigger by one, or exactly 1 "
            "reversion must be given",
        )
        for i in range(int(self._times.size) - 1):
            qassert.require(
                self._times[i] < self._times[i + 1],
                f"times must be increasing ({self._times[i]}@{i}, "
                f"{self._times[i + 1]}@{i + 1})",
            )

    # --- piecewise accessors --------------------------------------------

    def _vol(self, index: int) -> float:
        # C++ parity: gsrprocesscore.cpp:363-367.
        if index >= self._vols.size:
            return float(self._vols[-1])
        return float(self._vols[index])

    def _rev(self, index: int) -> float:
        # C++ parity: gsrprocesscore.cpp:369-373.
        if index >= self._reversions.size:
            return float(self._reversions[-1])
        return float(self._reversions[index])

    def _rev_zero_at(self, index: int) -> bool:
        # C++ parity: gsrprocesscore.cpp:375-379.
        if index >= len(self._rev_zero):
            return self._rev_zero[-1]
        return self._rev_zero[index]

    def _lower_index(self, t: float) -> int:
        # C++ parity: gsrprocesscore.cpp:330-333 — upper_bound on times.
        return int(np.searchsorted(self._times, t, side="right"))

    def _upper_index(self, t: float) -> int:
        # C++ parity: gsrprocesscore.cpp:335-342.
        if t < QL_MIN_POSITIVE_REAL:
            return 0
        return int(np.searchsorted(self._times, t - QL_EPSILON, side="right")) + 1

    def _time2(self, index: int) -> float:
        # C++ parity: gsrprocesscore.cpp:353-361.
        if index == 0:
            return 0.0
        if index > self._times.size:
            return self._T
        return float(self._times[index - 1])

    def _capped_time(self, index: int, cap: float | None = None) -> float:
        # C++ parity: gsrprocesscore.cpp:344-346.
        return min(cap, self._time2(index)) if cap is not None else self._time2(index)

    def _floored_time(self, index: int, floor: float | None = None) -> float:
        # C++ parity: gsrprocesscore.cpp:348-351.
        return (
            max(floor, self._time2(index)) if floor is not None else self._time2(index)
        )

    # --- scalar accessors ----------------------------------------------

    def sigma(self, t: float) -> float:
        # C++ parity: gsrprocesscore.hpp:101-103 (inline).
        return self._vol(self._lower_index(t))

    def reversion(self, t: float) -> float:
        # C++ parity: gsrprocesscore.hpp:105-107 (inline).
        return self._rev(self._lower_index(t))

    # --- closed-form quantities -----------------------------------------

    def y(self, t: float) -> float:
        """``y(t)`` — accumulated quadratic forward-state covariance.

        # C++ parity: gsrprocesscore.cpp:282-304.
        """
        k = self._cache4.get(t)
        if k is not None:
            return k

        res = 0.0
        for i in range(0, self._upper_index(t)):
            res2 = 1.0
            for j in range(i + 1, self._upper_index(t)):
                res2 *= math.exp(
                    -2.0 * self._rev(j) * (self._capped_time(j + 1, t) - self._time2(j))
                )
            v2 = self._vol(i) * self._vol(i)
            ct = self._capped_time(i + 1, t)
            tt = self._time2(i)
            if self._rev_zero_at(i):
                res2 *= v2 * (ct - tt)
            else:
                rev = self._rev(i)
                res2 *= v2 / (2.0 * rev) * (1.0 - math.exp(-2.0 * rev * (ct - tt)))
            res += res2

        self._cache4[t] = res
        return res

    def G(self, t: float, w: float) -> float:  # noqa: N802 — math symbol
        """``G(t, w)`` — Hull-White-style integral.

        # C++ parity: gsrprocesscore.cpp:306-328.
        """
        key = (t, w)
        k = self._cache5.get(key)
        if k is not None:
            return k

        res = 0.0
        for i in range(self._lower_index(t), self._upper_index(w)):
            res2 = 1.0
            for j in range(self._lower_index(t), i):
                res2 *= math.exp(
                    -self._rev(j) * (self._time2(j + 1) - self._floored_time(j, t))
                )
            cap = self._capped_time(i + 1, w)
            fl = self._floored_time(i, t)
            if self._rev_zero_at(i):
                res2 *= cap - fl
            else:
                rev = self._rev(i)
                res2 *= (1.0 - math.exp(-rev * (cap - fl))) / rev
            res += res2

        self._cache5[key] = res
        return res

    def variance(self, w: float, dt: float) -> float:
        """Conditional variance ``Var[x(w+dt) | F_w]``.

        # C++ parity: gsrprocesscore.cpp:252-280.
        """
        t = w + dt
        key = (w, t)
        k = self._cache3.get(key)
        if k is not None:
            return k

        res = 0.0
        for k_ in range(self._lower_index(w), self._upper_index(t)):
            res2 = self._vol(k_) * self._vol(k_)
            fl = self._floored_time(k_, w)
            cap = self._capped_time(k_ + 1, t)
            if self._rev_zero_at(k_):
                res2 *= -(fl - cap)
            else:
                rev = self._rev(k_)
                res2 *= (1.0 - math.exp(2.0 * rev * (fl - cap))) / (2.0 * rev)
            for i in range(k_ + 1, self._upper_index(t)):
                res2 *= math.exp(
                    -2.0 * self._rev(i) * (self._capped_time(i + 1, t) - self._time2(i))
                )
            res += res2

        self._cache3[key] = res
        return res

    def expectation_x0_dep_part(self, w: float, xw: float, dt: float) -> float:
        """x0-dependent part of conditional expectation.

        # C++ parity: gsrprocesscore.cpp:84-99.
        """
        t = w + dt
        key = (w, t)
        k = self._cache1.get(key)
        if k is not None:
            return xw * k

        res2 = 1.0
        for i in range(self._lower_index(w), self._upper_index(t)):
            res2 *= math.exp(
                -self._rev(i) * (self._capped_time(i + 1, t) - self._floored_time(i, w))
            )
        self._cache1[key] = res2
        return res2 * xw

    def expectation_rn_part(self, w: float, dt: float) -> float:
        """Risk-neutral-measure x0-independent part of conditional expectation.

        # C++ parity: gsrprocesscore.cpp:101-172.
        """
        t = w + dt
        key = (w, t)
        k = self._cache2a.get(key)
        if k is not None:
            return k

        res = 0.0
        for k_idx in range(self._lower_index(w), self._upper_index(t)):
            # l < k_idx
            for el in range(0, k_idx):
                res2 = 1.0
                # alpha_l
                vl = self._vol(el)
                if self._rev_zero_at(el):
                    res2 *= vl * vl * (self._time2(el + 1) - self._time2(el))
                else:
                    revl = self._rev(el)
                    res2 *= (
                        vl
                        * vl
                        / (2.0 * revl)
                        * (1.0 - math.exp(-2.0 * revl * (self._time2(el + 1) - self._time2(el))))
                    )
                # zeta_i (i > k_idx)
                for i in range(k_idx + 1, self._upper_index(t)):
                    res2 *= math.exp(
                        -self._rev(i) * (self._capped_time(i + 1, t) - self._time2(i))
                    )
                # beta_j (j between l+1 and k_idx-1)
                for j in range(el + 1, k_idx):
                    res2 *= math.exp(
                        -2.0 * self._rev(j) * (self._time2(j + 1) - self._time2(j))
                    )
                # zeta_k beta_k
                tk = self._time2(k_idx)
                fl = self._floored_time(k_idx, w)
                cap = self._capped_time(k_idx + 1, t)
                if self._rev_zero_at(k_idx):
                    res2 *= (
                        2.0 * tk - fl - cap
                        - 2.0 * (tk - cap)
                    )
                else:
                    rev = self._rev(k_idx)
                    res2 *= (
                        math.exp(rev * (2.0 * tk - fl - cap))
                        - math.exp(2.0 * rev * (tk - cap))
                    ) / rev
                res += res2
            # l = k_idx
            res2 = 1.0
            # alpha_k zeta_k
            vk = self._vol(k_idx)
            fl = self._floored_time(k_idx, w)
            cap = self._capped_time(k_idx + 1, t)
            tk = self._time2(k_idx)
            if self._rev_zero_at(k_idx):
                res2 *= (
                    vk
                    * vk
                    / 4.0
                    * (
                        4.0 * (cap - tk) ** 2
                        - (
                            (fl - 2.0 * tk + cap) ** 2
                            + (cap - fl) ** 2
                        )
                    )
                )
            else:
                rev = self._rev(k_idx)
                res2 *= (
                    vk
                    * vk
                    / (2.0 * rev * rev)
                    * (
                        math.exp(-2.0 * rev * (cap - tk))
                        + 1.0
                        - (
                            math.exp(-rev * (fl - 2.0 * tk + cap))
                            + math.exp(-rev * (cap - fl))
                        )
                    )
                )
            # zeta_i (i > k_idx)
            for i in range(k_idx + 1, self._upper_index(t)):
                res2 *= math.exp(
                    -self._rev(i) * (self._capped_time(i + 1, t) - self._time2(i))
                )
            res += res2

        self._cache2a[key] = res
        return res

    def expectation_tf_part(self, w: float, dt: float) -> float:
        """T-forward-measure correction part of conditional expectation.

        # C++ parity: gsrprocesscore.cpp:174-250.
        """
        t = w + dt
        key = (w, t)
        k = self._cache2b.get(key)
        if k is not None:
            return k

        res = 0.0
        for k_idx in range(self._lower_index(w), self._upper_index(t)):
            res2 = 0.0
            # l > k_idx
            for el in range(k_idx + 1, self._upper_index(self._T)):
                res3 = 1.0
                # eta_l
                if self._rev_zero_at(el):
                    res3 *= self._capped_time(el + 1, self._T) - self._time2(el)
                else:
                    revl = self._rev(el)
                    res3 *= (
                        1.0
                        - math.exp(
                            -revl * (self._capped_time(el + 1, self._T) - self._time2(el))
                        )
                    ) / revl
                # zeta_i (i > k_idx)
                for i in range(k_idx + 1, self._upper_index(t)):
                    res3 *= math.exp(
                        -self._rev(i) * (self._capped_time(i + 1, t) - self._time2(i))
                    )
                # gamma_j (j > k_idx)
                for j in range(k_idx + 1, el):
                    res3 *= math.exp(
                        -self._rev(j) * (self._time2(j + 1) - self._time2(j))
                    )
                # zeta_k gamma_k
                cap_t = self._capped_time(k_idx + 1, t)
                tk1 = self._time2(k_idx + 1)
                fl = self._floored_time(k_idx, w)
                if self._rev_zero_at(k_idx):
                    res3 *= (
                        cap_t
                        - tk1
                        - (2.0 * fl - cap_t - tk1)
                    ) / 2.0
                else:
                    rev = self._rev(k_idx)
                    res3 *= (
                        math.exp(rev * (cap_t - tk1))
                        - math.exp(rev * (2.0 * fl - cap_t - tk1))
                    ) / (2.0 * rev)
                res2 += res3
            # l = k_idx
            res3 = 1.0
            cap_t = self._capped_time(k_idx + 1, t)
            cap_T = self._capped_time(k_idx + 1, self._T)  # noqa: N806 — math symbol
            fl = self._floored_time(k_idx, w)
            if self._rev_zero_at(k_idx):
                res3 *= (
                    -(cap_t - cap_T) ** 2
                    - 2.0 * (cap_t - fl) ** 2
                    + (2.0 * fl - cap_T - cap_t) ** 2
                ) / 4.0
            else:
                rev = self._rev(k_idx)
                res3 *= (
                    2.0
                    - math.exp(rev * (cap_t - cap_T))
                    - (
                        2.0 * math.exp(-rev * (cap_t - fl))
                        - math.exp(rev * (2.0 * fl - cap_T - cap_t))
                    )
                ) / (2.0 * rev * rev)
            for i in range(k_idx + 1, self._upper_index(t)):
                res3 *= math.exp(
                    -self._rev(i) * (self._capped_time(i + 1, t) - self._time2(i))
                )
            res2 += res3
            res += -self._vol(k_idx) * self._vol(k_idx) * res2

        self._cache2b[key] = res
        return res


class GsrProcess(ForwardMeasureProcess1D):
    """GSR stochastic process — piecewise vol + reversion, T-forward measure.

    # C++ parity: ``class GsrProcess : public ForwardMeasureProcess1D`` in
    # ql/processes/gsrprocess.{hpp,cpp} (v1.42.1).

    Backs ``Gsr`` (the Gaussian1d short-rate model). Provides the
    StochasticProcess1D surface (``x0``, ``drift_1d``, ``diffusion_1d``,
    ``expectation_1d``, ``std_deviation_1d``, ``variance_1d``) plus the
    GSR-specific inspectors (``sigma(t)``, ``reversion(t)``, ``y(t)``,
    ``G(t, T)``).

    Construction:

    - ``times`` — step times (length N-1 for N pieces)
    - ``vols`` — piecewise volatilities (length N)
    - ``reversions`` — piecewise mean-reversions (length N or 1)
    - ``T`` — forward-measure horizon (defaults 60Y)
    - ``reference_date``, ``day_counter`` — optional, used only by
      ``time(Date)`` (matches C++)

    """

    __slots__ = (
        "_core",
        "_day_counter",
        "_reference_date",
    )

    def __init__(
        self,
        times: npt.NDArray[np.float64],
        vols: npt.NDArray[np.float64],
        reversions: npt.NDArray[np.float64],
        T: float = 60.0,  # noqa: N803 — math symbol
        reference_date: Date | None = None,
        day_counter: DayCounter | None = None,
    ) -> None:
        # C++ parity: gsrprocess.cpp:26-35.
        super().__init__(T=T)
        self._core: _GsrProcessCore = _GsrProcessCore(times, vols, reversions, T)
        self._reference_date: Date | None = reference_date
        self._day_counter: DayCounter | None = day_counter
        self.flush_cache()

    # --- mutators (forward to the core) --------------------------------

    def set_times(self, times: npt.NDArray[np.float64]) -> None:
        # C++ parity: gsrprocess.hpp:74 (private setter, friend to Gsr).
        self._core.set_times(times)

    def set_vols(self, vols: npt.NDArray[np.float64]) -> None:
        # C++ parity: gsrprocess.hpp:75 (private setter, friend to Gsr).
        self._core.set_vols(vols)

    def set_reversions(self, reversions: npt.NDArray[np.float64]) -> None:
        # C++ parity: gsrprocess.hpp:76 (private setter, friend to Gsr).
        self._core.set_reversions(reversions)

    def flush_cache(self) -> None:
        """Reset memoization caches.

        # C++ parity: gsrprocess.hpp:90-92 (inline).
        """
        self._core.flush_cache()

    def set_forward_measure_time(self, T: float) -> None:  # noqa: N803 — math symbol
        """Update forward-measure horizon, flushing the cache first.

        # C++ parity: gsrprocess.hpp:85-88 (inline).
        """
        self.flush_cache()
        super().set_forward_measure_time(T)

    def _check_T(self, t: float) -> None:  # noqa: N802 — math symbol
        # C++ parity: gsrprocess.cpp:37-42.
        qassert.require(
            t <= self.get_forward_measure_time() and t >= 0.0,
            f"t ({t}) must not be greater than forward measure time "
            f"({self.get_forward_measure_time()}) and non-negative",
        )

    # --- StochasticProcess1D surface -----------------------------------

    def x0(self) -> float:
        # C++ parity: gsrprocess.cpp:51.
        return 0.0

    def drift_1d(self, t: float, x: float) -> float:
        # C++ parity: gsrprocess.cpp:53-57.
        return (
            self._core.y(t)
            - self._core.G(t, self.get_forward_measure_time())
            * self.sigma(t)
            * self.sigma(t)
            - self.reversion(t) * x
        )

    def diffusion_1d(self, t: float, x: float) -> float:
        # C++ parity: gsrprocess.cpp:59-62.
        _ = x  # diffusion is x-independent
        self._check_T(t)
        return self.sigma(t)

    def expectation_1d(self, t0: float, x0: float, dt: float) -> float:
        # C++ parity: gsrprocess.cpp:64-69.
        self._check_T(t0 + dt)
        return (
            self._core.expectation_x0_dep_part(t0, x0, dt)
            + self._core.expectation_rn_part(t0, dt)
            + self._core.expectation_tf_part(t0, dt)
        )

    def std_deviation_1d(self, t0: float, x0: float, dt: float) -> float:
        # C++ parity: gsrprocess.cpp:73-75.
        return math.sqrt(self.variance_1d(t0, x0, dt))

    def variance_1d(self, t0: float, x0: float, dt: float) -> float:
        # C++ parity: gsrprocess.cpp:77-80.
        _ = x0  # variance is x0-independent
        self._check_T(t0 + dt)
        return self._core.variance(t0, dt)

    # --- additional inspectors -----------------------------------------

    def sigma(self, t: float) -> float:
        # C++ parity: gsrprocess.cpp:82.
        return self._core.sigma(t)

    def reversion(self, t: float) -> float:
        # C++ parity: gsrprocess.cpp:84.
        return self._core.reversion(t)

    def y(self, t: float) -> float:
        # C++ parity: gsrprocess.cpp:86-89.
        self._check_T(t)
        return self._core.y(t)

    def G(self, t: float, w: float, x: float = 0.0) -> float:  # noqa: N802 — math symbol
        """G(t, w) integral.

        The ``x`` argument is accepted for C++-parity but is unused
        (G is x-independent in the GSR forward measure).

        # C++ parity: gsrprocess.cpp:91-100.
        """
        _ = x
        qassert.require(
            w >= t, f"G(t,w) should be called with w ({w}) not lesser than t ({t})"
        )
        qassert.require(
            t >= 0.0 and w <= self.get_forward_measure_time(),
            f"G(t,w) should be called with (t,w)=({t},{w}) in Range "
            f"[0,{self.get_forward_measure_time()}].",
        )
        return self._core.G(t, w)

    def time(self, date: Date) -> float:
        """Year fraction from the process's reference date to ``date``.

        # C++ parity: gsrprocess.cpp:44-49.

        Raises if neither ``reference_date`` nor ``day_counter`` were
        provided at construction time.
        """
        qassert.require(
            self._reference_date is not None and self._day_counter is not None,
            "time can not be computed without reference date and day counter",
        )
        # Help pyright narrow the optional types.
        assert self._reference_date is not None
        assert self._day_counter is not None
        return self._day_counter.year_fraction(self._reference_date, date)


__all__ = ["GsrProcess"]

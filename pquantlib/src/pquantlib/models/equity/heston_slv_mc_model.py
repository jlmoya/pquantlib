"""HestonSlvMcModel — Heston SLV calibration via MC bucketing.

# C++ parity: ql/models/equity/hestonslvmcmodel.{hpp,cpp} (v1.42.1).

The Monte-Carlo variant of the Heston stochastic-local-vol calibration.
Instead of solving the Fokker-Planck PDE (as ``HestonSlvFdmModel``
does), the calibration draws ``n_paths`` joint Heston paths and at
each time step buckets the simulated ``(S, V)`` pairs by ``S``; within
each bucket it estimates ``E[V | S, t]`` and sets the leverage
function::

    L(S_bucket, t)^2 = sigma_LV(S_bucket, t)^2 / E[V | S_bucket, t]

This is mathematically equivalent to the FDM variant (both invert the
same conditional-expectation identity) but the MC approach scales
better in dimension and is simpler to implement — at the cost of
sampling noise that requires LOOSE tolerances.

Reference:
- van der Stoep, A.W., Grzelak, L.A., Oosterlee, C.W. 2013. The
  Heston Stochastic-Local Volatility Model: Efficient Monte Carlo
  Simulation. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2278122

L11-W1-D scope: ports the MC bucketing calibration. The C++
``BrownianGeneratorFactory`` abstraction is replaced by an inline
``numpy.random.Generator`` (the L11-W1-D scope doesn't require
plugging in alternative Brownian schemes — a future commit can
add the BrownianGeneratorFactory abstraction if needed for
``QuasiRandomBridgedBrownianGenerator`` parity).

Divergences from C++:
- ``BrownianGeneratorFactory`` collapses to a NumPy ``Generator``
  (seedable via ``rng`` kwarg).
- The C++ code uses ``boost::multi_array`` to pre-allocate all
  Brownian increments; Python uses a single ``(n_paths, n_steps, 2)``
  ndarray for the same effect.
- ``HestonSLVProcess.evolve`` is not yet ported. The Python
  implementation evolves the Heston SDE with the leverage scaling
  applied via the FixedLocalVolSurface inline (matches the C++
  HestonSLVProcess::evolve full-truncation semantics — see
  hestonslvprocess.cpp:48-72).
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from pquantlib.models.equity.heston_model import HestonModel
from pquantlib.processes.heston_process import HestonProcess
from pquantlib.termstructures.volatility.equity_fx.fixed_local_vol_surface import (
    FixedLocalVolSurface,
)
from pquantlib.termstructures.volatility.equity_fx.local_vol_term_structure import (
    LocalVolTermStructure,
)
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.time_grid import TimeGrid


class HestonSlvMcModel:
    """Heston-SLV model calibrated via MC bucketing.

    # C++ parity: ``class HestonSLVMCModel : public LazyObject`` in
    # hestonslvmcmodel.hpp:43-73 (v1.42.1).
    """

    __slots__ = (
        "_end_date",
        "_heston_model",
        "_leverage_function",
        "_local_vol",
        "_mixing_factor",
        "_n_bins",
        "_n_paths",
        "_rng",
        "_time_grid",
        "_time_steps_per_year",
    )

    def __init__(
        self,
        *,
        local_vol: LocalVolTermStructure,
        heston_model: HestonModel,
        end_date: Date,
        time_steps_per_year: int = 365,
        n_bins: int = 201,
        calibration_paths: int = 1 << 15,
        mixing_factor: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Construct the model.

        # C++ parity: ``HestonSLVMCModel`` ctor in
        # hestonslvmcmodel.hpp:45-53.

        Parameters
        ----------
        local_vol:
            Target local-vol surface to re-price.
        heston_model:
            Underlying Heston model.
        end_date:
            Calibration horizon.
        time_steps_per_year:
            Time-step density (max 365/day; the MC bucketing tolerates
            sparser grids).
        n_bins:
            Number of spot-bucket strikes per time slice.
        calibration_paths:
            Number of MC paths in the bucketing.
        mixing_factor:
            Heston-vol mixing scale.
        rng:
            NumPy random generator (replaces C++ BrownianGeneratorFactory).
            Defaults to a freshly-seeded default_rng.
        """
        self._local_vol: LocalVolTermStructure = local_vol
        self._heston_model: HestonModel = heston_model
        self._end_date: Date = end_date
        self._time_steps_per_year: int = time_steps_per_year
        self._n_bins: int = n_bins
        self._n_paths: int = calibration_paths
        self._mixing_factor: float = mixing_factor
        self._rng: np.random.Generator = (
            rng if rng is not None else np.random.default_rng()
        )
        self._leverage_function: FixedLocalVolSurface | None = None

        # Build the time grid (single mandatory date at end).
        process = heston_model.process()
        dc = process.risk_free_rate().day_counter()
        ref = process.risk_free_rate().reference_date()
        end_time = dc.year_fraction(ref, end_date)
        n_steps = max(2, int(end_time * time_steps_per_year))
        self._time_grid: TimeGrid = TimeGrid.with_mandatory_and_steps(
            [end_time], n_steps
        )

    # --- inspectors -----------------------------------------------------

    def heston_process(self) -> HestonProcess:
        """The Heston process driving the stochastic-vol dynamics.

        # C++ parity: hestonslvmcmodel.hpp:55.
        """
        return self._heston_model.process()

    def local_vol(self) -> LocalVolTermStructure:
        """The target local-vol surface.

        # C++ parity: hestonslvmcmodel.hpp:56.
        """
        return self._local_vol

    def leverage_function(self) -> FixedLocalVolSurface:
        """The calibrated leverage function ``L(S, t)``.

        # C++ parity: hestonslvmcmodel.hpp:57 + ``performCalculations``
        # in hestonslvmcmodel.cpp:85-188.

        Lazy: computes the calibration on first call, caches the
        result.
        """
        if self._leverage_function is None:
            self._leverage_function = self._calibrate()
        return self._leverage_function

    def time_grid(self) -> TimeGrid:
        """The time grid used by the bucketing."""
        return self._time_grid

    def mixing_factor(self) -> float:
        """The Heston-vol mixing scale."""
        return self._mixing_factor

    def end_date(self) -> Date:
        """The calibration horizon."""
        return self._end_date

    # --- calibration ----------------------------------------------------

    def _calibrate(self) -> FixedLocalVolSurface:  # noqa: PLR0915 - parity with C++
        """MC-bucketing calibration of the leverage function.

        # C++ parity: ``HestonSLVMCModel::performCalculations`` in
        # hestonslvmcmodel.cpp:85-188.

        Length matches the C++ structure so the time-step / bucketing
        loop reads in parallel with the source.
        """
        process = self._heston_model.process()
        ref_date = process.risk_free_rate().reference_date()
        dc = process.risk_free_rate().day_counter()
        spot = process.s0().value()
        v0 = process.v0
        rho = process.rho
        sqrt_one_minus_rho2 = math.sqrt(1.0 - rho * rho)
        kappa = process.kappa
        theta = process.theta
        sigma = process.sigma * self._mixing_factor

        n_steps = len(self._time_grid) - 1
        n_paths = self._n_paths
        n_bins = self._n_bins

        # Initial leverage L(K, 0) = sigma_LV(0, S0) / sqrt(V0).
        lv0_value = self._local_vol.local_vol_at_time(
            self._time_grid[0], spot, extrapolate=True
        )
        lv0 = lv0_value / math.sqrt(v0)

        # Strike grids per time slice: spread spot * (1 + j*dx) around spot.
        u = n_bins // 2
        dx0 = spot * math.sqrt(np.finfo(np.float64).eps)
        all_strikes: list[list[float]] = []
        for _ in range(n_steps + 1):
            all_strikes.append(
                [spot + (j - u) * dx0 for j in range(n_bins)]
            )

        # Leverage matrix initialized to lv0 in column 0 (t=0).
        leverage: npt.NDArray[np.float64] = np.zeros((n_bins, n_steps + 1), dtype=np.float64)
        leverage[:, 0] = lv0

        # Build the surface (so we can read L during the evolve loop).
        surface = FixedLocalVolSurface(
            reference_date=ref_date,
            times=list(self._time_grid),
            strikes=all_strikes,
            local_vol_matrix=leverage,
            day_counter=dc,
        )

        # Per-path state: (S, V) — start all paths at (spot, v0).
        s_paths = np.full(n_paths, spot, dtype=np.float64)
        v_paths = np.full(n_paths, v0, dtype=np.float64)

        for n in range(1, n_steps + 1):
            t_prev = self._time_grid[n - 1]
            t_cur = self._time_grid[n]
            dt = t_cur - t_prev
            sqrt_dt = math.sqrt(dt)

            # Draw correlated 2D Brownian increments.
            dw = self._rng.standard_normal((n_paths, 2))
            dw_s = dw[:, 0]
            dw_v = rho * dw[:, 0] + sqrt_one_minus_rho2 * dw[:, 1]

            # Read forward rates over [t_prev, t_cur].
            r = process.risk_free_rate().forward_rate(
                t_prev,
                t_cur,
                Compounding.Continuous,
                Frequency.NoFrequency,
                True,
            ).rate()
            q = process.dividend_yield().forward_rate(
                t_prev,
                t_cur,
                Compounding.Continuous,
                Frequency.NoFrequency,
                True,
            ).rate()

            # Evolve V with FullTruncation.
            v_pos = np.maximum(v_paths, 0.0)
            sqrt_v = np.sqrt(v_pos)
            v_paths = v_paths + kappa * (theta - v_pos) * dt + sigma * sqrt_v * sqrt_dt * dw_v
            # Read L(S, t_prev) per path.
            l_per_path = np.array(
                [surface.local_vol_at_time(t_prev, s, extrapolate=True) for s in s_paths],
                dtype=np.float64,
            )
            # Evolve S with the leverage scaling applied to the vol term.
            drift = (r - q - 0.5 * l_per_path * l_per_path * v_pos) * dt
            diff = l_per_path * sqrt_v * sqrt_dt * dw_s
            s_paths = s_paths * np.exp(drift + diff)

            # Sort path states by S; bucket into n_bins groups; compute
            # E[V | S in bucket] and L(S_bucket, t_cur).
            order = np.argsort(s_paths)
            s_sorted = s_paths[order]
            v_sorted = v_paths[order]
            k_floor = n_paths // n_bins
            m = n_paths % n_bins
            new_strikes = [0.0] * n_bins
            new_col = np.zeros(n_bins, dtype=np.float64)
            s_offset = 0
            for j in range(n_bins):
                inc = k_floor + (1 if j < m else 0)
                e = s_offset + inc
                # Use slice endpoints as the bucket strike representative.
                new_strikes[j] = 0.5 * (s_sorted[e - 1] + s_sorted[s_offset])
                bucket_v_mean = float(v_sorted[s_offset:e].mean()) if inc > 0 else v0
                # Read sigma_LV at the bucket strike.
                bucket_lv = self._local_vol.local_vol_at_time(
                    t_prev, new_strikes[j], extrapolate=True
                )
                # L^2 = sigma_LV^2 / E[V | S, t]; clamp E[V] > eps.
                bucket_v_mean = max(bucket_v_mean, 1e-12)
                new_col[j] = math.sqrt(bucket_lv * bucket_lv / bucket_v_mean)
                s_offset = e
            surface.set_column(n, new_strikes, new_col)

        return surface


__all__ = ["HestonSlvMcModel"]

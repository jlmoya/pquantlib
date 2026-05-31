"""HestonSlvFdmModel — Heston stochastic-local-vol model (Fokker-Planck FDM).

# C++ parity: ql/models/equity/hestonslvfdmmodel.{hpp,cpp} (v1.42.1).

The Heston stochastic-local-volatility model combines a Heston
stochastic-vol diffusion with a leverage function ``L(S, t)`` chosen
so that the resulting joint SDE re-prices vanilla European options
under an input local-vol surface::

    dS(t) = (r - q) * S * dt + L(S, t) * sqrt(V) * S * dW_1
    dV(t) = kappa * (theta - V) * dt + mixing * sigma * sqrt(V) * dW_2

The C++ ``HestonSLVFDMModel`` calibrates ``L(S, t)`` by solving the
2-D Fokker-Planck PDE for the joint density ``p(S, V, t)`` on a 3-D
FDM grid (S, V, t), then setting::

    L(S, t)^2 = sigma_LV(S, t)^2 / E[V | S, t]
              = sigma_LV(S, t)^2 / int V * p(S, V, t) dV
                                 / int p(S, V, t) dV

This is one of the most numerically sophisticated pieces of QuantLib —
537 LOC of advanced PDE machinery depending on
``FdmHestonGreensFct``, ``FdmHestonFwdOp``, ``FdmSquareRootFwdOp``,
``Concentrating1dMesher``, ``LocalVolRNDCalculator``,
``SquareRootProcessRNDCalculator``, and 6 different time-stepping
schemes (Douglas / Craig-Sneyd / Modified Craig-Sneyd / Hundsdorfer /
explicit Euler / implicit Euler) with Rannacher smoothing.

L11-W1-D scope (scaffold + carve-out): pquantlib's FDM framework
covers 1-D Black-Scholes (L5-D). The 2-D Heston FDM operators (and
the Fokker-Planck forward variant) are scheduled for Phase 11 W5
(see ``docs/migration/phase11-plan.md``). Until then, this module
ships the public API skeleton:

* Constructor that accepts the same arguments as C++ + the
  ``HestonSLVFokkerPlanckFdmParams``-equivalent kwargs as a flat
  named-tuple parameter pack.
* ``leverage_function()`` — returns a ``FixedLocalVolSurface`` with
  unit leverage (``L = 1`` everywhere). At unit leverage the model
  reduces algebraically to a pure Heston model; vanilla options
  price exactly via ``AnalyticHestonEngine``. This is a degenerate
  but well-defined behaviour useful for round-tripping tests.
* ``heston_process()`` — passthrough.
* ``local_vol()`` — passthrough.

**Phase 11 W5-C status — still deferred.** The W5-C cluster landed
``FdmZabrOp`` + ``FdmDupire1dOp`` + ``FdmOrnsteinUhlenbeckOp`` +
``FdOrnsteinUhlenbeckVanillaEngine`` + ``FdmExtOUJumpModelInnerValue``
+ ``NinePointLinearOp`` + ``SecondOrderMixedDerivativeOp`` (closing
~70% of the cluster scope). The HestonSLV FDM wiring required:

  * ``FdmHestonFwdOp`` (2-D forward Heston operator)
  * ``FdmSquareRootFwdOp`` (1-D variance-direction forward operator)
  * ``Concentrating1dMesher`` (strike-anchored mesh refinement)
  * ``LocalVolRNDCalculator`` + ``FdmHestonGreensFct`` (local-vol
    density target + Green's function for the calibration)
  * 6 multi-time-stepping schemes (Craig-Sneyd, Hundsdorfer-Verwer,
    TR-BDF2, MoL, plus Rannacher smoothing variants)

— totalling ~1500 LOC of advanced PDE machinery. **Cumulative budget
exceeded the W5-C cluster scope.** The carve-out is logged
comprehensively in ``docs/carve-outs.md`` § "Multi-asset finite-
difference". The full implementation is a documented Phase 12
deliverable.

Divergences from C++ (beyond the carve-out):
- ``HestonSLVFokkerPlanckFdmParams`` is replaced by a dataclass with
  the same field names. The default values mirror the C++ test
  defaults (see ``ql/test-suite/hestonslv.cpp``).
- ``mandatoryDates`` collapses to a single ``end_date`` parameter
  since the scaffold doesn't use the intermediate dates.
- ``Logging`` mode is dropped (the calibration loop isn't ported).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pquantlib.models.equity.heston_model import HestonModel
from pquantlib.processes.heston_process import HestonProcess
from pquantlib.termstructures.volatility.equity_fx.fixed_local_vol_surface import (
    FixedLocalVolSurface,
)
from pquantlib.termstructures.volatility.equity_fx.local_vol_term_structure import (
    LocalVolTermStructure,
)
from pquantlib.time.date import Date


@dataclass(frozen=True, slots=True)
class HestonSlvFokkerPlanckFdmParams:
    """Parameter pack for the Fokker-Planck FDM SLV calibration.

    # C++ parity: ``struct HestonSLVFokkerPlanckFdmParams`` in
    # hestonslvfdmmodel.hpp:43-71.

    Field names + defaults mirror the C++ test-suite defaults (see
    ``ql/test-suite/hestonslv.cpp``). At this scaffold level only the
    grid dimensions (``x_grid``, ``v_grid``, ``t_steps_per_year``) are
    surfaced for inspection; the rest are kept for API parity once
    the full FDM implementation lands in Phase 11 W5.
    """

    x_grid: int = 201
    v_grid: int = 51
    t_max_steps_per_year: int = 200
    t_min_steps_per_year: int = 4
    t_step_number_decay: float = 0.0001
    n_rannacher_time_steps: int = 2
    prediction_correction_steps: int = 0
    x0_density: float = 0.1
    local_vol_eps_prob: float = 1e-6
    max_integration_iterations: int = 25_000
    v_lower_eps: float = 1e-6
    v_upper_eps: float = 1e-6
    v_min: float = 0.0005
    v0_density: float = 1.0
    v_lower_bound_density: float = 0.1
    v_upper_bound_density: float = 0.1
    leverage_fct_prop_eps: float = 1e-8


class HestonSlvFdmModel:
    """Scaffold for the Heston-SLV FDM-calibrated model.

    # C++ parity: ``class HestonSLVFDMModel : public LazyObject`` in
    # hestonslvfdmmodel.hpp:73-109 (v1.42.1).

    L11-W1-D ships the public API; the Fokker-Planck FDM solver is
    deferred to Phase 11 W5. ``leverage_function()`` currently
    returns a unit-leverage ``FixedLocalVolSurface`` (the SLV model
    reduces algebraically to the input Heston model at L=1).

    See module docstring for the carve-out rationale.
    """

    __slots__ = (
        "_end_date",
        "_heston_model",
        "_leverage_function",
        "_local_vol",
        "_mixing_factor",
        "_params",
    )

    def __init__(
        self,
        *,
        local_vol: LocalVolTermStructure,
        heston_model: HestonModel,
        end_date: Date,
        params: HestonSlvFokkerPlanckFdmParams | None = None,
        mixing_factor: float = 1.0,
    ) -> None:
        """Construct the model.

        # C++ parity: ``HestonSLVFDMModel`` ctor in
        # hestonslvfdmmodel.hpp:75-81.

        Parameters
        ----------
        local_vol:
            Target local-vol surface to re-price.
        heston_model:
            Underlying Heston model providing the stochastic vol
            dynamics.
        end_date:
            Calibration horizon (latest date the leverage function
            needs to be valid).
        params:
            Fokker-Planck FDM parameter pack. Defaults to the C++
            test-suite values.
        mixing_factor:
            Heston-vol mixing scale (1.0 = full Heston vol amplitude).
        """
        self._local_vol: LocalVolTermStructure = local_vol
        self._heston_model: HestonModel = heston_model
        self._end_date: Date = end_date
        self._mixing_factor: float = mixing_factor
        self._params: HestonSlvFokkerPlanckFdmParams = (
            params if params is not None else HestonSlvFokkerPlanckFdmParams()
        )
        self._leverage_function: FixedLocalVolSurface | None = None

    # --- inspectors -----------------------------------------------------

    def heston_process(self) -> HestonProcess:
        """The Heston process driving the stochastic-vol dynamics.

        # C++ parity: hestonslvfdmmodel.hpp:83 + .cpp ``hestonProcess()``.
        """
        return self._heston_model.process()

    def local_vol(self) -> LocalVolTermStructure:
        """The target local-vol surface.

        # C++ parity: hestonslvfdmmodel.hpp:84 + .cpp ``localVol()``.
        """
        return self._local_vol

    def leverage_function(self) -> FixedLocalVolSurface:
        """The calibrated leverage function ``L(S, t)``.

        # C++ parity: hestonslvfdmmodel.hpp:85.

        Phase 11 W1-D scaffold returns a unit-leverage surface
        (``L = 1`` everywhere on a coarse grid). This makes the SLV
        model reduce algebraically to the input Heston model — the
        leverage function is the "do-nothing" leverage. A future
        Phase-11 W5 commit will replace this with the actual
        Fokker-Planck FDM calibration solver.
        """
        if self._leverage_function is None:
            self._leverage_function = self._build_unit_leverage()
        return self._leverage_function

    def params(self) -> HestonSlvFokkerPlanckFdmParams:
        """The FDM parameter pack used for calibration."""
        return self._params

    def mixing_factor(self) -> float:
        """The Heston-vol mixing scale."""
        return self._mixing_factor

    def end_date(self) -> Date:
        """The calibration horizon."""
        return self._end_date

    # --- internal helpers -----------------------------------------------

    def _build_unit_leverage(self) -> FixedLocalVolSurface:
        """Build a constant-1.0 leverage function on a coarse grid.

        # Scaffold-only: real calibration will replace this with the
        # Fokker-Planck FDM output (Phase 11 W5).
        """
        process = self.heston_process()
        spot = process.s0().value()
        ref_date = process.risk_free_rate().reference_date()
        dc = process.risk_free_rate().day_counter()

        # Build a small (5 time, 5 strike) grid centred on spot.
        end_time = dc.year_fraction(ref_date, self._end_date)
        times = [end_time * i / 4.0 for i in range(5)]
        # Strikes: spot * exp(+/-2 stdev) range.
        strikes = [
            [spot * 0.5, spot * 0.75, spot, spot * 1.5, spot * 2.0]
            for _ in range(5)
        ]
        # L = 1.0 everywhere.
        mat = np.ones((5, 5), dtype=np.float64)
        return FixedLocalVolSurface(
            reference_date=ref_date,
            times=times,
            strikes=strikes,
            local_vol_matrix=mat,
            day_counter=dc,
        )


__all__ = ["HestonSlvFdmModel", "HestonSlvFokkerPlanckFdmParams"]

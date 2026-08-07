"""HestonSLVFDMModel — Heston stochastic-local-vol model (Fokker-Planck FDM).

# C++ parity: ql/models/equity/hestonslvfdmmodel.{hpp,cpp} (v1.43).

The Heston stochastic-local-volatility model combines a Heston
stochastic-vol diffusion with a leverage function ``L(S, t)`` chosen so
that the resulting joint SDE re-prices vanilla European options under an
input local-vol surface::

    dS(t) = (r - q) * S * dt + L(S, t) * sqrt(V) * S * dW_1
    dV(t) = kappa * (theta - V) * dt + mixing * sigma * sqrt(V) * dW_2

``L`` is calibrated by solving the 2-D Fokker-Planck PDE forward for the
joint density ``p(S, V, t)`` and setting, slice by slice,

    L(S, t)^2 = sigma_LV(S, t)^2 * int p dV / int V p dV

which is what ``performCalculations`` does: seed with
:class:`FdmHestonGreensFct`, then step :class:`FdmHestonFwdOp` forward,
reading the leverage column off the density at every time step and feeding
it straight back into the operator.

Two things are easy to get wrong, and both are load-bearing:

* the leverage column for step ``i`` is written into the surface BEFORE the
  step that produces ``p`` at time ``t_i``, and the operator holds a
  reference to that same surface — so the write is what makes the
  calibration self-consistent, not a bookkeeping afterthought;
* the first ``nRannacherTimeSteps`` steps use implicit Euler regardless of
  the requested scheme (Rannacher smoothing of the Dirac-like initial
  density); the requested scheme only takes over from step
  ``nRannacherTimeSteps + 2``.

A note on the C++ this replaces
-------------------------------
This module was a scaffold whose ``leverage_function()`` returned ``L = 1``
everywhere on a 5x5 grid, behind a carve-out saying ``FdmHestonFwdOp``,
``FdmSquareRootFwdOp``, ``Concentrating1dMesher``, ``LocalVolRNDCalculator``,
``FdmHestonGreensFct`` and the six multi-dimensional schemes were "a
documented Phase 12 deliverable". Every one of them had shipped and is
cross-validated; the carve-out was stale.

Divergences from C++, all deliberate:

* ``HestonSLVFokkerPlanckFdmParams`` is a frozen dataclass rather than a
  POD struct, and spells C++'s ``predictionCorretionSteps`` correctly.
  Its default is 0 — which makes the C++ inner loop body never run — so
  callers who want a calibration must pass a positive value, as the C++
  test-suite does.
* C++ ``FdmScheme`` / ``FdmSchemeWrapper<T>`` exist only to give the six
  schemes a common virtual base; Python's duck typing needs neither, so
  ``_fdm_scheme_factory`` returns the scheme itself.
* Python's scheme ``step`` RETURNS the stepped array instead of mutating in
  place (the convention throughout ``methods.finitedifferences.schemes``),
  so ``pn = scheme.step(pn, t)``.
* ``FixedLocalVolSurface.set_column`` replaces C++'s
  ``setInterpolation<Linear>()``: C++ rebuilds every slice's interpolation
  after writing one column, Python rebuilds the one slice that changed.
  Same end state, and it is the only slice whose data moved.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.integrals.discrete_integrals import DiscreteSimpsonIntegral
from pquantlib.math.interpolations.bilinear import Bilinear
from pquantlib.methods.finitedifferences.meshers.concentrating_1d_mesher import (
    Concentrating1dMesher,
)
from pquantlib.methods.finitedifferences.meshers.fdm_1d_mesher import Fdm1dMesher
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.meshers.predefined_1d_mesher import (
    Predefined1dMesher,
)
from pquantlib.methods.finitedifferences.operators.fdm_heston_fwd_op import (
    FdmHestonFwdOp,
)
from pquantlib.methods.finitedifferences.operators.fdm_linear_op_composite import (
    FdmLinearOpComposite,
)
from pquantlib.methods.finitedifferences.operators.fdm_square_root_fwd_op import (
    TransformationType,
)
from pquantlib.methods.finitedifferences.schemes.craig_sneyd_scheme import (
    CraigSneydScheme,
)
from pquantlib.methods.finitedifferences.schemes.douglas_scheme import DouglasScheme
from pquantlib.methods.finitedifferences.schemes.explicit_euler_scheme import (
    ExplicitEulerScheme,
)
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import (
    FdmSchemeDesc,
    FdmSchemeType,
)
from pquantlib.methods.finitedifferences.schemes.hundsdorfer_scheme import (
    HundsdorferScheme,
)
from pquantlib.methods.finitedifferences.schemes.implicit_euler_scheme import (
    ImplicitEulerScheme,
)
from pquantlib.methods.finitedifferences.schemes.modified_craig_sneyd_scheme import (
    ModifiedCraigSneydScheme,
)
from pquantlib.methods.finitedifferences.utilities.fdm_heston_greens_fct import (
    FdmHestonGreensFct,
    FdmHestonGreensFctAlgorithm,
)
from pquantlib.methods.finitedifferences.utilities.fdm_mesher_integral import (
    FdmMesherIntegral,
)
from pquantlib.methods.finitedifferences.utilities.local_vol_rnd_calculator import (
    LocalVolRNDCalculator,
)
from pquantlib.methods.finitedifferences.utilities.square_root_process_rnd_calculator import (
    SquareRootProcessRNDCalculator,
)
from pquantlib.models.equity.heston_model import HestonModel
from pquantlib.patterns.lazy_object import LazyObject
from pquantlib.processes.heston_process import HestonProcess
from pquantlib.termstructures.volatility.equity_fx.fixed_local_vol_surface import (
    FixedLocalVolSurface,
)
from pquantlib.termstructures.volatility.equity_fx.local_vol_term_structure import (
    LocalVolTermStructure,
)
from pquantlib.time.date import Date
from pquantlib.time.time_grid import TimeGrid


@dataclass(frozen=True, slots=True)
class HestonSLVFokkerPlanckFdmParams:
    """Parameter pack for the Fokker-Planck FDM SLV calibration.

    # C++ parity: ``struct HestonSLVFokkerPlanckFdmParams``
    # (hestonslvfdmmodel.hpp:43-71).

    Field names and defaults mirror the C++ test-suite defaults; the C++
    struct itself has no defaults. Note ``prediction_correction_steps``
    defaults to 0, which makes the calibration inner loop never execute and
    the leverage function stay at its seed value — that is faithful to what
    C++ does with 0, not an omission. The C++ tests pass 2.
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
    #: # C++ parity: ``FdmHestonGreensFct::Algorithm greensAlgorithm``.
    greens_algorithm: FdmHestonGreensFctAlgorithm = (
        FdmHestonGreensFctAlgorithm.Gaussian
    )
    #: # C++ parity: ``FdmSquareRootFwdOp::TransformationType trafoType``.
    trafo_type: TransformationType = TransformationType.Plain
    #: # C++ parity: ``FdmSchemeDesc schemeDesc``.
    scheme_desc: FdmSchemeDesc = field(default_factory=FdmSchemeDesc.modified_craig_sneyd)


@dataclass(frozen=True, slots=True)
class LogEntry:
    """One snapshot of the Fokker-Planck density during calibration.

    # C++ parity: ``struct HestonSLVFDMModel::LogEntry``
    # (hestonslvfdmmodel.hpp:87-91) — ``{const Time t; const shared_ptr<Array>
    # prob; const shared_ptr<FdmMesherComposite> mesher;}``.

    C++ holds ``prob`` and ``mesher`` by shared pointer because the mesher is
    shared with the operator and the array is copied out of the working
    density (``ext::make_shared<Array>(p)``, hestonslvfdmmodel.cpp:422 and
    :523 — a copy, so a later step cannot mutate a recorded entry). The
    Python entry copies the array for the same reason and holds the mesher by
    reference, which is what a Python shared_ptr is.

    Entries are recorded only when the model was constructed with
    ``logging=True``; C++ guards both push_backs on ``logging_``.
    """

    t: float
    prob: Array
    mesher: FdmMesherComposite


#: C++ hardcodes these two bounds when clamping the leverage
#: (hestonslvfdmmodel.cpp:484).
_L_MIN: Final[float] = 0.001
_L_MAX: Final[float] = 50.0


def _fdm_scheme_factory(
    desc: FdmSchemeDesc, op: FdmLinearOpComposite
) -> HundsdorferScheme | DouglasScheme | CraigSneydScheme | ModifiedCraigSneydScheme | ImplicitEulerScheme | ExplicitEulerScheme:
    """# C++ parity: ``fdmSchemeFactory`` (hestonslvfdmmodel.cpp:215-248).

    The C++ ``FdmScheme`` / ``FdmSchemeWrapper<T>`` pair exists only to give
    six unrelated scheme types a common virtual base so a ``shared_ptr`` can
    hold any of them. Python needs neither, so the scheme is returned as is.
    """
    if desc.type == FdmSchemeType.HundsdorferType:
        return HundsdorferScheme(desc.theta, desc.mu, op)
    if desc.type == FdmSchemeType.DouglasType:
        return DouglasScheme(desc.theta, op)
    if desc.type == FdmSchemeType.CraigSneydType:
        return CraigSneydScheme(desc.theta, desc.mu, op)
    if desc.type == FdmSchemeType.ModifiedCraigSneydType:
        return ModifiedCraigSneydScheme(desc.theta, desc.mu, op)
    if desc.type == FdmSchemeType.ImplicitEulerType:
        return ImplicitEulerScheme(op)
    if desc.type == FdmSchemeType.ExplicitEulerType:
        return ExplicitEulerScheme(op)
    qassert.fail("Unknown scheme type")
    raise AssertionError  # unreachable


def _variance_mesher(
    rnd: SquareRootProcessRNDCalculator,
    t0: float,
    t1: float,
    v_grid: int,
    v0: float,
    params: HestonSLVFokkerPlanckFdmParams,
) -> Fdm1dMesher:
    """# C++ parity: ``varianceMesher`` (hestonslvfdmmodel.cpp:53-121).

    The bounds are the widest CDF inversions over eleven equally spaced times
    in ``[t0, t1]``, floored at ``params.v_min``. All three transformation
    types build the same three critical points; only ``Log`` takes logs first.
    """
    lower_bound = math.inf
    upper_bound = -math.inf
    for i in range(11):
        t = t0 + i / 10.0 * (t1 - t0)
        lower_bound = min(lower_bound, rnd.invcdf(params.v_lower_eps, t))
        upper_bound = max(upper_bound, rnd.invcdf(1.0 - params.v_upper_eps, t))

    lower_bound = max(lower_bound, params.v_min)

    if params.trafo_type == TransformationType.Log:
        lower_bound = math.log(lower_bound)
        upper_bound = math.log(upper_bound)
        v0_center = math.log(v0)
    elif params.trafo_type in (TransformationType.Plain, TransformationType.Power):
        v0_center = v0
    else:
        qassert.fail("transformation type is not implemented")
        raise AssertionError  # unreachable

    c_points = [
        (lower_bound, params.v_lower_bound_density, False),
        (v0_center, params.v0_density, True),
        (upper_bound, params.v_upper_bound_density, False),
    ]
    return Concentrating1dMesher.from_critical_points(
        lower_bound, upper_bound, v_grid, c_points, 1e-8
    )


def _integrate_pdf(
    p: Array,
    mesher: FdmMesherComposite,
    trafo_type: TransformationType,
    alpha: float,
) -> float:
    """# C++ parity: ``integratePDF`` (hestonslvfdmmodel.cpp:123-144)."""
    integral = FdmMesherIntegral(mesher, DiscreteSimpsonIntegral())
    if trafo_type != TransformationType.Power:
        return integral.integrate(p)
    nu = mesher.locations(1)
    return integral.integrate(p * np.power(nu, alpha - 1.0))


def _rescale_pdf(
    p: Array,
    mesher: FdmMesherComposite,
    trafo_type: TransformationType,
    alpha: float,
) -> Array:
    """# C++ parity: ``rescalePDF`` (hestonslvfdmmodel.cpp:147-153)."""
    return p / _integrate_pdf(p, mesher, trafo_type, alpha)


def _reshape_pdf(
    p: Array, old_mesher: FdmMesherComposite, new_mesher: FdmMesherComposite
) -> Array:
    """# C++ parity: ``reshapePDF<Bilinear>`` (hestonslvfdmmodel.cpp:156-193).

    Bilinear re-interpolation of the density from one (x, v) grid onto
    another, with ZERO outside the old grid's range rather than
    extrapolation — the density really is negligible there, and
    extrapolating a bilinear surface past its corners produces negative
    probabilities.
    """
    old_layout = old_mesher.layout()
    new_layout = new_mesher.layout()
    qassert.require(
        old_layout.size() == new_layout.size() and old_layout.size() == p.shape[0],
        "inconsistent mesher or vector size given",
    )
    dim = old_layout.dim()
    # C++ Matrix m(dim[1], dim[0]) filled by a flat row-wise copy of p, i.e.
    # the layout's flat index runs with direction 0 fastest.
    m = np.asarray(p, dtype=np.float64).reshape(dim[1], dim[0])

    xs = old_mesher.get_fdm_1d_meshers()[0].locations()
    ys = old_mesher.get_fdm_1d_meshers()[1].locations()
    interpol = Bilinear().interpolate(np.asarray(xs), np.asarray(ys), m)

    p_new = np.zeros(p.shape[0], dtype=np.float64)
    x_min, x_max = interpol.x_min, interpol.x_max
    y_min, y_max = interpol.y_min, interpol.y_max
    for it in new_layout.iter():
        x = new_mesher.location(it, 0)
        v = new_mesher.location(it, 1)
        if x > x_max or x < x_min or v > y_max or v < y_min:
            p_new[it.index] = 0.0
        else:
            p_new[it.index] = interpol(x, v)
    return p_new


class HestonSLVFDMModel(LazyObject):
    """Heston stochastic-local-vol model calibrated by a Fokker-Planck FDM.

    # C++ parity: ``class HestonSLVFDMModel : public LazyObject``
    # (hestonslvfdmmodel.hpp:73-109).
    """

    #: # C++ parity: the nested ``LogEntry`` struct, exposed under the owner
    #: so call sites can write ``HestonSLVFDMModel.LogEntry`` as C++ does.
    LogEntry = LogEntry

    def __init__(
        self,
        local_vol: LocalVolTermStructure,
        heston_model: HestonModel,
        end_date: Date,
        params: HestonSLVFokkerPlanckFdmParams | None = None,
        logging: bool = False,
        mandatory_dates: Sequence[Date] | None = None,
        mixing_factor: float = 1.0,
    ) -> None:
        """# C++ parity: the constructor at hestonslvfdmmodel.cpp:251-264."""
        super().__init__()
        self._local_vol: LocalVolTermStructure = local_vol
        self._heston_model: HestonModel = heston_model
        self._end_date: Date = end_date
        self._params: HestonSLVFokkerPlanckFdmParams = (
            params if params is not None else HestonSLVFokkerPlanckFdmParams()
        )
        self._mandatory_dates: list[Date] = (
            list(mandatory_dates) if mandatory_dates is not None else []
        )
        self._mixing_factor: float = mixing_factor
        self._logging: bool = logging
        self._leverage_function: FixedLocalVolSurface | None = None
        self._log_entries: list[LogEntry] = []

        local_vol.register_with(self)
        heston_model.register_with(self)

    # --- inspectors -----------------------------------------------------

    def heston_process(self) -> HestonProcess:
        """# C++ parity: ``HestonSLVFDMModel::hestonProcess`` (cpp:266-268)."""
        return self._heston_model.process()

    def local_vol(self) -> LocalVolTermStructure:
        """# C++ parity: ``HestonSLVFDMModel::localVol`` (cpp:270-272)."""
        return self._local_vol

    def leverage_function(self) -> FixedLocalVolSurface:
        """The calibrated leverage function ``L(S, t)``.

        # C++ parity: ``HestonSLVFDMModel::leverageFunction`` (cpp:274-279) —
        # ``calculate(); return leverageFunction_;``.
        """
        self.calculate()
        assert self._leverage_function is not None
        return self._leverage_function

    def log_entries(self) -> list[LogEntry]:
        """Density snapshots recorded during calibration.

        # C++ parity: ``HestonSLVFDMModel::logEntries`` (cpp:531-535). Note
        # C++ calls ``performCalculations()`` DIRECTLY here, not
        # ``calculate()``, so the whole calibration re-runs on every call and
        # the entries are rebuilt from scratch. Reproduced: the list a caller
        # gets back is therefore always freshly computed, and calling this on
        # a model built with ``logging=False`` returns an empty list after
        # doing all the work anyway.
        """
        self._perform_calculations()
        return list(self._log_entries)

    def params(self) -> HestonSLVFokkerPlanckFdmParams:
        """The FDM parameter pack used for calibration."""
        return self._params

    def mixing_factor(self) -> float:
        """The Heston-vol mixing scale."""
        return self._mixing_factor

    def end_date(self) -> Date:
        """The calibration horizon."""
        return self._end_date

    # --- calibration ----------------------------------------------------

    def _perform_calculations(self) -> None:  # noqa: PLR0915 — one C++ function
        """# C++ parity: ``HestonSLVFDMModel::performCalculations`` (cpp:281-529)."""
        self._log_entries.clear()

        params = self._params
        heston_process = self._heston_model.process()
        spot = heston_process.s0()
        r_ts = heston_process.risk_free_rate()
        q_ts = heston_process.dividend_yield()

        v0 = heston_process.v0
        kappa = heston_process.kappa
        theta = heston_process.theta
        sigma = heston_process.sigma
        mixed_sigma = self._mixing_factor * sigma
        alpha = 2.0 * kappa * theta / (mixed_sigma * mixed_sigma)

        x_grid = params.x_grid
        v_grid = params.v_grid

        dc = r_ts.day_counter()
        reference_date = r_ts.reference_date()
        t_end = dc.year_fraction(reference_date, self._end_date)

        qassert.require(
            reference_date < self._end_date,
            "reference date must be smaller than final calibration date",
        )
        qassert.require(
            self._local_vol.max_time() >= t_end,
            "final calibration maturity exceeds local volatility surface",
        )

        # Exponentially decaying time-step scheme (cpp:314-330).
        max_dt = 1.0 / params.t_max_steps_per_year
        min_dt = 1.0 / params.t_min_steps_per_year
        t_idx = 0.0
        times: list[float] = [t_idx]
        while t_idx < t_end:
            decay_factor = math.exp(-params.t_step_number_decay * t_idx)
            dt = max_dt * decay_factor + min_dt * (1.0 - decay_factor)
            t_idx += dt
            times.append(min(t_end, t_idx))
        times.extend(
            dc.year_fraction(reference_date, d) for d in self._mandatory_dates
        )

        time_grid = TimeGrid.with_mandatory(times)

        local_vol_rnd = LocalVolRNDCalculator(
            spot,
            r_ts,
            q_ts,
            self._local_vol,
            x_grid=x_grid,
            x0_density=params.x0_density,
            local_vol_prob_eps=params.local_vol_eps_prob,
            max_iter=params.max_integration_iterations,
            time_grid=time_grid,
        )
        rescale_steps = local_vol_rnd.rescale_time_steps()
        square_root_rnd = SquareRootProcessRNDCalculator(
            v0, kappa, theta, mixed_sigma
        )
        trafo_type = params.trafo_type

        # 1-D meshers, one per time step (cpp:352-376).
        x_mesher: list[Fdm1dMesher] = [local_vol_rnd.mesher(0.0)]
        v_mesher: list[Fdm1dMesher] = [Predefined1dMesher([v0] * v_grid)]
        rescale_idx = 0
        for i in range(1, time_grid.size()):
            x_mesher.append(local_vol_rnd.mesher(time_grid.at(i)))
            if rescale_idx < len(rescale_steps) and i == rescale_steps[rescale_idx]:
                rescale_idx += 1
                t1 = (
                    time_grid.at(rescale_steps[rescale_idx])
                    if rescale_idx < len(rescale_steps)
                    else time_grid.back()
                )
                v_mesher.append(
                    _variance_mesher(
                        square_root_rnd,
                        time_grid.at(rescale_steps[rescale_idx - 1]),
                        t1,
                        v_grid,
                        v0,
                        params,
                    )
                )
            else:
                v_mesher.append(v_mesher[-1])

        mesher = FdmMesherComposite(x_mesher[1], v_mesher[1])

        lv0 = self._local_vol.local_vol_at_time(0.0, spot.value()) / math.sqrt(v0)

        # Leverage matrix: (x_grid rows) x (time_grid.size() columns).
        leverage = np.empty((x_grid, time_grid.size()), dtype=np.float64)
        leverage[:, 0] = lv0
        leverage[:, 1] = lv0

        v_strikes: list[list[float]] = []
        for i in range(time_grid.size()):
            locs = x_mesher[i].locations()
            if float(locs[0]) == float(locs[-1]):
                v_strikes.append([math.exp(float(locs[0]))] * x_grid)
            else:
                v_strikes.append([math.exp(float(x)) for x in locs])

        leverage_fct = FixedLocalVolSurface(
            reference_date=reference_date,
            times=list(time_grid.times),
            strikes=v_strikes,
            local_vol_matrix=leverage,
            day_counter=dc,
        )

        heston_fwd_op: FdmLinearOpComposite = FdmHestonFwdOp(
            mesher, heston_process, trafo_type, leverage_fct, self._mixing_factor
        )

        p = FdmHestonGreensFct(mesher, heston_process, trafo_type, lv0).get(
            time_grid.at(1), params.greens_algorithm
        )

        if self._logging:
            self._log_entries.append(
                LogEntry(time_grid.at(1), np.array(p, copy=True), mesher)
            )

        simpson = DiscreteSimpsonIntegral()

        # C++ iterates `for (Size i=2; i < times.size(); ++i)` — the RAW times
        # vector — while indexing timeGrid->at(i) and xMesher[i]
        # (hestonslvfdmmodel.cpp:426). Those two counts differ whenever
        # TimeGrid dedupes or sorts, which mandatoryDates can cause; C++ then
        # runs off the end. The bound is transcribed rather than "fixed" so
        # the two ports agree wherever C++ is well defined, and the mismatch
        # is asserted instead of silently papered over.
        qassert.require(
            len(times) == time_grid.size(),
            "TimeGrid changed the time count; C++ would index out of bounds "
            f"({len(times)} raw times vs {time_grid.size()} grid points)",
        )
        for i in range(2, len(times)):
            t = time_grid.at(i)
            dt = t - time_grid.at(i - 1)

            if (
                mesher.get_fdm_1d_meshers()[0] is not x_mesher[i]
                or mesher.get_fdm_1d_meshers()[1] is not v_mesher[i]
            ):
                new_mesher = FdmMesherComposite(x_mesher[i], v_mesher[i])
                p = _reshape_pdf(p, mesher, new_mesher)
                mesher = new_mesher
                p = _rescale_pdf(p, mesher, trafo_type, alpha)
                heston_fwd_op = FdmHestonFwdOp(
                    mesher,
                    heston_process,
                    trafo_type,
                    leverage_fct,
                    self._mixing_factor,
                )

            pn = p
            x = np.exp(np.asarray(mesher.get_fdm_1d_meshers()[0].locations()))
            v = np.asarray(mesher.get_fdm_1d_meshers()[1].locations())

            # Predictor-corrector steps (cpp:454-517). With
            # prediction_correction_steps == 0 this body never runs, no
            # leverage column is written for step i, and no forward step is
            # taken — exactly what C++ does with 0.
            for _r in range(params.prediction_correction_steps):
                # Rannacher smoothing: implicit Euler for the first few steps
                # regardless of the requested scheme.
                fdm_scheme_desc = (
                    FdmSchemeDesc.implicit_euler()
                    if i < params.n_rannacher_time_steps + 2
                    else params.scheme_desc
                )
                fdm_scheme = _fdm_scheme_factory(fdm_scheme_desc, heston_fwd_op)

                # C++ loops `for (Size j=0; j < x.size(); ++j)` and writes
                # (*L)[j][i]; x.size() equals xGrid by construction. Asserted
                # rather than assumed, because a partially filled np.empty is
                # exactly the uninitialised-read shape this port keeps finding
                # in the C++ it is porting.
                qassert.require(
                    x.shape[0] == x_grid,
                    f"x mesher has {x.shape[0]} nodes, expected x_grid={x_grid}",
                )
                col = np.empty(x_grid, dtype=np.float64)
                for j in range(x.shape[0]):
                    p_slice = pn[j::x_grid][:v_grid]
                    if trafo_type == TransformationType.Power:
                        p_int = simpson(v, np.power(v, alpha - 1.0) * p_slice)
                        vp_int = simpson(v, np.power(v, alpha) * p_slice)
                    elif trafo_type == TransformationType.Log:
                        p_int = simpson(v, p_slice)
                        vp_int = simpson(v, np.exp(v) * p_slice)
                    else:
                        p_int = simpson(v, p_slice)
                        vp_int = simpson(v, v * p_slice)

                    # C++ writes `const Real scale = pInt/vpInt;` and divides in
                    # IEEE-754, so a zero denominator gives NaN (0/0, at nodes
                    # where the density has decayed to nothing on the whole v
                    # slice) or an infinity. Both are LOAD-BEARING: the very next
                    # line tests `scale >= 0.0`, which NaN fails, and the leverage
                    # falls back to 1.0. Python raises ZeroDivisionError instead,
                    # so the IEEE result is reproduced explicitly.
                    if vp_int == 0.0:
                        scale = (
                            math.nan
                            if p_int == 0.0
                            else math.copysign(1.0, p_int)
                            * math.copysign(1.0, vp_int)
                            * math.inf
                        )
                    else:
                        scale = p_int / vp_int
                    local_vol = self._local_vol.local_vol_at_time(t, float(x[j]))
                    # C++ names this `l`; renamed to avoid the ambiguous-name lint.
                    lev = local_vol * math.sqrt(scale) if scale >= 0.0 else 1.0
                    col[j] = min(_L_MAX, max(_L_MIN, lev))

                leverage[:, i] = col
                # C++ calls setInterpolation<Linear>() inside the j loop and
                # again after it; only the completed column matters, so the
                # surface is refreshed once, here.
                leverage_fct.set_column(i, v_strikes[i], leverage[:, i])

                s_lower_bound = max(
                    float(x[0]),
                    math.exp(local_vol_rnd.invcdf(params.leverage_fct_prop_eps, t)),
                )
                s_upper_bound = min(
                    float(x[-1]),
                    math.exp(
                        local_vol_rnd.invcdf(1.0 - params.leverage_fct_prop_eps, t)
                    ),
                )
                lower_l = leverage_fct.local_vol_at_time(t, s_lower_bound)
                upper_l = leverage_fct.local_vol_at_time(t, s_upper_bound)

                for j in range(x.shape[0]):
                    if x[j] < s_lower_bound:
                        leverage[j, i] = lower_l
                    elif x[j] > s_upper_bound:
                        leverage[j, i] = upper_l
                leverage_fct.set_column(i, v_strikes[i], leverage[:, i])

                pn = p
                fdm_scheme.set_step(dt)
                # Python schemes RETURN the stepped array; C++ mutates in place.
                pn = fdm_scheme.step(pn, t)

            p = pn
            p = _rescale_pdf(p, mesher, trafo_type, alpha)

            if self._logging:
                self._log_entries.append(
                    LogEntry(t, np.array(p, copy=True), mesher)
                )

        self._leverage_function = leverage_fct


__all__ = [
    "HestonSLVFDMModel",
    "HestonSLVFokkerPlanckFdmParams",
    "LogEntry",
]

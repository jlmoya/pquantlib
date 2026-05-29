"""GSR — Gaussian one-factor short-rate model (forward-measure formulation).

# C++ parity: ql/models/shortrate/onefactormodels/gsr.{hpp,cpp}
# @ v1.42.1 (099987f0).

Subclasses ``Gaussian1dModel`` and ``CalibratedModel``. Holds a
piecewise-constant volatility ``sigma(t)`` and a (potentially
piecewise) mean reversion ``kappa(t)``. The C++ formulation is in the
T-forward measure with a closed-form zerobond decomposition based on
the affine-form representation (Hull 2010 ch. 31 / Caspers 2013).

Construction:

- ``term_structure`` — yield curve the model fits to.
- ``volstepdates`` — list of dates where the volatility (and
  optionally reversion) switches. Length N-1 for N pieces.
- ``volatilities`` — N piecewise volatilities.
- ``reversion`` — either a single float (constant reversion) or a list
  of N floats (piecewise reversion on the same step dates).
- ``T`` — forward-measure horizon in years; default 60.0.

Closed-form discount-bond formula (gsr.cpp:186-207):

    P(t, T | y) = D(t, T) * exp(-x * G(t, T) - 0.5 * y(t) * G(t, T)^2)

where ``D(t, T) = curve.discount(T) / curve.discount(t)`` is the
curve-implied discount, ``x = y * sigma_0t + e_0t`` is the unnormalized
state, ``G(t, T)`` and ``y(t)`` come from the GSR process core.

Numeraire (gsr.cpp:209-222):

    N(t, y) = P(t, T_fwd | y)

i.e. the numeraire is the discount bond to the forward-measure
horizon T_fwd (at t=0, this reduces to the curve discount).

# C++ parity divergences:
- C++ has four overloaded ctors:
  (1) constant reversion + ``vector<Real>`` volatilities
  (2) piecewise reversion + ``vector<Real>`` volatilities
  (3) constant reversion + ``vector<Handle<Quote>>`` volatilities
  (4) piecewise reversion + ``vector<Handle<Quote>>`` volatilities
  PQuantLib collapses (1)+(2) into the public ctor (``reversion``
  accepts ``float | list[float]``). The Handle<Quote>-based variants
  (3)+(4) are deferred — they only matter for calibration with
  observable quotes, which is not exercised by the L10-B probe and
  can be added when MarketModels-style observable calibration is
  ported.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.models.model import CalibratedModel
from pquantlib.models.parameter import (
    ConstantParameter,
    NoConstraint,
    PiecewiseConstantParameter,
)
from pquantlib.models.shortrate.gaussian1d_model import Gaussian1dModel
from pquantlib.processes.gsr_process import GsrProcess

if TYPE_CHECKING:
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol
    from pquantlib.termstructures.yield_term_structure import YieldTermStructure
    from pquantlib.time.date import Date


class Gsr(Gaussian1dModel, CalibratedModel):
    """Gaussian short-rate (GSR) 1-factor model — forward-measure formulation.

    # C++ parity: ``class Gsr : public Gaussian1dModel, public
    # CalibratedModel`` in gsr.hpp (v1.42.1).

    Diamond-inheritance note: in C++ both bases are independent
    ``public`` parents. Python's MRO linearizes them as
    ``Gsr -> Gaussian1dModel -> TermStructureConsistentModel ->
    LazyObject -> CalibratedModel -> Model -> Observable``.
    Both bases'  ``__init__`` are called explicitly (mirrors G2's
    pattern in models.shortrate.twofactor.g2) — neither uses
    cooperative ``super().__init__()`` to bridge into the other.

    Attributes:
    - ``volatility()`` — piecewise volatility array.
    - ``reversion()`` — piecewise (or constant) reversion array.
    - ``numeraire_time()`` / ``numeraire_time(T)`` — forward-measure horizon.
    """

    def __init__(
        self,
        term_structure: YieldTermStructure,
        volstepdates: list[Date],
        volatilities: list[float],
        reversion: float | list[float],
        T: float = 60.0,  # noqa: N803 — math symbol
    ) -> None:
        # C++ parity: gsr.cpp:26-90 — four ctors collapsed; we keep just
        # the (float-list reversion + float-list volatilities) signature
        # since that's the only variant used in this project.
        Gaussian1dModel.__init__(self, term_structure=term_structure)
        CalibratedModel.__init__(self, n_arguments=2)

        # C++ parity: gsr.cpp:31-32 — arguments_[0] is reversion,
        # arguments_[1] is sigma. We mirror the convention.
        self._volstepdates: list[Date] = list(volstepdates)
        # Materialize observed volatility / reversion lists. We don't
        # need the Handle<Quote> indirection (carve-out: floating
        # variants deferred), so just store the raw vectors.
        self._volatilities: list[float] = list(volatilities)
        if isinstance(reversion, (int, float)):
            self._reversions: list[float] = [float(reversion)]
        else:
            self._reversions = [float(r) for r in reversion]

        # Will be populated by _update_times().
        self._volsteptimes: list[float] = []
        self._volsteptimes_array: npt.NDArray[np.float64] = np.zeros(
            len(self._volstepdates), dtype=np.float64
        )

        # Build the GsrProcess + Parameters.
        self._initialize(T)

        # Register with the term structure so its discount changes
        # propagate to the model.
        term_structure.register_with(self)

    # --- helpers --------------------------------------------------------

    def _update_times(self) -> None:
        """Refresh the volstep-times array from current evaluation date.

        # C++ parity: gsr.cpp:100-119 — ``Gsr::updateTimes``.
        """
        self._volsteptimes = []
        ts = self.term_structure
        for j, d in enumerate(self._volstepdates):
            t = ts.time_from_reference(d)
            self._volsteptimes.append(t)
            self._volsteptimes_array[j] = t
            if j == 0:
                qassert.require(
                    self._volsteptimes[0] > 0.0,
                    f"volsteptimes must be positive ({self._volsteptimes[0]})",
                )
            else:
                qassert.require(
                    self._volsteptimes[j] > self._volsteptimes[j - 1],
                    f"volsteptimes must be strictly increasing "
                    f"({self._volsteptimes[j - 1]}@{j - 1}, "
                    f"{self._volsteptimes[j]}@{j})",
                )
        if self._state_process is not None:
            # Cast through assert for pyright narrowing.
            assert isinstance(self._state_process, GsrProcess)
            self._state_process.flush_cache()
            self._state_process.set_times(self._volsteptimes_array)

    def _initialize(self, T: float) -> None:  # noqa: N803 — math symbol
        """Build the parameter objects + state process.

        # C++ parity: gsr.cpp:137-184 — ``Gsr::initialize``.
        """
        self._volsteptimes_array = np.zeros(len(self._volstepdates), dtype=np.float64)
        self._update_times()

        qassert.require(
            len(self._volatilities) == len(self._volsteptimes) + 1,
            f"there must be n+1 volatilities ({len(self._volatilities)}) for "
            f"n volatility step times ({len(self._volsteptimes)})",
        )
        qassert.require(
            len(self._reversions) == 1
            or len(self._reversions) == len(self._volsteptimes) + 1,
            f"there must be 1 or n+1 reversions ({len(self._reversions)}) for "
            f"n volatility step times ({len(self._volsteptimes)})",
        )

        # arguments[0] = reversion parameter; arguments[1] = sigma parameter.
        if len(self._reversions) == 1:
            # C++ parity: gsr.cpp:153-154.
            self._arguments[0] = ConstantParameter(self._reversions[0], NoConstraint())
        else:
            # C++ parity: gsr.cpp:155-159.
            rev_param = PiecewiseConstantParameter(list(self._volsteptimes), NoConstraint())
            for i, r in enumerate(self._reversions):
                rev_param.set_param(i, r)
            self._arguments[0] = rev_param

        sigma_param = PiecewiseConstantParameter(list(self._volsteptimes), NoConstraint())
        for i, v in enumerate(self._volatilities):
            sigma_param.set_param(i, v)
        self._arguments[1] = sigma_param

        # Build the state process. C++ parity: gsr.cpp:167-168.
        self._state_process = GsrProcess(
            self._volsteptimes_array,
            self._arguments[1].params,
            self._arguments[0].params,
            T,
        )

    # --- inspectors -----------------------------------------------------

    def reversion(self) -> npt.NDArray[np.float64]:
        """Piecewise (or constant) reversion array.

        # C++ parity: gsr.hpp:65 (inline).
        """
        return self._arguments[0].params

    def volatility(self) -> npt.NDArray[np.float64]:
        """Piecewise volatility array.

        # C++ parity: gsr.hpp:66 (inline).
        """
        return self._arguments[1].params

    def numeraire_time(self) -> float:
        """Forward-measure horizon in years.

        # C++ parity: gsr.hpp:191-194 (inline).
        """
        # The state process is always a GsrProcess (set in _initialize).
        assert isinstance(self._state_process, GsrProcess)
        return self._state_process.get_forward_measure_time()

    def set_numeraire_time(self, T: float) -> None:  # noqa: N803 — math symbol
        """Update forward-measure horizon (notifies observers).

        # C++ parity: gsr.hpp:196-199 (inline).
        """
        assert isinstance(self._state_process, GsrProcess)
        self._state_process.set_forward_measure_time(T)

    # --- update / generate_arguments ------------------------------------

    def update(self) -> None:
        # C++ parity: gsr.cpp:92-98 — flush the GSR-process cache and
        # then run the standard LazyObject invalidation chain.
        if self._state_process is not None:
            assert isinstance(self._state_process, GsrProcess)
            self._state_process.flush_cache()
            self._state_process.notify_observers()
        # LazyObject.update() invalidates the calculated flag.
        super().update()

    def generate_arguments(self) -> None:
        """Refresh the state process from the model arguments after calibration.

        # C++ parity: gsr.hpp:144-149 — ``Gsr::generateArguments``.
        """
        assert isinstance(self._state_process, GsrProcess)
        self._state_process.flush_cache()
        self._state_process.set_vols(self._arguments[1].params)
        self._state_process.set_reversions(self._arguments[0].params)
        self.notify_observers()

    def _perform_calculations(self) -> None:
        # C++ parity: gsr.hpp:153-156 — ``Gsr::performCalculations``.
        Gaussian1dModel._perform_calculations(self)
        self._update_times()

    # --- closed-form zerobond / numeraire -------------------------------

    def zerobond_impl(
        self,
        T: float,  # noqa: N803 — math symbol
        t: float,
        y: float,
        yts: YieldTermStructureProtocol | None,
    ) -> float:
        """Discount-bond price ``P(t, T | y)``.

        # C++ parity: gsr.cpp:186-207.
        """
        self.calculate()

        if t == 0.0:
            ts = yts if yts is not None else self.term_structure
            return ts.discount(T, True)

        assert isinstance(self._state_process, GsrProcess)
        p = self._state_process

        # Unnormalize the state.
        x = y * p.std_deviation_1d(0.0, 0.0, t) + p.expectation_1d(0.0, 0.0, t)
        g_tT = p.G(t, T, x)  # noqa: N806 — math symbol

        if yts is not None:
            d = yts.discount(T, True) / yts.discount(t, True)
        else:
            d = (
                self.term_structure.discount(T, True)
                / self.term_structure.discount(t, True)
            )

        # exp(-x * G(t, T) - 0.5 * y(t) * G(t, T)^2)
        return d * np.exp(-x * g_tT - 0.5 * p.y(t) * g_tT * g_tT)

    def numeraire_impl(
        self,
        t: float,
        y: float,
        yts: YieldTermStructureProtocol | None,
    ) -> float:
        """Numeraire ``N(t, y) = P(t, T_fwd | y)``.

        # C++ parity: gsr.cpp:209-222.

        At t=0 the standardized state ``y=0`` and the numeraire is just
        the curve discount to the forward-measure horizon.
        """
        self.calculate()

        assert isinstance(self._state_process, GsrProcess)
        p = self._state_process
        T_fwd = p.get_forward_measure_time()  # noqa: N806 — math symbol

        if t == 0:
            if yts is None:
                return self.term_structure.discount(T_fwd, True)
            return yts.discount(T_fwd)
        return self.zerobond(T_fwd, t, y, yts)


__all__ = ["Gsr"]

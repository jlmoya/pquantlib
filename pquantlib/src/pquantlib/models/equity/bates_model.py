"""BatesModel — Heston + Merton-jumps calibrated model for equity options.

# C++ parity: ql/models/equity/batesmodel.{hpp,cpp} (v1.42.1).

Subclass of ``HestonModel`` that exposes 3 additional calibration
parameters (nu, delta, lambda) in the parameter vector::

    arguments_ = [
        theta, kappa, sigma, rho, v0,   # inherited from Heston (indices 0..4)
        nu,                               # index 5 — mean log-jump  (NoConstraint)
        delta,                            # index 6 — jump std dev   (Positive)
        lambda,                           # index 7 — jump intensity (Positive)
    ]

``generate_arguments`` rebuilds the underlying ``BatesProcess`` from
the optimizer's current params on each iteration.

L4-C scope: lands the parameter vector + the generate_arguments
hook so a future BatesEngine (deferred to L5) can pick up calibrated
params from ``model.process()``. The L4-C ``AnalyticHestonEngine``
does NOT thread the Bates jump CF into its characteristic function;
calibrating with this engine + a BatesModel would simply reduce to
Heston calibration (jump params would not affect the model price).
The plain Heston engine is still useful here for a pure-Heston
calibration warmstart before swapping in the jump engine.

Divergences from C++:

* Sister jump-model classes (``BatesDetJumpModel``,
  ``BatesDoubleExpModel``, ``BatesDoubleExpDetJumpModel``) are
  out of scope — they extend BatesModel further with stochastic
  jump intensity / double-exponential jump distributions. Listed
  in the L4-C carve-outs.
"""

from __future__ import annotations

from pquantlib.math.optimization.constraint import NoConstraint, PositiveConstraint
from pquantlib.models.equity.heston_model import HestonModel
from pquantlib.models.parameter import ConstantParameter
from pquantlib.processes.bates_process import BatesProcess


class BatesModel(HestonModel):
    """Bates Heston + Merton-jump model.

    # C++ parity: ``class BatesModel : public HestonModel`` in
    # ql/models/equity/batesmodel.hpp:43-53 (v1.42.1).
    """

    # Override the base ``_N_ARGUMENTS`` so CalibratedModel allocates
    # 8 parameter slots instead of 5.
    _N_ARGUMENTS: int = 8

    def __init__(self, process: BatesProcess) -> None:
        # super().__init__ runs HestonModel.__init__ which populates
        # arguments_[0..4] from the BatesProcess (which IS-A HestonProcess).
        super().__init__(process)
        # The HestonModel.__init__ call sets self._process to a freshly
        # constructed HestonProcess (via generate_arguments) — re-wrap
        # as a BatesProcess by re-running BatesModel.generate_arguments
        # below. But first, populate the jump-parameter slots.
        #
        # C++ parity: batesmodel.cpp:27 — arguments_.resize(8) + assign
        # slots 5,6,7. The Python __init__ pre-allocated 8 slots via
        # _N_ARGUMENTS override, so we just assign.

        # C++ parity: batesmodel.cpp:29-30 — nu has NoConstraint
        # (a log-jump mean can be negative).
        self._arguments[5] = ConstantParameter(process.nu, NoConstraint())
        # C++ parity: batesmodel.cpp:31-32 — delta is Positive.
        self._arguments[6] = ConstantParameter(process.delta, PositiveConstraint())
        # C++ parity: batesmodel.cpp:33-34 — lambda is Positive.
        self._arguments[7] = ConstantParameter(process.lambda_, PositiveConstraint())

        # Note: HestonModel.__init__ has already called generate_arguments
        # once which rebuilt a plain HestonProcess (no jump params).
        # Now re-run with the full Bates parameters so self._process is a
        # BatesProcess again.
        # C++ parity: batesmodel.cpp:36 — BatesModel::generateArguments().
        self.generate_arguments()

    # --- jump-parameter accessors ---------------------------------------

    def nu(self) -> float:
        """Mean log-jump size (params[5] evaluated at t=0).

        # C++ parity: ``BatesModel::nu`` in batesmodel.hpp:47.
        """
        return self._arguments[5](0.0)

    def delta(self) -> float:
        """Jump-size standard deviation (params[6] evaluated at t=0).

        # C++ parity: ``BatesModel::delta`` in batesmodel.hpp:48.
        """
        return self._arguments[6](0.0)

    def lambda_(self) -> float:
        """Poisson jump intensity (params[7] evaluated at t=0).

        # C++ parity: ``BatesModel::lambda`` in batesmodel.hpp:49.

        Trailing underscore because ``lambda`` is a Python keyword.
        """
        return self._arguments[7](0.0)

    def process(self) -> BatesProcess:  # type: ignore[override]
        """The underlying ``BatesProcess``.

        # C++ parity: inherited ``HestonModel::process`` but the
        # ``ext::shared_ptr<HestonProcess>`` covariance over the
        # derived type is implicit in C++. In Python we narrow the
        # return type via an override to ``BatesProcess`` — at runtime
        # the same object as ``HestonModel.process`` but the static
        # type matches the C++ caller-side dynamic_cast convention.
        """
        p = self._process
        assert isinstance(p, BatesProcess), (
            "BatesModel.process must be a BatesProcess; "
            "generate_arguments may have produced a plain HestonProcess"
        )
        return p

    # --- generateArguments override -------------------------------------

    def generate_arguments(self) -> None:
        """Rebuild the underlying ``BatesProcess`` from current params.

        # C++ parity: ``BatesModel::generateArguments`` in
        # batesmodel.cpp:39-45.

        Called twice during construction (once by HestonModel.__init__
        before the jump-slots are populated, then again from BatesModel
        __init__ after assignment). The first call's
        ``_arguments[5..7]`` were freshly-allocated default Parameter()
        objects (size 0); attempting to read them via ``__call__``
        would raise. We guard with ``hasattr`` so the first call
        from the base ctor produces a plain HestonProcess; the
        second call (post-slot-assignment) produces a BatesProcess.
        """
        # ``_arguments`` always exists (set by CalibratedModel.__init__).
        # The jump slots are the 6th/7th/8th entries; if they're plain
        # Parameter() defaults (size 0), the impl is None — calling
        # would raise. Detect by checking ``size``.
        if len(self._arguments) < 8 or self._arguments[5].size == 0:
            # First-pass (called from HestonModel.__init__) — defer to
            # the base implementation that builds a plain HestonProcess.
            # The second pass (after BatesModel.__init__ assigns slots)
            # will rebuild as a BatesProcess.
            super().generate_arguments()
            return

        # All jump slots are populated — rebuild as BatesProcess.
        self._process = BatesProcess(
            risk_free_rate=self._process.risk_free_rate(),
            dividend_yield=self._process.dividend_yield(),
            s0=self._process.s0(),
            v0=self.v0(),
            kappa=self.kappa(),
            theta=self.theta(),
            sigma=self.sigma(),
            rho=self.rho(),
            lambda_=self.lambda_(),
            nu=self.nu(),
            delta=self.delta(),
        )


__all__ = ["BatesModel"]

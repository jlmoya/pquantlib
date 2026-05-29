"""ZabrSmileSection — closed-form ZABR smile at one expiry.

# C++ parity: ql/termstructures/volatility/zabrsmilesection.hpp
#             (v1.42.1).

A ZABR-parameterised smile section: given
``(alpha, beta, nu, rho, gamma)`` the volatility at any strike is
computed via :func:`zabr_volatility`. ``gamma = 1`` collapses to a
SABR smile.

Construction modes mirror :class:`SabrSmileSection`: either
``exercise_time`` (time-anchored) OR ``exercise_date`` +
``day_counter`` (date-anchored, optionally with ``reference_date``
for floating mode).

This port supports only the ``ShortMaturityLognormal`` and
``ShortMaturityNormal`` evaluation modes. The FD modes
(``LocalVolatility``, ``FullFd``, ``ProjectedHedge``) require the
ZABR FD engine and are deferred — they raise
``LibraryException`` from the underlying
:func:`zabr_volatility` call.
"""

from __future__ import annotations

import math

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.math.interpolations.zabr_formula import (
    ZabrEvaluation,
    zabr_volatility,
)
from pquantlib.termstructures.volatility.smile_section import SmileSection
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.time.date import Date


class ZabrSmileSection(SmileSection):
    """Closed-form ZABR smile at a single expiry.

    Args:
        forward: ATM forward (must satisfy ``forward + shift > 0``).
        zabr_params: 5-tuple ``(alpha, beta, nu, rho, gamma)``.
        exercise_time / exercise_date / day_counter / reference_date:
            same construction modes as :class:`SmileSection`. When
            ``exercise_date`` is given without an explicit
            ``day_counter`` the C++ default ``Actual365Fixed`` is used
            (matching ``ZabrSmileSection`` date-overload, ``zabr
            smilesection.hpp:54-59``).
        volatility_type: ``ShiftedLognormal`` (default) or ``Normal``.
            Note: the smile's ``volatility_type`` controls how
            downstream pricing helpers interpret the returned vol;
            the actual ZABR evaluation mode is selected via the
            ``evaluation`` argument.
        shift: shifted-lognormal shift; default 0.
        evaluation: ZABR evaluation mode. ``ShortMaturityLognormal``
            (default) returns the Hagan-style closed-form lognormal
            vol; ``ShortMaturityNormal`` returns the Bachelier (normal)
            vol. The FD modes raise ``LibraryException``.
    """

    def __init__(
        self,
        *,
        forward: float,
        zabr_params: tuple[float, float, float, float, float],
        exercise_time: float | None = None,
        exercise_date: Date | None = None,
        day_counter: DayCounter | None = None,
        reference_date: Date | None = None,
        volatility_type: VolatilityType = VolatilityType.ShiftedLognormal,
        shift: float = 0.0,
        evaluation: ZabrEvaluation = ZabrEvaluation.ShortMaturityLognormal,
    ) -> None:
        # Default day-counter for date mode — matches C++ default ctor arg.
        if exercise_date is not None and day_counter is None:
            day_counter = Actual365Fixed()
        super().__init__(
            exercise_date=exercise_date,
            exercise_time=exercise_time,
            day_counter=day_counter,
            reference_date=reference_date,
            volatility_type=volatility_type,
            shift=shift,
        )
        alpha, beta, nu, rho, gamma = zabr_params
        if forward + shift <= 0.0:
            from pquantlib.exceptions import LibraryException  # noqa: PLC0415

            raise LibraryException(
                f"at-the-money forward + shift must be positive: "
                f"{forward} with shift {shift} not allowed"
            )
        self._forward: float = forward
        self._alpha: float = alpha
        self._beta: float = beta
        self._nu: float = nu
        self._rho: float = rho
        self._gamma: float = gamma
        self._evaluation: ZabrEvaluation = evaluation

    # --- inspectors ----------------------------------------------------

    def alpha(self) -> float:
        return self._alpha

    def beta(self) -> float:
        return self._beta

    def nu(self) -> float:
        return self._nu

    def rho(self) -> float:
        return self._rho

    def gamma(self) -> float:
        return self._gamma

    def evaluation(self) -> ZabrEvaluation:
        return self._evaluation

    # --- SmileSection overrides ---------------------------------------

    def min_strike(self) -> float:
        # C++: 0.0 (zabrsmilesection.hpp:61).
        return 0.0

    def max_strike(self) -> float:
        # C++: QL_MAX_REAL.
        return math.inf

    def atm_level(self) -> float:
        # C++: model_->forward() — equals forward_ for short-maturity arms.
        return self._forward

    def _volatility_impl(self, strike: float) -> float:
        # C++ clamps strike to 1e-6 for the lognormal arm
        # (zabrsmilesection.hpp:301 — ``strike = std::max(1E-6, strike)``).
        clamped = max(1.0e-6, strike)
        return zabr_volatility(
            clamped,
            self._forward,
            self._exercise_time,
            self._alpha,
            self._beta,
            self._nu,
            self._rho,
            self._gamma,
            mode=self._evaluation,
        )

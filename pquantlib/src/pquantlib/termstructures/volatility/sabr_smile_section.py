"""SabrSmileSection — closed-form Hagan 2002 smile at one expiry.

# C++ parity: ql/termstructures/volatility/sabrsmilesection.{hpp,cpp} (v1.42.1).

A SABR-parameterised smile section: given (alpha, beta, nu, rho) the
volatility at any strike is computed via :func:`shifted_sabr_volatility`.

The C++ class accepts ``sabrParameters`` as a 4-vector; we keep the
4-tuple signature for clarity. Either ``exercise_time`` OR
``exercise_date`` must be provided (the latter requires a
``day_counter``).
"""

from __future__ import annotations

import math

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.math.interpolations.sabr_formula import (
    shifted_sabr_volatility,
    validate_sabr_parameters,
)
from pquantlib.termstructures.volatility.smile_section import SmileSection
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.time.date import Date


class SabrSmileSection(SmileSection):
    """Closed-form Hagan-2002 SABR smile at a single expiry.

    Args:
        forward: ATM forward (must satisfy ``forward + shift > 0``).
        sabr_params: 4-tuple ``(alpha, beta, nu, rho)``.
        exercise_time / exercise_date / day_counter / reference_date:
            same construction modes as :class:`SmileSection`. When
            ``exercise_date`` is given without an explicit ``day_counter``
            the C++ default ``Actual365Fixed`` is used (matching
            ``SabrSmileSection`` date-overload).
        volatility_type: ``ShiftedLognormal`` (default) or ``Normal``.
        shift: shifted-lognormal shift; default 0.
    """

    def __init__(
        self,
        *,
        forward: float,
        sabr_params: tuple[float, float, float, float],
        exercise_time: float | None = None,
        exercise_date: Date | None = None,
        day_counter: DayCounter | None = None,
        reference_date: Date | None = None,
        volatility_type: VolatilityType = VolatilityType.ShiftedLognormal,
        shift: float = 0.0,
    ) -> None:
        # Default day-counter for date mode mirrors C++ constructor
        # (sabrsmilesection.hpp:43 ``const DayCounter& dc = Actual365Fixed()``).
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
        alpha, beta, nu, rho = sabr_params
        if forward + shift <= 0.0:
            from pquantlib.exceptions import LibraryException  # noqa: PLC0415

            raise LibraryException(
                f"at-the-money forward + shift must be positive: "
                f"{forward} with shift {shift} not allowed"
            )
        validate_sabr_parameters(alpha, beta, nu, rho)
        self._forward: float = forward
        self._alpha: float = alpha
        self._beta: float = beta
        self._nu: float = nu
        self._rho: float = rho

    # --- inspectors ----------------------------------------------------

    def alpha(self) -> float:
        return self._alpha

    def beta(self) -> float:
        return self._beta

    def nu(self) -> float:
        return self._nu

    def rho(self) -> float:
        return self._rho

    # --- SmileSection overrides ---------------------------------------

    def min_strike(self) -> float:
        # C++: -shift_
        return -self._shift

    def max_strike(self) -> float:
        # C++: QL_MAX_REAL
        return math.inf

    def atm_level(self) -> float:
        return self._forward

    def _volatility_impl(self, strike: float) -> float:
        # C++ clamps strike to max(0.00001 - shift, strike) to avoid the
        # singular point at strike + shift = 0.
        clamped = max(0.00001 - self._shift, strike)
        return shifted_sabr_volatility(
            clamped, self._forward, self._exercise_time,
            self._alpha, self._beta, self._nu, self._rho,
            self._shift, self._volatility_type,
        )

    def _variance_impl(self, strike: float) -> float:
        clamped = max(0.00001 - self._shift, strike)
        v = shifted_sabr_volatility(
            clamped, self._forward, self._exercise_time,
            self._alpha, self._beta, self._nu, self._rho,
            self._shift, self._volatility_type,
        )
        return v * v * self._exercise_time

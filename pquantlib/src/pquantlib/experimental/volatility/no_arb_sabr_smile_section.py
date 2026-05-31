"""NoArbSabrSmileSection — Doust no-arbitrage SABR smile at one expiry.

# C++ parity: ql/experimental/volatility/noarbsabrsmilesection.{hpp,cpp} (v1.42.1).

A SmileSection backed by a :class:`NoArbSabrModel`. The smile's
``option_price`` / ``digital_option_price`` / ``density`` come directly
from the no-arb terminal density; ``volatility`` is the Black-implied
vol of the model option price (with a Hagan-2002 fallback), exactly
:func:`no_arb_sabr_volatility`.

Only zero shift is supported (matching the C++ ``init()`` requirement
``shift_ == 0.0``); the ``shift`` argument exists for interface parity
and must be zero.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.experimental.volatility.no_arb_sabr import (
    NoArbSabrModel,
    no_arb_sabr_volatility,
)
from pquantlib.payoffs import OptionType
from pquantlib.termstructures.volatility.smile_section import SmileSection
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.time.date import Date


class NoArbSabrSmileSection(SmileSection):
    """No-arbitrage SABR smile section (Doust 2012).

    Args:
        forward: ATM forward (positive).
        sabr_params: 4-tuple ``(alpha, beta, nu, rho)``.
        exercise_time / exercise_date / day_counter / reference_date:
            same construction modes as :class:`SmileSection`. The date
            overload defaults to ``Actual365Fixed`` (matching C++).
        shift: must be zero (other shifts unimplemented in C++).
        volatility_type: ``ShiftedLognormal`` (default) or ``Normal`` —
            forwarded to the inherited interface; the model itself is
            lognormal.
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
        shift: float = 0.0,
        volatility_type: VolatilityType = VolatilityType.ShiftedLognormal,
    ) -> None:
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
        # C++ init(): 4 params, forward > 0, shift == 0, build model.
        qassert.require(forward > 0.0, f"forward ({forward}) must be positive")
        qassert.require(
            shift == 0.0,
            f"shift ({shift}) must be zero, other shifts are not implemented yet",
        )
        alpha, beta, nu, rho = sabr_params
        self._forward: float = forward
        self._alpha: float = alpha
        self._beta: float = beta
        self._nu: float = nu
        self._rho: float = rho
        self._model: NoArbSabrModel = NoArbSabrModel(
            self.exercise_time(), forward, alpha, beta, nu, rho
        )

    # --- inspectors -----------------------------------------------------

    def model(self) -> NoArbSabrModel:
        return self._model

    def alpha(self) -> float:
        return self._alpha

    def beta(self) -> float:
        return self._beta

    def nu(self) -> float:
        return self._nu

    def rho(self) -> float:
        return self._rho

    # --- SmileSection overrides ----------------------------------------

    def min_strike(self) -> float:
        # C++: 0.0
        return 0.0

    def max_strike(self) -> float:
        # C++: QL_MAX_REAL
        return math.inf

    def atm_level(self) -> float:
        return self._forward

    def option_price(
        self, strike: float, option_type: int = 1, discount: float = 1.0
    ) -> float:
        """Black/Bachelier-free option price from the no-arb model.

        # C++ parity: ``NoArbSabrSmileSection::optionPrice``
        # (noarbsabrsmilesection.cpp:59-64) — overrides the base
        # SmileSection helper to price directly off the model density.
        """
        call = self._model.option_price(strike)
        ot = OptionType(option_type)
        if ot == OptionType.Call:
            return discount * call
        return discount * (call - (self._forward - strike))

    def digital_option_price(
        self,
        strike: float,
        option_type: int = 1,
        discount: float = 1.0,
        gap: float = 1.0e-5,
    ) -> float:
        """Digital price from the no-arb model.

        # C++ parity: ``NoArbSabrSmileSection::digitalOptionPrice``
        # (noarbsabrsmilesection.cpp:66-70).
        """
        call = self._model.digital_option_price(strike)
        ot = OptionType(option_type)
        if ot == OptionType.Call:
            return discount * call
        return discount * (1.0 - call)

    def density(
        self, strike: float, discount: float = 1.0, gap: float = 1.0e-4
    ) -> float:
        """Risk-neutral density from the no-arb model.

        # C++ parity: ``NoArbSabrSmileSection::density``
        # (noarbsabrsmilesection.cpp:72-74).
        """
        return discount * self._model.density(strike)

    def _volatility_impl(self, strike: float) -> float:
        # C++ volatilityImpl: implied vol from the model option price,
        # Hagan-2002 fallback — exactly no_arb_sabr_volatility.
        return no_arb_sabr_volatility(
            strike, self._forward, self.exercise_time(),
            self._alpha, self._beta, self._nu, self._rho,
        )


__all__ = ["NoArbSabrSmileSection"]

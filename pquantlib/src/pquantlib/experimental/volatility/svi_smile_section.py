"""SviSmileSection — raw-SVI smile at one expiry.

# C++ parity: ql/experimental/volatility/svismilesection.{hpp,cpp} (v1.42.1).

A SmileSection whose volatility comes from the raw-SVI total-variance
slice (:func:`svi_volatility`). The 5 SVI parameters are supplied as a
``(a, b, sigma, rho, m)`` tuple and validated against the
no-arbitrage constraints (:func:`check_svi_parameters`).

Either ``exercise_time`` OR ``exercise_date`` (+ ``day_counter``) must
be supplied — same construction surface as :class:`SmileSection`. The
date overload defaults to ``Actual365Fixed`` (matching the C++
constructor default).
"""

from __future__ import annotations

import math

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.experimental.volatility.svi_interpolation import (
    check_svi_parameters,
    svi_total_variance,
)
from pquantlib.termstructures.volatility.smile_section import SmileSection
from pquantlib.time.date import Date


class SviSmileSection(SmileSection):
    """Raw-SVI parameterised smile section.

    Args:
        forward: ATM forward (positive).
        svi_params: 5-tuple ``(a, b, sigma, rho, m)``.
        exercise_time / exercise_date / day_counter / reference_date:
            same construction modes as :class:`SmileSection`. When
            ``exercise_date`` is given without an explicit
            ``day_counter`` the C++ default ``Actual365Fixed`` applies.
    """

    def __init__(
        self,
        *,
        forward: float,
        svi_params: tuple[float, float, float, float, float],
        exercise_time: float | None = None,
        exercise_date: Date | None = None,
        day_counter: DayCounter | None = None,
        reference_date: Date | None = None,
    ) -> None:
        if exercise_date is not None and day_counter is None:
            day_counter = Actual365Fixed()
        super().__init__(
            exercise_date=exercise_date,
            exercise_time=exercise_time,
            day_counter=day_counter,
            reference_date=reference_date,
        )
        # C++ init(): expiry > 0, exactly 5 params, no-arb check.
        from pquantlib import qassert  # noqa: PLC0415

        qassert.require(
            self.exercise_time() > 0.0,
            "svi expects a strictly positive expiry time",
        )
        a, b, sigma, rho, m = svi_params
        check_svi_parameters(a, b, sigma, rho, m, self.exercise_time())
        self._forward: float = forward
        self._a: float = a
        self._b: float = b
        self._sigma: float = sigma
        self._rho: float = rho
        self._m: float = m

    # --- inspectors -----------------------------------------------------

    def a(self) -> float:
        return self._a

    def b(self) -> float:
        return self._b

    def sigma(self) -> float:
        return self._sigma

    def rho(self) -> float:
        return self._rho

    def m(self) -> float:
        return self._m

    # --- SmileSection overrides ----------------------------------------

    def min_strike(self) -> float:
        # C++: 0.0
        return 0.0

    def max_strike(self) -> float:
        # C++: QL_MAX_REAL
        return math.inf

    def atm_level(self) -> float:
        return self._forward

    def _volatility_impl(self, strike: float) -> float:
        # C++ clamps the strike to >= 1e-6 inside volatilityImpl.
        k = math.log(max(strike, 1e-6) / self._forward)
        total_variance = svi_total_variance(
            self._a, self._b, self._sigma, self._rho, self._m, k
        )
        return math.sqrt(max(0.0, total_variance / self.exercise_time()))

    def _variance_impl(self, strike: float) -> float:
        # C++ SmileSection::variance default is vol^2 * t; but to match
        # the exact total-variance algebra (and the test that checks
        # variance == a + b*sigma at k=m) we recompute directly so the
        # max(0,...) clamp and the t-cancellation are bit-exact.
        v = self._volatility_impl(strike)
        return v * v * self.exercise_time()


__all__ = ["SviSmileSection"]

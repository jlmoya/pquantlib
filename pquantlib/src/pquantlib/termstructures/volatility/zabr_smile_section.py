"""ZabrSmileSection — ZABR smile at one expiry, in four evaluation modes.

# C++ parity: ql/termstructures/volatility/zabrsmilesection.hpp (v1.43).

C++ selects the evaluation mode with a template tag parameter
(``ZabrShortMaturityLognormal``, ``ZabrShortMaturityNormal``,
``ZabrLocalVolatility``, ``ZabrFullFd``); Python selects it with the
``evaluation`` argument, a
:class:`~pquantlib.math.interpolations.zabr_formula.ZabrEvaluation`.
The tag types themselves are not modelled as classes — they carry no
state and exist only for overload resolution.

The four modes are genuinely different pipelines, not four formulas:

* ``ShortMaturityLognormal`` — ``volatility(K)`` is the closed-form
  ZABR lognormal expansion (with ``K`` floored at 1e-6);
  ``option_price`` is Black on that vol.
* ``ShortMaturityNormal`` — ``option_price`` is Bachelier on the ZABR
  *normal* vol, but ``volatility(K)`` is the implied **lognormal** vol
  back-solved from that Bachelier price (Call above the forward, Put
  below), returning 0.0 if the inversion fails. It is NOT the normal
  vol itself.
* ``LocalVolatility`` — a refined strike grid is priced in one shot off
  the 1-D Dupire PDE (:meth:`ZabrModel.fd_price_vector`), splined, and
  extended past the last strike by an exponential tail fitted from a
  one-sided finite difference. ``volatility(K)`` then runs the
  ``ShortMaturityNormal`` back-solve on those FD prices.
* ``FullFd`` — same grid, spline and tail, but each grid point is
  priced with its own 2-D ZABR PDE (:meth:`ZabrModel.full_fd_price`).
  Correspondingly expensive: the C++ test-suite lowers ``fd_refinement``
  "to speed up the test".

Construction mirrors :class:`SabrSmileSection`: either ``exercise_time``
(time-anchored) OR ``exercise_date`` + ``day_counter`` (date-anchored,
optionally with ``reference_date`` for floating mode).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.exceptions import LibraryException
from pquantlib.math.interpolations.cubic_interpolation import (
    BoundaryCondition,
    CubicInterpolation,
    DerivativeApprox,
)
from pquantlib.math.interpolations.zabr_formula import ZabrEvaluation
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.black_formula import (
    bachelier_black_formula,
    black_formula_implied_std_dev,
)
from pquantlib.termstructures.volatility.smile_section import SmileSection
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.termstructures.volatility.zabr import ZabrModel
from pquantlib.time.date import Date

# C++ ``defaultMoney`` in zabrsmilesection.hpp — the same 21-entry table
# SmileSectionUtils uses; the comment there reads "this is shared with
# SmileSectionUtils - unify later ?".
_DEFAULT_MONEY: tuple[float, ...] = (
    0.0, 0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
    1.0, 1.25, 1.5, 1.75, 2.0, 5.0, 7.5, 10.0, 15.0, 20.0,
)  # fmt: skip

_FD_MODES: frozenset[ZabrEvaluation] = frozenset(
    {ZabrEvaluation.LocalVolatility, ZabrEvaluation.FullFd}
)

# C++ ``static const Real eps`` in init3 — the gap for the one-sided first
# derivative used to fit the exponential right tail.
_TAIL_EPS: float = 1e-5


class ZabrSmileSection(SmileSection):
    """ZABR smile at a single expiry.

    Args:
        forward: ATM forward.
        zabr_params: 5-tuple ``(alpha, beta, nu, rho, gamma)``.
        exercise_time / exercise_date / day_counter / reference_date:
            same construction modes as :class:`SmileSection`. When
            ``exercise_date`` is given without an explicit
            ``day_counter`` the C++ default ``Actual365Fixed`` is used.
        volatility_type: ``ShiftedLognormal`` (default) or ``Normal``.
            NOTE: C++ ``ZabrSmileSection`` has no such parameter — it
            always inherits ``SmileSection``'s ShiftedLognormal default.
            It is retained here because the XABR swaption cube passes it.
        shift: shifted-lognormal shift; default 0. Also absent in C++.
        evaluation: ZABR evaluation mode; see the module docstring.
        moneyness: strike grid, as multiples of the forward, for the two
            FD modes. Empty/``None`` selects the C++ default table.
            Ignored by the short-maturity modes.
        fd_refinement: number of interior points inserted between
            consecutive moneyness strikes for the FD modes.
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
        moneyness: Sequence[float] | None = None,
        fd_refinement: int = 5,
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
            raise LibraryException(
                f"at-the-money forward + shift must be positive: "
                f"{forward} with shift {shift} not allowed"
            )
        if evaluation == ZabrEvaluation.ProjectedHedge:
            raise LibraryException(
                "ZABR mode ProjectedHedge has no counterpart in C++ QuantLib v1.43."
            )

        self._forward: float = forward
        self._evaluation: ZabrEvaluation = evaluation
        self._fd_refinement: int = fd_refinement
        # Kept because the section-level nu() accessor reports the value the
        # caller passed in, unlike model().nu() — see its docstring.
        self._input_nu: float = nu
        # C++ builds the model from exerciseTime(), i.e. after the base
        # class has resolved date- vs time-anchored construction.
        self._model: ZabrModel = ZabrModel(
            self.exercise_time(), forward, alpha, beta, nu, rho, gamma
        )

        self._strikes: list[float] = []
        self._call_prices: list[float] = []
        self._call_price_fct: CubicInterpolation | None = None
        self._a: float = 0.0
        self._b: float = 0.0

        if evaluation in _FD_MODES:
            self._init_fd_strikes(moneyness)
            self._init_fd_prices()
            self._init_fd_interpolation()

    # --- FD initialisation (C++ init / init2 / init3) ---------------------

    def _init_fd_strikes(self, moneyness: Sequence[float] | None) -> None:
        """Refined strike grid for the FD modes.

        # C++ parity: ``init(moneyness, ZabrLocalVolatility)``.
        """
        tmp = list(moneyness) if moneyness else list(_DEFAULT_MONEY)
        last_f = 0.0
        first_strike = True
        for i in tmp:
            f = i * self._forward
            if f > 0.0:
                if not first_strike:
                    for j in range(1, self._fd_refinement + 1):
                        self._strikes.append(
                            last_f + float(j) * (f - last_f) / (self._fd_refinement + 1)
                        )
                first_strike = False
                last_f = f
                self._strikes.append(f)

    def _init_fd_prices(self) -> None:
        """Price the grid.

        # C++ parity: ``init2(ZabrLocalVolatility)`` / ``init2(ZabrFullFd)``.
        """
        if self._evaluation == ZabrEvaluation.LocalVolatility:
            self._call_prices = self._model.fd_price_vector(self._strikes)
        else:
            self._call_prices = [self._model.full_fd_price(k) for k in self._strikes]

    def _init_fd_interpolation(self) -> None:
        """Spline the grid and fit the exponential right tail.

        # C++ parity: ``init3(ZabrLocalVolatility)``.
        """
        self._strikes.insert(0, 0.0)
        self._call_prices.insert(0, self._forward)

        self._call_price_fct = CubicInterpolation(
            np.asarray(self._strikes, dtype=np.float64),
            np.asarray(self._call_prices, dtype=np.float64),
            DerivativeApprox.Spline,
            True,
            BoundaryCondition.SecondDerivative,
            0.0,
            BoundaryCondition.SecondDerivative,
            0.0,
        )
        self._call_price_fct.enable_extrapolation()

        # On the right side we extrapolate exponentially (a spline does not
        # make sense there); precompute the two parameters.
        k_max = self._strikes[-1]
        c0 = self._call_price_fct(k_max)
        c0p = (self._call_price_fct(k_max - _TAIL_EPS) - c0) / _TAIL_EPS
        self._a = c0p / c0
        self._b = math.log(c0) + self._a * k_max

    # --- inspectors -------------------------------------------------------

    def model(self) -> ZabrModel:
        """# C++ parity: ``ZabrSmileSection::model``."""
        return self._model

    def alpha(self) -> float:
        return self._model.alpha()

    def beta(self) -> float:
        return self._model.beta()

    def nu(self) -> float:
        """The nu this section was CONSTRUCTED with.

        C++ ``ZabrSmileSection`` has no parameter accessors at all — it
        exposes ``model()`` and nothing else — so this is a pquantlib
        convenience. It deliberately reports the constructor argument;
        ``model().nu()`` reports the transformed ``nu * alpha**(1-gamma)``
        that the formulas actually use.
        """
        return self._input_nu

    def rho(self) -> float:
        return self._model.rho()

    def gamma(self) -> float:
        return self._model.gamma()

    def evaluation(self) -> ZabrEvaluation:
        return self._evaluation

    # --- SmileSection overrides ------------------------------------------

    def min_strike(self) -> float:
        # C++: 0.0.
        return 0.0

    def max_strike(self) -> float:
        # C++: QL_MAX_REAL.
        return math.inf

    def atm_level(self) -> float:
        # C++: model_->forward().
        return self._model.forward()

    def option_price(
        self,
        strike: float,
        option_type: int = 1,
        discount: float = 1.0,
    ) -> float:
        """Undiscounted-then-discounted call/put price in the active mode.

        # C++ parity: ``ZabrSmileSection::optionPrice`` and its four
        # tag-dispatched overloads.
        """
        if self._evaluation == ZabrEvaluation.ShortMaturityLognormal:
            return super().option_price(strike, option_type, discount)
        if self._evaluation == ZabrEvaluation.ShortMaturityNormal:
            return bachelier_black_formula(
                OptionType(option_type),
                strike,
                self._forward,
                self._model.normal_volatility(strike) * math.sqrt(self.exercise_time()),
                discount,
            )
        assert self._call_price_fct is not None
        if strike <= self._strikes[-1]:
            call = self._call_price_fct(strike, allow_extrapolation=True)
        else:
            call = math.exp(-self._a * strike + self._b)
        if option_type == OptionType.Call:
            return call * discount
        return (call - (self._forward - strike)) * discount

    def _volatility_impl(self, strike: float) -> float:
        """# C++ parity: ``ZabrSmileSection::volatilityImpl`` (four overloads)."""
        if self._evaluation == ZabrEvaluation.ShortMaturityLognormal:
            # C++ clamps the strike at 1e-6 for the lognormal arm.
            return self._model.lognormal_volatility(max(1.0e-6, strike))
        # ShortMaturityNormal, LocalVolatility and FullFd all report an
        # implied LOGNORMAL vol backed out of the mode's own option price.
        return self._implied_lognormal_volatility(strike)

    def _implied_lognormal_volatility(self, strike: float) -> float:
        """Back-solve a lognormal vol from this mode's option price.

        # C++ parity: ``volatilityImpl(strike, ZabrShortMaturityNormal)``
        # — which swallows every exception and returns 0.0, e.g. when the
        # price has underflowed to zero far out of the money.
        """
        implied_vol = 0.0
        try:
            option_type = (
                OptionType.Call if strike >= self._model.forward() else OptionType.Put
            )
            implied_vol = black_formula_implied_std_dev(
                option_type,
                strike,
                self._model.forward(),
                self.option_price(strike, int(option_type), 1.0),
                1.0,
            ) / math.sqrt(self.exercise_time())
        except Exception:
            return 0.0
        return implied_vol


__all__ = ["ZabrSmileSection"]

"""BlackVolTermStructure view of an Andreasen-Huge interpolation.

# C++ parity: ql/termstructures/volatility/equityfx/andreasenhugevolatilityadapter.hpp +
#             andreasenhugevolatilityadapter.cpp (v1.43).

:class:`AndreasenHugeVolatilityInterpl` produces *prices*; anything expecting a
``BlackVolTermStructure`` wants *variances*. This adapter closes the gap the
only way that is model-free: it asks the interpolation for the out-of-the-money
option price at ``(t, strike)`` — put below the forward, call above, so vega is
largest and the root is best conditioned — and inverts the Black formula.

All the term-structure metadata (calendar, day counter, settlement days,
reference date) is delegated to the interpolation's risk-free curve, exactly as
C++ does. When that curve was built on an explicit reference date its calendar
is empty and its settlement days are unset, so those two accessors *throw* —
that pass-through is the documented behaviour, not a gap in this class.

**Missing prerequisite, implemented locally.** The inversion C++ uses is
``blackFormulaImpliedStdDevLiRS`` (ql/pricingengines/blackformula.hpp:208, the
Li-2008 / Rational-Search fixed-point iteration) seeded by
``blackFormulaImpliedStdDevApproximationRS`` (hpp:155). Neither is present in
``pquantlib.pricingengines.black_formula``, whose module docstring lists the
"Chambers / RS / LiRS approximations" as deferred. They are ported here as
module-level functions so this adapter can reproduce C++ rather than
substituting a different root-finder. They are named exactly as they will be
once they move to ``black_formula.py``, which is where they belong and where
they should go when that module is next touched — the move is then a cut and
paste plus a re-export.
"""

from __future__ import annotations

import math
from typing import Final

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.math.distributions.inverse_cumulative_normal import (
    MaddockInverseCumulativeNormal,
)
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines import _ieee_arithmetic as _ieee
from pquantlib.termstructures.volatility.equity_fx.andreasen_huge_volatility_interpl import (
    AndreasenHugeVolatilityInterpl,
)
from pquantlib.termstructures.volatility.equity_fx.black_vol_term_structure import (
    BlackVarianceTermStructure,
)
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date

# C's <cmath> M_2_PI and M_PI_2, used verbatim by the RS approximation.
_M_2_PI: Final[float] = 2.0 / math.pi
_M_PI_2: Final[float] = math.pi / 2.0

_N: Final[CumulativeNormalDistribution] = CumulativeNormalDistribution()
_INV_N: Final[MaddockInverseCumulativeNormal] = MaddockInverseCumulativeNormal()


def _sign(x: float) -> float:
    """# C++ parity: ``boost::math::sign`` — -1 / 0 / +1."""
    if x > 0.0:
        return 1.0
    if x < 0.0:
        return -1.0
    return 0.0


def _af(x: float) -> float:
    """# C++ parity: the anonymous-namespace ``Af`` (blackformula.cpp:263-266)."""
    return 0.5 * (1.0 + _sign(x) * _ieee.sqrt(1.0 - math.exp(-_M_2_PI * x * x)))


def _check_parameters(strike: float, forward: float, displacement: float) -> None:
    """# C++ parity: ``checkParameters`` (blackformula.cpp)."""
    qassert.require(displacement >= 0.0, f"displacement ({displacement}) must be non-negative")
    qassert.require(
        strike + displacement >= 0.0,
        f"strike + displacement ({strike} + {displacement}) must be non-negative",
    )
    qassert.require(
        forward + displacement > 0.0,
        f"forward + displacement ({forward} + {displacement}) must be positive",
    )


def black_formula_implied_std_dev_approximation_rs(
    option_type: OptionType,
    strike: float,
    forward: float,
    market_value: float,
    discount: float = 1.0,
    displacement: float = 0.0,
) -> float:
    """Rational-search closed-form implied std-dev approximation.

    # C++ parity: ``blackFormulaImpliedStdDevApproximationRS``
    # (ql/pricingengines/blackformula.cpp:269-318). Belongs in
    # ``pquantlib.pricingengines.black_formula``; see the module docstring.

    Fragile far out of the money: the ``C`` term below is a difference of two
    nearly equal squares, so at a near-zero price it rounds negative,
    ``log(beta)`` becomes NaN and every downstream value follows. C++ has the
    same failure mode and it is pinned as such in the probe.
    """
    _check_parameters(strike, forward, displacement)
    qassert.require(market_value >= 0.0, f"blackPrice ({market_value}) must be non-negative")
    qassert.require(discount > 0.0, f"discount ({discount}) must be positive")

    f = forward + displacement
    k = strike + displacement

    ey = f / k
    ey2 = ey * ey
    y = _ieee.log(ey)
    alpha = market_value / (k * discount)
    r = 2 * alpha + ((-ey + 1.0) if option_type == OptionType.Call else (ey - 1.0))
    r2 = r * r

    a = math.exp((1.0 - _M_2_PI) * y)
    a_cap = (a - 1.0 / a) ** 2
    b = math.exp(_M_2_PI * y)
    b_cap = 4.0 * (b + 1 / b) - 2 * k / f * (a + 1.0 / a) * (ey2 + 1 - r2)
    c_cap = (r2 - (ey - 1) ** 2) * ((ey + 1) ** 2 - r2) / ey2

    beta = 2 * c_cap / (b_cap + _ieee.sqrt(b_cap * b_cap + 4.0 * a_cap * c_cap))
    gamma = -_M_PI_2 * _ieee.log(beta)

    if y >= 0.0:
        m0 = k * discount * (
            (ey * _af(_ieee.sqrt(2 * y)) - 0.5)
            if option_type == OptionType.Call
            else (0.5 - ey * _af(-_ieee.sqrt(2 * y)))
        )
        if market_value <= m0:
            return _ieee.sqrt(gamma + y) - _ieee.sqrt(gamma - y)
        return _ieee.sqrt(gamma + y) + _ieee.sqrt(gamma - y)

    m0 = k * discount * (
        (0.5 * ey - _af(-_ieee.sqrt(-2 * y)))
        if option_type == OptionType.Call
        else (_af(_ieee.sqrt(-2 * y)) - 0.5 * ey)
    )
    if market_value <= m0:
        return _ieee.sqrt(gamma - y) - _ieee.sqrt(gamma + y)
    return _ieee.sqrt(gamma + y) + _ieee.sqrt(gamma - y)


def _np(x: float, v: float) -> float:
    """# C++ parity: anonymous-namespace ``Np`` (blackformula.cpp:457-459)."""
    return _N(x / v + 0.5 * v)


def _nm(x: float, v: float) -> float:
    """# C++ parity: anonymous-namespace ``Nm`` (blackformula.cpp:460-462)."""
    return math.exp(-x) * _N(x / v - 0.5 * v)


def _phi(x: float, v: float) -> float:
    """# C++ parity: anonymous-namespace ``phi`` (blackformula.cpp:463-467)."""
    ax = 2 * abs(x)
    v2 = v * v
    return (v2 - ax) / (v2 + ax)


def _big_f(v: float, x: float, cs: float, w: float) -> float:
    """# C++ parity: anonymous-namespace ``F`` (blackformula.cpp:468-470)."""
    return cs + _nm(x, v) + w * _np(x, v)


def _big_g(v: float, x: float, cs: float, w: float) -> float:
    """# C++ parity: anonymous-namespace ``G`` (blackformula.cpp:471-479)."""
    q = _big_f(v, x, cs, w) / (1 + w)
    # C++ uses the boost quantile here, noting that Acklam's inverse plus a
    # Halley step is both less accurate and slower.
    k = _INV_N(q)
    return k + _ieee.sqrt(k * k + 2 * abs(x))


def black_formula_implied_std_dev_li_rs(
    option_type: OptionType,
    strike: float,
    forward: float,
    black_price: float,
    discount: float = 1.0,
    displacement: float = 0.0,
    guess: float | None = None,
    w: float = 1.0,
    accuracy: float = 1e-6,
    max_iterations: int = 100,
) -> float:
    """Li (2008) rational-search implied standard deviation.

    # C++ parity: ``blackFormulaImpliedStdDevLiRS``
    # (ql/pricingengines/blackformula.cpp:483-543). Belongs in
    # ``pquantlib.pricingengines.black_formula``; see the module docstring.

    ``guess=None`` is C++'s ``Null<Real>()`` and selects the RS seed above.
    """
    qassert.require(discount > 0.0, f"discount ({discount}) must be positive")
    qassert.require(black_price >= 0.0, f"option price ({black_price}) must be non-negative")

    strike = strike + displacement
    forward = forward + displacement

    if guess is None:
        # C++ passes `displacement` on unchanged even though strike and
        # forward have already absorbed it, so the shift is applied twice
        # inside the approximation. Reproduced verbatim; with the
        # displacement 0.0 this adapter always uses, the two agree exactly.
        guess = black_formula_implied_std_dev_approximation_rs(
            option_type, strike, forward, black_price, discount, displacement
        )
    else:
        qassert.require(guess >= 0.0, f"stdDev guess ({guess}) must be non-negative")

    x = _ieee.log(forward / strike)
    cs = (
        black_price / (forward * discount)
        if option_type == OptionType.Call
        else black_price / (forward * discount) + 1.0 - strike / forward
    )
    qassert.require(cs >= 0.0, f"normalized call price ({cs}) must be positive")

    if x > 0:
        # in-out duality: reflect onto the other side so the iteration always
        # runs on the out-of-the-money branch.
        cs = forward / strike * cs + 1.0 - forward / strike
        qassert.require(cs >= 0.0, "negative option price from in-out duality")
        x = -x

    n_iter = 0
    vk = guess
    vkp1 = guess
    dv = 0.0
    while True:
        vk = vkp1
        alpha_k = (1 + w) / (1 + _phi(x, vk))
        vkp1 = alpha_k * _big_g(vk, x, cs, w) + (1 - alpha_k) * vk
        dv = abs(vkp1 - vk)
        # C++ do/while: `dv > accuracy && ++nIter < maxIterations`.
        n_iter += 1
        if not (dv > accuracy and n_iter < max_iterations):
            break

    qassert.require(dv <= accuracy, "max iterations exceeded")
    qassert.require(vk >= 0.0, f"stdDev ({vk}) must be non-negative")
    return vk


class AndreasenHugeVolatilityAdapter(BlackVarianceTermStructure):
    """``BlackVarianceTermStructure`` over an Andreasen-Huge interpolation.

    # C++ parity: ``class AndreasenHugeVolatilityAdapter``
    # (andreasenhugevolatilityadapter.hpp:33).

    Args:
        vol_interpl: the calibrated interpolation to read prices from.
        eps: accuracy handed to the LiRS inversion (C++ default 1e-6).
    """

    def __init__(
        self, vol_interpl: AndreasenHugeVolatilityInterpl, eps: float = 1e-6
    ) -> None:
        # C++ default-constructs its base, i.e. the delegated mode: no stored
        # reference date, calendar or day counter — every accessor is
        # overridden below to read the risk-free curve instead.
        super().__init__(business_day_convention=BusinessDayConvention.Following)
        self._eps: float = eps
        self._vol_interpl: AndreasenHugeVolatilityInterpl = vol_interpl

    # --- TermStructure interface, all delegated ----------------------------

    def max_date(self) -> Date:
        """# C++ parity: andreasenhugevolatilityadapter.cpp:48-50."""
        return self._vol_interpl.max_date()

    def min_strike(self) -> float:
        """# C++ parity: cpp:51-53."""
        return self._vol_interpl.min_strike()

    def max_strike(self) -> float:
        """# C++ parity: cpp:54-56."""
        return self._vol_interpl.max_strike()

    def calendar(self) -> Calendar:
        """# C++ parity: cpp:57-59 — the risk-free curve's calendar.

        Throws when that curve has none, exactly as C++ does.
        """
        return self._vol_interpl.risk_free_rate().calendar()

    def day_counter(self) -> DayCounter:
        """# C++ parity: cpp:60-62."""
        return self._vol_interpl.risk_free_rate().day_counter()

    def settlement_days(self) -> int:
        """# C++ parity: cpp:66-68.

        Throws when the risk-free curve was built on an explicit reference
        date, exactly as C++ does.
        """
        return self._vol_interpl.risk_free_rate().settlement_days()

    def reference_date(self) -> Date:
        """# C++ parity: cpp:63-65."""
        return self._vol_interpl.risk_free_rate().reference_date()

    def atm_level(self, t: float) -> float:
        """# C++ parity: cpp:69-71 — the forward."""
        return self._vol_interpl.fwd(t)

    # --- BlackVarianceTermStructure hook -----------------------------------

    def _black_variance_impl(self, t: float, strike: float) -> float:
        """# C++ parity: ``blackVarianceImpl`` (cpp:33-45).

        The out-of-the-money side is chosen per query, then the Black formula
        is inverted with the LiRS iteration (1000 iterations, accuracy
        ``eps``) and the resulting std-dev squared.
        """
        fwd = self._vol_interpl.fwd(t)
        option_type = OptionType.Put if fwd > strike else OptionType.Call
        npv = self._vol_interpl.option_price(t, strike, option_type)
        std_dev = black_formula_implied_std_dev_li_rs(
            option_type,
            strike,
            fwd,
            npv,
            self._vol_interpl.risk_free_rate().discount(t),
            0.0,
            None,
            1.0,
            self._eps,
            1000,
        )
        return std_dev * std_dev


__all__ = [
    "AndreasenHugeVolatilityAdapter",
    "black_formula_implied_std_dev_approximation_rs",
    "black_formula_implied_std_dev_li_rs",
]

"""VannaVolga 3-point smile interpolation.

# C++ parity: ql/experimental/barrieroption/vannavolgainterpolation.hpp
#             (v1.43) — ``VannaVolgaInterpolation`` (line 39),
#             ``VannaVolga`` (line 58), ``detail::VannaVolgaInterpolationImpl``
#             (line 82).

The Vanna-Volga method interpolates an FX volatility smile from exactly
three pillars (25-delta put, ATM, 25-delta call). Given a strike ``k``
it builds a hedge portfolio replicating the smile-implied call premium
and inverts Black to recover the implied vol.

Because the interpolation needs only three strikes and never reuses the
generic ``Interpolation`` x-search/extrapolation machinery, the Python
port implements it as a small self-contained callable rather than
subclassing :class:`Interpolation`. The ``value`` formula mirrors C++
``VannaVolgaInterpolationImpl::value`` verbatim.

:class:`VannaVolga` is the interpolation factory/traits class; it carries
the ``requiredPoints = 3`` traits constant that the C++
``Interpolation::templateImpl`` base uses to reject short input ranges.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.math.distributions.normal_distribution import NormalDistribution
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.black_formula import (
    black_formula,
    black_formula_implied_std_dev,
)


class VannaVolgaInterpolation:
    """Vanna-Volga smile interpolation over three strike pillars.

    Args:
        strikes: the three strikes (sorted ascending: 25P, ATM, 25C).
        vols: the three quoted vols at those strikes.
        spot: spot FX rate.
        d_discount: domestic discount factor to ``T``.
        f_discount: foreign discount factor to ``T``.
        t: time to maturity.
    """

    __slots__ = (
        "_atm_vol",
        "_d_discount",
        "_f_discount",
        "_fwd",
        "_premia_bs",
        "_premia_mkt",
        "_spot",
        "_strikes",
        "_t",
        "_vegas",
        "_vols",
    )

    def __init__(
        self,
        strikes: Sequence[float],
        vols: Sequence[float],
        spot: float,
        d_discount: float,
        f_discount: float,
        t: float,
    ) -> None:
        qassert.require(
            len(strikes) == 3,
            "Vanna Volga Interpolator only interpolates 3 volatilities in strike space",
        )
        self._strikes: tuple[float, float, float] = (strikes[0], strikes[1], strikes[2])
        self._vols: tuple[float, float, float] = (vols[0], vols[1], vols[2])
        self._spot: float = spot
        self._d_discount: float = d_discount
        self._f_discount: float = f_discount
        self._t: float = t

        # atmVol is the second (middle) pillar.
        self._atm_vol: float = self._vols[1]
        self._fwd: float = spot * f_discount / d_discount

        self._premia_bs: list[float] = []
        self._premia_mkt: list[float] = []
        self._vegas: list[float] = []
        sqrt_t = math.sqrt(t)
        for i in range(3):
            self._premia_bs.append(
                black_formula(
                    OptionType.Call, self._strikes[i], self._fwd, self._atm_vol * sqrt_t, d_discount
                )
            )
            self._premia_mkt.append(
                black_formula(
                    OptionType.Call, self._strikes[i], self._fwd, self._vols[i] * sqrt_t, d_discount
                )
            )
            self._vegas.append(self._vega(self._strikes[i]))

    def __call__(self, k: float) -> float:
        return self.value(k)

    def value(self, k: float) -> float:
        """Smile vol at strike ``k``."""
        k0, k1, k2 = self._strikes
        sqrt_t = math.sqrt(self._t)

        x1 = (
            self._vega(k)
            / self._vegas[0]
            * (math.log(k1 / k) * math.log(k2 / k))
            / (math.log(k1 / k0) * math.log(k2 / k0))
        )
        x2 = (
            self._vega(k)
            / self._vegas[1]
            * (math.log(k / k0) * math.log(k2 / k))
            / (math.log(k1 / k0) * math.log(k2 / k1))
        )
        x3 = (
            self._vega(k)
            / self._vegas[2]
            * (math.log(k / k0) * math.log(k / k1))
            / (math.log(k2 / k0) * math.log(k2 / k1))
        )

        c_bs = black_formula(OptionType.Call, k, self._fwd, self._atm_vol * sqrt_t, self._d_discount)
        c = (
            c_bs
            + x1 * (self._premia_mkt[0] - self._premia_bs[0])
            + x2 * (self._premia_mkt[1] - self._premia_bs[1])
            + x3 * (self._premia_mkt[2] - self._premia_bs[2])
        )
        std = black_formula_implied_std_dev(OptionType.Call, k, self._fwd, c, self._d_discount)
        return std / sqrt_t

    def _vega(self, k: float) -> float:
        d1 = (math.log(self._fwd / k) + 0.5 * self._atm_vol**2 * self._t) / (
            self._atm_vol * math.sqrt(self._t)
        )
        norm = NormalDistribution()
        return self._spot * self._d_discount * math.sqrt(self._t) * norm(d1)

    # --- the three C++ Interpolation::Impl overrides that are QL_FAIL stubs ---

    def primitive(self, k: float) -> float:
        """Always raises.

        # C++ parity: ``VannaVolgaInterpolationImpl::primitive``
        # (vannavolgainterpolation.hpp:123-125) is ``QL_FAIL("Vanna Volga
        # primitive not implemented")``.
        """
        del k
        raise LibraryException("Vanna Volga primitive not implemented")

    def derivative(self, k: float) -> float:
        """Always raises.

        # C++ parity: vannavolgainterpolation.hpp:126-128.
        """
        del k
        raise LibraryException("Vanna Volga derivative not implemented")

    def second_derivative(self, k: float) -> float:
        """Always raises.

        # C++ parity: vannavolgainterpolation.hpp:129-131.
        """
        del k
        raise LibraryException("Vanna Volga secondDerivative not implemented")


class VannaVolga:
    """VannaVolga-interpolation factory and traits.

    # C++ parity: ``class VannaVolga``
    # (vannavolgainterpolation.hpp:58-77).

    Binds the market context (spot, the two discount factors, the maturity)
    once and stamps out a :class:`VannaVolgaInterpolation` per strike/vol
    triple. ``required_points`` is the C++
    ``static const Size requiredPoints = 3`` traits constant that
    ``Interpolation::templateImpl`` uses to reject short input ranges — for
    Vanna-Volga the interpolation additionally requires *exactly* three
    points, which :class:`VannaVolgaInterpolation` enforces.

    Unlike :class:`Svi` / :class:`NoArbSabr` this factory carries no
    ``global`` flag: C++ ``VannaVolga`` does not declare one, so the
    interpolated-curve machinery treats it as a local (piecewise)
    interpolation.

    Args:
        spot: spot FX rate.
        d_discount: domestic discount factor to ``t``.
        f_discount: foreign discount factor to ``t``.
        t: time to maturity.
    """

    #: C++ ``static const Size requiredPoints = 3``.
    required_points: Final[int] = 3

    __slots__ = ("_d_discount", "_f_discount", "_spot", "_t")

    def __init__(
        self,
        spot: float,
        d_discount: float,
        f_discount: float,
        t: float,
    ) -> None:
        self._spot: float = spot
        self._d_discount: float = d_discount
        self._f_discount: float = f_discount
        self._t: float = t

    def interpolate(
        self, strikes: Sequence[float], vols: Sequence[float]
    ) -> VannaVolgaInterpolation:
        """Build a :class:`VannaVolgaInterpolation` over one strike triple.

        # C++ parity: ``VannaVolga::interpolate``
        # (vannavolgainterpolation.hpp:66-70).
        """
        return VannaVolgaInterpolation(
            strikes, vols, self._spot, self._d_discount, self._f_discount, self._t
        )


__all__ = ["VannaVolga", "VannaVolgaInterpolation"]

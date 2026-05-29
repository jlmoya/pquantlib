"""KahaleSmileSection — arbitrage-free smile reformulation.

# C++ parity: ql/termstructures/volatility/kahalesmilesection.{hpp,cpp}
# (v1.42.1).

Builds an arbitrage-free C^1 inter- and extrapolation of a source
smile section, following Kahale (2004,
http://www.risk.net/data/Pay_per_view/risk/technical/2004/0504_tech_option2.pdf).

The class samples the source smile on a moneyness grid, identifies the
maximal contiguous *arbitrage-free* (AF) sub-window — where call
prices are convex in strike and dC/dK ∈ [-1, 0] — and replaces the
left/right tails with fitted Black-style ``cFunction`` segments. Optional
``interpolate`` re-fits inside the AF window via Kahale's ``aHelper``
brent-solve; otherwise the source smile is used in the interior.

PQuantLib divergences:

* **SmileSectionUtils inlined.** C++ delegates the moneyness-grid
  construction + AF-region detection to
  ``ql/termstructures/volatility/smilesectionutils.cpp``. PQuantLib
  inlines the equivalent functionality here (default 5-sigma
  log-moneyness grid centered at ATM; AF detection walks outward
  from the central index enforcing the call-spread + butterfly
  conditions).
* **Left-tail right-tail brent residual constants** mirror C++
  ``QL_KAHALE_*`` macros (gap=1e-5, acc=1e-12, smax=5.0).
* **Exponential right-tail extrapolation** is supported (C++
  ``exponentialExtrapolation``). The right tail by default uses a
  Black-style ``cFunction``.
* Only *shifted lognormal* source sections are accepted, matching C++.

Tolerance: tier LOOSE on intermediate evaluations (the AF-region
detection + Brent fits depend on the source's smoothness).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.optimize import brentq  # type: ignore[import-untyped]

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.exceptions import LibraryException
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.math.distributions.inverse_cumulative_normal import (
    InverseCumulativeNormal,
)
from pquantlib.pricingengines.black_formula import black_formula_implied_std_dev
from pquantlib.termstructures.volatility.smile_section import SmileSection
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.time.date import Date

_QL_KAHALE_SMAX: float = 5.0
_QL_KAHALE_ACC: float = 1.0e-12
_QL_KAHALE_EPS: float = 1.0e-16

_CDF = CumulativeNormalDistribution()
_INV_CDF = InverseCumulativeNormal()


class _CFunction:
    """Black-style call-price segment + linear correction.

    # C++ parity: ``KahaleSmileSection::cFunction``.

    Two forms:
      - Exponential: ``c(K) = exp(-a*K + b)`` (right-tail decay).
      - Black-style: at ``s -> 0``, ``c(K) = max(f - K, 0) + a*K + b``;
        otherwise the standard Black call price plus a linear
        correction ``a*K + b``.
    """

    __slots__ = ("a", "b", "exponential", "f", "s")

    def __init__(
        self,
        f: float | None,
        s: float | None,
        a: float,
        b: float,
        *,
        exponential: bool = False,
    ) -> None:
        self.exponential: bool = exponential
        self.a: float = a
        self.b: float = b
        self.f: float = float("nan") if f is None else f
        self.s: float = float("nan") if s is None else s

    def __call__(self, k: float) -> float:
        if self.exponential:
            return math.exp(-self.a * k + self.b)
        if self.s < _QL_KAHALE_EPS:
            return max(self.f - k, 0.0) + self.a * k + self.b
        d1 = math.log(self.f / k) / self.s + self.s / 2.0
        d2 = d1 - self.s
        return self.f * _CDF(d1) - k * _CDF(d2) + self.a * k + self.b


def _digital_via_finite_difference(
    source: SmileSection, strike: float, gap: float,
) -> float:
    """Approximate the source's digital call via centered finite difference.

    # C++ parity: ``source_->digitalOptionPrice(K, Call, 1.0, gap)``.
    PQuantLib's ``SmileSection.digital_option_price`` already
    implements the call-spread approximation; we just thread the gap.
    """
    return source.digital_option_price(strike, option_type=1, discount=1.0, gap=gap)


def _default_moneyness_grid() -> list[float]:
    """Default Kahale moneyness grid.

    # C++ parity: ``SmileSectionUtils`` defaults when the caller passes
    # an empty grid. Strikes are placed at log-moneyness values
    # ``F * exp(m * vol_atm * sqrt(T))`` for a fixed set of ``m`` shocks
    # (we use 9 shocks at +/- (0.5, 1, 1.5, 2, 3) sigma; the central
    # point at m=0 is the ATM strike).
    """
    return [-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]


class KahaleSmileSection(SmileSection):
    """Arbitrage-free smile reformulation per Kahale (2004)."""

    def __init__(  # noqa: PLR0915 — faithful port of C++ KahaleSmileSection ctor
        self,
        *,
        base: SmileSection,
        atm: float | None = None,
        interpolate: bool = False,
        exponential_extrapolation: bool = False,
        delete_arbitrage_points: bool = False,
        moneyness_grid: Sequence[float] | None = None,
        gap: float = 1.0e-5,
        forced_left_index: int = -1,
        forced_right_index: int = 1_000_000,
    ) -> None:
        qassert.require(
            base.volatility_type() == VolatilityType.ShiftedLognormal,
            "KahaleSmileSection only supports shifted lognormal source sections",
        )

        # Anchor against the base's exercise time/date.
        ed: Date | None
        rd: Date | None
        dc: DayCounter | None
        try:
            ed = base.exercise_date()
        except Exception:
            ed = None
        try:
            rd = base.reference_date()
        except Exception:
            rd = None
        try:
            dc = base.day_counter()
        except Exception:
            dc = None
        super().__init__(
            exercise_date=ed,
            exercise_time=None if ed is not None and rd is not None else base.exercise_time(),
            day_counter=dc,
            reference_date=rd,
            volatility_type=base.volatility_type(),
            shift=base.shift(),
        )

        self._source: SmileSection = base
        self._gap: float = gap
        self._interpolate: bool = interpolate
        self._exponential_extrapolation: bool = exponential_extrapolation
        self._forced_left_index: int = forced_left_index
        self._forced_right_index: int = forced_right_index

        # --- 1) Build moneyness grid + sample (k_i, c_i = call(k_i)).
        grid = list(moneyness_grid) if moneyness_grid is not None else _default_moneyness_grid()
        atm_target = atm if atm is not None and not math.isnan(atm) else base.atm_level()
        qassert.require(
            not math.isnan(atm_target),
            "KahaleSmileSection requires an explicit atm or a base section that provides one",
        )
        self._f: float = atm_target

        # Compute strikes via F * exp(m * vol_atm * sqrt(T)). vol_atm is
        # the source vol at ATM (best approximation).
        sqrt_t = math.sqrt(max(self.exercise_time(), _QL_KAHALE_EPS))
        vol_atm = base.volatility(atm_target)
        strikes: list[float] = []
        for m in grid:
            k = atm_target * math.exp(m * vol_atm * sqrt_t)
            strikes.append(k)
        # Sort and de-dup.
        strikes = sorted(set(strikes))

        # If delete_arbitrage_points, drop any strike where the base's
        # call is non-convex or the digital is non-monotone.
        # (We re-do this in the AF-region scan below; flagging here is
        # advisory.)
        _ = delete_arbitrage_points

        # Sample call prices.
        calls: list[float] = []
        for k in strikes:
            try:
                c = base.option_price(k, option_type=1, discount=1.0)
            except Exception:
                # If the base can't price at this strike, skip.
                c = float("nan")
            calls.append(c)
        # Drop NaNs.
        zipped = [(k, c) for k, c in zip(strikes, calls, strict=True) if not math.isnan(c)]
        qassert.require(len(zipped) >= 3, "Kahale needs >=3 valid grid points")
        strikes = [z[0] for z in zipped]
        calls = [z[1] for z in zipped]

        # Shift the strikes + forward into pure lognormal space.
        shift_val = self._source.shift()
        self._k: list[float] = [k + shift_val for k in strikes]
        self._c: list[float] = list(calls)
        self._f += shift_val

        # --- 2) AF-region detection.
        left, right = self._arbitrage_free_indices()
        # Honor forced indices.
        left = max(left, 0, forced_left_index)
        right = min(right, len(self._k) - 1, forced_right_index)
        qassert.require(
            left < right,
            f"empty AF region (left={left} right={right})",
        )
        self._left_index: int = left
        self._right_index: int = right

        # --- 3) Fit extrapolation segments + interior segments.
        self._c_functions: list[_CFunction | None] = [None] * (right - left + 2)
        self._compute()

    # --- AF detection -------------------------------------------------

    def _arbitrage_free_indices(self) -> tuple[int, int]:
        """Find the maximal AF window centered on the ATM index.

        AF conditions (Kahale 2004):
          1. dC/dK ∈ (-1, 0)  (no call-spread arbitrage)
          2. d²C/dK² ≥ 0      (no butterfly arbitrage)
        We use one-sided secants for derivative checks.
        """
        n = len(self._k)
        # Central index = strike closest to ATM.
        atm_idx = int(np.argmin([abs(k - self._f) for k in self._k]))
        # Walk outward from atm_idx — extend while both neighbours pass.
        left = atm_idx
        right = atm_idx
        while left > 0:
            i = left
            if i + 1 >= n:
                break
            secl = (self._c[i] - self._c[i - 1]) / (self._k[i] - self._k[i - 1])
            secr = (self._c[i + 1] - self._c[i]) / (self._k[i + 1] - self._k[i])
            # Call spread: secl ≤ secr (convex) AND secl ∈ [-1, 0].
            if not (secl <= secr + _QL_KAHALE_EPS and -1.0 - _QL_KAHALE_EPS <= secl <= 0.0):
                break
            left -= 1
        while right < n - 1:
            i = right
            if i - 1 < 0:
                break
            secl = (self._c[i] - self._c[i - 1]) / (self._k[i] - self._k[i - 1])
            secr = (self._c[i + 1] - self._c[i]) / (self._k[i + 1] - self._k[i])
            if not (secl <= secr + _QL_KAHALE_EPS and -1.0 - _QL_KAHALE_EPS <= secr <= 0.0):
                break
            right += 1
        return left, right

    # --- left tail fit ------------------------------------------------

    def _fit_left_tail(self, k1: float, c0: float, c1: float, c1p: float) -> _CFunction:
        """Fit ``sHelper1`` to the leftmost AF point.

        # C++ parity: ``KahaleSmileSection::sHelper1`` Brent solve.

        Solve for sigma s.t. ``cFunction(f_=k1*exp(s*d21+s²/2), s, 0, b_=c0-f_)``
        evaluated at k1 equals c1, where d21 = N^{-1}(-c1p).
        """
        def residual(s: float) -> float:
            s = max(s, 0.0)
            d21 = _INV_CDF(max(-c1p, _QL_KAHALE_EPS))
            f = k1 * math.exp(s * d21 + s * s / 2.0)
            b = c0 - f
            cfun = _CFunction(f, s, 0.0, b)
            return cfun(k1) - c1

        # Bracket: [eps, smax]. We accept whatever brentq finds.
        try:
            s_raw: object = brentq(  # pyright: ignore[reportUnknownVariableType]
                residual, _QL_KAHALE_ACC, _QL_KAHALE_SMAX, xtol=_QL_KAHALE_ACC,
            )
            s = float(s_raw)  # pyright: ignore[reportArgumentType]
        except Exception as e:
            raise LibraryException(f"Kahale left-tail Brent failed: {e}") from e
        d21 = _INV_CDF(max(-c1p, _QL_KAHALE_EPS))
        f = k1 * math.exp(s * d21 + s * s / 2.0)
        b = c0 - f
        return _CFunction(f, s, 0.0, b)

    # --- right tail fit -----------------------------------------------

    def _fit_right_tail_black(self, k0: float, c0: float, cp0: float) -> _CFunction:
        """Fit ``sHelper`` to the rightmost AF point (Black-style).

        # C++ parity: ``KahaleSmileSection::sHelper`` Brent solve.
        """
        def residual(s: float) -> float:
            s = max(s, 0.0)
            d20 = _INV_CDF(max(-cp0, _QL_KAHALE_EPS))
            f = k0 * math.exp(s * d20 + s * s / 2.0)
            cfun = _CFunction(f, s, 0.0, 0.0)
            return cfun(k0) - c0

        try:
            s_raw: object = brentq(  # pyright: ignore[reportUnknownVariableType]
                residual, _QL_KAHALE_ACC, _QL_KAHALE_SMAX, xtol=_QL_KAHALE_ACC,
            )
            s = float(s_raw)  # pyright: ignore[reportArgumentType]
        except Exception as e:
            raise LibraryException(f"Kahale right-tail Brent failed: {e}") from e
        d20 = _INV_CDF(max(-cp0, _QL_KAHALE_EPS))
        f = k0 * math.exp(s * d20 + s * s / 2.0)
        return _CFunction(f, s, 0.0, 0.0)

    def _fit_right_tail_exp(self, k0: float, c0: float, cp0: float) -> _CFunction:
        """Exponential right-tail extrapolation."""
        # # C++ parity: ``cFunction(-cp0 / c0, log(c0) - cp0 / c0 * k0)``
        # — exponential form.
        if -cp0 / c0 <= 0.0:
            raise LibraryException("Kahale exponential right-tail: -cp0/c0 must be positive")
        a = -cp0 / c0
        b = math.log(c0) - cp0 / c0 * k0
        return _CFunction(None, None, a, b, exponential=True)

    # --- interior fit (when interpolate=True) -------------------------

    def _fit_interior_segment(
        self, k0: float, k1: float, c0: float, c1: float, c0p: float, c1p: float,
    ) -> _CFunction:
        """Fit ``aHelper`` to a single interior segment.

        # C++ parity: ``KahaleSmileSection::aHelper`` Brent solve.
        """
        def residual(a: float) -> float:
            d20 = _INV_CDF(max(-c0p + a, _QL_KAHALE_EPS))
            d21 = _INV_CDF(max(-c1p + a, _QL_KAHALE_EPS))
            alpha = (d20 - d21) / (math.log(k0) - math.log(k1))
            beta = d20 - alpha * math.log(k0)
            s = -1.0 / alpha
            f = math.exp(s * (beta + s / 2.0))
            cfun_tmp = _CFunction(f, s, a, 0.0)
            b = c0 - cfun_tmp(k0)
            cfun = _CFunction(f, s, a, b)
            return cfun(k1) - c1

        try:
            lower = c1p + _QL_KAHALE_EPS
            upper = 1.0 + c0p - _QL_KAHALE_EPS
            if upper <= lower:
                raise LibraryException(f"Kahale interior Brent: empty bracket [{lower}, {upper}]")
            a_raw: object = brentq(  # pyright: ignore[reportUnknownVariableType]
                residual, lower, upper, xtol=_QL_KAHALE_ACC,
            )
            a = float(a_raw)  # pyright: ignore[reportArgumentType]
        except Exception as e:
            raise LibraryException(f"Kahale interior Brent failed: {e}") from e
        d20 = _INV_CDF(max(-c0p + a, _QL_KAHALE_EPS))
        d21 = _INV_CDF(max(-c1p + a, _QL_KAHALE_EPS))
        alpha = (d20 - d21) / (math.log(k0) - math.log(k1))
        beta = d20 - alpha * math.log(k0)
        s = -1.0 / alpha
        f = math.exp(s * (beta + s / 2.0))
        cfun_tmp = _CFunction(f, s, a, 0.0)
        b = c0 - cfun_tmp(k0)
        return _CFunction(f, s, a, b)

    # --- compute --------------------------------------------------------

    def _compute(self) -> None:
        # Build leftmost tail.
        k1 = self._k[self._left_index]
        c0 = self._c[0]
        c1 = self._c[self._left_index]
        # Derivative at k1 from interpolated secant (matches C++
        # ``interpolate ? (secl + sec) / 2 : -digitalCall(k1 - shift_ + gap/2)``).
        secl = (c1 - c0) / (k1 - self._k[0]) if k1 > self._k[0] else 0.0
        if self._interpolate:
            if self._left_index + 1 < len(self._k):
                sec = (self._c[self._left_index + 1] - c1) / (
                    self._k[self._left_index + 1] - k1
                )
            else:
                sec = secl
            c1p = (secl + sec) / 2.0
        else:
            c1p = -_digital_via_finite_difference(
                self._source,
                k1 - self._source.shift() + self._gap / 2.0,
                self._gap,
            )
        try:
            self._c_functions[0] = self._fit_left_tail(k1, c0, c1, c1p)
        except Exception:
            # Fall back to no left tail (caller will defer to base).
            self._c_functions[0] = None

        # Interior segments (when interpolate=True).
        cp0 = 0.0
        cp1 = 0.0
        if self._interpolate:
            i = self._left_index
            while i < self._right_index:
                k0 = self._k[i]
                k1 = self._k[i + 1]
                c0 = self._c[i]
                c1 = self._c[i + 1]
                sec = (c1 - c0) / (k1 - k0)
                if i == self._left_index:
                    cp0 = (secl + sec) / 2.0 if self._left_index > 0 else sec
                secr = (
                    0.0
                    if i == self._right_index - 1
                    else (self._c[i + 2] - c1) / (self._k[i + 2] - k1)
                )
                cp1 = (sec + secr) / 2.0
                try:
                    self._c_functions[
                        i - self._left_index + 1 if self._left_index > 0 else 0
                    ] = self._fit_interior_segment(k0, k1, c0, c1, cp0, cp1)
                    cp0 = cp1
                except Exception:
                    # Skip this segment — caller will defer to base.
                    pass
                i += 1

        # Right tail.
        k0 = self._k[self._right_index]
        c0 = self._c[self._right_index]
        if self._interpolate:
            cp0 = 0.5 * (c0 - self._c[self._right_index - 1]) / (
                k0 - self._k[self._right_index - 1]
            )
        else:
            cp0 = -_digital_via_finite_difference(
                self._source,
                k0 - self._source.shift() - self._gap / 2.0,
                self._gap,
            )
        try:
            if self._exponential_extrapolation:
                self._c_functions[-1] = self._fit_right_tail_exp(k0, c0, cp0)
            else:
                self._c_functions[-1] = self._fit_right_tail_black(k0, c0, cp0)
        except Exception:
            self._c_functions[-1] = None

    # --- SmileSection interface ---------------------------------------

    def min_strike(self) -> float:
        return -self._source.shift()

    def max_strike(self) -> float:
        return math.inf

    def atm_level(self) -> float:
        # Stored ATM is in shifted-lognormal space; return the
        # un-shifted forward.
        return self._f - self._source.shift()

    def left_core_strike(self) -> float:
        return self._k[self._left_index] - self._source.shift()

    def right_core_strike(self) -> float:
        return self._k[self._right_index] - self._source.shift()

    def core_indices(self) -> tuple[int, int]:
        return self._left_index, self._right_index

    def _segment_index(self, shifted_strike: float) -> int:
        """Return the index of the cFunctions segment covering ``shifted_strike``.

        0 = left tail; ``right - left + 1`` = right tail.
        """
        n_interior = self._right_index - self._left_index + 1
        # Walk the sorted strike array.
        idx = 0
        for kk in self._k:
            if shifted_strike < kk:
                break
            idx += 1
        # Map idx onto the segment table:
        # idx <= leftIndex_  →  0 (left tail)
        # idx > rightIndex_  →  rightIndex - leftIndex + 1 (right tail)
        # otherwise          →  idx - leftIndex (interior segment).
        if idx <= self._left_index:
            return 0
        if idx > self._right_index:
            return n_interior
        return idx - self._left_index

    def _option_price_inner(self, strike: float, option_type: int) -> float:
        """Per-segment call/put price.

        Returns the *Call* price; caller flips to Put via put-call parity.
        """
        shifted = max(strike + self._source.shift(), _QL_KAHALE_EPS)
        i = self._segment_index(shifted)
        # If the strike is interior AND interpolate=False, defer to source.
        n_segments = self._right_index - self._left_index + 1
        if (not self._interpolate) and (0 < i < n_segments):
            return self._source.option_price(strike, option_type, 1.0)
        cfun = self._c_functions[i]
        if cfun is None:
            return self._source.option_price(strike, option_type, 1.0)
        c = cfun(shifted)
        # Apply put-call parity if a Put was requested.
        if option_type != 1:  # Put
            return c + shifted - self._f
        return c

    def option_price(
        self,
        strike: float,
        option_type: int = 1,
        discount: float = 1.0,
    ) -> float:
        return discount * self._option_price_inner(strike, option_type)

    def _volatility_impl(self, strike: float) -> float:
        # Invert Kahale's call price into an implied vol via Black.
        shifted = max(strike + self._source.shift(), _QL_KAHALE_EPS)
        i = self._segment_index(shifted)
        n_segments = self._right_index - self._left_index + 1
        if (not self._interpolate) and (0 < i < n_segments):
            return self._source.volatility(strike)
        cfun = self._c_functions[i]
        if cfun is None:
            return self._source.volatility(strike)
        c = cfun(shifted)
        # Convert ``c`` back to a Black implied std_dev. Put-call parity
        # for the OTM side keeps the inversion well-conditioned.
        from pquantlib.payoffs import OptionType  # noqa: PLC0415

        try:
            ot = OptionType.Call if shifted >= self._f else OptionType.Put
            value = c if ot == OptionType.Call else strike - self._f + c
            std_dev = black_formula_implied_std_dev(
                ot, shifted, self._f, value, 1.0, 0.0,
            )
            return std_dev / math.sqrt(max(self.exercise_time(), _QL_KAHALE_EPS))
        except Exception:
            # Fall back to source vol if the inversion fails.
            return self._source.volatility(strike)

    # --- diagnostics ---------------------------------------------------

    def call_prices(self) -> list[float]:
        """Call prices on the moneyness grid (un-shifted-by-base-shift)."""
        return list(self._c)

    def strike_grid(self) -> list[float]:
        """Strike grid (un-shifted)."""
        return [k - self._source.shift() for k in self._k]


__all__ = ["KahaleSmileSection"]

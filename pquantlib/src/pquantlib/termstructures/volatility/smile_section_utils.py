"""SmileSectionUtils — sanitised strike grid + arbitrage-free window.

# C++ parity: ql/termstructures/volatility/smilesectionutils.{hpp,cpp}
# (v1.43).

Samples a :class:`SmileSection` on a moneyness grid and reports the
maximal contiguous run of strikes over which the resulting call prices
are free of call-spread and butterfly arbitrage.

Moneyness is expressed in **absolute** terms for normal sections
(``k = f + m``) and in **relative** terms for shifted-lognormal ones
(``k = m*(f + shift) - shift``).

Two details are load-bearing and easy to reinvent incorrectly:

* The default grid is a hard-coded table — 21 entries for shifted
  lognormal, a different 27 entries for normal. It is not derived from
  the section's own volatility, expiry or any sigma scaling.
* The call-price vector is not index-aligned with the strike vector in
  the naive way: for shifted lognormal, ``call_prices()[0]`` is the
  analytic ``f + shift`` and the pricing loop starts at strike index 1;
  for normal it starts at index 0.

``delete_arbitrage_points`` erases grid points that fall outside the
arbitrage-free window and re-runs the scan until it is stable, so it
shortens the reported grid rather than only the reported indices.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pquantlib import qassert
from pquantlib.math.closeness import close
from pquantlib.math.constants import QL_EPSILON
from pquantlib.payoffs import OptionType
from pquantlib.termstructures.volatility.smile_section import SmileSection
from pquantlib.termstructures.volatility.volatility_type import VolatilityType

# C++ ``defaultMoney`` (smilesectionutils.cpp) — relative moneyness for
# shifted-lognormal sections.
_DEFAULT_MONEY: tuple[float, ...] = (
    0.0, 0.01, 0.05, 0.10, 0.25, 0.40,
    0.50, 0.60, 0.70, 0.80, 0.90, 1.0,
    1.25, 1.5, 1.75, 2.0, 5.0, 7.5,
    10.0, 15.0, 20.0,
)  # fmt: skip

# C++ ``defaultMoneyNormal`` — absolute moneyness for normal sections.
_DEFAULT_MONEY_NORMAL: tuple[float, ...] = (
    -0.20, -0.15, -0.10, -0.075, -0.05, -0.04, -0.03,
    -0.02, -0.015, -0.01, -0.0075, -0.0050, -0.0025, 0.0,
    0.0025, 0.0050, 0.0075, 0.01, 0.015, 0.02, 0.03,
    0.04, 0.05, 0.075, 0.10, 0.15, 0.20,
)  # fmt: skip


class SmileSectionUtils:
    """Moneyness grid + arbitrage-free window for a smile section.

    # C++ parity: ``class SmileSectionUtils``.

    Args:
        section: the smile to sample.
        moneyness_grid: strictly increasing moneyness values; empty or
            ``None`` selects the built-in default for the section's
            volatility type. Must be non-negative unless the section is
            normal.
        atm: overrides the section's own ATM level; ``None`` means take
            ``section.atm_level()``.
        delete_arbitrage_points: drop grid points outside the
            arbitrage-free window and rescan until stable.
    """

    def __init__(  # noqa: PLR0915 — one-to-one with the C++ constructor body
        self,
        section: SmileSection,
        moneyness_grid: Sequence[float] | None = None,
        atm: float | None = None,
        delete_arbitrage_points: bool = False,
    ) -> None:
        is_normal = section.volatility_type() == VolatilityType.Normal

        if moneyness_grid:
            qassert.require(
                is_normal or moneyness_grid[0] >= 0.0,
                f"moneyness grid should only contain non negative values ({moneyness_grid[0]})",
            )
            for i in range(len(moneyness_grid) - 1):
                qassert.require(
                    moneyness_grid[i] < moneyness_grid[i + 1],
                    "moneyness grid should contain strictly increasing values "
                    f"({moneyness_grid[i]},{moneyness_grid[i + 1]} at indices {i}, {i + 1})",
                )

        if atm is None:
            # C++ checks against Null<Real>; PQuantLib sections signal
            # "no atm level" with NaN (see SmileSection.option_price).
            self._f: float = section.atm_level()
            qassert.require(
                not math.isnan(self._f),
                "atm level must be provided by source section or given in the constructor",
            )
        else:
            self._f = atm

        if moneyness_grid:
            tmp = list(moneyness_grid)
        else:
            tmp = list(_DEFAULT_MONEY_NORMAL if is_normal else _DEFAULT_MONEY)

        shift = section.shift()

        self._m: list[float] = []
        self._k: list[float] = []
        self._c: list[float] = []

        if not is_normal and tmp[0] > QL_EPSILON:
            self._m.append(0.0)
            self._k.append(-shift)

        min_strike_added = False
        max_strike_added = False
        for i in tmp:
            k = (self._f + i) if is_normal else (i * (self._f + shift) - shift)
            if (not is_normal and i <= QL_EPSILON) or (
                section.min_strike() <= k <= section.max_strike()
            ):
                if not min_strike_added or not close(k, section.min_strike()):
                    self._m.append(i)
                    self._k.append(k)
                if close(k, section.max_strike()):
                    max_strike_added = True
            else:
                # If the section provides a limited strike range we put the
                # respective endpoint in our grid in order to not lose too
                # much information.
                if k < section.min_strike() and not min_strike_added:
                    self._m.append(
                        (section.min_strike() - self._f)
                        if is_normal
                        else ((section.min_strike() + shift) / self._f)
                    )
                    self._k.append(section.min_strike())
                    min_strike_added = True
                if k > section.max_strike() and not max_strike_added:
                    self._m.append(
                        (section.max_strike() - self._f)
                        if is_normal
                        else ((section.max_strike() + shift) / self._f)
                    )
                    self._k.append(section.max_strike())
                    max_strike_added = True

        # Only known in closed form for shifted lognormal vols; otherwise we
        # include the lower strike in the loop below.
        if not is_normal:
            self._c.append(self._f + shift)

        for i in range(0 if is_normal else 1, len(self._k)):
            self._c.append(section.option_price(self._k[i], OptionType.Call, 1.0))

        target = (0.0 if is_normal else 1.0) - QL_EPSILON
        central_index = 0
        while central_index < len(self._m) and self._m[central_index] <= target:
            central_index += 1
        qassert.require(
            central_index < len(self._k) - 1 and central_index > 1,
            f"Atm point in moneyness grid ({central_index}) too close to boundary.",
        )

        # Shift the central index to the right if necessary (sometimes even
        # the atm point lies in an arbitrageable area).
        while central_index < len(self._k) - 1 and not self._af(
            central_index, central_index, central_index + 1
        ):
            central_index += 1
        qassert.require(central_index < len(self._k), "central index is at right boundary")

        self._left_index: int = central_index
        self._right_index: int = central_index

        done = False
        while not done:
            is_af = True
            done = True

            while is_af and self._right_index < len(self._k) - 1:
                self._right_index += 1
                is_af = self._af(
                    self._left_index, self._right_index, self._right_index
                ) and self._af(self._left_index, self._right_index - 1, self._right_index)
            if not is_af:
                self._right_index -= 1

            is_af = True
            while is_af and self._left_index > 1:
                self._left_index -= 1
                is_af = self._af(
                    self._left_index, self._left_index, self._right_index
                ) and self._af(self._left_index, self._left_index + 1, self._right_index)
            if not is_af:
                self._left_index += 1

            self._right_index = max(self._right_index, self._left_index)

            if delete_arbitrage_points and self._left_index > 1:
                del self._m[self._left_index - 1]
                del self._k[self._left_index - 1]
                del self._c[self._left_index - 1]
                self._left_index -= 1
                if self._right_index > 0:
                    self._right_index -= 1
                done = False
            if delete_arbitrage_points and self._right_index < len(self._k) - 1:
                del self._m[self._right_index + 1]
                del self._k[self._right_index + 1]
                del self._c[self._right_index + 1]
                if self._right_index > 0:
                    self._right_index -= 1
                done = False

        qassert.require(
            self._right_index > self._left_index,
            "arbitrage free region must at least contain two points "
            f"(only index is {self._left_index})",
        )

    # --- inspectors -------------------------------------------------------

    def arbitragefree_region(self) -> tuple[float, float]:
        """Lowest and highest arbitrage-free strike.

        # C++ parity: ``SmileSectionUtils::arbitragefreeRegion``.
        """
        return self._k[self._left_index], self._k[self._right_index]

    def arbitragefree_indices(self) -> tuple[int, int]:
        """Indices into :meth:`strike_grid` bounding the arbitrage-free region.

        # C++ parity: ``SmileSectionUtils::arbitragefreeIndices``.
        """
        return self._left_index, self._right_index

    def money_grid(self) -> list[float]:
        """# C++ parity: ``SmileSectionUtils::moneyGrid``."""
        return self._m

    def strike_grid(self) -> list[float]:
        """# C++ parity: ``SmileSectionUtils::strikeGrid``."""
        return self._k

    def call_prices(self) -> list[float]:
        """# C++ parity: ``SmileSectionUtils::callPrices``."""
        return self._c

    def atm_level(self) -> float:
        """# C++ parity: ``SmileSectionUtils::atmLevel``."""
        return self._f

    # --- internals --------------------------------------------------------

    def _af(self, i0: int, i: int, i1: int) -> bool:
        """Arbitrage-free test at grid point ``i`` within window ``[i0, i1]``.

        # C++ parity: ``SmileSectionUtils::af``.

        The left secant must satisfy ``-1 <= dC/dK <= 0`` (no call-spread
        arbitrage); if ``i`` is interior, the right secant must be no
        smaller and still non-positive (convexity).
        """
        if i == 0:
            return True
        im = i - 1 if i - 1 >= i0 else 0
        q1 = (self._c[i] - self._c[im]) / (self._k[i] - self._k[im])
        if q1 < -1.0 or q1 > 0.0:
            return False
        if i >= i1:
            return True
        q2 = (self._c[i + 1] - self._c[i]) / (self._k[i + 1] - self._k[i])
        return q1 <= q2 <= 0.0


__all__ = ["SmileSectionUtils"]

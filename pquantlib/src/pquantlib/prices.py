"""Price types, mid-price helpers and the OHLC interval price.

# C++ parity: ql/prices.{hpp,cpp} @ v1.43.

Contents mirror the C++ header one-for-one:

* ``PriceType``          — C++ ``enum PriceType`` (prices.hpp:35).
* ``mid_equivalent``     — C++ ``Real midEquivalent(...)`` (prices.cpp:27).
* ``mid_safe``           — C++ ``Real midSafe(...)`` (prices.cpp:45).
* ``IntervalPrice``      — C++ ``class IntervalPrice`` (prices.hpp:66).
* ``IntervalPriceType``  — C++ nested ``IntervalPrice::Type`` (prices.hpp:70),
  flattened to module scope as this port does for every nested enum
  (cf. ``PositionType`` for ``Position::Type``).

# C++ parity divergence — ``Null<Real>()``:
# C++ uses the ``Null<Real>()`` sentinel (``std::numeric_limits<float>::max()``)
# for "price not available". This port represents it as ``None``, which is the
# convention used throughout (see e.g. pquantlib/cashflows/dividend.py,
# pquantlib/instruments/asset_swap.py). ``mid_equivalent``/``mid_safe``
# therefore accept ``float | None`` and treat ``None`` exactly as C++ treats
# ``Null<Real>()``: as an unavailable input, distinct from a non-positive one.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum

from pquantlib import qassert
from pquantlib.time.date import Date
from pquantlib.time.time_series import TimeSeries


class PriceType(IntEnum):
    """Price flavour.

    # C++ parity: ``enum PriceType`` (ql/prices.hpp:35). Integer values
    # follow C++ declaration order.
    """

    Bid = 0
    Ask = 1
    Last = 2
    Close = 3
    Mid = 4
    """Arithmetic average of bid and ask."""
    MidEquivalent = 5
    """Mid when available, else bid, ask, last or close — see :func:`mid_equivalent`."""
    MidSafe = 6
    """Mid only when BOTH bid and ask are available — see :func:`mid_safe`."""


class IntervalPriceType(IntEnum):
    """Which corner of an OHLC bar.

    # C++ parity: nested ``IntervalPrice::Type`` (ql/prices.hpp:70) —
    # ``{ Open, Close, High, Low }``, integer values in declaration order.
    """

    Open = 0
    Close = 1
    High = 2
    Low = 3


def mid_equivalent(
    bid: float | None,
    ask: float | None,
    last: float | None,
    close: float | None,
) -> float:
    """Mid price if available, else a suitable substitute.

    # C++ parity: ``Real midEquivalent(Real, Real, Real, Real)``
    # (ql/prices.cpp:27-43). The branch structure is reproduced exactly:
    # a valid bid short-circuits the last/close fallbacks even when the ask
    # is missing, so ``mid_equivalent(1, None, 2, 3) == 1`` (NOT 2).

    A price counts as available when it is neither ``None`` (C++
    ``Null<Real>()``) nor non-positive.

    Raises:
        LibraryException: when every input is unavailable.
    """
    if bid is not None and bid > 0.0:
        if ask is not None and ask > 0.0:
            return (bid + ask) / 2.0
        return bid
    if ask is not None and ask > 0.0:
        return ask
    if last is not None and last > 0.0:
        return last
    if close is None or close <= 0.0:
        qassert.fail("all input prices are invalid")
    return close


def mid_safe(bid: float | None, ask: float | None) -> float:
    """Mid price, requiring BOTH bid and ask.

    # C++ parity: ``Real midSafe(Real, Real)`` (ql/prices.cpp:45-53).

    Raises:
        LibraryException: when either side is unavailable.
    """
    if bid is None or bid <= 0.0:
        qassert.fail("invalid bid price")
    if ask is None or ask <= 0.0:
        qassert.fail("invalid ask price")
    return (bid + ask) / 2.0


class IntervalPrice:
    """One OHLC bar.

    # C++ parity: ``class IntervalPrice`` (ql/prices.hpp:66, ql/prices.cpp:83).

    The default constructor leaves every field unset (C++ ``Null<Real>()``,
    here ``None``); the four-argument constructor takes them in the C++ order
    ``(open, close, high, low)`` — note that is NOT the conventional OHLC
    ordering, and matching C++ matters because callers pass positionally.
    """

    __slots__ = ("_close", "_high", "_low", "_open")

    def __init__(
        self,
        open_: float | None = None,
        close: float | None = None,
        high: float | None = None,
        low: float | None = None,
    ) -> None:
        self._open: float | None = open_
        self._close: float | None = close
        self._high: float | None = high
        self._low: float | None = low

    # --- inspectors -------------------------------------------------------

    def open(self) -> float | None:
        """C++ ``IntervalPrice::open()``."""
        return self._open

    def close(self) -> float | None:
        """C++ ``IntervalPrice::close()``."""
        return self._close

    def high(self) -> float | None:
        """C++ ``IntervalPrice::high()``."""
        return self._high

    def low(self) -> float | None:
        """C++ ``IntervalPrice::low()``."""
        return self._low

    def value(self, t: IntervalPriceType) -> float | None:
        """C++ ``IntervalPrice::value(Type)`` (ql/prices.cpp:87-101)."""
        match t:
            case IntervalPriceType.Open:
                return self._open
            case IntervalPriceType.Close:
                return self._close
            case IntervalPriceType.High:
                return self._high
            case IntervalPriceType.Low:
                return self._low
        qassert.fail("Unknown price type")

    # --- modifiers --------------------------------------------------------

    def set_value(self, value: float | None, t: IntervalPriceType) -> None:
        """C++ ``IntervalPrice::setValue(Real, Type)`` (ql/prices.cpp:103-119)."""
        match t:
            case IntervalPriceType.Open:
                self._open = value
            case IntervalPriceType.Close:
                self._close = value
            case IntervalPriceType.High:
                self._high = value
            case IntervalPriceType.Low:
                self._low = value
            case _:
                qassert.fail("Unknown price type")

    def set_values(
        self,
        open_: float | None,
        close: float | None,
        high: float | None,
        low: float | None,
    ) -> None:
        """C++ ``IntervalPrice::setValues(Real, Real, Real, Real)`` (ql/prices.cpp:121)."""
        self._open = open_
        self._close = close
        self._high = high
        self._low = low

    def __repr__(self) -> str:
        return (
            f"IntervalPrice(open={self._open!r}, close={self._close!r}, "
            f"high={self._high!r}, low={self._low!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IntervalPrice):
            return NotImplemented
        return (self._open, self._close, self._high, self._low) == (
            other._open,
            other._close,
            other._high,
            other._low,
        )

    def __hash__(self) -> int:
        return hash((self._open, self._close, self._high, self._low))

    # --- helper functions -------------------------------------------------

    @staticmethod
    def make_series(
        dates: Sequence[Date],
        open_: Sequence[float | None],
        close: Sequence[float | None],
        high: Sequence[float | None],
        low: Sequence[float | None],
    ) -> TimeSeries[IntervalPrice]:
        """C++ ``IntervalPrice::makeSeries(...)`` (ql/prices.cpp:126-152)."""
        n = len(dates)
        qassert.require(
            len(open_) == n and len(close) == n and len(high) == n and len(low) == n,
            f"size mismatch ({n}, {len(open_)}, {len(close)}, {len(high)}, {len(low)})",
        )
        out: TimeSeries[IntervalPrice] = TimeSeries()
        for i, d in enumerate(dates):
            out[d] = IntervalPrice(open_[i], close[i], high[i], low[i])
        return out

    @staticmethod
    def extract_values(
        ts: TimeSeries[IntervalPrice], t: IntervalPriceType
    ) -> list[float | None]:
        """C++ ``IntervalPrice::extractValues(...)`` (ql/prices.cpp:154-162).

        Values come out in date order — C++ iterates a ``std::map``, which is
        key-ordered; ``TimeSeries.values()`` sorts by date for the same reason.
        """
        return [p.value(t) for p in ts.values()]

    @staticmethod
    def extract_component(
        ts: TimeSeries[IntervalPrice], t: IntervalPriceType
    ) -> TimeSeries[float | None]:
        """C++ ``IntervalPrice::extractComponent(...)`` (ql/prices.cpp:164-170)."""
        out: TimeSeries[float | None] = TimeSeries()
        for d, v in zip(ts.dates(), IntervalPrice.extract_values(ts, t), strict=True):
            out[d] = v
        return out


__all__ = [
    "IntervalPrice",
    "IntervalPriceType",
    "PriceType",
    "mid_equivalent",
    "mid_safe",
]

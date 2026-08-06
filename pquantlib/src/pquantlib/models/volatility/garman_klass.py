"""Garman-Klass volatility estimators over OHLC bars.

# C++ parity: ql/models/volatility/garmanklass.hpp @ v1.43
# (header-only — there is no .cpp).

Estimators from Garman and Klass, "On the Estimation of the Security Price
from Historical Data". Volatilities are expressed on an annual basis.

How the C++ template hierarchy is expressed in Python
-----------------------------------------------------
C++ has ``template <class T> class GarmanKlassOpenClose : public T``
(garmanklass.hpp:77-104) — a base class parameterised by its own derived
behaviour. It overrides ``calculate`` to bring in the overnight gap between
the previous close and today's open, and calls ``T::calculatePoint`` for the
intraday part. The three concrete open-close estimators are then
``GarmanKlassOpenClose<GarmanKlassSimpleSigma>``,
``GarmanKlassOpenClose<ParkinsonSigma>`` and
``GarmanKlassOpenClose<GarmanKlassSigma4>``, each with its own weight ``a``.

Python's equivalent of "a base class parameterised by the derived behaviour"
is cooperative multiple inheritance: :class:`GarmanKlassOpenClose` derives
from :class:`GarmanKlassAbstract`, overrides ``calculate``, and leaves
``_calculate_point`` abstract; each concrete class lists it FIRST and the
intraday estimator SECOND::

    class GarmanKlassSigma1(GarmanKlassOpenClose, GarmanKlassSimpleSigma)

so the MRO is ``Sigma1 -> OpenClose -> SimpleSigma -> Abstract``: ``calculate``
resolves to the open-close override and ``_calculate_point`` to the intraday
formula. The resulting subclass relationships are the same as C++'s
(``GarmanKlassSigma1`` IS-A ``GarmanKlassSimpleSigma`` there too), and
``GarmanKlassOpenClose`` stays uninstantiable on its own, exactly like an
unspecialised template.

``fabs``, and where it is missing
---------------------------------
``GarmanKlassAbstract::calculate`` takes ``sqrt(fabs(point)/y)``
(garmanklass.hpp:55) but ``GarmanKlassOpenClose::calculate`` takes
``sqrt(sigma2/y)`` with NO fabs (garmanklass.hpp:100). For an
OHLC-consistent bar (``high >= max(open, close)``, ``low <= min(open, close)``)
that asymmetry is invisible, because it forces ``(u-d)^2 >= c^2`` and every
point formula stays non-negative. It becomes visible for a degenerate bar —
``high == low == open != close`` makes ``GarmanKlassSigma4`` and
``GarmanKlassSigma5`` negative — and there the open-close variants would
produce a NaN. Both behaviours are reproduced as-is.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from pquantlib import qassert
from pquantlib.prices import IntervalPrice
from pquantlib.time.time_series import TimeSeries
from pquantlib.volatility_model import LocalVolatilityEstimator


def _component(value: float | None, name: str) -> float:
    """Unwrap an ``IntervalPrice`` field, rejecting the "unavailable" sentinel.

    # C++ parity divergence. C++ ``IntervalPrice`` stores ``Null<Real>()``
    # (= ``numeric_limits<float>::max()``) for an unset field and the
    # Garman-Klass formulas feed it straight into ``std::log``, silently
    # producing nonsense. pquantlib's ``IntervalPrice`` represents that
    # sentinel as ``None`` (see pquantlib/prices.py), so the port diagnoses
    # the missing field instead.
    """
    qassert.require(value is not None, f"IntervalPrice.{name}() is not available")
    assert value is not None  # narrowing only; qassert.require raised otherwise
    return value


class GarmanKlassAbstract(LocalVolatilityEstimator[IntervalPrice], ABC):
    """Base of the Garman-Klass family: one estimate per bar.

    # C++ parity: ``class GarmanKlassAbstract :
    # public LocalVolatilityEstimator<IntervalPrice>``
    # (ql/models/volatility/garmanklass.hpp:41-60).

    Subclasses supply ``_calculate_point``, the per-bar variance proxy;
    this class annualises it. Unlike the open-close variants, EVERY input
    date appears in the output — no bar is dropped.
    """

    __slots__ = ("_year_fraction",)

    def __init__(self, y: float) -> None:
        # C++ parity: garmanklass.hpp:47-48.
        self._year_fraction: float = y

    def year_fraction(self) -> float:
        """Interval length in years. # C++ parity: protected ``yearFraction_``."""
        return self._year_fraction

    @abstractmethod
    def _calculate_point(self, p: IntervalPrice) -> float:
        """Per-bar variance proxy over the interval.

        # C++ parity: ``virtual Real calculatePoint(const IntervalPrice&) = 0``
        # (garmanklass.hpp:45) — protected, hence the leading underscore.
        """
        ...

    def calculate(self, quote_series: TimeSeries[IntervalPrice]) -> TimeSeries[float]:
        """``sqrt(|point(bar)| / year_fraction)`` for every bar.

        # C++ parity: ``GarmanKlassAbstract::calculate``
        # (garmanklass.hpp:49-59). Note the ``std::fabs`` — see the module
        # docstring for when it matters.
        """
        retval: TimeSeries[float] = TimeSeries()
        for date, price in quote_series.items():
            retval[date] = math.sqrt(abs(self._calculate_point(price)) / self._year_fraction)
        return retval


class GarmanKlassSimpleSigma(GarmanKlassAbstract):
    """Close-to-open squared log return.

    # C++ parity: ``class GarmanKlassSimpleSigma``
    # (garmanklass.hpp:62-72).
    """

    __slots__ = ()

    def _calculate_point(self, p: IntervalPrice) -> float:
        # C++ parity: garmanklass.hpp:68-71.
        c = math.log(_component(p.close(), "close") / _component(p.open(), "open"))
        return c * c


class GarmanKlassOpenClose(GarmanKlassAbstract, ABC):
    """Mixes the overnight gap into another estimator's intraday variance.

    # C++ parity: ``template <class T> class GarmanKlassOpenClose : public T``
    # (garmanklass.hpp:77-104). See the module docstring for how the
    # template parameter becomes a second base class here.

    ``market_open_fraction`` (C++ ``f_``) is the fraction of the day the
    market is open; ``a`` is the weight given to the overnight term. Because
    each estimate needs the PREVIOUS bar's close, the output starts at the
    second date (garmanklass.hpp:89-90).
    """

    __slots__ = ("_a", "_f")

    def __init__(self, y: float, market_open_fraction: float, a: float) -> None:
        # C++ parity: garmanklass.hpp:83-85 — ``T(y), f_(...), a_(...)``.
        super().__init__(y)
        self._f: float = market_open_fraction
        self._a: float = a

    def market_open_fraction(self) -> float:
        """Fraction of the day the market is open. # C++ parity: protected ``f_``."""
        return self._f

    def a(self) -> float:
        """Weight of the overnight term. # C++ parity: protected ``a_``."""
        return self._a

    def calculate(self, quote_series: TimeSeries[IntervalPrice]) -> TimeSeries[float]:
        """Weighted overnight-gap + intraday variance, annualised.

        # C++ parity: ``GarmanKlassOpenClose<T>::calculate``
        # (garmanklass.hpp:86-103). There is deliberately NO ``fabs`` here,
        # unlike ``GarmanKlassAbstract::calculate``.
        """
        retval: TimeSeries[float] = TimeSeries()
        items = quote_series.items()
        for i in range(1, len(items)):
            date, cur = items[i]
            prev = items[i - 1][1]
            c0 = math.log(_component(prev.close(), "close"))
            o1 = math.log(_component(cur.open(), "open"))
            sigma2 = self._a * (o1 - c0) * (o1 - c0) / self._f + (
                1 - self._a
            ) * self._calculate_point(cur) / (1 - self._f)
            retval[date] = math.sqrt(sigma2 / self._year_fraction)
        return retval


class GarmanKlassSigma1(GarmanKlassOpenClose, GarmanKlassSimpleSigma):
    """Open-close variant of :class:`GarmanKlassSimpleSigma`, ``a = 0.5``.

    # C++ parity: ``class GarmanKlassSigma1 :
    # public GarmanKlassOpenClose<GarmanKlassSimpleSigma>``
    # (garmanklass.hpp:107-114).
    """

    __slots__ = ()

    def __init__(self, y: float, market_open_fraction: float) -> None:
        # C++ parity: garmanklass.hpp:110-113 — the weight is hard-coded 0.5.
        super().__init__(y, market_open_fraction, 0.5)


class ParkinsonSigma(GarmanKlassAbstract):
    """High-low range estimator, ``(u - d)^2 / (4 ln 2)``.

    # C++ parity: ``class ParkinsonSigma`` (garmanklass.hpp:117-128).

    Both ``u`` and ``d`` are measured against the OPEN, not against each
    other's midpoint, so ``open`` is read even though the formula is a
    function of the high-low range alone.
    """

    __slots__ = ()

    def _calculate_point(self, p: IntervalPrice) -> float:
        # C++ parity: garmanklass.hpp:123-127.
        open_ = _component(p.open(), "open")
        u = math.log(_component(p.high(), "high") / open_)
        d = math.log(_component(p.low(), "low") / open_)
        return (u - d) * (u - d) / 4.0 / math.log(2.0)


class GarmanKlassSigma3(GarmanKlassOpenClose, ParkinsonSigma):
    """Open-close variant of :class:`ParkinsonSigma`, ``a = 0.17``.

    # C++ parity: ``class GarmanKlassSigma3 :
    # public GarmanKlassOpenClose<ParkinsonSigma>`` (garmanklass.hpp:131-138).
    """

    __slots__ = ()

    def __init__(self, y: float, market_open_fraction: float) -> None:
        # C++ parity: garmanklass.hpp:134-137 — the weight is hard-coded 0.17.
        super().__init__(y, market_open_fraction, 0.17)


class GarmanKlassSigma4(GarmanKlassAbstract):
    """Four-field estimator with the published 0.511 / 0.019 / 0.383 weights.

    # C++ parity: ``class GarmanKlassSigma4`` (garmanklass.hpp:142-154).
    """

    __slots__ = ()

    def _calculate_point(self, p: IntervalPrice) -> float:
        # C++ parity: garmanklass.hpp:148-153.
        open_ = _component(p.open(), "open")
        u = math.log(_component(p.high(), "high") / open_)
        d = math.log(_component(p.low(), "low") / open_)
        c = math.log(_component(p.close(), "close") / open_)
        return 0.511 * (u - d) * (u - d) - 0.019 * (c * (u + d) - 2 * u * d) - 0.383 * c * c


class GarmanKlassSigma5(GarmanKlassAbstract):
    """Range/close estimator, ``0.5 (u - d)^2 - (2 ln 2 - 1) c^2``.

    # C++ parity: ``class GarmanKlassSigma5`` (garmanklass.hpp:156-168).
    """

    __slots__ = ()

    def _calculate_point(self, p: IntervalPrice) -> float:
        # C++ parity: garmanklass.hpp:162-167.
        open_ = _component(p.open(), "open")
        u = math.log(_component(p.high(), "high") / open_)
        d = math.log(_component(p.low(), "low") / open_)
        c = math.log(_component(p.close(), "close") / open_)
        return 0.5 * (u - d) * (u - d) - (2.0 * math.log(2.0) - 1.0) * c * c


class GarmanKlassSigma6(GarmanKlassOpenClose, GarmanKlassSigma4):
    """Open-close variant of :class:`GarmanKlassSigma4`, ``a = 0.012``.

    # C++ parity: ``class GarmanKlassSigma6 :
    # public GarmanKlassOpenClose<GarmanKlassSigma4>``
    # (garmanklass.hpp:170-177).
    """

    __slots__ = ()

    def __init__(self, y: float, market_open_fraction: float) -> None:
        # C++ parity: garmanklass.hpp:173-176 — the weight is hard-coded 0.012.
        super().__init__(y, market_open_fraction, 0.012)


__all__ = [
    "GarmanKlassAbstract",
    "GarmanKlassOpenClose",
    "GarmanKlassSigma1",
    "GarmanKlassSigma3",
    "GarmanKlassSigma4",
    "GarmanKlassSigma5",
    "GarmanKlassSigma6",
    "GarmanKlassSimpleSigma",
    "ParkinsonSigma",
]

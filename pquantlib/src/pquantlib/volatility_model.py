"""Root-level volatility-model abstract bases.

# C++ parity: ql/volatilitymodel.hpp @ v1.43.

Two unrelated interfaces live in this header, and the difference between
them is the whole point:

* :class:`LocalVolatilityEstimator` maps a series of *quotes* (of some
  arbitrary type ``T`` — a price, an OHLC bar, ...) onto a series of
  volatilities. It is a one-way estimator: it has no state to fit.
* :class:`VolatilityCompositor` maps a series of *volatilities* onto a
  series of volatilities, and can additionally be ``calibrate``\\ d against
  one. Its input and output are the same type; the estimator's are not.

C++ templates ``LocalVolatilityEstimator`` on the quote type and gives
``VolatilityCompositor`` a ``time_series`` typedef for ``TimeSeries<Volatility>``.
The Python port uses PEP 695 generics for the former (matching
``TimeSeries[T]`` in pquantlib.time.time_series) and spells the latter out
as ``TimeSeries[float]``, because ``Volatility`` is a bare ``Real`` alias in
C++ (ql/types.hpp) with no runtime identity of its own.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pquantlib.time.time_series import TimeSeries


class LocalVolatilityEstimator[T](ABC):
    """Estimates a volatility series from a series of quotes of type ``T``.

    # C++ parity: ``template <class T> class LocalVolatilityEstimator``
    # (ql/volatilitymodel.hpp:33-39).

    Volatilities are expressed on an annual basis.
    """

    __slots__ = ()

    @abstractmethod
    def calculate(self, quote_series: TimeSeries[T]) -> TimeSeries[float]:
        """Estimated volatility for each date of ``quote_series``.

        # C++ parity: ``virtual TimeSeries<Volatility> calculate(
        # const TimeSeries<T>&) = 0`` (volatilitymodel.hpp:37-38).

        Implementations are free to return FEWER dates than they were
        given — every estimator in ql/models/volatility/ that needs a
        previous observation drops the first date.
        """
        ...


class VolatilityCompositor(ABC):
    """Transforms a volatility series, and can be fitted to one.

    # C++ parity: ``class VolatilityCompositor``
    # (ql/volatilitymodel.hpp:41-47). The C++ ``time_series`` typedef for
    # ``TimeSeries<Volatility>`` is spelled out as ``TimeSeries[float]``.
    """

    __slots__ = ()

    @abstractmethod
    def calculate(self, volatility_series: TimeSeries[float]) -> TimeSeries[float]:
        """Transformed volatility series.

        # C++ parity: ``virtual time_series calculate(const time_series&) = 0``
        # (volatilitymodel.hpp:45).
        """
        ...

    @abstractmethod
    def calibrate(self, volatility_series: TimeSeries[float]) -> None:
        """Fit this compositor's parameters to ``volatility_series``.

        # C++ parity: ``virtual void calibrate(const time_series&) = 0``
        # (volatilitymodel.hpp:46). Pure-virtual in C++, so a compositor
        # with nothing to fit must still say so — ``ConstantEstimator``
        # overrides it with an empty body.
        """
        ...


__all__ = ["LocalVolatilityEstimator", "VolatilityCompositor"]

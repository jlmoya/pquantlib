"""DeltaVolQuote — FX delta-convention volatility quote.

# C++ parity: ql/quotes/deltavolquote.{hpp,cpp} (v1.42.1).

A ``DeltaVolQuote`` carries a single FX-market volatility quotation
together with the delta convention (Spot / Fwd / premium-adjusted
variants) and maturity it is quoted against. It can additionally
represent an ATM quote (the ``AtmType`` discriminant) when constructed
without an explicit delta.

The quote IS-A :class:`Quote` and observes the wrapped vol quote so
that downstream observers (e.g. VannaVolga engines) are notified when
the underlying volatility changes.
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib import qassert
from pquantlib.quotes.quote import Quote


class DeltaType(IntEnum):
    """FX delta-quotation convention.

    # C++ parity: ``DeltaVolQuote::DeltaType``. Declaration-order values:
    # Spot=0, Fwd=1, PaSpot=2, PaFwd=3.
    """

    Spot = 0  # Spot Delta, e.g. usual Black-Scholes delta
    Fwd = 1  # Forward Delta
    PaSpot = 2  # Premium-Adjusted Spot Delta
    PaFwd = 3  # Premium-Adjusted Forward Delta


class AtmType(IntEnum):
    """ATM-quotation convention.

    # C++ parity: ``DeltaVolQuote::AtmType``. Declaration-order values:
    # AtmNull=0 .. AtmPutCall50=6.
    """

    AtmNull = 0  # Default, if not an atm quote
    AtmSpot = 1  # K = S_0
    AtmFwd = 2  # K = F
    AtmDeltaNeutral = 3  # Call Delta = Put Delta
    AtmVegaMax = 4  # K such that Vega is Maximum
    AtmGammaMax = 5  # K such that Gamma is Maximum
    AtmPutCall50 = 6  # K such that Call Delta = 0.50 (Fwd delta only)


class DeltaVolQuote(Quote):
    """Quotation of an FX volatility versus delta (or an ATM type).

    Two constructions mirror the C++ overloads:

    * ``DeltaVolQuote(delta, vol, maturity, delta_type)`` — a standard
      delta-quoted vol (``atm_type`` defaults to ``AtmNull``).
    * ``DeltaVolQuote(vol, delta_type, maturity, atm_type)`` — an ATM
      quote (``delta`` is left as ``Null``/``None``).

    The two are disambiguated by argument order/types, exactly as the
    C++ overload set is. The Python port exposes the second form via the
    classmethod :meth:`atm` for an unambiguous call site.
    """

    __slots__ = ("_atm_type", "_delta", "_delta_type", "_maturity", "_vol")

    def __init__(
        self,
        delta: float,
        vol: Quote,
        maturity: float,
        delta_type: DeltaType,
    ) -> None:
        super().__init__()
        self._delta: float | None = delta
        self._vol: Quote = vol
        self._maturity: float = maturity
        self._delta_type: DeltaType = delta_type
        self._atm_type: AtmType = AtmType.AtmNull
        self._vol.register_with(self)

    @classmethod
    def atm(
        cls,
        vol: Quote,
        delta_type: DeltaType,
        maturity: float,
        atm_type: AtmType,
    ) -> DeltaVolQuote:
        """Construct an ATM-quoted vol (second C++ ctor overload).

        # C++ parity: ``DeltaVolQuote(Handle<Quote> vol, DeltaType,
        # Time maturity, AtmType)``.
        """
        # Build with a placeholder delta then overwrite the discriminants.
        obj = cls(0.0, vol, maturity, delta_type)
        obj._delta = None
        obj._atm_type = atm_type
        return obj

    # --- Quote interface ------------------------------------------------

    def value(self) -> float:
        return self._vol.value()

    def is_valid(self) -> bool:
        # C++ ``isValid`` is ``!vol_.empty() && vol_->isValid()``. PQuantLib
        # passes a direct (non-Handle) Quote, so emptiness can't arise.
        return self._vol.is_valid()

    # --- Observer -------------------------------------------------------

    def update(self) -> None:
        # C++ ``DeltaVolQuote::update`` simply forwards notification.
        self.notify_observers()

    # --- accessors ------------------------------------------------------

    def delta(self) -> float:
        qassert.require(self._delta is not None, "delta not provided (ATM quote)")
        assert self._delta is not None
        return self._delta

    def maturity(self) -> float:
        return self._maturity

    def atm_type(self) -> AtmType:
        return self._atm_type

    def delta_type(self) -> DeltaType:
        return self._delta_type


__all__ = ["AtmType", "DeltaType", "DeltaVolQuote"]

"""SwaptionVolatilityCube — abstract Xabr-style swaption vol cube.

# C++ parity: ql/termstructures/volatility/swaption/swaptionvolcube.{hpp,cpp}
# (v1.42.1).

Extends :class:`SwaptionVolatilityDiscrete` with a strike-spread
dimension. The cube is the standard Xabr scaffolding: an ATM vol
surface (handle to a ``SwaptionVolatilityStructure``) plus a
``(option_tenor, swap_tenor, strike_spread)`` grid of vol-spread Quotes.

Subclasses (``SabrSwaptionVolatilityCube``,
``InterpolatedSwaptionVolatilityCube``) implement
:meth:`smile_section_impl` to return a :class:`SmileSection` at any
``(option_time, swap_length)``.

The cube routes ``_volatility_impl(t, s, K)`` through the smile section
via ``smile_section_impl(t, s).volatility(K)`` — matching the C++
behaviour in ``swaptionvolcube.hpp:118-130``.

This Python port keeps the API minimal: the C++ ``fillVolatilityCube``
+ ``createSparseSmiles`` + ``spreadVolInterpolation`` machinery (which
implements the "fit-then-densify" path of ``XabrSwaptionVolatilityCube``)
is encapsulated by the concrete subclasses rather than this base. The
base only ensures the grid sanity checks + Quote registration.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pquantlib import qassert
from pquantlib.indexes.swap_index import SwapIndex
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.volatility.smile_section import SmileSection
from pquantlib.termstructures.volatility.swaption.swaption_volatility_discrete import (
    SwaptionVolatilityDiscrete,
)
from pquantlib.termstructures.volatility.swaption.swaption_volatility_structure import (
    SwaptionVolatilityStructure,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.time.date import Date
from pquantlib.time.period import Period


@runtime_checkable
class AtmSwapIndexProtocol(Protocol):
    """Anything that exposes ``tenor()`` and ``fixing(date) -> float``.

    The cube uses the swap-index base solely to compute the ATM strike
    at each option date — so it suffices to accept any object that
    satisfies this protocol. In practice the user passes a fully-formed
    :class:`SwapIndex`.
    """

    def tenor(self) -> Period: ...

    def fixing(self, fixing_date: Date, forecast_todays_fixing: bool = False) -> float: ...


class SwaptionVolatilityCube(SwaptionVolatilityDiscrete):
    """Abstract Xabr-style swaption volatility cube.

    Args:
        atm_vol_structure: ATM-vol-only surface (a
            :class:`SwaptionVolatilityStructure`). The cube layers
            strike-spread vols on top of it.
        option_tenors: vector of option expiries (ascending Periods).
        swap_tenors: vector of swap tenors (ascending Periods).
        strike_spreads: vector of strike-spread offsets relative to the
            ATM forward at each grid point (must be strictly ascending).
        vol_spreads: row-major ``(option_tenors x swap_tenors) x
            strike_spreads`` matrix of vol-spread :class:`Quote` objects.
            Each row contains the strike-spread vols for one
            ``(option_tenor, swap_tenor)`` pair.
        swap_index_base: long-tenor :class:`SwapIndex` for ATM-strike
            queries when ``swap_tenor > short_swap_index_base.tenor()``.
        short_swap_index_base: short-tenor :class:`SwapIndex` for the
            opposite case.
        vega_weighted_smile_fit: passed to subclass fitters (Sabr cube).
    """

    def __init__(
        self,
        *,
        atm_vol_structure: SwaptionVolatilityStructure,
        option_tenors: Sequence[Period],
        swap_tenors: Sequence[Period],
        strike_spreads: Sequence[float],
        vol_spreads: Sequence[Sequence[Quote]],
        swap_index_base: SwapIndex | AtmSwapIndexProtocol,
        short_swap_index_base: SwapIndex | AtmSwapIndexProtocol,
        vega_weighted_smile_fit: bool = False,
    ) -> None:
        # The ATM structure may have either a fixed reference_date or
        # a settlement_days mode; we forward both fields if available.
        try:
            settlement_days = atm_vol_structure.settlement_days()
        except Exception:
            settlement_days = None
        super().__init__(
            business_day_convention=atm_vol_structure.business_day_convention(),
            option_tenors=option_tenors,
            swap_tenors=swap_tenors,
            calendar=atm_vol_structure.calendar(),
            day_counter=atm_vol_structure.day_counter(),
            reference_date=atm_vol_structure.reference_date(),
            settlement_days=settlement_days,
        )
        qassert.require(
            len(strike_spreads) > 0,
            "empty strike-spread vector",
        )
        for i in range(1, len(strike_spreads)):
            qassert.require(
                strike_spreads[i - 1] < strike_spreads[i],
                f"non-increasing strike spreads at index {i}",
            )
        n_opt = len(option_tenors)
        n_swap = len(swap_tenors)
        n_strikes = len(strike_spreads)
        qassert.require(
            len(vol_spreads) == n_opt * n_swap,
            f"vol-spreads outer dim {len(vol_spreads)} must equal "
            f"n_option_tenors * n_swap_tenors = {n_opt} * {n_swap} = {n_opt * n_swap}",
        )
        for i, row in enumerate(vol_spreads):
            qassert.require(
                len(row) == n_strikes,
                f"vol-spreads row {i} has length {len(row)}; expected {n_strikes}",
            )
        # Short-swap-index tenor must be no longer than the swap-index.
        qassert.require(
            short_swap_index_base.tenor() <= swap_index_base.tenor(),
            f"short index tenor ({short_swap_index_base.tenor()}) must be "
            f"<= swap-index tenor ({swap_index_base.tenor()})",
        )

        self._atm_vol: SwaptionVolatilityStructure = atm_vol_structure
        self._strike_spreads: list[float] = list(strike_spreads)
        # Flatten vol-spreads matrix into row-major list-of-lists for
        # quick ``(j*n_swap + k)`` indexing (mirrors C++).
        self._vol_spreads: list[list[Quote]] = [list(r) for r in vol_spreads]
        self._swap_index_base: SwapIndex | AtmSwapIndexProtocol = swap_index_base
        self._short_swap_index_base: SwapIndex | AtmSwapIndexProtocol = short_swap_index_base
        self._vega_weighted_smile_fit: bool = vega_weighted_smile_fit
        # Register as observer of each spread Quote so any update fires
        # this cube's notify chain.
        for row in self._vol_spreads:
            for q in row:
                q.register_with(self)

    # --- inspectors --------------------------------------------------------

    def strike_spreads(self) -> list[float]:
        return list(self._strike_spreads)

    def vol_spreads(self) -> list[list[Quote]]:
        return [list(r) for r in self._vol_spreads]

    def swap_index_base(self) -> SwapIndex | AtmSwapIndexProtocol:
        return self._swap_index_base

    def short_swap_index_base(self) -> SwapIndex | AtmSwapIndexProtocol:
        return self._short_swap_index_base

    def vega_weighted_smile_fit(self) -> bool:
        return self._vega_weighted_smile_fit

    def atm_vol(self) -> SwaptionVolatilityStructure:
        return self._atm_vol

    # --- ATM strike --------------------------------------------------------

    def atm_strike(self, option_date: Date, swap_tenor: Period) -> float:
        """Compute the ATM swap rate at ``(option_date, swap_tenor)``.

        # C++ parity: SwaptionVolatilityCube::atmStrike
        #             (swaptionvolcube.cpp:89-144) — simplified: PQuantLib
        #             reuses the existing SwapIndex's ``.fixing(date)``
        #             rather than rebuilding a tenor-customised index.
        """
        if swap_tenor > self._short_swap_index_base.tenor():
            return self._swap_index_base.fixing(option_date)
        return self._short_swap_index_base.fixing(option_date)

    # --- subclass hooks ----------------------------------------------------

    @abstractmethod
    def smile_section_impl(
        self,
        option_time: float,
        swap_length: float,
    ) -> SmileSection:
        """Subclass: return the SmileSection at ``(option_time, swap_length)``."""

    def smile_section(
        self,
        option_expiry: Period | Date | float,
        swap_tenor: Period | float,
    ) -> SmileSection:
        """Return the smile section at ``(option_expiry, swap_tenor)``.

        # C++ parity: SwaptionVolatilityStructure::smileSection (lifted
        # via the ``using ::smileSection;`` declaration in the cube
        # header).
        """
        if isinstance(option_expiry, Period):
            t = self.time_from_reference(self.option_date_from_tenor(option_expiry))
        elif isinstance(option_expiry, Date):
            t = self.time_from_reference(option_expiry)
        else:
            t = float(option_expiry)
        length = (
            self.swap_length(swap_tenor) if isinstance(swap_tenor, Period) else swap_tenor
        )
        return self.smile_section_impl(t, length)

    # --- volatility dispatch -----------------------------------------------

    def _volatility_impl(
        self, option_time: float, swap_length: float, strike: float
    ) -> float:
        # C++ parity: SwaptionVolatilityCube::volatilityImpl
        return self.smile_section_impl(option_time, swap_length).volatility(strike)

    def _shift_impl(self, option_time: float, swap_length: float) -> float:
        # C++ parity: SwaptionVolatilityCube::shiftImpl
        # Cube shifts come from the ATM structure.
        return self._atm_vol.shift(option_time, swap_length, extrapolate=True)

    # --- TermStructure interface (delegate to ATM surface) ----------------

    def max_date(self) -> Date:
        return self._atm_vol.max_date()

    def min_strike(self) -> float:
        return -float("inf")

    def max_strike(self) -> float:
        return float("inf")

    def max_swap_tenor(self) -> Period:
        return self._atm_vol.max_swap_tenor()

    def volatility_type(self) -> VolatilityType:
        return self._atm_vol.volatility_type()

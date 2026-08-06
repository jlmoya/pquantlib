"""NoExceptLocalVolSurface — Dupire local vol with a fallback on failure.

# C++ parity: ql/termstructures/volatility/equityfx/noexceptlocalvolsurface.hpp
# (v1.43, header-only).

Thin wrapper around :class:`LocalVolSurface`: when the Dupire calculation
refuses to produce a number — a non-monotone variance in time, or a
negative local variance from a surface that is not smooth enough — the
surface returns ``illegal_local_vol_overwrite`` instead of propagating the
error. Calibrations that sweep a whole (t, S) grid need this: a single bad
node would otherwise abort the sweep.

The C++ ``catch (Error&)`` catches exactly QuantLib's own error type, so
this port catches :class:`LibraryException` (PQuantLib's ``QL_REQUIRE`` /
``QL_ENSURE`` analogue) and nothing wider. A ``ValueError`` out of ``log``
or a ``ZeroDivisionError`` still propagates, as they do in C++.
"""

from __future__ import annotations

from pquantlib.exceptions import LibraryException
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.volatility.equity_fx.black_vol_term_structure import (
    BlackVolTermStructure,
)
from pquantlib.termstructures.volatility.equity_fx.local_vol_surface import (
    LocalVolSurface,
)
from pquantlib.termstructures.yield_term_structure import YieldTermStructure


class NoExceptLocalVolSurface(LocalVolSurface):
    """Local vol surface that substitutes a constant where Dupire fails."""

    def __init__(
        self,
        *,
        black_ts: BlackVolTermStructure,
        risk_free_ts: YieldTermStructure,
        dividend_ts: YieldTermStructure,
        underlying: float | Quote,
        illegal_local_vol_overwrite: float,
    ) -> None:
        super().__init__(
            black_ts=black_ts,
            risk_free_ts=risk_free_ts,
            dividend_ts=dividend_ts,
            underlying=underlying,
        )
        self._illegal_local_vol_overwrite: float = illegal_local_vol_overwrite

    def _local_vol_impl(self, t: float, underlying_level: float) -> float:
        try:
            return super()._local_vol_impl(t, underlying_level)
        except LibraryException:
            return self._illegal_local_vol_overwrite


__all__ = ["NoExceptLocalVolSurface"]

"""FdmQuantoHelper — market data for the quanto drift adjustment.

# C++ parity: ql/methods/finitedifferences/utilities/fdmquantohelper.{hpp,cpp}
# @ v1.43 (6b57206e0).

The adjustment applied to the drift of a quanto-ed equity between ``t1``
and ``t2``::

    r_domestic - r_foreign + equityVol * fxVol * equityFxCorrelation

with both rates taken as continuously-compounded forward rates and
``fxVol`` the forward FX Black vol struck at ``exch_rate_atm_level``.

C++ overloads ``quantoAdjustment`` on ``Real`` vs ``Array``; Python cannot
overload on argument type without a union return, so the vector flavour
gets its own name :meth:`FdmQuantoHelper.quanto_adjustment_array`.
"""

from __future__ import annotations

from typing import final

import numpy as np

from pquantlib.math.array import Array
from pquantlib.patterns.observer import Observable
from pquantlib.termstructures.volatility.equity_fx.black_vol_term_structure import (
    BlackVolTermStructure,
)
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.compounding import Compounding


@final
class FdmQuantoHelper(Observable):
    """Market data needed for the quanto adjustment.

    # C++ parity: ``class FdmQuantoHelper : public Observable``. The C++
    # members are public; the Python port exposes them as read-only
    # accessors of the same names in snake_case.
    """

    def __init__(
        self,
        r_ts: YieldTermStructure,
        f_ts: YieldTermStructure,
        fx_vol_ts: BlackVolTermStructure,
        equity_fx_correlation: float,
        exch_rate_atm_level: float,
    ) -> None:
        super().__init__()
        self.r_ts: YieldTermStructure = r_ts
        self.f_ts: YieldTermStructure = f_ts
        self.fx_vol_ts: BlackVolTermStructure = fx_vol_ts
        self.equity_fx_correlation: float = equity_fx_correlation
        self.exch_rate_atm_level: float = exch_rate_atm_level

    def _rates_and_fx_vol(self, t1: float, t2: float) -> tuple[float, float, float]:
        """The three market quantities shared by both overloads."""
        r_domestic = self.r_ts.forward_rate(t1, t2, Compounding.Continuous).rate()
        r_foreign = self.f_ts.forward_rate(t1, t2, Compounding.Continuous).rate()
        fx_vol = self.fx_vol_ts.black_forward_vol_at_time(t1, t2, self.exch_rate_atm_level)
        return r_domestic, r_foreign, fx_vol

    def quanto_adjustment(self, equity_vol: float, t1: float, t2: float) -> float:
        """Scalar quanto drift adjustment.

        # C++ parity: ``Rate quantoAdjustment(Volatility, Time, Time) const``.
        """
        r_domestic, r_foreign, fx_vol = self._rates_and_fx_vol(t1, t2)
        return r_domestic - r_foreign + equity_vol * fx_vol * self.equity_fx_correlation

    def quanto_adjustment_array(self, equity_vol: Array, t1: float, t2: float) -> Array:
        """Vector quanto drift adjustment, one entry per equity vol.

        # C++ parity: ``Array quantoAdjustment(const Array&, Time, Time) const``.
        """
        r_domestic, r_foreign, fx_vol = self._rates_and_fx_vol(t1, t2)
        out = np.empty(equity_vol.shape[0], dtype=np.float64)
        for i in range(equity_vol.shape[0]):
            out[i] = (
                r_domestic - r_foreign + float(equity_vol[i]) * fx_vol * self.equity_fx_correlation
            )
        return out


__all__ = ["FdmQuantoHelper"]

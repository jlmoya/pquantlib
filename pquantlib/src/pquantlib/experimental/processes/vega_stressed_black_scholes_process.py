"""VegaStressedBlackScholesProcess — GBSM with a local vega-stress bump.

# C++ parity: ql/experimental/processes/vegastressedblackscholesprocess.{hpp,cpp}
# (v1.42.1).

A :class:`GeneralizedBlackScholesProcess` whose ``diffusion`` is bumped
by a constant ``stress_level`` inside a rectangular (time, asset) region
``[lower_time, upper_time] x [lower_asset, upper_asset]``. Used for local
vega stress tests — only the volatility inside the stress box is shifted.

Outside the box (or with the default stress level of 0) the diffusion is
identical to the base GBSM.

The borders + stress level are mutable; each setter calls ``update()`` so
dependent observers reprice (mirroring the C++ ``setXxx`` methods).
"""

from __future__ import annotations

from typing import final

from pquantlib.processes.euler_discretization import EulerDiscretization
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.processes.stochastic_process_1d import (
    StochasticProcess1DDiscretization,
)
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.volatility.equity_fx.black_vol_term_structure import (
    BlackVolTermStructure,
)
from pquantlib.termstructures.yield_term_structure import YieldTermStructure


@final
class VegaStressedBlackScholesProcess(GeneralizedBlackScholesProcess):
    """GBSM process supporting a local vega stress test.

    # C++ parity: ``class VegaStressedBlackScholesProcess : public
    # GeneralizedBlackScholesProcess`` in
    # vegastressedblackscholesprocess.hpp:32.

    Parameters
    ----------
    x0
        Spot quote.
    dividend_ts
        Dividend-yield curve.
    risk_free_ts
        Risk-free curve.
    black_vol_ts
        Black volatility surface.
    lower_time_border, upper_time_border
        Time window of the stress region (defaults 0 .. 1e6).
    lower_asset_border, upper_asset_border
        Asset window of the stress region (defaults 0 .. 1e6).
    stress_level
        Additive volatility bump inside the region (default 0).
    discretization
        Discretization driver (default ``EulerDiscretization``).
    """

    def __init__(
        self,
        *,
        x0: Quote,
        dividend_ts: YieldTermStructure,
        risk_free_ts: YieldTermStructure,
        black_vol_ts: BlackVolTermStructure,
        lower_time_border: float = 0.0,
        upper_time_border: float = 1000000.0,
        lower_asset_border: float = 0.0,
        upper_asset_border: float = 1000000.0,
        stress_level: float = 0.0,
        discretization: StochasticProcess1DDiscretization | None = None,
    ) -> None:
        if discretization is None:
            discretization = EulerDiscretization()
        super().__init__(
            x0=x0,
            dividend_ts=dividend_ts,
            risk_free_ts=risk_free_ts,
            black_vol_ts=black_vol_ts,
            discretization=discretization,
        )
        self._lower_time_border: float = float(lower_time_border)
        self._upper_time_border: float = float(upper_time_border)
        self._lower_asset_border: float = float(lower_asset_border)
        self._upper_asset_border: float = float(upper_asset_border)
        self._stress_level: float = float(stress_level)

    def diffusion_1d(self, t: float, x: float) -> float:
        """Base GBSM local vol, bumped by ``stress_level`` inside the box.

        # C++ parity: ``VegaStressedBlackScholesProcess::diffusion``.
        """
        base = super().diffusion_1d(t, x)
        if (
            self._lower_time_border <= t <= self._upper_time_border
            and self._lower_asset_border <= x <= self._upper_asset_border
        ):
            return base + self._stress_level
        return base

    # --- accessors / mutators (each mutator notifies observers) ---------

    def get_lower_time_border_for_stress_test(self) -> float:
        # C++ parity: getLowerTimeBorderForStressTest().
        return self._lower_time_border

    def set_lower_time_border_for_stress_test(self, ltb: float) -> None:
        # C++ parity: setLowerTimeBorderForStressTest() + update().
        self._lower_time_border = float(ltb)
        self.update()

    def get_upper_time_border_for_stress_test(self) -> float:
        # C++ parity: getUpperTimeBorderForStressTest().
        return self._upper_time_border

    def set_upper_time_border_for_stress_test(self, utb: float) -> None:
        # C++ parity: setUpperTimeBorderForStressTest() + update().
        self._upper_time_border = float(utb)
        self.update()

    def get_lower_asset_border_for_stress_test(self) -> float:
        # C++ parity: getLowerAssetBorderForStressTest().
        return self._lower_asset_border

    def set_lower_asset_border_for_stress_test(self, lab: float) -> None:
        # C++ parity: setLowerAssetBorderForStressTest() + update().
        self._lower_asset_border = float(lab)
        self.update()

    def get_upper_asset_border_for_stress_test(self) -> float:
        # C++ parity: getUpperAssetBorderForStressTest().
        return self._upper_asset_border

    def set_upper_asset_border_for_stress_test(self, uba: float) -> None:
        # C++ parity: setUpperAssetBorderForStressTest() + update().
        self._upper_asset_border = float(uba)
        self.update()

    def get_stress_level(self) -> float:
        # C++ parity: getStressLevel().
        return self._stress_level

    def set_stress_level(self, sl: float) -> None:
        # C++ parity: setStressLevel() + update().
        self._stress_level = float(sl)
        self.update()


__all__ = ["VegaStressedBlackScholesProcess"]

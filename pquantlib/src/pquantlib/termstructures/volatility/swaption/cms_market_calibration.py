"""CmsMarketCalibration — calibrate a swaption-vol cube to CMS market quotes.

# C++ parity: ql/termstructures/volatility/swaption/cmsmarketcalibration.{hpp,cpp}
# (v1.42.1).

The C++ class is a calibrator that adjusts the SABR parameters of a
:class:`SabrSwaptionVolatilityCube` (or any cube derived from
``XabrSwaptionVolatilityCube``) so the implied CMS-coupon vols
reproduce the bid/ask spreads recorded in a :class:`CmsMarket`. The
loop:

  1. Define a cost function ``cost(params) → spread_or_price_errors``.
  2. Project the parameter array through a sign-preserving transform
     (``betaTransformDirect`` / ``reversionTransformDirect``).
  3. Minimise via Levenberg-Marquardt under a ``NoConstraint``.

PQuantLib lands a **structural-loop** subset:

  * Stores the inputs (``vol_cube``, ``cms_market``, ``weights``,
    ``calibration_type``).
  * Exposes :meth:`compute` that wraps ``scipy.optimize.least_squares``
    in the same shape as the C++ method. The actual residual function
    is supplied as a callable parameter — the C++ method derives it
    from ``CmsMarket.weighted*Errors``, which is the deferred part.
  * Exposes the four static parameter-transform helpers
    (``beta_transform_direct/inverse``,
    ``reversion_transform_direct/inverse``) — these are pure-math
    functions and ported as-is.

Documented divergences:

  * The C++ ``compute`` overloads (Array vs Matrix) collapse into a
    single Python method taking a vector ``guess`` plus a callable
    ``residual_fn``. The C++ Matrix overload corresponds to a
    per-(option-tenor, swap-tenor) repeat of the single-cube fit; this
    is delegated to user code via repeated calls.
  * The ``OnSpread`` / ``OnPrice`` / ``OnForwardCmsPrice``
    ``CalibrationType`` enum is preserved — the user picks which slice
    of the C++ cost function to minimise; the residual callable should
    return the matching slice.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import IntEnum
from typing import Any

import numpy as np
from scipy.optimize import least_squares  # type: ignore[import-untyped]

from pquantlib import qassert
from pquantlib.termstructures.volatility.swaption.cms_market import CmsMarket
from pquantlib.termstructures.volatility.swaption.swaption_volatility_cube import (
    SwaptionVolatilityCube,
)


class CalibrationType(IntEnum):
    """Which CMS-market residual slice to minimise.

    # C++ parity: ``CmsMarketCalibration::CalibrationType`` enum
    # (cmsmarketcalibration.hpp:41).
    """

    OnSpread = 0
    OnPrice = 1
    OnForwardCmsPrice = 2


class CmsMarketCalibration:
    """Calibrate a swaption vol cube to CMS market quotes.

    Args:
        volatility_cube: a :class:`SwaptionVolatilityCube` whose SABR
            parameters will be calibrated (currently held opaquely by
            the loop; the callable ``residual_fn`` handed to
            :meth:`compute` is the only state-mutator).
        cms_market: the :class:`CmsMarket` providing reference bid/ask
            spreads.
        weights: a 2-D weighting matrix
            (``(n_swap_lengths, n_swap_indexes)``).
        calibration_type: :class:`CalibrationType` selecting the residual
            slice (``OnSpread`` / ``OnPrice`` / ``OnForwardCmsPrice``).

    # C++ parity: ``CmsMarketCalibration``
    # (cmsmarketcalibration.hpp:43-94).
    """

    def __init__(
        self,
        volatility_cube: SwaptionVolatilityCube,
        cms_market: CmsMarket,
        weights: Sequence[Sequence[float]],
        calibration_type: CalibrationType = CalibrationType.OnSpread,
    ) -> None:
        qassert.require(
            len(weights) == cms_market.n_exercise,
            f"CmsMarketCalibration: weights outer ({len(weights)}) "
            f"must match cms_market.n_exercise ({cms_market.n_exercise})",
        )
        for i, row in enumerate(weights):
            qassert.require(
                len(row) == cms_market.n_swap_indexes,
                f"CmsMarketCalibration: weights row {i} length "
                f"({len(row)}) must match cms_market.n_swap_indexes "
                f"({cms_market.n_swap_indexes})",
            )
        self._vol_cube: SwaptionVolatilityCube = volatility_cube
        self._cms_market: CmsMarket = cms_market
        self._weights: np.ndarray = np.asarray(weights, dtype=np.float64)
        self._calibration_type: CalibrationType = calibration_type
        self._error: float = float("nan")
        self._end_criteria: str = "None"

    # --- inspectors -------------------------------------------------------

    @property
    def volatility_cube(self) -> SwaptionVolatilityCube:
        return self._vol_cube

    @property
    def cms_market(self) -> CmsMarket:
        return self._cms_market

    @property
    def weights(self) -> np.ndarray:
        return self._weights.copy()

    @property
    def calibration_type(self) -> CalibrationType:
        return self._calibration_type

    @property
    def error(self) -> float:
        """Final RMS error after :meth:`compute`."""
        return self._error

    @property
    def end_criteria(self) -> str:
        """Diagnostic string indicating which scipy termination tripped."""
        return self._end_criteria

    # --- parameter transforms ---------------------------------------------
    # # C++ parity: 4 static helpers at cmsmarketcalibration.hpp:75-89.

    @staticmethod
    def beta_transform_direct(y: float) -> float:
        """Map unconstrained ``y`` → constrained beta ∈ (0, 1).

        # C++ parity: ``betaTransformDirect``.
        """
        beta = float(np.exp(-(y * y))) if abs(y) < 10.0 else 0.0
        return max(min(beta, 0.999999), 0.000001)

    @staticmethod
    def beta_transform_inverse(beta: float) -> float:
        """Map constrained ``beta`` → unconstrained ``y``.

        # C++ parity: ``betaTransformInverse``.
        """
        return float(np.sqrt(-np.log(beta)))

    @staticmethod
    def reversion_transform_direct(y: float) -> float:
        """Map unconstrained ``y`` → constrained mean reversion ≥ 0.

        # C++ parity: ``reversionTransformDirect``.
        """
        return float(np.sqrt(y))

    @staticmethod
    def reversion_transform_inverse(reversion: float) -> float:
        """Map constrained ``reversion`` → unconstrained ``y``.

        # C++ parity: ``reversionTransformInverse``.
        """
        return reversion * reversion

    # --- compute (calibration loop) ---------------------------------------

    def compute(
        self,
        guess: Sequence[float],
        residual_fn: Callable[[np.ndarray], np.ndarray],
        *,
        max_nfev: int = 10000,
        xtol: float = 1.0e-8,
        ftol: float = 1.0e-8,
        gtol: float = 1.0e-8,
    ) -> np.ndarray:
        """Run the Levenberg-Marquardt loop against ``residual_fn``.

        Args:
            guess: initial parameter vector (the SABR-cube + mean-reversion
                parameters being calibrated; user-chosen layout).
            residual_fn: callable mapping the current parameter vector to
                a numpy array of residuals. The user is responsible for
                installing the parameters into the cube and reading the
                appropriate :attr:`cms_market.bid_ask_spreads` slice to
                compute the residual.
            max_nfev / xtol / ftol / gtol: scipy.optimize.least_squares
                tuning knobs. Defaults track the C++ ``EndCriteria(10000,
                1e3, 1e-8, 1e-8, 1e-8)`` settings.

        Returns:
            The calibrated parameter array.

        # C++ parity: ``CmsMarketCalibration::compute`` (Array overload,
        # cmsmarketcalibration.cpp:55-105).
        """
        qassert.require(
            len(guess) > 0,
            "CmsMarketCalibration.compute: empty guess",
        )
        x0 = np.asarray(guess, dtype=np.float64)
        result: Any = least_squares(  # pyright: ignore[reportUnknownVariableType]
            residual_fn,
            x0,
            method="trf",
            max_nfev=max_nfev,
            xtol=xtol,
            ftol=ftol,
            gtol=gtol,
        )
        # Persist diagnostics.
        fun_arr: np.ndarray = np.asarray(
            result.fun,  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            dtype=np.float64,
        )
        n = fun_arr.shape[0]
        if n > 1:
            self._error = float(
                np.sqrt(n * float(np.mean(fun_arr * fun_arr)) / (n - 1))
            )
        else:
            self._error = float(np.sqrt(float(np.mean(fun_arr * fun_arr))))
        self._end_criteria = str(
            result.message  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        )
        return np.asarray(
            result.x,  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            dtype=np.float64,
        )


__all__ = ["CalibrationType", "CmsMarketCalibration"]

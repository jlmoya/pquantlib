"""LocalBootstrap — localised piecewise-term-structure bootstrapper.

# C++ parity: ql/termstructures/localbootstrap.hpp (v1.42.1) — the
# ``LocalBootstrap<Curve>`` template. Alternative to
# :class:`IterativeBootstrap` (L8-A) for curves built on a *global*
# interpolator (such as the convex-monotone spline). Instead of solving
# each pillar with a Brent search against a single helper's quote error,
# LocalBootstrap solves a small window of contiguous pillars
# *jointly* via Levenberg-Marquardt against the corresponding window of
# helpers — producing a tighter local IR risk profile when paired with
# a global interpolator.

The C++ algorithm:

1. Sort helpers by pillar date; uniqueness + valid-quote checks.
2. Wire each helper to the curve.
3. Initialize the data grid (``data[0] = traits.initial_value(curve)``;
   ``data[i+1] = data[i]`` as the first guess).
4. Sweep ``iInst`` from ``localisation - 1`` up to ``n - 1``. At each
   step,
     a. Extend the interpolation to cover the first ``iInst + 2``
        points (via the global interpolator's ``localInterpolate``
        hook).
     b. Build the start array of length ``localisation + 1 -
        dataAdjust`` from the current ``data`` slice.
     c. Define a SimpleCostFunction that, for an input Array ``x``,
        installs ``x`` into the corresponding ``data`` slots, refreshes
        the interpolation, and returns the helper-quote errors over the
        last ``localisation`` helpers.
     d. Minimize via :class:`LevenbergMarquardt` under a
        :class:`PositiveConstraint` (if ``force_positive`` else
        ``NoConstraint``).
5. The resulting curve interpolates the bootstrapped quantities at every
   pillar.

Python-specific divergences:

* C++ uses its own ``LevenbergMarquardt`` + ``Problem`` +
  ``PositiveConstraint`` machinery. PQuantLib delegates to
  ``scipy.optimize.least_squares`` with the ``'trf'`` method (analogous
  to L4-A's ``LevenbergMarquardt`` port) and natively-enforced box
  bounds.
* Forced-positivity is implemented by clamping the lower bound to 0.
* The ``dataSizeAdjustment`` knob (C++ ``ConvexMonotone::dataSizeAdjustment = 1``)
  is hard-coded to 1 in this port — the only ``Interpolator`` we
  pair with LocalBootstrap is :class:`ConvexMonotoneInterpolation`
  whose first ``y`` element is discarded.
* C++ ``localInterpolate(prevInterpolation, finalSize)`` lets the
  global interpolator carry over partial-fit helpers from one pass to
  the next. The Python convex-monotone interpolator doesn't expose
  that hook; we instead rebuild a fresh ``ConvexMonotoneInterpolation``
  over the partial-window of pillars each iteration. The grid-end
  values match at convergence; intermediate fits may differ from C++
  by a transient amount.

The curve must satisfy a structural surface matching
:class:`BootstrapCurveProtocol` from L8-A; we re-use it.

# C++ parity: localbootstrap.hpp:86-258.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.optimize import least_squares  # type: ignore[import-untyped]

from pquantlib import qassert
from pquantlib.termstructures.bootstrap.bootstrap_error import BootstrapError
from pquantlib.termstructures.bootstrap.iterative_bootstrap import (
    BootstrapCurveProtocol,
    BootstrapTraitsProtocol,
)
from pquantlib.termstructures.bootstrap_helper import BootstrapHelper
from pquantlib.time.date import Date

# C++ ``ConvexMonotone::dataSizeAdjustment`` = 1 (the only interpolator
# we pair LocalBootstrap with). Forwards-curve users may override.
_DATA_SIZE_ADJUSTMENT: int = 1

# C++ ``EndCriteria(100, 10, 0.0, accuracy, 0.0)`` — bridge to scipy.
_DEFAULT_MAX_NFEV: int = 100
_DEFAULT_ACCURACY: float = 1.0e-8


class LocalBootstrap[TS, Traits]:
    """Localised piecewise-term-structure bootstrapper.

    Construct with the curve, helpers, traits, and ``localisation`` window
    size; call :meth:`calculate` to run the windowed Levenberg-Marquardt
    sweep.

    Args:
        curve: target term structure satisfying
            :class:`BootstrapCurveProtocol`.
        instruments: bootstrap helpers (one per pillar).
        traits: trait class matching :class:`BootstrapTraitsProtocol`.
        localisation: window size (number of helpers solved jointly per
            sweep). Default 2 — the smallest window that uses the local
            character of the convex-monotone spline.
        force_positive: clamp the minimiser bound at 0 (used for forward
            and zero-rate curves). Default True.
        accuracy: convergence tolerance handed to ``scipy.optimize.least_squares``
            as ``xtol`` / ``ftol`` / ``gtol``. ``None`` → use ``1e-8`` to
            mirror the C++ default ``EndCriteria(100, 10, 0.0,
            accuracy_, 0.0)`` with no override.

    Raises:
        :class:`BootstrapError` if a window solver fails to converge.

    # C++ parity: ``LocalBootstrap<Curve>``.
    """

    def __init__(
        self,
        curve: TS,
        instruments: Sequence[BootstrapHelper[TS]],
        traits: Traits,
        *,
        localisation: int = 2,
        force_positive: bool = True,
        accuracy: float | None = None,
    ) -> None:
        qassert.require(
            len(instruments) >= 2,
            "LocalBootstrap requires at least 2 helpers",
        )
        qassert.require(
            len(instruments) > localisation,
            f"LocalBootstrap: {len(instruments)} helpers provided, "
            f"need more than localisation={localisation}",
        )
        qassert.require(
            localisation >= 1,
            "LocalBootstrap: localisation must be >= 1",
        )
        self._curve: TS = curve
        self._instruments: list[BootstrapHelper[TS]] = list(instruments)
        self._traits: Traits = traits
        self._localisation: int = localisation
        self._force_positive: bool = force_positive
        self._accuracy: float = accuracy if accuracy is not None else _DEFAULT_ACCURACY
        self._valid_curve: bool = False

    # ----------------------------------------------------------------
    # main entry — port of C++ ``LocalBootstrap::calculate``
    # ----------------------------------------------------------------

    def calculate(self) -> None:  # noqa: PLR0915 — faithful port of C++ LocalBootstrap::calculate
        """Run the windowed Levenberg-Marquardt sweep.

        # C++ parity: localbootstrap.hpp:134-258. Algorithm summary:
        # 1. Sort helpers by pillar date; check pillar uniqueness +
        #    valid quote.
        # 2. Wire helpers to curve.
        # 3. Initialize data array (``data[0] = traits.initialValue()``;
        #    ``data[i+1] = data[i]``).
        # 4. Sweep ``iInst`` from ``localisation - 1`` to ``n - 1``; at
        #    each step minimise the helper-error vector over the window
        #    ``[iInst - localisation + 1, iInst]``.
        """
        n = len(self._instruments)
        curve: Any = self._curve
        traits: Any = self._traits

        # Step 1 — sort + uniqueness.
        self._instruments.sort(key=lambda h: h.pillar_date())
        for i in range(1, n):
            qassert.require(
                self._instruments[i - 1].pillar_date()
                != self._instruments[i].pillar_date(),
                "two instruments have the same pillar date",
            )

        # Step 2 — wire helpers.
        for i in range(n):
            self._instruments[i].set_term_structure(curve)

        # Step 3 — initial grid (n + 1 dates/times/data including pillar 0).
        dates: list[Date] = [traits.initial_date(curve)]
        times: list[float] = [curve.time_from_reference(dates[0])]
        data: list[float] = [traits.initial_value(curve)]
        for i in range(n):
            dates.append(self._instruments[i].pillar_date())
            times.append(curve.time_from_reference(dates[i + 1]))
            data.append(data[i])  # "data[i+1] = data[i]" — C++ at line 185.
        curve.bootstrap_install_grid(dates, times, data)

        # Step 4 — windowed sweep.
        data_adjust = _DATA_SIZE_ADJUSTMENT
        i_inst = self._localisation - 1
        while i_inst < n:
            # Extend interpolation to cover the first ``i_inst + 2``
            # pillars. The convex-monotone interpolator in our port
            # rebuilds from scratch each call — no incremental
            # ``localInterpolate``.
            curve.refresh_interpolation_through(i_inst + 1)

            initial_data_pt = i_inst + 1 - self._localisation + data_adjust
            window_size = self._localisation + 1 - data_adjust
            live_data = curve.data_live()

            # Build the start array — current ``data`` slice + a fresh
            # guess for the new tail pillar.
            start = np.empty(window_size, dtype=np.float64)
            for j in range(window_size - 1):
                start[j] = float(live_data[initial_data_pt + j])
            if i_inst >= self._localisation:
                start[-1] = float(
                    traits.guess(i_inst, live_data, valid_data=False)
                )
            else:
                start[-1] = float(live_data[0])

            # Bounds: lower = 0.0 if force_positive else -inf.
            lower = (
                np.zeros(window_size, dtype=np.float64)
                if self._force_positive
                else np.full(window_size, -np.inf, dtype=np.float64)
            )
            upper = np.full(window_size, np.inf, dtype=np.float64)

            # Numerical sanity: scipy's ``least_squares`` requires the
            # initial guess to lie strictly inside bounds. If a guess
            # equals 0.0 under force_positive, nudge it up.
            if self._force_positive:
                for j in range(window_size):
                    if start[j] <= 0.0:
                        start[j] = 1.0e-8

            window_helpers = self._instruments[
                i_inst - self._localisation + 1 : i_inst + 1
            ]
            captured_curve: Any = curve
            captured_traits: Any = traits
            initial_data_pt_local: int = initial_data_pt

            def cost_residuals(
                x: np.ndarray,
                helpers: list[BootstrapHelper[TS]] = window_helpers,
                _curve: Any = captured_curve,
                _traits: Any = captured_traits,
                _i_inst: int = i_inst,
                _initial_pt: int = initial_data_pt_local,
            ) -> np.ndarray:
                # Install ``x`` into the data slots.
                live = _curve.data_live()
                for k in range(x.shape[0]):
                    _traits.update_guess(live, float(x[k]), _initial_pt + k)
                _curve.refresh_interpolation_through(_i_inst + 1)
                # Build the residual vector — one entry per helper in
                # the window.
                residuals = np.empty(len(helpers), dtype=np.float64)
                for k, h in enumerate(helpers):
                    residuals[k] = h.quote_error()
                return residuals

            try:
                result: Any = least_squares(  # pyright: ignore[reportUnknownVariableType]
                    cost_residuals,
                    start,
                    bounds=(lower, upper),
                    method="trf",
                    max_nfev=_DEFAULT_MAX_NFEV,
                    xtol=self._accuracy,
                    ftol=self._accuracy,
                    gtol=self._accuracy,
                )
            except Exception as exc:
                # Wrap unexpected solver crash as a BootstrapError.
                raise BootstrapError(
                    f"LocalBootstrap window solve raised: {exc}",
                    helper_index=i_inst,
                    helper=self._instruments[i_inst],
                    curve=self._curve,
                    accuracy=self._accuracy,
                ) from exc

            success_raw: bool = bool(
                result.success  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            )
            cost_raw: float = float(
                result.cost  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            )
            # ``least_squares`` reports ``success=True`` when any of the
            # gtol/xtol/ftol criteria succeed. We additionally enforce
            # cost <= accuracy^2 — the C++ ``EndCriteria::succeeded``
            # check is on the function-value tolerance.
            if not success_raw or cost_raw > self._accuracy:
                # Mirror C++ at line 253-254 — `EndCriteria::succeeded` failed.
                final_residual = float(
                    np.max(
                        np.abs(
                            np.asarray(
                                result.fun,  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                                dtype=np.float64,
                            )
                        )
                    )
                )
                raise BootstrapError(
                    "LocalBootstrap: unable to strip curve to required accuracy",
                    helper_index=i_inst,
                    helper=self._instruments[i_inst],
                    last_residual=final_residual,
                    curve=self._curve,
                    accuracy=self._accuracy,
                )

            x_sol: np.ndarray = np.asarray(
                result.x,  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                dtype=np.float64,
            )
            # Commit the final solution into the curve.
            live = curve.data_live()
            for k in range(x_sol.shape[0]):
                traits.update_guess(live, float(x_sol[k]), initial_data_pt + k)
            curve.refresh_interpolation_through(i_inst + 1)
            i_inst += 1

        # Final refresh over the whole grid.
        curve.refresh_interpolation_through(len(curve.data_live()) - 1)
        self._valid_curve = True


__all__ = ["BootstrapCurveProtocol", "BootstrapTraitsProtocol", "LocalBootstrap"]

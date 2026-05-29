"""MarkovFunctional — Markov-Functional 1-factor short-rate model.

# C++ parity: ql/models/shortrate/onefactormodels/markovfunctional.{hpp,cpp}
# @ v1.42.1 (099987f0).

The Markov-Functional model is a one-factor short-rate model whose
**numeraire is tabulated** (rather than expressed in closed form) on a
2-D ``(time, state)`` grid. The tabulation is calibrated by inverting a
strip of swaption (or caplet) digital prices: at each calibration time
``t_i``, the discrete numeraire row ``N(t_i, y_j)`` is chosen so that
the model reproduces the market digital prices implied by the input
smile section at strike ``swapRate(y_j)``.

References:
- Hunt, Kennedy, Pelsser. "Markov-Functional Interest Rate Models."
  Finance and Stochastics (2000).
- Caspers (2013). "Markov Functional 1F." http://ssrn.com/abstract=2183721

This Python port covers the **swaption-strip-calibrated** branch (the
caplet branch in C++ is deferred — same algorithm but with a single
floating-period leg in place of the swap's annuity). The C++ optional
smile pretreatment (Kahale / SABR) and the experimental ``AdjustYts`` /
``AdjustDigitals`` adjustments are also deferred — the Python ctor
accepts settings to select pretreatment, but only ``NoSmilePretreatment``
+ ``NoPayoffExtrapolation`` (the most degenerate / well-behaved
combination) are exercised by the W1-A probe.

Constructor convention:

The C++ signature is
``MarkovFunctional(termStructure, reversion, volstepdates, volatilities,
swaptionVol, swaptionExpiries, swaptionTenors, swapIndexBase,
ModelSettings)``. The Python port collapses (swaptionVol,
swaptionExpiries, swaptionTenors, swapIndexBase) into the more idiomatic
(swap_indexes, swaption_volatilities) pair: ``swap_indexes[i]`` carries
the tenor for expiry ``i`` and ``swaption_volatilities[i]`` is a
Quote-wrapped lognormal volatility for that expiry. The
``smile_step_dates`` list serves dual duty as both the vol-step grid and
the calibration-expiry grid (matching the C++ "smile step dates ==
swaption expiries" convention seen in the test suite).

State process: a ``Gaussian1dGsrProcess`` (delegating to ``GsrProcess``)
matching the convention of the existing ``Gsr`` model. The forward
measure horizon ``T_fwd`` is set internally to the last calibration
payment date — equivalently, the discount-bond ``P(t, T_fwd | y)`` is
the model's numeraire.

Carve-outs (documented inline below):

- Caplet-based calibration path (``cap_volatilities`` ctor arg) — only
  a stub that raises ``NotImplementedError`` is provided. Re-add by
  porting the C++ ``makeCapletCalibrationPoint`` branch.
- Kahale / SABR smile pretreatment — Python ports of
  ``KahaleSmileSection`` and ``SabrInterpolatedSmileSection`` exist in
  ``termstructures.volatility`` but are not wired into the calibration
  here. Re-add by selecting the smile-section factory in
  ``_update_smiles`` based on ``settings.adjustments``.
- ``ModelOutputs`` (the C++ trace + diagnostic struct) — Python omits
  the diagnostic surface entirely; ``_ModelOutputs`` here is a lightweight
  dict that records the per-expiry atm / annuity / adjustment factors
  needed by the engine.
- ``arbitrageIndices`` / ``forceArbitrageIndices`` — only meaningful
  with Kahale pretreatment; deferred along with it.
- ``Gauss-Hermite`` integration uses ``numpy.polynomial.hermite.hermgauss``
  rather than QL's hand-rolled ``GaussHermiteIntegration`` — the
  underlying recursion is the same (Golub-Welsch), so the resulting
  nodes/weights match QL to TIGHT.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.math.interpolations.cubic_interpolation import CubicNaturalSpline
from pquantlib.math.optimization.constraint import PositiveConstraint
from pquantlib.math.solvers1d.brent import Brent
from pquantlib.models.model import CalibratedModel
from pquantlib.models.parameter import PiecewiseConstantParameter
from pquantlib.models.shortrate.gaussian1d_model import Gaussian1dModel
from pquantlib.processes.gaussian1d_gsr_process import Gaussian1dGsrProcess
from pquantlib.termstructures.volatility.atm_smile_section import AtmSmileSection
from pquantlib.termstructures.volatility.flat_smile_section import FlatSmileSection

if TYPE_CHECKING:
    from pquantlib.indexes.swap_index import SwapIndex
    from pquantlib.quotes.quote import Quote
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol
    from pquantlib.termstructures.volatility.smile_section import SmileSection
    from pquantlib.termstructures.yield_term_structure import YieldTermStructure
    from pquantlib.time.date import Date


# C++ parity: markovfunctional.hpp:115-258 — collapsed into a frozen
# dataclass. Several flag-style bit-or settings collapse to plain bool
# fields; the rarely-used ``smileMoneynessCheckpoints_`` field is
# omitted since no pretreatment path is wired here.
@dataclass(frozen=True, slots=True)
class MarkovFunctionalSettings:
    """Numerical/grid settings for the MarkovFunctional model.

    Defaults track the C++ ``ModelSettings`` ctor with
    ``Kahale + SmileExponentialExtrapolation`` overridden to
    ``NoSmilePretreatment`` (Python carve-out — Kahale pretreatment is
    deferred).
    """

    y_grid_points: int = 32
    y_std_devs: float = 5.0
    gauss_hermite_points: int = 16
    digital_gap: float = 1.0e-5
    market_rate_accuracy: float = 1.0e-7
    lower_rate_bound: float = 0.001
    upper_rate_bound: float = 1.5
    no_payoff_extrapolation: bool = True
    extrapolate_payoff_flat: bool = False
    adjust_digitals: bool = False
    adjust_yts: bool = False
    # KahaleSmile / SabrSmile / CustomSmile — all carve-outs in the
    # Python port. Settings struct accepts them as bool flags for
    # forward compatibility but the calibration path raises
    # NotImplementedError if any is True.
    kahale_smile: bool = False
    sabr_smile: bool = False
    custom_smile: bool = False


# C++ parity: markovfunctional.hpp:260-271 — internal calibration-point
# record. PQuantLib stores it as a mutable dataclass keyed by expiry
# in the model's ``_calibration_points`` dict.
@dataclass(slots=True)
class _CalibrationPoint:
    is_caplet: bool
    tenor: Any  # Period
    payment_dates: list[Any] = field(default_factory=lambda: [])  # list[Date]
    year_fractions: list[float] = field(default_factory=lambda: [])
    atm: float = math.nan
    annuity: float = math.nan
    smile_section: SmileSection | None = None
    raw_smile_section: SmileSection | None = None
    min_rate_digital: float = math.nan
    max_rate_digital: float = math.nan


def _gaussian_shifted_polynomial_integral(
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    h: float,
    x0: float,
    x1: float,
) -> float:
    r"""Closed-form Gaussian integral of a shifted quartic polynomial.

    # C++ parity: ``Gaussian1dModel::gaussianShiftedPolynomialIntegral``
    # (gaussian1dmodel.cpp:285-326).

    Computes
        (2 pi)^{-1/2} \int_{x0}^{x1} p(x) exp(-x^2/2) dx
    where
        p(x) = a (x - h)^4 + b (x - h)^3 + c (x - h)^2 + d (x - h) + e.

    Used by the MarkovFunctional bootstrap and swaption engine for
    integrating cubic-spline payoff coefficients against the standard
    normal kernel on a y-grid segment. The closed form decomposes the
    polynomial moments via the recursion::

        M_0(x) = N(x)
        M_1(x) = -phi(x)
        M_n(x) = -x^{n-1} phi(x) + (n-1) M_{n-2}(x)

    where ``N`` is the standard normal CDF and ``phi`` the density. The
    full integral is then a closed-form combination of these moments
    evaluated at ``x0`` and ``x1`` with a binomial expansion of
    ``(x - h)^k``.
    """
    # Helper aliases.
    sqrt_2pi = math.sqrt(2.0 * math.pi)
    phi0 = math.exp(-0.5 * x0 * x0) / sqrt_2pi
    phi1 = math.exp(-0.5 * x1 * x1) / sqrt_2pi
    # math.erf gives N(x) via 0.5 * (1 + erf(x / sqrt(2))).
    n0 = 0.5 * (1.0 + math.erf(x0 / math.sqrt(2.0)))
    n1 = 0.5 * (1.0 + math.erf(x1 / math.sqrt(2.0)))

    # Moments M_n(x) = \int_{-inf}^{x} t^n phi(t) dt, evaluated at x0,x1.
    # M_0(x) = N(x), M_1(x) = -phi(x), M_n(x) = -x^{n-1} phi(x) + (n-1) M_{n-2}(x).
    m0_0 = n0
    m0_1 = n1
    m1_0 = -phi0
    m1_1 = -phi1
    m2_0 = -x0 * phi0 + m0_0
    m2_1 = -x1 * phi1 + m0_1
    m3_0 = -(x0 * x0) * phi0 + 2.0 * m1_0
    m3_1 = -(x1 * x1) * phi1 + 2.0 * m1_1
    m4_0 = -(x0 ** 3) * phi0 + 3.0 * m2_0
    m4_1 = -(x1 ** 3) * phi1 + 3.0 * m2_1

    # Definite integrals \int_{x0}^{x1} t^n phi(t) dt.
    int0 = m0_1 - m0_0
    int1 = m1_1 - m1_0
    int2 = m2_1 - m2_0
    int3 = m3_1 - m3_0
    int4 = m4_1 - m4_0

    # Now we need \int (x - h)^k phi(x) dx for k = 0..4. Expand via
    # binomial.
    h2 = h * h
    h3 = h2 * h
    h4 = h3 * h

    # (x - h)^0 = 1
    poly0 = int0
    # (x - h)^1 = x - h
    poly1 = int1 - h * int0
    # (x - h)^2 = x^2 - 2 h x + h^2
    poly2 = int2 - 2.0 * h * int1 + h2 * int0
    # (x - h)^3 = x^3 - 3 h x^2 + 3 h^2 x - h^3
    poly3 = int3 - 3.0 * h * int2 + 3.0 * h2 * int1 - h3 * int0
    # (x - h)^4 = x^4 - 4 h x^3 + 6 h^2 x^2 - 4 h^3 x + h^4
    poly4 = int4 - 4.0 * h * int3 + 6.0 * h2 * int2 - 4.0 * h3 * int1 + h4 * int0

    return a * poly4 + b * poly3 + c * poly2 + d * poly1 + e * poly0


# C++ parity: ``ModelOutputs`` (markovfunctional.hpp:282-302) collapsed
# to a tiny per-expiry diagnostics holder.
@dataclass(slots=True)
class _ModelOutputs:
    expiries: list[Any] = field(default_factory=lambda: [])  # list[Date]
    times: list[float] = field(default_factory=lambda: [])
    atm: list[float] = field(default_factory=lambda: [])
    annuity: list[float] = field(default_factory=lambda: [])
    adjustment_factors: list[float] = field(default_factory=lambda: [])


class MarkovFunctional(Gaussian1dModel, CalibratedModel):
    """Markov-Functional 1-factor short-rate model.

    # C++ parity: ``class MarkovFunctional : public Gaussian1dModel,
    # public CalibratedModel`` (markovfunctional.hpp:99-536).

    Subclasses ``Gaussian1dModel`` (for the
    ``numeraire / zerobond / forward_rate / swap_rate / swap_annuity /
    state_process / y_grid`` public surface) and ``CalibratedModel``
    (for the ``Parameter`` / argument tracking — sigma is the
    calibratable parameter; reversion is constant per C++).

    The discrete numeraire is stored as a ``Matrix`` shaped
    ``(num_calibration_times + 2, 2 * y_grid_points + 1)`` plus a list
    of ``CubicNaturalSpline`` interpolators (one per row).
    """

    def __init__(
        self,
        term_structure: YieldTermStructure,
        reversion: float,
        volatility: list[float],
        smile_step_dates: list[Date],
        swap_indexes: Sequence[SwapIndex],
        cap_volatilities: Sequence[Quote] | None = None,
        swaption_volatilities: Sequence[Quote] | None = None,
        settings: MarkovFunctionalSettings | None = None,
    ) -> None:
        # Parent ctors. Gaussian1dModel sets up the LazyObject + curve
        # registration. CalibratedModel allocates the arguments array.
        Gaussian1dModel.__init__(self, term_structure=term_structure)
        CalibratedModel.__init__(self, n_arguments=1)

        self._settings: MarkovFunctionalSettings = settings or MarkovFunctionalSettings()
        # Forward-compatibility: smile pretreatment carve-outs.
        qassert.require(
            not (self._settings.kahale_smile or self._settings.sabr_smile or self._settings.custom_smile),
            "Kahale / SABR / Custom smile pretreatment is deferred in the Python port",
        )
        # Caplet calibration branch — also deferred for now.
        if cap_volatilities is not None:
            raise NotImplementedError(
                "MarkovFunctional caplet-based calibration is deferred in the "
                "Python port. Use swaption_volatilities."
            )
        qassert.require(
            swaption_volatilities is not None,
            "swaption_volatilities is required (caplet calibration is deferred)",
        )
        assert swaption_volatilities is not None
        qassert.require(
            len(swaption_volatilities) == len(swap_indexes),
            f"len(swaption_volatilities) ({len(swaption_volatilities)}) != "
            f"len(swap_indexes) ({len(swap_indexes)})",
        )
        qassert.require(
            len(volatility) == len(smile_step_dates) + 1,
            f"len(volatility) ({len(volatility)}) must equal "
            f"len(smile_step_dates) + 1 ({len(smile_step_dates) + 1})",
        )
        qassert.require(
            len(swap_indexes) >= 1,
            "at least one swap index / expiry is required",
        )

        # Volatility step grid + per-expiry calibration anchors.
        self._volstepdates: list[Date] = list(smile_step_dates)
        self._volatilities_init: list[float] = list(volatility)
        self._reversion_value: float = float(reversion)

        # The calibration expiries (one per swap_index). Following the
        # C++ convention used in the test-suite, expiries are taken
        # from the smile_step_dates: ``swaption_expiries == volstepdates``
        # when the lengths match (N volstep dates + N swaptions), or
        # ``swaption_expiries == volstepdates[: len(swap_indexes)]``
        # when there are fewer swaptions than volstep dates.
        qassert.require(
            len(smile_step_dates) >= len(swap_indexes),
            "smile_step_dates must contain at least len(swap_indexes) dates",
        )
        self._swap_indexes: list[SwapIndex] = list(swap_indexes)
        self._swaption_volatilities: list[Quote] = list(swaption_volatilities)
        self._swaption_expiries: list[Date] = list(smile_step_dates)[: len(swap_indexes)]

        # Mutable state populated by the calibration bootstrap.
        self._volsteptimes: list[float] = []
        self._volsteptimes_array: npt.NDArray[np.float64] = np.zeros(
            len(self._volstepdates), dtype=np.float64
        )
        # Sorted dict-equivalent: list of (expiry_date, _CalibrationPoint)
        # tuples ordered by expiry. Python uses a dict and re-sorts on
        # iteration; matches C++ ``std::map``'s ordering semantics.
        self._calibration_points: dict[Any, _CalibrationPoint] = {}
        self._times: list[float] = []
        self._numeraire_date: Date | None = None
        self._numeraire_time: float = 0.0
        # y-grid (standardized state) — populated by initialize.
        self._y_grid_array: npt.NDArray[np.float64] = np.zeros(0, dtype=np.float64)
        # Discrete numeraire matrix [time_idx, y_idx].
        self._discrete_numeraire: npt.NDArray[np.float64] = np.zeros((0, 0), dtype=np.float64)
        # Cubic interpolators along y, one per time row.
        self._numeraire_interp: list[CubicNaturalSpline] = []
        # Gauss-Hermite nodes + weights normalized for integration
        # against the standard normal density:
        #   E[g(Y)] = (1 / sqrt(pi)) * sum w_i * g(x_i * sqrt(2))
        # where (x_i, w_i) are numpy.polynomial.hermite.hermgauss roots
        # for \int exp(-x^2) f(x) dx ≈ sum w_i f(x_i).
        #
        # Divergence from C++: the C++ uses QL's GaussHermiteIntegration
        # which returns *bare* quadrature weights (no exp(-x^2)
        # folded in), then multiplies them by exp(-x^2) / sqrt(pi).
        # numpy's hermgauss already folds exp(-x^2) in, so we only
        # divide by sqrt(pi).
        nodes, weights = np.polynomial.hermite.hermgauss(self._settings.gauss_hermite_points)
        nodes_scaled = nodes * math.sqrt(2.0)
        weights_scaled = weights / math.sqrt(math.pi)
        self._gauss_hermite_x: npt.NDArray[np.float64] = nodes_scaled
        self._gauss_hermite_w: npt.NDArray[np.float64] = weights_scaled
        # Diagnostic outputs.
        self._model_outputs: _ModelOutputs = _ModelOutputs()

        # Initialize the model — builds the state process, calibration
        # points, and the numeraire tabulation. Runs the full bootstrap.
        self._initialize()

        term_structure.register_with(self)

    # ----- C++ parity: markovfunctional.cpp:134-234 -----------------------

    def _initialize(self) -> None:
        # Update vol-step times in float-year form.
        self._update_times1()

        # Build the per-expiry calibration points (one per swap-index).
        for expiry, swap_idx, vol_q in zip(
            self._swaption_expiries,
            self._swap_indexes,
            self._swaption_volatilities,
            strict=True,
        ):
            self._make_swaption_calibration_point(expiry, swap_idx, vol_q)

        # Find a numeraire date covering all payments. C++ parity:
        # markovfunctional.cpp:170-203 (iterative back-fill loop). For
        # the simplified Python port, we take the latest payment date
        # across all calibration points. No back-fill of intermediate
        # expiries is needed since we don't compute on a fine grid
        # between payments (we never query numeraire beyond the
        # furthest expiry's last payment).
        latest_payment: Date | None = None
        for cp in self._calibration_points.values():
            last = cp.payment_dates[-1]
            if latest_payment is None or last > latest_payment:
                latest_payment = last
        assert latest_payment is not None
        self._numeraire_date = latest_payment

        self._update_times2()

        # arguments[0] = sigma (piecewise-constant). C++ parity:
        # markovfunctional.cpp:207-211.
        sigma_param = PiecewiseConstantParameter(list(self._volsteptimes), PositiveConstraint())
        for i, v in enumerate(self._volatilities_init):
            sigma_param.set_param(i, v)
        self._arguments[0] = sigma_param

        # State process — Gaussian1dGsrProcess with the same vol grid
        # and reversion. The forward measure horizon is the numeraire
        # time. C++ parity: markovfunctional.cpp:213-214.
        self._state_process = Gaussian1dGsrProcess(
            term_structure=self._term_structure,  # type: ignore[arg-type]
            volstepdates=self._volstepdates,
            volatilities=self._volatilities_init,
            reversion=self._reversion_value,
            T=self._numeraire_time,
        )

        # Build the standardized y-grid (state grid).
        self._y_grid_array = self.y_grid(
            self._settings.y_std_devs,
            self._settings.y_grid_points,
        )

        # Initialize the discrete numeraire matrix to 1.0 everywhere.
        # C++ parity: markovfunctional.cpp:218-219.
        n_times = len(self._times)
        n_y = 2 * self._settings.y_grid_points + 1
        self._discrete_numeraire = np.ones((n_times, n_y), dtype=np.float64)
        self._numeraire_interp = []
        for _ in range(n_times):
            self._numeraire_interp.append(
                CubicNaturalSpline(
                    x_seq=self._y_grid_array,  # type: ignore[arg-type]
                    y_seq=np.ones(n_y, dtype=np.float64),  # type: ignore[arg-type]
                )
            )

        # Update smile sections + numeraire tabulation — this is the
        # full bootstrap.
        self._update_smiles()
        self._update_numeraire_tabulation()

    def _update_times1(self) -> None:
        # C++ parity: markovfunctional.cpp:94-112.
        self._volsteptimes = []
        ts = self._term_structure
        assert ts is not None
        self._volsteptimes_array = np.zeros(len(self._volstepdates), dtype=np.float64)
        for j, d in enumerate(self._volstepdates):
            t = ts.time_from_reference(d)
            self._volsteptimes.append(t)
            self._volsteptimes_array[j] = t
            if j == 0:
                qassert.require(
                    self._volsteptimes[0] > 0.0,
                    f"volsteptimes must be positive ({self._volsteptimes[0]})",
                )
            else:
                qassert.require(
                    self._volsteptimes[j] > self._volsteptimes[j - 1],
                    f"volsteptimes must be strictly increasing "
                    f"({self._volsteptimes[j - 1]}@{j - 1}, "
                    f"{self._volsteptimes[j]}@{j})",
                )

    def _update_times2(self) -> None:
        # C++ parity: markovfunctional.cpp:114-131.
        ts = self._term_structure
        assert ts is not None
        assert self._numeraire_date is not None
        self._numeraire_time = ts.time_from_reference(self._numeraire_date)
        self._times = [0.0]
        self._model_outputs.expiries.clear()
        for expiry in sorted(self._calibration_points.keys()):
            self._times.append(ts.time_from_reference(expiry))
            self._model_outputs.expiries.append(expiry)
        self._times.append(self._numeraire_time)
        self._model_outputs.times = list(self._times)

    def _make_swaption_calibration_point(
        self,
        expiry: Date,
        swap_idx: SwapIndex,
        _vol_quote: Quote,
    ) -> None:
        # C++ parity: markovfunctional.cpp:236-261.
        qassert.require(
            expiry not in self._calibration_points,
            f"swaption expiry ({expiry}) occurs more than once in calibration set",
        )

        # The C++ uses the cached ``underlyingSwap`` helper from
        # Gaussian1dModel; we delegate directly to the swap_index.
        underlying = swap_idx.underlying_swap(expiry)
        # Untyped sub-API at module boundary — VanillaSwap lives in
        # ``instruments/`` which is below this layer in the dependency
        # graph; runtime ``cast`` keeps the call site readable.
        from typing import cast  # noqa: PLC0415
        sched: Any = cast("Any", underlying).fixed_schedule()
        cal: Any = sched.calendar
        pay_conv: Any = cast("Any", underlying).payment_convention()
        dc: Any = swap_idx.day_counter()
        dates: list[Any] = list(sched.dates)

        cp = _CalibrationPoint(is_caplet=False, tenor=swap_idx.tenor())
        for k in range(1, len(dates)):
            prev_d = expiry if k == 1 else dates[k - 1]
            curr_d = dates[k]
            cp.year_fractions.append(dc.year_fraction(prev_d, curr_d))
            cp.payment_dates.append(cal.adjust(curr_d, pay_conv))
        self._calibration_points[expiry] = cp

    def _update_smiles(self) -> None:
        # C++ parity: markovfunctional.cpp:286-439 (NoSmilePretreatment
        # branch only — Kahale / SABR / Custom carved out).
        ts = self._term_structure
        assert ts is not None
        for expiry, cp in self._calibration_points.items():
            annuity = 0.0
            for k, pay_date in enumerate(cp.payment_dates):
                annuity += cp.year_fractions[k] * ts.discount(pay_date, True)
            cp.annuity = annuity
            cp.atm = (
                ts.discount(expiry, True)
                - ts.discount(cp.payment_dates[-1], True)
            ) / annuity
            # Build the flat / smile section for this expiry. With a
            # single Quote per expiry the smile is flat ; we wrap in
            # AtmSmileSection to override the atm.
            # Locate the vol quote for this expiry (matching by index
            # in self._swaption_expiries).
            idx = self._swaption_expiries.index(expiry)
            vol_q = self._swaption_volatilities[idx]
            t = ts.time_from_reference(expiry)
            raw = FlatSmileSection(
                volatility=vol_q.value(),
                exercise_date=expiry,
                exercise_time=t,
                day_counter=ts.day_counter(),
                reference_date=ts.reference_date(),
                atm_level=cp.atm,
            )
            cp.raw_smile_section = raw
            cp.smile_section = AtmSmileSection(base=raw, atm=cp.atm)
            cp.min_rate_digital = cp.smile_section.digital_option_price(
                self._settings.lower_rate_bound,
                1,
                cp.annuity,
                self._settings.digital_gap,
            )
            cp.max_rate_digital = cp.smile_section.digital_option_price(
                self._settings.upper_rate_bound,
                1,
                cp.annuity,
                self._settings.digital_gap,
            )
            self._model_outputs.atm.append(cp.atm)
            self._model_outputs.annuity.append(cp.annuity)

    def _update_numeraire_tabulation(self) -> None:
        # C++ parity: markovfunctional.cpp:441-607.
        # This is the central calibration bootstrap.
        ts = self._term_structure
        assert ts is not None
        # Iterate calibration points in reverse expiry order, filling
        # ``discrete_numeraire`` row by row.
        sorted_expiries = sorted(self._calibration_points.keys(), reverse=True)
        idx = len(self._times) - 2  # last interior row
        numeraire0 = ts.discount(self._numeraire_time, True)
        self._model_outputs.adjustment_factors = []

        for expiry in sorted_expiries:
            cp = self._calibration_points[expiry]
            assert cp.smile_section is not None
            assert cp.raw_smile_section is not None

            # 1) Compute the deflated annuity on the y-grid.
            # discreteDeflatedAnnuities[j] = sum_k yf_k * P(t_i, T_k, y_j) / N(t_i, y_j)
            # which equals sum_k yf_k * P(t_i, T_k, y_j) * (1/N) — we
            # compute the "deflated zerobond" array via gauss-hermite
            # integration over the standardized state.
            normalization = ts.discount(self._times[idx], True) / numeraire0

            discrete_deflated_annuities = np.zeros(
                self._y_grid_array.size, dtype=np.float64
            )
            deflated_final_payments: npt.NDArray[np.float64] = np.zeros(0, dtype=np.float64)
            for k, pay_date in enumerate(cp.payment_dates):
                pay_t = ts.time_from_reference(pay_date)
                deflated_final_payments = self._deflated_zerobond_array(
                    pay_t, self._times[idx], self._y_grid_array
                )
                discrete_deflated_annuities += deflated_final_payments * cp.year_fractions[k]

            # 2) Build the cubic spline of deflated_annuities in y.
            # Used by the digital-price integration loop below.
            try:
                deflated_annuities_interp = CubicNaturalSpline(
                    x_seq=self._y_grid_array,  # type: ignore[arg-type]
                    y_seq=discrete_deflated_annuities,  # type: ignore[arg-type]
                )
            except Exception:
                deflated_annuities_interp = None

            # 3) Right-to-left sweep over the y-grid. For each j we
            # accumulate the cumulative digital and invert the input
            # smile to find the model swap rate at that y.
            digital = 0.0
            swap_rate_0 = self._settings.upper_rate_bound / 2.0
            y_arr = self._y_grid_array
            y_size = y_arr.size

            for j in range(y_size - 1, -1, -1):
                integral = 0.0
                if j == y_size - 1:
                    if not self._settings.no_payoff_extrapolation:
                        # Flat or polynomial extrapolation — both
                        # disabled in Python carve-out beyond the
                        # nominal grid. (Re-add when porting the
                        # ExtrapolatePayoffFlat behaviour.)
                        pass
                # Integrate the cubic-spline segment over y in
                # [y_j, y_{j+1}] against the std normal density.
                elif deflated_annuities_interp is not None:
                    # scipy returns segment coefficients in
                    # (cubic, quadratic, linear) order, which maps
                    # to QL's (cCoef, bCoef, aCoef) — see comment
                    # in _segment_polynomial_coeffs.
                    cubic, quadratic, linear = self._segment_polynomial_coeffs(
                        deflated_annuities_interp,
                        discrete_deflated_annuities,
                        j,
                    )
                    # _gaussian_shifted_polynomial_integral signature:
                    #   (quartic, cubic, quadratic, linear, const, h, x0, x1)
                    # where p(x) = a(x-h)^4 + b(x-h)^3 + c(x-h)^2 + d(x-h) + e.
                    integral = _gaussian_shifted_polynomial_integral(
                        0.0, cubic, quadratic, linear,
                        float(discrete_deflated_annuities[j]),
                        float(y_arr[j]),
                        float(y_arr[j]),
                        float(y_arr[j + 1]),
                    )
                if integral < 0:
                    integral = 0.0
                digital += integral * numeraire0

                if digital >= cp.min_rate_digital:
                    swap_rate = self._settings.lower_rate_bound
                elif digital <= cp.max_rate_digital:
                    swap_rate = self._settings.upper_rate_bound
                else:
                    swap_rate = self._market_swap_rate(
                        expiry, cp, digital, swap_rate_0
                    )
                swap_rate_0 = swap_rate
                numeraire_jb = 1.0 / max(
                    swap_rate * float(discrete_deflated_annuities[j])
                    + float(deflated_final_payments[j]),
                    1.0e-6,
                )
                self._discrete_numeraire[idx, j] = numeraire_jb * normalization

            self._model_outputs.adjustment_factors.append(1.0)

            # Refresh the interpolator for this row (used by
            # subsequent rows' deflated-zerobond integral).
            self._numeraire_interp[idx] = CubicNaturalSpline(
                x_seq=self._y_grid_array,  # type: ignore[arg-type]
                y_seq=self._discrete_numeraire[idx].copy(),  # type: ignore[arg-type]
            )

            idx -= 1

    def _segment_polynomial_coeffs(
        self,
        interp: CubicNaturalSpline,
        ys: npt.NDArray[np.float64],
        j: int,
    ) -> tuple[float, float, float]:
        """Return scipy cubic segment coefficients as ``(cubic, quadratic, linear)``.

        # C++ parity: ``CubicInterpolation::aCoefficients()`` /
        # ``bCoefficients()`` / ``cCoefficients()`` from
        # cubicinterpolation.hpp.
        #
        # QL's segment representation (cubicinterpolation.hpp):
        #   s(y) = ys[j] + aCoef[j] (y - y_j) + bCoef[j] (y - y_j)^2
        #               + cCoef[j] (y - y_j)^3
        # so ``aCoef = linear``, ``bCoef = quadratic``, ``cCoef = cubic``.
        #
        # scipy's CubicSpline polynomial convention (PPoly with c shape
        # ``(4, n-1)``) is:
        #   s(y) = c[0,j] (y - y_j)^3 + c[1,j] (y - y_j)^2
        #        + c[2,j] (y - y_j)   + c[3,j]
        # so ``c[0] = cubic, c[1] = quadratic, c[2] = linear,
        #     c[3] = const (= ys[j])``.
        #
        # The (cubic, quadratic, linear) ordering matches the order the
        # caller wants to pass to ``_gaussian_shifted_polynomial_integral``
        # (whose signature is ``(quartic, cubic, quadratic, linear, ...)``).
        """
        _ = ys
        # pyright: ignore[reportPrivateUsage] — scipy CubicSpline access
        spline = interp._spline  # type: ignore[reportPrivateUsage]
        c_arr = spline.c
        return float(c_arr[0, j]), float(c_arr[1, j]), float(c_arr[2, j])

    def _market_swap_rate(
        self,
        expiry: Date,
        cp: _CalibrationPoint,
        digital_price: float,
        guess: float,
    ) -> float:
        # C++ parity: markovfunctional.cpp:814-828.
        _ = expiry  # unused
        assert cp.smile_section is not None

        def residual(strike: float) -> float:
            assert cp.smile_section is not None
            return (
                cp.smile_section.digital_option_price(
                    strike,
                    1,  # Call
                    cp.annuity,
                    self._settings.digital_gap,
                )
                - digital_price
            )

        b = Brent()
        # Bracket within (lower, upper) bounds with the C++ epsilon
        # buffer.
        eps = 1.0e-5
        return b.solve(
            residual,
            self._settings.market_rate_accuracy,
            min(
                max(guess, self._settings.lower_rate_bound + eps),
                self._settings.upper_rate_bound - eps,
            ),
            self._settings.lower_rate_bound,
            self._settings.upper_rate_bound,
        )

    # --- numeraire / zerobond array implementations ---------------------

    def _numeraire_array(
        self, t: float, y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        # C++ parity: markovfunctional.cpp:704-741.
        ts = self._term_structure
        assert ts is not None
        if t < 1.0e-15:
            return np.full(y.size, ts.discount(self._numeraire_time, True))

        inverse_normalization = ts.discount(self._numeraire_time, True) / ts.discount(t, True)

        tz = min(t, self._times[-1])
        # i = first index with self._times[i] > t (clamped to len-1).
        i = int(np.searchsorted(np.asarray(self._times), t, side="right"))
        if i >= len(self._times):
            i = len(self._times) - 1
        if i == 0:
            i = 1

        ta = self._times[i - 1]
        tb = self._times[i]
        dt = tb - ta
        dt = max(dt, 1.0e-15)

        result = np.zeros(y.size, dtype=np.float64)
        y_min = float(self._y_grid_array[0])
        y_max = float(self._y_grid_array[-1])
        for j in range(y.size):
            yv = float(y[j])
            yv = max(yv, y_min)
            yv = min(yv, y_max)
            na = float(self._numeraire_interp[i - 1](yv))
            nb = float(self._numeraire_interp[i](yv))
            # Linear-in-1/N interpolation between (ta, na) and (tb, nb).
            result[j] = inverse_normalization / ((tz - ta) / nb + (tb - tz) / na) * dt
        return result

    def _deflated_zerobond_array(
        self, T: float, t: float, y: npt.NDArray[np.float64]  # noqa: N803
    ) -> npt.NDArray[np.float64]:
        # C++ parity: markovfunctional.cpp:748-774.
        assert self._state_process is not None
        sp = self._state_process

        std_dev_0_t = sp.std_deviation_1d(0.0, 0.0, t) if t > 1.0e-15 else 0.0
        std_dev_0_T = sp.std_deviation_1d(0.0, 0.0, T)  # noqa: N806
        std_dev_t_T = sp.std_deviation_1d(t, 0.0, max(T - t, 1.0e-15))  # noqa: N806

        result = np.zeros(y.size, dtype=np.float64)
        gh_x = self._gauss_hermite_x
        gh_w = self._gauss_hermite_w
        n_gh = gh_x.size
        for j in range(y.size):
            ya = np.zeros(n_gh, dtype=np.float64)
            for i in range(n_gh):
                ya[i] = (float(y[j]) * std_dev_0_t + std_dev_t_T * float(gh_x[i])) / max(
                    std_dev_0_T, 1.0e-15
                )
            res = self._numeraire_array(T, ya)
            for i in range(n_gh):
                result[j] += float(gh_w[i]) / float(res[i])

        return result

    def _zerobond_array(
        self, T: float, t: float, y: npt.NDArray[np.float64]  # noqa: N803
    ) -> npt.NDArray[np.float64]:
        # C++ parity: markovfunctional.cpp:743-746.
        return self._deflated_zerobond_array(T, t, y) * self._numeraire_array(t, y)

    # --- Gaussian1dModel hook implementations ---------------------------

    def numeraire_impl(
        self,
        t: float,
        y: float,
        yts: YieldTermStructureProtocol | None,
    ) -> float:
        # C++ parity: markovfunctional.cpp:776-791.
        ts = self._term_structure
        assert ts is not None
        if t == 0.0:
            if yts is None:
                return ts.discount(self._numeraire_time, True)
            return yts.discount(self._numeraire_time)

        ya = np.array([y], dtype=np.float64)
        base = float(self._numeraire_array(t, ya)[0])
        if yts is None:
            return base
        # External-yts adjustment factor.
        return base * (
            yts.discount(self._numeraire_time)
            / yts.discount(t)
            * ts.discount(t, True)
            / ts.discount(self._numeraire_time, True)
        )

    def zerobond_impl(
        self,
        T: float,  # noqa: N803
        t: float,
        y: float,
        yts: YieldTermStructureProtocol | None,
    ) -> float:
        # C++ parity: markovfunctional.cpp:793-805.
        ts = self._term_structure
        assert ts is not None
        if t == 0.0:
            if yts is None:
                return ts.discount(T, True)
            return yts.discount(T)

        ya = np.array([y], dtype=np.float64)
        base = float(self._zerobond_array(T, t, ya)[0])
        if yts is None:
            return base
        # External-yts adjustment factor.
        return base * (
            yts.discount(T) / yts.discount(t) * ts.discount(t, True) / ts.discount(T, True)
        )

    # --- inspectors -----------------------------------------------------

    def get_numeraire_date(self) -> Date:
        """Numeraire date — the forward-measure horizon expressed as a Date.

        # C++ parity: ``MarkovFunctional::numeraireDate()`` (markovfunctional.hpp:328).
        # Renamed to ``get_numeraire_date`` to avoid colliding with
        # ``Gaussian1dModel.numeraire_date(reference_date, y, yts)``
        # which is the Date-overload of ``numeraire(...)``.
        """
        assert self._numeraire_date is not None
        return self._numeraire_date

    def numeraire_time(self) -> float:
        """Forward-measure horizon in years.

        # C++ parity: markovfunctional.hpp:329 (inline).
        """
        return self._numeraire_time

    def volatility(self) -> npt.NDArray[np.float64]:
        """Piecewise volatility array.

        # C++ parity: markovfunctional.hpp:331 (inline).
        """
        return self._arguments[0].params

    def reversion_value(self) -> float:
        """Constant reversion value used to build the state process."""
        return self._reversion_value

    def model_outputs(self) -> _ModelOutputs:
        """Diagnostic per-expiry calibration outputs (atm / annuity / adj factors)."""
        return self._model_outputs

    # --- engine-facing accessors ---------------------------------------
    #
    # These are not part of the C++ public surface but the engine
    # subclass needs structured access without going through the
    # protected `_` underscore fields. They mirror the C++
    # ``MarkovFunctional::modelSettings()``-style passthrough.

    def settings(self) -> MarkovFunctionalSettings:
        """Numerical / grid settings used by the model.

        # C++ parity: ``MarkovFunctional::modelSettings()`` accessor.
        """
        return self._settings

    def swap_indexes(self) -> list[SwapIndex]:
        """List of swap indexes used to construct the calibration points.

        # C++ parity: ``MarkovFunctional::swapIndexBase_`` (singular —
        # the C++ uses a single base index since the tenors are
        # provided separately as ``swaptionTenors``). Python's
        # plural form mirrors the more flexible ctor signature.
        """
        return self._swap_indexes

    def gaussian_shifted_polynomial_integral_helper(
        self,
        a: float,
        b: float,
        c: float,
        d: float,
        e: float,
        h: float,
        x0: float,
        x1: float,
    ) -> float:
        """Public wrapper for ``_gaussian_shifted_polynomial_integral``.

        Used by ``MarkovFunctionalSwaptionEngine`` (and any future
        Gaussian1d-flavoured engine) to access the closed-form
        polynomial-integral helper without going through the
        underscore-prefixed module-level function.
        """
        return _gaussian_shifted_polynomial_integral(a, b, c, d, e, h, x0, x1)


__all__ = [
    "MarkovFunctional",
    "MarkovFunctionalSettings",
]

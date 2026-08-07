"""Gaussian1dModel — abstract base for one-factor Gaussian short-rate models.

# C++ parity: ql/models/shortrate/onefactormodels/gaussian1dmodel.{hpp,cpp}
# @ v1.42.1 (099987f0).

The Gaussian1d framework decomposes the short rate as

    r_t = phi(t) + x_t

where ``phi(t)`` is the term-structure-fitting function (held by the
curve) and ``x_t`` is the standardized state variable
(zero-mean / unit-variance form of the model's native state ``y``).
Subclasses implement the closed-form ``zerobond_impl(T, t, y, yts)`` and
``numeraire_impl(t, y, yts)`` and supply a ``StochasticProcess1D``
state process via ``_state_process``.

Concrete subclasses in this codebase:
- ``Gsr`` — piecewise-vol + piecewise-reversion Gaussian short rate
  (forward measure formulation).

Deferred (carve-out per Phase 10 design):
- ``MarkovFunctional`` — Markov-functional model.

This module also exposes:
- ``Gaussian1dModel.zerobond(T, t, y, yts)`` / ``zerobond_date(maturity,
  reference_date, y, yts)`` — public entry points dispatching through
  ``zerobond_impl``.
- ``forward_rate(fixing, reference_date, y, ibor_index)`` —
  Ibor-coupon forward rate under the model.
- ``swap_rate``, ``swap_annuity`` — par-swap rate and annuity for a
  given swap index.
- ``y_grid(stdDevs, gridPoints, T, t, y)`` — the AMC integration grid.

Divergences from C++ noted inline.

Python notes:

- The C++ class is a multiple-inheritance crossover
  ``Gaussian1dModel : public TermStructureConsistentModel, public
  LazyObject``. PQuantLib mirrors that.
- The C++ ``swapCache_`` is an ``unordered_map`` of recently-built
  ``VanillaSwap`` from swap indexes; PQuantLib uses a plain ``dict``
  keyed by ``(swap_index.name(), fixing_serial, tenor_repr)``.
- C++ ``zerobondOption(...)`` (option on a zero-bond) uses cubic
  interpolation over a 1-D state grid + a closed-form polynomial
  integral (``gaussianShiftedPolynomialIntegral``). The polynomial
  integral helpers were ported in Phase 11 W1-B (used by the
  Gaussian1d cap/floor + float-float-swaption + non-standard-swaption
  engines that landed there). The ``zerobondOption`` method itself
  remains deferred — none of the W1-B engines call it; they call the
  polynomial-integral helpers directly with payoff-replication
  coefficients from a per-state cubic spline.
- C++ ``gaussianPolynomialIntegral`` / ``gaussianShiftedPolynomialIntegral``
  are static utilities — ported as static methods on
  :class:`Gaussian1dModel` in Phase 11 W1-B.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.math.interpolations.cubic_interpolation import (
    BoundaryCondition,
    CubicInterpolation,
    DerivativeApprox,
)
from pquantlib.models.model import TermStructureConsistentModel
from pquantlib.patterns.lazy_object import LazyObject
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.patterns.observer import Observable
from pquantlib.payoffs import OptionType

if TYPE_CHECKING:
    from pquantlib.indexes.ibor_index import IborIndex
    from pquantlib.indexes.swap_index import SwapIndex
    from pquantlib.processes.stochastic_process_1d import StochasticProcess1D
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol
    from pquantlib.termstructures.yield_term_structure import YieldTermStructure
    from pquantlib.time.date import Date
    from pquantlib.time.period import Period

EXTRAPOLATION_BOUND: float = 100.0
"""# C++ parity: gaussian1dmodel.cpp:187-200 and every Gaussian1d engine use
the literal +/-100.0 as the effective infinite bound; beyond ~7 standard
deviations the normal density is numerically zero."""


def payoff_interpolation(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> CubicInterpolation:
    """The exact interpolant every Gaussian1d payoff replication uses.

    # C++ parity: ``CubicInterpolation(z.begin(), z.end(), p.begin(),
    # CubicInterpolation::Spline, true, CubicInterpolation::Lagrange, 0.0,
    # CubicInterpolation::Lagrange, 0.0)`` — note ``monotonic == true``
    # (Hyman filter) and Lagrange end conditions. This is NOT a natural
    # cubic spline; substituting one changes the answer at the 1e-5 level.
    """
    return CubicInterpolation(
        x,
        y,
        derivative_approx=DerivativeApprox.Spline,
        monotonic=True,
        left_condition=BoundaryCondition.Lagrange,
        left_value=0.0,
        right_condition=BoundaryCondition.Lagrange,
        right_value=0.0,
    )


class Gaussian1dModel(TermStructureConsistentModel, LazyObject, ABC):
    """Abstract one-factor Gaussian short-rate model.

    # C++ parity: ``class Gaussian1dModel : public
    # TermStructureConsistentModel, public LazyObject`` in
    # gaussian1dmodel.hpp (v1.42.1).

    Subclasses must implement:
    - ``zerobond_impl(T, t, y, yts)`` — zero-coupon-bond price under
      the model.
    - ``numeraire_impl(t, y, yts)`` — numeraire (under the model's
      measure) at time ``t`` given standardized state ``y``.
    - ``_state_process`` slot must be populated (typically a
      ``GsrProcess`` or similar).
    """

    # __slots__ intentionally omitted — TermStructureConsistentModel
    # doesn't declare __slots__ to permit multiple-inheritance with
    # other slotted classes, so adding slots here would force every
    # subclass to carefully manage the diamond. We declare attributes
    # as regular class members instead.

    def __init__(self, term_structure: YieldTermStructure) -> None:
        # C++ parity: gaussian1dmodel.hpp:181-184.
        #
        # Diamond-inheritance note: subclasses (e.g. Gsr) also inherit
        # from ``CalibratedModel`` whose ctor signature takes
        # ``n_arguments``. To avoid an MRO-cooperative super() failure
        # we bypass both bases' ``__init__`` chain by manually
        # initialising the Observable + LazyObject + Term-structure
        # slots in-place. ``Observable.__init__`` is reached
        # idempotently by all branches; calling it once here is the
        # safest entry point.
        Observable.__init__(self)
        # LazyObject state.
        self._calculated: bool = False
        # TermStructureConsistentModel state.
        self._term_structure: YieldTermStructure | None = term_structure
        # Mirror C++ `registerWith(Settings::instance().evaluationDate())` —
        # the model must invalidate its cached evaluation date when the
        # global evaluation date changes.
        ObservableSettings().register_with(self)
        # Set by subclasses in their ctor.
        self._state_process: StochasticProcess1D | None = None
        # Mutable cached fields populated by `_perform_calculations`.
        self._evaluation_date: Date | None = None
        self._enforces_todays_historic_fixings: bool = False
        # C++ parity: gaussian1dmodel.hpp:177 — VanillaSwap cache keyed
        # by (swap_index_name, fixing_serial, tenor_repr).
        self._swap_cache: dict[tuple[str, int, str], object] = {}

    # --- LazyObject: refresh evaluation-date snapshot --------------------

    def _perform_calculations(self) -> None:
        # C++ parity: gaussian1dmodel.hpp:191-195.
        self._evaluation_date = ObservableSettings().evaluation_date_or_today()
        # Python parity: ``enforcesTodaysHistoricFixings`` is not
        # currently surfaced by ObservableSettings; default False.
        self._enforces_todays_historic_fixings = False

    # --- subclass hooks --------------------------------------------------

    @abstractmethod
    def numeraire_impl(
        self,
        t: float,
        y: float,
        yts: YieldTermStructureProtocol | None,
    ) -> float:
        """Numeraire ``N(t, y)`` under the model's measure.

        # C++ parity: ``numeraireImpl`` pure virtual in
        # gaussian1dmodel.hpp:186.
        """

    @abstractmethod
    def zerobond_impl(
        self,
        T: float,  # noqa: N803 — math symbol
        t: float,
        y: float,
        yts: YieldTermStructureProtocol | None,
    ) -> float:
        """Discount-bond price ``P(t, T | y)`` under the model.

        # C++ parity: ``zerobondImpl`` pure virtual in
        # gaussian1dmodel.hpp:188-189.
        """

    def state_process(self) -> StochasticProcess1D:
        """Underlying 1-D state process.

        # C++ parity: gaussian1dmodel.hpp:74 + 224-228 (inline).
        """
        qassert.require(self._state_process is not None, "state process not set")
        # Pyright requires the explicit narrowing.
        assert self._state_process is not None
        return self._state_process

    # --- public numeraire / zerobond surface ----------------------------

    def numeraire(
        self,
        t: float,
        y: float = 0.0,
        yts: YieldTermStructureProtocol | None = None,
    ) -> float:
        # C++ parity: gaussian1dmodel.hpp:230-235 (inline).
        return self.numeraire_impl(t, y, yts)

    def zerobond(
        self,
        T: float,  # noqa: N803 — math symbol
        t: float = 0.0,
        y: float = 0.0,
        yts: YieldTermStructureProtocol | None = None,
    ) -> float:
        # C++ parity: gaussian1dmodel.hpp:237-241 (inline).
        return self.zerobond_impl(T, t, y, yts)

    def numeraire_date(
        self,
        reference_date: Date,
        y: float = 0.0,
        yts: YieldTermStructureProtocol | None = None,
    ) -> float:
        """Numeraire at a ``Date`` reference (overload of ``numeraire``).

        # C++ parity: gaussian1dmodel.hpp:243-248 (inline).
        """
        t = self.term_structure.time_from_reference(reference_date)
        return self.numeraire(t, y, yts)

    def zerobond_date(
        self,
        maturity: Date,
        reference_date: Date | None = None,
        y: float = 0.0,
        yts: YieldTermStructureProtocol | None = None,
    ) -> float:
        """Discount-bond price at a ``Date`` reference + ``Date`` maturity.

        # C++ parity: gaussian1dmodel.hpp:250-259 (inline).
        """
        ts = self.term_structure
        t_T = ts.time_from_reference(maturity)  # noqa: N806 — math symbol
        t_t = ts.time_from_reference(reference_date) if reference_date is not None else 0.0
        return self.zerobond(t_T, t_t, y, yts)

    def zerobond_option(
        self,
        option_type: OptionType,
        expiry: Date,
        value_date: Date,
        maturity: Date,
        strike: float,
        reference_date: Date | None = None,
        y: float = 0.0,
        yts: YieldTermStructureProtocol | None = None,
        y_std_devs: float = 7.0,
        y_grid_points: int = 64,
        extrapolate_payoff: bool = True,
        flat_payoff_extrapolation: bool = False,
    ) -> float:
        """Price of an option on the discount bond ``P(valueDate, maturity)``.

        # C++ parity: ``Gaussian1dModel::zerobondOption`` in
        # gaussian1dmodel.cpp:143-207 (v1.43).

        Payoff replication on the model's y-grid: build the conditional
        payoff at every grid node, fit the SAME cubic interpolant C++
        uses (``CubicInterpolation(Spline, monotonic=True, Lagrange BC
        at both ends)``) and integrate it against the standard-normal
        density in closed form. The optional tails are extended either
        flat or with the boundary segment's cubic; in the non-flat case
        the extended tail is type-dependent — Call extends only the upper
        tail, Put only the lower.
        """
        self.calculate()

        ts = self.term_structure
        fixing_time = ts.time_from_reference(expiry)
        reference_time = (
            0.0 if reference_date is None else ts.time_from_reference(reference_date)
        )

        yg = self.y_grid(y_std_devs, y_grid_points, fixing_time, reference_time, y)
        z = self.y_grid(y_std_devs, y_grid_points)

        sign = 1.0 if option_type == OptionType.Call else -1.0
        p = np.zeros(yg.size, dtype=np.float64)
        for i in range(int(yg.size)):
            yi = float(yg[i])
            exp_val_dsc = self.zerobond_date(value_date, expiry, yi, yts)
            discount = self.zerobond_date(maturity, expiry, yi, yts) / exp_val_dsc
            p[i] = (
                max(sign * (discount - strike), 0.0)
                / self.numeraire(fixing_time, yi, yts)
                * exp_val_dsc
            )

        payoff = payoff_interpolation(z, p)
        a_c = payoff.a_coefficients()
        b_c = payoff.b_coefficients()
        c_c = payoff.c_coefficients()

        z_size = int(z.size)
        price = 0.0
        for i in range(z_size - 1):
            price += Gaussian1dModel.gaussian_shifted_polynomial_integral(
                0.0, c_c[i], b_c[i], a_c[i], float(p[i]), float(z[i]),
                float(z[i]), float(z[i + 1]),
            )
        if extrapolate_payoff:
            last = z_size - 2
            if flat_payoff_extrapolation:
                price += Gaussian1dModel.gaussian_shifted_polynomial_integral(
                    0.0, 0.0, 0.0, 0.0, float(p[last]), float(z[last]),
                    float(z[z_size - 1]), EXTRAPOLATION_BOUND,
                )
                price += Gaussian1dModel.gaussian_shifted_polynomial_integral(
                    0.0, 0.0, 0.0, 0.0, float(p[0]), float(z[0]),
                    -EXTRAPOLATION_BOUND, float(z[0]),
                )
            elif option_type == OptionType.Call:
                price += Gaussian1dModel.gaussian_shifted_polynomial_integral(
                    0.0, c_c[last], b_c[last], a_c[last],
                    float(p[last]), float(z[last]),
                    float(z[z_size - 1]), EXTRAPOLATION_BOUND,
                )
            else:
                price += Gaussian1dModel.gaussian_shifted_polynomial_integral(
                    0.0, c_c[0], b_c[0], a_c[0], float(p[0]), float(z[0]),
                    -EXTRAPOLATION_BOUND, float(z[0]),
                )

        return price * self.numeraire(reference_time, y, yts)

    # --- forward-rate / swap-rate / swap-annuity ------------------------

    def forward_rate(
        self,
        fixing: Date,
        reference_date: Date | None = None,
        y: float = 0.0,
        ibor_index: IborIndex | None = None,
    ) -> float:
        """Forward rate of an Ibor coupon at ``fixing`` under the model.

        # C++ parity: gaussian1dmodel.cpp:27-51.
        """
        qassert.require(ibor_index is not None, "no ibor index given")
        # Help pyright narrow the type.
        assert ibor_index is not None
        self.calculate()

        if self._evaluation_date is not None:
            cutoff = self._evaluation_date if self._enforces_todays_historic_fixings else (
                self._evaluation_date - 1
            )
            if fixing <= cutoff:
                return ibor_index.fixing(fixing)

        yts = ibor_index.forecast_term_structure()  # may be None — fall back to model curve

        value_date = ibor_index.value_date(fixing)
        end_date = ibor_index.fixing_calendar().advance_period(
            value_date,
            ibor_index.tenor(),
            ibor_index.business_day_convention(),
            ibor_index.end_of_month(),
        )
        dcf = ibor_index.day_counter().year_fraction(value_date, end_date)

        p_val = self.zerobond_date(value_date, reference_date, y, yts)
        p_end = self.zerobond_date(end_date, reference_date, y, yts)
        return (p_val - p_end) / (dcf * p_end)

    def _cached_underlying_swap(
        self,
        swap_index: SwapIndex,
        fixing: Date,
        tenor: Period,
    ) -> object:
        """Cache + return the underlying VanillaSwap for ``(swap_index, fixing, tenor)``.

        # C++ parity: gaussian1dmodel.hpp:202-217 inline private member.

        Divergence from C++: the C++ helper clones the swap index with
        a new tenor (``swap_index.clone(tenor)``) before calling
        ``underlying_swap``. PQuantLib's ``SwapIndex.clone`` only
        supports the curve-overload variant — and in practice all callers
        pass ``tenor == swap_index.tenor()`` (the smile section / the
        swaption vol's volatility-impl path), so the tenor argument is
        documented but unused. A future port of ``Gaussian1dCapFloor*``
        / ``Gaussian1dFloatFloatSwaption*`` engines that exercise the
        tenor mismatch would need to extend ``SwapIndex.clone`` first.
        """
        # The cache key still includes the tenor representation so a
        # future tenor-aware clone can be wired in without breaking
        # downstream observers.
        key = (
            swap_index.name(),
            fixing.serial_number(),
            f"{tenor.length}{tenor.units!r}",
        )
        cached = self._swap_cache.get(key)
        if cached is None:
            cached = swap_index.underlying_swap(fixing)
            self._swap_cache[key] = cached
        return cached

    def swap_rate(
        self,
        fixing: Date,
        tenor: Period,
        reference_date: Date | None = None,
        y: float = 0.0,
        swap_index: SwapIndex | None = None,
    ) -> float:
        """Par-swap rate under the model.

        # C++ parity: gaussian1dmodel.cpp:53-111.

        Implementation handles the dual-curve case (separate
        forwarding/discounting). The OIS-index branch falls through to
        the floating-schedule == fixed-schedule simplification.
        """
        qassert.require(swap_index is not None, "no swap index given")
        assert swap_index is not None
        self.calculate()

        if self._evaluation_date is not None:
            cutoff = self._evaluation_date if self._enforces_todays_historic_fixings else (
                self._evaluation_date - 1
            )
            if fixing <= cutoff:
                return swap_index.fixing(fixing)

        ytsf = swap_index.ibor_index().forecast_term_structure()
        ytsd = swap_index.discounting_term_structure()

        # Untyped sub-API at module boundary — VanillaSwap lives in
        # ``instruments/`` which is below this layer in the dependency
        # graph; runtime ``cast`` keeps the call site readable while
        # silencing pyright.
        from typing import Any, cast  # noqa: PLC0415
        underlying: Any = self._cached_underlying_swap(swap_index, fixing, tenor)
        fixed_sched: Any = underlying.fixed_schedule()
        # C++ parity: gaussian1dmodel.cpp:79-85 — OIS index check; for
        # OIS the float schedule == fixed schedule. We don't dispatch on
        # type here (Python: just always use the float schedule from the
        # underlying swap; identical to fixed for OIS).
        float_sched: Any = underlying.floating_schedule()
        pay_conv: Any = underlying.payment_convention()

        annuity = self.swap_annuity(fixing, tenor, reference_date, y, swap_index)

        if ytsf is None and ytsd is None:
            # Simple 100-formula valid in one-curve setup.
            first = cast("Date", fixed_sched.dates[0])
            last = cast("Date", fixed_sched.dates[-1])
            adjusted_last = cast(
                "Date", fixed_sched.calendar.adjust(last, pay_conv)
            )
            floatleg = (
                self.zerobond_date(first, reference_date, y)
                - self.zerobond_date(adjusted_last, reference_date, y)
            )
        else:
            floatleg = 0.0
            for i in range(1, len(float_sched.dates)):
                prev = cast("Date", float_sched.dates[i - 1])
                curr = cast("Date", float_sched.dates[i])
                p_prev = self.zerobond_date(prev, reference_date, y, ytsf)
                p_curr = self.zerobond_date(curr, reference_date, y, ytsf)
                pay_date = cast("Date", float_sched.calendar.adjust(curr, pay_conv))
                p_pay = self.zerobond_date(pay_date, reference_date, y, ytsd)
                floatleg += (p_prev / p_curr - 1.0) * p_pay
        return floatleg / annuity

    def swap_annuity(
        self,
        fixing: Date,
        tenor: Period,
        reference_date: Date | None = None,
        y: float = 0.0,
        swap_index: SwapIndex | None = None,
    ) -> float:
        """Discounted fixed-leg annuity for the given swap.

        # C++ parity: gaussian1dmodel.cpp:113-141.
        """
        qassert.require(swap_index is not None, "no swap index given")
        assert swap_index is not None
        self.calculate()

        ytsd = swap_index.discounting_term_structure()

        from typing import Any, cast  # noqa: PLC0415
        underlying: Any = self._cached_underlying_swap(swap_index, fixing, tenor)
        fixed_sched: Any = underlying.fixed_schedule()
        pay_conv: Any = underlying.payment_convention()

        annuity = 0.0
        for j in range(1, len(fixed_sched.dates)):
            dj_minus_1 = cast("Date", fixed_sched.dates[j - 1])
            dj = cast("Date", fixed_sched.dates[j])
            pay_date = cast("Date", fixed_sched.calendar.adjust(dj, pay_conv))
            disc = self.zerobond_date(pay_date, reference_date, y, ytsd)
            dcf = swap_index.day_counter().year_fraction(dj_minus_1, dj)
            annuity += disc * dcf
        return annuity

    # --- shifted-polynomial Gaussian integral ----------------------------

    @staticmethod
    def gaussian_polynomial_integral(
        a: float,  # quartic coef
        b: float,  # cubic coef
        c: float,  # quadratic coef
        d: float,  # linear coef
        e: float,  # constant
        y0: float,
        y1: float,
    ) -> float:
        r"""Integral over [y0, y1] of (a*y^4 + b*y^3 + c*y^2 + d*y + e)*phi(y)/2 dy.

        # C++ parity: ``Gaussian1dModel::gaussianPolynomialIntegral`` in
        # gaussian1dmodel.cpp:207-238.

        Used by the cubic-interpolated payoff replication in the
        Gaussian1d engines (cap/floor + swaption variants). Substitutes
        ``x = y / sqrt(2)`` and writes the integrals in closed form in
        terms of ``erf`` and ``exp``.

        The C++ ``GAUSS1D_ENABLE_NTL`` arbitrary-precision codepath is
        not ported — the standard-precision branch is what every
        downstream test uses.
        """
        # # C++ parity: substitutions (gaussian1dmodel.cpp:226-228).
        sqrt2 = math.sqrt(2.0)
        aa = 4.0 * a
        ba = 2.0 * sqrt2 * b
        ca = 2.0 * c
        da = sqrt2 * d
        x0 = y0 / sqrt2
        x1 = y1 / sqrt2
        sqrtpi = math.sqrt(math.pi)
        term_x1 = (
            0.125 * (3.0 * aa + 2.0 * ca + 4.0 * e) * math.erf(x1)
            - (1.0 / (4.0 * sqrtpi))
            * math.exp(-x1 * x1)
            * (
                2.0 * aa * x1 * x1 * x1
                + 3.0 * aa * x1
                + 2.0 * ba * (x1 * x1 + 1.0)
                + 2.0 * ca * x1
                + 2.0 * da
            )
        )
        term_x0 = (
            0.125 * (3.0 * aa + 2.0 * ca + 4.0 * e) * math.erf(x0)
            - (1.0 / (4.0 * sqrtpi))
            * math.exp(-x0 * x0)
            * (
                2.0 * aa * x0 * x0 * x0
                + 3.0 * aa * x0
                + 2.0 * ba * (x0 * x0 + 1.0)
                + 2.0 * ca * x0
                + 2.0 * da
            )
        )
        return term_x1 - term_x0

    @staticmethod
    def gaussian_shifted_polynomial_integral(
        a: float,  # quartic coef
        b: float,  # cubic coef
        c: float,  # quadratic coef
        d: float,  # linear coef
        e: float,  # constant
        h: float,  # shift
        x0: float,
        x1: float,
    ) -> float:
        r"""Integral of the *shifted* polynomial under standard normal density.

        # C++ parity: ``Gaussian1dModel::gaussianShiftedPolynomialIntegral``
        # in gaussian1dmodel.cpp:240-247.

        Computes ``∫_{x0}^{x1} P(y - h) * phi(y) dy`` where
        ``P(t) = a t^4 + b t^3 + c t^2 + d t + e``. Expanding the
        substitution into the unshifted basis ``a' t^4 + b' t^3 +
        c' t^2 + d' t + e'`` and delegating to
        :meth:`gaussian_polynomial_integral`.
        """
        return Gaussian1dModel.gaussian_polynomial_integral(
            a,
            -4.0 * a * h + b,
            6.0 * a * h * h - 3.0 * b * h + c,
            -4.0 * a * h * h * h + 3.0 * b * h * h - 2.0 * c * h + d,
            a * h * h * h * h - b * h * h * h + c * h * h - d * h + e,
            x0,
            x1,
        )

    # --- y-grid utility -------------------------------------------------

    def y_grid(
        self,
        std_devs: float,
        grid_points: int = 64,
        T: float = 1.0,  # noqa: N803 — math symbol
        t: float = 0.0,
        y: float = 0.0,
    ) -> npt.NDArray[np.float64]:
        """Standardized state-variable grid for AMC integration.

        # C++ parity: gaussian1dmodel.cpp:249-284.

        Builds a ``2 * grid_points + 1`` grid of standardized state
        values at time ``T`` conditional on ``y(t) = y``, covering
        ``+/- std_devs`` standard deviations. The grid is built in the
        state ``y`` (standardized) frame.

        Uses that the state-process std-deviation is independent of
        the state (one of the defining assumptions of Gaussian1d).
        """
        qassert.require(
            self._state_process is not None, "state process not set"
        )
        assert self._state_process is not None
        sp = self._state_process

        std_dev_0_T = sp.std_deviation_1d(0.0, 0.0, T)  # noqa: N806 — math symbol
        e_0_T = sp.expectation_1d(0.0, 0.0, T)  # noqa: N806 — math symbol

        if t < 1.0e-16:
            std_dev_t_T = std_dev_0_T  # noqa: N806 — math symbol
            e_t_T = e_0_T  # noqa: N806 — math symbol
        else:
            std_dev_0_t = sp.std_deviation_1d(0.0, 0.0, t)
            std_dev_t_T = sp.std_deviation_1d(t, 0.0, T - t)  # noqa: N806 — math symbol
            e_0_t = sp.expectation_1d(0.0, 0.0, t)
            x_t = y * std_dev_0_t + e_0_t
            e_t_T = sp.expectation_1d(t, x_t, T - t)  # noqa: N806 — math symbol

        h = std_devs / float(grid_points)
        result = np.zeros(2 * grid_points + 1, dtype=np.float64)
        for j in range(-grid_points, grid_points + 1):
            result[j + grid_points] = (
                e_t_T + std_dev_t_T * float(j) * h - e_0_T
            ) / std_dev_0_T
        return result


__all__ = ["EXTRAPOLATION_BOUND", "Gaussian1dModel", "payoff_interpolation"]

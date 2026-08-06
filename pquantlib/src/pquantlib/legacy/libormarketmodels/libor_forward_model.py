"""LiborForwardModel — LIBOR forward model with exact caps and Rebonato swaptions.

# C++ parity: ql/legacy/libormarketmodels/liborforwardmodel.{hpp,cpp} (v1.43).

References:

- Stefan Weber, 2005, *Efficient Calibration for Libor Market Models*
- Damiano Brigo, Fabio Mercurio, Massimo Morini, 2003, *Different Covariance
  Parameterizations of Libor Market Model and Joint Caps/Swaptions
  Calibration*

The model is both a ``CalibratedModel`` (its arguments are the concatenation of
the volatility model's and the correlation model's) and an ``AffineModel``
(``discount_bond_option`` prices a caplet exactly off the integrated
covariance). ``get_swaption_volatility_matrix`` builds an ATM swaption
volatility surface with Rebonato's approximation.

Handle indirection: C++ reads the forwarding curve through
``process->index()->forwardingTermStructure()``, an ``Handle``. This port
threads the term structure object directly.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.legacy.libormarketmodels.lfm_covar_proxy import LfmCovarianceProxy
from pquantlib.legacy.libormarketmodels.lfm_process import LiborForwardModelProcess
from pquantlib.legacy.libormarketmodels.lm_corr_model import LmCorrelationModel
from pquantlib.legacy.libormarketmodels.lm_vol_model import LmVolatilityModel
from pquantlib.math.array import Array
from pquantlib.models.model import AffineModel, CalibratedModel
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.black_formula import black_formula
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.termstructures.volatility.swaption.swaption_volatility_matrix import (
    SwaptionVolatilityMatrix,
)
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

# C++ uses 100 * numeric_limits<Real>::epsilon() to decide whether a requested
# caplet maturity coincides with a process accrual date.
_MATURITY_TOLERANCE = 100.0 * float(np.finfo(np.float64).eps)


class LiborForwardModel(CalibratedModel, AffineModel):
    """LIBOR forward model.

    # C++ parity: ``class LiborForwardModel : public CalibratedModel,
    # public AffineModel`` (liborforwardmodel.hpp:51-83).
    """

    def __init__(
        self,
        process: LiborForwardModelProcess,
        vola_model: LmVolatilityModel,
        corr_model: LmCorrelationModel,
    ) -> None:
        # C++ parity: liborforwardmodel.cpp:27-49.
        super().__init__(len(vola_model.params()) + len(corr_model.params()))
        self._covar_proxy: LfmCovarianceProxy = LfmCovarianceProxy(vola_model, corr_model)
        self._process: LiborForwardModelProcess = process
        self._swaption_vola: SwaptionVolatilityMatrix | None = None

        k = len(vola_model.params())
        for i, p in enumerate(vola_model.params()):
            self._arguments[i] = p
        for i, p in enumerate(corr_model.params()):
            self._arguments[k + i] = p

        size = process.size()
        self._accrual_period: list[float] = [0.0] * size
        self._f: list[float] = [0.0] * size
        initial_values = process.initial_values()
        for i in range(size):
            self._accrual_period[i] = (
                process.accrual_end_times()[i] - process.accrual_start_times()[i]
            )
            self._f[i] = 1.0 / (1.0 + self._accrual_period[i] * initial_values[i])

    # --- inspectors -------------------------------------------------------

    def covar_proxy(self) -> LfmCovarianceProxy:
        """The covariance proxy built from the volatility + correlation models.

        Python addition: C++ keeps ``covarProxy_`` protected with no accessor.
        """
        return self._covar_proxy

    def process(self) -> LiborForwardModelProcess:
        """The underlying LFM process.

        Python addition: C++ keeps ``process_`` protected with no accessor.
        """
        return self._process

    # --- Model interface --------------------------------------------------

    def generate_arguments(self) -> None:
        """No derived state on the model itself.

        # C++ parity: ``LiborForwardModel`` does not override
        # ``CalibratedModel::generateArguments`` (a no-op virtual); the
        # rewiring happens in ``setParams`` instead.
        """

    def set_params(self, params: npt.NDArray[np.float64]) -> None:
        """Push the flat parameter vector into the sub-models.

        # C++ parity: liborforwardmodel.cpp:51-62 — the first
        # ``volatilityModel()->params().size()`` arguments go to the
        # volatility model, the rest to the correlation model, and the cached
        # swaption volatility matrix is dropped.
        """
        super().set_params(params)

        k = len(self._covar_proxy.volatility_model().params())

        self._covar_proxy.volatility_model().set_params(self._arguments[:k])
        self._covar_proxy.correlation_model().set_params(self._arguments[k:])

        self._swaption_vola = None

    # --- AffineModel interface --------------------------------------------

    def discount(self, t: float) -> float:
        """# C++ parity: liborforwardmodel.cpp:203-205 — "meaningless within
        this context"; delegates to the forwarding curve.
        """
        return self._term_structure().discount(t)

    def discount_bond(
        self, now: float, maturity: float, factors: npt.NDArray[np.float64]
    ) -> float:
        """# C++ parity: liborforwardmodel.cpp:207-209 — ignores ``now`` and
        ``factors`` entirely, exactly as C++ does.
        """
        return self.discount(maturity)

    def discount_bond_option(
        self,
        option_type: OptionType,
        strike: float,
        maturity: float,
        bond_maturity: float,
    ) -> float:
        """Exact caplet/floorlet price via the Black formula.

        # C++ parity: liborforwardmodel.cpp:64-103. The bond-option strike
        ``strike`` is converted into a cap rate ``(1/strike - 1)/tenor``, the
        option type is FLIPPED (a bond put is a caplet), and the result is
        rescaled by ``1 / (1 + capRate * tenor)``.
        """
        accrual_start_times = self._process.accrual_start_times()
        accrual_end_times = self._process.accrual_end_times()

        qassert.require(
            accrual_start_times[0] <= maturity and accrual_start_times[-1] >= maturity,
            "capet maturity does not fit to the process",
        )

        # C++ std::lower_bound over accrualStartTimes
        i = int(np.searchsorted(np.asarray(accrual_start_times), maturity, side="left"))

        qassert.require(
            i < self._process.size()
            and abs(maturity - accrual_start_times[i]) < _MATURITY_TOLERANCE
            and abs(bond_maturity - accrual_end_times[i]) < _MATURITY_TOLERANCE,
            "irregular fixings are not (yet) supported",
        )

        tenor = accrual_end_times[i] - accrual_start_times[i]
        forward = float(self._process.initial_values()[i])
        cap_rate = (1.0 / strike - 1.0) / tenor
        var = self._covar_proxy.integrated_covariance_scalar(
            i, i, self._process.fixing_times()[i]
        )
        dis = self._term_structure().discount(bond_maturity)

        black = black_formula(
            OptionType.Call if option_type == OptionType.Put else OptionType.Put,
            cap_rate,
            forward,
            math.sqrt(var),
        )

        npv = dis * tenor * black

        return npv / (1.0 + cap_rate * tenor)

    # --- swap rates -------------------------------------------------------

    def _w_0(self, alpha: int, beta: int) -> Array:
        """Swap-rate weights of the forward rates alpha+1 .. beta.

        # C++ parity: ``LiborForwardModel::w_0`` (liborforwardmodel.cpp:105-128).
        """
        omega = np.zeros(beta + 1, dtype=np.float64)
        qassert.require(alpha < beta, "alpha needs to be smaller than beta")

        s = 0.0
        for k in range(alpha + 1, beta + 1):
            b = self._accrual_period[k]
            for j in range(alpha + 1, k + 1):
                b *= self._f[j]
            s += b

        for i in range(alpha + 1, beta + 1):
            a = self._accrual_period[i]
            for j in range(alpha + 1, i + 1):
                a *= self._f[j]
            omega[i] = a / s

        return omega

    def s_0(self, alpha: int, beta: int) -> float:
        """Forward swap rate implied by the model's initial forward curve.

        # C++ parity: ``LiborForwardModel::S_0`` (liborforwardmodel.cpp:130-140).
        """
        w = self._w_0(alpha, beta)
        f = self._process.initial_values()

        fwd_rate = 0.0
        for i in range(alpha + 1, beta + 1):
            fwd_rate += w[i] * f[i]

        return fwd_rate

    def get_swaption_volatility_matrix(self) -> SwaptionVolatilityMatrix:
        """ATM swaption volatility matrix via Rebonato's approximation.

        # C++ parity: ``LiborForwardModel::getSwaptionVolatilityMatrix``
        # (liborforwardmodel.cpp:142-199). Valid only for regular fixings and
        # assuming the fixed and floating legs share a frequency. The result is
        # cached in a mutable member and dropped by :meth:`set_params`.

        # C++ parity divergence: C++ constructs the matrix from a vector of
        # option DATES (``SwaptionVolatilityDiscrete``'s date constructor).
        # PQuantLib's ``SwaptionVolatilityDiscrete`` only exposes the
        # option-TENOR constructor, so each exercise date is passed as
        # ``Period(date - today, Days)``. With a ``NullCalendar`` — the
        # calendar C++ uses here — every day is a business day, so
        # ``optionDateFromTenor`` maps that tenor back onto exactly the same
        # date under any business-day convention, and hence onto the same
        # option times. The surface is therefore numerically identical.
        """
        if self._swaption_vola is not None:
            return self._swaption_vola

        index = self._process.index()
        today = self._process.fixing_dates()[0]

        size = self._process.size() // 2
        volatilities = np.zeros((size, size), dtype=np.float64)

        exercises = self._process.fixing_dates()[1 : size + 1]

        lengths = [(i + 1) * index.tenor() for i in range(size)]

        f = self._process.initial_values()
        for k in range(size):
            alpha = k
            t_alpha = self._process.fixing_times()[alpha + 1]

            var = np.zeros((size, size), dtype=np.float64)
            for i in range(alpha + 1, k + size + 1):
                for j in range(i, k + size + 1):
                    value = self._covar_proxy.integrated_covariance_scalar(i, j, t_alpha)
                    var[i - alpha - 1, j - alpha - 1] = value
                    var[j - alpha - 1, i - alpha - 1] = value

            for level in range(1, size + 1):
                beta = level + k
                w = self._w_0(alpha, beta)

                total = 0.0
                for i in range(alpha + 1, beta + 1):
                    for j in range(alpha + 1, beta + 1):
                        total += (
                            w[i] * w[j] * f[i] * f[j] * var[i - alpha - 1, j - alpha - 1]
                        )
                volatilities[k, level - 1] = math.sqrt(total / t_alpha) / self.s_0(
                    alpha, beta
                )

        option_tenors = [Period(d - today, TimeUnit.Days) for d in exercises]

        self._swaption_vola = SwaptionVolatilityMatrix(
            business_day_convention=BusinessDayConvention.Following,
            option_tenors=option_tenors,
            swap_tenors=lengths,
            volatilities=volatilities,
            calendar=NullCalendar(),
            day_counter=index.day_counter(),
            reference_date=today,
        )
        return self._swaption_vola

    # --- helpers ----------------------------------------------------------

    def _term_structure(self) -> YieldTermStructureProtocol:
        ts = self._process.index().forecast_term_structure()
        qassert.require(ts is not None, "null term structure set to this instance of the index")
        assert ts is not None
        return ts


__all__ = ["LiborForwardModel"]

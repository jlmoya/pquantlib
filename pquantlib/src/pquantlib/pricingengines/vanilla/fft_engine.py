"""FFTEngine — Carr-Madan FFT pricing engine base for European vanillas.

# C++ parity: ql/experimental/variancegamma/fftengine.{hpp,cpp} (v1.42.1).

The FFT engine values *all* options with the same expiry in a single FFT
pass, so individual pricing is inefficient — collect the options in a list
and call :meth:`precalculate` once before reading each option's NPV.

References
----------
Carr, P. and D. B. Madan (1998), "Option Valuation using the fast Fourier
transform," Journal of Computational Finance, 2, 61-73.

Algorithm (per expiry date, mirrors the C++ ``precalculate``):

1. Choose ``n`` (a power of two) large enough for the maximum strike.
2. Build the integrand grid ``fti[i]`` from the discounted characteristic
   function (the abstract :meth:`complex_fourier_transform`), damped by the
   Carr-Madan ``alpha = 1.25`` factor and Simpson-rule weights.
3. Run a forward FFT (``numpy.fft.fft`` — same negative-exponent convention
   as QuantLib's ``FastFourierTransform::transform``).
4. Recover call prices on the log-strike grid and linearly interpolate to
   each requested strike; put prices follow by put-call parity.

Python divergences:

* The C++ ``std::map<Date, std::map<shared_ptr<StrikedTypePayoff>, Real>>``
  cache becomes ``dict[int, dict[StrikedTypePayoff, float]]`` keyed by the
  expiry's serial number (value-based, matching ``std::map<Date>``) and the
  payoff object identity (matching ``shared_ptr`` pointer keys).
* QuantLib's ``FastFourierTransform`` is replaced by ``numpy.fft.fft`` —
  both compute the unnormalised ``sum_j in[j] exp(-2 pi i j k / n)``.
* ``clone()`` returns a fresh engine of the same concrete type (used by
  :meth:`calculate` to price a single uncached option).
"""

from __future__ import annotations

import math
from abc import abstractmethod

import numpy as np

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.option import OptionArguments
from pquantlib.payoffs import OptionType, StrikedTypePayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.stochastic_process_1d import StochasticProcess1D
from pquantlib.time.date import Date


class FFTEngine(GenericEngine[OptionArguments, OneAssetOptionResults]):
    """Abstract Carr-Madan FFT engine for European vanilla options.

    # C++ parity: ``class FFTEngine : public VanillaOption::engine`` in
    # fftengine.hpp:47.

    Concrete subclasses provide the per-process characteristic function
    plus discount / dividend accessors and ``clone()``.
    """

    def __init__(self, process: StochasticProcess1D, log_strike_spacing: float) -> None:
        super().__init__(OptionArguments(), OneAssetOptionResults())
        self._process: StochasticProcess1D = process
        self._lambda: float = float(log_strike_spacing)
        # date-serial -> {payoff -> price}
        self._result_map: dict[int, dict[StrikedTypePayoff, float]] = {}
        process.register_with(self)

    # --- abstract hooks (subclass per process) --------------------------

    @abstractmethod
    def clone(self) -> FFTEngine:
        """Return a fresh engine of the same concrete type.

        # C++ parity: ``FFTEngine::clone`` (pure virtual).
        """
        ...

    @abstractmethod
    def precalculate_expiry(self, d: Date) -> None:
        """Cache any per-expiry discount factors / variances.

        # C++ parity: ``FFTEngine::precalculateExpiry`` (pure virtual).
        """
        ...

    @abstractmethod
    def complex_fourier_transform(self, u: complex) -> complex:
        """Characteristic function ``phi(u)`` of log-spot at the expiry.

        # C++ parity: ``FFTEngine::complexFourierTransform`` (pure virtual).
        """
        ...

    @abstractmethod
    def discount_factor(self, d: Date) -> float:
        """Risk-free discount factor to date ``d``.

        # C++ parity: ``FFTEngine::discountFactor`` (pure virtual).
        """
        ...

    @abstractmethod
    def dividend_yield(self, d: Date) -> float:
        """Dividend discount factor to date ``d``.

        # C++ parity: ``FFTEngine::dividendYield`` (pure virtual).
        """
        ...

    # --- engine interface -----------------------------------------------

    def calculate(self) -> None:
        """Read the cached price, or price a single uncached option.

        # C++ parity: ``FFTEngine::calculate``.
        """
        args = self._arguments
        results = self._results

        qassert.require(args.exercise is not None, "no exercise given")
        qassert.require(args.payoff is not None, "no payoff given")
        assert args.exercise is not None
        assert args.payoff is not None

        qassert.require(
            args.exercise.type() == Exercise.Type.European,
            "not an European Option",
        )
        qassert.require(
            isinstance(args.payoff, StrikedTypePayoff),
            "non-striked payoff given",
        )
        assert isinstance(args.payoff, StrikedTypePayoff)
        payoff: StrikedTypePayoff = args.payoff

        key = args.exercise.last_date().serial_number()
        inner = self._result_map.get(key)
        if inner is not None and payoff in inner:
            results.value = inner[payoff]
            return

        # Not precalculated — do the whole FFT for this single option.
        self._calculate_uncached(payoff, args.exercise)

    def update(self) -> None:
        """Invalidate the cache then propagate.

        # C++ parity: ``FFTEngine::update`` — clears resultMap_ + base update.
        """
        self._result_map.clear()
        super().update()

    def _calculate_uncached(
        self, payoff: StrikedTypePayoff, exercise: Exercise
    ) -> None:
        """Price a single option by cloning the engine and precalculating.

        # C++ parity: ``FFTEngine::calculateUncached``.
        """
        option = VanillaOption(payoff, exercise)
        temp_engine = self.clone()
        temp_engine.precalculate([option])
        option.set_pricing_engine(temp_engine)
        self._results.value = option.npv()

    # --- the FFT itself --------------------------------------------------

    def precalculate(self, option_list: list[VanillaOption]) -> None:
        """Run the Carr-Madan FFT for every expiry in ``option_list``.

        # C++ parity: ``FFTEngine::precalculate``.
        """
        self._result_map.clear()

        # Group payoffs by expiry date.
        payoff_map: dict[int, tuple[Date, list[StrikedTypePayoff]]] = {}
        for opt in option_list:
            qassert.require(
                opt.exercise().type() == Exercise.Type.European,
                "not an European Option",
            )
            p = opt.payoff()
            qassert.require(isinstance(p, StrikedTypePayoff), "non-striked payoff given")
            assert isinstance(p, StrikedTypePayoff)
            d = opt.exercise().last_date()
            key = d.serial_number()
            if key not in payoff_map:
                payoff_map[key] = (d, [])
            payoff_map[key][1].append(p)

        i1 = 1j
        alpha = 1.25

        for _key, (expiry_date, payoffs) in payoff_map.items():
            # n large enough for the maximum strike, rounded up to 2^k.
            max_strike = max(po.strike() for po in payoffs)
            n_r = 2.0 * (math.log(max_strike) + self._lambda) / self._lambda
            log2_n = int(math.log(n_r) / math.log(2.0)) + 1
            n = 1 << log2_n

            # Strike range (eq. 19,20) + grid spacing (eq. 23).
            b = n * self._lambda / 2.0
            eta = 2.0 * math.pi / (self._lambda * n)

            df = self.discount_factor(expiry_date)
            div = self.dividend_yield(expiry_date)

            self.precalculate_expiry(expiry_date)

            fti = np.zeros(n, dtype=np.complex128)
            for i in range(n):
                v_j = eta * i
                # Simpson weights.
                sw = (
                    eta
                    * (3.0 + (-1.0 if (i % 2) == 0 else 1.0) - (1.0 if i == 0 else 0.0))
                    / 3.0
                )
                psi = df * self.complex_fourier_transform(v_j - (alpha + 1.0) * i1)
                psi = psi / (
                    alpha * alpha + alpha - v_j * v_j + i1 * (2.0 * alpha + 1.0) * v_j
                )
                fti[i] = np.exp(i1 * b * v_j) * sw * psi

            # Forward FFT — numpy.fft.fft matches QuantLib's
            # FastFourierTransform::transform (negative-exponent DFT).
            results_fft = np.fft.fft(fti)

            prices = np.zeros(n, dtype=np.float64)
            strikes = np.zeros(n, dtype=np.float64)
            for i in range(n):
                k_u = -b + self._lambda * i
                prices[i] = (math.exp(-alpha * k_u) / math.pi) * float(results_fft[i].real)
                strikes[i] = math.exp(k_u)

            interp = LinearInterpolation(strikes, prices)
            inner = self._result_map.setdefault(expiry_date.serial_number(), {})
            for po in payoffs:
                call_price = interp(po.strike(), allow_extrapolation=True)
                if po.option_type() == OptionType.Call:
                    inner[po] = call_price
                elif po.option_type() == OptionType.Put:
                    # Put-call parity: C - S0*div + K*df.
                    inner[po] = call_price - self._process.x0() * div + po.strike() * df
                else:
                    qassert.fail("Invalid option type")


__all__ = ["FFTEngine"]

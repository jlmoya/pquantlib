"""SmileSectionRNDCalculator — risk-neutral density implied by a smile.

# C++ parity: ql/methods/finitedifferences/utilities/smilesectionrndcalculator.{hpp,cpp}
# (new in v1.43).

Reads the risk-neutral terminal distribution of the underlying off a
:class:`~pquantlib.termstructures.volatility.smile_section.SmileSection`
via the Breeden-Litzenberger identity: the undiscounted digital call
price at strike ``K`` *is* ``P(S > K)``, and the density is its
derivative.

Every abscissa is in log-space: ``pdf`` and ``cdf`` take ``x = ln(S)``,
and ``invcdf`` returns ``ln(K)``.

Three behaviours are worth knowing before relying on this class, none of
them visible from the signatures:

* ``pdf`` and ``cdf`` evaluate the smile directly and **never build the
  quantile grid**, so ``n_strikes`` and ``n_std`` affect ``invcdf`` only.
* The grid is forced monotone with a running maximum and then
  deduplicated — points whose CDF gains at most 1e-12 on the previous
  kept one are dropped — so it is normally shorter than ``n_strikes``,
  and ``invcdf`` at extreme probabilities clamps onto the *surviving*
  endpoints rather than extrapolating.
* ``invcdf`` validates in the order exercise time → build grid → ``p``
  range. On a smile with no ATM level, ``invcdf(-1.0)`` therefore
  reports the missing ATM level, not the invalid probability.

The two finite-difference helpers it leans on take **different** default
gaps — ``density`` 1e-4, ``digital_option_price`` 1e-5 — and both are
inherited from C++'s defaults. A port that unifies them will not
reproduce the wings.

Like :class:`~pquantlib.methods.finitedifferences.utilities.gbsm_rnd_calculator.GBSMRNDCalculator`,
this has no ``RiskNeutralDensityCalculator`` base class: that C++
interface is a single header of three pure virtuals, and this port
implements the three methods directly.
"""

from __future__ import annotations

import math

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.closeness import close_enough
from pquantlib.math.constants import QL_EPSILON
from pquantlib.math.interpolations.cubic_interpolation import (
    MonotonicCubicNaturalSpline,
)
from pquantlib.payoffs import OptionType
from pquantlib.termstructures.volatility.smile_section import SmileSection

# C++ parity: the ``dedupTol`` constant in ``initialize()``.
_DEDUP_TOL = 1.0e-12


class SmileSectionRNDCalculator:
    """Risk-neutral terminal density / CDF / quantile implied by a smile.

    # C++ parity: ``class SmileSectionRNDCalculator``.

    Args:
        smile: the section the distribution is read off. It must supply an
            ATM level — wrap it in
            :class:`~pquantlib.termstructures.volatility.atm_smile_section.AtmSmileSection`
            if it does not. Only ``invcdf`` needs one.
        n_strikes: number of grid points laid out for the quantile
            function; at least 4.
        n_std: half-width of the strike grid, in ATM lognormal standard
            deviations; must be positive.
    """

    __slots__ = (
        "_cdf",
        "_initialized",
        "_n_std",
        "_n_strikes",
        "_quantile_fn",
        "_smile",
        "_strikes",
    )

    def __init__(
        self,
        smile: SmileSection,
        n_strikes: int = 200,
        n_std: float = 5.0,
    ) -> None:
        # The C++ constructor also rejects a null SmileSection. A typed
        # ``SmileSection`` parameter statically excludes None, so that check is
        # skipped here — pyright would flag it as ``reportUnnecessaryComparison``,
        # and the repo drops such guards elsewhere for the same reason.
        qassert.require(n_strikes >= 4, f"at least 4 strikes required, got {n_strikes}")
        qassert.require(n_std > 0.0, f"nStd must be positive, got {n_std}")
        self._smile: SmileSection = smile
        self._n_strikes: int = n_strikes
        self._n_std: float = n_std
        self._initialized: bool = False
        self._strikes: Array = np.empty(0, dtype=np.float64)
        self._cdf: Array = np.empty(0, dtype=np.float64)
        self._quantile_fn: MonotonicCubicNaturalSpline | None = None

    # --- internals ---------------------------------------------------------

    def _check_time(self, t: float) -> None:
        # C++ parity: ``SmileSectionRNDCalculator::checkTime``. ``close_enough``,
        # not ``==`` — but still not a way to reprice at another maturity.
        t_ref = self._smile.exercise_time()
        qassert.require(
            close_enough(t, t_ref),
            f"SmileSectionRNDCalculator: requested t={t} does not match "
            f"smile exercise time {t_ref}",
        )

    def _initialize(self) -> None:
        """Build the CDF grid and the monotone spline that inverts it.

        # C++ parity: ``SmileSectionRNDCalculator::initialize``. Lazy and
        # idempotent — the grid is a pure function of the smile and the two
        # constructor arguments.
        """
        if self._initialized:
            return

        forward = self._smile.atm_level()
        qassert.require(
            not math.isnan(forward),
            "SmileSectionRNDCalculator: smile.atm_level() returned null; "
            "wrap with AtmSmileSection to supply one",
        )

        t = self._smile.exercise_time()
        sigma_atm = self._smile.volatility(forward)
        log_std = sigma_atm * math.sqrt(t)
        k_min = max(forward * math.exp(-self._n_std * log_std), QL_EPSILON)
        k_max = forward * math.exp(self._n_std * log_std)

        strikes: list[float] = []
        cdf: list[float] = []
        last_cdf = -1.0
        for i in range(self._n_strikes):
            k = k_min + (k_max - k_min) * i / (self._n_strikes - 1)
            c = min(
                max(
                    1.0
                    - self._smile.digital_option_price(k, int(OptionType.Call), 1.0),
                    0.0,
                ),
                1.0,
            )
            # Breeden-Litzenberger output can dip in the wings, so take a
            # running maximum; then drop points that add nothing, because a
            # spline through a flat run is not invertible.
            c_mono = max(c, last_cdf)
            if c_mono - last_cdf > _DEDUP_TOL:
                strikes.append(k)
                cdf.append(c_mono)
                last_cdf = c_mono

        qassert.require(
            len(cdf) >= 4,
            f"SmileSectionRNDCalculator: too few unique CDF points ({len(cdf)}) "
            "after deduplication",
        )

        self._strikes = np.asarray(strikes, dtype=np.float64)
        self._cdf = np.asarray(cdf, dtype=np.float64)
        # Abscissae are the CDF values and ordinates the strikes — i.e. the
        # quantile function, not the distribution function.
        self._quantile_fn = MonotonicCubicNaturalSpline(self._cdf, self._strikes)
        self._initialized = True

    # --- the three RiskNeutralDensityCalculator methods --------------------

    def pdf(self, x: float, t: float | None = None) -> float:
        """Density of ``ln(S)`` at ``x``.

        # C++ parity: ``pdf(Real, Time)`` and the one-argument overload, which
        # simply passes the smile's own exercise time. Deliberately does *not*
        # touch the quantile grid.
        """
        t = self._smile.exercise_time() if t is None else t
        self._check_time(t)
        s = math.exp(x)
        return s * self._smile.density(s, 1.0)

    def cdf(self, x: float, t: float | None = None) -> float:
        """Cumulative probability that ``ln(S)`` is at or below ``x``.

        # C++ parity: ``cdf(Real, Time)`` plus the one-argument overload.
        """
        t = self._smile.exercise_time() if t is None else t
        self._check_time(t)
        s = math.exp(x)
        return 1.0 - self._smile.digital_option_price(s, int(OptionType.Call), 1.0)

    def invcdf(self, p: float, t: float | None = None) -> float:
        """Quantile of ``ln(S)`` at probability ``p``.

        # C++ parity: ``invcdf(Real, Time)`` plus the one-argument overload.
        # The order of the checks is load-bearing and mirrors C++: exercise
        # time first, then the grid is built — which is where a missing ATM
        # level is reported — and only then is ``p`` range-checked.
        """
        t = self._smile.exercise_time() if t is None else t
        self._check_time(t)
        self._initialize()
        qassert.require(0.0 < p < 1.0, f"p must be in (0, 1), got {p}")
        assert self._quantile_fn is not None
        p_clamped = min(max(p, float(self._cdf[0])), float(self._cdf[-1]))
        return math.log(self._quantile_fn(p_clamped))


__all__ = ["SmileSectionRNDCalculator"]

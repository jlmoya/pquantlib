"""AnalyticGjrGarchEngine — Edgeworth-expansion European-option engine.

# C++ parity: ql/pricingengines/vanilla/analyticgjrgarchengine.{hpp,cpp}
# (v1.42.1).

Approximates the European call price under GJR-GARCH dynamics by
projecting the GJR-GARCH log-return into a Black-Scholes-like form
with the first four moments (mean, variance, skewness, kurtosis) and
applying a Gram-Charlier / Edgeworth correction::

    C_approx = C_BS + k3 * A3 + (k4 - 3) * A4

where C_BS is the modified Black-Scholes call (with drift adjustment
``del = (ex - r*T + sigma/2)/stdev``), and ``A3``, ``A4`` are the
Edgeworth correction terms.

Reference: Duan, J.C., Gauthier, G., Simonato, J.G., Sasseville, C.,
2006. "Approximating the GJR-GARCH and EGARCH option pricing models
analytically", *Journal of Computational Finance*, 9(3), Spring 2006.

L11-W1-D scope: ports the Edgeworth-expansion formulas verbatim from
C++ analyticgjrgarchengine.cpp. The triple loop over T (number of
days to expiry) is preserved as a numpy/Python loop — the inner
moment accumulators contain mixed product+conditional updates that
don't vectorize cleanly without obfuscating the C++ correspondence.

Divergences from C++:
- The C++ engine caches intermediate constants (``b1_``, ``m1_`` etc.)
  via mutable members; calling ``calculate`` twice with the same
  parameters reuses the moments. Python recomputes each time
  (the cache logic is < 1% of total cost; preserving it adds
  significant complexity without measurable benefit).
- ``Real`` collapses to ``float`` throughout.
- Names ``m1``, ``m2``, ``v1``, ``v2``, ``z1``, ``x1``, ``del`` etc.
  are preserved verbatim for line-by-line C++ correspondence. ``del``
  is renamed ``delta_`` (Python keyword shadow). ``A3``, ``A4`` are
  upper-case because that's how they appear in the Duan paper.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import Exercise
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.models.equity.gjr_garch_model import GjrGarchModel
from pquantlib.option import OptionArguments
from pquantlib.payoffs import OptionType, StrikedTypePayoff
from pquantlib.pricingengines.generic_engine import GenericEngine


class AnalyticGjrGarchEngine(GenericEngine[OptionArguments, OneAssetOptionResults]):
    """European-option engine for GJR-GARCH via Edgeworth expansion.

    # C++ parity: ``class AnalyticGJRGARCHEngine`` in
    # analyticgjrgarchengine.hpp:52-83 (v1.42.1).
    """

    def __init__(
        self,
        model: GjrGarchModel,
        integration_order: int = 144,
    ) -> None:
        """Construct from a ``GjrGarchModel``.

        # C++ parity: ``AnalyticGJRGARCHEngine`` ctor in
        # analyticgjrgarchengine.cpp:32-36.

        Parameters
        ----------
        model:
            The GJR-GARCH model whose parameters drive the Edgeworth
            expansion.
        integration_order:
            Unused — kept as a kwarg for API parity with sibling
            engines (the GJR-GARCH engine is closed-form, no
            integration). Defaults to 144 to match
            ``AnalyticHestonEngine`` / ``AnalyticPTDHestonEngine``.
        """
        super().__init__(OptionArguments(), OneAssetOptionResults())
        self._model: GjrGarchModel = model
        # C++ parity: analyticgjrgarchengine.hpp:65 — register with
        # model so the engine invalidates when parameters change.
        model.register_with(self)

    # --- inspectors -----------------------------------------------------

    def model(self) -> GjrGarchModel:
        """The underlying GjrGarchModel."""
        return self._model

    # --- helpers --------------------------------------------------------

    def _compute_coefficients(
        self,
        b1: float,
        b2: float,
        b3: float,
        la: float,
    ) -> tuple[float, float, float, float, float, float, float, float, float]:
        """Return (m1, m2, m3, v1, v2, v3, z1, z2, x1) — recursion constants.

        # C++ parity: analyticgjrgarchengine.cpp:106-145.
        """
        n_cdf = CumulativeNormalDistribution()(la)
        n_pdf = math.exp(-la * la / 2.0) / (
            math.sqrt(math.pi) * math.sqrt(2.0)
        )

        # m1, m2, m3 — successive moments of the latent variance recursion.
        m1 = b1 + (b2 + b3 * n_cdf) * (1.0 + la * la) + b3 * la * n_pdf
        m2 = (
            b1 * b1
            + b2 * b2 * (math.pow(la, 4) + 6.0 * la * la + 3.0)
            + (b3 * b3 + 2.0 * b2 * b3)
            * (
                math.pow(la, 4) * n_cdf
                + math.pow(la, 3) * n_pdf
                + 6.0 * la * la * n_cdf
                + 5.0 * la * n_pdf
                + 3.0 * n_cdf
            )
            + 2.0 * b1 * b2 * (1.0 + la * la)
            + 2.0 * b3 * b1 * (la * la * n_cdf + la * n_pdf + n_cdf)
        )
        m3 = (
            math.pow(b1, 3)
            + (3.0 * b3 * b3 * b1 + 6.0 * b1 * b2 * b3)
            * (
                math.pow(la, 3) * n_pdf
                + 5.0 * la * n_pdf
                + 3.0 * n_cdf
                + math.pow(la, 4) * n_cdf
                + 6.0 * la * la * n_cdf
            )
            + math.pow(b2, 3)
            * (15.0 + math.pow(la, 6) + 15.0 * math.pow(la, 4) + 45.0 * la * la)
            + (math.pow(b3, 3) + 3.0 * b2 * b2 * b3 + 3.0 * b3 * b3 * b2)
            * (
                math.pow(la, 5) * n_pdf
                + 14.0 * math.pow(la, 3) * n_pdf
                + 33.0 * la * n_pdf
                + 15.0 * n_cdf
                + 15.0 * math.pow(la, 4) * n_cdf
                + 45.0 * la * la * n_cdf
                + math.pow(la, 6) * n_cdf
            )
            + 3.0 * b1 * b1 * b2 * (1.0 + la * la)
            + 3.0 * b1 * b1 * b3 * (la * n_pdf + n_cdf + la * la * n_cdf)
            + 3.0 * b1 * b2 * b2 * (3.0 + math.pow(la, 4) + 6.0 * la * la)
        )

        v1 = -2.0 * b2 * la - 2.0 * b3 * (n_pdf + la * n_cdf)
        v2 = (
            -4.0 * b2 * b2 * (3.0 * la + math.pow(la, 3))
            - (4.0 * b3 * b3 + 8.0 * b2 * b3)
            * (la * la * n_pdf + 2.0 * n_pdf + math.pow(la, 3) * n_cdf + 3.0 * la * n_cdf)
            - 4.0 * b1 * b2 * la
            - 4.0 * b3 * b1 * (n_pdf + la * n_cdf)
        )
        v3 = (
            -12.0 * b3 * b1 * (b3 + 2.0 * b2)
            * (la * la * n_pdf + 2.0 * n_pdf + math.pow(la, 3) * n_cdf + 3.0 * la * n_cdf)
            - 6.0 * math.pow(b2, 3) * la * (15.0 + math.pow(la, 4) + 10.0 * la * la)
            - 6.0 * b3 * (b3 * b3 + 3.0 * b2 * b2 + 3.0 * b3 * b2)
            * (
                9.0 * la * la * n_pdf
                + 8.0 * n_pdf
                + 15.0 * la * n_cdf
                + math.pow(la, 4) * n_pdf
                + math.pow(la, 5) * n_cdf
                + 10.0 * math.pow(la, 3) * n_cdf
            )
            - 6.0 * b1 * b1 * b2 * la
            - 6.0 * b3 * b1 * b1 * (n_pdf + la * n_cdf)
            - 12.0 * b2 * b2 * b1 * (3.0 * la + math.pow(la, 3))
        )

        z1 = b1 + b2 * (3.0 + la * la) + b3 * (la * n_pdf + 3.0 * n_cdf + la * la * n_cdf)
        z2 = (
            b1 * b1
            + b2 * b2 * (15.0 + math.pow(la, 4) + 18.0 * la * la)
            + (b3 * b3 + 2.0 * b2 * b3)
            * (
                math.pow(la, 3) * n_pdf
                + 17.0 * la * n_pdf
                + 15.0 * n_cdf
                + math.pow(la, 4) * n_cdf
                + 18.0 * la * la * n_cdf
            )
            + 2.0 * b1 * b2 * (3.0 + la * la)
            + 2.0 * b3 * b1 * (la * n_pdf + 3.0 * n_cdf + la * la * n_cdf)
        )

        x1 = -6.0 * b2 * la - 2.0 * b3 * (4.0 * n_pdf + 3.0 * la * n_cdf)

        return m1, m2, m3, v1, v2, v3, z1, z2, x1

    def _compute_moments(  # noqa: PLR0915 - mirror C++ structure for parity
        self,
        b0: float,
        h1: float,
        b1: float,
        b2: float,
        b3: float,
        la: float,
        r: float,
        big_t: int,
    ) -> tuple[float, float, float, float]:
        """Return (ex, sigma, k3, k4) — first four moments of log-return.

        # C++ parity: analyticgjrgarchengine.cpp:147-258.
        """
        m1, m2, m3, v1, v2, _v3, z1, _z2, x1 = self._compute_coefficients(b1, b2, b3, la)

        # Precompute m1^i, m2^i, m3^i for i in [0, T).
        m1ai = [1.0] * big_t
        m2ai = [1.0] * big_t
        m3ai = [1.0] * big_t
        for i in range(1, big_t):
            m1ai[i] = m1ai[i - 1] * m1
            m2ai[i] = m2ai[i - 1] * m2
            m3ai[i] = m3ai[i - 1] * m3

        # Accumulators for the moment sums.
        s_eh = 0.0
        s_eh2 = 0.0
        s_eh3 = 0.0
        s_ehh = 0.0
        s_eh1_2eh = 0.0
        s_ehh2 = 0.0
        s_eh2h = 0.0
        s_eh1_2eh2 = 0.0
        s_eh3_2eh = 0.0
        s_ehe2h = 0.0
        s_eh3_2e3h = 0.0
        s_ehhh = 0.0
        s_eh1_2ehh = 0.0
        s_ehh1_2eh = 0.0
        s_eh1_2eh1_2eh = 0.0

        for i in range(big_t):
            m1i = m1ai[i]
            m2i = m2ai[i]
            m3i = m3ai[i]
            m1im2i = m1i - m2i
            m1im3i = m1i - m3i
            m2im3i = m2i - m3i

            eh = b0 * (1.0 - m1i) / (1.0 - m1) + m1i * h1
            eh2 = (
                b0 * b0 * (
                    (1.0 + m1) * (1.0 - m2i) / (1.0 - m2)
                    - 2.0 * m1 * m1im2i / (m1 - m2)
                ) / (1.0 - m1)
                + 2.0 * b0 * m1 * m1im2i * h1 / (m1 - m2)
                + m2i * h1 * h1
            )
            eh3 = (
                math.pow(b0, 3) * (
                    (1.0 - m3i) / (1.0 - m3)
                    + 3.0 * m2 * ((1.0 - m3i) / (1.0 - m3) - m2im3i / (m2 - m3)) / (1.0 - m2)
                    + 3.0 * m1 * ((1.0 - m3i) / (1.0 - m3) - m1im3i / (m1 - m3)) / (1.0 - m1)
                    + 6.0 * m1 * m2 * (
                        ((1.0 - m3i) / (1.0 - m3) - m2im3i / (m2 - m3)) / (1.0 - m2)
                        + (m2im3i / (m2 - m3) - m1im3i / (m1 - m3)) / (m1 - m2)
                    ) / (1.0 - m1)
                )
                + 3.0 * b0 * b0 * m1 * h1 * (
                    m1im3i / (m1 - m3)
                    + 2.0 * m2 * (m1im3i / (m1 - m3) - m2im3i / (m2 - m3)) / (m1 - m2)
                )
                + 3.0 * b0 * m2 * h1 * h1 * m2im3i / (m2 - m3)
                + m3i * h1 * h1 * h1
            )

            eh3_2 = 0.375 * math.pow(eh, -0.5) * eh2 + 0.625 * math.pow(eh, 1.5)
            eh5_2 = 1.875 * math.pow(eh, 0.5) * eh2 - 0.875 * math.pow(eh, 2.5)

            s_eh += eh
            s_eh2 += eh2
            s_eh3 += eh3

            for j in range(big_t - i - 1):
                ehh = b0 * eh * (1.0 - m1ai[j + 1]) / (1.0 - m1) + eh2 * m1ai[j + 1]
                ehh2 = (
                    b0 * b0 * eh * (
                        (1.0 + m1) * (1.0 - m2ai[j + 1]) / (1.0 - m2)
                        - 2.0 * m1 * (m1ai[j + 1] - m2ai[j + 1]) / (m1 - m2)
                    ) / (1.0 - m1)
                    + 2.0 * b0 * m1 * eh2 * (m1ai[j + 1] - m2ai[j + 1]) / (m1 - m2)
                    + m2ai[j + 1] * eh3
                )
                eh2h = (
                    b0 * eh2 * (1.0 - m1ai[j + 1]) / (1.0 - m1)
                    + m1ai[j + 1] * eh3
                )
                eh1_2eh = v1 * m1ai[j] * eh3_2
                eh1_2eh2 = (
                    2.0 * b0 * v1 * (m1ai[j + 1] - m2ai[j + 1]) * eh3_2 / (m1 - m2)
                    + v2 * m2ai[j] * eh5_2
                )
                ehij = b0 * (1.0 - m1ai[i + j + 1]) / (1.0 - m1) + m1ai[i + j + 1] * h1
                ehh3_2 = (
                    0.375 * ehh2 / math.sqrt(ehij)
                    + 0.75 * math.sqrt(ehij) * ehh
                    - 0.125 * math.pow(ehij, 1.5) * eh
                )
                eh3_2eh = v1 * m1ai[j] * eh5_2
                eh3_2e3h = x1 * m1ai[j] * eh5_2
                eh1_2eh3_2 = (
                    0.375 * eh1_2eh2 / math.sqrt(ehij) + 0.75 * math.sqrt(ehij) * eh1_2eh
                )

                s_ehh += ehh
                s_eh1_2eh += eh1_2eh
                s_ehh2 += ehh2
                s_eh2h += eh2h
                s_eh1_2eh2 += eh1_2eh2
                s_eh3_2eh += eh3_2eh
                s_ehe2h += (
                    b0 * eh * (1.0 - m1ai[j + 1]) / (1.0 - m1)
                    + z1 * m1ai[j] * eh2
                )
                s_eh3_2e3h += eh3_2e3h

                for k in range(big_t - i - j - 2):
                    ehhh = (
                        b0 * ehh * (1.0 - m1ai[k + 1]) / (1.0 - m1)
                        + m1ai[k + 1] * ehh2
                    )
                    eh1_2ehh = (
                        b0 * eh1_2eh * (1.0 - m1ai[k + 1]) / (1.0 - m1)
                        + m1ai[k + 1] * eh1_2eh2
                    )
                    s_ehhh += ehhh
                    s_eh1_2ehh += eh1_2ehh
                    s_ehh1_2eh += v1 * m1ai[k] * ehh3_2
                    s_eh1_2eh1_2eh += v1 * m1ai[k] * eh1_2eh3_2

        # Final moments — verbatim from gjrgarchengine.cpp:228-256.
        ex = big_t * r - 0.5 * s_eh
        sd1 = 2.0 * s_ehh + s_eh2
        sd2 = s_eh
        sd3 = s_eh1_2eh
        ex2 = (
            big_t * big_t * r * r
            - big_t * r * s_eh
            + 0.25 * sd1
            + sd2
            - sd3
        )
        st1 = 6.0 * s_ehhh + 3.0 * s_ehh2 + 3.0 * s_eh2h + s_eh3
        st2 = 3.0 * s_eh1_2eh
        st3 = (
            2.0 * s_ehh1_2eh + 2.0 * s_eh1_2ehh + 2.0 * s_eh3_2eh + s_eh1_2eh2
        )
        st4 = s_ehe2h + s_ehh + s_eh2 + 2.0 * s_eh1_2eh1_2eh
        ex3 = (
            math.pow(big_t * r, 3)
            - 1.5 * big_t * big_t * r * r * s_eh
            + 3.0 * big_t * r * (sd1 / 4.0 + sd2 - sd3)
            + (st2 - st1 / 8.0 + 3.0 * st3 / 4.0 - 3.0 * st4 / 2.0)
        )
        sq2 = 6.0 * s_ehe2h + 12.0 * s_eh1_2eh1_2eh + 3.0 * s_eh2
        sq4 = 2.0 * s_ehhh + 2.0 * s_ehh2
        sq5 = (
            3.0 * s_ehh1_2eh + 3.0 * s_eh1_2ehh + 3.0 * s_eh3_2eh
            + 3.0 * s_eh1_2eh2 + s_eh3_2e3h
        )
        ex4 = (
            math.pow(big_t * r, 4)
            - 2.0 * math.pow(big_t * r, 3) * s_eh
            + 6.0 * big_t * big_t * r * r * (sd1 / 4.0 + sd2 - sd3)
            + big_t * r * (4.0 * st2 - st1 / 2.0 + 3.0 * st3 - 6.0 * st4)
            + (sq2 + 3.0 * sq4 / 2.0 - 2.0 * sq5)
        )

        sigma = ex2 - ex * ex
        k3 = ex3 - 3.0 * sigma * ex - ex * ex * ex
        k4 = ex4 + 6.0 * ex * ex * ex2 - 3.0 * ex * ex * ex * ex - 4.0 * ex * ex3
        k3 /= math.pow(sigma, 1.5)
        k4 /= math.pow(sigma, 2)
        return ex, sigma, k3, k4

    # --- calculate ------------------------------------------------------

    def calculate(self) -> None:
        """Run the engine: fill ``results.value`` for a European vanilla.

        # C++ parity: ``AnalyticGJRGARCHEngine::calculate`` in
        # analyticgjrgarchengine.cpp:38-298.
        """
        args = self._arguments
        results = self._results

        qassert.require(args.exercise is not None, "no exercise given")
        qassert.require(args.payoff is not None, "no payoff given")
        assert args.exercise is not None
        assert args.payoff is not None
        qassert.require(
            args.exercise.type() == Exercise.Type.European,
            "not an European option",
        )
        qassert.require(
            isinstance(args.payoff, StrikedTypePayoff), "non-striked payoff given"
        )
        assert isinstance(args.payoff, StrikedTypePayoff)
        payoff: StrikedTypePayoff = args.payoff

        process = self._model.process()

        risk_free_discount = process.risk_free_rate().discount(args.exercise.last_date())
        dividend_discount = process.dividend_yield().discount(args.exercise.last_date())

        spot_price = process.s0().value()
        qassert.require(spot_price > 0.0, "negative or null underlying given")

        strike_price = payoff.strike()
        term = process.time(args.exercise.last_date())
        # T — number of days; round to nearest int.
        big_t = round(process.days_per_year * term)
        # r — per-day risk-neutral drift.
        r = -math.log(risk_free_discount / dividend_discount) / (
            process.days_per_year * term
        )
        h1 = process.v0
        b0 = process.omega
        b2 = process.alpha
        b1 = process.beta
        b3 = process.gamma
        la = process.lambda_

        ex, sigma, k3, k4 = self._compute_moments(b0, h1, b1, b2, b3, la, r, big_t)

        # Compute call option price.
        s = spot_price
        x = strike_price
        stdev = math.sqrt(sigma)
        delta_ = (ex - r * big_t + sigma / 2.0) / stdev
        d = (math.log(s / x) + (r * big_t + sigma / 2.0)) / stdev
        d_ = d + delta_
        cum_norm = CumulativeNormalDistribution()
        c_bs = (
            s * math.exp(delta_ * stdev) * cum_norm(d_)
            - x * math.exp(-r * big_t) * cum_norm(d_ - stdev)
        )
        a3 = (
            s * math.exp(delta_ * stdev) * stdev
            * (
                (2.0 * stdev - d_) * math.exp(-d_ * d_ / 2.0) / math.sqrt(2.0 * math.pi)
                + sigma * cum_norm(d_)
            )
            / 6.0
        )
        a4 = (
            s * math.exp(delta_ * stdev) * stdev
            * (
                (d_ * d_ - 1.0 - 3.0 * stdev * (d_ - stdev))
                * math.exp(-d_ * d_ / 2.0) / math.sqrt(2.0 * math.pi)
                - sigma * stdev * cum_norm(d_)
            )
            / 24.0
        )
        c_approx = c_bs + k3 * a3 + (k4 - 3.0) * a4

        results.reset()
        if payoff.option_type() == OptionType.Call:
            results.value = c_approx
        elif payoff.option_type() == OptionType.Put:
            # Put-call parity (C++ form):
            # P = C + K * Df_rf / Df_div - S
            results.value = (
                c_approx + strike_price * risk_free_discount / dividend_discount - spot_price
            )
        else:
            raise LibraryException(f"unknown option type: {payoff.option_type()}")

    def update(self) -> None:
        """Observer.update — model parameters or curves changed."""
        self.notify_observers()


__all__ = ["AnalyticGjrGarchEngine"]

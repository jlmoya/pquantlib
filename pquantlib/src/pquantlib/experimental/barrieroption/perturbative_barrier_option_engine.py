"""PerturbativeBarrierOptionEngine — perturbative up-and-out barrier engine.

# C++ parity: ql/experimental/barrieroption/perturbativebarrieroptionengine.{hpp,cpp}
# (v1.43).

Implements the approach of L. Fatone, M. C. Recchioni and F. Zirilli
(<http://www.econ.univpm.it/recchioni/finance/w3/>): the up-and-out put price
is expanded to order 0, 1 or 2 in the time dependence of the drift and
variance. The C++ header warns "This was reported to fail tests on Mac OS X
10.8.4".

The C++ source is a near-literal Fortran transliteration carrying Alan Genz's
bivariate/trivariate normal routines and M. C. Recchioni's Hart/Miller normal
CDF. It is reproduced here function-for-function, including:

* the private ``PHID`` normal CDF rather than QuantLib's own
  ``CumulativeNormalDistribution`` (they differ in the last few ulp, and the
  published reference values are PHID's);
* file-scope mutable state (``H1``/``H2``/``H3``/``R23``/``RUA``/``RUB``/
  ``AR``/``RUC``/``NUC``) that ``tvtl`` writes and the Kronrod integrand
  reads — modelled here as a module-level state object, see
  :data:`_TVTL_STATE`;
* one-based array indexing (``limit[1..3]``, ``sigmarho[1..3]``, ``XGK[1..12]``),
  kept so the numerics can be diffed line-by-line against the C++;
* two transcription defects in ``_pntgnd`` — see the parity notes there.

Performance note: order 2 runs a 1000 x 100 quadrature whose integrand calls
the adaptive trivariate normal a dozen times per point. The C++ test suite
comments its own order-2 assertion out as "Too slow, skip"
(test-suite/barrieroption.cpp:1355-1373); the Python port is slower still, so
order 2 is implemented and reachable but is not exercised by the test suite.
The functions the second-order term composes are cross-validated one at a
time instead.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Final

from pquantlib import qassert
from pquantlib.instruments.barrier_option import (
    BarrierOptionArguments,
    BarrierType,
)
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.time.compounding import Compounding

# # C++ parity: ``#define PI 3.14159265358979324`` and the repeated
# # ``static double ppi = 3.14159265358979324`` locals. Both round to the same
# # double as ``math.pi``; kept as the literal the C++ writes.
_PI: Final[float] = 3.14159265358979324
_HALF_PI: Final[float] = 1.57079632679489661923132169163975


# ---------------------------------------------------------------------------
# C++ arithmetic helpers
# ---------------------------------------------------------------------------


def _sign(a: float, b: float) -> float:
    """# C++ parity: the file-local ``SIGN`` (perturbativebarrieroptionengine.cpp:36-42).

    Note ``b == 0.0`` takes the negative branch.
    """
    return math.fabs(a) if b > 0.0 else -math.fabs(a)


def _pow(base: float, exponent: float) -> float:
    """``std::pow`` semantics.

    Python's ``**`` returns a *complex* for a negative base with a fractional
    exponent, where C++ returns NaN; ``math.pow`` raises instead. This helper
    restores the C++/IEEE-754 result so the transliterated code can reach the
    same branches.
    """
    if base < 0.0 and exponent != math.floor(exponent):
        return math.nan
    if base == 0.0 and exponent < 0.0:
        return math.inf
    try:
        return math.pow(base, exponent)
    except OverflowError:  # pragma: no cover - defensive, C++ returns inf
        return math.inf


def _fdiv(a: float, b: float) -> float:
    """IEEE-754 division: ``x/0`` is +/-inf, ``0/0`` is NaN (C++ semantics)."""
    if b == 0.0:
        if a == 0.0 or math.isnan(a):
            return math.nan
        return math.copysign(math.inf, a) * math.copysign(1.0, b)
    return a / b


# ---------------------------------------------------------------------------
# Normal CDF (Hart / Miller algorithm 5666, via M. C. Recchioni)
# ---------------------------------------------------------------------------


def _phid(z: float) -> float:
    """Standard normal CDF, accurate to ~1e-15.

    # C++ parity: ``PHID`` (perturbativebarrieroptionengine.cpp:308-370).

    # C++ parity note: this is *not* QuantLib's ``CumulativeNormalDistribution``.
    # The engine ships its own so its published reference values reproduce;
    # they differ in the last ulp or two.
    """
    p0 = 220.2068679123761
    p1 = 221.2135961699311
    p2 = 112.0792914978709
    p3 = 33.91286607838300
    p4 = 6.373962203531650
    p5 = 0.7003830644436881
    p6 = 0.03526249659989109

    q0 = 440.4137358247522
    q1 = 793.8265125199484
    q2 = 637.3336333788311
    q3 = 296.5642487796737
    q4 = 86.78073220294608
    q5 = 16.064177579206950
    q6 = 1.7556671631826420
    q7 = 0.088388347648318440
    rootpi = 2.506628274631001
    cutoff = 7.071067811865475

    zabs = math.fabs(z)
    if zabs > 37:
        p = 0.0
    else:
        expntl = math.exp(-zabs * zabs / 2)
        if zabs < cutoff:
            p = (
                expntl
                * (
                    (((((p6 * zabs + p5) * zabs + p4) * zabs + p3) * zabs + p2) * zabs + p1)
                    * zabs
                    + p0
                )
                / (
                    (
                        (((((q7 * zabs + q6) * zabs + q5) * zabs + q4) * zabs + q3) * zabs + q2)
                        * zabs
                        + q1
                    )
                    * zabs
                    + q0
                )
            )
        else:
            p = expntl / (zabs + 1 / (zabs + 2 / (zabs + 3 / (zabs + 4 / (zabs + 0.65))))) / rootpi
    if z > 0:
        p = 1 - p
    return p


# ---------------------------------------------------------------------------
# Bivariate normal (Drezner / Wesolowsky, via A. Genz)
# ---------------------------------------------------------------------------

# Gauss-Legendre points and weights for N = 6, 12, 20, one-based on both axes.
_WL: Final[tuple[tuple[float, ...], ...]] = (
    (0.0, 0.1713244923791705, 0.4717533638651177e-01, 0.1761400713915212e-01),
    (0.0, 0.3607615730481384, 0.1069393259953183, 0.4060142980038694e-01),
    (0.0, 0.4679139345726904, 0.1600783285433464, 0.6267204833410906e-01),
    (0.0, 0.0, 0.2031674267230659, 0.8327674157670475e-01),
    (0.0, 0.0, 0.2334925365383547, 0.1019301198172404),
    (0.0, 0.0, 0.2491470458134029, 0.1181945319615184),
    (0.0, 0.0, 0.0, 0.1316886384491766),
    (0.0, 0.0, 0.0, 0.1420961093183821),
    (0.0, 0.0, 0.0, 0.1491729864726037),
    (0.0, 0.0, 0.0, 0.1527533871307259),
)
_XL: Final[tuple[tuple[float, ...], ...]] = (
    (0.0, -0.9324695142031522, -0.9815606342467191, -0.9931285991850949),
    (0.0, -0.6612093864662647, -0.9041172563704750, -0.9639719272779138),
    (0.0, -0.2386191860831970, -0.7699026741943050, -0.9122344282513259),
    (0.0, 0.0, -0.5873179542866171, -0.8391169718222188),
    (0.0, 0.0, -0.3678314989981802, -0.7463319064601508),
    (0.0, 0.0, -0.1252334085114692, -0.6360536807265150),
    (0.0, 0.0, 0.0, -0.5108670019508271),
    (0.0, 0.0, 0.0, -0.3737060887154196),
    (0.0, 0.0, 0.0, -0.2277858511416451),
    (0.0, 0.0, 0.0, -0.7652652113349733e-01),
)


def _nd2(a: float, b: float, rho: float) -> float:  # noqa: PLR0915 — transliteration
    """P(X > a, Y > b) for a standard bivariate normal with correlation ``rho``.

    # C++ parity: ``ND2`` (perturbativebarrieroptionengine.cpp:1280-1451).
    """
    twopi = 6.283185307179586
    r = rho
    dh = a
    dk = b

    if math.fabs(r) < 0.3:
        ng = 1
        lg = 3
    elif math.fabs(r) < 0.75:
        ng = 2
        lg = 6
    else:
        ng = 3
        lg = 10

    h = dh
    k = dk
    hk = h * k
    bvn = 0.0

    if math.fabs(r) < 0.925:
        if math.fabs(r) > 0:
            hs = (h * h + k * k) / 2
            asr = math.asin(r)
            for i in range(1, lg + 1):
                for is_ in (-1, 1):
                    sn = math.sin(asr * (is_ * _XL[i - 1][ng] + 1) / 2)
                    bvn = bvn + _WL[i - 1][ng] * math.exp((sn * hk - hs) / (1.0 - sn * sn))
            bvn = bvn * asr / (2 * twopi)
        bvn = bvn + _phid(-h) * _phid(-k)
    else:
        if r < 0:
            k = -k
            hk = -hk
        if math.fabs(r) < 1:
            as_ = (1 - r) * (1 + r)
            aa = _pow(as_, 0.5)
            bs = _pow(h - k, 2)
            c = (4 - hk) / 8
            d = (12 - hk) / 16
            asr = -(bs / as_ + hk) / 2
            if asr > -100:
                bvn = (
                    aa
                    * math.exp(asr)
                    * (1 - c * (bs - as_) * (1 - d * bs / 5) / 3 + c * d * as_ * as_ / 5)
                )
            if -hk < 100:
                bb = _pow(bs, 0.5)
                bvn = bvn - math.exp(-hk / 2) * _pow(twopi, 0.5) * _phid(-bb / aa) * bb * (
                    1 - c * bs * (1 - d * bs / 5) / 3
                )
            aa = aa / 2
            for i in range(1, lg + 1):
                for is_ in (-1, 1):
                    xs = _pow(aa * (is_ * _XL[i - 1][ng] + 1), 2)
                    rs = _pow(1 - xs, 2)
                    asr = -(bs / xs + hk) / 2
                    if asr > -100:
                        bvn = bvn + aa * _WL[i - 1][ng] * math.exp(asr) * (
                            math.exp(-hk * (1 - rs) / (2 * (1 + rs))) / rs
                            - (1 + c * xs * (1 + d * xs))
                        )
            bvn = -bvn / twopi
        if r > 0:
            bvn = bvn + _phid(-max(h, k))
        else:
            bvn = -bvn
            if k > h:
                bvn = bvn + _phid(k) - _phid(h)
    return bvn


# ---------------------------------------------------------------------------
# First-order helper functions
# ---------------------------------------------------------------------------


def _ff(p: float, tt: float, a: float, b: float, gm: float) -> float:
    """# C++ parity: ``ff`` — the function F(p,tt,a,b,gm) (.cpp:386-399)."""
    aa = -(b * p - b * tt + a) / _pow(2.0 * (tt - p), 0.5)
    caux = 2.0 * _pow(_PI, 0.5) * _phid(aa)
    aa = b * b - (1.0 - gm) * (1.0 - gm)
    aa = aa / 4.0
    return math.exp(-0.5 * a * b) * math.exp(aa * (tt - p)) * caux


def _v(p: float, tt: float, a: float, b: float, gm: float) -> float:
    """# C++ parity: ``v`` — the function E(p,tt,a,b,gm) (.cpp:406-418)."""
    aa = -(p * (a - b) + b * tt) / _pow(2.0 * p * tt * (tt - p), 0.5)
    caux = _phid(aa)
    aa = (
        math.exp(_pow(a - b, 2) / (4.0 * tt))
        * math.exp(_pow(1.0 - gm, 2) * tt / 4.0)
        * _pow(tt, 0.5)
    )
    return caux / aa


def _llold(p: float, tt: float, a: float, b: float, c: float, gm: float) -> float:
    """# C++ parity: ``llold`` — the function L(p,tt,a,b,c,gm) (.cpp:425-440)."""
    xx = (-a + b * (tt - p)) / _pow(2.0 * (tt - p), 0.5)
    yy = (-a + b * tt + c) / _pow(2.0 * tt, 0.5)
    rho = _pow((tt - p) / tt, 0.5)
    aa = b * b - (1.0 - gm) * (1.0 - gm)
    aa = aa / 4.0
    caux = _nd2(-xx, -yy, rho)
    return 2.0 * _pow(_PI, 0.5) * math.exp(-a * b * 0.5) * math.exp(aa * (tt - p)) * caux


# ---------------------------------------------------------------------------
# Second-order helper functions
# ---------------------------------------------------------------------------


def _dvv(s: float, p: float, tt: float, a: float, b: float, gm: float) -> float:
    """# C++ parity: ``dvv`` — D_E(s,p,tt,a,b,gm) (.cpp:455-484)."""
    aa = (a * p + b * (tt - p)) / _pow(2.0 * p * tt * (tt - p), 0.5)
    caux = _phid(aa)
    aa = (
        math.exp((a - b) * (a - b) / (4.0 * tt))
        * math.exp(_pow(1.0 - gm, 2) * tt / 4.0)
        * _pow(tt, 0.5)
    )
    caux = -caux / aa

    xx = (a * p + b * (tt - p)) / _pow(2.0 * tt * p * (tt - p), 0.5)
    yy = (a * s + b * (tt - s)) / _pow(2.0 * tt * s * (tt - s), 0.5)
    rho = _pow((s * (tt - p)) / (p * (tt - s)), 0.5)
    caux1 = _nd2(-xx, -yy, rho) / aa

    aa = (
        math.exp((a + b) * (a + b) / (4.0 * tt))
        * math.exp(_pow(1.0 - gm, 2) * tt / 4.0)
        * _pow(tt, 0.5)
    )
    xx = (a * p - b * (tt - p)) / _pow(2.0 * tt * p * (tt - p), 0.5)
    yy = (a * s - b * (tt - s)) / _pow(2.0 * tt * s * (tt - s), 0.5)
    rho = _pow((s * (tt - p)) / (p * (tt - s)), 0.5)
    caux2 = _nd2(-xx, -yy, rho) / aa
    return (caux + caux1 + caux2) / (2.0 * _pow(_PI, 0.5))


def _dff(s: float, p: float, tt: float, a: float, b: float, gm: float) -> float:
    """# C++ parity: ``dff`` — D_F(s,p,tt,a,b,gm) (.cpp:491-516)."""
    xx = (a - b * (tt - p)) / _pow(2.0 * (tt - p), 0.5)
    caux = -_phid(xx) * math.exp(-0.5 * a * b)

    xx = (a + b * (tt - p)) / _pow(2.0 * (tt - p), 0.5)
    yy = (a + b * (tt - s)) / _pow(2.0 * (tt - s), 0.5)
    rho = _pow((tt - p) / (tt - s), 0.5)
    caux1 = math.exp(0.5 * a * b) * _nd2(-xx, -yy, rho)

    xx = (a - b * (tt - p)) / _pow(2.0 * (tt - p), 0.5)
    yy = (a - b * (tt - s)) / _pow(2.0 * (tt - s), 0.5)
    rho = _pow((tt - p) / (tt - s), 0.5)
    caux2 = math.exp(-0.5 * a * b) * _nd2(-xx, -yy, rho)

    aa = math.exp((b * b - (1.0 - gm) * (1.0 - gm)) * (tt - s) / 4.0)
    return (caux + caux1 + caux2) * aa


def _dll(s: float, p: float, tt: float, a: float, b: float, c: float, gm: float) -> float:
    """# C++ parity: ``dll`` — D_L(s,p,a,b,c,gm) (.cpp:524-555)."""
    epsi = 1.0e-12
    limit = [0.0, 0.0, 0.0, 0.0]
    sigmarho = [0.0, 0.0, 0.0, 0.0]

    limit[1] = (a + b * (tt - p)) / _pow(2.0 * (tt - p), 0.5)
    limit[2] = (a + b * (tt - s)) / _pow(2.0 * (tt - s), 0.5)
    limit[3] = (a + b * tt + c) / _pow(2.0 * tt, 0.5)
    sigmarho[1] = _pow((tt - p) / (tt - s), 0.5)
    sigmarho[2] = _pow((tt - p) / tt, 0.5)
    sigmarho[3] = _pow((tt - s) / tt, 0.5)
    caux = math.exp(0.5 * a * b) * _tvtl(0, limit, sigmarho, epsi)

    limit[1] = (a - b * (tt - p)) / _pow(2.0 * (tt - p), 0.5)
    limit[2] = (-a + b * (tt - s)) / _pow(2.0 * (tt - s), 0.5)
    limit[3] = (-a + b * tt + c) / _pow(2.0 * tt, 0.5)
    sigmarho[1] = -_pow((tt - p) / (tt - s), 0.5)
    sigmarho[2] = -_pow((tt - p) / tt, 0.5)
    sigmarho[3] = _pow((tt - s) / tt, 0.5)
    caux1 = -math.exp(-0.5 * a * b) * _tvtl(0, limit, sigmarho, epsi)

    aa = math.exp((b * b - (1.0 - gm) * (1.0 - gm)) * (tt - s) / 4.0)
    return (caux + caux1) * aa


def _ddff(s: float, p: float, tt: float, a: float, b: float, gm: float) -> float:
    """# C++ parity: ``ddff`` — d/da of D_F (.cpp:562-611)."""
    xx = (a - b * (tt - p)) / _pow(2.0 * (tt - p), 0.5)
    caux = _phid(xx) * math.exp(-0.5 * a * b)

    xx = (a + b * (tt - p)) / _pow(2.0 * (tt - p), 0.5)
    yy = (a + b * (tt - s)) / _pow(2.0 * (tt - s), 0.5)
    rho = _pow((tt - p) / (tt - s), 0.5)
    caux1 = math.exp(0.5 * a * b) * _nd2(-xx, -yy, rho)

    xx = (a - b * (tt - p)) / _pow(2.0 * (tt - p), 0.5)
    yy = (a - b * (tt - s)) / _pow(2.0 * (tt - s), 0.5)
    rho = _pow((tt - p) / (tt - s), 0.5)
    caux2 = -math.exp(-0.5 * a * b) * _nd2(-xx, -yy, rho)

    caux = 0.5 * b * (caux + caux1 + caux2)

    xx = (a + b * (tt - p)) / _pow(2.0 * (tt - p), 0.5)
    yy = b * _pow(p - s, 0.5) / _pow(2.0, 0.5)
    caux1 = (
        math.exp(-0.5 * xx * xx)
        * math.exp(0.5 * a * b)
        * _phid(yy)
        / (2.0 * _pow(_PI * (tt - p), 0.5))
    )

    xx = (a + b * (tt - s)) / _pow(2.0 * (tt - s), 0.5)
    yy = a * _pow(p - s, 0.5) / _pow(2.0 * (tt - p) * (tt - s), 0.5)
    caux2 = (
        math.exp(-0.5 * xx * xx)
        * math.exp(0.5 * a * b)
        * _phid(yy)
        / (2.0 * _pow(_PI * (tt - s), 0.5))
    )

    xx = (a - b * (tt - p)) / _pow(2.0 * (tt - p), 0.5)
    yy = b * _pow(p - s, 0.5) / _pow(2.0, 0.5)
    caux3 = (
        -math.exp(-0.5 * xx * xx)
        * math.exp(-0.5 * a * b)
        * _phid(yy)
        / (2.0 * _pow(_PI * (tt - p), 0.5))
    )

    xx = (a - b * (tt - s)) / _pow(2.0 * (tt - s), 0.5)
    yy = a * _pow(p - s, 0.5) / _pow(2.0 * (tt - p) * (tt - s), 0.5)
    caux4 = (
        math.exp(-0.5 * xx * xx)
        * math.exp(-0.5 * a * b)
        * _phid(yy)
        / (2.0 * _pow(_PI * (tt - s), 0.5))
    )

    # # C++ parity note: this factor uses (tt - p), while the sibling ``dff``
    # # uses (tt - s) for the same expression (.cpp:512 vs :606). Verbatim.
    aa = math.exp((b * b - (1.0 - gm) * (1.0 - gm)) * (tt - p) / 4.0)
    return (caux + caux1 + caux2 + caux3 + caux4) * aa


def _ddll(s: float, p: float, tt: float, ax: float, bx: float, c: float, gm: float) -> float:
    """# C++ parity: ``ddll`` — d/da of D_L (.cpp:618-674)."""
    epsi = 1.0e-12
    limit = [0.0, 0.0, 0.0, 0.0]
    sigmarho = [0.0, 0.0, 0.0, 0.0]

    limit[1] = (ax + bx * (tt - p)) / _pow(2.0 * (tt - p), 0.5)
    limit[2] = (ax + bx * (tt - s)) / _pow(2.0 * (tt - s), 0.5)
    limit[3] = (ax + bx * tt + c) / _pow(2.0 * tt, 0.5)
    sigmarho[1] = _pow((tt - p) / (tt - s), 0.5)
    sigmarho[2] = _pow((tt - p) / tt, 0.5)
    sigmarho[3] = _pow((tt - s) / tt, 0.5)

    caux = 0.5 * bx * _tvtl(0, limit, sigmarho, epsi)
    caux = caux + _derivn3(limit, sigmarho, 1) / _pow(2.0 * (tt - p), 0.5)
    caux = caux + _derivn3(limit, sigmarho, 2) / _pow(2.0 * (tt - s), 0.5)
    caux = caux + _derivn3(limit, sigmarho, 3) / _pow(2.0 * tt, 0.5)
    caux = math.exp(0.5 * ax * bx) * caux

    limit[1] = (ax - bx * (tt - p)) / _pow(2.0 * (tt - p), 0.5)
    limit[2] = (-ax + bx * (tt - s)) / _pow(2.0 * (tt - s), 0.5)
    limit[3] = (-ax + bx * tt + c) / _pow(2.0 * tt, 0.5)
    sigmarho[1] = -_pow((tt - p) / (tt - s), 0.5)
    sigmarho[2] = -_pow((tt - p) / tt, 0.5)
    sigmarho[3] = _pow((tt - s) / tt, 0.5)

    caux1 = 0.5 * bx * _tvtl(0, limit, sigmarho, epsi)
    caux1 = caux1 - _derivn3(limit, sigmarho, 1) / _pow(2.0 * (tt - p), 0.5)
    caux1 = caux1 + _derivn3(limit, sigmarho, 2) / _pow(2.0 * (tt - s), 0.5)
    caux1 = caux1 + _derivn3(limit, sigmarho, 3) / _pow(2.0 * tt, 0.5)
    caux1 = math.exp(-0.5 * ax * bx) * caux1

    aa = math.exp((bx * bx - (1.0 - gm) * (1.0 - gm)) * (tt - s) / 4.0)
    return (caux + caux1) * aa


def _ddvv(s: float, p: float, tt: float, a: float, b: float, gm: float) -> float:
    """# C++ parity: ``ddvv`` — d/da of D_E (.cpp:681-738)."""
    aa = (a * p + b * (tt - p)) / _pow(2.0 * p * tt * (tt - p), 0.5)
    caux = _phid(aa)
    aa = math.exp(-(a - b) * (a - b) / (4.0 * tt)) / tt
    caux = 0.5 * aa * caux * (a - b)

    xx = (a * p + b * (tt - p)) / _pow(2.0 * tt * p * (tt - p), 0.5)
    yy = (a * s + b * (tt - s)) / _pow(2.0 * tt * s * (tt - s), 0.5)
    rho = _pow((s * (tt - p)) / (p * (tt - s)), 0.5)
    caux1 = -0.5 * aa * _nd2(-xx, -yy, rho) * (a - b)

    aa = math.exp(-(a + b) * (a + b) / (4.0 * tt)) / tt
    xx = (a * p - b * (tt - p)) / _pow(2.0 * tt * p * (tt - p), 0.5)
    yy = (a * s - b * (tt - s)) / _pow(2.0 * tt * s * (tt - s), 0.5)
    rho = _pow((s * (tt - p)) / (p * (tt - s)), 0.5)
    caux2 = -0.5 * aa * _nd2(-xx, -yy, rho) * (a + b)

    aa = -b * _pow((p - s) / _pow(2.0 * p * s, 0.5), 0.5)
    aux = _pow(p / (_PI * tt * (tt - p)), 0.5) * _phid(aa)

    xx = (a + b) * (a + b) / (4.0 * tt)
    yy = _pow(a * p - b * (tt - p), 2) / (4.0 * p * tt * (tt - p))
    caux3 = aux * math.exp(-xx) * math.exp(-yy) / 2.0

    xx = (a - b) * (a - b) / (4.0 * tt)
    yy = _pow(a * p + b * (tt - p), 2) / (4.0 * p * tt * (tt - p))
    caux4 = aux * math.exp(-xx) * math.exp(-yy) / 2.0

    aa = a * _pow((p - s) / _pow(2.0 * (tt - p) * (tt - s), 0.5), 0.5)
    aux = _pow(s / (_PI * tt * (tt - s)), 0.5) * _phid(aa)

    xx = (a + b) * (a + b) / (4.0 * tt)
    yy = _pow(a * s - b * (tt - s), 2) / (4.0 * s * tt * (tt - s))
    caux5 = aux * math.exp(-xx) * math.exp(-yy) / 2.0

    xx = (a - b) * (a - b) / (4.0 * tt)
    yy = _pow(a * s + b * (tt - s), 2) / (4.0 * s * tt * (tt - s))
    caux6 = aux * math.exp(-xx) * math.exp(-yy) / 2.0

    aux = math.exp((1.0 - gm) * (1.0 - gm) * tt / 4.0) * _pow(tt, 0.5)
    return (caux + caux1 + caux2 + caux3 + caux4 + caux5 + caux6) / (
        aux * 2.0 * _pow(_PI, 0.5)
    )


def _derivn3(limit: list[float], sigmarho: list[float], idx: int) -> float:
    """Derivative of the trivariate normal CDF w.r.t. one integration limit.

    # C++ parity: ``derivn3`` (.cpp:746-787). ``limit`` / ``sigmarho`` are
    # one-based (index 0 unused), as in C++.
    """
    sc = _pow(2.0 * _PI, 0.5)
    if idx == 1:
        aa = math.exp(-0.5 * _pow(limit[1], 2))
        xx = (limit[3] - sigmarho[2] * limit[1]) / _pow(1.0 - _pow(sigmarho[2], 2), 0.5)
        yy = (limit[2] - sigmarho[1] * limit[1]) / _pow(1.0 - _pow(sigmarho[1], 2), 0.5)
        rho = (sigmarho[3] - sigmarho[1] * sigmarho[2]) / _pow(
            (1.0 - sigmarho[1] * sigmarho[1]) * (1.0 - sigmarho[2] * sigmarho[2]), 0.5
        )
    elif idx == 2:
        aa = math.exp(-0.5 * limit[2] * limit[2])
        xx = (limit[1] - sigmarho[1] * limit[2]) / _pow(1.0 - _pow(sigmarho[1], 2), 0.5)
        yy = (limit[3] - sigmarho[3] * limit[2]) / _pow(1.0 - _pow(sigmarho[3], 2), 0.5)
        rho = (sigmarho[2] - sigmarho[1] * sigmarho[3]) / _pow(
            (1.0 - sigmarho[1] * sigmarho[1]) * (1.0 - sigmarho[3] * sigmarho[3]), 0.5
        )
    else:
        aa = math.exp(-0.5 * limit[3] * limit[3])
        xx = (limit[1] - sigmarho[2] * limit[3]) / _pow(1.0 - _pow(sigmarho[2], 2), 0.5)
        yy = (limit[2] - sigmarho[3] * limit[3]) / _pow(1.0 - _pow(sigmarho[3], 2), 0.5)
        rho = (sigmarho[1] - sigmarho[2] * sigmarho[3]) / _pow(
            (1.0 - sigmarho[2] * sigmarho[2]) * (1.0 - sigmarho[3] * sigmarho[3]), 0.5
        )
    return aa * _nd2(-xx, -yy, rho) / sc


# ---------------------------------------------------------------------------
# Trivariate normal (A. Genz's adaptive Plackett integration)
# ---------------------------------------------------------------------------


class _TvtlState:
    """The C++ file-scope globals shared between ``tvtl`` and the integrand.

    # C++ parity: ``Real H1, H2, H3, R23, RUA, RUB, AR, RUC; int NUC;``
    # (perturbativebarrieroptionengine.cpp:54-55).

    C++ declares these at namespace scope, so ``tvtl`` sets them and
    ``KRNRDT``/``TVTMFN`` read them without passing them down. Reproduced
    with a module-level singleton; like the C++, it is not thread-safe.
    """

    __slots__ = ("ar", "h1", "h2", "h3", "nuc", "r23", "rua", "rub", "ruc")

    def __init__(self) -> None:
        self.h1 = 0.0
        self.h2 = 0.0
        self.h3 = 0.0
        self.r23 = 0.0
        self.rua = 0.0
        self.rub = 0.0
        self.ar = 0.0
        self.ruc = 0.0
        self.nuc = 0


_TVTL_STATE: Final[_TvtlState] = _TvtlState()


def _sincs(x: float) -> tuple[float, float]:
    """sin(x) and cos(x)^2, series-approximated near |x| = pi/2.

    # C++ parity: ``SINCS`` (.cpp:938-957); C++ returns via reference params.
    """
    ee = _pow(_HALF_PI - math.fabs(x), 2)
    if ee < 5e-5:
        sx = _sign(1 - ee * (1 - ee / 12) / 2, x)
        cs = ee * (1 - ee * (1 - 2 * ee / 15) / 3)
    else:
        sx = math.sin(x)
        cs = 1 - sx * sx
    return sx, cs


def _pntgnd(
    nuc: int, ba: float, bb: float, bc: float, ra: float, rb: float, r: float, rr: float
) -> float:
    """Plackett formula integrand.

    # C++ parity: ``PNTGND`` (.cpp:1253-1275).

    # C++ parity note (1): ``FT`` is computed as
    # ``std::pow(BA - R*BB, 0.5)/RR + BB*BB`` where Genz's original Fortran
    # squares the numerator: ``FT = ( (BA-R*BB)**2/RR + BB**2 )``. The 0.5
    # exponent is a transcription defect and can make FT NaN for a negative
    # base. Reproduced verbatim.
    # C++ parity note (2): the ``else`` arm that divides by ``NUC`` sits
    # *inside* ``if (NUC < 1)``, so it can only ever run with NUC <= 0 —
    # i.e. it divides by zero. Genz's original has the NUC test the other way
    # round. Reproduced verbatim; only reachable when BT <= -10 or FT >= 100.
    """
    result = 0.0
    dt = rr * (rr - _pow(ra - rb, 2) - 2 * ra * rb * (1 - r))
    if dt > 0:
        bt = (bc * rr + ba * (r * rb - ra) + bb * (r * ra - rb)) / _pow(dt, 0.5)
        ft = _fdiv(_pow(ba - r * bb, 0.5), rr) + bb * bb
        if nuc < 1:
            if bt > -10 and ft < 100:
                result = math.exp(-ft / 2)
                if bt < 10:
                    result = result * _phid(bt)
            else:
                ft = _pow(1 + _fdiv(ft, float(nuc)), 0.5)
                result = _studnt(nuc, _fdiv(bt, ft)) / _pow(ft, float(nuc))
    return result


def _tvtmfn(
    x: float,
    h1: float,
    h2: float,
    h3: float,
    r23: float,
    rua: float,
    rub: float,
    ar: float,
    ruc: float,
    nuc: int,
) -> float:
    """Plackett formula integrands.

    # C++ parity: ``TVTMFN`` (.cpp:913-934).
    """
    zro = 0.0
    result = 0.0
    r12, rr2 = _sincs(rua * x)
    r13, rr3 = _sincs(rub * x)
    if math.fabs(rua) > 0:
        result += rua * _pntgnd(nuc, h1, h2, h3, r13, r23, r12, rr2)
    if math.fabs(rub) > 0:
        result += rub * _pntgnd(nuc, h1, h3, h2, r12, r23, r13, rr3)
    if nuc > 0:
        r, rr = _sincs(ar + ruc * x)
        result -= ruc * _pntgnd(nuc, h2, h3, h1, zro, zro, r, rr)
    return result


# Kronrod rule abscissae and weights, one-based (index 0 unused).
_WG: Final[tuple[float, ...]] = (
    0.0,
    0.2729250867779007,
    0.05566856711617449,
    0.1255803694649048,
    0.1862902109277352,
    0.2331937645919914,
    0.2628045445102478,
)
_XGK: Final[tuple[float, ...]] = (
    0.0,
    0.0000000000000000,
    0.9963696138895427,
    0.9782286581460570,
    0.9416771085780681,
    0.8870625997680953,
    0.8160574566562211,
    0.7301520055740492,
    0.6305995201619651,
    0.5190961292068118,
    0.3979441409523776,
    0.2695431559523450,
    0.1361130007993617,
)
_WGK: Final[tuple[float, ...]] = (
    0.0,
    0.1365777947111183,
    0.9765441045961290e-02,
    0.2715655468210443e-01,
    0.4582937856442671e-01,
    0.6309742475037484e-01,
    0.7866457193222764e-01,
    0.9295309859690074e-01,
    0.1058720744813894,
    0.1167395024610472,
    0.1251587991003195,
    0.1312806842298057,
    0.1351935727998845,
)


def _krnrdt(a: float, b: float) -> tuple[float, float]:
    """21-point Gauss-Kronrod rule over ``[a, b]``; returns (integral, error).

    # C++ parity: ``KRNRDT`` (.cpp:1004-1086); C++ returns the error via a
    # reference parameter. The integrand arguments come from the module-level
    # :data:`_TVTL_STATE`, exactly as C++ reads its file-scope globals.
    """
    st = _TVTL_STATE
    n = 11
    wid = (b - a) / 2.0
    cen = (b + a) / 2.0

    fc = _tvtmfn(cen, st.h1, st.h2, st.h3, st.r23, st.rua, st.rub, st.ar, st.ruc, st.nuc)
    resg = fc * _WG[1]
    resk = fc * _WGK[1]

    for j in range(1, n + 1):
        t = wid * _XGK[j + 1]
        fc = _tvtmfn(
            cen - t, st.h1, st.h2, st.h3, st.r23, st.rua, st.rub, st.ar, st.ruc, st.nuc
        ) + _tvtmfn(
            cen + t, st.h1, st.h2, st.h3, st.r23, st.rua, st.rub, st.ar, st.ruc, st.nuc
        )
        resk = resk + _WGK[j + 1] * fc
        if (j - 2 * int(j / 2)) == 0:
            resg = resg + _WG[1 + j // 2] * fc
    return wid * resk, math.fabs(wid * (resk - resg))


def _adonet(a: float, b: float, tol: float) -> float:
    """One-dimensional globally adaptive integration.

    # C++ parity: ``ADONET`` (.cpp:965-1001). Arrays are one-based with
    # ``NL = 100`` subintervals.
    """
    nl = 100
    ei = [0.0] * (nl + 1)
    ai = [0.0] * (nl + 1)
    bi = [0.0] * (nl + 1)
    fi = [0.0] * (nl + 1)

    ai[1] = a
    bi[1] = b
    err = 1.0
    ip = 1
    im = 1
    fin = 0.0
    while (4 * err) > tol and im < nl:
        im = im + 1
        bi[im] = bi[ip]
        ai[im] = (ai[ip] + bi[ip]) / 2.0
        bi[ip] = ai[im]
        fi[ip], ei[ip] = _krnrdt(ai[ip], bi[ip])
        fi[im], ei[im] = _krnrdt(ai[im], bi[im])

        err = 0.0
        fin = 0.0
        for i in range(1, im + 1):
            if ei[i] > ei[ip]:
                ip = i
            fin = fin + fi[i]
            err = err + ei[i] * ei[i]
        err = _pow(err, 0.5)
    return fin


def _studnt(nu: int, t: float) -> float:
    """Student-t distribution function.

    # C++ parity: ``STUDNT`` (.cpp:1089-1126).
    """
    zro = 0.0
    one = 1.0
    if nu < 1:
        return _phid(t)
    if nu == 1:
        return (1 + 2.0 * math.atan(t) / _PI) / 2.0
    if nu == 2:
        return (1 + t / _pow(2.0 + t * t, 0.5)) / 2.0
    tt = t * t
    cssthe = 1 / (1 + tt / float(nu))
    polyn = 1.0
    for j in range(nu - 2, 1, -2):
        polyn = 1.0 + (j - 1.0) * cssthe * polyn / float(j)
    if (nu - 2 * int(nu / 2)) == 1:
        rn = float(nu)
        ts = t / _pow(rn, 0.5)
        result = (1.0 + 2.0 * (math.atan(ts) + ts * cssthe * polyn) / _PI) / 2.0
    else:
        snthe = t / _pow(nu + tt, 0.5)
        result = (1 + snthe * polyn) / 2.0
    return max(zro, min(result, one))


def _bvtl(  # noqa: PLR0915 — transliteration
    nu: int, dh: float, dk: float, r: float
) -> float:
    """Bivariate t probability (Dunnett/Sobel, via A. Genz).

    # C++ parity: ``BVTL`` (.cpp:1129-1248).
    """
    one = 1.0
    eps = 1e-15
    if nu < 1:
        return _nd2(-dh, -dk, r)
    if (1 - r) <= eps:
        return _studnt(nu, min(dh, dk))
    if (r + 1) <= eps:
        if dh > -dk:
            return _studnt(nu, dh) - _studnt(nu, -dk)
        return 0.0

    tpi = 2.0 * _PI
    snu = _pow(float(nu), 0.5)
    ors = 1.0 - r * r
    hrk = dh - r * dk
    krh = dk - r * dh
    if (math.fabs(hrk) + ors) > 0:
        xnhk = hrk * hrk / (hrk * hrk + ors * (nu + dk * dk))
        xnkh = krh * krh / (krh * krh + ors * (nu + dh * dh))
    else:
        xnhk = 0.0
        xnkh = 0.0

    hs = int(_sign(one, dh - r * dk))
    ks = int(_sign(one, dk - r * dh))
    if (nu - 2 * int(nu / 2)) == 0:
        bvt = math.atan2(_pow(ors, 0.5), -r) / tpi
        gmph = dh / _pow(16 * (nu + dh * dh), 0.5)
        gmpk = dk / _pow(16 * (nu + dk * dk), 0.5)
        btnckh = 2 * math.atan2(_pow(xnkh, 0.5), _pow(1 - xnkh, 0.5)) / _PI
        btpdkh = 2 * _pow(xnkh * (1 - xnkh), 0.5) / _PI
        btnchk = 2 * math.atan2(_pow(xnhk, 0.5), _pow(1 - xnhk, 0.5)) / _PI
        btpdhk = 2 * _pow(xnhk * (1 - xnhk), 0.5) / _PI
        for j in range(1, nu // 2 + 1):
            bvt = bvt + gmph * (1 + ks * btnckh)
            bvt = bvt + gmpk * (1 + hs * btnchk)
            btnckh = btnckh + btpdkh
            btpdkh = 2 * j * btpdkh * (1 - xnkh) / (2 * j + 1)
            btnchk = btnchk + btpdhk
            btpdhk = 2 * j * btpdhk * (1 - xnhk) / (2 * j + 1)
            gmph = gmph * (2 * j - 1) / (2 * j * (1 + dh * dh / nu))
            gmpk = gmpk * (2 * j - 1) / (2 * j * (1 + dk * dk / nu))
    else:
        qhrk = _pow(dh * dh + dk * dk - 2 * r * dh * dk + nu * ors, 0.5)
        hkrn = dh * dk + r * nu
        hkn = dh * dk - nu
        hpk = dh + dk
        bvt = math.atan2(-snu * (hkn * qhrk + hpk * hkrn), hkn * hkrn - nu * hpk * qhrk) / tpi
        if bvt < -eps:
            bvt = bvt + 1
        gmph = dh / (tpi * snu * (1 + dh * dh / nu))
        gmpk = dk / (tpi * snu * (1 + dk * dk / nu))
        btnckh = _pow(xnkh, 0.5)
        btpdkh = btnckh
        btnchk = _pow(xnhk, 0.5)
        btpdhk = btnchk
        for j in range(1, (nu - 1) // 2 + 1):
            bvt = bvt + gmph * (1 + ks * btnckh)
            bvt = bvt + gmpk * (1 + hs * btnchk)
            btpdkh = (2 * j - 1) * btpdkh * (1 - xnkh) / (2 * j)
            btnckh = btnckh + btpdkh
            btpdhk = (2 * j - 1) * btpdhk * (1 - xnhk) / (2 * j)
            btnchk = btnchk + btpdhk
            gmph = 2 * j * gmph / ((2 * j + 1) * (1 + dh * dh / nu))
            gmpk = 2 * j * gmpk / ((2 * j + 1) * (1 + dk * dk / nu))
    return bvt


def _tvtl(nu: int, limit: list[float], sigmarho: list[float], epsi: float) -> float:
    """Trivariate normal / t probability P(X_i < limit_i, i = 1,2,3).

    # C++ parity: ``tvtl`` (.cpp:800-907). ``sigmarho`` holds r21, r31, r23
    # in that order, one-based.
    """
    st = _TVTL_STATE
    one = 1.0
    zro = 0.0
    eps = max(1.0e-14, epsi)
    pt = _PI / 2.0

    st.nuc = nu
    st.h1 = limit[1]
    st.h2 = limit[2]
    st.h3 = limit[3]
    r12 = sigmarho[1]
    r13 = sigmarho[2]
    st.r23 = sigmarho[3]

    # Sort R's and check for special cases.
    if math.fabs(r12) > math.fabs(r13):
        st.h2 = st.h3
        st.h3 = limit[2]
        r12 = r13
        r13 = sigmarho[1]

    if math.fabs(r13) > math.fabs(st.r23):
        st.h1 = st.h2
        st.h2 = limit[1]
        st.r23 = r13
        r13 = sigmarho[3]

    tvt = 0.0
    if (math.fabs(st.h1) + math.fabs(st.h2) + math.fabs(st.h3)) < eps:
        tvt = (1 + (math.asin(r12) + math.asin(r13) + math.asin(st.r23)) / pt) / 8.0
    elif nu < 1 and (math.fabs(r12) + math.fabs(r13)) < eps:
        tvt = _phid(st.h1) * _bvtl(nu, st.h2, st.h3, st.r23)
    elif nu < 1 and (math.fabs(r13) + math.fabs(st.r23)) < eps:
        tvt = _phid(st.h3) * _bvtl(nu, st.h1, st.h2, r12)
    elif nu < 1 and (math.fabs(r12) + math.fabs(st.r23)) < eps:
        tvt = _phid(st.h2) * _bvtl(nu, st.h1, st.h3, r13)
    elif (1.0 - st.r23) < eps:
        tvt = _bvtl(nu, st.h1, min(st.h2, st.h3), r12)
    elif (st.r23 + 1.0) < eps:
        if st.h2 > -st.h3:
            tvt = _bvtl(nu, st.h1, st.h2, r12) - _bvtl(nu, st.h1, -st.h3, r12)
    else:
        # Compute singular TVT value.
        if nu < 1:
            tvt = _bvtl(nu, st.h2, st.h3, st.r23) * _phid(st.h1)
        elif st.r23 > 0:
            tvt = _bvtl(nu, st.h1, min(st.h2, st.h3), zro)
        elif st.h2 > -st.h3:
            tvt = _bvtl(nu, st.h1, st.h2, zro) - _bvtl(nu, st.h1, -st.h3, zro)

        # Use numerical integration to compute probability.
        st.rua = math.asin(r12)
        st.rub = math.asin(r13)
        st.ar = math.asin(st.r23)
        st.ruc = _sign(pt, st.ar) - st.ar
        tvt = tvt + _adonet(zro, one, eps) / (4.0 * pt)
    return max(zro, min(tvt, one))


# ---------------------------------------------------------------------------
# The perturbative price itself
# ---------------------------------------------------------------------------


def _barrier_upd(  # noqa: PLR0915 — one-to-one with the C++ body
    kprice: float,
    stock: float,
    hbarr: float,
    taumin: float,
    taumax: float,
    iord: int,
    igm: int,
    integr: Callable[[float, float], float],
    integalpha: Callable[[float, float], float],
    integs: Callable[[float, float], float],
    alpha: Callable[[float], float],
    sigmaq: Callable[[float], float],
) -> float:
    """Perturbative up-and-out put price to order ``iord`` (0, 1 or 2).

    # C++ parity: ``BarrierUPD`` (perturbativebarrieroptionengine.cpp:79-305).
    """
    # # C++ parity: the three-way ``if(igm==0) ... else if(igm==1) ... else``
    # # (.cpp:104-110); the trailing ``else`` also zeroes gm.
    if igm == 0:
        gm = 0.0
    elif igm == 1:
        gm = integalpha(taumin, taumax) / (0.5 * integs(taumin, taumax))
    else:
        gm = 0.0

    # # C++ parity: ``xstar=log(kprice/hbarr); if(xstar>0.0) xstar=0.0;``
    # # (.cpp:120-122) — the boxed comment above it spells the intent as
    # # ``xstar=min(0,log(kprice/hbarr))``. ``min`` agrees bit-for-bit: it
    # # returns its first argument whenever the comparison is false, so NaN
    # # and -0.0 propagate exactly as the ``if`` leaves them.
    xstar = min(math.log(kprice / hbarr), 0.0)
    sigmat = integs(taumin, taumax)
    disc = -integr(taumin, taumax)

    # Change of variable.
    s0 = stock / hbarr

    # --- zero-th order term P_0 ------------------------------------------
    d1 = (xstar - math.log(s0) + (1.0 - gm) * 0.5 * sigmat) / math.sqrt(sigmat)
    d2 = (xstar + math.log(s0) + (1.0 - gm) * 0.5 * sigmat) / math.sqrt(sigmat)
    d3 = (xstar - math.log(s0) - (1.0 + gm) * 0.5 * sigmat) / math.sqrt(sigmat)
    d4 = (xstar + math.log(s0) - (1.0 + gm) * 0.5 * sigmat) / math.sqrt(sigmat)

    e1 = _phid(d1)
    e2 = _phid(d2)
    e3 = _phid(d3)
    e4 = _phid(d4)

    v0 = kprice * e1 - kprice * _pow(s0, 1.0 - gm) * e2
    v0 = v0 + math.exp(gm * 0.5 * sigmat) * (
        -hbarr * s0 * e3 + hbarr * _pow(s0, -gm) * e4
    )
    v0 = v0 * math.exp(disc)

    if iord == 0:
        return v0

    # --- first order term P_1 --------------------------------------------
    npoint = 1000
    npoint2 = 100
    dt = (taumax - taumin) / float(npoint)
    tt = 0.5 * integs(taumin, taumax)
    x = math.log(s0)
    et = math.exp(0.5 * (1.0 - gm) * x)
    dsqpi = _pow(_PI, 0.5)

    v1 = 0.0
    for i in range(1, npoint + 1):
        v1p = 0.0
        tmp = taumin + dt * float(2 * i - 1) * 0.5
        p = 0.5 * integs(tmp, taumax)

        # Function E(p,tt,a,b,gm)
        ccaux = (
            _v(p, tt, x, xstar, gm)
            + _v(p, tt, x, -xstar, gm)
            - _v(p, tt, -x, xstar, gm)
            - _v(p, tt, -x, -xstar, gm)
        )
        auxnew = ccaux * (
            -kprice * math.exp(-xstar * 0.5 * (1.0 - gm))
            + hbarr * math.exp(xstar * 0.5 * (1.0 + gm))
        )
        v1p = v1p + auxnew

        # Function L(p,tt,a,b,c,gm)
        b = gm - 1.0
        c = -xstar
        ccaux = _llold(p, tt, x, b, c, gm) - _llold(p, tt, -x, b, c, gm)
        v1p = v1p + kprice * (1.0 - gm) * ccaux

        b = -(gm + 1.0)
        c = xstar
        ccaux = _llold(p, tt, x, b, c, gm) - _llold(p, tt, -x, b, c, gm)
        v1p = v1p + -math.exp(gm * p) * hbarr * ccaux

        b = gm + 1.0
        c = -xstar
        ccaux = _llold(p, tt, x, b, c, gm) - _llold(p, tt, -x, b, c, gm)
        v1p = v1p + math.exp(gm * p) * hbarr * gm * ccaux

        # Function F(p,tt,a,b,c,gm)
        b = gm - 1.0
        v1p = v1p + -kprice * (1.0 - gm) * (_ff(p, tt, x, b, gm) - _ff(p, tt, -x, b, gm))

        b = gm + 1.0
        v1p = v1p + -math.exp(gm * p) * gm * hbarr * (
            _ff(p, tt, x, b, gm) - _ff(p, tt, -x, b, gm)
        )

        v1 = v1 + (alpha(tmp) - gm * 0.5 * sigmaq(tmp)) * v1p

    v1 = math.exp(disc) * et * v1 * dt / (dsqpi * 2.0)

    if iord == 1:
        return v0 + v1

    # --- second order term P_2 -------------------------------------------
    v2 = 0.0
    for i in range(1, npoint + 1):
        v2p = 0.0
        tmp = taumin + dt * float(2 * i - 1) * 0.5
        p = 0.5 * integs(tmp, taumax)
        dtp = (taumax - tmp) / float(npoint2)

        for j in range(1, npoint2 + 1):
            tmp1 = tmp + dtp * float(2 * j - 1) * 0.50
            s = 0.50 * integs(tmp1, taumax)

            caux = _dll(s, p, tt, -x, -1.0 + gm, -xstar, gm) - _dll(
                s, p, tt, x, -1.0 + gm, -xstar, gm
            )
            v2pp = caux * kprice * (1.0 - gm)

            caux = _dll(s, p, tt, -x, -1.0 - gm, xstar, gm) - _dll(
                s, p, tt, x, -1.0 - gm, xstar, gm
            )
            v2pp = v2pp - math.exp(gm * s) * hbarr * caux

            caux = _dll(s, p, tt, -x, 1.0 + gm, -xstar, gm) - _dll(
                s, p, tt, x, 1.0 + gm, -xstar, gm
            )
            v2pp = v2pp + math.exp(gm * s) * gm * hbarr * caux

            caux = _dvv(s, p, tt, -x, xstar, gm) - _dvv(s, p, tt, x, xstar, gm)
            caux = caux + (_dvv(s, p, tt, -x, -xstar, gm) - _dvv(s, p, tt, x, -xstar, gm))
            caux2 = hbarr * math.exp(0.5 * (1.0 + gm) * xstar) - kprice * math.exp(
                -0.5 * (1.0 - gm) * xstar
            )
            v2pp = v2pp + caux2 * caux

            caux = _dff(s, p, tt, -x, -1.0 + gm, gm) - _dff(s, p, tt, x, -1.0 + gm, gm)
            v2pp = v2pp - (1.0 - gm) * kprice * caux

            caux = _dff(s, p, tt, -x, 1.0 + gm, gm) - _dff(s, p, tt, x, 1.0 + gm, gm)
            v2pp = v2pp - math.exp(gm * s) * gm * hbarr * caux

            v2pp = v2pp * 0.5 * (1.0 - gm)

            caux = -_ddll(s, p, tt, -x, -1.0 + gm, -xstar, gm) + _ddll(
                s, p, tt, x, -1.0 + gm, -xstar, gm
            )
            v2pp = v2pp + caux * kprice * (1.0 - gm)

            caux = -_ddll(s, p, tt, -x, -1.0 - gm, xstar, gm) + _ddll(
                s, p, tt, x, -1.0 - gm, xstar, gm
            )
            v2pp = v2pp - math.exp(gm * s) * hbarr * caux

            caux = -_ddll(s, p, tt, -x, 1.0 + gm, -xstar, gm) + _ddll(
                s, p, tt, x, 1.0 + gm, -xstar, gm
            )
            v2pp = v2pp + math.exp(gm * s) * gm * hbarr * caux

            caux = -_ddvv(s, p, tt, -x, xstar, gm) + _ddvv(s, p, tt, x, xstar, gm)
            caux = caux + (-_dvv(s, p, tt, -x, -xstar, gm) + _dvv(s, p, tt, x, -xstar, gm))
            caux2 = hbarr * math.exp(0.5 * (1.0 + gm) * xstar) - kprice * math.exp(
                -0.5 * (1 - gm) * xstar
            )
            v2pp = v2pp + caux2 * caux

            caux = -_ddff(s, p, tt, -x, -1 + gm, gm) + _ddff(s, p, tt, x, -1 + gm, gm)
            v2pp = v2pp - (1.0 - gm) * kprice * caux

            caux = -_ddff(s, p, tt, -x, 1.0 + gm, gm) + _ddff(s, p, tt, x, 1.0 + gm, gm)
            v2pp = v2pp - math.exp(gm * s) * gm * hbarr * caux

            v2p = v2p + (alpha(tmp1) - gm * 0.5 * sigmaq(tmp1)) * v2pp

        v2 = v2 + v2p * (alpha(tmp) - gm * 0.5 * sigmaq(tmp)) * dtp

    v2 = math.exp(disc) * et * v2 * dt
    return v0 + v1 + v2


class PerturbativeBarrierOptionEngine(
    GenericEngine[BarrierOptionArguments, OneAssetOptionResults]
):
    """Perturbative engine for up-and-out put barrier options.

    # C++ parity: ``class PerturbativeBarrierOptionEngine``
    # (perturbativebarrieroptionengine.hpp:40-51 + .cpp:1510-1547).
    """

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        order: int = 1,
        zero_gamma: bool = False,
    ) -> None:
        super().__init__(BarrierOptionArguments(), OneAssetOptionResults())
        self._process: GeneralizedBlackScholesProcess = process
        self._order: int = order
        self._zero_gamma: bool = zero_gamma
        process.register_with(self)

    def calculate(self) -> None:
        """# C++ parity: ``PerturbativeBarrierOptionEngine::calculate``
        # (.cpp:1516-1547)."""
        qassert.require(
            self._arguments.barrier_type == BarrierType.UpOut,
            "this engine only manages up-and-out options",
        )
        qassert.require(
            self._arguments.rebate == 0.0,
            "this engine does not manage non-null rebates",
        )
        payoff = self._arguments.payoff
        qassert.require(
            isinstance(payoff, PlainVanillaPayoff)
            and payoff.option_type() == OptionType.Put,
            "this engine only manages put options",
        )
        assert isinstance(payoff, PlainVanillaPayoff)

        stock = self._process.x0()
        kprice = payoff.strike()
        hbarr = self._arguments.barrier
        assert hbarr is not None

        exercise = self._arguments.exercise
        assert exercise is not None
        tau_min = 0.0
        tau_max = self._process.time(exercise.last_date())

        qassert.require(self._order <= 2, "order must be <= 2")

        igm = 0 if self._zero_gamma else 1

        r = self._process.risk_free_rate()
        q = self._process.dividend_yield()
        vol = self._process.black_volatility()
        spot = self._process.x0()

        def integr(t1: float, t2: float) -> float:
            # # C++ parity: ``integr_adapter`` (.cpp:1453-1460).
            return r.forward_rate(t1, t2, Compounding.Continuous).rate() * (t2 - t1)

        def integalpha(t1: float, t2: float) -> float:
            # # C++ parity: ``integalpha_adapter`` (.cpp:1462-1473).
            a = (
                r.forward_rate(t1, t2, Compounding.Continuous).rate()
                - q.forward_rate(t1, t2, Compounding.Continuous).rate()
            )
            return a * (t2 - t1)

        def alpha(t: float) -> float:
            # # C++ parity: ``alpha_adapter`` (.cpp:1475-1484).
            return (
                r.forward_rate(t, t, Compounding.Continuous).rate()
                - q.forward_rate(t, t, Compounding.Continuous).rate()
            )

        def sigmaq(t: float) -> float:
            # # C++ parity: ``sigmaq_adapter`` (.cpp:1486-1495).
            sigma = vol.black_forward_vol_at_time(t, t, spot, True)
            return sigma * sigma

        def integs(t1: float, t2: float) -> float:
            # # C++ parity: ``integs_adapter`` (.cpp:1497-1505).
            return vol.black_forward_variance_at_time(t1, t2, spot, True)

        self._results.value = _barrier_upd(
            kprice,
            stock,
            hbarr,
            tau_min,
            tau_max,
            self._order,
            igm,
            integr,
            integalpha,
            integs,
            alpha,
            sigmaq,
        )


__all__ = ["PerturbativeBarrierOptionEngine"]

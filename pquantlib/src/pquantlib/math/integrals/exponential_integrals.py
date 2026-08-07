"""Sine and cosine integrals ``Si`` / ``Ci``, real and complex.

# C++ parity: ql/math/integrals/exponentialintegrals.{hpp,cpp}, namespace
# ``QuantLib::ExponentialIntegral`` (v1.43).

References carried over from the C++ header:

- B. Rowe et al: *GALSIM: The modular galaxy image simulation toolkit*,
  https://arxiv.org/abs/1407.7676 — the rational approximations behind the
  real-argument branches.
- V. Pegoraro, P. Slusallek: *On the Evaluation of the Complex-Valued
  Exponential Integral*, https://www.sci.utah.edu/~vpegorar/research/2011_JGT.pdf
- https://functions.wolfram.com/GammaBetaErf/ExpIntegralEi/introductions/ExpIntegrals/ShowAll.html
  — the ``Si``/``Ci`` reduction onto ``E1``.

C++ overloads ``Si``/``Ci`` on ``Real`` versus ``std::complex<Real>``; Python
has no overloading, so :func:`si` and :func:`ci` dispatch on the runtime type
and are declared with ``typing.overload`` so a ``float`` argument is typed to a
``float`` result and a ``complex`` argument to a ``complex`` result. This is the
same shape ``EulerDiscretization.drift`` uses for its two C++ overloads.

``e1`` and ``ei`` are public because they are declared in the C++ header, not
merely because ``si``/``ci`` need them.

Cross-validated against C++ v1.43 by
``migration-harness/cpp/probes/v143_math_expint/probe.cpp`` →
``migration-harness/references/v143/math/expint.json``.
"""

from __future__ import annotations

import cmath
import math
from typing import Final, overload

from pquantlib import qassert
from pquantlib.math.constants import M_PI_2, QL_EPSILON, QL_MAX_REAL

__all__ = ["M_EULER_MASCHERONI", "ci", "e1", "ei", "si"]

# C++ parity: the ``M_EULER_MASCHERONI`` macro defined at the top of
# exponentialintegrals.hpp (it is not one of the standard ``M_*`` constants,
# so it does not live in pquantlib.math.constants).
M_EULER_MASCHERONI: Final[float] = 0.5772156649015328606065121

# C++ parity: the ``constexpr``/``const`` locals of ``Ei`` (.cpp:132-138). They
# are hoisted to module level because they do not depend on the argument.
_DIST: Final[float] = 4.5
_MAX_ERROR: Final[float] = 5.0 * QL_EPSILON
_Z_INF: Final[float] = math.log(0.01 * QL_MAX_REAL) + math.log(100.0)
_Z_ASYM: Final[float] = 2.0 - 1.035 * math.log(_MAX_ERROR)


# --- exponential_integrals_helper (.cpp:33-74) --------------------------
#
# Named-namespace helpers in C++, i.e. an implementation detail of this
# translation unit; module-private here. Used only by the real-argument
# ``x > 4`` branches.


def _f(x: float) -> float:
    """# C++ parity: ``exponential_integrals_helper::f`` (.cpp:39-55)."""
    x2 = 1.0 / (x * x)

    return (
        1
        + x2
        * (
            7.44437068161936700618e2
            + x2
            * (
                1.96396372895146869801e5
                + x2
                * (
                    2.37750310125431834034e7
                    + x2
                    * (
                        1.43073403821274636888e9
                        + x2
                        * (
                            4.33736238870432522765e10
                            + x2
                            * (
                                6.40533830574022022911e11
                                + x2
                                * (
                                    4.20968180571076940208e12
                                    + x2
                                    * (
                                        1.00795182980368574617e13
                                        + x2 * (4.94816688199951963482e12 - x2 * 4.94701168645415959931e11)
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    ) / (
        x
        * (
            1
            + x2
            * (
                7.46437068161927678031e2
                + x2
                * (
                    1.97865247031583951450e5
                    + x2
                    * (
                        2.41535670165126845144e7
                        + x2
                        * (
                            1.47478952192985464958e9
                            + x2
                            * (
                                4.58595115847765779830e10
                                + x2
                                * (
                                    7.08501308149515401563e11
                                    + x2
                                    * (
                                        5.06084464593475076774e12
                                        + x2 * (1.43468549171581016479e13 + x2 * 1.11535493509914254097e13)
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )


def _g(x: float) -> float:
    """# C++ parity: ``exponential_integrals_helper::g`` (.cpp:57-73)."""
    x2 = 1.0 / (x * x)

    return (
        x2
        * (
            1
            + x2
            * (
                8.1359520115168615e2
                + x2
                * (
                    2.35239181626478200e5
                    + x2
                    * (
                        3.12557570795778731e7
                        + x2
                        * (
                            2.06297595146763354e9
                            + x2
                            * (
                                6.83052205423625007e10
                                + x2
                                * (
                                    1.09049528450362786e12
                                    + x2
                                    * (
                                        7.57664583257834349e12
                                        + x2
                                        * (
                                            1.81004487464664575e13
                                            + x2 * (6.43291613143049485e12 - x2 * 1.36517137670871689e12)
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
        / (
            1
            + x2
            * (
                8.19595201151451564e2
                + x2
                * (
                    2.40036752835578777e5
                    + x2
                    * (
                        3.26026661647090822e7
                        + x2
                        * (
                            2.23355543278099360e9
                            + x2
                            * (
                                7.87465017341829930e10
                                + x2
                                * (
                                    1.39866710696414565e12
                                    + x2
                                    * (
                                        1.17164723371736605e13
                                        + x2 * (4.01839087307656620e13 + x2 * 3.99653257887490811e13)
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )


def _sign(x: float) -> float:
    """# C++ parity: ``boost::math::sign`` — ``(x == 0) ? 0 : signbit ? -1 : 1``.

    The zero test comes first, so ``-0.0`` maps to ``0``, not ``-1``. That is
    what makes the ``+/- i*pi`` accumulator in :func:`ei` vanish on the real
    axis.
    """
    if x == 0.0:
        return 0.0
    return math.copysign(1.0, x)


# --- ExponentialIntegral::Si / Ci, real argument -------------------------


def _si_real(x: float) -> float:
    """# C++ parity: ``ExponentialIntegral::Si(Real)`` (.cpp:77-98)."""
    if x < 0:
        return -_si_real(-x)
    if x <= 4.0:
        x2 = x * x

        return (
            x
            * (
                1
                + x2
                * (
                    -4.54393409816329991e-2
                    + x2
                    * (
                        1.15457225751016682e-3
                        + x2
                        * (
                            -1.41018536821330254e-5
                            + x2
                            * (
                                9.43280809438713025e-8
                                + x2
                                * (
                                    -3.53201978997168357e-10
                                    + x2 * (7.08240282274875911e-13 - x2 * 6.05338212010422477e-16)
                                )
                            )
                        )
                    )
                )
            )
            / (
                1
                + x2
                * (
                    1.01162145739225565e-2
                    + x2
                    * (
                        4.99175116169755106e-5
                        + x2
                        * (
                            1.55654986308745614e-7
                            + x2
                            * (
                                3.28067571055789734e-10
                                + x2 * (4.5049097575386581e-13 + x2 * 3.21107051193712168e-16)
                            )
                        )
                    )
                )
            )
        )

    return M_PI_2 - _f(x) * math.cos(x) - _g(x) * math.sin(x)


def _ci_real(x: float) -> float:
    """# C++ parity: ``ExponentialIntegral::Ci(Real)`` (.cpp:100-121)."""
    qassert.require(x >= 0, "x < 0 => Ci(x) = Ci(-x) + i*pi")

    if x <= 4.0:
        x2 = x * x

        # C++ ``std::log(0.0)`` is ``-inf``; Python's ``math.log`` raises on 0,
        # so the pole is spelled out. Either way Ci(0) == -inf.
        log_x = math.log(x) if x > 0.0 else -math.inf

        return (
            M_EULER_MASCHERONI
            + log_x
            + x2
            * (
                -0.25
                + x2
                * (
                    7.51851524438898291e-3
                    + x2
                    * (
                        -1.27528342240267686e-4
                        + x2
                        * (
                            1.05297363846239184e-6
                            + x2
                            * (
                                -4.68889508144848019e-9
                                + x2 * (1.06480802891189243e-11 - x2 * 9.93728488857585407e-15)
                            )
                        )
                    )
                )
            )
            / (
                1
                + x2
                * (
                    1.1592605689110735e-2
                    + x2
                    * (
                        6.72126800814254432e-5
                        + x2
                        * (
                            2.55533277086129636e-7
                            + x2
                            * (
                                6.97071295760958946e-10
                                + x2
                                * (
                                    1.38536352772778619e-12
                                    + x2 * (1.89106054713059759e-15 + x2 * 1.39759616731376855e-18)
                                )
                            )
                        )
                    )
                )
            )
        )

    return _f(x) * math.sin(x) - _g(x) * math.cos(x)


# --- ExponentialIntegral::Ei / E1, complex argument ----------------------


def _match(z1: complex, z2: complex) -> bool:
    """# C++ parity: the ``match`` lambda inside ``Ei`` (.cpp:143-149)."""
    d = z1 - z2
    return abs(d.real) <= _MAX_ERROR * abs(z1.real) and abs(d.imag) <= _MAX_ERROR * abs(z1.imag)


def _ei_acc(z: complex, acc: complex) -> complex:
    """# C++ parity: ``ExponentialIntegral::Ei(z, acc)`` (.cpp:123-197).

    The two-argument form is not declared in the C++ header — it exists so
    :func:`e1` can fold the branch-cut offset into the series — so it stays
    module-private here too.
    """
    if z.real == 0.0 and z.imag == 0.0:
        return complex(-math.inf, 0.0)

    qassert.require(z.real < _Z_INF, f"argument error {z}")

    # NOTE: the C++ ``if (z.real() > z_inf) return exp(z)/z + acc;`` right after
    # the QL_REQUIRE above is dead code — the require already rejects
    # ``z.real() >= z_inf`` — so it is deliberately not ported.

    abs_z = abs(z)

    if abs_z > 1.1 * _Z_ASYM:
        # Asymptotic series sum_k k!/z^k, stopped once a term no longer moves
        # either component by more than MAX_ERROR relative.
        value = acc + complex(0.0, _sign(z.imag) * math.pi)
        s = cmath.exp(z) / z
        for i in range(1, math.floor(abs_z) + 2):
            if _match(value + s, value):
                return value + s
            value += s
            s *= i / z
        qassert.fail(f"series conversion issue for Ei({z})")

    if abs_z > _DIST and (z.real < 0 or abs(z.imag) > _DIST):
        # 47-level backward continued fraction.
        frac = 0j
        for k in range(47, 0, -1):
            frac = -float(k * k) / (2.0 * k + 1.0 - z + frac)
        return (acc + complex(0.0, _sign(z.imag) * math.pi)) - cmath.exp(z) / (1.0 - z + frac)

    # Power series with the half-harmonic accumulator ``nn``, terminated on
    # exact floating-point stagnation of the partial sum.
    s = 0j
    sn = z
    nn = 1.0

    n = 2
    while n < 1000 and s + sn * nn != s:
        s += sn * nn

        if n & 1:
            nn += 1 / (2.0 * (n // 2) + 1)

        sn *= -z / float(2 * n)
        n += 1

    qassert.require(n < 1000, f"series conversion issue for Ei({z})")

    r = (M_EULER_MASCHERONI + acc) + cmath.log(z) + cmath.exp(0.5 * z) * s

    if z.imag != 0.0:
        return r
    return complex(r.real, acc.imag)


def ei(z: complex) -> complex:
    """Exponential integral ``Ei``.

    # C++ parity: ``ExponentialIntegral::Ei(const std::complex<Real>&)``
    # (.cpp:199-201).
    """
    return _ei_acc(z, 0j)


def e1(z: complex) -> complex:
    """Exponential integral ``E1``.

    # C++ parity: ``ExponentialIntegral::E1(const std::complex<Real>&)``
    # (.cpp:203-213).
    """
    if z.imag < 0.0:
        return -_ei_acc(-z, complex(0.0, -math.pi))
    if z.imag > 0.0 or z.real < 0.0:
        return -_ei_acc(-z, complex(0.0, math.pi))
    return -_ei_acc(-z, 0j)


# --- ExponentialIntegral::Si / Ci, complex argument ----------------------


def _si_complex(z: complex) -> complex:
    """# C++ parity: ``ExponentialIntegral::Si(const std::complex<Real>&)``
    # (.cpp:217-235).
    """
    if abs(z) <= 0.2:
        s = 0j
        nn = z
        k = 2
        while k < 100 and s != s + nn:
            s += nn
            nn *= -z * z / ((2.0 * k - 2) * (2 * k - 1) * (2 * k - 1)) * (2.0 * k - 3)
            k += 1
        qassert.require(k < 100, f"series conversion issue for Si({z})")

        return s

    i = 1j
    # The branch-cut offset is +pi on the closed first quadrant and the whole
    # open fourth quadrant, -pi elsewhere. Note the asymmetry on the imaginary
    # axis: (0, -y) takes -pi while (x, -y) with x > 0 takes +pi.
    plus_pi = (z.real >= 0 and z.imag >= 0) or (z.real > 0 and z.imag < 0)
    return 0.5 * i * (e1(-i * z) - e1(i * z) - complex(0.0, math.pi if plus_pi else -math.pi))


def _ci_complex(z: complex) -> complex:
    """# C++ parity: ``ExponentialIntegral::Ci(const std::complex<Real>&)``
    # (.cpp:237-247).
    """
    i = 1j

    acc = 0j
    if z.real < 0.0 and z.imag >= 0.0:
        acc = complex(0.0, math.pi)
    elif z.real <= 0.0 and z.imag <= 0.0:
        acc = complex(0.0, -math.pi)

    return -0.5 * (e1(-i * z) + e1(i * z)) + acc


# --- public dispatchers --------------------------------------------------


@overload
def si(z: float) -> float: ...


@overload
def si(z: complex) -> complex: ...


def si(z: float | complex) -> float | complex:
    """Sine integral ``Si(z) = int_0^z sin(t)/t dt``.

    # C++ parity: the ``ExponentialIntegral::Si`` overload pair —
    # ``Real Si(Real)`` (.cpp:77) and
    # ``std::complex<Real> Si(const std::complex<Real>&)`` (.cpp:217).
    """
    if isinstance(z, complex):
        return _si_complex(z)
    return _si_real(z)


@overload
def ci(z: float) -> float: ...


@overload
def ci(z: complex) -> complex: ...


def ci(z: float | complex) -> float | complex:
    """Cosine integral ``Ci(z) = gamma + ln z + int_0^z (cos t - 1)/t dt``.

    The real overload raises for ``x < 0`` (``Ci(-x) = Ci(x) + i*pi`` is not
    representable as a ``float``); the complex overload accepts the whole plane.

    # C++ parity: the ``ExponentialIntegral::Ci`` overload pair —
    # ``Real Ci(Real)`` (.cpp:100) and
    # ``std::complex<Real> Ci(const std::complex<Real>&)`` (.cpp:237).
    """
    if isinstance(z, complex):
        return _ci_complex(z)
    return _ci_real(z)

"""Gauss-Kronrod 1-D integration (adaptive and non-adaptive).

# C++ parity: ql/math/integrals/kronrodintegral.hpp +
# ql/math/integrals/kronrodintegral.cpp (v1.43) classes
# ``GaussKronrodAdaptive`` and ``GaussKronrodNonAdaptive``.

:class:`GaussKronrodAdaptive` applies a 15-point Gauss-Kronrod rule with
G7 nested inside K15, recursively halving the interval whenever the
K15/G7 error estimate exceeds the tolerance.

:class:`GaussKronrodNonAdaptive` is the QUADPACK ``qng`` cascade: the
10-, 21-, 43- and 87-point Gauss-Kronrod-Patterson rules applied in
succession (each reusing its predecessor's evaluations) until the
rescaled error estimate meets either the absolute or the relative
accuracy. It is the default integrator of
:class:`~pquantlib.cashflows.linear_tsr_pricer.LinearTsrPricer`, which is
why bit-level parity with the C++ rule matters: substituting the
adaptive rule would move every CMS convexity adjustment.
"""

from __future__ import annotations

from typing import Final

from pquantlib import qassert
from pquantlib.math.constants import QL_EPSILON, QL_MIN_POSITIVE_REAL
from pquantlib.math.integrals.integrator import Integrator, RealFunction

# weights for 7-point Gauss-Legendre integration (4 unique values, symmetric).
# C++ parity: kronrodintegral.cpp static const Real g7w[].
_G7W: Final[tuple[float, ...]] = (
    0.417959183673469,
    0.381830050505119,
    0.279705391489277,
    0.129484966168870,
)
# weights for 15-point Gauss-Kronrod integration.
# C++ parity: kronrodintegral.cpp static const Real k15w[].
_K15W: Final[tuple[float, ...]] = (
    0.209482141084728,
    0.204432940075298,
    0.190350578064785,
    0.169004726639267,
    0.140653259715525,
    0.104790010322250,
    0.063092092629979,
    0.022935322010529,
)
# abscissae for 15-point Gauss-Kronrod integration.
# C++ parity: kronrodintegral.cpp static const Real k15t[].
_K15T: Final[tuple[float, ...]] = (
    0.000000000000000,
    0.207784955007898,
    0.405845151377397,
    0.586087235467691,
    0.741531185599394,
    0.864864423359769,
    0.949107912342758,
    0.991455371120813,
)


class GaussKronrodAdaptive(Integrator):
    """Adaptive Gauss-Kronrod (G7 / K15 nested rule, recursive interval bisection)."""

    def __init__(self, tolerance: float, max_evaluations: int) -> None:
        super().__init__(tolerance, max_evaluations)
        qassert.require(
            max_evaluations >= 15,
            f"required maxEvaluations ({max_evaluations}) not allowed. It must be >= 15",
        )

    def _integrate(self, f: RealFunction, a: float, b: float) -> float:
        return self._integrate_recursively(f, a, b, self._absolute_accuracy)

    def _integrate_recursively(self, f: RealFunction, a: float, b: float, tolerance: float) -> float:
        half_length = (b - a) / 2
        center = (a + b) / 2

        fc = f(center)
        g7 = fc * _G7W[0]
        k15 = fc * _K15W[0]

        # calculate g7 and half of k15
        # j runs 1..3, j2 runs 2,4,6
        for j, j2 in ((1, 2), (2, 4), (3, 6)):
            t = half_length * _K15T[j2]
            fsum = f(center - t) + f(center + t)
            g7 += fsum * _G7W[j]
            k15 += fsum * _K15W[j2]

        # calculate other half of k15 (odd j2 indices: 1,3,5,7)
        for j2 in (1, 3, 5, 7):
            t = half_length * _K15T[j2]
            fsum = f(center - t) + f(center + t)
            k15 += fsum * _K15W[j2]

        # multiply by (b - a) / 2
        g7 *= half_length
        k15 *= half_length

        # 15 more function evaluations have been used
        self._increase_number_of_evaluations(15)

        # error is <= |k15 - g7|; if larger than tolerance, split & recurse
        if abs(k15 - g7) < tolerance:
            return k15
        qassert.require(
            self._evaluations + 30 <= self._max_evaluations,
            "maximum number of function evaluations exceeded",
        )
        return self._integrate_recursively(f, a, center, tolerance / 2) + self._integrate_recursively(
            f, center, b, tolerance / 2
        )


# ---------------------------------------------------------------------------
# GaussKronrodNonAdaptive — QUADPACK "qng" 10/21/43/87-point cascade
# ---------------------------------------------------------------------------
#
# Gauss-Kronrod-Patterson quadrature coefficients for use in the QUADPACK
# routine qng, calculated with 101-decimal-digit arithmetic by
# L. W. Fullerton, Bell Labs, Nov 1981.
# C++ parity: kronrodintegral.cpp static const Real x1/w10/x2/w21a/w21b/
# x3/w43a/w43b/x4/w87a/w87b — transcribed verbatim, same order.

# abscissae common to the 10-, 21-, 43- and 87-point rule
_X1: Final[tuple[float, ...]] = (
    0.973906528517171720077964012084452,
    0.865063366688984510732096688423493,
    0.679409568299024406234327365114874,
    0.433395394129247190799265943165784,
    0.148874338981631210884826001129720,
)
# weights of the 10-point formula
_W10: Final[tuple[float, ...]] = (
    0.066671344308688137593568809893332,
    0.149451349150580593145776339657697,
    0.219086362515982043995534934228163,
    0.269266719309996355091226921569469,
    0.295524224714752870173892994651338,
)
# abscissae common to the 21-, 43- and 87-point rule
_X2: Final[tuple[float, ...]] = (
    0.995657163025808080735527280689003,
    0.930157491355708226001207180059508,
    0.780817726586416897063717578345042,
    0.562757134668604683339000099272694,
    0.294392862701460198131126603103866,
)
# weights of the 21-point formula for abscissae x1
_W21A: Final[tuple[float, ...]] = (
    0.032558162307964727478818972459390,
    0.075039674810919952767043140916190,
    0.109387158802297641899210590325805,
    0.134709217311473325928054001771707,
    0.147739104901338491374841515972068,
)
# weights of the 21-point formula for abscissae x2
_W21B: Final[tuple[float, ...]] = (
    0.011694638867371874278064396062192,
    0.054755896574351996031381300244580,
    0.093125454583697605535065465083366,
    0.123491976262065851077958109831074,
    0.142775938577060080797094273138717,
    0.149445554002916905664936468389821,
)
# abscissae common to the 43- and 87-point rule
_X3: Final[tuple[float, ...]] = (
    0.999333360901932081394099323919911,
    0.987433402908088869795961478381209,
    0.954807934814266299257919200290473,
    0.900148695748328293625099494069092,
    0.825198314983114150847066732588520,
    0.732148388989304982612354848755461,
    0.622847970537725238641159120344323,
    0.499479574071056499952214885499755,
    0.364901661346580768043989548502644,
    0.222254919776601296498260928066212,
    0.074650617461383322043914435796506,
)
# weights of the 43-point formula for abscissae x1, x3
_W43A: Final[tuple[float, ...]] = (
    0.016296734289666564924281974617663,
    0.037522876120869501461613795898115,
    0.054694902058255442147212685465005,
    0.067355414609478086075553166302174,
    0.073870199632393953432140695251367,
    0.005768556059769796184184327908655,
    0.027371890593248842081276069289151,
    0.046560826910428830743339154433824,
    0.061744995201442564496240336030883,
    0.071387267268693397768559114425516,
)
# weights of the 43-point formula for abscissae x3
_W43B: Final[tuple[float, ...]] = (
    0.001844477640212414100389106552965,
    0.010798689585891651740465406741293,
    0.021895363867795428102523123075149,
    0.032597463975345689443882222526137,
    0.042163137935191811847627924327955,
    0.050741939600184577780189020092084,
    0.058379395542619248375475369330206,
    0.064746404951445885544689259517511,
    0.069566197912356484528633315038405,
    0.072824441471833208150939535192842,
    0.074507751014175118273571813842889,
    0.074722147517403005594425168280423,
)
# abscissae of the 87-point rule
_X4: Final[tuple[float, ...]] = (
    0.999902977262729234490529830591582,
    0.997989895986678745427496322365960,
    0.992175497860687222808523352251425,
    0.981358163572712773571916941623894,
    0.965057623858384619128284110607926,
    0.943167613133670596816416634507426,
    0.915806414685507209591826430720050,
    0.883221657771316501372117548744163,
    0.845710748462415666605902011504855,
    0.803557658035230982788739474980964,
    0.757005730685495558328942793432020,
    0.706273209787321819824094274740840,
    0.651589466501177922534422205016736,
    0.593223374057961088875273770349144,
    0.531493605970831932285268948562671,
    0.466763623042022844871966781659270,
    0.399424847859218804732101665817923,
    0.329874877106188288265053371824597,
    0.258503559202161551802280975429025,
    0.185695396568346652015917141167606,
    0.111842213179907468172398359241362,
    0.037352123394619870814998165437704,
)
# weights of the 87-point formula for abscissae x1, x2, x3
_W87A: Final[tuple[float, ...]] = (
    0.008148377384149172900002878448190,
    0.018761438201562822243935059003794,
    0.027347451050052286161582829741283,
    0.033677707311637930046581056957588,
    0.036935099820427907614589586742499,
    0.002884872430211530501334156248695,
    0.013685946022712701888950035273128,
    0.023280413502888311123409291030404,
    0.030872497611713358675466394126442,
    0.035693633639418770719351355457044,
    0.000915283345202241360843392549948,
    0.005399280219300471367738743391053,
    0.010947679601118931134327826856808,
    0.016298731696787335262665703223280,
    0.021081568889203835112433060188190,
    0.025370969769253827243467999831710,
    0.029189697756475752501446154084920,
    0.032373202467202789685788194889595,
    0.034783098950365142750781997949596,
    0.036412220731351787562801163687577,
    0.037253875503047708539592001191226,
)
# weights of the 87-point formula for abscissae x4
_W87B: Final[tuple[float, ...]] = (
    0.000274145563762072350016527092881,
    0.001807124155057942948341311753254,
    0.004096869282759164864458070683480,
    0.006758290051847378699816577897424,
    0.009549957672201646536053581325377,
    0.012329447652244853694626639963780,
    0.015010447346388952376697286041943,
    0.017548967986243191099665352925900,
    0.019938037786440888202278192730714,
    0.022194935961012286796332102959499,
    0.024339147126000805470360647041454,
    0.026374505414839207241503786552615,
    0.028286910788771200659968002987960,
    0.030052581128092695322521110347341,
    0.031646751371439929404586051078883,
    0.033050413419978503290785944862689,
    0.034255099704226061787082821046821,
    0.035262412660156681033782717998428,
    0.036076989622888701185500318003895,
    0.036698604498456094498018047441094,
    0.037120549269832576114119958413599,
    0.037334228751935040321235449094698,
    0.037361073762679023410321241766599,
)


def _rescale_error(err: float, result_abs: float, result_asc: float) -> float:
    """QUADPACK error rescaling.

    # C++ parity: kronrodintegral.cpp:26-45 ``rescaleError``.
    """
    err = abs(err)
    if result_asc != 0.0 and err != 0.0:
        scale = (200.0 * err / result_asc) ** 1.5
        err = result_asc * scale if scale < 1.0 else result_asc
    if result_abs > QL_MIN_POSITIVE_REAL / (50.0 * QL_EPSILON):
        min_err = 50.0 * QL_EPSILON * result_abs
        err = max(err, min_err)
    return err


class GaussKronrodNonAdaptive(Integrator):
    """Non-adaptive 10/21/43/87-point Gauss-Kronrod cascade (QUADPACK ``qng``).

    # C++ parity: kronrodintegral.hpp:52-64 + kronrodintegral.cpp:216-336.

    Applies the 10-, 21-, 43- and 87-point rules in succession until the
    rescaled error estimate is below ``absolute_accuracy`` or below
    ``relative_accuracy * |result|``; each rule reuses every evaluation of
    its predecessors. Intended for smooth integrands.
    """

    def __init__(
        self,
        absolute_accuracy: float,
        max_evaluations: int,
        relative_accuracy: float,
    ) -> None:
        super().__init__(absolute_accuracy, max_evaluations)
        self._relative_accuracy: float = relative_accuracy

    def set_relative_accuracy(self, relative_accuracy: float) -> None:
        self._relative_accuracy = relative_accuracy

    def relative_accuracy(self) -> float:
        return self._relative_accuracy

    def _integrate(self, f: RealFunction, a: float, b: float) -> float:  # noqa: PLR0915  (faithful C++ port — qng keeps every partial rule in a local)
        qassert.require(a < b, "b must be greater than a)")

        half_length = 0.5 * (b - a)
        center = 0.5 * (b + a)
        f_center = f(center)

        # array of function values which have been computed
        savfun: list[float] = [0.0] * 21
        fv1: list[float] = [0.0] * 5
        fv2: list[float] = [0.0] * 5
        fv3: list[float] = [0.0] * 5
        fv4: list[float] = [0.0] * 5

        # --- 10- and 21-point formulae -------------------------------------
        res10 = 0.0
        res21 = _W21B[5] * f_center
        res_abs = _W21B[5] * abs(f_center)

        for k in range(5):
            abscissa = half_length * _X1[k]
            fval1 = f(center + abscissa)
            fval2 = f(center - abscissa)
            fval = fval1 + fval2
            res10 += _W10[k] * fval
            res21 += _W21A[k] * fval
            res_abs += _W21A[k] * (abs(fval1) + abs(fval2))
            savfun[k] = fval
            fv1[k] = fval1
            fv2[k] = fval2

        for k in range(5):
            abscissa = half_length * _X2[k]
            fval1 = f(center + abscissa)
            fval2 = f(center - abscissa)
            fval = fval1 + fval2
            res21 += _W21B[k] * fval
            res_abs += _W21B[k] * (abs(fval1) + abs(fval2))
            savfun[k + 5] = fval
            fv3[k] = fval1
            fv4[k] = fval2

        result = res21 * half_length
        res_abs *= half_length
        mean = 0.5 * res21
        resasc = _W21B[5] * abs(f_center - mean)

        for k in range(5):
            resasc += _W21A[k] * (abs(fv1[k] - mean) + abs(fv2[k] - mean)) + _W21B[k] * (
                abs(fv3[k] - mean) + abs(fv4[k] - mean)
            )

        err = _rescale_error((res21 - res10) * half_length, res_abs, resasc)
        resasc *= half_length

        if err < self._absolute_accuracy or err < self._relative_accuracy * abs(result):
            self._set_absolute_error(err)
            self._set_number_of_evaluations(21)
            return result

        # --- 43-point formula ----------------------------------------------
        res43 = _W43B[11] * f_center
        for k in range(10):
            res43 += savfun[k] * _W43A[k]
        for k in range(11):
            abscissa = half_length * _X3[k]
            fval = f(center + abscissa) + f(center - abscissa)
            res43 += fval * _W43B[k]
            savfun[k + 10] = fval

        result = res43 * half_length
        err = _rescale_error((res43 - res21) * half_length, res_abs, resasc)

        if err < self._absolute_accuracy or err < self._relative_accuracy * abs(result):
            self._set_absolute_error(err)
            self._set_number_of_evaluations(43)
            return result

        # --- 87-point formula ----------------------------------------------
        res87 = _W87B[22] * f_center
        for k in range(21):
            res87 += savfun[k] * _W87A[k]
        for k in range(22):
            abscissa = half_length * _X4[k]
            res87 += _W87B[k] * (f(center + abscissa) + f(center - abscissa))

        result = res87 * half_length
        err = _rescale_error((res87 - res43) * half_length, res_abs, resasc)

        self._set_absolute_error(err)
        self._set_number_of_evaluations(87)
        return result


__all__ = ["GaussKronrodAdaptive", "GaussKronrodNonAdaptive"]

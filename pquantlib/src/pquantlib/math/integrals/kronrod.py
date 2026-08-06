"""Gauss-Kronrod 1-D integration.

# C++ parity: ql/math/integrals/kronrodintegral.{hpp,cpp} (v1.43).

Two rules:

* :class:`GaussKronrodNonAdaptive` — the fixed Gauss-Kronrod-Patterson ladder
  from QUADPACK's ``qng``: the 10-, 21-, 43- and 87-point rules are applied in
  succession, each reusing every function value its predecessor computed, and
  the first one whose error estimate meets either the absolute or the relative
  accuracy wins. Fast for smooth integrands; it never subdivides. It is also
  the default integrator of
  :class:`~pquantlib.cashflows.linear_tsr_pricer.LinearTsrPricer`, which is
  why bit-level parity with the C++ rule matters: substituting the adaptive
  rule would move every CMS convexity adjustment.
* :class:`GaussKronrodAdaptive` — the 15-point rule with G7 nested inside K15,
  bisecting the interval whenever the K15/G7 discrepancy exceeds the
  tolerance. More robust on less smooth integrands, but it recomputes every
  point after a split.

References:

* Gauss-Kronrod-Patterson coefficients computed to 101 decimal digits by
  L. W. Fullerton, Bell Labs, Nov 1981 (QUADPACK ``qng``).
* NMS - Numerical Analysis Library,
  <http://www.math.iastate.edu/burkardt/f_src/nms/nms.html>
"""

from __future__ import annotations

import math
from typing import Final

from pquantlib import qassert
from pquantlib.math.constants import QL_EPSILON, QL_MIN_POSITIVE_REAL
from pquantlib.math.integrals.integrator import Integrator, RealFunction


def _rescale_error(err: float, result_abs: float, result_asc: float) -> float:
    """QUADPACK's error rescaling.

    # C++ parity: kronrodintegral.cpp:25-41 (``rescaleError``, file-static).
    """
    err = math.fabs(err)
    if result_asc != 0 and err != 0:
        scale = math.pow((200 * err / result_asc), 1.5)
        err = result_asc * scale if scale < 1 else result_asc
    if result_abs > QL_MIN_POSITIVE_REAL / (50 * QL_EPSILON):
        err = max(err, 50 * QL_EPSILON * result_abs)
    return err


# --- Gauss-Kronrod-Patterson coefficients for QUADPACK's qng --------------
# x1, abscissae common to the 10-, 21-, 43- and 87-point rule.
_X1: Final[tuple[float, ...]] = (
    0.973906528517171720077964012084452,
    0.865063366688984510732096688423493,
    0.679409568299024406234327365114874,
    0.433395394129247190799265943165784,
    0.148874338981631210884826001129720,
)

# w10, weights of the 10-point formula.
_W10: Final[tuple[float, ...]] = (
    0.066671344308688137593568809893332,
    0.149451349150580593145776339657697,
    0.219086362515982043995534934228163,
    0.269266719309996355091226921569469,
    0.295524224714752870173892994651338,
)

# x2, abscissae common to the 21-, 43- and 87-point rule.
_X2: Final[tuple[float, ...]] = (
    0.995657163025808080735527280689003,
    0.930157491355708226001207180059508,
    0.780817726586416897063717578345042,
    0.562757134668604683339000099272694,
    0.294392862701460198131126603103866,
)

# w21a, weights of the 21-point formula for abscissae x1.
_W21A: Final[tuple[float, ...]] = (
    0.032558162307964727478818972459390,
    0.075039674810919952767043140916190,
    0.109387158802297641899210590325805,
    0.134709217311473325928054001771707,
    0.147739104901338491374841515972068,
)

# w21b, weights of the 21-point formula for abscissae x2.
_W21B: Final[tuple[float, ...]] = (
    0.011694638867371874278064396062192,
    0.054755896574351996031381300244580,
    0.093125454583697605535065465083366,
    0.123491976262065851077958109831074,
    0.142775938577060080797094273138717,
    0.149445554002916905664936468389821,
)

# x3, abscissae common to the 43- and 87-point rule.
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

# w43a, weights of the 43-point formula for abscissae x1, x3.
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

# w43b, weights of the 43-point formula for abscissae x3.
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

# x4, abscissae of the 87-point rule.
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

# w87a, weights of the 87-point formula for abscissae x1, x2, x3.
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

# w87b, weights of the 87-point formula for abscissae x4.
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

# --- 15-point Gauss-Kronrod with G7 nested (adaptive rule) ---------------

# weights for 7-point Gauss-Legendre integration
# (only 4 values out of 7 are given as they are symmetric).
# C++ parity: kronrodintegral.cpp:362-365 (``g7w``).
_G7W: Final[tuple[float, ...]] = (
    0.417959183673469,
    0.381830050505119,
    0.279705391489277,
    0.129484966168870,
)
# weights for 15-point Gauss-Kronrod integration.
# C++ parity: kronrodintegral.cpp:367-374 (``k15w``).
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
# abscissae (evaluation points) for 15-point Gauss-Kronrod integration.
# C++ parity: kronrodintegral.cpp:377-384 (``k15t``).
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

class GaussKronrodNonAdaptive(Integrator):
    """Non-adaptive 10/21/43/87-point Gauss-Kronrod-Patterson rule.

    # C++ parity: kronrodintegral.hpp:52-64, kronrodintegral.cpp:218-351.
    """

    __slots__ = ("_relative_accuracy",)

    def __init__(
        self, absolute_accuracy: float, max_evaluations: int, relative_accuracy: float
    ) -> None:
        # C++ parity: kronrodintegral.cpp:227-231.
        super().__init__(absolute_accuracy, max_evaluations)
        self._relative_accuracy: float = relative_accuracy

    def set_relative_accuracy(self, relative_accuracy: float) -> None:
        # C++ parity: kronrodintegral.cpp:218-220.
        self._relative_accuracy = relative_accuracy

    def relative_accuracy(self) -> float:
        # C++ parity: kronrodintegral.cpp:223-225.
        return self._relative_accuracy

    def _integrate(  # noqa: PLR0915 — faithful port of a long C++ method
        self, f: RealFunction, a: float, b: float
    ) -> float:
        # C++ parity: kronrodintegral.cpp:233-351, transcribed rule by rule.
        fv1 = [0.0] * 5
        fv2 = [0.0] * 5
        fv3 = [0.0] * 5
        fv4 = [0.0] * 5
        savfun = [0.0] * 21  # function values already computed

        qassert.require(a < b, "b must be greater than a)")

        half_length = 0.5 * (b - a)
        center = 0.5 * (b + a)
        f_center = f(center)

        # Compute the integral using the 10- and 21-point formula.
        res10 = 0.0
        res21 = _W21B[5] * f_center
        res_abs = _W21B[5] * math.fabs(f_center)

        for k in range(5):
            abscissa = half_length * _X1[k]
            fval1 = f(center + abscissa)
            fval2 = f(center - abscissa)
            fval = fval1 + fval2
            res10 += _W10[k] * fval
            res21 += _W21A[k] * fval
            res_abs += _W21A[k] * (math.fabs(fval1) + math.fabs(fval2))
            savfun[k] = fval
            fv1[k] = fval1
            fv2[k] = fval2

        for k in range(5):
            abscissa = half_length * _X2[k]
            fval1 = f(center + abscissa)
            fval2 = f(center - abscissa)
            fval = fval1 + fval2
            res21 += _W21B[k] * fval
            res_abs += _W21B[k] * (math.fabs(fval1) + math.fabs(fval2))
            savfun[k + 5] = fval
            fv3[k] = fval1
            fv4[k] = fval2

        result = res21 * half_length
        res_abs *= half_length
        mean = 0.5 * res21
        resasc = _W21B[5] * math.fabs(f_center - mean)

        for k in range(5):
            resasc += _W21A[k] * (
                math.fabs(fv1[k] - mean) + math.fabs(fv2[k] - mean)
            ) + _W21B[k] * (math.fabs(fv3[k] - mean) + math.fabs(fv4[k] - mean))

        err = _rescale_error((res21 - res10) * half_length, res_abs, resasc)
        resasc *= half_length

        # test for convergence.
        if err < self._absolute_accuracy or err < self.relative_accuracy() * math.fabs(result):
            self._set_absolute_error(err)
            self._set_number_of_evaluations(21)
            return result

        # compute the integral using the 43-point formula.
        res43 = _W43B[11] * f_center

        for k in range(10):
            res43 += savfun[k] * _W43A[k]

        for k in range(11):
            abscissa = half_length * _X3[k]
            fval = f(center + abscissa) + f(center - abscissa)
            res43 += fval * _W43B[k]
            savfun[k + 10] = fval

        # test for convergence.
        result = res43 * half_length
        err = _rescale_error((res43 - res21) * half_length, res_abs, resasc)

        if err < self._absolute_accuracy or err < self.relative_accuracy() * math.fabs(result):
            self._set_absolute_error(err)
            self._set_number_of_evaluations(43)
            return result

        # compute the integral using the 87-point formula.
        res87 = _W87B[22] * f_center

        for k in range(21):
            res87 += savfun[k] * _W87A[k]

        for k in range(22):
            abscissa = half_length * _X4[k]
            res87 += _W87B[k] * (f(center + abscissa) + f(center - abscissa))

        # test for convergence.
        result = res87 * half_length
        err = _rescale_error((res87 - res43) * half_length, res_abs, resasc)

        self._set_absolute_error(err)
        self._set_number_of_evaluations(87)
        return result

class GaussKronrodAdaptive(Integrator):
    """Adaptive Gauss-Kronrod (G7 / K15 nested rule, recursive bisection).

    # C++ parity: kronrodintegral.hpp:85-97, kronrodintegral.cpp:353-448.
    """

    __slots__ = ()

    def __init__(self, tolerance: float, max_evaluations: int) -> None:
        # C++ parity: kronrodintegral.cpp:442-448.
        super().__init__(tolerance, max_evaluations)
        qassert.require(
            max_evaluations >= 15,
            f"required maxEvaluations ({max_evaluations}) not allowed. It must be >= 15",
        )

    def _integrate(self, f: RealFunction, a: float, b: float) -> float:
        # C++ parity: kronrodintegral.cpp:353-358.
        return self._integrate_recursively(f, a, b, self._absolute_accuracy)

    def _integrate_recursively(
        self, f: RealFunction, a: float, b: float, tolerance: float
    ) -> float:
        # C++ parity: kronrodintegral.cpp:386-439.
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


__all__ = ["GaussKronrodAdaptive", "GaussKronrodNonAdaptive"]

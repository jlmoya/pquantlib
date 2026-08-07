"""XabrSwaptionVolatilityCube — model-parameterised xabr-style swaption cube.

# C++ parity: ql/termstructures/volatility/swaption/sabrswaptionvolatilitycube.hpp
#             template ``XabrSwaptionVolatilityCube<Model>``, its private
#             nested ``Cube`` and ``PrivateObserver``, plus
#             ``ql/termstructures/volatility/swaption/zabrswaptionvolatilitycube.hpp``
#             (the typedef ``XabrSwaptionVolatilityCube<SwaptionVolCubeZabrModel<>>``)
#             and ``ql/experimental/volatility/noarbsabrswaptionvolatilitycube.hpp``.
#             (v1.43).

The C++ class is a template parameterised by an ``XabrModelTraits``
specialisation that picks the interpolation class (``SABRInterpolation``,
``ZabrInterpolation<...>``, ``NoArbSabrInterpolation``) and the
corresponding smile section. The PQuantLib generalisation uses an
:class:`XabrModelKind` IntEnum dispatch instead of C++ templates.

Three classes live here, mirroring the header:

* :class:`Cube` — C++ ``XabrSwaptionVolatilityCube<Model>::Cube``
  (hpp:130-181 declaration, hpp:986-1256 implementation). A stack of
  ``n_layers`` matrices over an ``(option_times x swap_lengths)`` grid,
  each layer carrying its own 2-D interpolation so that a parameter
  vector can be read off at ANY ``(option_time, swap_length)``, not only
  at a grid pillar.
* :class:`PrivateObserver` — C++
  ``XabrSwaptionVolatilityCube<Model>::PrivateObserver`` (hpp:268-279).
  Registered with every parameter-guess Quote; a quote move re-seeds the
  guess cube and invalidates the calculation.
* :class:`XabrSwaptionVolatilityCube` itself.

Both nested C++ classes are *private*; Python has no access control, so
they are module-level here. Names are kept verbatim.

Documented divergences vs C++:

* The **ATM-calibrated dense branch** is not ported: C++
  ``fillVolatilityCube`` / ``createSparseSmiles`` / ``spreadVolInterpolation``
  (hpp:646-858) densify the vol cube onto the union of the ATM surface's
  pillars before a second calibration pass, and ``smileSectionImpl``
  then reads ``denseParameters_`` instead of ``sparseParameters_``
  (hpp:877-884). This port always takes the ``isAtmCalibrated == false``
  arm, i.e. ``smileSectionImpl`` reads ``sparseParameters_``. There is
  consequently no ``isAtmCalibrated`` constructor flag, no
  ``denseSabrParameters()`` and no ``volCubeAtmCalibrated()``.
  :class:`Cube` itself IS complete — ``expand_layers`` / ``set_point``
  are ported and cross-validated — so the dense branch can be added on
  top without touching it.
* ``end_criteria`` has no PQuantLib counterpart (the fitters expose
  ``converged()`` instead), so the last metadata layer of the parameter
  cube stores ``0.0`` for a converged fit and ``1.0`` otherwise. Those
  are exactly ``EndCriteria::None`` and ``EndCriteria::MaxIterations``,
  the only two values C++ ever tests for (hpp:486-497).
* ``max_error_tolerance`` / ``error_accept`` / ``use_max_error`` are not
  reproduced as *guards*: C++ raises when the fit residual exceeds them
  (hpp:486-512). The residuals are still computed and stored in the
  parameter cube's metadata layers, so a caller can apply its own
  threshold.
* The C++ optimiser knobs ``endCriteria`` / ``optMethod`` /
  ``maxGuesses`` are not constructor arguments here and are not
  forwarded to the fitters. All three fitters are SciPy
  ``least_squares(method="trf")`` delegations whose ``ftol`` / ``xtol``
  / ``gtol`` are hard-coded internally (1e-12 for SABR and ZABR, 1e-10
  for no-arb SABR) and are NOT exposed; ``max_nfev`` / ``max_guesses`` /
  ``multi_start_seed`` are their own knobs, likewise not plumbed
  through the cube. The one place this is provably immaterial is the
  all-parameters-fixed path, where both C++ and this port skip the
  optimiser entirely (xabrinterpolation.hpp:162-169) — which is why the
  cross-validation reference is built that way.
* SABR, ZABR and no-arb SABR share one constructor: the only per-mode
  difference is the parameter count (4 / 5 / 4) and the classes
  instantiated for the fit and the smile section.
"""

from __future__ import annotations

import weakref
from bisect import bisect_left
from collections.abc import Sequence
from enum import IntEnum

import numpy as np

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.swap_index import SwapIndex
from pquantlib.math.interpolations.backwardflat_linear_interpolation import (
    BackwardflatLinearInterpolation,
)
from pquantlib.math.interpolations.bilinear import BilinearInterpolation
from pquantlib.math.interpolations.flat_extrapolation_2d import FlatExtrapolator2D
from pquantlib.math.interpolations.interpolation_2d import Interpolation2D
from pquantlib.math.interpolations.sabr_interpolation import SabrInterpolation
from pquantlib.math.interpolations.zabr_formula import ZabrEvaluation
from pquantlib.math.interpolations.zabr_interpolation import ZabrInterpolation
from pquantlib.math.matrix import Matrix
from pquantlib.quotes.quote import Quote
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.sabr_smile_section import SabrSmileSection
from pquantlib.termstructures.volatility.smile_section import SmileSection
from pquantlib.termstructures.volatility.swaption.swaption_volatility_cube import (
    AtmSwapIndexProtocol,
    SwaptionVolatilityCube,
)
from pquantlib.termstructures.volatility.swaption.swaption_volatility_structure import (
    SwaptionVolatilityStructure,
)
from pquantlib.termstructures.volatility.zabr_smile_section import ZabrSmileSection
from pquantlib.time.date import Date
from pquantlib.time.period import Period


class XabrModelKind(IntEnum):
    """Selector for the xabr model used in :class:`XabrSwaptionVolatilityCube`.

    # C++ parity: indirectly mapped from the template parameter triple
    # ``SwaptionVolCubeSabrModel`` / ``SwaptionVolCubeZabrModel<Kernel>``
    # / ``SwaptionVolCubeNoArbSabrModel``.
    """

    SABR = 0  # 4-param Hagan 2002 SABR via SabrInterpolation + SabrSmileSection
    ZABR = 1  # 5-param ZABR via ZabrInterpolation + ZabrSmileSection
    NOARB_SABR = 2  # 4-param Doust no-arb SABR via NoArbSabrInterpolation


def _n_params_for(kind: XabrModelKind) -> int:
    """``XabrModelTraits<Model>::nParams`` for each supported model.

    # C++ parity: primary template ``nParams = 4``
    # (sabrswaptionvolatilitycube.hpp:80); ``SwaptionVolCubeZabrModel``
    # specialises it to 5 (zabrswaptionvolatilitycube.hpp).
    """
    return 5 if kind == XabrModelKind.ZABR else 4


# =====================================================================
#                                Cube
# =====================================================================


class Cube:
    """Layered ``(option_times x swap_lengths)`` grid with per-layer 2-D interpolation.

    # C++ parity: ``XabrSwaptionVolatilityCube<Model>::Cube``
    # (sabrswaptionvolatilitycube.hpp:130-181, 986-1256).

    Each of the ``n_layers`` layers is a :data:`Matrix` shaped
    ``(len(option_times), len(swap_lengths))`` — rows index option times,
    columns index swap lengths. ``option_dates`` runs parallel to
    ``option_times`` and ``swap_tenors`` parallel to ``swap_lengths``.

    The interpolations are built over the **transposed** layers
    (hpp:1010, 1225): ``Interpolation2D`` indexes ``z[y, x]`` with
    ``x = option_time`` and ``y = swap_length``, while ``points_[k]``
    is stored ``[option_time][swap_length]``. Getting that backwards
    yields plausible-looking but wrong numbers everywhere off the
    diagonal, so the transpose is load-bearing.

    Args:
        option_dates: option dates, parallel to ``option_times``.
        swap_tenors: swap tenors, parallel to ``swap_lengths``.
        option_times: ascending option times; at least 2 (hpp:998).
        swap_lengths: ascending swap lengths; at least 2 (hpp:999).
        n_layers: how many matrices the cube stacks.
        extrapolation: stored, never read — see :meth:`extrapolation`.
        backward_flat: pick ``BackwardflatLinearInterpolation`` over
            ``BilinearInterpolation`` for layers 0..4 (hpp:1018).
    """

    def __init__(
        self,
        option_dates: Sequence[Date],
        swap_tenors: Sequence[Period],
        option_times: Sequence[float],
        swap_lengths: Sequence[float],
        n_layers: int,
        extrapolation: bool = True,
        backward_flat: bool = False,
    ) -> None:
        # C++ parity: hpp:998-1004.
        qassert.require(len(option_times) > 1, "Cube::Cube(...): optionTimes.size()<2")
        qassert.require(len(swap_lengths) > 1, "Cube::Cube(...): swapLengths.size()<2")
        qassert.require(
            len(option_times) == len(option_dates),
            "Cube::Cube(...): optionTimes/optionDates mismatch",
        )
        qassert.require(
            len(swap_tenors) == len(swap_lengths),
            "Cube::Cube(...): swapTenors/swapLengths mismatch",
        )

        self._option_times: list[float] = [float(t) for t in option_times]
        self._swap_lengths: list[float] = [float(s) for s in swap_lengths]
        self._option_dates: list[Date] = list(option_dates)
        self._swap_tenors: list[Period] = list(swap_tenors)
        self._n_layers: int = int(n_layers)
        self._extrapolation: bool = extrapolation
        self._backward_flat: bool = backward_flat

        # C++ parity: hpp:1006-1007 — nLayers zero matrices of the grid shape.
        points: list[Matrix] = [
            np.zeros((len(self._option_times), len(self._swap_lengths)), dtype=np.float64)
            for _ in range(self._n_layers)
        ]
        self._points: list[Matrix] = points
        self._transposed_points: list[Matrix] = []
        self._interpolators: list[Interpolation2D] = []
        # C++ builds the interpolators inside the ctor loop (hpp:1008-1033)
        # and only THEN calls setPoints; the effect is identical to setting
        # the points first and rebuilding, which is what happens here.
        self.set_points(points)
        self.update_interpolators()

    # ----- construction helpers -------------------------------------------

    def copy(self) -> Cube:
        """Return an independent copy with freshly built interpolators.

        # C++ parity: ``Cube::Cube(const Cube&)`` (hpp:1037-1061) and
        # ``Cube::operator=`` (hpp:1063-1093). Both rebuild the
        # interpolators from ``transposedPoints_`` and then ``setPoints``.
        #
        # Note: C++ ``operator=`` *appends* to ``interpolators_`` without
        # clearing it first, so an assigned-to Cube keeps its previous
        # interpolators at indices ``[0, nLayers)`` until
        # ``updateInterpolators()`` is called. Every C++ call site does
        # call it, so the latent bug is unobservable; it is not
        # reproduced here.
        """
        other = Cube(
            self._option_dates,
            self._swap_tenors,
            self._option_times,
            self._swap_lengths,
            self._n_layers,
            self._extrapolation,
            self._backward_flat,
        )
        other.set_points([p.copy() for p in self._points])
        other.update_interpolators()
        return other

    # ----- mutators --------------------------------------------------------

    def set_element(
        self,
        index_of_layer: int,
        index_of_row: int,
        index_of_column: int,
        x: float,
    ) -> None:
        """``points_[layer][row][column] = x``.

        # C++ parity: ``Cube::setElement`` (hpp:1095-1106). ``row`` indexes
        # option times, ``column`` indexes swap lengths.
        """
        qassert.require(
            0 <= index_of_layer < self._n_layers,
            "Cube::setElement: incompatible IndexOfLayer ",
        )
        qassert.require(
            0 <= index_of_row < len(self._option_times),
            "Cube::setElement: incompatible IndexOfRow",
        )
        qassert.require(
            0 <= index_of_column < len(self._swap_lengths),
            "Cube::setElement: incompatible IndexOfColumn",
        )
        self._points[index_of_layer][index_of_row, index_of_column] = float(x)

    def set_points(self, x: Sequence[Matrix]) -> None:
        """Replace every layer at once.

        # C++ parity: ``Cube::setPoints`` (hpp:1108-1118). C++ only checks
        # ``x[0]``'s shape; the loop below checks every layer, which can
        # only reject inputs C++ would have accepted and then corrupted.
        """
        qassert.require(
            len(x) == self._n_layers,
            "Cube::setPoints: incompatible number of layers ",
        )
        expected = (len(self._option_times), len(self._swap_lengths))
        for layer in x:
            arr = np.asarray(layer, dtype=np.float64)
            qassert.require(arr.shape[0] == expected[0], "Cube::setPoints: incompatible size 1")
            qassert.require(arr.shape[1] == expected[1], "Cube::setPoints: incompatible size 2")
        self._points = [np.array(layer, dtype=np.float64, copy=True) for layer in x]

    def set_layer(self, i: int, x: Matrix) -> None:
        """Replace layer ``i``.

        # C++ parity: ``Cube::setLayer`` (hpp:1120-1130).
        """
        qassert.require(0 <= i < self._n_layers, "Cube::setLayer: incompatible number of layer ")
        arr = np.asarray(x, dtype=np.float64)
        qassert.require(
            arr.shape[0] == len(self._option_times), "Cube::setLayer: incompatible size 1"
        )
        qassert.require(
            arr.shape[1] == len(self._swap_lengths), "Cube::setLayer: incompatible size 2"
        )
        self._points[i] = np.array(arr, dtype=np.float64, copy=True)

    def set_point(
        self,
        option_date: Date,
        swap_tenor: Period,
        option_time: float,
        swap_length: float,
        point: Sequence[float],
    ) -> None:
        """Write one full parameter vector, growing the grid if needed.

        # C++ parity: ``Cube::setPoint`` (hpp:1132-1166).

        The cell is located by SEARCHING ``option_times`` / ``swap_lengths``
        (``binary_search`` for "is it already a pillar", ``lower_bound``
        for "where does it belong"), not by index. A time or length that
        is not already a pillar triggers :meth:`expand_layers`, and the
        supplied date/tenor then overwrite the placeholders inserted there.

        Note that C++ does NOT call ``updateInterpolators`` here; callers
        do it once after a batch of ``setPoint`` calls (hpp:641, 715). The
        same holds in this port: until then, :meth:`__call__` still reads
        the pre-expansion interpolators.
        """
        # C++ parity: hpp:1139-1142.
        expand_option_times = not _binary_search(self._option_times, option_time)
        expand_swap_lengths = not _binary_search(self._swap_lengths, swap_length)

        # C++ parity: hpp:1147-1153 — std::lower_bound.
        option_times_index = bisect_left(self._option_times, option_time)
        swap_lengths_index = bisect_left(self._swap_lengths, swap_length)

        if expand_option_times or expand_swap_lengths:
            self.expand_layers(
                option_times_index,
                expand_option_times,
                swap_lengths_index,
                expand_swap_lengths,
            )

        # C++ parity: hpp:1159-1160.
        qassert.require(
            len(point) >= self._n_layers,
            f"Cube::setPoint: point has {len(point)} entries; need {self._n_layers}",
        )
        for k in range(self._n_layers):
            self._points[k][option_times_index, swap_lengths_index] = float(point[k])

        # C++ parity: hpp:1162-1165.
        self._option_times[option_times_index] = float(option_time)
        self._swap_lengths[swap_lengths_index] = float(swap_length)
        self._option_dates[option_times_index] = option_date
        self._swap_tenors[swap_lengths_index] = swap_tenor

    def expand_layers(
        self,
        i: int,
        expand_option_times: bool,
        j: int,
        expand_swap_lengths: bool,
    ) -> None:
        """Insert a row before ``i`` and/or a column before ``j``, shifting the rest.

        # C++ parity: ``Cube::expandLayers`` (hpp:1168-1197).

        The inserted option time / swap length is ``0.0`` and the inserted
        date / tenor are the null ``Date()`` / ``Period()`` — placeholders
        that :meth:`set_point` overwrites immediately. Existing values move
        to ``u+1`` (resp. ``v+1``) if and only if their old index is ``>= i``
        (resp. ``>= j``) AND that axis is being expanded, so a row insert and
        a column insert compose without interfering.

        Neither ``transposedPoints_`` nor ``interpolators_`` are resized
        here — C++ leaves them stale until ``updateInterpolators()``, and so
        does this port.
        """
        # C++ parity: hpp:1170-1171 — note ``<=``: inserting at the end is legal.
        qassert.require(
            0 <= i <= len(self._option_times), "Cube::expandLayers: incompatible size 1"
        )
        qassert.require(
            0 <= j <= len(self._swap_lengths), "Cube::expandLayers: incompatible size 2"
        )

        old_points = self._points
        old_rows = old_points[0].shape[0] if old_points else len(self._option_times)
        old_cols = old_points[0].shape[1] if old_points else len(self._swap_lengths)

        # C++ parity: hpp:1173-1180.
        if expand_option_times:
            self._option_times.insert(i, 0.0)
            self._option_dates.insert(i, Date())
        if expand_swap_lengths:
            self._swap_lengths.insert(j, 0.0)
            self._swap_tenors.insert(j, Period())

        # C++ parity: hpp:1182-1195.
        new_points: list[Matrix] = [
            np.zeros((len(self._option_times), len(self._swap_lengths)), dtype=np.float64)
            for _ in range(self._n_layers)
        ]
        for k in range(self._n_layers):
            for u in range(old_rows):
                index_of_row = u + 1 if (u >= i and expand_option_times) else u
                for v in range(old_cols):
                    index_of_col = v + 1 if (v >= j and expand_swap_lengths) else v
                    new_points[k][index_of_row, index_of_col] = old_points[k][u, v]
        self.set_points(new_points)

    # ----- inspectors ------------------------------------------------------

    def option_dates(self) -> list[Date]:
        """# C++ parity: ``Cube::optionDates`` (hpp:159-161). C++ returns a
        const reference; this port returns a copy, matching the convention
        set by ``Interpolation2D.x_values``.
        """
        return list(self._option_dates)

    def swap_tenors(self) -> list[Period]:
        """# C++ parity: ``Cube::swapTenors`` (hpp:162-164)."""
        return list(self._swap_tenors)

    def option_times(self) -> list[float]:
        """# C++ parity: ``Cube::optionTimes`` (hpp:1213-1216)."""
        return list(self._option_times)

    def swap_lengths(self) -> list[float]:
        """# C++ parity: ``Cube::swapLengths`` (hpp:1218-1221)."""
        return list(self._swap_lengths)

    def points(self) -> list[Matrix]:
        """# C++ parity: ``Cube::points`` (hpp:1199-1202)."""
        return [p.copy() for p in self._points]

    def n_layers(self) -> int:
        """Layer count. No C++ accessor — ``nLayers_`` is a private member."""
        return self._n_layers

    def extrapolation(self) -> bool:
        """The ``extrapolation`` constructor flag.

        # C++ parity: ``Cube::extrapolation_`` (hpp:178) is stored by the
        # constructor, the copy constructor and ``operator=`` and then
        # NEVER READ: every layer is unconditionally wrapped in a
        # ``FlatExtrapolator2D`` with extrapolation enabled
        # (hpp:1030-1032, 1239-1241). The flag is carried here for parity
        # and is likewise inert.
        """
        return self._extrapolation

    def backward_flat(self) -> bool:
        """The ``backwardFlat`` constructor flag. No C++ accessor."""
        return self._backward_flat

    # ----- evaluation ------------------------------------------------------

    def __call__(self, option_time: float, swap_length: float) -> list[float]:
        """Interpolate every layer at ``(option_time, swap_length)``.

        # C++ parity: ``Cube::operator()`` (hpp:1204-1211).

        Outside the grid the value is flat-extrapolated (each layer is
        wrapped in a ``FlatExtrapolator2D``, hpp:1030-1032), so the result
        equals the value on the nearest boundary point.
        """
        return [
            float(self._interpolators[k](option_time, swap_length))
            for k in range(self._n_layers)
        ]

    def update_interpolators(self) -> None:
        """Re-transpose every layer and rebuild its interpolation.

        # C++ parity: ``Cube::updateInterpolators`` (hpp:1223-1243).

        Must be called after any mutation — ``set_element`` /
        ``set_layer`` / ``set_points`` / ``set_point`` / ``expand_layers``
        all leave the interpolators stale, exactly as in C++, because the
        2-D interpolations own a copy of the grid.
        """
        times = np.asarray(self._option_times, dtype=np.float64)
        lengths = np.asarray(self._swap_lengths, dtype=np.float64)
        transposed: list[Matrix] = []
        interpolators: list[Interpolation2D] = []
        for k in range(self._n_layers):
            # C++ parity: hpp:1225 — the interpolation sees the TRANSPOSE.
            tp = np.ascontiguousarray(self._points[k].T)
            transposed.append(tp)
            inner: Interpolation2D
            # C++ parity: hpp:1227 — the threshold is the literal ``k <= 4``,
            # not ``k < nParams``. The header calls this out at hpp:1011-1017:
            # for a 4-parameter model the forward-rate metadata layer (index 4)
            # also gets BackwardflatLinear, while for a 5-parameter model the
            # forward-rate layer (index 5) gets Bilinear.
            if k <= 4 and self._backward_flat:
                inner = BackwardflatLinearInterpolation(times, lengths, tp)
            else:
                inner = BilinearInterpolation(times, lengths, tp)
            # C++ parity: hpp:1239-1241.
            flat = FlatExtrapolator2D(inner)
            flat.enable_extrapolation()
            interpolators.append(flat)
        self._transposed_points = transposed
        self._interpolators = interpolators

    def transposed_points(self) -> list[Matrix]:
        """The mirrored, transposed layers the interpolations are built on.

        # C++ parity: ``Cube::transposedPoints_`` (hpp:177) — a ``mutable``
        # member with no accessor. Exposed here because it is only ever
        # filled by :meth:`update_interpolators`, and a test that reads it
        # is a test that the transpose actually happened.
        """
        return [p.copy() for p in self._transposed_points]

    def browse(self) -> Matrix:
        """Flatten the cube to a ``(n_swap_lengths * n_option_times) x (n_layers + 2)`` matrix.

        # C++ parity: ``Cube::browse`` (hpp:1245-1256).

        Row ``i * n_option_times + j`` carries ``swap_lengths[i]`` in
        column 0, ``option_times[j]`` in column 1, and
        ``points_[k][j][i]`` in column ``2 + k``. Note the row-major order
        is over SWAP LENGTHS first.
        """
        n_opt = len(self._option_times)
        n_swap = len(self._swap_lengths)
        result: Matrix = np.zeros((n_swap * n_opt, self._n_layers + 2), dtype=np.float64)
        for i in range(n_swap):
            for j in range(n_opt):
                row = i * n_opt + j
                result[row, 0] = self._swap_lengths[i]
                result[row, 1] = self._option_times[j]
                for k in range(self._n_layers):
                    result[row, 2 + k] = self._points[k][j, i]
        return result


def _binary_search(values: Sequence[float], x: float) -> bool:
    """``std::binary_search`` over a sorted sequence."""
    i = bisect_left(values, x)
    return i < len(values) and values[i] == x


# =====================================================================
#                           PrivateObserver
# =====================================================================


class PrivateObserver:
    """Observer that re-seeds the guess cube when a guess Quote moves.

    # C++ parity: ``XabrSwaptionVolatilityCube<Model>::PrivateObserver``
    # (sabrswaptionvolatilitycube.hpp:268-279).

    The cube registers one of these with every parameter-guess Quote
    (``registerWithParametersGuess``, hpp:341-347) rather than registering
    itself, because the reaction is not the plain "invalidate" that a
    vol-spread move triggers: the guess cube has to be rebuilt from the
    new quote values FIRST, and only then is the calculation invalidated.

    C++ holds the cube by raw pointer — non-owning, because the cube owns
    the observer. This port holds a :class:`weakref.ref` for the same
    reason: an owning back-reference would make cube and observer an
    uncollectable cycle.
    """

    __slots__ = ("__weakref__", "_v")

    def __init__(self, v: XabrSwaptionVolatilityCube) -> None:
        self._v: weakref.ReferenceType[XabrSwaptionVolatilityCube] = weakref.ref(v)

    def update(self) -> None:
        """# C++ parity: ``PrivateObserver::update`` (hpp:272-275)."""
        cube = self._v()
        if cube is None:
            # The cube is gone; C++ cannot reach this state because the
            # observer dies with the cube that owns it.
            return
        cube.set_parameter_guess()
        cube.update()


# Initial-guess shapes. Each cell is either the parameter tuple itself or
# the Quotes carrying it; C++ only accepts ``Handle<Quote>``.
_GuessCell = Sequence[float] | Sequence[Quote]
_XabrGuess = Sequence[Sequence[_GuessCell]]


class XabrSwaptionVolatilityCube(SwaptionVolatilityCube):
    """xabr-style swaption volatility cube, model_kind-dispatched.

    Args:
        model_kind: which xabr model to use (SABR, ZABR or NOARB_SABR).
        atm_vol_structure / option_tenors / swap_tenors / strike_spreads
            / vol_spreads / swap_index_base / short_swap_index_base /
            vega_weighted_smile_fit: same as
            :class:`SwaptionVolatilityCube`.
        initial_guess: optional outer list shape
            ``n_option_tenors x n_swap_tenors``, each cell a sequence of
            ``n_params`` initial parameters — either plain floats or
            :class:`Quote` objects. Quotes are observed: moving one
            re-seeds the guess cube through :class:`PrivateObserver` and
            invalidates the fit. Plain floats are wrapped in
            :class:`SimpleQuote`, so they are observable too but nothing
            else holds a reference with which to move them.
            If ``None``, no guess cube is built and each cell falls back
            to the underlying interpolation's own default initial guess
            — a PQuantLib extension; C++ requires ``parametersGuess``.
        is_parameter_fixed: per-mode parameter-fix mask shared across
            grid cells (4 entries for SABR / NOARB_SABR, 5 for ZABR).
        backward_flat: C++ ``backwardFlat`` — use backward-flat-in-option-time
            interpolation for cube layers 0..4 instead of bilinear.
        cutoff_strike: C++ ``cutoffStrike`` — a shifted strike below this
            is dropped from the calibration slice (hpp:442).
        zabr_evaluation: ZABR evaluation mode. Ignored unless
            ``model_kind == XabrModelKind.ZABR``.
    """

    def __init__(
        self,
        *,
        model_kind: XabrModelKind,
        atm_vol_structure: SwaptionVolatilityStructure,
        option_tenors: Sequence[Period],
        swap_tenors: Sequence[Period],
        strike_spreads: Sequence[float],
        vol_spreads: Sequence[Sequence[Quote]],
        swap_index_base: SwapIndex | AtmSwapIndexProtocol,
        short_swap_index_base: SwapIndex | AtmSwapIndexProtocol,
        vega_weighted_smile_fit: bool = False,
        initial_guess: _XabrGuess | None = None,
        is_parameter_fixed: Sequence[bool] = (False, False, False, False),
        backward_flat: bool = False,
        cutoff_strike: float = 0.0001,
        zabr_evaluation: ZabrEvaluation = ZabrEvaluation.ShortMaturityLognormal,
    ) -> None:
        super().__init__(
            atm_vol_structure=atm_vol_structure,
            option_tenors=option_tenors,
            swap_tenors=swap_tenors,
            strike_spreads=strike_spreads,
            vol_spreads=vol_spreads,
            swap_index_base=swap_index_base,
            short_swap_index_base=short_swap_index_base,
            vega_weighted_smile_fit=vega_weighted_smile_fit,
        )
        self._model_kind: XabrModelKind = model_kind
        n_expected = _n_params_for(model_kind)
        qassert.require(
            len(is_parameter_fixed) == n_expected,
            f"is_parameter_fixed has length {len(is_parameter_fixed)}; "
            f"expected {n_expected} for {model_kind.name}",
        )
        self._n_params: int = n_expected
        self._is_param_fixed: tuple[bool, ...] = tuple(is_parameter_fixed)
        self._zabr_evaluation: ZabrEvaluation = zabr_evaluation
        self._backward_flat: bool = backward_flat
        self._cutoff_strike: float = cutoff_strike

        # C++ parity: ``parametersGuessQuotes_`` (hpp:254) is a FLAT
        # ``[j*nSwapTenors + k]`` list of per-cell parameter Quote vectors.
        self._parameters_guess_quotes: list[list[Quote]] | None = (
            None if initial_guess is None else self._normalise_guess(initial_guess)
        )
        self._parameters_guess: Cube | None = None

        # Cached calculation. C++ gets this from LazyObject; PQuantLib's
        # TermStructure is a plain Observable, so the two-flag contract is
        # reproduced here (see ``_calculate`` / ``update``).
        self._calculated: bool = False
        self._market_vol_cube: Cube | None = None
        self._sparse_parameters: Cube | None = None

        # C++ parity: hpp:336-338 — the observer is created, registered with
        # every guess quote, and the guess cube seeded, all in the ctor.
        self._private_observer: PrivateObserver = PrivateObserver(self)
        self._register_with_parameters_guess()
        self.set_parameter_guess()

    # --- guess plumbing ----------------------------------------------------

    def _normalise_guess(self, initial_guess: _XabrGuess) -> list[list[Quote]]:
        """Flatten a ``[j][k]`` guess grid into C++'s ``[j*nSwap + k]`` layout."""
        n_opt = len(self._option_tenors)
        n_swap = len(self._swap_tenors)
        qassert.require(
            len(initial_guess) == n_opt,
            f"initial_guess has {len(initial_guess)} rows; expected {n_opt}",
        )
        flat: list[list[Quote]] = []
        for j in range(n_opt):
            row = initial_guess[j]
            qassert.require(
                len(row) == n_swap,
                f"initial_guess row {j} has {len(row)} cells; expected {n_swap}",
            )
            for k in range(n_swap):
                cell = row[k]
                if len(cell) != self._n_params:
                    raise LibraryException(
                        f"{self._model_kind.name} cube expects a "
                        f"{self._n_params}-tuple guess at cell ({j},{k}); "
                        f"got length {len(cell)}"
                    )
                quotes: list[Quote] = [
                    p if isinstance(p, Quote) else SimpleQuote(float(p)) for p in cell
                ]
                flat.append(quotes)
        return flat

    def _register_with_parameters_guess(self) -> None:
        """# C++ parity: ``registerWithParametersGuess`` (hpp:341-347)."""
        if self._parameters_guess_quotes is None:
            return
        n_opt = len(self._option_tenors)
        n_swap = len(self._swap_tenors)
        for i in range(self._n_params):
            for j in range(n_opt):
                for k in range(n_swap):
                    self._parameters_guess_quotes[j * n_swap + k][i].register_with(
                        self._private_observer
                    )

    def set_parameter_guess(self) -> None:
        """Rebuild the guess :class:`Cube` from the guess Quotes.

        # C++ parity: ``setParameterGuess`` (hpp:349-364). Called from the
        # constructor and from :class:`PrivateObserver` on every guess-quote
        # move. A no-op when the cube was built without an ``initial_guess``.
        """
        if self._parameters_guess_quotes is None:
            self._parameters_guess = None
            return
        n_opt = len(self._option_tenors)
        n_swap = len(self._swap_tenors)
        guess = Cube(
            self._option_dates,
            self._swap_tenors,
            self._option_times,
            self._swap_lengths,
            self._n_params,
            True,
            self._backward_flat,
        )
        for i in range(self._n_params):
            for j in range(n_opt):
                for k in range(n_swap):
                    guess.set_element(
                        i, j, k, self._parameters_guess_quotes[j * n_swap + k][i].value()
                    )
        guess.update_interpolators()
        self._parameters_guess = guess

    # --- lazy-calculation contract ----------------------------------------

    def update(self) -> None:
        """Invalidate the cached calibration and propagate.

        # C++ parity: ``LazyObject::update`` — the cube is a LazyObject in
        # C++ and its ``performCalculations`` re-runs on the next access.
        """
        self._calculated = False
        super().update()

    def _calculate(self) -> None:
        """# C++ parity: ``LazyObject::calculate``."""
        if not self._calculated:
            self._calculated = True
            try:
                self._perform_calculations()
            except BaseException:
                self._calculated = False
                raise

    def _perform_calculations(self) -> None:
        """Rebuild ``market_vol_cube_`` from the quotes, then calibrate.

        # C++ parity: ``performCalculations`` (hpp:366-399), taking the
        # ``isAtmCalibrated == false`` arm (see the module docstring).
        """
        n_opt = len(self._option_tenors)
        n_swap = len(self._swap_tenors)
        n_strikes = len(self._strike_spreads)

        # C++ parity: hpp:371-386.
        market = Cube(
            self._option_dates,
            self._swap_tenors,
            self._option_times,
            self._swap_lengths,
            n_strikes,
        )
        for j in range(n_opt):
            for k in range(n_swap):
                atm_forward = self.atm_strike(self._option_dates[j], self._swap_tenors[k])
                atm_vol = self._atm_vol.volatility(
                    self._option_dates[j], self._swap_tenors[k], atm_forward, extrapolate=True
                )
                for i in range(n_strikes):
                    market.set_element(
                        i, j, k, atm_vol + self._vol_spreads[j * n_swap + k][i].value()
                    )
        market.update_interpolators()
        self._market_vol_cube = market

        # C++ parity: hpp:388-390.
        sparse = self._xabr_calibration(market)
        sparse.update_interpolators()
        self._sparse_parameters = sparse

    # --- calibration -------------------------------------------------------

    def _xabr_calibration(self, market_vol_cube: Cube) -> Cube:
        """Fit one xabr slice per grid cell into a fresh parameter :class:`Cube`.

        # C++ parity: ``sabrCalibration`` (hpp:411-535). The returned cube
        # has ``n_params + 4`` layers: the model parameters, then forwards,
        # rms errors, max errors and the end-criteria code.
        """
        option_times = market_vol_cube.option_times()
        swap_lengths = market_vol_cube.swap_lengths()
        option_dates = market_vol_cube.option_dates()
        swap_tenors = market_vol_cube.swap_tenors()
        n_strikes = len(self._strike_spreads)
        shape = (len(option_times), len(swap_lengths))

        params = [np.zeros(shape, dtype=np.float64) for _ in range(self._n_params)]
        forwards: Matrix = np.zeros(shape, dtype=np.float64)
        errors: Matrix = np.zeros(shape, dtype=np.float64)
        max_errors: Matrix = np.zeros(shape, dtype=np.float64)
        end_criteria: Matrix = np.zeros(shape, dtype=np.float64)

        market_points = market_vol_cube.points()

        for j in range(len(option_times)):
            for k in range(len(swap_lengths)):
                atm_forward = self.atm_strike(option_dates[j], swap_tenors[k])
                shift_tmp = self._atm_vol.shift(
                    option_times[j], swap_lengths[k], extrapolate=True
                )
                strikes: list[float] = []
                volatilities: list[float] = []
                for i in range(n_strikes):
                    strike = atm_forward + self._strike_spreads[i]
                    # C++ parity: hpp:442 — the guard is against
                    # ``cutoffStrike_`` (default 1e-4), not against zero.
                    if strike + shift_tmp >= self._cutoff_strike:
                        strikes.append(strike)
                        volatilities.append(float(market_points[i][j, k]))

                # C++ parity: hpp:448-449 — the guess is the guess CUBE
                # interpolated at this pillar, not a per-cell constant.
                guess = (
                    None
                    if self._parameters_guess is None
                    else self._parameters_guess(option_times[j], swap_lengths[k])
                )
                fitted, rms, max_err, converged = self._fit_one_cell(
                    guess, strikes, volatilities, option_times[j], atm_forward, shift_tmp
                )
                for p in range(self._n_params):
                    params[p][j, k] = fitted[p]
                forwards[j, k] = atm_forward
                errors[j, k] = rms
                max_errors[j, k] = max_err
                # See the module docstring: 0 == EndCriteria::None,
                # 1 == EndCriteria::MaxIterations.
                end_criteria[j, k] = 0.0 if converged else 1.0

        # C++ parity: hpp:517-531.
        cube = Cube(
            option_dates,
            swap_tenors,
            option_times,
            swap_lengths,
            self._n_params + 4,
            True,
            self._backward_flat,
        )
        for p in range(self._n_params):
            cube.set_layer(p, params[p])
        cube.set_layer(self._n_params, forwards)
        cube.set_layer(self._n_params + 1, errors)
        cube.set_layer(self._n_params + 2, max_errors)
        cube.set_layer(self._n_params + 3, end_criteria)
        return cube

    def _fit_one_cell(
        self,
        guess: Sequence[float] | None,
        strikes: list[float],
        vols: list[float],
        option_time: float,
        forward: float,
        shift_local: float,
    ) -> tuple[tuple[float, ...], float, float, bool]:
        """Dispatch to the model-specific fit kernel.

        # C++ parity: ``XabrModelTraits<Model>::createInterpolation``
        # (hpp:82-99) plus the ``->update()`` call at hpp:465.

        Returns ``(params, rms_error, max_error, converged)``.
        """
        if self._model_kind == XabrModelKind.SABR:
            return self._fit_sabr_cell(guess, strikes, vols, option_time, forward, shift_local)
        if self._model_kind == XabrModelKind.NOARB_SABR:
            return self._fit_noarb_sabr_cell(guess, strikes, vols, option_time, forward)
        return self._fit_zabr_cell(guess, strikes, vols, option_time, forward)

    def _fit_sabr_cell(
        self,
        guess: Sequence[float] | None,
        strikes: list[float],
        vols: list[float],
        option_time: float,
        forward: float,
        shift_local: float,
    ) -> tuple[tuple[float, ...], float, float, bool]:
        a0, b0, n0, r0 = (
            (None, None, None, None)
            if guess is None
            else (guess[0], guess[1], guess[2], guess[3])
        )
        interp = SabrInterpolation(
            strikes,
            vols,
            option_time,
            forward,
            alpha=a0,
            beta=b0,
            nu=n0,
            rho=r0,
            alpha_is_fixed=self._is_param_fixed[0],
            beta_is_fixed=self._is_param_fixed[1],
            nu_is_fixed=self._is_param_fixed[2],
            rho_is_fixed=self._is_param_fixed[3],
            vega_weighted=self._vega_weighted_smile_fit,
            shift=shift_local,
            volatility_type=self.volatility_type(),
        )
        return (
            (interp.alpha(), interp.beta(), interp.nu(), interp.rho()),
            interp.rms_error(),
            interp.max_error(),
            interp.converged(),
        )

    def _fit_zabr_cell(
        self,
        guess: Sequence[float] | None,
        strikes: list[float],
        vols: list[float],
        option_time: float,
        forward: float,
    ) -> tuple[tuple[float, ...], float, float, bool]:
        a0, b0, n0, r0, g0 = (
            (None, None, None, None, None)
            if guess is None
            else (guess[0], guess[1], guess[2], guess[3], guess[4])
        )
        interp = ZabrInterpolation(
            strikes=strikes,
            volatilities=vols,
            expiry_time=option_time,
            forward=forward,
            alpha=a0,
            beta=b0,
            nu=n0,
            rho=r0,
            gamma=g0,
            alpha_is_fixed=self._is_param_fixed[0],
            beta_is_fixed=self._is_param_fixed[1],
            nu_is_fixed=self._is_param_fixed[2],
            rho_is_fixed=self._is_param_fixed[3],
            gamma_is_fixed=self._is_param_fixed[4],
            vega_weighted=self._vega_weighted_smile_fit,
            evaluation=self._zabr_evaluation,
        )
        return (
            (interp.alpha(), interp.beta(), interp.nu(), interp.rho(), interp.gamma()),
            interp.rms_error(),
            interp.max_error(),
            interp.converged(),
        )

    def _fit_noarb_sabr_cell(
        self,
        guess: Sequence[float] | None,
        strikes: list[float],
        vols: list[float],
        option_time: float,
        forward: float,
    ) -> tuple[tuple[float, ...], float, float, bool]:
        # Imported lazily so the no-arb model + its absorption-table
        # asset are only pulled in when a NOARB_SABR cube is built.
        from pquantlib.experimental.volatility.no_arb_sabr_interpolation import (  # noqa: PLC0415
            NoArbSabrInterpolation,
        )

        a0, b0, n0, r0 = (
            (None, None, None, None)
            if guess is None
            else (guess[0], guess[1], guess[2], guess[3])
        )
        interp = NoArbSabrInterpolation(
            strikes,
            vols,
            option_time,
            forward,
            alpha=a0,
            beta=b0,
            nu=n0,
            rho=r0,
            alpha_is_fixed=self._is_param_fixed[0],
            beta_is_fixed=self._is_param_fixed[1],
            nu_is_fixed=self._is_param_fixed[2],
            rho_is_fixed=self._is_param_fixed[3],
            vega_weighted=self._vega_weighted_smile_fit,
        )
        return (
            (interp.alpha(), interp.beta(), interp.nu(), interp.rho()),
            interp.rms_error(),
            interp.max_error(),
            interp.converged(),
        )

    # --- inspectors --------------------------------------------------------

    def model_kind(self) -> XabrModelKind:
        return self._model_kind

    def zabr_evaluation(self) -> ZabrEvaluation:
        return self._zabr_evaluation

    def n_params(self) -> int:
        """``XabrModelTraits<Model>::nParams`` for this cube's model."""
        return self._n_params

    def backward_flat(self) -> bool:
        """The C++ ``backwardFlat_`` constructor flag (hpp:264)."""
        return self._backward_flat

    def cutoff_strike(self) -> float:
        """The C++ ``cutoffStrike_`` constructor flag (hpp:265)."""
        return self._cutoff_strike

    def parameters_guess(self) -> Cube | None:
        """The guess :class:`Cube`, or ``None`` when built without one.

        # C++ parity: ``parametersGuess_`` (hpp:255) — private, no accessor.
        """
        return self._parameters_guess

    def sparse_parameters(self) -> Cube:
        """The fitted parameter :class:`Cube`.

        # C++ parity: ``sparseParameters_`` (hpp:250) — private, reachable
        # in C++ only through ``sparseSabrParameters()``.
        """
        self._calculate()
        assert self._sparse_parameters is not None
        return self._sparse_parameters

    def sparse_xabr_parameters(self) -> Matrix:
        """``sparseParameters_.browse()``.

        # C++ parity: ``sparseSabrParameters`` (hpp:886-889).
        """
        return self.sparse_parameters().browse()

    def market_vol_cube(self, i: int | None = None) -> Matrix:
        """Layer ``i`` of the market vol cube, or its ``browse()`` when ``i`` is None.

        # C++ parity: ``marketVolCube(Size i)`` (hpp:215-217) and
        # ``marketVolCube()`` (hpp:896-899).
        """
        self._calculate()
        assert self._market_vol_cube is not None
        if i is None:
            return self._market_vol_cube.browse()
        return self._market_vol_cube.points()[i]

    def xabr_parameters(self, j: int, k: int) -> tuple[float, ...]:
        """Return the fitted parameter tuple at grid cell ``(j, k)``.

        Returns a 4-tuple ``(alpha, beta, nu, rho)`` in SABR / NOARB_SABR
        mode and a 5-tuple ``(alpha, beta, nu, rho, gamma)`` in ZABR mode.
        """
        points = self.sparse_parameters().points()
        return tuple(float(points[p][j, k]) for p in range(self._n_params))

    def fitted_forward(self, j: int, k: int) -> float:
        """Return the ATM forward at grid cell ``(j, k)``.

        # C++ parity: layer ``Traits::nParams`` of the parameter cube
        # (hpp:528).
        """
        return float(self.sparse_parameters().points()[self._n_params][j, k])

    # --- smile section -----------------------------------------------------

    def _smile_section_from_cube(
        self, option_time: float, swap_length: float, parameters_cube: Cube
    ) -> SmileSection:
        """Build the smile section from a parameter cube evaluated at ``(t, s)``.

        # C++ parity: ``XabrSwaptionVolatilityCube::smileSection(Time, Time,
        # const Cube&)`` (hpp:860-875).
        """
        self._calculate()
        all_parameters = parameters_cube(option_time, swap_length)
        # C++ parity: hpp:868 — the forward lives at index ``Traits::nParams``.
        forward = all_parameters[self._n_params]
        model_params = all_parameters[: self._n_params]
        shift_tmp = self._atm_vol.shift(option_time, swap_length, extrapolate=True)

        if self._model_kind == XabrModelKind.SABR:
            alpha, beta, nu, rho = model_params
            return SabrSmileSection(
                forward=forward,
                sabr_params=(alpha, beta, nu, rho),
                exercise_time=option_time,
                volatility_type=self.volatility_type(),
                shift=shift_tmp,
            )
        if self._model_kind == XabrModelKind.NOARB_SABR:
            from pquantlib.experimental.volatility.no_arb_sabr_smile_section import (  # noqa: PLC0415
                NoArbSabrSmileSection,
            )

            alpha, beta, nu, rho = model_params
            return NoArbSabrSmileSection(
                forward=forward,
                sabr_params=(alpha, beta, nu, rho),
                exercise_time=option_time,
                volatility_type=self.volatility_type(),
                shift=0.0,
            )
        alpha, beta, nu, rho, gamma = model_params
        return ZabrSmileSection(
            forward=forward,
            zabr_params=(alpha, beta, nu, rho, gamma),
            exercise_time=option_time,
            volatility_type=self.volatility_type(),
            shift=0.0,
            evaluation=self._zabr_evaluation,
        )

    def smile_section_impl(self, option_time: float, swap_length: float) -> SmileSection:
        """Return the smile section at ``(option_time, swap_length)``.

        # C++ parity: ``smileSectionImpl`` (hpp:877-884), ``isAtmCalibrated
        # == false`` arm. The parameters are the parameter cube INTERPOLATED
        # at ``(option_time, swap_length)`` — bilinear (or backward-flat in
        # option time) inside the grid, flat outside it.
        """
        return self._smile_section_from_cube(option_time, swap_length, self.sparse_parameters())

    def recalibrate(self) -> None:
        """Force a re-fit at every grid cell.

        # C++ parity: ``LazyObject::recalculate`` — invalidate, run
        # ``performCalculations`` again, then notify. Quote-driven
        # invalidation happens on its own (vol spreads notify the cube
        # directly; guess quotes go through :class:`PrivateObserver`), so
        # this is only needed when something the cube does not observe has
        # changed.
        """
        self._calculated = False
        self._calculate()
        self.notify_observers()


__all__ = ["Cube", "PrivateObserver", "XabrModelKind", "XabrSwaptionVolatilityCube"]

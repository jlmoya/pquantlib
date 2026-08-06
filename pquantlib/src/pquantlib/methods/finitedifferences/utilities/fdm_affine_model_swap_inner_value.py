"""FdmAffineModelSwapInnerValue — swaption exercise value on an FD grid.

# C++ parity: ql/methods/finitedifferences/utilities/fdmaffinemodelswapinnervalue.{hpp,cpp}
# @ v1.43 (6b57206e0).

At an exercise node the calculator prices the *whole* underlying swap under
the affine model state read off the mesher::

    disTs = FdmAffineModelTermStructure(state(disModel), ..., exerciseDate, ...)
    fwdTs = FdmAffineModelTermStructure(state(fwdModel), ..., exerciseDate, ...)
    npv   = -sum(fixed coupons)  + sum(floating coupons),  each discounted on disTs
            (only coupons whose accrual starts on/after the exercise date)
    value = max(0, npv * (-1 if receiver))

The C++ class is a template on ``ModelType`` with two explicit
specialisations of ``getState`` — ``HullWhite`` and ``G2``. Python has no
templates, so the port is one class that dispatches on the runtime model
type and rejects anything else, which is exactly the set of instantiations
that link in C++.

**Handles.** C++ keeps two ``RelinkableHandle<YieldTermStructure>`` members
so the swap's cloned index keeps pointing at whatever curve the calculator
installs. PQuantLib does not port the ``Handle`` indirection, so this module
defines a private :class:`_RelinkableYieldTermStructure` that plays the same
role — a stable object the cloned index binds to once, whose inner link is
swapped per exercise date.

**Divergence — OIS.** The C++ overnight branch rebuilds the swap with
``paymentLag``, ``paymentCalendar``, ``telescopicValueDates``,
``averagingMethod``, ``lookbackDays``, ``lockoutDays`` and
``applyObservationShift``. PQuantLib's ``OvernightIndexedSwap`` exposes no
public accessor for the first three and does not model the last four, so the
rebuilt OIS uses the constructor defaults for them. Anything else about the
OIS (type, nominals, both schedules, fixed rate/day-count, spread, payment
convention, index) is carried over faithfully.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import final

import numpy as np

from pquantlib import qassert
from pquantlib.cashflows.coupon import Coupon
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.indexes.overnight_index import OvernightIndex
from pquantlib.instruments.fixed_vs_floating_swap import FixedVsFloatingSwap
from pquantlib.instruments.overnight_indexed_swap import OvernightIndexedSwap
from pquantlib.instruments.swap import SwapType
from pquantlib.instruments.vanilla_swap import VanillaSwap
from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.operators.fdm_linear_op_layout import (
    FdmLinearOpIterator,
)
from pquantlib.methods.finitedifferences.utilities.fdm_affine_model_term_structure import (
    FdmAffineModelTermStructure,
)
from pquantlib.methods.finitedifferences.utilities.fdm_inner_value_calculator import (
    FdmInnerValueCalculator,
)
from pquantlib.models.shortrate.onefactor.hull_white import HullWhite
from pquantlib.models.shortrate.twofactor.g2 import G2
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date

#: The two model types C++ explicitly specialises ``getState`` for.
AffineSwapModel = HullWhite | G2


def _calendar_or_none(ts: YieldTermStructure) -> Calendar | None:
    """The curve's calendar, or ``None`` when it has none.

    # C++ parity: ``TermStructure::calendar()`` returns the stored
    # ``Calendar``, which for curves built without one (``FlatForward``,
    # for instance) is a default-constructed *empty* Calendar — perfectly
    # legal to copy around. PQuantLib spells the empty calendar ``None``
    # and makes ``TermStructure.calendar()`` raise for it, so the read has
    # to be guarded here to keep the C++ call shape.
    """
    try:
        return ts.calendar()
    except LibraryException:
        return None


@final
class _RelinkableYieldTermStructure:
    """Stable curve object whose inner link can be swapped.

    # C++ parity: the ``RelinkableHandle<YieldTermStructure> disTs_, fwdTs_``
    # members. PQuantLib threads term structures directly, so the relinking
    # behaviour is provided by this tiny forwarder instead. It satisfies
    # ``YieldTermStructureProtocol`` (the surface ``IborIndex`` needs).
    """

    __slots__ = ("_link",)

    def __init__(self) -> None:
        self._link: FdmAffineModelTermStructure | None = None

    def empty(self) -> bool:
        """# C++ parity: ``Handle::empty()``."""
        return self._link is None

    def link_to(self, ts: FdmAffineModelTermStructure) -> None:
        """# C++ parity: ``RelinkableHandle::linkTo``."""
        self._link = ts

    def current_link(self) -> FdmAffineModelTermStructure:
        """# C++ parity: ``Handle::currentLink()``."""
        qassert.require(self._link is not None, "empty Handle cannot be dereferenced")
        assert self._link is not None
        return self._link

    def reference_date(self) -> Date:
        return self.current_link().reference_date()

    def max_date(self) -> Date:
        return self.current_link().max_date()

    def day_counter(self) -> DayCounter:
        return self.current_link().day_counter()

    def discount(self, t: float | Date, extrapolate: bool = False) -> float:
        return self.current_link().discount(t, extrapolate)


@final
class FdmAffineModelSwapInnerValue(FdmInnerValueCalculator):
    """Underlying-swap exercise value under an affine short-rate model.

    # C++ parity: ``template <class ModelType> class
    # FdmAffineModelSwapInnerValue : public FdmInnerValueCalculator``.
    """

    def __init__(
        self,
        dis_model: AffineSwapModel,
        fwd_model: AffineSwapModel,
        swap: FixedVsFloatingSwap,
        exercise_dates: Mapping[float, Date],
        mesher: FdmMesher,
        direction: int,
    ) -> None:
        # C++ declares disTs_/fwdTs_ before the models, so they are default
        # constructed (empty) before the swap clone binds to fwdTs_.
        self._dis_ts: _RelinkableYieldTermStructure = _RelinkableYieldTermStructure()
        self._fwd_ts: _RelinkableYieldTermStructure = _RelinkableYieldTermStructure()
        self._dis_model: AffineSwapModel = dis_model
        self._fwd_model: AffineSwapModel = fwd_model
        self._index = swap.ibor_index()
        self._swap: FixedVsFloatingSwap = self._clone_swap(swap)
        self._exercise_dates: dict[float, Date] = dict(exercise_dates)
        self._mesher: FdmMesher = mesher
        self._direction: int = direction

    def _clone_swap(self, swap: FixedVsFloatingSwap) -> FixedVsFloatingSwap:
        """Rebuild ``swap`` with its floating index bound to ``fwd_ts``.

        # C++ parity: the immediately-invoked lambda in the member-init list
        # of ``FdmAffineModelSwapInnerValue``.
        """
        if isinstance(swap, OvernightIndexedSwap):
            on_index = swap.overnight_index()
            # C++ parity: ``QL_REQUIRE(clonedIndex, "failed to clone
            # OvernightIndex")`` — there the guard catches a failed
            # ``dynamic_pointer_cast``; here it catches an index that does not
            # expose ``clone`` (PQuantLib's ``OvernightIndexProtocol`` does
            # not declare it, unlike C++'s ``OvernightIndex``).
            qassert.require(
                isinstance(on_index, OvernightIndex), "failed to clone OvernightIndex"
            )
            assert isinstance(on_index, OvernightIndex)
            return OvernightIndexedSwap(
                swap.swap_type(),
                swap.overnight_nominals(),
                swap.fixed_schedule(),
                swap.fixed_rate(),
                swap.fixed_day_count(),
                on_index.clone(self._fwd_ts),
                swap.spread(),
                payment_adjustment=swap.payment_convention(),
                overnight_schedule=swap.overnight_schedule(),
            )

        # C++ calls ``swap->iborIndex()->clone(fwdTs_)`` on the ``IborIndex``
        # interface, which declares ``clone``; PQuantLib's
        # ``IborIndexProtocol`` does not, so narrow to the concrete class.
        index = self._index
        qassert.require(
            isinstance(index, IborIndex),
            "FdmAffineModelSwapInnerValue needs a cloneable IborIndex",
        )
        assert isinstance(index, IborIndex)
        return VanillaSwap(
            swap.swap_type(),
            swap.nominal(),
            swap.fixed_schedule(),
            swap.fixed_rate(),
            swap.fixed_day_count(),
            swap.floating_schedule(),
            index.clone(self._fwd_ts),
            swap.spread(),
            swap.floating_day_count(),
            swap.payment_convention(),
        )

    def _get_state(self, model: object, t: float, iterator: FdmLinearOpIterator) -> Array:
        """The model state implied by the grid node.

        # C++ parity: the two explicit ``getState`` specialisations in
        # fdmaffinemodelswapinnervalue.cpp — ``HullWhite`` maps the grid
        # coordinate through the short-rate dynamics; ``G2`` reads two
        # consecutive mesher directions verbatim.

        ``model`` is typed ``object`` on purpose: C++ restricts the admissible
        models at *link* time (only those two specialisations exist), so the
        Python equivalent is a runtime check that stays live even when a
        caller ignores the declared ``AffineSwapModel`` union.
        """
        if isinstance(model, HullWhite):
            return np.array(
                [
                    model.dynamics().short_rate(
                        t, self._mesher.location(iterator, self._direction)
                    )
                ],
                dtype=np.float64,
            )
        if isinstance(model, G2):
            return np.array(
                [
                    self._mesher.location(iterator, self._direction),
                    self._mesher.location(iterator, self._direction + 1),
                ],
                dtype=np.float64,
            )
        qassert.fail(
            "FdmAffineModelSwapInnerValue supports HullWhite and G2 only "
            "(the two C++ template specialisations)"
        )

    def inner_value(self, iterator: FdmLinearOpIterator, t: float) -> float:
        """Exercise value of the underlying swap at the node.

        # C++ parity: ``FdmAffineModelSwapInnerValue::innerValue``.
        """
        iter_exercise_date = self._exercise_dates[t]

        dis_rate = self._get_state(self._dis_model, t, iterator)
        fwd_rate = self._get_state(self._fwd_model, t, iterator)

        if self._dis_ts.empty() or iter_exercise_date != self._dis_ts.reference_date():
            discount: YieldTermStructure = self._dis_model.term_structure
            self._dis_ts.link_to(
                FdmAffineModelTermStructure(
                    dis_rate,
                    _calendar_or_none(discount),
                    discount.day_counter(),
                    iter_exercise_date,
                    discount.reference_date(),
                    self._dis_model,
                )
            )

            fwd: YieldTermStructure = self._fwd_model.term_structure
            self._fwd_ts.link_to(
                FdmAffineModelTermStructure(
                    fwd_rate,
                    _calendar_or_none(fwd),
                    fwd.day_counter(),
                    iter_exercise_date,
                    fwd.reference_date(),
                    self._fwd_model,
                )
            )
        else:
            self._dis_ts.current_link().set_variable(dis_rate)
            self._fwd_ts.current_link().set_variable(fwd_rate)

        npv = 0.0
        for j in range(2):
            for cf in self._swap.leg(j):
                # C++ parity: ``coupon_cast(i)->accrualStartDate()`` — a
                # ``dynamic_pointer_cast<Coupon>`` dereferenced without a null
                # check, i.e. C++ takes "every leg entry is a Coupon" as a
                # precondition. Both legs of a FixedVsFloatingSwap are coupons
                # by construction, so the assert only narrows the type.
                assert isinstance(cf, Coupon)
                npv += (
                    cf.amount() * self._dis_ts.discount(cf.date())
                    if cf.accrual_start_date() >= iter_exercise_date
                    else 0.0
                )
            if j == 0:
                npv *= -1.0
        if self._swap.swap_type() == SwapType.Receiver:
            npv *= -1.0

        return max(0.0, npv)

    def avg_inner_value(self, iterator: FdmLinearOpIterator, t: float) -> float:
        """Same as :meth:`inner_value`.

        # C++ parity: ``FdmAffineModelSwapInnerValue::avgInnerValue``.
        """
        return self.inner_value(iterator, t)


__all__ = ["AffineSwapModel", "FdmAffineModelSwapInnerValue"]

"""Utility class to manage a cycle of mutually dependent curves.

# C++ parity: ql/termstructures/multicurve.{hpp,cpp} (v1.43)

A "multi curve" is a set of yield curves that cannot be built one after
another because each needs the others: a 3M forwarding curve whose swap
helpers discount off an OIS curve that is itself a spread over the 3M
curve; a 3M/6M pair joined by basis swaps. :class:`MultiCurve` builds
such a cycle by handing every member's residuals to ONE optimizer, via
:class:`~pquantlib.termstructures.global_bootstrap.MultiCurveBootstrap`.

The recipe (C++ multicurve.hpp:41-78, restated for this port):

1. Create an :class:`InternalCurveLink` for every member that something
   else has to reference BEFORE it exists.
2. Build each member, pointing at those links.
3. Build the :class:`MultiCurve`.
4. Register each member: :meth:`MultiCurve.add_bootstrapped_curve` for
   curves driven by a
   :class:`~pquantlib.termstructures.global_bootstrap.GlobalBootstrap`,
   :meth:`MultiCurve.add_non_bootstrapped_curve` for everything else
   (spreaded curves, say). Both take the member's link and the member.

Divergence from C++: no ``Handle``
----------------------------------

The C++ class is, mechanically, about handle relinking and about breaking
``shared_ptr`` ownership cycles. pquantlib has NEITHER ``Handle`` nor
``RelinkableHandle`` — they are allowlisted in
``migration-harness/check_coverage.py`` with the documented approach
"thread the pointed-to object directly and re-register observers
explicitly". So this class is re-derived rather than transliterated:

- **``RelinkableHandle<YieldTermStructure>`` →**
  :class:`InternalCurveLink`. C++ needs a handle because a curve must be
  referenceable before it is constructed; that need survives the port, so
  the link survives, but it is a forwarding ``YieldTermStructure``
  rather than a smart-pointer indirection. Exactly as in
  ``linkTo(..., registerAsObserver=false)`` (multicurve.cpp:58), the link
  does NOT observe its target — that is what keeps notifications from
  going round the cycle forever.
- **``null_deleter``** (multicurve.cpp:58) exists so the internal handle
  can point at a curve it does not own. Python reference semantics make
  ownership a non-question; there is nothing to port and nothing faked.
- **``ext::enable_shared_from_this``** (multicurve.hpp:82) exists so
  ``addCurve`` can build an aliasing ``shared_ptr`` that co-owns the
  MultiCurve. See :meth:`MultiCurve.add_bootstrapped_curve` for how the
  "every cycle member stays alive as long as any one of them is" property
  is preserved without it.
- **The external handle** the C++ methods return is, in this port, the
  curve itself.

What is genuinely portable — and what the cross-validation tests pin — is
the RESULT: a cycle of curves that reprices every member's helpers, with
the spread relation between members holding exactly.
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pquantlib import qassert
from pquantlib.termstructures.global_bootstrap import (
    MultiCurveBootstrap,
    MultiCurveBootstrapContributor,
)
from pquantlib.termstructures.yield_term_structure import YieldTermStructure

if TYPE_CHECKING:
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.math.optimization.end_criteria import EndCriteria
    from pquantlib.math.optimization.optimization_method import OptimizationMethod
    from pquantlib.time.date import Date


class MultiCurveBootstrapProvider(ABC):
    """A curve that can contribute to a joint multi-curve solve.

    # C++ parity: ``struct MultiCurveBootstrapProvider`` at
    # multicurve.hpp:36-39. ``PiecewiseYieldCurve`` implements it by
    # returning the address of its own bootstrap when that bootstrap is a
    # ``MultiCurveBootstrapContributor``, and ``nullptr`` otherwise
    # (piecewiseyieldcurve.hpp:148-154).
    """

    @abstractmethod
    def multi_curve_bootstrap_contributor(
        self,
    ) -> MultiCurveBootstrapContributor | None:
        """C++ multicurve.hpp:38 — ``None`` when the curve is not globally
        bootstrapped."""


class InternalCurveLink(YieldTermStructure):
    """Forward reference to a curve that does not exist yet.

    The pquantlib stand-in for ``RelinkableHandle<YieldTermStructure>``
    (multicurve.hpp:91). Construct one, point the cycle's other members at
    it, and :class:`MultiCurve` links it to the real curve when that curve
    is registered.

    Deliberately NOT an observer of its target: C++ links the internal
    handle with ``registerAsObserver = false`` (multicurve.cpp:58-59) so
    that notifications do not circulate around the cycle. Refreshing the
    cycle is :meth:`MultiCurve.update`'s job, not the link's.
    """

    def __init__(self) -> None:
        # No reference date / day counter of its own — every query is
        # forwarded, so a link is only usable once it has been linked.
        super().__init__()
        self._target: YieldTermStructure | None = None

    # -- linking -----------------------------------------------------------

    def link_to(self, target: YieldTermStructure) -> None:
        """C++ ``internalHandle.linkTo(curve, false)`` — multicurve.cpp:58."""
        self._target = target

    def empty(self) -> bool:
        """C++ ``Handle::empty()`` — multicurve.cpp:35."""
        return self._target is None

    def target(self) -> YieldTermStructure:
        """The linked curve; raises while the link is still empty."""
        qassert.require(
            self._target is not None,
            "InternalCurveLink: not linked yet — the curve must be added to "
            "its MultiCurve before it can be evaluated",
        )
        assert self._target is not None
        return self._target

    # -- forwarded TermStructure interface ---------------------------------

    def reference_date(self) -> Date:
        return self.target().reference_date()

    def max_date(self) -> Date:
        return self.target().max_date()

    def day_counter(self) -> DayCounter:
        return self.target().day_counter()

    def time_from_reference(self, d: Date) -> float:
        return self.target().time_from_reference(d)

    def allows_extrapolation(self) -> bool:
        return self.target().allows_extrapolation()

    def _discount_impl(self, t: float) -> float:
        target: Any = self.target()
        return float(target._discount_impl(t))


class MultiCurve:
    """Builds a set of curves that form a dependency cycle.

    # C++ parity: ``class MultiCurve`` at multicurve.hpp:79-104 and
    # multicurve.cpp:25-77.

    C++ derives from ``Observer`` (and from ``enable_shared_from_this``);
    this port keeps the observer role — :meth:`update` is the notification
    fan-out that keeps every member in step — and drops the second base,
    which has no Python meaning.
    """

    def __init__(
        self,
        accuracy: float | None = None,
        optimizer: OptimizationMethod | None = None,
        end_criteria: EndCriteria | None = None,
    ) -> None:
        """Merge of the two C++ constructors (multicurve.cpp:25-30).

        Both simply forward to the matching ``MultiCurveBootstrap``
        constructor, so the merged signature is the merged signature of
        those two.
        """
        self._multi_curve_bootstrap: MultiCurveBootstrap = MultiCurveBootstrap(
            accuracy, optimizer, end_criteria
        )
        # C++ ``curves_`` (multicurve.hpp:103) — strong references to every
        # member, which is what keeps the cycle alive.
        self._curves: list[Any] = []
        # Re-entrancy latch, see ``update``.
        self._updating: bool = False

    # -- registration ------------------------------------------------------

    def add_bootstrapped_curve(
        self,
        internal_link: InternalCurveLink | None,
        curve: Any,
        contributor: MultiCurveBootstrapContributor | None = None,
    ) -> Any:
        """Register a globally bootstrapped member of the cycle.

        # C++ parity: ``MultiCurve::addBootstrappedCurve``, multicurve.cpp:32-43.

        C++ recovers the contributor with
        ``dynamic_pointer_cast<MultiCurveBootstrapProvider>(curve)``,
        because in C++ the bootstrap is a MEMBER of the curve. pquantlib's
        ``PiecewiseYieldCurve`` hard-codes ``IterativeBootstrap`` as its
        bootstrap (piecewise_yield_curve.py:30-33 documents the carve-out),
        so a ``GlobalBootstrap`` is driven alongside the curve rather than
        from inside it; pass it as ``contributor`` when the curve is not
        itself a :class:`MultiCurveBootstrapProvider`.

        ``internal_link`` may be ``None`` when nothing in the cycle had to
        reference this curve before it existed — a case C++ cannot have,
        since there the handle is how the curve is referenced at all.
        """
        if contributor is None:
            qassert.require(
                isinstance(curve, MultiCurveBootstrapProvider),
                "curve must be a MultiCurveBootstrapProvider",
            )
            provider: MultiCurveBootstrapProvider = curve
            found = provider.multi_curve_bootstrap_contributor()
            qassert.require(
                found is not None,
                "curve does not provide a valid multi curve bootstrap contributor",
            )
            assert found is not None
            contributor = found
        # C++ parity: multicurve.cpp:41.
        self._multi_curve_bootstrap.add(contributor)
        return self._add_curve(internal_link, curve)

    def add_non_bootstrapped_curve(
        self, internal_link: InternalCurveLink | None, curve: Any
    ) -> Any:
        """Register a member that is not bootstrapped (e.g. a spreaded curve).

        # C++ parity: ``MultiCurve::addNonBootstrappedCurve``,
        # multicurve.cpp:45-53. Such a curve takes part in the solve only as
        # an OBSERVER: it is refreshed after every cost-function argument
        # update so that the bootstrapped members see it up to date
        # (globalbootstrap.cpp:74-75).
        """
        qassert.require(curve is not None, "curve must not be null")
        # C++ parity: multicurve.cpp:51.
        self._multi_curve_bootstrap.add_observer(curve)
        return self._add_curve(internal_link, curve)

    def _add_curve(self, internal_link: InternalCurveLink | None, curve: Any) -> Any:
        """C++ parity: ``MultiCurve::addCurve``, multicurve.cpp:55-71."""
        if internal_link is not None:
            # C++ parity: multicurve.cpp:35 / :48 — an already-linked handle
            # means the curve was added twice.
            qassert.require(
                internal_link.empty(),
                "internal handle must be empty; was the curve added already?",
            )
            internal_link.link_to(curve)

        # C++ parity: multicurve.cpp:68 — ``registerWithObservables(curve)``,
        # i.e. the MultiCurve observes the member so that a quote change on
        # any member is fanned out to all of them by ``update``.
        register_with = getattr(curve, "register_with", None)
        if callable(register_with):
            register_with(self)

        # C++ parity: multicurve.cpp:69 — ``curves_.push_back(curve)``.
        self._curves.append(curve)

        # C++ builds an aliasing shared_ptr so the returned external handle
        # CO-OWNS the MultiCurve (multicurve.cpp:60-67); that is what
        # guarantees "all member curves are kept alive until none of the
        # curves and the MultiCurve itself is referenced" (multicurve.hpp:74).
        # pquantlib's Observable holds its observers weakly
        # (patterns/observer.py:27), so registration alone would NOT keep the
        # MultiCurve alive. A back-reference on the curve reproduces the C++
        # co-ownership: holding any member keeps the MultiCurve alive, which
        # keeps every other member alive. Python's cycle collector reclaims
        # the whole group once nothing outside it refers to any member.
        # A slotted curve cannot carry the back-reference; the caller then has
        # to keep the MultiCurve alive itself.
        with contextlib.suppress(AttributeError):
            curve._multi_curve_owner = self

        return curve

    # -- Observer ----------------------------------------------------------

    def update(self) -> None:
        """Fan a notification out to every member of the cycle.

        # C++ parity: ``MultiCurve::update``, multicurve.cpp:73-76.

        C++ terminates because each member is a ``LazyObject`` whose
        ``update()`` only re-notifies when it was previously calculated, so
        the second lap round the cycle is silent. pquantlib's
        ``TermStructure.update`` notifies unconditionally
        (term_structure.py:143-153), so the same "notify each member once"
        semantics need an explicit latch here.
        """
        if self._updating:
            return
        self._updating = True
        try:
            for c in self._curves:
                c.update()
        finally:
            self._updating = False

    # -- inspectors --------------------------------------------------------

    def curves(self) -> list[Any]:
        """C++ ``curves_`` (multicurve.hpp:103)."""
        return list(self._curves)

    def bootstrap(self) -> MultiCurveBootstrap:
        """C++ ``multiCurveBootstrap_`` (multicurve.hpp:102)."""
        return self._multi_curve_bootstrap


__all__ = ["InternalCurveLink", "MultiCurve", "MultiCurveBootstrapProvider"]

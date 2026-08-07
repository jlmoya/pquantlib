"""DefaultProbKey — index key for default-probability curves per issuer.

# C++ parity: ql/experimental/credit/defaultprobabilitykey.{hpp,cpp} (v1.42.1).

The key aggregates the (currency, seniority, set of event types)
contractual conditions that determine which default-probability term
structure applies. Used as a dictionary key by ``Issuer``.

``NorthAmericaCorpDefaultKey`` is the ISDA standard key for North-American
corporate US debt: FailureToPay + Bankruptcy + optional Restructuring
event types.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pquantlib import qassert
from pquantlib.currencies.currency import Currency
from pquantlib.experimental.credit.default_type import (
    AtomicDefault,
    DefaultType,
    FailureToPay,
    Restructuring,
    Seniority,
)
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


@dataclass(frozen=True, slots=True)
class DefaultProbKey:
    """Index key for default-probability curves.

    Fields:
      - ``event_types``: list of contractual ``DefaultType`` instances
        the contract is sensitive to. Must have unique atomic-default
        types (else ``LibraryException``).
      - ``currency``: bond + protection-leg currency.
      - ``seniority``: reference-bonds seniority.

    Equality follows C++ ``operator==``: same seniority + currency +
    event-type set (set comparison via element-wise DefaultType equality).
    """

    event_types: tuple[DefaultType, ...] = field(default_factory=tuple)
    currency: Currency = field(default_factory=Currency)
    seniority: Seniority = Seniority.NoSeniority

    def __post_init__(self) -> None:
        # Reject duplicated atomic-default types in event_types.
        # # C++ parity: defaultprobabilitykey.cpp:65-70.
        atomic_set: set[AtomicDefault] = set()
        for et in self.event_types:
            qassert.require(
                et.default_type not in atomic_set,
                "Duplicated event type in contract definition",
            )
            atomic_set.add(et.default_type)

    def size(self) -> int:
        """Number of event types this contract is sensitive to."""
        return len(self.event_types)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DefaultProbKey):
            return NotImplemented
        if self.seniority != other.seniority:
            return False
        if self.currency != other.currency:
            return False
        if len(self.event_types) != len(other.event_types):
            return False
        # All my types must appear in other (and vice-versa via size match).
        # # C++ parity: defaultprobabilitykey.cpp:44-57.
        return all(et in self.event_types for et in other.event_types)

    def __hash__(self) -> int:
        # Hash on a frozenset of (default_type, restructuring_type) to
        # honour set-equality semantics regardless of event_types order.
        return hash(
            (
                self.seniority,
                self.currency,
                frozenset(
                    (et.default_type, et.restructuring_type) for et in self.event_types
                ),
            )
        )


class NorthAmericaCorpDefaultKey(DefaultProbKey):
    """ISDA standard contractual default key for North-American corporate US debt.

    # C++ parity: ql/experimental/credit/defaultprobabilitykey.hpp:71-81 +
    # .cpp:73-91 (v1.43).

    Always includes ``FailureToPay(grace, amount)`` then
    ``Bankruptcy(XR)``. Appends a ``Restructuring(restructuring_type)``
    entry iff ``restructuring_type != NoRestructuring`` — so the key holds
    three event types by default and two when restructuring is switched off.

    Equality and hashing come from :class:`DefaultProbKey`, so an instance of
    this class compares equal to a hand-built ``DefaultProbKey`` carrying the
    same currency / seniority / event-type set (matching the C++ free
    ``operator==``, which takes ``DefaultProbKey`` references).
    """

    __slots__ = ()

    def __init__(
        self,
        currency: Currency,
        seniority: Seniority,
        grace_failure_to_pay: Period | None = None,
        amount_failure: float = 1.0e6,
        restructuring_type: Restructuring = Restructuring.FullRestructuring,
    ) -> None:
        # # C++ parity note: the C++ default for ``graceFailureToPay`` is
        # ``Period(30, Days)`` and for ``resType`` is ``Restructuring::CR``
        # (== FullRestructuring). ``Period()`` — the null period the C++
        # test-suite passes explicitly — is a *different* value and is
        # reachable here by passing ``Period()``, not by omitting the arg.
        grace = (
            grace_failure_to_pay
            if grace_failure_to_pay is not None
            else Period(30, TimeUnit.Days)
        )
        events: list[DefaultType] = [
            FailureToPay(grace_period=grace, amount_required=amount_failure),
            # No specifics for Bankruptcy.
            DefaultType(AtomicDefault.Bankruptcy, Restructuring.NoRestructuring),
        ]
        if restructuring_type != Restructuring.NoRestructuring:
            events.append(DefaultType(AtomicDefault.Restructuring, restructuring_type))
        super().__init__(
            event_types=tuple(events),
            currency=currency,
            seniority=seniority,
        )


def make_north_america_corp_default_key(
    currency: Currency,
    seniority: Seniority,
    grace_failure_to_pay: Period | None = None,
    amount_failure: float = 1.0e6,
    restructuring_type: Restructuring = Restructuring.FullRestructuring,
) -> DefaultProbKey:
    """Deprecated factory kept for callers written before the class existed.

    Prefer :class:`NorthAmericaCorpDefaultKey` — it is the exact C++ name.
    """
    return NorthAmericaCorpDefaultKey(
        currency,
        seniority,
        grace_failure_to_pay,
        amount_failure,
        restructuring_type,
    )


__all__ = [
    "DefaultProbKey",
    "NorthAmericaCorpDefaultKey",
    "make_north_america_corp_default_key",
]

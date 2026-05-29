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


def make_north_america_corp_default_key(
    currency: Currency,
    seniority: Seniority,
    grace_failure_to_pay: Period | None = None,
    amount_failure: float = 1.0e6,
    restructuring_type: Restructuring = Restructuring.FullRestructuring,
) -> DefaultProbKey:
    """ISDA standard contractual default key for North-American corporate US debt.

    # C++ parity: ql/experimental/credit/defaultprobabilitykey.cpp:73 —
    # the C++ NorthAmericaCorpDefaultKey is a DefaultProbKey subclass that
    # populates the event_types vector. Python collapses this to a free
    # factory function since the result is just a populated DefaultProbKey
    # (no behavioural override).

    Always includes FailureToPay(grace, amount) + Bankruptcy(XR). Adds a
    Restructuring(restructuring_type) entry iff
    ``restructuring_type != NoRestructuring``.
    """
    grace = (
        grace_failure_to_pay
        if grace_failure_to_pay is not None
        else Period(30, TimeUnit.Days)
    )
    events: list[DefaultType] = [
        FailureToPay(grace_period=grace, amount_required=amount_failure),
        DefaultType(AtomicDefault.Bankruptcy, Restructuring.NoRestructuring),
    ]
    if restructuring_type != Restructuring.NoRestructuring:
        events.append(DefaultType(AtomicDefault.Restructuring, restructuring_type))
    return DefaultProbKey(
        event_types=tuple(events),
        currency=currency,
        seniority=seniority,
    )

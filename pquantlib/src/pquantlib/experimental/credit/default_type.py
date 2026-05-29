"""DefaultType — atomic credit-event types.

# C++ parity: ql/experimental/credit/defaulttype.{hpp,cpp} (v1.42.1).

The C++ header defines three things:

- ``Seniority`` enum with bond seniorities (SecDom..PrefT1 + NoSeniority).
- ``AtomicDefault::Type`` enum with ISDA atomic-default types
  (Restructuring..MergerEvent + ObligationAcceleration / ObligationDefault
  / CrossDefault synonyms).
- ``Restructuring::Type`` enum with restructuring sub-types
  (NoRestructuring..AnyRestructuring + Markit XR/MR/MM/CR synonyms).
- ``DefaultType`` class wrapping (AtomicDefault::Type, Restructuring::Type)
  with equality on both fields + ``isRestructuring()`` + containment
  predicates.
- ``FailureToPay`` subclass adding a grace ``Period`` + minimum amount.

Python port: each enum is an ``IntEnum`` (so integer indices match C++
when serialised through the probe harness). C++ uses ``struct
AtomicDefault { enum Type { ... } }`` and ``struct Restructuring { enum
Type { ... } }`` as nested-enum carriers — Python collapses these to
flat IntEnum classes named ``AtomicDefault`` and ``Restructuring``.

``DefaultType`` and ``FailureToPay`` are plain frozen-slots dataclasses
(no Observable parent in C++ either). Equality on ``DefaultType`` is the
C++ pair-equality of (default-type, restructuring-type).

# C++ parity divergence: AtomicDefault::ObligationAcceleration ==
# Acceleration, ObligationDefault == Default, CrossDefault == Default
# are encoded as class-attributes that point at the same IntEnum
# instance (Python IntEnum forbids aliased members in 3.11+ unless
# given a different name, so we expose the aliases as plain class vars
# rather than enum members).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from pquantlib.time.period import Period


class Seniority(IntEnum):
    """ISDA bond seniorities.

    Markit synonyms (SeniorSec, SeniorUnSec, SubTier1, SubUpperTier2,
    SubLoweTier2) are exposed as class-attribute aliases below.
    """

    SecDom = 0
    SnrFor = 1
    SubLT2 = 2
    JrSubT2 = 3
    PrefT1 = 4
    # Unassigned value, allows for default RR quote
    NoSeniority = 5


# Markit-parlance aliases. Python IntEnum disallows duplicate-name
# aliases as members; expose them as plain class-attribute pointers.
Seniority.SeniorSec = Seniority.SecDom  # type: ignore[attr-defined]
Seniority.SeniorUnSec = Seniority.SnrFor  # type: ignore[attr-defined]
Seniority.SubTier1 = Seniority.PrefT1  # type: ignore[attr-defined]
Seniority.SubUpperTier2 = Seniority.JrSubT2  # type: ignore[attr-defined]
Seniority.SubLoweTier2 = Seniority.SubLT2  # type: ignore[attr-defined]


class AtomicDefault(IntEnum):
    """Atomic (single-event) default types per ISDA.

    Aliases ``ObligationAcceleration``/``ObligationDefault``/
    ``CrossDefault`` collapse onto the same value as C++
    ``AtomicDefault::Type``.
    """

    # Includes one of the restructuring cases
    Restructuring = 0
    Bankruptcy = 1
    FailureToPay = 2
    RepudiationMoratorium = 3
    Acceleration = 4
    Default = 5
    # Other non-isda
    Downgrade = 6
    MergerEvent = 7


# Synonym aliases — exposed as class attributes (not enum members) to
# avoid IntEnum's duplicate-value-with-different-name restriction.
AtomicDefault.ObligationAcceleration = AtomicDefault.Acceleration  # type: ignore[attr-defined]
AtomicDefault.ObligationDefault = AtomicDefault.Default  # type: ignore[attr-defined]
AtomicDefault.CrossDefault = AtomicDefault.Default  # type: ignore[attr-defined]


class Restructuring(IntEnum):
    """Restructuring sub-types per ISDA / Markit notation."""

    NoRestructuring = 0
    ModifiedRestructuring = 1
    ModifiedModifiedRestructuring = 2
    FullRestructuring = 3
    AnyRestructuring = 4


# Markit notation aliases.
Restructuring.XR = Restructuring.NoRestructuring  # type: ignore[attr-defined]
Restructuring.MR = Restructuring.ModifiedRestructuring  # type: ignore[attr-defined]
Restructuring.MM = Restructuring.ModifiedModifiedRestructuring  # type: ignore[attr-defined]
Restructuring.CR = Restructuring.FullRestructuring  # type: ignore[attr-defined]


@dataclass(frozen=True, slots=True)
class DefaultType:
    """Atomic credit-event type.

    Wraps an (AtomicDefault, Restructuring) pair. Equality is on both
    fields (mirrors C++ ``operator==``).
    """

    default_type: AtomicDefault = AtomicDefault.Bankruptcy
    restructuring_type: Restructuring = Restructuring.NoRestructuring

    def is_restructuring(self) -> bool:
        """True iff ``restructuring_type != NoRestructuring``."""
        return self.restructuring_type != Restructuring.NoRestructuring

    def contains_default_type(self, def_type: AtomicDefault) -> bool:
        """True iff ``def_type`` matches our atomic default type.

        # C++ parity: DefaultType::containsDefaultType(AtomicDefault::Type)
        # at defaulttype.hpp:134 — strict equality, no event-hierarchy logic.
        """
        return self.default_type == def_type

    def contains_restructuring_type(self, res_type: Restructuring) -> bool:
        """True iff ``res_type`` matches our restructuring type, or is AnyRestructuring.

        # C++ parity: DefaultType::containsRestructuringType at defaulttype.hpp:138.
        # AnyRestructuring is a wildcard.
        """
        return res_type in (self.restructuring_type, Restructuring.AnyRestructuring)


_NULL_PERIOD: Period = Period()


@dataclass(frozen=True, slots=True)
class FailureToPay(DefaultType):
    """Failure-to-pay credit-event type with grace period + minimum amount.

    # C++ parity: ql/experimental/credit/defaulttype.hpp:165 (FailureToPay)
    # — fixes default_type to FailureToPay + restructuring to XR.

    Hand-rolls ``__init__`` so the two base-class fields are pinned to
    FailureToPay/NoRestructuring regardless of constructor args.
    """

    grace_period: Period = _NULL_PERIOD
    amount_required: float = 1.0e6

    def __init__(
        self,
        grace_period: Period | None = None,
        amount_required: float = 1.0e6,
    ) -> None:
        # Frozen dataclass requires object.__setattr__.
        object.__setattr__(self, "default_type", AtomicDefault.FailureToPay)
        object.__setattr__(
            self, "restructuring_type", Restructuring.NoRestructuring
        )
        object.__setattr__(
            self,
            "grace_period",
            grace_period if grace_period is not None else _NULL_PERIOD,
        )
        object.__setattr__(self, "amount_required", amount_required)

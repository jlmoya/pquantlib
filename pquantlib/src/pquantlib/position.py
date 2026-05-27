"""Position type — Long / Short discriminant.

# C++ parity: ql/position.hpp (v1.42.1)
#
#     struct Position {
#         enum Type { Long, Short };
#     };
#
# Python idiomatically uses IntEnum (matching ``OptionType`` in
# pquantlib.payoffs) so it composes with the same idioms used by
# ForwardTypePayoff and ForwardRateAgreement.
"""

from __future__ import annotations

from enum import IntEnum


class PositionType(IntEnum):
    """Long / Short discriminant for forward-style payoffs.

    # C++ parity: ``Position::Type`` (ql/position.hpp). Integer values
    # match the C++ enum order: Long=0, Short=1.
    """

    Long = 0
    Short = 1

    def __str__(self) -> str:
        return "Long" if self == PositionType.Long else "Short"


__all__ = ["PositionType"]

"""Global library settings (singleton).

# C++ parity: ql/settings.hpp (v1.42.1) — class Settings::instance().

The C++ ``Settings`` singleton carries a handful of boolean flags that
affect library-wide behavior (e.g. enforcement of business-day conventions
during schedule generation, payment-date inclusion semantics). This module
seeds the most common flags; further flags land alongside the consumers
that need them in later L1 stages and Phase 2+.
"""

from __future__ import annotations

from pquantlib.patterns.singleton import Singleton


class ObservableSettings(Singleton):
    """Library-wide mutable flags."""

    enforces_business_day_convention: bool = True
    include_today_in_payments: bool = False
    include_reference_date_events: bool = True

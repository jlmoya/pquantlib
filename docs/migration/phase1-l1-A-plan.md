# Phase 1 L1-A Implementation Plan — Pilot cluster

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:subagent-driven-development`.
> Use one worktree (`../pquantlib-phase1-A`). Stages are sequential within
> L1-A; tasks within a stage may be parallelized only if they touch disjoint
> module trees.

**Goal:** Land all foundations + time + daycounters + calendars + first math batch on `main`, behind tag `pquantlib-phase1-l1-A-complete`.

**Architecture:** Five sequential stages (0=harness, 1=foundations, 2=time, 3=daycounters, 4=calendars, 5=first math). Each stage ends with a `feat(<stage>): close L1-A stage N` checkpoint commit.

**Tech Stack:** Python 3.14, numpy 2.4+, mpmath 1.4+, pytest 9+, pyright 1.1+, ruff 0.15+, C++ QuantLib v1.42.1 @ `099987f0`.

**Date:** 2026-05-23
**Predecessor:** `pquantlib-phase0-bootstrap` @ `7e2c6ec`

---

## Task 0 — Spawn worktree

```bash
cd /Users/josemoya/Projects/PycharmProjects/pquantlib
git worktree add -b phase1-A ../pquantlib-phase1-A main
cd ../pquantlib-phase1-A
uv sync
uv run pytest  # confirm 2/0/0 baseline
```

DoD: worktree exists at `../pquantlib-phase1-A`, baseline tests pass.

---

## Stage 0 — Harness bootstrap

### Task 0.1 — Clone QuantLib submodule

```bash
cd ../pquantlib-phase1-A
git submodule add https://github.com/lballabio/QuantLib.git migration-harness/cpp/quantlib
cd migration-harness/cpp/quantlib
git checkout 099987f0ca2c11c505dc4348cdb9ce01a598e1e5
cd ../../..
git submodule update --init --recursive
```

Verify: `migration-harness/cpp/quantlib/ql/version.hpp` exists and reports v1.42.1.

### Task 0.2 — Build C++ QuantLib

```bash
./migration-harness/build-cpp.sh
```

Expected: `migration-harness/cpp/build/libQuantLib.a` (or `.dylib`) exists.

If this fails: trigger A_LA_2 (likely missing boost/cmake/clang). Document the environment fix and pause.

### Task 0.3 — Write the first sentinel probe

File: `migration-harness/cpp/probes/sentinel/sentinel_probe.cpp`

```cpp
// First probe — proves the harness emits parseable JSON
#include <ql/version.hpp>
#include <iostream>
#include <iomanip>
int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";
    std::cout << "  \"quantlib_version\": \"" << QL_VERSION << "\",\n";
    std::cout << "  \"sentinel_value\": " << 1.4142135623730951 << "\n";
    std::cout << "}\n";
    return 0;
}
```

Add to `migration-harness/cpp/probes/CMakeLists.txt`.

```bash
./migration-harness/generate-references.sh sentinel/sentinel_probe
cat migration-harness/references/sentinel/sentinel.json
```

Commit: `infra(harness): bootstrap C++ submodule + sentinel probe`

DoD: `migration-harness/references/sentinel/sentinel.json` exists, parses as valid JSON, contains `"quantlib_version": "1.42.1"` and the `sqrt(2)` sentinel.

---

## Stage 1 — Foundations

### Task 1.1 — `pquantlib.exceptions`

File: `pquantlib/src/pquantlib/exceptions.py`

```python
"""Exception hierarchy.

# C++ parity: ql/errors.hpp (v1.42.1)
"""

from __future__ import annotations


class LibraryException(RuntimeError):
    """Base class for all PQuantLib exceptions.

    Mirrors C++ ``QL::Error``. Carries an optional cause and an optional
    source-location hint. Construction has NO side effects (no stderr
    print, no logging) — historical bug from jquantlib pre-de95bb17.
    """

    def __init__(self, message: str, *, where: str | None = None) -> None:
        super().__init__(message)
        self.where = where
```

File: `pquantlib/src/pquantlib/qassert.py`

```python
"""C++ ``QL_REQUIRE`` / ``QL_FAIL`` analogues as free functions.

# C++ parity: ql/errors.hpp QL_REQUIRE / QL_FAIL macros (v1.42.1)
"""

from __future__ import annotations

from typing import NoReturn

from pquantlib.exceptions import LibraryException


def require(condition: object, message: str, *, where: str | None = None) -> None:
    if not condition:
        raise LibraryException(message, where=where)


def fail(message: str, *, where: str | None = None) -> NoReturn:
    raise LibraryException(message, where=where)
```

Tests: `pquantlib/tests/test_exceptions.py`, `pquantlib/tests/test_qassert.py`.
- `LibraryException` constructs without printing (capture stderr, assert empty).
- `require(False, "msg")` raises `LibraryException("msg")`.
- `require(True, "msg")` returns None.
- `fail("msg")` raises `LibraryException("msg")`.

Commit: `feat(exceptions): port LibraryException + qassert helpers`

### Task 1.2 — `pquantlib.testing.tolerance`

File: `pquantlib/src/pquantlib/testing/__init__.py` (empty + `py.typed`)
File: `pquantlib/src/pquantlib/testing/tolerance.py`

```python
"""Tolerance-tier assertions for cross-validation tests.

# C++ parity: none — this is harness, not a port.

Tiers:
- EXACT: bit-identical via ``struct.pack('!d', ...)``.
- TIGHT: ``math.isclose(abs_tol=1e-14, rel_tol=1e-12)``.
- LOOSE: ``math.isclose(abs_tol=1e-8, rel_tol=1e-8)``.

Per-test exceptions go through ``custom(..., reason="...")`` with a
mandatory written rationale.
"""

from __future__ import annotations

import math
import struct
from typing import Final

_TIGHT_ABS: Final[float] = 1e-14
_TIGHT_REL: Final[float] = 1e-12
_LOOSE_ABS: Final[float] = 1e-8
_LOOSE_REL: Final[float] = 1e-8


def exact(actual: float, expected: float, *, reason: str | None = None) -> None:
    if struct.pack("!d", actual) != struct.pack("!d", expected):
        raise AssertionError(
            f"EXACT tier mismatch: actual={actual!r} expected={expected!r}"
            + (f" (reason: {reason})" if reason else "")
        )


def tight(actual: float, expected: float, *, reason: str | None = None) -> None:
    if not math.isclose(actual, expected, abs_tol=_TIGHT_ABS, rel_tol=_TIGHT_REL):
        raise AssertionError(
            f"TIGHT tier mismatch: actual={actual!r} expected={expected!r}"
            + (f" (reason: {reason})" if reason else "")
        )


def loose(actual: float, expected: float, *, reason: str | None = None) -> None:
    if not math.isclose(actual, expected, abs_tol=_LOOSE_ABS, rel_tol=_LOOSE_REL):
        raise AssertionError(
            f"LOOSE tier mismatch: actual={actual!r} expected={expected!r}"
            + (f" (reason: {reason})" if reason else "")
        )


def custom(
    actual: float,
    expected: float,
    *,
    abs_tol: float,
    rel_tol: float,
    reason: str,
) -> None:
    if not math.isclose(actual, expected, abs_tol=abs_tol, rel_tol=rel_tol):
        raise AssertionError(
            f"custom tier mismatch (abs_tol={abs_tol}, rel_tol={rel_tol}, "
            f"reason: {reason}): actual={actual!r} expected={expected!r}"
        )
```

Tests: `pquantlib/tests/testing/test_tolerance.py`.
- `exact(0.1+0.2, 0.3)` raises (FP non-associativity).
- `tight(0.1+0.2, 0.3)` passes.
- `loose(1.0, 1.00000001)` passes.
- `loose(1.0, 1.0001)` raises.
- `custom(...)` requires `reason`.

Commit: `feat(testing): add tolerance-tier assertion helpers`

### Task 1.3 — `pquantlib.testing.reference_reader`

File: `pquantlib/src/pquantlib/testing/reference_reader.py`

```python
"""Load JSON reference values produced by the C++ probe harness.

# C++ parity: none — this is harness, not a port.

Probe outputs live at:
    migration-harness/references/<topic>/<class>.json

Resolution: search upward from the test file's location to find the
``migration-harness/references/`` directory. This makes the loader
work both in the main worktree and in cluster worktrees.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_HARNESS_REL = Path("migration-harness") / "references"


def _find_references_root(start: Path) -> Path:
    cur = start.resolve()
    for ancestor in (cur, *cur.parents):
        candidate = ancestor / _HARNESS_REL
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find {_HARNESS_REL} starting from {start}"
    )


def load(key: str, *, start: Path | None = None) -> dict[str, Any]:
    """Load ``migration-harness/references/<key>.json`` as a dict.

    ``key`` is the topic/class path without extension, e.g. ``"math/beta"``.
    ``start`` defaults to this file's directory.
    """
    root = _find_references_root(start if start is not None else Path(__file__).parent)
    path = root / f"{key}.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
```

Tests: `pquantlib/tests/testing/test_reference_reader.py`.
- Load the `sentinel/sentinel.json` written in Stage 0 and assert version + sqrt(2) sentinel using tolerance helpers.
- Loading a missing key raises `FileNotFoundError`.

Commit: `feat(testing): add reference_reader loader for C++ probe JSONs`

### Task 1.4 — `pquantlib.patterns`

File: `pquantlib/src/pquantlib/patterns/__init__.py`
File: `pquantlib/src/pquantlib/patterns/observer.py`

```python
"""Observer / Observable pattern.

# C++ parity: ql/patternsobservable.hpp + ql/utilities/observablesettings.hpp (v1.42.1)
"""

from __future__ import annotations

import weakref
from typing import Protocol, runtime_checkable


@runtime_checkable
class Observer(Protocol):
    def update(self) -> None: ...


class Observable:
    """Observable subject. Holds weak references to observers."""

    def __init__(self) -> None:
        self._observers: weakref.WeakSet[Observer] = weakref.WeakSet()

    def register_with(self, observer: Observer) -> None:
        self._observers.add(observer)

    def unregister_with(self, observer: Observer) -> None:
        self._observers.discard(observer)

    def notify_observers(self) -> None:
        for obs in list(self._observers):
            obs.update()
```

File: `pquantlib/src/pquantlib/patterns/singleton.py`

```python
"""Singleton metaclass.

# C++ parity: ql/patterns/singleton.hpp (v1.42.1)
"""

from __future__ import annotations

from typing import Any, ClassVar


class _SingletonMeta(type):
    _instances: ClassVar[dict[type, Any]] = {}

    def __call__(cls, *args: object, **kwargs: object) -> Any:  # noqa: ANN401
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class Singleton(metaclass=_SingletonMeta):
    """Subclass to get singleton semantics: ``S() is S()`` is ``True``."""
```

File: `pquantlib/src/pquantlib/patterns/lazy_object.py`

```python
"""Lazy-evaluation mixin.

# C++ parity: ql/patterns/lazyobject.hpp (v1.42.1)
"""

from __future__ import annotations

from abc import abstractmethod

from pquantlib.patterns.observer import Observable


class LazyObject(Observable):
    """Subclass and implement ``_perform_calculations``.

    Call ``calculate()`` to trigger calculation if not yet done.
    Override ``update()`` (from Observer protocol via Observable) to
    invalidate the cache and trigger re-notification.
    """

    def __init__(self) -> None:
        super().__init__()
        self._calculated: bool = False

    @abstractmethod
    def _perform_calculations(self) -> None: ...

    def calculate(self) -> None:
        if not self._calculated:
            self._perform_calculations()
            self._calculated = True

    def update(self) -> None:
        self._calculated = False
        self.notify_observers()
```

File: `pquantlib/src/pquantlib/patterns/observable_settings.py`

```python
"""Global mutable library settings (singleton).

# C++ parity: ql/settings.hpp (v1.42.1)
"""

from __future__ import annotations

from pquantlib.patterns.singleton import Singleton


class ObservableSettings(Singleton):
    """Global flags affecting library-wide behavior.

    Extended per-feature as L1-A and later phases land them. Phase 0
    seed includes the toggles C++ uses across termstructures and time.
    """

    enforces_business_day_convention: bool = True
    include_today_in_payments: bool = False
    include_reference_date_events: bool = True
```

File: `pquantlib/src/pquantlib/patterns/visitor.py`

```python
"""Visitor pattern as a Protocol.

# C++ parity: ql/patterns/visitor.hpp (v1.42.1)

Python's structural typing collapses C++/Java's separate
Visitor/Visitable/PolymorphicVisitor/PolymorphicVisitable family into
one Protocol pair.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Visitor[T](Protocol):
    def visit(self, target: T) -> None: ...


@runtime_checkable
class Visitable(Protocol):
    def accept(self, visitor: Visitor) -> None: ...  # type: ignore[type-arg]
```

Tests for all patterns: `pquantlib/tests/patterns/test_*.py`.
- Observer: register → notify → all receive `update()`; weak-ref drops observer when last user-ref dies.
- Singleton: `S() is S()`, two distinct subclasses give distinct singletons.
- LazyObject: `calculate()` triggers once; `update()` invalidates; second `calculate()` re-runs.
- ObservableSettings: returns same instance; mutations visible across imports.
- Visitor: simple Visitable accepts a Visitor; runtime_checkable works.

Commit (per-module or batched): `feat(patterns): port Observer/Observable, Singleton, LazyObject, ObservableSettings, Visitor`

### Task 1.5 — `pquantlib.util`

Most of jquantlib's `util/*` collapses in Python:
- `Pair`, `ComparablePair` → drop; use `tuple[A, B]` directly. No module created.
- `Std` → drop; the helpers in it (`equal`, `min_element`, etc.) are stdlib in Python.
- `Visitor`/`Visitable`/`PolymorphicVisitor`/`PolymorphicVisitable` → already in `patterns.visitor`.
- `Observable`/`Observer`/`ObservableValue`/`DefaultObservable`/`WeakReferenceObservable` → all consolidated in `patterns.observer`.
- `LazyObject` → already in `patterns.lazy_object`.
- `ObservableSettings` → already in `patterns.observable_settings`.

Result: no `pquantlib.util` module is created in L1-A. Documented divergence (Phase 1 design decision #9 + L1-A decision #9).

### Task 1.6 — Stage 1 checkpoint

```bash
uv run pytest && uv run pyright && uv run ruff check && uv run ruff format --check
```

Commit: `feat(stage1): close L1-A stage 1 — foundations`

---

## Stage 2 — Time core

### Task 2.1 — Time enums (one commit)

Files (one per enum):
- `pquantlib/src/pquantlib/time/__init__.py`
- `pquantlib/src/pquantlib/time/weekday.py` — Weekday(IntEnum)
- `pquantlib/src/pquantlib/time/month.py` — Month(IntEnum)
- `pquantlib/src/pquantlib/time/time_unit.py` — TimeUnit(IntEnum): Days, Weeks, Months, Years, Hours, Minutes, Seconds, Milliseconds, Microseconds
- `pquantlib/src/pquantlib/time/frequency.py` — Frequency(IntEnum)
- `pquantlib/src/pquantlib/time/business_day_convention.py` — BusinessDayConvention(IntEnum)
- `pquantlib/src/pquantlib/time/date_generation.py` — DateGeneration(IntEnum)

Probe: `migration-harness/cpp/probes/time/enums_probe.cpp` — emit each enum value as integer; test asserts each Python enum value matches.

Commit: `feat(time): port enums (Weekday, Month, TimeUnit, Frequency, BusinessDayConvention, DateGeneration)`

### Task 2.2 — `time.period`

File: `pquantlib/src/pquantlib/time/period.py`

`Period` is `@dataclass(frozen=True, slots=True)` over `(length: int, units: TimeUnit)`. Methods: `years()`, `months()`, `weeks()`, `days()`, `frequency()`, normalization (`normalize`), arithmetic (`+`, `-`, `*`, `==`, `<`).

Probe: `migration-harness/cpp/probes/time/period_probe.cpp` — emit arithmetic results, frequency conversions, comparison outcomes.

Test: `pquantlib/tests/time/test_period.py`. All TIGHT or EXACT.

Commit: `feat(time): port Period`

### Task 2.3 — `time.date` (THE date)

File: `pquantlib/src/pquantlib/time/date.py`

```python
@dataclass(frozen=True, slots=True)
class Date:
    """Serial-day-number date.

    # C++ parity: ql/time/date.hpp (v1.42.1) — minimum serial 367, max 109575.
    """
    serial: int  # days since 1899-12-30 epoch (Excel convention)
```

Methods (all `# C++ parity:`-annotated): `year()`, `month()`, `day_of_month()`, `weekday()`, `day_of_year()`, `serial_number()`, `is_leap(year)` classmethod, `end_of_month()`, `is_end_of_month()`, `next_weekday(day)`, `nth_weekday(n, day, m, y)` classmethod, arithmetic (`+ int`, `+ Period`, `- int`, `- Period`, `- Date`, comparisons), parsing helpers, `min_date()`/`max_date()` classmethods.

Probe: `migration-harness/cpp/probes/time/date_probe.cpp` — exercise every method with ~50 test inputs. EXACT tolerance.

Test: `pquantlib/tests/time/test_date.py`. All EXACT (integer arithmetic).

Commit: `feat(time): port Date`

### Task 2.4 — Date/Period parsers

Files: `pquantlib/src/pquantlib/time/date_parser.py`, `period_parser.py`.

`Period("3M")` and `Date.parse("2024-03-15")` style helpers. Free functions, not classes.

Probe: parse a battery of strings, emit parsed `(length, unit)` / `(y, m, d)`. EXACT.

Commit: `feat(time): port DateParser + PeriodParser`

### Task 2.5 — Calendar abstract + base impls

Files:
- `pquantlib/src/pquantlib/time/calendar.py` — abstract Calendar
- `pquantlib/src/pquantlib/time/calendars/__init__.py`
- `pquantlib/src/pquantlib/time/calendars/null_calendar.py`
- `pquantlib/src/pquantlib/time/calendars/weekends_only.py`
- `pquantlib/src/pquantlib/time/calendars/joint_calendar.py`
- `pquantlib/src/pquantlib/time/calendars/bespoke_calendar.py`

Calendar interface:
```python
class Calendar(ABC):
    @abstractmethod
    def name(self) -> str: ...
    def is_business_day(self, d: Date) -> bool: ...
    def is_holiday(self, d: Date) -> bool: ...
    def is_weekend(self, w: Weekday) -> bool: ...
    def advance(self, d: Date, p: Period, bdc: BusinessDayConvention, end_of_month: bool = False) -> Date: ...
    def adjust(self, d: Date, bdc: BusinessDayConvention) -> Date: ...
    def add_holiday(self, d: Date) -> None: ...
    def remove_holiday(self, d: Date) -> None: ...
    def business_days_between(self, from_d: Date, to_d: Date, include_first: bool = True, include_last: bool = False) -> int: ...
    def holiday_list(self, from_d: Date, to_d: Date, include_weekends: bool = False) -> tuple[Date, ...]: ...
```

Subclasses override `name()` + a protected `_is_business_day(d)` and `_is_weekend(w)`.

Probes: one per calendar emitting (a) business-day check across 50+ dates, (b) `advance(d, p, bdc)` for 20+ scenarios, (c) holiday list for one year.

Commit: `feat(time): port Calendar abstract + Null/WeekendsOnly/Joint/Bespoke`

### Task 2.6 — Schedule + MakeSchedule

File: `pquantlib/src/pquantlib/time/schedule.py`

`Schedule` carries `tuple[Date, ...]` + generation params (effective date, termination date, tenor, calendar, convention, termination convention, date generation rule, end-of-month). `MakeSchedule` is a builder pattern with `with_*` methods.

Probe: 10+ Schedule scenarios (semi-annual, IMM, end-of-month, custom regular periods).

Commit: `feat(time): port Schedule + MakeSchedule`

### Task 2.7 — IMM / ASX / ECB

Files: `pquantlib/src/pquantlib/time/imm.py`, `asx.py`, `ecb.py`.

Free functions for IMM/ASX date arithmetic + ECB date set.

Probe: emit known IMM dates for 2020-2030; ASX equivalents; ECB date set.

Commit: `feat(time): port IMM, ASX, ECB date helpers`

### Task 2.8 — TimeGrid, TimeSeries, Series

Files: `pquantlib/src/pquantlib/time/time_grid.py`, `time_series.py`, `series.py`.

`TimeGrid`: float array of times with mandatory + extra points. `TimeSeries[T]`: bisect-backed dict over Date keys. `Series[T]`: ordered key/value pairs.

Probe: 5+ TimeGrid construction scenarios; TimeSeries lookup behaviors.

Commit: `feat(time): port TimeGrid, TimeSeries, Series`

### Task 2.9 — Stage 2 checkpoint

```bash
uv run pytest && uv run pyright && uv run ruff check && uv run ruff format --check
```

Commit: `feat(stage2): close L1-A stage 2 — time core`

---

## Stage 3 — Day counters

### Task 3.1 — DayCounter abstract

File: `pquantlib/src/pquantlib/daycounters/__init__.py`
File: `pquantlib/src/pquantlib/daycounters/day_counter.py`

```python
class DayCounter(ABC):
    @abstractmethod
    def name(self) -> str: ...
    @abstractmethod
    def day_count(self, d1: Date, d2: Date) -> int: ...
    @abstractmethod
    def year_fraction(self, d1: Date, d2: Date, ref_start: Date | None = None, ref_end: Date | None = None) -> float: ...
```

### Task 3.2 — 11 day counter concretes

One file each under `pquantlib/src/pquantlib/daycounters/`:
- `actual360.py`, `actual364.py`, `actual36525.py`, `actual365fixed.py`, `actual366.py`
- `actualactual.py` (with `Convention` IntEnum: ISMA, Bond, ISDA, Historical, Actual365, AFB, Euro)
- `business252.py`
- `one_day_counter.py`, `simple_day_counter.py`
- `thirty360.py` (with `Convention` IntEnum: USA, BondBasis, European, EurobondBasis, Italian, German, ISMA, ISDA, NASD)
- `thirty365.py`

Probes: one probe per day counter emitting year-fraction for 30+ (d1, d2) pairs covering month-end, leap year, boundary, IMM cases. TIGHT tier default; LOOSE for ActualActual.ISMA corner cases (justified inline).

Commits: per day counter, or batched (e.g., all Actual* in one commit; all Thirty* in one).

### Task 3.3 — Stage 3 checkpoint

Commit: `feat(stage3): close L1-A stage 3 — day counters`

---

## Stage 4 — Calendars

### Task 4.1 — Western + Eastern bases

Files: `pquantlib/src/pquantlib/time/calendars/western.py`, `eastern.py`.

Internal helper bases with default weekend rules (Sat/Sun for Western, Fri/Sat for some Eastern variants — see C++ `ql/time/calendars/saudiarabia.hpp` for the canonical case).

### Task 4.2 — ~43 sovereign/exchange calendars

Alphabetical batches (~10 per commit):

1. Argentina, Australia, Austria, Botswana, Brazil, Canada, Chile, China, CostaRica, CzechRepublic → commit `feat(calendars): port Argentina..CzechRepublic`
2. Denmark, Finland, France, Germany, HongKong, Hungary, Iceland, India, Indonesia, Israel → commit `feat(calendars): port Denmark..Israel`
3. Italy, Japan, Mexico, NewZealand, Norway, Poland, Romania, Russia, SaudiArabia, Singapore → commit `feat(calendars): port Italy..Singapore`
4. Slovakia, SouthAfrica, SouthKorea, Sweden, Switzerland, Taiwan, Target, Thailand, Turkey, Ukraine, UnitedKingdom, UnitedStates → commit `feat(calendars): port Slovakia..UnitedStates`

Per-calendar probe: emit holiday set for 2000–2030. Test asserts set equality. EXACT tier.

### Task 4.3 — Stage 4 checkpoint

Commit: `feat(stage4): close L1-A stage 4 — calendars`

---

## Stage 5 — First math batch

### Task 5.1 — `math.constants` + `math.closeness`

File: `pquantlib/src/pquantlib/math/__init__.py`, `constants.py`, `closeness.py`.

`Constants`: `M_PI`, `M_E`, `QL_EPSILON`, `QL_MAX_REAL`, etc. Module-level `Final[float]`. Cross-validated via probe.

`Closeness`: `close(a, b, n=42)` and `close_enough(a, b, n=42)` per `ql/math/comparison.hpp`.

Commit: `feat(math): port constants + Closeness`

### Task 5.2 — `math.rounding`

File: `pquantlib/src/pquantlib/math/rounding.py`

Class `Rounding` with `Type` IntEnum (None, Up, Down, Closest, Floor, Ceiling) and `precision: int`, `digit: int`. `__call__(value: float) -> float` rounds per type.

Probe: emit rounded values for 50+ inputs across all 5 types.

Commit: `feat(math): port Rounding`

### Task 5.3 — `math.factorial`

File: `pquantlib/src/pquantlib/math/factorial.py`

`Factorial.get(n)` — tabulated for n ≤ 170, Stirling for larger. `Factorial.ln(n)`. Backed by a tuple of precomputed values (mirrors C++ `static const Real` array).

Probe: emit `factorial(0)`..`factorial(170)`, `ln_factorial(0)`..`ln_factorial(1000)`.

Commit: `feat(math): port Factorial`

### Task 5.4 — `math.error_function`

File: `pquantlib/src/pquantlib/math/error_function.py`

`ErrorFunction()(x)` — Chebyshev approximation per Numerical Recipes / C++ `ql/math/errorfunction.hpp`.

Probe: emit `erf(x)` for 100+ x in [-5, 5]. mpmath secondary check.

Commit: `feat(math): port ErrorFunction`

### Task 5.5 — `math.beta`

File: `pquantlib/src/pquantlib/math/beta.py`

`beta(a, b)`, `beta_continued_fraction`, `incomplete_beta(a, b, x)`. Free functions per `ql/math/beta.hpp`.

Probe: emit `beta(a,b)`, `incomplete_beta(a,b,x)` for 50+ (a, b, x). mpmath cross-check.

Commit: `feat(math): port Beta + IncompleteBeta`

### Task 5.6 — `math.bernstein_polynomial`

File: `pquantlib/src/pquantlib/math/bernstein_polynomial.py`

`BernsteinPolynomial.get(i, n, x)` per `ql/math/bernsteinpolynomial.hpp`.

Probe: 50+ (i, n, x) triples.

Commit: `feat(math): port BernsteinPolynomial`

### Task 5.7 — `math.pascal_triangle`

File: `pquantlib/src/pquantlib/math/pascal_triangle.py`

`PascalTriangle.get(order)` returning a row of binomial coefficients per `ql/math/pascaltriangle.hpp`.

Probe: rows 0..30.

Commit: `feat(math): port PascalTriangle`

### Task 5.8 — Stage 5 checkpoint

Commit: `feat(stage5): close L1-A stage 5 — first math batch`

---

## Stage 6 — Review + merge

### Task 6.1 — Spec-compliance review

Dispatch a fresh reviewer subagent with the worktree path. Reviewer verifies:
- Every public class/function has a `# C++ parity:` annotation citing v1.42.1.
- Every numerical test loads from `migration-harness/references/`.
- Tolerance tier is justified inline for any non-default override.
- No commits skip `pytest` / `pyright` / `ruff` (check CI/locally).
- No `Co-authored-by: Claude` trailer in any commit.

Reviewer writes findings to `docs/migration/phase1-l1-A-spec-review.md`. Fix-up loop until clean.

### Task 6.2 — Code-quality review

Dispatch `pr-review-toolkit:code-reviewer` on the full diff `main..phase1-A`. Findings into `docs/migration/phase1-l1-A-code-review.md`. Fix-up loop until clean.

### Task 6.3 — FF-merge to main

```bash
cd /Users/josemoya/Projects/PycharmProjects/pquantlib  # back to main worktree
git fetch
git merge --ff-only origin/phase1-A  # or local phase1-A if not pushed
uv run pytest && uv run pyright && uv run ruff check
git push origin main
```

### Task 6.4 — Tag + clean up worktree

```bash
git tag -a pquantlib-phase1-l1-A-complete -m "..."
git push origin pquantlib-phase1-l1-A-complete

git worktree remove ../pquantlib-phase1-A
git branch -D phase1-A
git push origin --delete phase1-A 2>/dev/null || true
```

### Task 6.5 — Write completion summary

File: `docs/migration/phase1-l1-A-completion.md`. Shape mirrors `phase0-completion.md`: actually-ported count vs estimate, surprises/lessons, carry-overs into B/C/D/E.

Commit: `docs(migration): close L1-A`

### Task 6.6 — Draft B/C/D/E plans

Now that L1-A has surfaced patterns, write:
- `phase1-l1-B-design.md` + `phase1-l1-B-plan.md` (math.copulas + distributions + statistics + currencies)
- `phase1-l1-C-design.md` + `phase1-l1-C-plan.md` (math.integrals + solvers1D + ode)
- `phase1-l1-D-design.md` + `phase1-l1-D-plan.md` (math.randomnumbers + optimization)
- `phase1-l1-E-design.md` + `phase1-l1-E-plan.md` (math.matrixutilities + interpolations + remaining math root)

Commit: `docs(migration): draft L1-B/C/D/E plans`

---

## Definition of done

- [ ] All stages 0–5 land
- [ ] Spec-compliance + code-quality reviews clean
- [ ] FF-merged to `main`
- [ ] Tag `pquantlib-phase1-l1-A-complete` pushed
- [ ] Completion doc written
- [ ] B/C/D/E plans drafted

## Carve-outs

Anything that needs to slip from L1-A:

- Will be enumerated as discovered. Default policy: any class that needs >2 days of pilot iteration to unblock gets pushed to a follow-up cluster with documented rationale.
- Likely candidates (heads-up, not commitments): SaudiArabia (Hijri calendar complexity); ActualActual.ISMA corner cases; `MultidimIntegral` if it's only used by L2+.

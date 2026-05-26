# Phase 1 L1-A Code Review

Scope: 47 commits / ~9000 lines from `main..phase1-A` covering exceptions/qassert, testing helpers, design patterns (observer / singleton / lazy_object / observable_settings / visitor), time module (enums, Period, Date, parsers, Calendar + 4 trivial + 41 sovereign calendars), daycounters (abstract + 11 concretes), Schedule + MakeSchedule, IMM/ASX/ECB, TimeGrid + TimeSeries, math primitives (constants, closeness, rounding, factorial, error_function, beta, bernstein_polynomial, pascal_triangle), and matching tests.

Methodology: read all newly-added source modules, cross-checked key algorithms against `migration-harness/cpp/quantlib/ql/...` sources, and exercised potential edge cases interactively (`uv run python -c '...'`). Skipped findings that fall into the project's stated "not bugs" exclusions (PLR0911/PLR0912 dispatch, switch-on-enum patterns, LOOSE-tier divergences on lgamma/erf, etc.).

Tests pass (411/0), pyright 0 errors, ruff clean. Findings below are quality concerns above and beyond the green baseline.

## Counts

- Blocker: 0
- Major: 3
- Minor: 6
- Nit: 4

---

## Correctness

### [MAJOR] pquantlib/src/pquantlib/time/time_grid.py:31 — `_CLOSE_ENOUGH_REL` is ~4500x looser than the C++ `close_enough(a, b, 42)` it claims to mirror

The module-level constants:

```python
_CLOSE_ENOUGH_ABS: Final[float] = 1e-14
_CLOSE_ENOUGH_REL: Final[float] = 42 * 1e-12     # ← 4.2e-11
```

are documented as mirroring C++ `close_enough(a, b, 42)`. That helper uses `tolerance = 42 * QL_EPSILON` (≈ 9.33e-15). The Python value `42 * 1e-12 = 4.2e-11` is ~4500× looser. Confirmed:

```
C++ 42 * QL_EPSILON = 9.325873406851315e-15
Python _CLOSE_ENOUGH_REL = 4.2e-11
ratio: 4.50e+03
```

`close(1.0, 1.0+1e-13)` is False in C++ but True under the Python tolerance — the test even verifies it with TIGHT tolerance against the C++ probe and passes because the probe values are far apart, but real callers of `TimeGrid.index()` and `_dedupe_close()` will see *too many* hits (false-positive merges + false-positive index match) for values 1e-15..4e-11 apart.

Fix: import `QL_EPSILON` from `pquantlib.math.constants` and use `_CLOSE_ENOUGH_REL: Final[float] = 42 * QL_EPSILON`. (The constant `1e-12` literal was almost certainly a typo for `QL_EPSILON`.) Add a regression test that constructs two grids differing by 1e-13 and asserts they are NOT close.

### [MAJOR] pquantlib/src/pquantlib/math/factorial.py:57 — `Factorial.get(n)` silently returns wrong values for negative `n` via Python negative indexing

```python
@staticmethod
def get(n: int) -> float:
    if n <= _TABULATED:
        return _FIRST_FACTORIALS[n]
    return math.exp(math.lgamma(n + 1))
```

`_TABULATED == 27` so `n == -1` satisfies `n <= 27`, then `_FIRST_FACTORIALS[-1]` returns the value for n=27 (=`1.0888...e28`) instead of failing. Verified at runtime:

```
Factorial.get(-1) → 1.0888869450418352e+28   # silently wrong
```

C++ uses `Natural` (unsigned) so this is impossible there; Python `int` accepts negative. `Factorial.ln(-1)` has the same defect (`math.log(_FIRST_FACTORIALS[-1])`). `PascalTriangle.get(order)` has the same problem (`cls._coefficients[-1]` returns the highest-cached row).

`BernsteinPolynomial.get(i, n, x)` calls `Factorial.get(n - i)` — if a caller passes `i > n` (which is a degenerate but possible misuse), the result is silently nonsensical.

Fix: add `qassert.require(n >= 0, "negative factorial argument")` at the top of `Factorial.get` and `Factorial.ln`, and `qassert.require(order >= 0, "negative pascal triangle order")` in `PascalTriangle.get`.

### [MAJOR] pquantlib/src/pquantlib/daycounters/actual_actual.py:267 — `_coupons_per_year` divides by zero if `months == 0`

```python
def _coupons_per_year(ref_start: Date, ref_end: Date) -> int:
    months = round(12 * (ref_end - ref_start) / 365.0)
    return round(12.0 / months)        # ← ZeroDivisionError if months == 0
```

This is reachable only via `_yf_with_ref_dates` which guards with `if reference_day_count < 16: coupons_per_year = 1` — so in the current call site, `months == 0` is impossible. But the helper itself is unsafe and the C++ version `findCouponsPerYear` (`lround(12.0/months)`) is also unsafe — both rely on caller-side filtering. Worth flagging because the helper is exposed at module scope and a future caller could trip the bug.

Fix: either inline the helper into `_yf_with_ref_dates` (single call site), or guard the helper with `qassert.require(months > 0, ...)`. Add a regression test that fakes a ref period shorter than 15 days and confirms the guard fires.

## Silent failures

### [MINOR] pquantlib/src/pquantlib/time/date_parser.py:39 — `parse_iso` lets a stdlib `ValueError` from `Month(int(s[5:7]))` leak

For inputs like `"2024-00-15"` or `"2024-13-15"`, the integer parses fine, then `Month(0)` / `Month(13)` raises Python `ValueError`, not `LibraryException`. C++ throws `QuantLib::Error` for the same input (via `Date(d, m, y)` validation). Confirmed:

```
parse_iso('2024-13-32') → ValueError: 13 is not a valid Month
```

Fix: wrap the `Month(month)` call (and the subsequent `Date.from_ymd`) in a try/except, or pre-validate `1 <= month <= 12 and 1 <= day <= 31` before constructing `Month(month)`.

### [MINOR] pquantlib/src/pquantlib/time/calendars/israel.py:443-444 (and similar) — `Israel` and `SHIR` `_is_business_day` raise on dates near `Date.min_date()`

The Israel TASE / SHIR predicates evaluate `_is_passover_1st(d - 5)`, `_is_yom_kippur(d + 1)`, `_is_simchat_torah(d - 7)`, etc. — each of these subtracts a small offset from `d`. At `d = Date.from_ymd(1, January, 1901)` (serial 367, the minimum), `d - 5 = Date(362)` is below `_MIN_SERIAL=367` and `Date.__post_init__` raises:

```
Israel(TASE).is_business_day(Date.from_ymd(1, January, 1901))
  → LibraryException: Date's serial number (362) outside allowed range [367-109574]
```

C++ doesn't hit this because its `holidaySet.count(d - 5)` doesn't construct a Date — it computes the serial then compares. In Python the arithmetic constructs a Date which validates. The Israel holiday tables only start in 2000, so the result for any date < 2000 is "not a holiday for these reasons" — but the user gets an exception, not False. Similar exposure on any other calendar that adds/subtracts a small number of days from a near-minimum date.

Fix: either short-circuit (`if y < 2000: return ...`) inside the Israel predicate, or relax Date arithmetic to clamp the result rather than raise (more invasive). Pragmatic option: add a guard `if d.year() < 2000: skip these checks` at the top of each Israel impl.

### [NIT] pquantlib/src/pquantlib/time/schedule.py:478-479 — `_final_safety_dedup` assumes `is_regular` is non-empty before `pop()`

The trailing-dedup block always calls `is_regular.pop()` (line 479) without checking `is_regular` is non-empty. The leading-dedup block (line 482-485) likewise indexes `is_regular[1]` and `is_regular.pop(0)`. In practice `from_rule` populates `is_regular` in lock-step with `dates`, so this isn't reachable. But the public `Schedule.__init__` accepts an empty `is_regular` sequence; the dedup helper is static so the assumption isn't structurally enforced. Not a hot fix; just add `if is_regular: is_regular.pop()` (and similar at the leading edge) for robustness, since the cost is one branch.

## Type strictness

### [MINOR] pquantlib/src/pquantlib/patterns/singleton.py:28-33 — `dict[Any, Any]` masks two narrower types

```python
_instances: ClassVar[dict[Any, Any]] = {}
def __call__(cls, *args: object, **kwargs: object) -> Any: ...
```

The module docstring explains the `dict[Any, Any]` keying (pyright sees `cls` as `Self@_SingletonMeta`, not `type`). That's defensible — but the **return** type of `__call__` could be `cls` (i.e. tighten to `Self`, since PEP 673 is supported in 3.14). At minimum, callers of `Singleton()` get back `Any` rather than the concrete subclass, which propagates `Any` into downstream call sites that subscript or call methods on the singleton. Concrete tests work because pyright trusts the subclass annotations.

Fix: explore `def __call__(cls, *args: object, **kwargs: object) -> Self:` (Python 3.11+) — would require `from typing import Self` and a check that pyright accepts it on a metaclass `__call__`.

### [NIT] pquantlib/src/pquantlib/testing/reference_reader.py:43 — `# type: ignore[no-any-return]` could be tightened with a `cast`

```python
return json.load(f)  # type: ignore[no-any-return]
```

`json.load` returns `Any`. A `cast(dict[str, Any], json.load(f))` would make the intent explicit without suppressing.

## Idiomatic / non-obvious

### [MINOR] pquantlib/src/pquantlib/daycounters/actual_actual.py:262, 291 — `_year_fraction_guess` is dead code (referenced only via `_ = _year_fraction_guess` to silence the unused-warning)

```python
def _year_fraction_guess(start: Date, end: Date) -> float:
    """Mirrors C++ ``yearFractionGuess`` ..."""
    return (end - start) / 365.0
...
_ = _year_fraction_guess
```

The comment claims it's "useful for diagnostics" but it's not in `__all__`, never called, never tested. Either expose it (add to `__all__` and write a test), or delete the function and the `_ =` line.

### [NIT] pquantlib/src/pquantlib/daycounters/business_252.py:9-17 — stale docstring claiming Brazil calendar is not ported

```python
# Divergence from C++:
# - The C++ default constructor uses ``Brazil()`` calendar. Brazil is not
#   yet ported (Stage 4); the Python port requires an explicit calendar.
```

Brazil is now ported (`pquantlib/src/pquantlib/time/calendars/brazil.py`). The docstring should either drop the divergence note (and align with C++ by defaulting the calendar to `Brazil()`) or update the text to "we deliberately require an explicit calendar — see Phase 1 decision #N" with a forward-reference.

### [NIT] pquantlib/src/pquantlib/time/calendars/* — inconsistent `__init__` validation of the Market enum

Some Market-enum calendars validate the input in `__init__` (Argentina, Chile, China, Hong Kong, India, Indonesia, South Korea, Thailand, United States, United Kingdom), others don't (Australia, Brazil, Canada, Czech Republic, Iceland, New Zealand, Ukraine). The non-validating ones still raise via `qassert.fail` in `name()` / `_is_business_day()`, so misuse is caught — but later (at first use, not construction). Pick one pattern; the validation-in-`__init__` form gives fail-fast behavior, which is preferable.

## Tests / coverage gaps

### [MINOR] pquantlib/tests/math/test_first_batch.py — no tests for negative-argument behavior of `Factorial`, `PascalTriangle`, or `BernsteinPolynomial`

Tied to the Major finding above: the Python implementations silently return wrong values for negative `n`/`order`/`i`. The tests should pin the (corrected) behavior:

```python
with pytest.raises(LibraryException):
    Factorial.get(-1)
with pytest.raises(LibraryException):
    PascalTriangle.get(-1)
```

### [MINOR] pquantlib/tests/time/test_date.py — no `next_weekday` edge case for "target is current weekday"

The implementation returns `d` itself when `target == d.weekday()` (verified above). The test `test_next_weekday_matches_cpp` exercises the cross-validation table but no Python-side assertion confirms the "returns d unchanged when already on target weekday" invariant. Add one explicit assertion since it's the most likely off-by-one site.

### [MINOR] pquantlib/tests/time/test_time_grid_and_series.py — no test exercises the `_CLOSE_ENOUGH_REL` boundary

Tied to the Major finding on time_grid.py. After the fix, add:

```python
def test_index_rejects_value_slightly_off_grid():
    tg = TimeGrid.regular(1.0, 4)
    # 1e-13 off the 0.25 node — C++ would reject (diff > 42*QL_EPSILON*0.25)
    with pytest.raises(LibraryException, match="inadequate"):
        tg.index(0.25 + 1e-13)
```

## Dead code / unused imports / leftover scaffolding

### [NIT] pquantlib/src/pquantlib/daycounters/actual_actual.py — `_year_fraction_guess` dead code (covered above under Idiomatic).

(No other dead-code findings — the 41 calendar modules and the math/time core all have every defined symbol referenced or under test.)

---

## Summary of top-three highest-priority items

1. **time_grid.py `_CLOSE_ENOUGH_REL = 42 * 1e-12`** — 4500× looser than the C++ value it claims to mirror. One-line fix (`42 * QL_EPSILON`). Will produce subtly wrong `TimeGrid.index()` and `_dedupe_close` decisions for values within ~1e-13 to ~4e-11 of a node. **MAJOR**.

2. **`Factorial.get(-1)` returns a silently wrong tabulated value** via Python negative indexing. Same defect in `Factorial.ln` and `PascalTriangle.get`. Defensive `qassert.require(n >= 0, ...)` on each. **MAJOR**.

3. **`_coupons_per_year` ZeroDivisionError** if `months == 0`. Currently guarded by the only caller, but the helper is exposed at module scope. Inline-or-guard. **MAJOR**.

The remaining findings are minor robustness / consistency improvements. Phase 1 L1-A is in solid shape — 411 tests green, cross-validated against C++ probes, idiomatic Python, well-documented divergences, consistent patterns across 41 sovereign calendars and 11 day-counters.

"""Market-model grid utilities — time-grid merge / subset / tau helpers.

# C++ parity: ql/models/marketmodels/utilities.{hpp,cpp} (v1.42.1).

Free functions shared across the marketmodels cluster:

- ``merge_times(times)`` — merge a list of strictly-increasing time
  vectors into one sorted-unique grid, returning the merged grid plus a
  per-input boolean presence matrix.
- ``is_in_subset(s, subset)`` — boolean vector flagging which elements of
  ``s`` appear in ``subset`` (both must be strictly increasing).
- ``check_increasing_times(times)`` — validate strictly-increasing,
  positive-first-time.
- ``check_increasing_times_and_calculate_taus(times)`` — validate and
  return the consecutive-difference ``taus`` vector.

Divergences from C++:

- C++ ``mergeTimes`` returns ``isPresent`` as ``std::vector<std::valarray<bool>>``
  and *appends* to caller-provided out-parameters. PQuantLib returns a
  ``tuple[list[float], list[list[bool]]]`` (fresh containers) — out-params
  are unidiomatic in Python.
- C++ ``isInSubset`` returns a ``std::valarray<bool>``; we return
  ``list[bool]``.
"""

from __future__ import annotations

import bisect

from pquantlib import qassert


def check_increasing_times(times: list[float]) -> None:
    """Require ``times`` strictly increasing with a positive first time.

    # C++ parity: utilities.cpp checkIncreasingTimes.
    """
    n = len(times)
    qassert.require(n > 0, "at least one time is required")
    qassert.require(times[0] > 0.0, f"first time ({times[0]}) must be greater than zero")
    for i in range(n - 1):
        qassert.require(
            times[i + 1] - times[i] > 0,
            f"non increasing rate times: times[{i}]={times[i]}, times[{i + 1}]={times[i + 1]}",
        )


def check_increasing_times_and_calculate_taus(times: list[float]) -> list[float]:
    """Validate ``times`` (>= 2, increasing, positive first) and return taus.

    # C++ parity: utilities.cpp checkIncreasingTimesAndCalculateTaus.

    Unlike C++ (which fills a caller-supplied ``taus`` out-param), this
    returns a freshly-allocated ``taus`` list of length ``len(times) - 1``.
    """
    n = len(times)
    qassert.require(n > 1, f"at least two times are required, {n} provided")
    qassert.require(times[0] > 0.0, f"first time ({times[0]}) must be greater than zero")
    taus = [0.0] * (n - 1)
    for i in range(n - 1):
        taus[i] = times[i + 1] - times[i]
        qassert.require(
            taus[i] > 0,
            f"non increasing rate times: times[{i}]={times[i]}, times[{i + 1}]={times[i + 1]}",
        )
    return taus


def merge_times(times: list[list[float]]) -> tuple[list[float], list[list[bool]]]:
    """Merge increasing time vectors into one sorted-unique grid + presence flags.

    # C++ parity: utilities.cpp mergeTimes.

    Returns ``(merged_times, is_present)`` where ``merged_times`` is the
    sorted unique union of all input times and ``is_present[i][j]`` is
    ``True`` iff ``merged_times[j]`` is present in ``times[i]``.
    """
    all_times: list[float] = []
    for time in times:
        all_times.extend(time)
    # sort + unique-compact
    all_times.sort()
    merged: list[float] = []
    for t in all_times:
        if not merged or merged[-1] != t:
            merged.append(t)

    is_present: list[list[bool]] = []
    for i in range(len(times)):
        row = [False] * len(merged)
        sorted_i = sorted(times[i])
        for j, t in enumerate(merged):
            # binary_search equivalent
            k = bisect.bisect_left(sorted_i, t)
            row[j] = k < len(sorted_i) and sorted_i[k] == t
        is_present.append(row)
    return merged, is_present


def is_in_subset(s: list[float], subset: list[float]) -> list[bool]:
    """Flag which elements of ``s`` appear in ``subset``.

    # C++ parity: utilities.cpp isInSubset.

    Both ``s`` and ``subset`` must be strictly increasing. Returns a list
    of booleans the same length as ``s``.
    """
    result = [False] * len(s)
    dim_subset = len(subset)
    if dim_subset == 0:
        return result
    dim_set = len(s)
    qassert.require(
        dim_set >= dim_subset,
        "set is required to be larger or equal than subset",
    )
    for i in range(dim_set):
        j = 0
        set_element = s[i]
        while True:
            subset_element = subset[j]
            result[i] = False
            # if smaller no hope, leave false and go to next i
            if set_element < subset_element:
                break
            # if match, set result[i] to true and go to next i
            if set_element == subset_element:
                result[i] = True
                break
            # if larger, leave false if at the end or go to next j
            if j == dim_subset - 1:
                break
            j += 1
    return result

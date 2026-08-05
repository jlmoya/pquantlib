"""Cross-validate FastFourierTransform against the v1.43 C++ probe.

Reference: ``migration-harness/references/v143/math/tail.json`` — ``fft``.
Three inputs: a complex length-4 (exactly a power of two), a real length-5
(padded into order 3), and a single point (order 0, the degenerate copy).
Forward and unnormalised-inverse outputs are both pinned in full.
"""

from __future__ import annotations

from typing import Any

from pquantlib.math.fast_fourier_transform import FastFourierTransform
from pquantlib.testing import tolerance


def test_min_order_matches_cpp_exactly(v143_tail: dict[str, Any]) -> None:
    for n, expected in v143_tail["fft"]["min_order"]:
        assert FastFourierTransform.min_order(n) == expected


def test_forward_and_inverse_match_cpp_tight(v143_tail: dict[str, Any]) -> None:
    for case in v143_tail["fft"]["cases"]:
        values = [complex(re, im) for re, im in case["input"]]
        fft = FastFourierTransform(case["order"])
        assert fft.output_size() == case["output_size"]
        for label, expected in (("forward", case["forward"]), ("inverse", case["inverse"])):
            got = fft.transform(values) if label == "forward" else fft.inverse_transform(values)
            assert len(got) == len(expected)
            for i, (actual, (re, im)) in enumerate(zip(got, expected, strict=True)):
                ctx = f"order={case['order']}, {label}[{i}]"
                tolerance.tight(actual.real, re, reason=ctx)
                tolerance.tight(actual.imag, im, reason=ctx)


def test_inverse_is_unnormalised(v143_tail: dict[str, Any]) -> None:
    """C++ conjugates the twiddle but does not divide by N.

    So forward-then-inverse scales by N. Pinned as a property because the
    missing 1/N is exactly the kind of thing a port "helpfully" adds.
    """
    case = next(c for c in v143_tail["fft"]["cases"] if c["order"] > 0)
    values = [complex(re, im) for re, im in case["input"]]
    fft = FastFourierTransform(case["order"])
    n = fft.output_size()
    round_trip = fft.inverse_transform(fft.transform(values))
    padded = [*values, *([0j] * (n - len(values)))]
    for actual, original in zip(round_trip, padded, strict=True):
        tolerance.tight(actual.real, n * original.real)
        tolerance.tight(actual.imag, n * original.imag)


def test_order_zero_is_a_copy(v143_tail: dict[str, Any]) -> None:
    case = next(c for c in v143_tail["fft"]["cases"] if c["order"] == 0)
    fft = FastFourierTransform(0)
    assert fft.output_size() == 1
    values = [complex(re, im) for re, im in case["input"]]
    tolerance.exact(fft.transform(values)[0].real, values[0].real)
    tolerance.exact(fft.transform(values)[0].imag, values[0].imag)

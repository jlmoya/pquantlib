"""SobolChartSample — generate (and optionally plot) a Sobol low-discrepancy set.

Port of ``org.jquantlib.samples.SobolChartSample``. The Java original used
JFreeChart in a Swing window; here :func:`run` always computes the Sobol
sequence and prints summary statistics, and the scatter plot is *optional*:
matplotlib is imported lazily inside an environment-variable guard so the
module imports and runs headlessly (and without matplotlib installed) under
CI / the smoke suite.

Set ``PQL_SAMPLES_PLOT`` (any non-empty value) to render the scatter plot to a
temp PNG; if matplotlib is missing the plot is skipped with a notice.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from pquantlib.math.randomnumbers.sobol_rsg import SobolRsg

_DIMENSION = 5
_SEED = 42
_SAMPLES = 500


@dataclass(frozen=True, slots=True)
class SobolResult:
    """The Sobol point cloud plus a couple of pinnable summary numbers."""

    dimension: int
    seed: int
    samples: int
    xs: list[float]
    ys: list[float]
    mean_x: float
    mean_y: float


def compute(samples: int = _SAMPLES) -> SobolResult:
    sobol = SobolRsg(_DIMENSION, _SEED)
    xs: list[float] = []
    ys: list[float] = []
    for _ in range(samples):
        for i in range(_DIMENSION):
            seq1 = sobol.next_sequence()
            seq2 = sobol.next_sequence()
            xs.append(float(seq1[i]))
            ys.append(float(seq2[i]))
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    return SobolResult(
        dimension=_DIMENSION,
        seed=_SEED,
        samples=samples,
        xs=xs,
        ys=ys,
        mean_x=mean_x,
        mean_y=mean_y,
    )


def _maybe_plot(result: SobolResult) -> None:
    """Render a scatter plot if ``PQL_SAMPLES_PLOT`` is set and matplotlib exists."""
    if not os.environ.get("PQL_SAMPLES_PLOT"):
        return
    # Lazy imports by design: matplotlib is NOT a workspace dependency, so it is
    # only imported inside this guard (which also requires PQL_SAMPLES_PLOT to be
    # set). This keeps the module importable + runnable headlessly without it.
    # pyright cannot resolve the optional matplotlib types — suppressed below.
    try:
        import matplotlib  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

        matplotlib.use("Agg")  # headless backend; no display required.  # pyright: ignore[reportUnknownMemberType]
        import matplotlib.pyplot as plt  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
    except ImportError:
        print("(PQL_SAMPLES_PLOT set but matplotlib is not installed — skipping plot)")
        return

    import tempfile  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(8, 5.7))  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    ax.scatter(result.xs, result.ys, s=2, c="navy")  # pyright: ignore[reportUnknownMemberType]
    ax.set_title("Sobol Chart Sample")  # pyright: ignore[reportUnknownMemberType]
    ax.set_xlabel("x")  # pyright: ignore[reportUnknownMemberType]
    ax.set_ylabel("y")  # pyright: ignore[reportUnknownMemberType]
    out = os.path.join(tempfile.gettempdir(), "sobol_chart_sample.png")
    fig.savefig(out)  # pyright: ignore[reportUnknownMemberType]
    plt.close(fig)  # pyright: ignore[reportUnknownMemberType]
    print(f"Sobol scatter plot written to {out}")


def run() -> None:
    print("::::: SobolChartSample :::::")
    r = compute()
    print(f"Sobol dimension: {r.dimension}, seed: {r.seed}, samples: {r.samples}")
    print(f"Generated {len(r.xs)} (x, y) points.")
    print(f"First point:  ({r.xs[0]:.6f}, {r.ys[0]:.6f})")
    print(f"Mean x: {r.mean_x:.6f}  Mean y: {r.mean_y:.6f}  (both ~0.5 for a uniform fill)")
    _maybe_plot(r)


if __name__ == "__main__":
    run()

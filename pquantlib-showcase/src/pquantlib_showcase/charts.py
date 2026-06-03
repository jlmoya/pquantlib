"""Reusable Plotly figures. Pure functions: data in, ``go.Figure`` out."""

from __future__ import annotations

from collections.abc import Sequence

import plotly.graph_objects as go

# A restrained, finance-desk palette.
ACCENT = "#2563eb"
ACCENT2 = "#db2777"
ACCENT3 = "#059669"
MUTED = "#94a3b8"
GRID = "#e2e8f0"
INK = "#0f172a"


def _base(fig: go.Figure, *, xtitle: str, ytitle: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=60, r=30, t=30, b=50),
        font=dict(family="Inter, system-ui, sans-serif", size=13, color=INK),
        xaxis=dict(title=xtitle, gridcolor=GRID, zeroline=False),
        yaxis=dict(title=ytitle, gridcolor=GRID, zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0),
        hovermode="x unified",
    )
    return fig


def line(
    x: Sequence[float],
    y: Sequence[float],
    *,
    xtitle: str,
    ytitle: str,
    name: str = "",
    color: str = ACCENT,
    fill: bool = False,
    percent_axis: bool = False,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(x),
            y=list(y),
            mode="lines",
            name=name,
            line=dict(color=color, width=2.5),
            fill="tozeroy" if fill else None,
            fillcolor="rgba(37,99,235,0.08)",
        )
    )
    _base(fig, xtitle=xtitle, ytitle=ytitle)
    if percent_axis:
        fig.update_yaxes(tickformat=".2%")
    return fig


def multi_line(
    x: Sequence[float],
    series: list[tuple[str, Sequence[float], str]],
    *,
    xtitle: str,
    ytitle: str,
    percent_axis: bool = False,
    percent_x: bool = False,
) -> go.Figure:
    """``series`` is a list of (name, y-values, color)."""
    fig = go.Figure()
    for name, y, color in series:
        fig.add_trace(
            go.Scatter(x=list(x), y=list(y), mode="lines", name=name, line=dict(color=color, width=2.5))
        )
    _base(fig, xtitle=xtitle, ytitle=ytitle)
    if percent_axis:
        fig.update_yaxes(tickformat=".2%")
    if percent_x:
        fig.update_xaxes(tickformat=".1%")
    return fig


def convergence(
    steps: Sequence[int], prices: Sequence[float], target: float, *, ytitle: str = "Price"
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(steps),
            y=list(prices),
            mode="lines+markers",
            name="Binomial tree",
            line=dict(color=ACCENT, width=2),
            marker=dict(size=4),
        )
    )
    fig.add_hline(
        y=target,
        line=dict(color=ACCENT2, width=2, dash="dash"),
        annotation_text=f"Closed-form  {target:.4f}",
        annotation_position="top right",
    )
    return _base(fig, xtitle="Tree steps (N)", ytitle=ytitle)


def smile(
    strikes: Sequence[float],
    market: Sequence[float] | None,
    model: Sequence[float],
    *,
    model_name: str = "Model",
    spot: float | None = None,
) -> go.Figure:
    fig = go.Figure()
    if market is not None:
        fig.add_trace(
            go.Scatter(
                x=list(strikes),
                y=list(market),
                mode="markers",
                name="Market",
                marker=dict(color=ACCENT2, size=10, symbol="circle"),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=list(strikes),
            y=list(model),
            mode="lines+markers",
            name=model_name,
            line=dict(color=ACCENT, width=2.5),
            marker=dict(size=5),
        )
    )
    if spot is not None:
        fig.add_vline(
            x=spot,
            line=dict(color=MUTED, width=1, dash="dot"),
            annotation_text="spot",
            annotation_position="bottom",
        )
    fig.update_yaxes(tickformat=".1%")
    return _base(fig, xtitle="Strike", ytitle="Implied volatility")


def payoff_and_value(
    spots: Sequence[float],
    payoff: Sequence[float],
    value: Sequence[float],
    *,
    spot: float,
    strike: float,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(spots),
            y=list(payoff),
            mode="lines",
            name="Payoff at expiry",
            line=dict(color=MUTED, width=2, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(spots), y=list(value), mode="lines", name="Value today", line=dict(color=ACCENT, width=2.5)
        )
    )
    fig.add_vline(
        x=spot,
        line=dict(color=ACCENT3, width=1.5, dash="dot"),
        annotation_text=f"spot {spot:g}",
        annotation_position="top left",
    )
    fig.add_vline(
        x=strike,
        line=dict(color=MUTED, width=1, dash="dot"),
        annotation_text=f"K {strike:g}",
        annotation_position="top right",
    )
    return _base(fig, xtitle="Underlying spot", ytitle="Option value")


def bars(
    labels: Sequence[str], values: Sequence[float], *, ytitle: str, colors: Sequence[str] | None = None
) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=list(labels),
            y=list(values),
            marker_color=list(colors) if colors else ACCENT,
            text=[f"{v:,.4f}" for v in values],
            textposition="outside",
        )
    )
    return _base(fig, xtitle="", ytitle=ytitle, height=380)

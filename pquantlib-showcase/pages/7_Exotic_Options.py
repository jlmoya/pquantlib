"""Exotic options: single barrier, double barrier, and discrete Asian — vs vanilla."""

from __future__ import annotations

import streamlit as st

from pquantlib_showcase import charts, ui
from pquantlib_showcase.quant import exotics

ui.setup_page("Exotic Options", icon="🚧")
ui.about_sidebar()
ui.hero(
    "Exotic options",
    "Path-dependent payoffs priced with closed-form analytics.",
    pills=["AnalyticBarrierEngine", "AnalyticDoubleBarrierEngine", "Geometric Asian"],
)

family = st.radio("Family", ["Single barrier", "Double barrier", "Asian (geometric)"], horizontal=True)

with st.sidebar:
    kind = st.radio("Option type", ["Call", "Put"], horizontal=True)
    spot = st.slider("Spot", 50.0, 150.0, 100.0, 1.0)
    strike = st.slider("Strike", 50.0, 150.0, 100.0, 1.0)
    r = st.slider("Risk-free rate", 0.0, 0.10, 0.05, 0.0025, format="%.4f")
    vol = st.slider("Volatility", 0.05, 0.80, 0.25, 0.01)
    t = st.slider("Maturity (years)", 0.1, 3.0, 1.0, 0.1)

if family == "Single barrier":
    bkind = st.sidebar.selectbox("Barrier", list(exotics.BARRIER_TYPES))
    barrier = st.sidebar.slider("Barrier level", 50.0, 150.0, 90.0, 1.0)
    rebate = st.sidebar.number_input("Rebate", 0.0, 20.0, 0.0)
    res = exotics.price_barrier(kind, bkind, spot, strike, barrier, rebate, r, 0.0, vol, t)
    if res.triggered:
        st.warning(
            f"With spot at {spot:g}, a **{bkind}** barrier at {barrier:g} is already breached at inception — "
            "the analytic engine only prices live barriers. Move the barrier to the other side of spot."
        )
    else:
        ui.metric_cards(
            [
                (f"{bkind} {kind.lower()}", f"{res.price:.4f}", None),
                ("Vanilla reference", f"{res.vanilla_price:.4f}", None),
                ("Barrier vs vanilla", f"{res.price - res.vanilla_price:+.4f}", None),
            ]
        )
        ui.note(
            "A knock-out is worth less than the vanilla — it can be extinguished early. "
            "A knock-in is its mirror image."
        )
    st.subheader("Price vs. barrier level")
    levels, prices, vanilla = exotics.barrier_vs_level(kind, bkind, spot, strike, rebate, r, 0.0, vol, t)
    fig = charts.line(levels, prices, xtitle="Barrier level", ytitle="Option price", name=bkind)
    fig.add_hline(
        y=vanilla,
        line=dict(color=charts.ACCENT2, width=2, dash="dash"),
        annotation_text=f"vanilla {vanilla:.3f}",
        annotation_position="top right",
    )
    fig.add_vline(x=spot, line=dict(color=charts.MUTED, width=1, dash="dot"), annotation_text="spot")
    st.plotly_chart(fig, use_container_width=True)

elif family == "Double barrier":
    bkind = st.sidebar.selectbox("Barrier", list(exotics.DOUBLE_BARRIER_TYPES))
    low = st.sidebar.slider("Lower barrier", 40.0, 100.0, 80.0, 1.0)
    high = st.sidebar.slider("Upper barrier", 100.0, 200.0, 120.0, 1.0)
    rebate = st.sidebar.number_input("Rebate", 0.0, 20.0, 0.0)
    res = exotics.price_double_barrier(kind, bkind, spot, strike, low, high, rebate, r, 0.0, vol, t)
    if res.triggered or not (low < spot < high):
        st.warning(f"Spot ({spot:g}) must sit strictly inside the corridor ({low:g}, {high:g}) to stay live.")
    else:
        ui.metric_cards(
            [
                (f"Double {bkind.lower()} {kind.lower()}", f"{res.price:.4f}", None),
                ("Vanilla reference", f"{res.vanilla_price:.4f}", None),
                ("Corridor vs vanilla", f"{res.price - res.vanilla_price:+.4f}", None),
            ]
        )
        ui.note(
            f"The option survives only while spot stays inside ({low:g}, {high:g}). A tighter corridor or "
            "volatility makes a knock-out cheaper. Priced with the Ikeda-Kunitomo analytic series."
        )

else:
    n_fix = st.sidebar.slider("Averaging dates", 4, 52, 12)
    res = exotics.price_asian(kind, spot, strike, r, 0.0, vol, t, n_fix)
    ui.metric_cards(
        [
            (f"Geometric Asian {kind.lower()}", f"{res.price:.4f}", None),
            ("Vanilla reference", f"{res.vanilla_price:.4f}", None),
            ("Averaging discount", f"{res.price - res.vanilla_price:+.4f}", None),
        ]
    )
    ui.note(
        "Averaging the underlying over many fixings dampens its effective volatility, so an Asian option "
        "is cheaper than the equivalent vanilla. Closed-form for the geometric average."
    )

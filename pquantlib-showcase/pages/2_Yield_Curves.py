"""Yield curves: flat and bootstrapped-from-deposits, with the full term structure."""

from __future__ import annotations

import streamlit as st

from pquantlib_showcase import charts, ui
from pquantlib_showcase.quant import curves

ui.setup_page("Yield Curves", icon="📉")
ui.about_sidebar()
ui.hero(
    "Yield curves",
    "Discount factors, zero rates and forwards — flat or bootstrapped from market deposits.",
    pills=["FlatForward", "PiecewiseYieldCurve", "DepositRateHelper"],
)

mode = st.radio("Curve construction", ["Flat curve", "Bootstrapped from deposits"], horizontal=True)

if mode == "Flat curve":
    rate = st.sidebar.slider("Flat rate", 0.0, 0.10, 0.04, 0.0025, format="%.4f")
    horizon = st.sidebar.slider("Horizon (years)", 1.0, 30.0, 10.0, 1.0)
    prof = curves.flat_curve_profile(rate, horizon)
    ui.metric_cards(
        [
            ("Discount @ 1Y", f"{prof.discount_factors[min(5, len(prof.discount_factors) - 1)]:.5f}", None),
            ("Zero rate", f"{rate:.3%}", None),
            ("Forward rate", f"{rate:.3%}", None),
        ]
    )
    st.plotly_chart(
        charts.line(
            prof.tenors,
            prof.discount_factors,
            xtitle="Maturity (years)",
            ytitle="Discount factor",
            name="Discount",
            fill=True,
        ),
        use_container_width=True,
    )
    st.plotly_chart(
        charts.multi_line(
            prof.tenors,
            [
                ("Zero rate", prof.zero_rates, charts.ACCENT),
                ("Instantaneous forward", prof.forward_rates, charts.ACCENT2),
            ],
            xtitle="Maturity (years)",
            ytitle="Rate",
            percent_axis=True,
        ),
        use_container_width=True,
    )
    ui.note("A flat curve has identical zero and forward rates — the simplest possible term structure.")
else:
    st.sidebar.subheader("Deposit quotes")
    quotes = []
    defaults = [0.020, 0.023, 0.026, 0.028, 0.030]
    for (label, _, _), dflt in zip(curves.DEPOSIT_PILLARS, defaults, strict=True):
        quotes.append(st.sidebar.slider(f"{label} deposit", 0.0, 0.08, dflt, 0.0005, format="%.4f"))
    res = curves.bootstrap_deposit_curve(quotes, max_years=1.0)
    st.subheader("Bootstrapped discount curve")
    ui.note("PQuantLib iteratively solves for discount factors that reprice every input deposit exactly.")
    st.plotly_chart(
        charts.line(
            res.profile.tenors,
            res.profile.discount_factors,
            xtitle="Maturity (years)",
            ytitle="Discount factor",
            name="Discount",
            fill=True,
        ),
        use_container_width=True,
    )
    st.plotly_chart(
        charts.multi_line(
            res.profile.tenors,
            [
                ("Zero rate", res.profile.zero_rates, charts.ACCENT),
                ("Forward", res.profile.forward_rates, charts.ACCENT3),
            ],
            xtitle="Maturity (years)",
            ytitle="Rate",
            percent_axis=True,
        ),
        use_container_width=True,
    )
    import pandas as pd

    st.dataframe(
        pd.DataFrame(
            {
                "Pillar": res.pillar_labels,
                "Years": res.pillar_years,
                "Quoted rate": [f"{q:.4%}" for q in res.pillar_quotes],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

"""
ForecastWise — MR D.I.Y. Revenue Forecasting Dashboard
========================================================
Streamlit app integrating Member 3's Moving Average & Least Squares Trend
forecasting with interactive scenario simulation and management insights.
"""

import sys, os, pathlib

# ── Make sure src/ is importable ──────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from forecasting import (
    moving_average_forecast,
    least_squares_fit,
    least_squares_forecast,
    forecast_next,
    rolling_one_step_forecasts,
    next_period_label,
    apply_scenario,
    MA,
    LS,
)
from accuracy import accuracy_table, select_method, mad, mse, mape, bias, rmse


# ═══════════════════════════════════════════════════════════════════════════
# Page config
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ForecastWise — MR D.I.Y.",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* KPI cards */
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px 24px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,.12);
        margin-bottom: 8px;
    }
    .kpi-card h3 {
        margin: 0 0 4px 0;
        font-size: 14px;
        font-weight: 500;
        opacity: 0.9;
    }
    .kpi-card h1 {
        margin: 0;
        font-size: 28px;
        font-weight: 700;
    }
    .kpi-card p {
        margin: 6px 0 0 0;
        font-size: 13px;
        opacity: 0.85;
    }
    .kpi-green  { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .kpi-blue   { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .kpi-orange { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
    .kpi-teal   { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #1a1a2e, #16213e);
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        margin: 24px 0 16px 0;
        font-size: 18px;
        font-weight: 600;
    }

    /* Insight box */
    .insight-box {
        background: #f0f4ff;
        border-left: 4px solid #667eea;
        padding: 16px 20px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
        font-size: 14px;
        line-height: 1.6;
    }
    .insight-box strong { color: #1a1a2e; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Data loading (cached)
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    rev = pd.read_csv(ROOT / "data" / "MRDIY_Cleaned_Revenue.csv")
    growth = pd.read_csv(ROOT / "data" / "MRDIY_Revenue_Growth_Analysis.csv")
    rev["t"] = np.arange(1, len(rev) + 1)
    # Parse year & quarter for grouping
    rev["year"] = rev["period"].str.split("-Q").str[0].astype(int)
    rev["quarter"] = "Q" + rev["period"].str.split("-Q").str[1]
    return rev, growth


df, growth_df = load_data()
revenue = df["revenue_rm_million"].to_numpy()


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏪 MR D.I.Y.")
    st.markdown("## ForecastWise Dashboard")
    st.markdown("**MR D.I.Y. Group (M) Berhad**")
    st.markdown("---")

    page = st.radio(
        "📑 Navigation",
        [
            "🏠 Main Dashboard",
            "📈 Revenue Overview",
            "🔮 Forecasting & Accuracy",
            "🎛️ Scenario Simulator",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown(
        """
        <div style='font-size:12px; color:grey; text-align:center;'>
        GFB3213 Forecasting Methods<br>
        September 2026 Semester<br>
        Data: 2022-Q1 to 2026-Q2
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Helper: render a KPI card
# ═══════════════════════════════════════════════════════════════════════════
def kpi(title, value, subtitle="", css_class="kpi-blue"):
    st.markdown(
        f"""<div class="kpi-card {css_class}">
                <h3>{title}</h3>
                <h1>{value}</h1>
                <p>{subtitle}</p>
            </div>""",
        unsafe_allow_html=True,
    )


def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def insight(text):
    st.markdown(f'<div class="insight-box">{text}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Shared computations
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data
def compute_forecasts():
    """Run rolling forecasts and accuracy comparisons."""
    val = rolling_one_step_forecasts(df, "2025-Q3")
    rob = rolling_one_step_forecasts(df, "2023-Q1")

    primary = accuracy_table(val["actual"], {MA: val["ma_forecast"], LS: val["ls_forecast"]})
    robust = accuracy_table(rob["actual"], {MA: rob["ma_forecast"], LS: rob["ls_forecast"]})

    selected, reason = select_method(primary, robust, decision_metrics=("MAD", "MSE"))

    # Next-quarter forecast
    next_q = next_period_label(df["period"].iloc[-1])
    fcst_ma = forecast_next(revenue, MA)
    fcst_ls = forecast_next(revenue, LS)
    a_all, b_all = least_squares_fit(revenue)

    return val, rob, primary, robust, selected, reason, next_q, fcst_ma, fcst_ls, a_all, b_all

val, rob, primary, robust, SELECTED, REASON, next_q, fcst_ma, fcst_ls, a_all, b_all = compute_forecasts()
base_forecast = fcst_ls if SELECTED == LS else fcst_ma


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                        PAGE 1: MAIN DASHBOARD                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
if page == "🏠 Main Dashboard":
    st.markdown("# 📊 ForecastWise — MR D.I.Y. Revenue Dashboard")
    st.markdown("*Comprehensive revenue analytics & forecasting for MR D.I.Y. Group (M) Berhad*")

    # ── KPI Cards ─────────────────────────────────────────────────────────
    section("📋 Key Performance Indicators")

    latest_rev = df["revenue_rm_million"].iloc[-1]
    total_rev = df["revenue_rm_million"].sum()
    avg_rev = df["revenue_rm_million"].mean()
    cagr = ((df["revenue_rm_million"].iloc[-1] / df["revenue_rm_million"].iloc[0]) ** (1 / ((len(df) - 1) / 4)) - 1) * 100

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Latest Revenue (2026-Q2)", f"RM {latest_rev:,.1f}M", "Quarter-on-Quarter", "kpi-blue")
    with c2:
        kpi("Total Revenue (18 Qtrs)", f"RM {total_rev:,.0f}M", f"2022-Q1 to 2026-Q2", "kpi-green")
    with c3:
        kpi("Avg Quarterly Revenue", f"RM {avg_rev:,.1f}M", f"Mean across {len(df)} quarters", "kpi-teal")
    with c4:
        kpi("Annualised Growth (CAGR)", f"{cagr:.1f}%", "Compounded annual rate", "kpi-orange")

    st.markdown("")

    # ── Interactive Revenue Trend ──────────────────────────────────────────
    section("📈 Revenue Trend (2022-Q1 — 2026-Q2)")

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=df["period"], y=df["revenue_rm_million"],
        mode="lines+markers",
        name="Actual Revenue",
        line=dict(color="#667eea", width=3),
        marker=dict(size=8, color="#667eea"),
        hovertemplate="<b>%{x}</b><br>Revenue: RM %{y:,.1f}M<extra></extra>",
    ))
    # Add trend line (LS)
    trend_y = a_all + b_all * df["t"].values
    fig_trend.add_trace(go.Scatter(
        x=df["period"], y=trend_y,
        mode="lines",
        name=f"Trend: Y = {a_all:,.1f} + {b_all:,.1f}t",
        line=dict(color="#f5576c", width=2, dash="dash"),
        hovertemplate="<b>%{x}</b><br>Trend: RM %{y:,.1f}M<extra></extra>",
    ))
    fig_trend.update_layout(
        height=420,
        xaxis_title="Quarter",
        yaxis_title="Revenue (RM Million)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # ── Model Comparison Summary ──────────────────────────────────────────
    section("🔮 Model Comparison Results")

    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown("#### Validation Accuracy (2025-Q3 → 2026-Q2)")
        styled_primary = primary.copy()
        styled_primary.columns = ["MAD (RM M)", "MSE (RM M²)", "MAPE (%)", "Bias (RM M)"]
        st.dataframe(styled_primary.round(2), use_container_width=True)

    with mc2:
        st.markdown("#### Robustness Backtest (2023-Q1 → 2026-Q2)")
        styled_robust = robust.copy()
        styled_robust.columns = ["MAD (RM M)", "MSE (RM M²)", "MAPE (%)", "Bias (RM M)"]
        st.dataframe(styled_robust.round(2), use_container_width=True)

    st.success(f"✅ **Selected Method: {SELECTED}** — {REASON}")

    fc1, fc2 = st.columns(2)
    with fc1:
        kpi("Next-Quarter Forecast", f"RM {base_forecast:,.1f}M", f"{next_q} ({SELECTED})", "kpi-green")
    with fc2:
        alt = fcst_ma if SELECTED == LS else fcst_ls
        alt_name = MA if SELECTED == LS else LS
        kpi("Alternative Method", f"RM {alt:,.1f}M", f"{next_q} ({alt_name})", "kpi-teal")

    # ── Management Insights ───────────────────────────────────────────────
    section("💡 Management Insights")

    insight(
        f"""
        <strong>1. Consistent Growth Trajectory:</strong> MR D.I.Y.'s revenue has grown at an average rate of
        <strong>RM {b_all:,.1f}M per quarter</strong> (≈ RM {b_all * 4:,.0f}M annualised), demonstrating strong,
        steady expansion from RM {df['revenue_rm_million'].iloc[0]:,.1f}M in 2022-Q1 to RM {df['revenue_rm_million'].iloc[-1]:,.1f}M
        in 2026-Q2.
        """
    )
    insight(
        f"""
        <strong>2. Seasonal Pattern Detected:</strong> Revenue consistently dips from Q2→Q3 and rebounds in Q4
        across all years (2022–2025). Management should plan inventory and staffing to capitalise on Q4 peaks
        and manage the Q3 lull.
        """
    )
    insight(
        f"""
        <strong>3. Forecast Reliability:</strong> The {SELECTED} model achieves a <strong>MAPE of
        {robust.loc[SELECTED, 'MAPE (%)']:.2f}%</strong> over 14 backtested quarters, meaning the typical
        forecast is off by less than 4% — highly reliable for quarterly planning. The next-quarter forecast
        of <strong>RM {base_forecast:,.1f}M</strong> for {next_q} should be used as the planning baseline.
        """
    )
    insight(
        """
        <strong>4. Risk Considerations:</strong> A straight-line trend assumes growth continues at a constant rate.
        External shocks (economic downturns, supply-chain disruptions, regulatory changes) are not captured.
        Use the <strong>Scenario Simulator</strong> tab to stress-test the forecast under different assumptions.
        """
    )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                       PAGE 2: REVENUE OVERVIEW                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
elif page == "📈 Revenue Overview":
    st.markdown("# 📈 Revenue Overview")
    st.markdown("*Historical revenue trend, quarterly comparisons, growth rates and key statistics*")

    # ── Key Statistics ────────────────────────────────────────────────────
    section("📋 Key Revenue Statistics")

    stats = df["revenue_rm_million"].describe()
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi("Mean", f"RM {stats['mean']:,.1f}M", "Avg quarterly", "kpi-blue")
    with c2:
        kpi("Median", f"RM {stats['50%']:,.1f}M", "Midpoint", "kpi-green")
    with c3:
        kpi("Std Dev", f"RM {stats['std']:,.1f}M", "Variation", "kpi-orange")
    with c4:
        kpi("Min", f"RM {stats['min']:,.1f}M", df["period"].iloc[df["revenue_rm_million"].argmin()], "kpi-teal")
    with c5:
        kpi("Max", f"RM {stats['max']:,.1f}M", df["period"].iloc[df["revenue_rm_million"].argmax()], "kpi-blue")

    # ── Historical Trend ──────────────────────────────────────────────────
    section("📈 Historical Revenue Trend")

    fig_hist = go.Figure()
    # Add bar chart background
    fig_hist.add_trace(go.Bar(
        x=df["period"], y=df["revenue_rm_million"],
        name="Revenue",
        marker_color="#667eea",
        opacity=0.3,
        hovertemplate="<b>%{x}</b><br>Revenue: RM %{y:,.1f}M<extra></extra>",
    ))
    # Add line
    fig_hist.add_trace(go.Scatter(
        x=df["period"], y=df["revenue_rm_million"],
        mode="lines+markers",
        name="Trend",
        line=dict(color="#667eea", width=3),
        marker=dict(size=7),
        hovertemplate="<b>%{x}</b><br>Revenue: RM %{y:,.1f}M<extra></extra>",
    ))
    fig_hist.update_layout(
        height=450,
        xaxis_title="Quarter",
        yaxis_title="Revenue (RM Million)",
        template="plotly_white",
        showlegend=False,
        margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── Quarterly Comparison (heatmap-style) ──────────────────────────────
    section("📊 Quarterly Comparison by Year")

    pivot = df.pivot_table(index="year", columns="quarter", values="revenue_rm_million", aggfunc="first")
    pivot = pivot.reindex(columns=["Q1", "Q2", "Q3", "Q4"])

    fig_q = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=[str(y) for y in pivot.index.tolist()],
        text=[[f"RM {v:,.1f}M" if not np.isnan(v) else "" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        textfont=dict(size=13, color="white"),
        colorscale="Viridis",
        hovertemplate="Year: %{y}<br>Quarter: %{x}<br>Revenue: %{text}<extra></extra>",
        colorbar_title="RM M",
    ))
    fig_q.update_layout(
        height=320,
        xaxis_title="Quarter",
        yaxis_title="Year",
        template="plotly_white",
        margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig_q, use_container_width=True)

    # ── Growth Rates ──────────────────────────────────────────────────────
    section("📉 Growth Rates")

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("#### Quarter-on-Quarter (QoQ) Growth")
        qoq = growth_df[["period", "QoQ_Growth_%"]].copy()
        qoq["QoQ_Growth_%"] = pd.to_numeric(qoq["QoQ_Growth_%"], errors="coerce")
        qoq_valid = qoq.dropna()

        colors = ["#38ef7d" if v >= 0 else "#f5576c" for v in qoq_valid["QoQ_Growth_%"]]
        fig_qoq = go.Figure(go.Bar(
            x=qoq_valid["period"], y=qoq_valid["QoQ_Growth_%"],
            marker_color=colors,
            hovertemplate="<b>%{x}</b><br>QoQ Growth: %{y:.1f}%<extra></extra>",
        ))
        fig_qoq.update_layout(
            height=350,
            xaxis_title="Quarter",
            yaxis_title="Growth (%)",
            template="plotly_white",
            margin=dict(t=20, b=40),
        )
        fig_qoq.add_hline(y=0, line_color="grey", line_dash="dash", line_width=1)
        st.plotly_chart(fig_qoq, use_container_width=True)

    with g2:
        st.markdown("#### Year-on-Year (YoY) Growth")
        yoy = growth_df[["period", "YoY_Growth_%"]].copy()
        yoy["YoY_Growth_%"] = pd.to_numeric(yoy["YoY_Growth_%"], errors="coerce")
        yoy_valid = yoy.dropna()

        colors_yoy = ["#4facfe" if v >= 0 else "#f5576c" for v in yoy_valid["YoY_Growth_%"]]
        fig_yoy = go.Figure(go.Bar(
            x=yoy_valid["period"], y=yoy_valid["YoY_Growth_%"],
            marker_color=colors_yoy,
            hovertemplate="<b>%{x}</b><br>YoY Growth: %{y:.1f}%<extra></extra>",
        ))
        fig_yoy.update_layout(
            height=350,
            xaxis_title="Quarter",
            yaxis_title="Growth (%)",
            template="plotly_white",
            margin=dict(t=20, b=40),
        )
        fig_yoy.add_hline(y=0, line_color="grey", line_dash="dash", line_width=1)
        st.plotly_chart(fig_yoy, use_container_width=True)

    # ── Full Data Table ───────────────────────────────────────────────────
    section("📋 Complete Revenue Dataset")
    display_df = growth_df.copy()
    display_df.columns = ["Period", "Revenue (RM M)", "QoQ Growth (%)", "YoY Growth (%)"]
    st.dataframe(display_df, use_container_width=True, height=400)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                     PAGE 3: FORECASTING & ACCURACY                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
elif page == "🔮 Forecasting & Accuracy":
    st.markdown("# 🔮 Forecasting & Accuracy Analysis")
    st.markdown("*Integrating Member 3's Moving Average and Least Squares Trend calculations*")

    # ── Method Selection ──────────────────────────────────────────────────
    section("⚙️ Model Selection & Configuration")

    ms1, ms2 = st.columns(2)
    with ms1:
        st.markdown("""
        **Method A — 4-Quarter Moving Average**

        $$F_{t} = \\frac{A_{t-1} + A_{t-2} + A_{t-3} + A_{t-4}}{4}$$

        Uses the 4 most recent actual quarters to smooth seasonal swings.
        Each average spans one Q1, Q2, Q3 and Q4.
        """)

    with ms2:
        a_all_disp, b_all_disp = a_all, b_all
        st.markdown(f"""
        **Method B — Least Squares Trend Projection**

        $$\\hat{{Y}} = a + bt$$

        Fitted on all 18 quarters:
        **Y = {a_all_disp:,.2f} + {b_all_disp:,.2f}t**

        Revenue grows ~RM {b_all_disp:,.1f}M per quarter on average.
        """)

    # ── Rolling Forecast Chart ────────────────────────────────────────────
    section("📈 Actual vs Rolling One-Quarter-Ahead Forecasts")

    fig_fc = go.Figure()
    # Actual
    fig_fc.add_trace(go.Scatter(
        x=df["period"], y=df["revenue_rm_million"],
        mode="lines+markers",
        name="Actual Revenue",
        line=dict(color="#0b0b0b", width=2.5),
        marker=dict(size=7, color="#0b0b0b"),
    ))
    # MA backtest
    fig_fc.add_trace(go.Scatter(
        x=rob["period"], y=rob["ma_forecast"],
        mode="lines+markers",
        name=MA,
        line=dict(color="#2a78d6", width=2),
        marker=dict(size=6, symbol="square", color="#2a78d6"),
    ))
    # LS backtest
    fig_fc.add_trace(go.Scatter(
        x=rob["period"], y=rob["ls_forecast"],
        mode="lines+markers",
        name=LS,
        line=dict(color="#eb6834", width=2),
        marker=dict(size=7, symbol="triangle-up", color="#eb6834"),
    ))
    # Validation region
    v_start = val["period"].iloc[0]
    v_end = val["period"].iloc[-1]
    all_periods = df["period"].tolist()
    v0_idx = all_periods.index(v_start)
    v1_idx = all_periods.index(v_end)
    fig_fc.add_vrect(
        x0=v0_idx - 0.5, x1=v1_idx + 0.5,
        fillcolor="rgba(241,240,236,0.5)", line_width=0,
        annotation_text="Validation Holdout", annotation_position="top",
    )
    fig_fc.update_layout(
        height=480,
        xaxis_title="Quarter",
        yaxis_title="Revenue (RM Million)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # ── Forecast Results ──────────────────────────────────────────────────
    section("📊 Forecast Results — Validation Quarters")

    val_display = val[["period", "actual", "ma_forecast", "ls_forecast", "ma_error", "ls_error"]].copy()
    val_display.columns = ["Quarter", "Actual (RM M)", "MA Forecast", "LS Forecast", "MA Error (A-F)", "LS Error (A-F)"]
    st.dataframe(val_display.round(3), use_container_width=True)

    # ── Absolute Error Chart ──────────────────────────────────────────────
    section("📊 Absolute Forecast Errors (Validation Quarters)")

    fig_err = go.Figure()
    fig_err.add_trace(go.Bar(
        x=val["period"], y=val["ma_error"].abs(),
        name=MA,
        marker_color="#2a78d6",
        text=[f"{abs(e):.1f}<br>{'under' if e > 0 else 'over'}" for e in val["ma_error"]],
        textposition="outside",
        textfont=dict(size=10),
    ))
    fig_err.add_trace(go.Bar(
        x=val["period"], y=val["ls_error"].abs(),
        name=LS,
        marker_color="#eb6834",
        text=[f"{abs(e):.1f}<br>{'under' if e > 0 else 'over'}" for e in val["ls_error"]],
        textposition="outside",
        textfont=dict(size=10),
    ))
    fig_err.update_layout(
        barmode="group",
        height=420,
        xaxis_title="Quarter",
        yaxis_title="Absolute Error |A - F| (RM Million)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig_err, use_container_width=True)

    # ── Accuracy Comparison ───────────────────────────────────────────────
    section("🎯 Accuracy Metrics — MAD, MSE & MAPE")

    ac1, ac2 = st.columns(2)
    with ac1:
        st.markdown(f"#### Validation Window ({val['period'].iloc[0]} → {val['period'].iloc[-1]}, n={len(val)})")
        p_disp = primary.copy()
        p_disp.columns = ["MAD (RM M)", "MSE (RM M²)", "MAPE (%)", "Bias (RM M)"]
        st.dataframe(p_disp.round(3), use_container_width=True)

        # Highlight winners
        for metric in ["MAD (RM M)", "MSE (RM M²)", "MAPE (%)"]:
            winner = p_disp[metric].idxmin()
            st.markdown(f"- **{metric}**: {winner} wins ({p_disp.loc[winner, metric]:,.3f})")

    with ac2:
        st.markdown(f"#### Robustness Backtest ({rob['period'].iloc[0]} → {rob['period'].iloc[-1]}, n={len(rob)})")
        r_disp = robust.copy()
        r_disp.columns = ["MAD (RM M)", "MSE (RM M²)", "MAPE (%)", "Bias (RM M)"]
        st.dataframe(r_disp.round(3), use_container_width=True)

        for metric in ["MAD (RM M)", "MSE (RM M²)", "MAPE (%)"]:
            winner = r_disp[metric].idxmin()
            st.markdown(f"- **{metric}**: {winner} wins ({r_disp.loc[winner, metric]:,.3f})")

    # ── Method Selection Result ───────────────────────────────────────────
    section("✅ Method Selection")
    st.success(f"**Selected Method: {SELECTED}**")
    st.info(f"**Selection Reason:** {REASON}")

    bias_ma_val = primary.loc[MA, "Bias (mean error)"]
    bias_ls_val = primary.loc[LS, "Bias (mean error)"]
    st.markdown(f"""
    **Bias Diagnostics (Validation):**
    - Moving Average bias: **{bias_ma_val:+,.2f} RM M** (systematically under-forecasts)
    - Least Squares bias: **{bias_ls_val:+,.2f} RM M** (slight over-forecast)

    The MA method lags behind a rising series. For MR D.I.Y., persistent under-forecasting risks
    under-stocking and under-staffing. The LS Trend captures the growth trajectory more accurately.
    """)

    # ── Next-Quarter Forecast ─────────────────────────────────────────────
    section(f"🔮 Next-Quarter Forecast: {next_q}")

    nf1, nf2 = st.columns(2)
    with nf1:
        kpi(f"{MA}", f"RM {fcst_ma:,.1f}M", f"Mean of last 4 quarters", "kpi-blue")
    with nf2:
        kpi(f"{LS} ✅", f"RM {fcst_ls:,.1f}M", f"Y = {a_all:,.1f} + {b_all:,.1f} × {len(revenue) + 1}", "kpi-green")

    band = rmse(val["actual"], val["ls_forecast"] if SELECTED == LS else val["ma_forecast"])
    st.markdown(f"**Typical error:** ±RM {band:,.1f}M (validation RMSE)")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                      PAGE 4: SCENARIO SIMULATOR                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
elif page == "🎛️ Scenario Simulator":
    st.markdown("# 🎛️ Scenario Simulator")
    st.markdown("*Simulate promotional uplift, consumer-spending slowdowns, and other adjustments*")

    section("⚙️ Scenario Configuration")

    sc1, sc2, sc3 = st.columns([1, 1, 1])
    with sc1:
        st.markdown(f"**Baseline Forecast ({next_q})**")
        st.markdown(f"### RM {base_forecast:,.1f}M")
        st.markdown(f"Method: {SELECTED}")

    with sc2:
        scenario_type = st.selectbox(
            "Select Scenario Type",
            [
                "Custom Adjustment",
                "🟢 Promotional Uplift (+5% to +20%)",
                "🔴 Consumer Slowdown (-5% to -20%)",
                "🟡 Moderate Growth (+2% to +5%)",
                "⚪ Stagnation (0%)",
            ],
        )

    with sc3:
        if scenario_type == "🟢 Promotional Uplift (+5% to +20%)":
            pct = st.slider("Uplift Percentage (%)", 5, 20, 10)
        elif scenario_type == "🔴 Consumer Slowdown (-5% to -20%)":
            pct = st.slider("Decline Percentage (%)", -20, -5, -10)
        elif scenario_type == "🟡 Moderate Growth (+2% to +5%)":
            pct = st.slider("Growth Percentage (%)", 2, 5, 3)
        elif scenario_type == "⚪ Stagnation (0%)":
            pct = 0
        else:
            pct = st.slider("Adjustment (%)", -30, 30, 0, step=1)

    # Compute scenario
    scenario_forecast = apply_scenario(base_forecast, pct)
    change_rm = scenario_forecast - base_forecast

    section("📊 Scenario Results")

    sr1, sr2, sr3 = st.columns(3)
    with sr1:
        kpi("Baseline Forecast", f"RM {base_forecast:,.1f}M", f"{next_q}", "kpi-blue")
    with sr2:
        css = "kpi-green" if pct >= 0 else "kpi-orange"
        kpi("Scenario Forecast", f"RM {scenario_forecast:,.1f}M", f"{pct:+d}% adjustment", css)
    with sr3:
        kpi("Revenue Impact", f"{'+'if change_rm >= 0 else ''}RM {change_rm:,.1f}M", f"vs baseline", "kpi-teal")

    # ── Multi-scenario comparison ─────────────────────────────────────────
    section("📊 Multi-Scenario Comparison")

    scenarios = [
        {"Scenario": "Severe Slowdown", "Adjustment (%)": -15, "Color": "#f5576c"},
        {"Scenario": "Moderate Slowdown", "Adjustment (%)": -5, "Color": "#ff9a76"},
        {"Scenario": "Baseline", "Adjustment (%)": 0, "Color": "#667eea"},
        {"Scenario": "Moderate Growth", "Adjustment (%)": 5, "Color": "#38ef7d"},
        {"Scenario": "Promotional Uplift", "Adjustment (%)": 10, "Color": "#11998e"},
        {"Scenario": "Strong Promotion", "Adjustment (%)": 15, "Color": "#00b4d8"},
    ]
    for s in scenarios:
        s["Forecast (RM M)"] = apply_scenario(base_forecast, s["Adjustment (%)"])
        s["Change vs Baseline"] = s["Forecast (RM M)"] - base_forecast

    scen_df = pd.DataFrame(scenarios)

    fig_scen = go.Figure()
    fig_scen.add_trace(go.Bar(
        x=scen_df["Scenario"],
        y=scen_df["Forecast (RM M)"],
        marker_color=scen_df["Color"],
        text=[f"RM {v:,.0f}M" for v in scen_df["Forecast (RM M)"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Forecast: RM %{y:,.1f}M<extra></extra>",
    ))
    fig_scen.add_hline(
        y=base_forecast, line_dash="dash", line_color="#667eea",
        annotation_text=f"Baseline: RM {base_forecast:,.0f}M",
    )
    fig_scen.update_layout(
        height=420,
        yaxis_title="Forecast Revenue (RM Million)",
        template="plotly_white",
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig_scen, use_container_width=True)

    st.dataframe(
        scen_df[["Scenario", "Adjustment (%)", "Forecast (RM M)", "Change vs Baseline"]].round(1),
        use_container_width=True,
    )

    # ── Forecast trajectory with scenario ─────────────────────────────────
    section("📈 Forecast Trajectory with Scenario")

    traj_periods = list(df["period"]) + [next_q]
    traj_actual = list(df["revenue_rm_million"]) + [None]
    traj_base = [None] * len(df) + [base_forecast]
    traj_scen = [None] * len(df) + [scenario_forecast]

    fig_traj = go.Figure()
    fig_traj.add_trace(go.Scatter(
        x=traj_periods, y=traj_actual,
        mode="lines+markers",
        name="Actual Revenue",
        line=dict(color="#0b0b0b", width=2.5),
        marker=dict(size=7),
    ))
    fig_traj.add_trace(go.Scatter(
        x=[df["period"].iloc[-1], next_q],
        y=[df["revenue_rm_million"].iloc[-1], base_forecast],
        mode="lines+markers",
        name="Baseline Forecast",
        line=dict(color="#667eea", width=2, dash="dash"),
        marker=dict(size=10, symbol="star", color="#667eea"),
    ))
    if pct != 0:
        fig_traj.add_trace(go.Scatter(
            x=[df["period"].iloc[-1], next_q],
            y=[df["revenue_rm_million"].iloc[-1], scenario_forecast],
            mode="lines+markers",
            name=f"Scenario ({pct:+d}%)",
            line=dict(color="#f5576c" if pct < 0 else "#38ef7d", width=2, dash="dot"),
            marker=dict(size=10, symbol="diamond", color="#f5576c" if pct < 0 else "#38ef7d"),
        ))
    fig_traj.update_layout(
        height=450,
        xaxis_title="Quarter",
        yaxis_title="Revenue (RM Million)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig_traj, use_container_width=True)

    # ── Management Recommendations ────────────────────────────────────────
    section("💡 Management Recommendations")

    if pct > 10:
        insight(
            f"""
            <strong>🟢 Strong Promotional Uplift Scenario ({pct:+d}%):</strong><br>
            • Expected revenue: <strong>RM {scenario_forecast:,.1f}M</strong> (↑ RM {change_rm:,.1f}M vs baseline)<br>
            • <strong>Recommend:</strong> Increase inventory by {pct}–{pct + 5}% to meet anticipated demand surge.
            Pre-position stock in high-traffic stores 2–3 weeks before the promotional period.<br>
            • <strong>Staffing:</strong> Hire temporary workers and extend operating hours during peak promotion days.<br>
            • <strong>Supply Chain:</strong> Coordinate with key suppliers for expedited restocking. Secure additional warehouse space.<br>
            • <strong>Marketing:</strong> Allocate incremental budget for digital ads & in-store signage to maximise awareness.
            """
        )
    elif pct > 0:
        insight(
            f"""
            <strong>🟡 Moderate Growth Scenario ({pct:+d}%):</strong><br>
            • Expected revenue: <strong>RM {scenario_forecast:,.1f}M</strong> (↑ RM {change_rm:,.1f}M vs baseline)<br>
            • <strong>Recommend:</strong> Maintain current inventory planning with a modest {pct}–{pct + 2}% buffer.<br>
            • <strong>Staffing:</strong> Schedule normal rostering with flexibility for peak days.<br>
            • <strong>Focus:</strong> Continue store expansion strategy and invest in customer loyalty programmes.
            """
        )
    elif pct == 0:
        insight(
            f"""
            <strong>⚪ Baseline Scenario (No Adjustment):</strong><br>
            • Expected revenue: <strong>RM {base_forecast:,.1f}M</strong><br>
            • <strong>Recommend:</strong> Proceed with standard quarterly plans. Use the LS Trend forecast as the
            primary target for procurement, staffing and capital expenditure planning.<br>
            • <strong>Monitor:</strong> Track weekly sales against the forecast and trigger contingency plans if
            cumulative variance exceeds ±5%.
            """
        )
    elif pct > -10:
        insight(
            f"""
            <strong>🟠 Mild Slowdown Scenario ({pct:+d}%):</strong><br>
            • Expected revenue: <strong>RM {scenario_forecast:,.1f}M</strong> (↓ RM {abs(change_rm):,.1f}M vs baseline)<br>
            • <strong>Recommend:</strong> Reduce discretionary spending (marketing events, non-essential capex) by 5–10%.<br>
            • <strong>Inventory:</strong> Tighten ordering to avoid overstocking; negotiate extended payment terms with suppliers.<br>
            • <strong>Pricing:</strong> Consider targeted markdowns on slow-moving categories to maintain foot traffic.
            """
        )
    else:
        insight(
            f"""
            <strong>🔴 Significant Slowdown Scenario ({pct:+d}%):</strong><br>
            • Expected revenue: <strong>RM {scenario_forecast:,.1f}M</strong> (↓ RM {abs(change_rm):,.1f}M vs baseline)<br>
            • <strong>Recommend:</strong> Implement cost-containment measures immediately. Freeze non-critical hiring and
            defer store renovations.<br>
            • <strong>Inventory:</strong> Cut purchase orders by {abs(pct)}–{abs(pct) + 5}%. Focus on fast-moving SKUs and
            essentials.<br>
            • <strong>Cash Flow:</strong> Accelerate receivables collection and renegotiate supplier terms for
            extended credit periods.<br>
            • <strong>Strategy:</strong> Shift marketing spend to value-oriented messaging. Emphasise MR D.I.Y.'s
            low-price positioning as a consumer advantage during economic headwinds.
            """
        )

    # Disclaimer
    st.markdown("---")
    st.caption(
        "⚠️ **Disclaimer:** Scenarios are hypothetical managerial adjustments applied to the statistical "
        "baseline forecast. They are not statistical predictions. Actual results depend on market conditions, "
        "consumer behaviour, competition, and company-specific factors."
    )

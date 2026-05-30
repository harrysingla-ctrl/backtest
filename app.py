"""
Regime Signal Dashboard
=======================
4-signal weekly system: NiftyBees · MID150BEES · GoldBees
Auto-fetches live weekly data from Yahoo Finance.
Falls back to manual input if fetch fails.

Run:  streamlit run app.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Regime Dashboard",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.rec-wrap {
    border-radius: 20px;
    padding: 28px 20px;
    text-align: center;
    margin: 20px 0 28px 0;
}
.rec-label {
    font-size: 11px;
    letter-spacing: 2.5px;
    color: #8a8a8a;
    text-transform: uppercase;
    margin-bottom: 10px;
    font-family: 'Space Mono', monospace;
}
.rec-value {
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -0.5px;
    line-height: 1.1;
}

.sig-card {
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 10px;
    font-size: 14.5px;
    line-height: 1.7;
}
.sig-card.indent { margin-left: 28px; }
.sig-title {
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 6px;
    font-family: 'Space Mono', monospace;
}
.sig-arrow {
    font-size: 15px;
    font-weight: 700;
    margin-top: 8px;
}

.val-row {
    display: flex;
    gap: 10px;
    margin: 16px 0;
    flex-wrap: wrap;
}
.val-box {
    flex: 1;
    min-width: 120px;
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 14px 12px;
    text-align: center;
}
.val-label {
    font-size: 10px;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-family: 'Space Mono', monospace;
    margin-bottom: 4px;
}
.val-number {
    font-size: 20px;
    font-weight: 700;
    font-family: 'Space Mono', monospace;
}
.val-delta {
    font-size: 12px;
    margin-top: 3px;
    font-family: 'Space Mono', monospace;
}

.divider { border-top: 1px solid #222; margin: 20px 0; }
.footer-note {
    font-size: 12px;
    color: #555;
    text-align: center;
    margin-top: 16px;
    font-family: 'Space Mono', monospace;
}

/* Dark override */
.stApp { background-color: #0d0d0d; color: #e0e0e0; }
</style>
""", unsafe_allow_html=True)


# ─── Constants ─────────────────────────────────────────────────────────────────
TICKERS = {
    "NIFTYBEES":  "NIFTYBEES.NS",
    "MID150BEES": "MID150BEES.NS",
    "GOLDBEES":   "GOLDBEES.NS",
}
WEEKS = 42          # fetch window (buffer beyond 26w)
SMA_PERIOD = 26
BUFFER_PCT = 0.02   # ±2% around SMA
ROC_THRESHOLD = -5.0


# ─── Data Layer ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_weekly_data() -> tuple[dict, list[str]]:
    end   = datetime.today()
    start = end - timedelta(weeks=WEEKS)
    results, errors = {}, []

    for name, ticker in TICKERS.items():
        try:
            df = yf.download(
                ticker, start=start, end=end,
                interval="1wk", progress=False, auto_adjust=True,
            )
            # yfinance may return MultiIndex columns for single tickers in newer versions
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            closes = df["Close"].dropna()
            if len(closes) < SMA_PERIOD:
                errors.append(f"{name}: only {len(closes)} weeks returned (need {SMA_PERIOD})")
            else:
                results[name] = closes
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    return results, errors


def compute_signals(data: dict) -> dict:
    out = {}

    # NiftyBees
    if "NIFTYBEES" in data:
        nb = data["NIFTYBEES"]
        out["nifty_price"] = float(nb.iloc[-1])
        out["nifty_sma"]   = float(nb.tail(SMA_PERIOD).mean())

    # MID150/Nifty ratio
    if "NIFTYBEES" in data and "MID150BEES" in data:
        nb  = data["NIFTYBEES"]
        mid = data["MID150BEES"]
        df  = pd.concat([nb, mid], axis=1, join="inner")
        df.columns = ["nifty", "mid"]
        ratio = df["mid"] / df["nifty"]
        out["ratio_value"] = float(ratio.iloc[-1])
        out["ratio_sma"]   = float(ratio.tail(SMA_PERIOD).mean())

    # GoldBees
    if "GOLDBEES" in data:
        gold = data["GOLDBEES"]
        out["gold_price"] = float(gold.iloc[-1])
        out["gold_sma"]   = float(gold.tail(SMA_PERIOD).mean())

    # Equity/Gold 26w ROC
    if "NIFTYBEES" in data and "GOLDBEES" in data:
        nb   = data["NIFTYBEES"]
        gold = data["GOLDBEES"]
        df   = pd.concat([nb, gold], axis=1, join="inner")
        df.columns = ["nifty", "gold"]
        eg = df["nifty"] / df["gold"]
        if len(eg) >= SMA_PERIOD:
            curr = float(eg.iloc[-1])
            past = float(eg.iloc[-SMA_PERIOD])
            out["equity_gold_roc"] = ((curr - past) / past) * 100.0

    return out


# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 8px 0 4px 0">
  <span style="font-size:26px; font-weight:800; letter-spacing:-0.5px">📊 Regime Dashboard</span><br>
  <span style="font-size:13px; color:#555; font-family:'Space Mono',monospace">
    NiftyBees · MID150BEES · GoldBees &nbsp;|&nbsp; Weekly 26-SMA System
  </span>
</div>
""", unsafe_allow_html=True)

# ─── Fetch ─────────────────────────────────────────────────────────────────────
with st.spinner("Fetching live weekly data…"):
    raw_data, fetch_errors = fetch_weekly_data()
    computed = compute_signals(raw_data)

REQUIRED_KEYS = [
    "nifty_price", "nifty_sma",
    "ratio_value", "ratio_sma",
    "gold_price",  "gold_sma",
    "equity_gold_roc",
]
fetch_ok = all(k in computed for k in REQUIRED_KEYS)

if fetch_errors:
    with st.expander("⚠️ Fetch warnings (tap to expand)"):
        for err in fetch_errors:
            st.warning(err)

# ─── Input Panel ───────────────────────────────────────────────────────────────
with st.expander(
    "📥 Signal Inputs — " + ("Auto-filled ✅" if fetch_ok else "Manual entry required ⚠️"),
    expanded=not fetch_ok,
):
    if fetch_ok:
        st.caption("All values auto-populated. Edit to override.")
    else:
        st.caption("Live fetch incomplete. Enter values manually below.")

    c1, c2 = st.columns(2)
    with c1:
        nifty_price     = st.number_input("NiftyBees Price",       value=float(computed.get("nifty_price",     0.0)), format="%.2f", step=1.0)
        nifty_sma       = st.number_input("NiftyBees 26w SMA",     value=float(computed.get("nifty_sma",       0.0)), format="%.2f", step=1.0)
        ratio_value     = st.number_input("MID150/Nifty Ratio",    value=float(computed.get("ratio_value",     0.0)), format="%.4f", step=0.0005)
        ratio_sma       = st.number_input("Ratio 26w SMA",         value=float(computed.get("ratio_sma",       0.0)), format="%.4f", step=0.0005)
    with c2:
        gold_price      = st.number_input("GoldBees Price",        value=float(computed.get("gold_price",      0.0)), format="%.2f", step=1.0)
        gold_sma        = st.number_input("GoldBees 26w SMA",      value=float(computed.get("gold_sma",        0.0)), format="%.2f", step=1.0)
        equity_gold_roc = st.number_input("EQ/Gold 26w ROC (%)",   value=float(computed.get("equity_gold_roc", 0.0)), format="%.2f", step=0.1)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─── Signal Logic ──────────────────────────────────────────────────────────────
if nifty_price > 0 and nifty_sma > 0:

    upper_band = nifty_sma * (1 + BUFFER_PCT)
    lower_band = nifty_sma * (1 - BUFFER_PCT)
    nifty_pct  = ((nifty_price - nifty_sma) / nifty_sma) * 100

    # S1
    if nifty_price > upper_band:
        s1 = "EQUITY"
    elif nifty_price < lower_band:
        s1 = "DEFENSIVE"
    else:
        s1 = "BUFFER"

    # S2
    s2 = "MIDCAP" if (ratio_sma > 0 and ratio_value > ratio_sma) else "LARGECAP"

    # S3 + S4
    s3 = (gold_sma > 0) and (gold_price > gold_sma)
    s4 = equity_gold_roc < ROC_THRESHOLD

    gold_pct   = ((gold_price - gold_sma) / gold_sma * 100) if gold_sma > 0 else 0.0
    ratio_pct  = ((ratio_value - ratio_sma) / ratio_sma * 100) if ratio_sma > 0 else 0.0

    # Final recommendation
    if s1 == "EQUITY":
        rec      = s2
        rec_icon = "🟠" if s2 == "MIDCAP" else "🔵"
        rec_clr  = "#FF8C00" if s2 == "MIDCAP" else "#1E90FF"
    elif s1 == "DEFENSIVE":
        if s3 and s4:
            rec, rec_icon, rec_clr = "GOLD",  "🥇", "#DAA520"
        else:
            rec, rec_icon, rec_clr = "CASH",  "💵", "#3CB371"
    else:
        rec, rec_icon, rec_clr = "HOLD — No Change", "⏸", "#9B59B6"

    # ── Recommendation card ───────────────────────────────────────────────────
    st.markdown(f"""
    <div class="rec-wrap" style="background:{rec_clr}18; border: 2px solid {rec_clr}55">
        <div class="rec-label">Current Recommendation</div>
        <div class="rec-value" style="color:{rec_clr}">{rec_icon}&nbsp; {rec}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Signal flow ───────────────────────────────────────────────────────────
    st.markdown("#### Signal Flow")

    # S1 card
    s1_meta = {
        "EQUITY":    ("#2E7D32", "🟢", "EQUITY REGIME"),
        "DEFENSIVE": ("#C62828", "🔴", "DEFENSIVE REGIME"),
        "BUFFER":    ("#E65100", "🟡", "BUFFER ZONE — Hold current position"),
    }
    s1_clr, s1_ico, s1_label = s1_meta[s1]

    st.markdown(f"""
    <div class="sig-card" style="border:1.5px solid {s1_clr}44; background:{s1_clr}0d">
        <div class="sig-title">{s1_ico} &nbsp;S1 — Equity Regime Gate</div>
        NiftyBees &nbsp;<b>₹{nifty_price:,.2f}</b> &nbsp;|&nbsp; 26w SMA &nbsp;<b>₹{nifty_sma:,.2f}</b>
        &nbsp;|&nbsp; <b>{nifty_pct:+.2f}%</b> from SMA<br>
        Buffer band: ₹{lower_band:,.2f} – ₹{upper_band:,.2f}
        <div class="sig-arrow" style="color:{s1_clr}">→ {s1_label}</div>
    </div>
    """, unsafe_allow_html=True)

    # S2 card (only in equity regime)
    if s1 == "EQUITY":
        s2_clr  = "#FF8C00" if s2 == "MIDCAP" else "#1E90FF"
        s2_ico  = "🟠" if s2 == "MIDCAP" else "🔵"
        s2_note = "Ratio above SMA → risk appetite high → Midcap leads" if s2 == "MIDCAP" \
                  else "Ratio below SMA → rotation to quality → Largecap"
        st.markdown(f"""
        <div class="sig-card indent" style="border:1.5px solid {s2_clr}44; background:{s2_clr}0d">
            <div class="sig-title">{s2_ico} &nbsp;S2 — Mid vs Large Selector</div>
            Ratio &nbsp;<b>{ratio_value:.4f}</b> &nbsp;|&nbsp; Ratio SMA &nbsp;<b>{ratio_sma:.4f}</b>
            &nbsp;|&nbsp; <b>{ratio_pct:+.2f}%</b><br>
            {s2_note}
            <div class="sig-arrow" style="color:{s2_clr}">→ {s2}</div>
        </div>
        """, unsafe_allow_html=True)

    # S3 + S4 card (only in defensive regime)
    elif s1 == "DEFENSIVE":
        s3_ico = "✅" if s3 else "❌"
        s4_ico = "✅" if s4 else "❌"
        both   = s3 and s4
        outcome_clr = "#DAA520" if both else "#3CB371"
        outcome_lbl = "Both TRUE → GOLD" if both else ("One or both FALSE → CASH")
        st.markdown(f"""
        <div class="sig-card indent" style="border:1.5px solid #44444488; background:#1a1a1a">
            <div class="sig-title">🛡 &nbsp;S3 + S4 — Gold or Cash?</div>
            {s3_ico} &nbsp;<b>S3 Gold Uptrend</b> &nbsp;|&nbsp;
            GoldBees <b>₹{gold_price:,.2f}</b> vs SMA <b>₹{gold_sma:,.2f}</b>
            &nbsp;(<b style="color:{'#DAA520' if s3 else '#888'}">{gold_pct:+.2f}%</b>)
            &nbsp;→ <b style="color:{'#DAA520' if s3 else '#888'}">{'Above ✓' if s3 else 'Below ✗'}</b><br><br>
            {s4_ico} &nbsp;<b>S4 EQ/Gold ROC</b> &nbsp;|&nbsp;
            26w ROC <b style="color:{'#DAA520' if s4 else '#888'}">{equity_gold_roc:+.2f}%</b>
            &nbsp;(threshold −5%)
            &nbsp;→ <b style="color:{'#DAA520' if s4 else '#888'}">{'Gold outrunning ✓' if s4 else 'Not confirmed ✗'}</b>
            <div class="sig-arrow" style="color:{outcome_clr}">→ {outcome_lbl}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Key Values row ────────────────────────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("#### Key Values")

    def delta_color(val): return "#3CB371" if val >= 0 else "#E74C3C"
    def delta_arrow(val): return "▲" if val >= 0 else "▼"

    st.markdown(f"""
    <div class="val-row">
        <div class="val-box">
            <div class="val-label">NiftyBees</div>
            <div class="val-number">₹{nifty_price:,.0f}</div>
            <div class="val-delta" style="color:{delta_color(nifty_pct)}">{delta_arrow(nifty_pct)} {abs(nifty_pct):.2f}% vs SMA</div>
        </div>
        <div class="val-box">
            <div class="val-label">MID/Nifty Ratio</div>
            <div class="val-number">{ratio_value:.3f}</div>
            <div class="val-delta" style="color:{delta_color(ratio_pct)}">{delta_arrow(ratio_pct)} {abs(ratio_pct):.2f}% vs SMA</div>
        </div>
        <div class="val-box">
            <div class="val-label">GoldBees</div>
            <div class="val-number">₹{gold_price:,.0f}</div>
            <div class="val-delta" style="color:{delta_color(gold_pct)}">{delta_arrow(gold_pct)} {abs(gold_pct):.2f}% vs SMA</div>
        </div>
        <div class="val-box">
            <div class="val-label">EQ/Gold ROC</div>
            <div class="val-number" style="color:{'#E74C3C' if equity_gold_roc < ROC_THRESHOLD else '#e0e0e0'}">{equity_gold_roc:+.1f}%</div>
            <div class="val-delta" style="color:#555">threshold −5%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    now_str = datetime.now().strftime("%d %b %Y, %H:%M")
    src_str = "Live · Yahoo Finance" if fetch_ok else "Manual / Partial data"
    st.markdown(f'<div class="footer-note">{src_str} &nbsp;·&nbsp; {now_str} IST</div>', unsafe_allow_html=True)

    col_r, _ = st.columns([1, 3])
    with col_r:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

else:
    st.info("Enter NiftyBees price and 26w SMA above to compute signals.")

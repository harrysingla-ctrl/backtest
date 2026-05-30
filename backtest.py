"""
Backtest: 4-Signal App vs Kiru's Donchian Strategy
====================================================
Weekly data · Yahoo Finance · No look-ahead bias
Signals at end of week N → returns earned in week N+1

Run:  streamlit run backtest.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# ── Config ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Backtest Analysis",
    page_icon="📈",
    layout="centered",
)

TICKERS = {
    "NIFTYBEES":  "NIFTYBEES.NS",
    "MID150BEES": "MID150BEES.NS",
    "GOLDBEES":   "GOLDBEES.NS",
}
SMA_PERIOD      = 26
BUFFER          = 0.02
ROC_PERIOD      = 26
DONCHIAN_PERIOD = 20
RISK_FREE_RATE  = 0.065   # 6.5% p.a. India liquid fund proxy
CASH_WEEKLY     = RISK_FREE_RATE / 52
START_DATE      = "2010-01-01"
INITIAL_CAPITAL = 100_000  # ₹1 lakh


# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #0d0d0d; color: #e0e0e0; }
.metric-card {
    background: #141414;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    margin-bottom: 8px;
}
.metric-label { font-size: 11px; color: #666; letter-spacing: 1.5px; font-family: 'Space Mono', monospace; text-transform: uppercase; margin-bottom: 4px; }
.metric-value { font-size: 24px; font-weight: 700; font-family: 'Space Mono', monospace; }
.section-title { font-size: 16px; font-weight: 700; letter-spacing: -0.3px; margin: 24px 0 12px 0; border-bottom: 1px solid #222; padding-bottom: 8px; }
.winner-badge { display: inline-block; padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: 700; font-family: 'Space Mono', monospace; margin-left: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Data ───────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_all():
    results, errors = {}, []
    for name, ticker in TICKERS.items():
        try:
            df = yf.download(
                ticker, start=START_DATE, end=datetime.today().strftime("%Y-%m-%d"),
                interval="1wk", progress=False, auto_adjust=True,
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            closes = df["Close"].dropna()
            if len(closes) < SMA_PERIOD + 10:
                errors.append(f"{name}: only {len(closes)} weeks — need more history")
            else:
                results[name] = closes
                st.toast(f"✅ {name}: {len(closes)} weeks fetched")
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    return results, errors


# ── Strategy 1: 4-Signal App ───────────────────────────────────────────────────
def run_4signal(data: dict) -> pd.DataFrame:
    """
    S1: NiftyBees vs 26w SMA ± 2% buffer → Equity / Defensive / Hold
    S2: MID150/Nifty ratio vs 26w SMA    → Midcap / Largecap
    S3: GoldBees vs 26w SMA              → Gold uptrend
    S4: NiftyBees/GoldBees 26w ROC < -5% → Gold outrunning equity
    Position in equity: MIDCAP or NIFTY
    Position in defensive: GOLD or CASH
    """
    nb   = data["NIFTYBEES"]
    mid  = data["MID150BEES"]
    gold = data["GOLDBEES"]

    df = pd.concat([nb, mid, gold], axis=1, join="inner")
    df.columns = ["nifty", "mid", "gold"]
    df = df.dropna()

    # Indicators — computed on data up to each row (no look-ahead)
    df["nifty_sma"]     = df["nifty"].rolling(SMA_PERIOD).mean()
    ratio_mid           = df["mid"] / df["nifty"]
    df["ratio_mid"]     = ratio_mid
    df["ratio_mid_sma"] = ratio_mid.rolling(SMA_PERIOD).mean()
    df["gold_sma"]      = df["gold"].rolling(SMA_PERIOD).mean()
    eg_ratio            = df["nifty"] / df["gold"]
    df["eg_roc"]        = eg_ratio.pct_change(ROC_PERIOD) * 100

    # Signal generation — at end of each week
    signals = []
    last_signal = "CASH"

    for _, row in df.iterrows():
        if any(pd.isna(row[c]) for c in ["nifty_sma", "ratio_mid_sma", "gold_sma", "eg_roc"]):
            signals.append(None)
            continue

        upper = row["nifty_sma"] * (1 + BUFFER)
        lower = row["nifty_sma"] * (1 - BUFFER)

        if row["nifty"] > upper:
            s1 = "EQUITY"
        elif row["nifty"] < lower:
            s1 = "DEFENSIVE"
        else:
            s1 = "BUFFER"  # hold current

        if s1 == "EQUITY":
            sig = "MID" if row["ratio_mid"] > row["ratio_mid_sma"] else "NIFTY"
        elif s1 == "DEFENSIVE":
            s3 = row["gold"] > row["gold_sma"]
            s4 = row["eg_roc"] < -5.0
            sig = "GOLD" if (s3 and s4) else "CASH"
        else:
            sig = last_signal  # buffer zone → hold

        last_signal = sig
        signals.append(sig)

    df["signal"] = signals
    df = df.dropna(subset=["signal"])

    # Returns — IMPORTANT: signal at week t → return earned in week t+1
    # Shift signals forward by 1 so position[t] drives return[t+1]
    df["nifty_ret"] = df["nifty"].pct_change()
    df["mid_ret"]   = df["mid"].pct_change()
    df["gold_ret"]  = df["gold"].pct_change()

    # active_position[t] = signal[t-1] (what we hold during week t)
    df["active_pos"] = df["signal"].shift(1)
    df = df.dropna(subset=["active_pos"])

    def pick_return(row):
        p = row["active_pos"]
        if p == "MID":   return row["mid_ret"]
        if p == "NIFTY": return row["nifty_ret"]
        if p == "GOLD":  return row["gold_ret"]
        return CASH_WEEKLY  # CASH earns liquid fund rate

    df["strat_ret"] = df.apply(pick_return, axis=1)
    return df


# ── Strategy 2: Kiru's Donchian ─────────────────────────────────────────────────
def run_donchian(data: dict) -> pd.DataFrame:
    """
    Turtle Trading + Dual Momentum on NiftyBees/GoldBees ratio.
    Apply 20-week Donchian Channel to the ratio.
    Ratio breaks above 20w high → hold NiftyBees
    Ratio breaks below 20w low  → hold GoldBees
    Always invested (no cash).
    """
    nb   = data["NIFTYBEES"]
    gold = data["GOLDBEES"]

    df = pd.concat([nb, gold], axis=1, join="inner")
    df.columns = ["nifty", "gold"]
    df = df.dropna()

    df["ratio"]    = df["nifty"] / df["gold"]

    # Donchian using shift(1) on rolling max/min to avoid look-ahead:
    # At end of week t, channel is based on weeks t-20 to t-1
    df["don_high"] = df["ratio"].shift(1).rolling(DONCHIAN_PERIOD).max()
    df["don_low"]  = df["ratio"].shift(1).rolling(DONCHIAN_PERIOD).min()

    signals = []
    last_sig = "NIFTY"

    for _, row in df.iterrows():
        if pd.isna(row["don_high"]) or pd.isna(row["don_low"]):
            signals.append(None)
            continue
        if row["ratio"] > row["don_high"]:
            last_sig = "NIFTY"
        elif row["ratio"] < row["don_low"]:
            last_sig = "GOLD"
        # else: hold last signal
        signals.append(last_sig)

    df["signal"] = signals
    df = df.dropna(subset=["signal"])

    df["nifty_ret"] = df["nifty"].pct_change()
    df["gold_ret"]  = df["gold"].pct_change()

    # Same no-look-ahead rule: signal[t] → return[t+1]
    df["active_pos"] = df["signal"].shift(1)
    df = df.dropna(subset=["active_pos"])

    df["strat_ret"] = np.where(
        df["active_pos"] == "NIFTY",
        df["nifty_ret"],
        df["gold_ret"],
    )
    return df


# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(ret: pd.Series, label: str) -> dict:
    r = ret.dropna()
    n_years = len(r) / 52

    total_ret = (1 + r).prod() - 1
    cagr      = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0

    cum       = (1 + r).cumprod()
    roll_max  = cum.cummax()
    dd        = (cum - roll_max) / roll_max
    max_dd    = dd.min()

    excess    = r - CASH_WEEKLY
    sharpe    = (excess.mean() / excess.std() * np.sqrt(52)) if excess.std() > 0 else 0
    calmar    = cagr / abs(max_dd) if max_dd != 0 else 0
    vol       = r.std() * np.sqrt(52)
    win_rate  = (r > 0).mean()

    return {
        "label":      label,
        "cagr":       cagr,
        "total_ret":  total_ret,
        "max_dd":     max_dd,
        "sharpe":     sharpe,
        "calmar":     calmar,
        "vol":        vol,
        "win_rate":   win_rate,
        "equity":     cum * INITIAL_CAPITAL,
    }

def yearly_table(ret: pd.Series) -> pd.Series:
    r = ret.dropna().copy()
    r.index = pd.to_datetime(r.index)
    return r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)

def fmt_pct(v, decimals=1, plus=False):
    s = f"{v*100:.{decimals}f}%"
    return ("+" + s) if (plus and v > 0) else s

def color_pct(val):
    """Return green/red colored HTML for a % string."""
    try:
        v = float(val.replace("%", "").replace("+", ""))
        clr = "#3CB371" if v >= 0 else "#E74C3C"
        return f'<span style="color:{clr};font-weight:600">{val}</span>'
    except:
        return val


# ── Main App ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:8px 0 4px 0">
  <span style="font-size:24px;font-weight:800;letter-spacing:-0.5px">📈 Backtest Analysis</span><br>
  <span style="font-size:13px;color:#555;font-family:'Space Mono',monospace">
    4-Signal App &nbsp;vs&nbsp; Kiru Donchian &nbsp;|&nbsp; Weekly · Yahoo Finance · No look-ahead bias
  </span>
</div>
""", unsafe_allow_html=True)

with st.spinner("Fetching full historical data from Yahoo Finance…"):
    raw, errors = fetch_all()

if errors:
    with st.expander("⚠️ Fetch warnings"):
        for e in errors: st.warning(e)

missing = [k for k in ["NIFTYBEES", "GOLDBEES"] if k not in raw]
if missing:
    st.error(f"Cannot run backtest — missing: {', '.join(missing)}")
    st.stop()

has_mid = "MID150BEES" in raw

if not has_mid:
    st.warning("MID150BEES data unavailable — 4-Signal strategy will use NIFTY for all equity phases (no midcap allocation).")

with st.spinner("Running backtests…"):
    df4 = run_4signal(raw)
    dfD = run_donchian(raw)

# Align on common period
common_start = max(df4.index[0], dfD.index[0])
common_end   = min(df4.index[-1], dfD.index[-1])
df4 = df4[(df4.index >= common_start) & (df4.index <= common_end)]
dfD = dfD[(dfD.index >= common_start) & (dfD.index <= common_end)]

# Benchmarks
nb_ret   = raw["NIFTYBEES"].pct_change().dropna()
gld_ret  = raw["GOLDBEES"].pct_change().dropna()
nb_ret   = nb_ret[(nb_ret.index >= common_start) & (nb_ret.index <= common_end)]
gld_ret  = gld_ret[(gld_ret.index >= common_start) & (gld_ret.index <= common_end)]

m4   = compute_metrics(df4["strat_ret"], "4-Signal App")
mD   = compute_metrics(dfD["strat_ret"], "Kiru Donchian")
mNB  = compute_metrics(nb_ret,           "Nifty B&H")
mGLD = compute_metrics(gld_ret,          "Gold B&H")

start_yr = pd.to_datetime(common_start).year
end_yr   = pd.to_datetime(common_end).year

st.markdown(f"""
<div style="font-size:12px;color:#555;font-family:'Space Mono',monospace;margin:8px 0 16px 0">
  Backtest period: {start_yr}–{end_yr} &nbsp;·&nbsp; {len(df4)} weeks &nbsp;·&nbsp; 
  Risk-free: {RISK_FREE_RATE*100:.1f}% p.a.
</div>
""", unsafe_allow_html=True)


# ── Summary Table ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📊 Summary Metrics</div>', unsafe_allow_html=True)

all_metrics = [m4, mD, mNB, mGLD]
best_cagr   = max(m["cagr"] for m in all_metrics)
best_dd     = max(m["max_dd"] for m in all_metrics)  # least negative = best
best_sharpe = max(m["sharpe"] for m in all_metrics)
best_calmar = max(m["calmar"] for m in all_metrics)

summary_rows = []
for m in all_metrics:
    cagr_w   = " 🏆" if m["cagr"]   == best_cagr   else ""
    dd_w     = " 🛡" if m["max_dd"] == best_dd     else ""
    sharpe_w = " ⭐" if m["sharpe"] == best_sharpe else ""
    calmar_w = " ⭐" if m["calmar"] == best_calmar else ""
    summary_rows.append({
        "Strategy":       m["label"],
        "CAGR":           fmt_pct(m["cagr"])   + cagr_w,
        "Total Return":   fmt_pct(m["total_ret"], 0),
        "Max Drawdown":   fmt_pct(m["max_dd"])  + dd_w,
        "Sharpe":         f"{m['sharpe']:.2f}"  + sharpe_w,
        "Calmar":         f"{m['calmar']:.2f}"  + calmar_w,
        "Ann. Vol":       fmt_pct(m["vol"]),
        "Weekly Win%":    fmt_pct(m["win_rate"]),
    })

summary_df = pd.DataFrame(summary_rows).set_index("Strategy")
st.dataframe(summary_df, use_container_width=True)
st.caption("🏆 Best CAGR &nbsp;·&nbsp; 🛡 Lowest drawdown &nbsp;·&nbsp; ⭐ Best risk-adjusted return")


# ── CAGR vs Drawdown Visual ────────────────────────────────────────────────────
st.markdown('<div class="section-title">⚖️ Return vs Risk</div>', unsafe_allow_html=True)

cols = st.columns(4)
for i, m in enumerate(all_metrics):
    with cols[i]:
        cagr_clr = "#3CB371" if m["cagr"] > 0.20 else ("#E6A817" if m["cagr"] > 0.10 else "#E74C3C")
        dd_clr   = "#3CB371" if m["max_dd"] > -0.20 else ("#E6A817" if m["max_dd"] > -0.40 else "#E74C3C")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{m['label']}</div>
            <div class="metric-value" style="color:{cagr_clr}">{fmt_pct(m['cagr'])}</div>
            <div style="font-size:12px;color:#555;font-family:'Space Mono',monospace;margin-top:4px">CAGR</div>
            <div style="height:1px;background:#222;margin:10px 0"></div>
            <div class="metric-value" style="color:{dd_clr};font-size:18px">{fmt_pct(m['max_dd'])}</div>
            <div style="font-size:12px;color:#555;font-family:'Space Mono',monospace;margin-top:4px">Max DD</div>
        </div>
        """, unsafe_allow_html=True)


# ── Year-by-Year Returns ───────────────────────────────────────────────────────
st.markdown('<div class="section-title">📅 Year-by-Year Returns</div>', unsafe_allow_html=True)

y4   = yearly_table(df4["strat_ret"])
yD   = yearly_table(dfD["strat_ret"])
yNB  = yearly_table(nb_ret)
yGLD = yearly_table(gld_ret)

yearly_df = pd.DataFrame({
    "4-Signal App":  y4,
    "Kiru Donchian": yD,
    "Nifty B&H":     yNB,
    "Gold B&H":      yGLD,
}).dropna(how="all")

# Format with color
def style_pct(val):
    if pd.isna(val): return ""
    color = "#3CB371" if val >= 0 else "#E74C3C"
    prefix = "+" if val >= 0 else ""
    return f"color: {color}; font-weight: 600"

styled = yearly_df.style.applymap(style_pct).format(lambda x: f"+{x*100:.1f}%" if x >= 0 else f"{x*100:.1f}%", na_rep="-")
st.dataframe(styled, use_container_width=True)


# ── Equity Curve ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📈 Growth of ₹1,00,000</div>', unsafe_allow_html=True)

eq_df = pd.DataFrame({
    "4-Signal App":  m4["equity"],
    "Kiru Donchian": mD["equity"],
    "Nifty B&H":     mNB["equity"],
    "Gold B&H":      mGLD["equity"],
}).dropna()

st.line_chart(eq_df, use_container_width=True)

# Final corpus
st.markdown('<div class="section-title">💰 Final Corpus</div>', unsafe_allow_html=True)
fc_cols = st.columns(4)
for i, m in enumerate(all_metrics):
    final_val = m["equity"].iloc[-1]
    multiple  = final_val / INITIAL_CAPITAL
    with fc_cols[i]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{m['label']}</div>
            <div class="metric-value" style="font-size:20px">₹{final_val:,.0f}</div>
            <div style="font-size:13px;color:#888;font-family:'Space Mono',monospace;margin-top:4px">{multiple:.1f}x</div>
        </div>
        """, unsafe_allow_html=True)


# ── Position Breakdown ─────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📦 Time Spent in Each Asset</div>', unsafe_allow_html=True)

pos4_counts = df4["active_pos"].value_counts(normalize=True).mul(100).round(1)
posD_counts = dfD["active_pos"].value_counts(normalize=True).mul(100).round(1)

pc1, pc2 = st.columns(2)
with pc1:
    st.markdown("**4-Signal App**")
    asset_colors = {"MID": "#FF8C00", "NIFTY": "#1E90FF", "GOLD": "#DAA520", "CASH": "#3CB371"}
    for asset, pct in pos4_counts.items():
        clr = asset_colors.get(asset, "#888")
        bar_w = int(pct)
        st.markdown(f"""
        <div style="margin-bottom:8px">
            <div style="font-size:13px;font-family:'Space Mono',monospace;margin-bottom:3px">
                <span style="color:{clr};font-weight:700">{asset}</span>
                <span style="float:right;color:#888">{pct:.1f}%</span>
            </div>
            <div style="background:#222;border-radius:4px;height:8px;overflow:hidden">
                <div style="background:{clr};width:{bar_w}%;height:100%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

with pc2:
    st.markdown("**Kiru Donchian**")
    for asset, pct in posD_counts.items():
        clr = asset_colors.get(asset, "#888")
        bar_w = int(pct)
        st.markdown(f"""
        <div style="margin-bottom:8px">
            <div style="font-size:13px;font-family:'Space Mono',monospace;margin-bottom:3px">
                <span style="color:{clr};font-weight:700">{asset}</span>
                <span style="float:right;color:#888">{pct:.1f}%</span>
            </div>
            <div style="background:#222;border-radius:4px;height:8px;overflow:hidden">
                <div style="background:{clr};width:{bar_w}%;height:100%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Number of switches
n_switches_4 = (df4["signal"] != df4["signal"].shift(1)).sum()
n_switches_D = (dfD["signal"] != dfD["signal"].shift(1)).sum()

st.markdown(f"""
<div style="font-size:13px;color:#666;font-family:'Space Mono',monospace;margin-top:8px">
  Switches — 4-Signal: {n_switches_4} &nbsp;|&nbsp; Donchian: {n_switches_D}
  &nbsp;&nbsp;(fewer = lower transaction cost)
</div>
""", unsafe_allow_html=True)


# ── Notes ──────────────────────────────────────────────────────────────────────
st.divider()
with st.expander("📋 Methodology Notes"):
    st.markdown("""
    **No look-ahead bias:** Signals computed at close of week N determine position held during week N+1.
    
    **4-Signal App:**  
    - S1 uses NiftyBees vs 26w SMA with ±2% buffer zone (hold current if inside band)  
    - S2 uses MID150BEES/NiftyBees ratio vs its 26w SMA  
    - S3 uses GoldBees vs its 26w SMA  
    - S4 uses 26w ROC of NiftyBees/GoldBees ratio < −5%  
    - Cash earns 6.5% p.a. (liquid fund proxy)

    **Kiru Donchian:**  
    - NiftyBees/GoldBees ratio with 20-week Donchian Channel  
    - Channel uses `shift(1).rolling(20).max/min` — strictly prior 20 weeks  
    - Always invested — no cash allocation  
    - Based on turtle trading + dual momentum described in video

    **Benchmarks:** Weekly close-to-close returns, fully invested, no rebalancing.

    **Caveats:** Past performance ≠ future results. Weekly Yahoo Finance data may have occasional gaps or adjusted price anomalies. No transaction costs or taxes modelled.
    """)

if st.button("🔄 Refresh Data", use_container_width=False):
    st.cache_data.clear()
    st.rerun()

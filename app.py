"""
Regime Signal Dashboard — Donchian Pro
=======================================
Pure Donchian Channel system: NiftyBees · MID150BEES · GoldBees
No moving averages. Breakout-based signals only.

Signals:
  S1 Asymmetric EQ/Gold Donchian  — 13w high → EQUITY, 26w low → GOLD
  S1 Absolute NiftyBees override  — 20w high → force EQUITY
  S1 Conflict → CASH              — ratio defensive + Nifty strong + Gold weak
  S2 MID/Nifty Donchian           — 20w high → MIDCAP, 20w low → LARGECAP

Run:  streamlit run app.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Regime Dashboard",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Constants ──────────────────────────────────────────────────────────────────
TICKERS = {
    "NIFTYBEES":  "NIFTYBEES.NS",
    "MID150BEES": "MID150BEES.NS",
    "GOLDBEES":   "GOLDBEES.NS",
}
D_FAST   = 13   # EQ entry: ratio above 13w high
D_SLOW   = 26   # Gold entry: ratio below 26w low
D_MID    = 20   # Mid/Large selector channel
D_NIFTY  = 20   # Absolute NiftyBees breakout window
FETCH_WK = 55   # Weeks of history to fetch

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background:#0d0d0d; color:#e0e0e0; }

.rec-wrap  { border-radius:20px; padding:28px 20px; text-align:center; margin:16px 0 24px 0; }
.rec-label { font-size:11px; letter-spacing:2.5px; color:#888; text-transform:uppercase;
             margin-bottom:10px; font-family:'Space Mono',monospace; }
.rec-value { font-size:40px; font-weight:800; letter-spacing:-0.5px; line-height:1.1; }

.sig-card  { border-radius:12px; padding:14px 16px; margin-bottom:10px;
             font-size:14px; line-height:1.75; }
.sig-title { font-weight:700; font-size:12px; letter-spacing:0.5px;
             text-transform:uppercase; margin-bottom:5px;
             font-family:'Space Mono',monospace; }
.indent    { margin-left:24px; }

.val-row   { display:flex; gap:8px; margin:14px 0; flex-wrap:wrap; }
.val-box   { flex:1; min-width:110px; background:#141414; border:1px solid #2a2a2a;
             border-radius:12px; padding:12px 10px; text-align:center; }
.val-lbl   { font-size:10px; color:#555; text-transform:uppercase;
             letter-spacing:1px; font-family:'Space Mono',monospace; margin-bottom:3px; }
.val-num   { font-size:18px; font-weight:700; font-family:'Space Mono',monospace; }
.val-sub   { font-size:11px; margin-top:2px; font-family:'Space Mono',monospace; }
.divider   { border-top:1px solid #222; margin:18px 0; }
.footer    { font-size:12px; color:#555; text-align:center;
             font-family:'Space Mono',monospace; margin-top:14px; }
</style>
""", unsafe_allow_html=True)


# ── Data Fetch ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_weekly(tickers: dict, weeks: int):
    end   = datetime.today()
    start = end - timedelta(weeks=weeks)
    out, errors = {}, []
    for name, sym in tickers.items():
        try:
            t  = yf.Ticker(sym)
            df = t.history(period="max", interval="1wk", auto_adjust=True)
            if df.empty:
                df_d = t.history(period="max", interval="1d", auto_adjust=True)
                if df_d.empty:
                    errors.append(f"{name}: no data"); continue
                df = df_d["Close"].resample("W-FRI").last().to_frame("Close")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            s = df["Close"].dropna()
            idx = s.index
            if hasattr(idx, "tz") and idx.tz is not None:
                idx = idx.tz_convert(None)
            s.index = pd.DatetimeIndex(idx).normalize()
            s = s[~s.index.duplicated(keep="last")]
            s = s.resample("W-FRI").last().dropna()
            s = s[s.index >= pd.Timestamp(start)]
            if len(s) >= D_SLOW + 2:
                out[name] = s
            else:
                errors.append(f"{name}: only {len(s)} weeks")
        except Exception as e:
            errors.append(f"{name}: {e}")
    return out, errors


def compute_signals(data: dict):
    """Compute Donchian Pro signals from weekly series."""
    nb   = data.get("NIFTYBEES")
    gold = data.get("GOLDBEES")
    mid  = data.get("MID150BEES")
    if nb is None or gold is None:
        return None

    # Align nifty + gold
    df = pd.concat([nb.rename("nifty"), gold.rename("gold")], axis=1, join="inner").dropna()
    if mid is not None:
        df = df.join(mid.rename("mid"), how="left")
        df["mid_live"] = df["mid"].notna()
        df["mid"] = df["mid"].fillna(df["nifty"])
    else:
        df["mid"] = df["nifty"]
        df["mid_live"] = False

    if len(df) < D_SLOW + 2:
        return None

    # ── Channels (shift(1) = no look-ahead) ───────────────────────────────────
    df["eg_ratio"]    = df["nifty"] / df["gold"]
    eg_sh             = df["eg_ratio"].shift(1)
    df["eg_13w_high"] = eg_sh.rolling(D_FAST).max()
    df["eg_26w_low"]  = eg_sh.rolling(D_SLOW).min()

    nifty_sh             = df["nifty"].shift(1)
    df["nifty_20w_high"] = nifty_sh.rolling(D_NIFTY).max()
    df["nifty_13w_high"] = nifty_sh.rolling(D_FAST).max()
    df["nifty_20w_low"]  = nifty_sh.rolling(D_NIFTY).min()

    gold_sh              = df["gold"].shift(1)
    df["gold_20w_high"]  = gold_sh.rolling(D_NIFTY).max()
    df["gold_20w_low"]   = gold_sh.rolling(D_NIFTY).min()

    df["mid_ratio"]    = df["mid"] / df["nifty"]
    mid_sh             = df["mid_ratio"].shift(1)
    df["mid_don_high"] = mid_sh.rolling(D_MID).max()
    df["mid_don_low"]  = mid_sh.rolling(D_MID).min()

    df = df.dropna()
    if df.empty:
        return None

    row = df.iloc[-1]

    # ── Current values ─────────────────────────────────────────────────────────
    cur_nifty   = float(row["nifty"])
    cur_gold    = float(row["gold"])
    cur_mid     = float(row["mid"])
    cur_eg      = float(row["eg_ratio"])
    cur_mid_r   = float(row["mid_ratio"])
    mid_live    = bool(row["mid_live"])

    eg_13h  = float(row["eg_13w_high"])
    eg_26l  = float(row["eg_26w_low"])
    n_20h   = float(row["nifty_20w_high"])
    n_13h   = float(row["nifty_13w_high"])
    n_20l   = float(row["nifty_20w_low"])
    g_20h   = float(row["gold_20w_high"])
    g_20l   = float(row["gold_20w_low"])
    mid_h   = float(row["mid_don_high"])
    mid_l   = float(row["mid_don_low"])

    # ── S1 regime ──────────────────────────────────────────────────────────────
    gold_trending   = cur_gold  > g_20h
    ratio_defensive = cur_eg    < eg_26l
    nifty_strong    = cur_nifty > n_13h
    ratio_bullish   = cur_eg    > eg_13h
    nifty_breakout  = cur_nifty > n_20h
    dual_weak       = (cur_nifty < n_20l) and (cur_gold < g_20l)

    if dual_weak:
        s1 = "CASH"
        s1_reason = "Both equity & gold below 20w lows — dual weakness"
    elif ratio_bullish or nifty_breakout:
        s1 = "EQUITY"
        trigger = []
        if ratio_bullish:   trigger.append(f"EQ/Gold ratio {cur_eg:.3f} > 13w high {eg_13h:.3f}")
        if nifty_breakout:  trigger.append(f"NiftyBees ₹{cur_nifty:.0f} > 20w high ₹{n_20h:.0f}")
        s1_reason = " & ".join(trigger)
    elif ratio_defensive and nifty_strong and gold_trending:
        s1 = "GOLD"
        s1_reason = f"Ratio defensive + NiftyBees strong but GoldBees trending ₹{cur_gold:.0f} > ₹{g_20h:.0f}"
    elif ratio_defensive and nifty_strong and not gold_trending:
        s1 = "CASH"
        s1_reason = "Conflict — ratio says gold but NiftyBees still strong, gold not trending"
    elif ratio_defensive:
        s1 = "GOLD"
        s1_reason = f"EQ/Gold ratio {cur_eg:.3f} < 26w low {eg_26l:.3f}"
    else:
        s1 = "HOLD"
        s1_reason = f"Ratio {cur_eg:.3f} between 26w low {eg_26l:.3f} and 13w high {eg_13h:.3f} — no breakout"

    # ── S2 mid/large ───────────────────────────────────────────────────────────
    if s1 == "EQUITY":
        if not mid_live:
            s2 = "NIFTY"
            s2_reason = "MID150BEES data unavailable — defaulting to largecap"
        elif cur_mid_r > mid_h:
            s2 = "MIDCAP"
            s2_reason = f"MID/Nifty ratio {cur_mid_r:.4f} > 20w high {mid_h:.4f}"
        elif cur_mid_r < mid_l:
            s2 = "NIFTY"
            s2_reason = f"MID/Nifty ratio {cur_mid_r:.4f} < 20w low {mid_l:.4f}"
        else:
            s2 = "HOLD"
            s2_reason = f"Ratio {cur_mid_r:.4f} between 20w low {mid_l:.4f} and high {mid_h:.4f} — hold"
    else:
        s2 = None
        s2_reason = None

    # ── Final recommendation ───────────────────────────────────────────────────
    if s1 == "EQUITY":
        rec = "HOLD (equity)" if s2 == "HOLD" else s2
    elif s1 == "HOLD":
        rec = "HOLD — No Change"
    else:
        rec = s1   # GOLD or CASH

    return {
        "rec": rec, "s1": s1, "s2": s2,
        "s1_reason": s1_reason, "s2_reason": s2_reason,
        "cur_nifty": cur_nifty, "cur_gold": cur_gold, "cur_mid": cur_mid,
        "cur_eg": cur_eg, "cur_mid_r": cur_mid_r,
        "eg_13h": eg_13h, "eg_26l": eg_26l,
        "n_20h": n_20h, "n_13h": n_13h, "n_20l": n_20l,
        "g_20h": g_20h, "g_20l": g_20l,
        "mid_h": mid_h, "mid_l": mid_l,
        "mid_live": mid_live,
        "ratio_bullish": ratio_bullish, "nifty_breakout": nifty_breakout,
        "ratio_defensive": ratio_defensive, "nifty_strong": nifty_strong,
        "gold_trending": gold_trending, "dual_weak": dual_weak,
        "as_of": df.index[-1],
    }


# ── App ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:8px 0 4px 0">
  <span style="font-size:26px;font-weight:800;letter-spacing:-0.5px">📊 Regime Dashboard</span><br>
  <span style="font-size:13px;color:#555;font-family:'Space Mono',monospace">
    Donchian Pro &nbsp;|&nbsp; NiftyBees · MID150BEES · GoldBees &nbsp;|&nbsp; Weekly
  </span>
</div>
""", unsafe_allow_html=True)

# ── Fetch ──────────────────────────────────────────────────────────────────────
with st.spinner("Fetching live weekly data…"):
    raw, fetch_errors = fetch_weekly(tuple(TICKERS.items()), FETCH_WK)

fetch_ok = all(k in raw for k in ["NIFTYBEES", "GOLDBEES"])

if fetch_errors:
    with st.expander("⚠️ Fetch warnings"):
        for e in fetch_errors: st.warning(e)

# ── Compute signals ────────────────────────────────────────────────────────────
sig = compute_signals(raw) if fetch_ok else None

# ── Manual override expander ───────────────────────────────────────────────────
with st.expander(
    "📥 Channel Values — " + ("Auto-filled ✅" if sig else "Manual entry required ⚠️"),
    expanded=not sig
):
    st.caption("All values auto-computed from last 26–55 weeks of closes.")
    c1, c2 = st.columns(2)
    with c1:
        nifty_price  = st.number_input("NiftyBees Price",        value=float(sig["cur_nifty"] if sig else 0), format="%.2f", step=1.0)
        eg_ratio     = st.number_input("EQ/Gold Ratio (current)",value=float(sig["cur_eg"]    if sig else 0), format="%.4f", step=0.001)
        eg_13h       = st.number_input("EQ/Gold 13w High",       value=float(sig["eg_13h"]    if sig else 0), format="%.4f", step=0.001)
        eg_26l       = st.number_input("EQ/Gold 26w Low",        value=float(sig["eg_26l"]    if sig else 0), format="%.4f", step=0.001)
        nifty_20h    = st.number_input("NiftyBees 20w High",     value=float(sig["n_20h"]     if sig else 0), format="%.2f", step=1.0)
    with c2:
        nifty_13h    = st.number_input("NiftyBees 13w High",     value=float(sig["n_13h"]     if sig else 0), format="%.2f", step=1.0)
        gold_price   = st.number_input("GoldBees Price",         value=float(sig["cur_gold"]  if sig else 0), format="%.2f", step=1.0)
        gold_20h     = st.number_input("GoldBees 20w High",      value=float(sig["g_20h"]     if sig else 0), format="%.2f", step=1.0)
        mid_ratio    = st.number_input("MID/Nifty Ratio",        value=float(sig["cur_mid_r"] if sig else 0), format="%.4f", step=0.001)
        mid_don_h    = st.number_input("MID/Nifty 20w High",     value=float(sig["mid_h"]     if sig else 0), format="%.4f", step=0.001)
        mid_don_l    = st.number_input("MID/Nifty 20w Low",      value=float(sig["mid_l"]     if sig else 0), format="%.4f", step=0.001)

# Use manual values if auto-compute failed or was overridden
if not sig:
    # Rebuild sig from manual inputs
    if nifty_price > 0 and eg_ratio > 0 and eg_13h > 0:
        gold_trending   = gold_price  > gold_20h
        ratio_defensive = eg_ratio    < eg_26l
        nifty_strong    = nifty_price > nifty_13h
        ratio_bullish   = eg_ratio    > eg_13h
        nifty_brkout    = nifty_price > nifty_20h

        if ratio_bullish or nifty_brkout:
            s1, s1r = "EQUITY", "Manual: equity signal active"
        elif ratio_defensive and nifty_strong and gold_trending:
            s1, s1r = "GOLD", "Manual: gold trending despite equity strength"
        elif ratio_defensive and nifty_strong:
            s1, s1r = "CASH", "Manual: conflict — cash"
        elif ratio_defensive:
            s1, s1r = "GOLD", "Manual: ratio defensive"
        else:
            s1, s1r = "HOLD", "Manual: no breakout"

        if s1 == "EQUITY":
            if mid_ratio > mid_don_h: s2, s2r = "MIDCAP", "Manual: mid breakout"
            elif mid_ratio < mid_don_l: s2, s2r = "NIFTY", "Manual: large breakout"
            else: s2, s2r = "HOLD", "Manual: mid channel hold"
        else:
            s2, s2r = None, None

        rec = ("HOLD (equity)" if s2 == "HOLD" else s2) if s1 == "EQUITY" \
              else ("HOLD — No Change" if s1 == "HOLD" else s1)

        sig = {
            "rec": rec, "s1": s1, "s2": s2,
            "s1_reason": s1r, "s2_reason": s2r,
            "cur_nifty": nifty_price, "cur_gold": gold_price,
            "cur_eg": eg_ratio, "cur_mid_r": mid_ratio,
            "eg_13h": eg_13h, "eg_26l": eg_26l,
            "n_20h": nifty_20h, "n_13h": nifty_13h,
            "g_20h": gold_20h, "mid_h": mid_don_h, "mid_l": mid_don_l,
            "ratio_bullish": ratio_bullish, "nifty_breakout": nifty_brkout,
            "ratio_defensive": ratio_defensive, "nifty_strong": nifty_strong,
            "gold_trending": gold_trending, "dual_weak": False,
            "mid_live": mid_ratio > 0,
            "as_of": datetime.now(),
        }

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Display ────────────────────────────────────────────────────────────────────
if sig:
    rec = sig["rec"]
    s1  = sig["s1"]
    s2  = sig["s2"]

    # Colour map
    REC_STYLE = {
        "MIDCAP":         ("#FF8C00", "🟠"),
        "NIFTY":          ("#1E90FF", "🔵"),
        "GOLD":           ("#DAA520", "🥇"),
        "CASH":           ("#3CB371", "💵"),
        "HOLD — No Change": ("#9B59B6", "⏸"),
        "HOLD (equity)":  ("#9B59B6", "⏸"),
    }
    clr, ico = REC_STYLE.get(rec, ("#888", "⚪"))

    # ── Recommendation card ──────────────────────────────────────────────────
    st.markdown(f"""
    <div class="rec-wrap" style="background:{clr}18;border:2.5px solid {clr}55">
        <div class="rec-label">Current Recommendation</div>
        <div class="rec-value" style="color:{clr}">{ico}&nbsp; {rec}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Signal flow ──────────────────────────────────────────────────────────
    st.markdown("#### Signal Breakdown")

    S1_META = {
        "EQUITY": ("#2E7D32", "🟢", "EQUITY REGIME"),
        "GOLD":   ("#DAA520", "🥇", "GOLD / DEFENSIVE"),
        "CASH":   ("#3CB371", "💵", "CASH / LIQUID"),
        "HOLD":   ("#9B59B6", "⏸", "HOLD — No breakout"),
    }
    s1_clr, s1_ico, s1_lbl = S1_META.get(s1, ("#888", "⚪", s1))

    # S1 indicators as checkboxes
    def tick(v): return "✅" if v else "❌"
    indicators = [
        (tick(sig["ratio_bullish"]),  f"EQ/Gold ratio <b>{sig['cur_eg']:.3f}</b> vs 13w high <b>{sig['eg_13h']:.3f}</b>"),
        (tick(sig["nifty_breakout"]), f"NiftyBees <b>₹{sig['cur_nifty']:.0f}</b> vs 20w high <b>₹{sig['n_20h']:.0f}</b>"),
        (tick(sig["ratio_defensive"]),f"Ratio below 26w low <b>{sig['eg_26l']:.3f}</b> (defensive)"),
        (tick(sig["nifty_strong"]),   f"NiftyBees above 13w high <b>₹{sig['n_13h']:.0f}</b>"),
        (tick(sig["gold_trending"]),  f"GoldBees <b>₹{sig['cur_gold']:.0f}</b> above 20w high <b>₹{sig['g_20h']:.0f}</b>"),
    ]
    ind_html = "<br>".join(f"{t} &nbsp;{d}" for t, d in indicators)

    st.markdown(f"""
    <div class="sig-card" style="border:1.5px solid {s1_clr}44;background:{s1_clr}0d">
        <div class="sig-title">{s1_ico} &nbsp;S1 — Regime Gate (Donchian)</div>
        {ind_html}
        <div style="margin-top:8px;font-weight:700;color:{s1_clr}">→ {s1_lbl}</div>
        <div style="font-size:12px;color:#666;margin-top:4px">{sig['s1_reason']}</div>
    </div>
    """, unsafe_allow_html=True)

    # S2
    if s1 == "EQUITY" and s2 is not None:
        s2_clr = "#FF8C00" if s2 == "MIDCAP" else ("#1E90FF" if s2 == "NIFTY" else "#9B59B6")
        s2_ico = "🟠" if s2 == "MIDCAP" else ("🔵" if s2 == "NIFTY" else "⏸")
        mid_pct = ((sig["cur_mid_r"] - sig["mid_h"]) / sig["mid_h"] * 100) if sig["mid_h"] else 0
        st.markdown(f"""
        <div class="sig-card indent" style="border:1.5px solid {s2_clr}44;background:{s2_clr}0d">
            <div class="sig-title">{s2_ico} &nbsp;S2 — Mid vs Large (Donchian)</div>
            MID/Nifty ratio &nbsp;<b>{sig['cur_mid_r']:.4f}</b><br>
            {'✅' if sig['cur_mid_r'] > sig['mid_h'] else '❌'} 20w High <b>{sig['mid_h']:.4f}</b>
            &nbsp;|&nbsp;
            {'✅' if sig['cur_mid_r'] < sig['mid_l'] else '❌'} 20w Low <b>{sig['mid_l']:.4f}</b>
            <div style="margin-top:8px;font-weight:700;color:{s2_clr}">→ {s2}</div>
            <div style="font-size:12px;color:#666;margin-top:4px">{sig['s2_reason']}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Key values ───────────────────────────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("#### Key Values")

    def dc(v): return "#3CB371" if v >= 0 else "#E74C3C"
    def da(v): return "▲" if v >= 0 else "▼"

    eg_pct  = (sig["cur_eg"] - sig["eg_13h"]) / sig["eg_13h"] * 100 if sig["eg_13h"] else 0
    nif_pct = (sig["cur_nifty"] - sig["n_20h"]) / sig["n_20h"] * 100 if sig["n_20h"] else 0
    gld_pct = (sig["cur_gold"] - sig["g_20h"]) / sig["g_20h"] * 100 if sig["g_20h"] else 0
    mid_pct2= (sig["cur_mid_r"] - sig["mid_h"]) / sig["mid_h"] * 100 if sig["mid_h"] else 0

    st.markdown(f"""
    <div class="val-row">
        <div class="val-box">
            <div class="val-lbl">NiftyBees</div>
            <div class="val-num">₹{sig['cur_nifty']:,.0f}</div>
            <div class="val-sub" style="color:{dc(nif_pct)}">{da(nif_pct)} {abs(nif_pct):.1f}% vs 20w high</div>
        </div>
        <div class="val-box">
            <div class="val-lbl">EQ/Gold Ratio</div>
            <div class="val-num">{sig['cur_eg']:.3f}</div>
            <div class="val-sub" style="color:{dc(eg_pct)}">{da(eg_pct)} {abs(eg_pct):.1f}% vs 13w high</div>
        </div>
        <div class="val-box">
            <div class="val-lbl">GoldBees</div>
            <div class="val-num">₹{sig['cur_gold']:,.0f}</div>
            <div class="val-sub" style="color:{dc(gld_pct)}">{da(gld_pct)} {abs(gld_pct):.1f}% vs 20w high</div>
        </div>
        <div class="val-box">
            <div class="val-lbl">MID/Nifty</div>
            <div class="val-num">{sig['cur_mid_r']:.3f}</div>
            <div class="val-sub" style="color:{dc(mid_pct2)}">{da(mid_pct2)} {abs(mid_pct2):.1f}% vs 20w high</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Channel reference
    st.markdown(f"""
    <div style="font-size:12px;color:#444;font-family:'Space Mono',monospace;
                background:#111;border-radius:8px;padding:10px 14px;margin-top:8px">
      EQ/Gold &nbsp;13w⬆ {sig['eg_13h']:.3f} &nbsp;|&nbsp; 26w⬇ {sig['eg_26l']:.3f}
      &nbsp;&nbsp;·&nbsp;&nbsp;
      NiftyBees &nbsp;20w⬆ ₹{sig['n_20h']:.0f} &nbsp;13w⬆ ₹{sig['n_13h']:.0f}
      &nbsp;&nbsp;·&nbsp;&nbsp;
      MID/Nifty &nbsp;20w⬆ {sig['mid_h']:.4f} &nbsp;20w⬇ {sig['mid_l']:.4f}
    </div>
    """, unsafe_allow_html=True)

    # Footer
    as_of = sig["as_of"]
    as_of_str = as_of.strftime("%d %b %Y") if hasattr(as_of, "strftime") else str(as_of)
    src = "✅ Live · Yahoo Finance" if fetch_ok else "⚠️ Manual / Partial"
    st.markdown(f'<div class="footer">{src} &nbsp;·&nbsp; As of week ending {as_of_str}</div>',
                unsafe_allow_html=True)

    col_r, _ = st.columns([1, 3])
    with col_r:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

else:
    st.info("Enter channel values above to compute signals.")

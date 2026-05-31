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

# ── Market Selection (sidebar) ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    MARKET = st.radio(
        "Market",
        ["🇮🇳 India (NSE)", "🇺🇸 US (NYSE)"],
        index=0,
        help="Same strategy rules applied to a different market — out-of-sample validation",
    )
IS_US = "US" in MARKET

# ── Constants (India defaults) ────────────────────────────────────────────────
TICKERS = {
    "NIFTYBEES":  "NIFTYBEES.NS",
    "MID150BEES": "MID150BEES.NS",
    "GOLDBEES":   "GOLDBEES.NS",
    "BONDBEES":   "GSEC10YBEES.NS",
}
SMA_PERIOD        = 26
SMA_FAST          = 13
BUFFER            = 0.02
ROC_PERIOD        = 26
DONCHIAN_PERIOD   = 20
DONCHIAN_FAST     = 13
DONCHIAN_SLOW     = 26
RISK_FREE_RATE    = 0.065
BOND_RATE         = 0.08
CASH_WEEKLY       = RISK_FREE_RATE / 52
BOND_WEEKLY_SYNTH = BOND_RATE / 52
START_DATE        = "2010-01-01"
INITIAL_CAPITAL   = 100_000
CURRENCY          = "₹"

# US market overrides — key names identical so all strategy functions unchanged
if IS_US:
    TICKERS = {
        "NIFTYBEES":  "SPY",   # S&P 500 ETF  → largecap equity
        "MID150BEES": "MDY",   # S&P MidCap 400 ETF → midcap
        "GOLDBEES":   "GLD",   # SPDR Gold Shares → gold
        "BONDBEES":   "TLT",   # iShares 20Y Treasury → bond
    }
    START_DATE        = "2005-01-01"   # GLD launched Nov 2004
    RISK_FREE_RATE    = 0.03
    BOND_RATE         = 0.04
    CASH_WEEKLY       = RISK_FREE_RATE / 52
    BOND_WEEKLY_SYNTH = BOND_RATE / 52
    INITIAL_CAPITAL   = 10_000
    CURRENCY          = "$"


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
# ── Midcap Index Splice ────────────────────────────────────────────────────────
MID_INDEX_TICKERS = [
    "^NIFMDCP150",    # Nifty Midcap 150 (preferred)
    "^NSMIDCP150",    # alternate symbol
    "NIFTYMIDCAP150.NS",
    "^NSMIDCP100",    # Nifty Midcap 100 (fallback)
    "^NIFMDCP100",
]

def _fetch_single_weekly(ticker_sym: str):
    """Fetch one ticker, return tz-naive Friday-resampled weekly Close or None."""
    try:
        t  = yf.Ticker(ticker_sym)
        df = t.history(period="max", interval="1wk", auto_adjust=True)
        if df.empty:
            df_d = t.history(period="max", interval="1d", auto_adjust=True)
            if df_d.empty:
                return None
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
        return s if len(s) >= 10 else None
    except Exception:
        return None

def build_spliced_mid(etf_series: pd.Series, toast_fn=None) -> tuple[pd.Series, str]:
    """
    Extend MID150BEES back in time using a Nifty Midcap index.
    Returns (spliced_series, note_string).
    Scale factor = median(ETF / index) over overlap → synthetic pre-ETF prices.
    """
    idx_data, idx_name = None, None
    for sym in MID_INDEX_TICKERS:
        data = _fetch_single_weekly(sym)
        if data is not None and len(data) > 52:
            idx_data, idx_name = data, sym
            break

    if idx_data is None:
        return etf_series, "No index proxy found — MID150BEES used as-is"

    # Overlap period (need ≥ 8 weeks for reliable scale factor)
    overlap = etf_series.index.intersection(idx_data.index)
    if len(overlap) < 8:
        return etf_series, f"{idx_name} found but overlap too short — MID150BEES used as-is"

    # Scale factor: median ratio ETF / index over overlap
    scale   = (etf_series[overlap] / idx_data[overlap]).median()

    # Synthetic prices for dates before ETF launch
    etf_start   = etf_series.index[0]
    pre_idx     = idx_data[idx_data.index < etf_start] * scale
    spliced     = pd.concat([pre_idx, etf_series]).sort_index()
    spliced     = spliced[~spliced.index.duplicated(keep="last")]

    n_synthetic = len(pre_idx)
    note = (f"Extended using **{idx_name}** — scale factor {scale:.4f} "
            f"({n_synthetic} synthetic weeks prepended before {etf_start.date()})")
    return spliced, note


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_all():
    results, errors = {}, []
    for name, ticker_sym in TICKERS.items():
        try:
            # ticker.history() is more reliable than yf.download() for NSE tickers
            t  = yf.Ticker(ticker_sym)
            df = t.history(period="max", interval="1wk", auto_adjust=True)

            if df.empty:
                # fallback: try daily then resample to weekly
                df_daily = t.history(period="max", interval="1d", auto_adjust=True)
                if df_daily.empty:
                    errors.append(f"{name} ({ticker_sym}): no data returned — ticker may be delisted or unavailable")
                    continue
                df = df_daily["Close"].resample("W-FRI").last().to_frame(name="Close")

            # Normalise columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            closes = df["Close"].dropna()

            # Strip timezone and normalise to midnight so all tickers align
            idx = closes.index
            if hasattr(idx, "tz") and idx.tz is not None:
                idx = idx.tz_convert(None)   # remove tz properly
            closes.index = pd.DatetimeIndex(idx).normalize()  # 00:00:00 for all

            # Remove any duplicate dates (safety)
            closes = closes[~closes.index.duplicated(keep="last")]

            # Resample to common weekly anchor (Friday) so all tickers align
            closes = closes.resample("W-FRI").last()

            # Trim to START_DATE
            closes = closes[closes.index >= pd.Timestamp(START_DATE)]
            closes = closes.dropna()

            # Base tickers need full history; MID150BEES accepted even if shorter
            min_len = SMA_PERIOD + 10 if name in ("NIFTYBEES", "GOLDBEES") else 4
            if len(closes) < min_len:
                errors.append(f"{name}: only {len(closes)} weeks — insufficient")
            else:
                results[name] = closes

        except Exception as exc:
            errors.append(f"{name}: {exc}")

    # ── Splice midcap back in time (India only — US MDY has 20yr history) ────────
    if not IS_US and "MID150BEES" in results:
        spliced, splice_note = build_spliced_mid(results["MID150BEES"])
        results["MID150BEES"] = spliced
        results["_splice_note"] = splice_note
    elif IS_US:
        results["_splice_note"] = None   # no splice needed for US
    else:
        results["_splice_note"] = "MID150BEES not fetched — using NIFTY for equity regime"

    return results, errors


# ── Strategy 1: 4-Signal App ───────────────────────────────────────────────────
def run_4signal(data: dict) -> pd.DataFrame:
    """
    Fully vectorised. No iterrows.
    S1: NiftyBees vs 26w SMA ±2% buffer → Equity / Defensive / Buffer(hold)
    S2: MID150/Nifty ratio vs 26w SMA   → MID / NIFTY
    S3: GoldBees vs 26w SMA             → gold uptrend
    S4: EQ/Gold 26w ROC < -5%          → gold outrunning equity
    Buffer zone → forward-fill previous signal.
    """
    nb   = data["NIFTYBEES"]
    gold = data["GOLDBEES"]

    # Base: inner join nifty+gold (full history from 2010)
    # Mid: left-joined so pre-2019 rows keep NaN → fallback to NIFTY
    df = pd.concat([nb.rename("nifty"), gold.rename("gold")], axis=1, join="inner").dropna()
    if "MID150BEES" in data:
        df = df.join(data["MID150BEES"].rename("mid"), how="left")
    else:
        df["mid"] = np.nan
    df["mid_live"] = df["mid"].notna()          # True only where ETF exists
    df["mid"] = df["mid"].fillna(df["nifty"])   # pre-2019: mid = nifty (no MID signal)

    # ── Indicators ──────────────────────────────────────────────────────────
    df["nifty_sma"]     = df["nifty"].rolling(SMA_PERIOD, min_periods=SMA_PERIOD).mean()
    df["ratio_mid"]     = df["mid"] / df["nifty"]
    df["ratio_mid_sma"] = df["ratio_mid"].rolling(SMA_PERIOD, min_periods=SMA_PERIOD).mean()
    df["gold_sma"]      = df["gold"].rolling(SMA_PERIOD, min_periods=SMA_PERIOD).mean()
    eg                  = df["nifty"] / df["gold"]
    df["eg_roc"]        = eg.pct_change(ROC_PERIOD)        # fraction, not %

    # Drop warmup rows (first SMA_PERIOD + ROC_PERIOD weeks)
    df = df.dropna(subset=["nifty_sma", "ratio_mid_sma", "gold_sma", "eg_roc"]).copy()
    if df.empty:
        return df

    # ── Signals (vectorised) ─────────────────────────────────────────────────
    upper = df["nifty_sma"] * (1.0 + BUFFER)
    lower = df["nifty_sma"] * (1.0 - BUFFER)

    eq_regime  = df["nifty"] > upper
    def_regime = df["nifty"] < lower
    buf_regime = ~eq_regime & ~def_regime

    s2_mid = df["ratio_mid"] > df["ratio_mid_sma"]
    s3     = df["gold"] > df["gold_sma"]
    s4     = df["eg_roc"] < -0.05   # -5% as fraction

    raw = pd.Series(np.nan, index=df.index, dtype=object)
    raw[eq_regime  &  s2_mid & df["mid_live"]]  = "MID"   # MID only when ETF exists
    raw[eq_regime  & (~s2_mid | ~df["mid_live"])] = "NIFTY" # else always largecap
    raw[def_regime &  s3 & s4]                  = "GOLD"
    raw[def_regime & ~(s3 & s4)]                = "CASH"
    # buf_regime stays NaN → forward-filled below

    df["signal"] = raw.ffill().fillna("CASH")

    # ── Returns ──────────────────────────────────────────────────────────────
    df["nifty_ret"] = df["nifty"].pct_change()
    df["mid_ret"]   = df["mid"].pct_change()
    df["gold_ret"]  = df["gold"].pct_change()

    # Signal at t → return earned at t+1
    df["active_pos"] = df["signal"].shift(1)
    df = df.iloc[1:].copy()   # drop first row (no prior signal)

    df["strat_ret"] = np.select(
        [df["active_pos"] == "MID",
         df["active_pos"] == "NIFTY",
         df["active_pos"] == "GOLD"],
        [df["mid_ret"], df["nifty_ret"], df["gold_ret"]],
        default=CASH_WEEKLY,
    )
    return df


# ── Strategy 2: Kiru's Donchian ─────────────────────────────────────────────────
def run_donchian(data: dict) -> pd.DataFrame:
    """
    Fully vectorised Donchian on NiftyBees/GoldBees ratio.
    20-week channel uses shift(1) to avoid look-ahead.
    Breakout above → NIFTY. Breakdown below → GOLD.
    Always invested (no cash). Forward-fill between breakouts.
    """
    nb   = data["NIFTYBEES"]
    gold = data["GOLDBEES"]

    df = pd.concat([nb.rename("nifty"), gold.rename("gold")], axis=1, join="inner").dropna()

    df["ratio"]    = df["nifty"] / df["gold"]

    # Channel based on prior DONCHIAN_PERIOD bars (shift avoids look-ahead)
    shifted        = df["ratio"].shift(1)
    df["don_high"] = shifted.rolling(DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max()
    df["don_low"]  = shifted.rolling(DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min()

    df = df.dropna(subset=["don_high", "don_low"]).copy()
    if df.empty:
        return df

    # Raw signal at breakout; NaN elsewhere → forward-fill → start default NIFTY
    raw = pd.Series(np.nan, index=df.index, dtype=object)
    raw[df["ratio"] > df["don_high"]] = "NIFTY"
    raw[df["ratio"] < df["don_low"]]  = "GOLD"
    df["signal"] = raw.ffill().fillna("NIFTY")

    df["nifty_ret"] = df["nifty"].pct_change()
    df["gold_ret"]  = df["gold"].pct_change()

    df["active_pos"] = df["signal"].shift(1)
    df = df.iloc[1:].copy()

    df["strat_ret"] = np.where(
        df["active_pos"] == "NIFTY",
        df["nifty_ret"],
        df["gold_ret"],
    )
    return df


# ── Strategy 3: Enhanced 5-Signal ──────────────────────────────────────────────
def run_enhanced(data: dict) -> pd.DataFrame:
    """
    Enhanced 5-Signal Strategy — three upgrades over the 4-Signal App:

    S1 (enhanced): Dual SMA confirmation — both 13w AND 26w must agree to
                   switch regime. Eliminates premature entries/exits in
                   range-bound markets like 2022.

    S2 (enhanced): Midcap quality filter — ratio > 26w SMA AND 13w ROC > 0.
                   Only allocates to midcap when momentum is genuinely
                   accelerating, not just marginally above the SMA.

    S3 / S4:       Unchanged from 4-Signal App.

    S5 (new):      Gilt/bond rotation — when defensive and gold fails,
                   check if a gilt ETF (GSEC10YBEES) is above its 13w SMA.
                   If yes → GILT. If no → CASH.
                   Falls back to synthetic 8% p.a. if ETF unavailable.
    """
    nb   = data["NIFTYBEES"]
    gold = data["GOLDBEES"]
    has_bond = "BONDBEES" in data
    has_mid  = "MID150BEES" in data

    # Base: inner join nifty + gold
    df = pd.concat([nb.rename("nifty"), gold.rename("gold")], axis=1, join="inner").dropna()

    # Left-join mid and bond (shorter histories)
    if has_mid:
        df = df.join(data["MID150BEES"].rename("mid"), how="left")
    else:
        df["mid"] = np.nan
    df["mid_live"] = df["mid"].notna()
    df["mid"] = df["mid"].fillna(df["nifty"])

    if has_bond:
        df = df.join(data["BONDBEES"].rename("bond"), how="left")
    else:
        df["bond"] = np.nan
    df["bond_live"] = df["bond"].notna()

    # ── Indicators ──────────────────────────────────────────────────────────
    # S1 enhanced: dual SMA
    df["nifty_sma26"] = df["nifty"].rolling(SMA_PERIOD, min_periods=SMA_PERIOD).mean()
    df["nifty_sma13"] = df["nifty"].rolling(SMA_FAST,   min_periods=SMA_FAST).mean()

    # S2 enhanced: ratio + momentum quality
    df["ratio_mid"]     = df["mid"] / df["nifty"]
    df["ratio_mid_sma"] = df["ratio_mid"].rolling(SMA_PERIOD, min_periods=SMA_PERIOD).mean()
    df["ratio_roc13"]   = df["ratio_mid"].pct_change(SMA_FAST)   # 13w momentum

    # S3 / S4 unchanged
    df["gold_sma"] = df["gold"].rolling(SMA_PERIOD, min_periods=SMA_PERIOD).mean()
    eg             = df["nifty"] / df["gold"]
    df["eg_roc"]   = eg.pct_change(ROC_PERIOD)

    # S5: gilt ETF vs 13w SMA
    df["bond_sma13"] = df["bond"].rolling(SMA_FAST, min_periods=SMA_FAST).mean()

    # Drop warmup
    df = df.dropna(subset=[
        "nifty_sma26", "nifty_sma13",
        "ratio_mid_sma", "ratio_roc13",
        "gold_sma", "eg_roc",
    ]).copy()
    if df.empty:
        return df

    # ── Signal generation (vectorised) ──────────────────────────────────────
    upper = df["nifty_sma26"] * (1.0 + BUFFER)
    lower = df["nifty_sma26"] * (1.0 - BUFFER)

    # S1 enhanced: slow AND fast must agree
    eq_regime  = (df["nifty"] > upper) & (df["nifty"] > df["nifty_sma13"])
    def_regime = (df["nifty"] < lower) & (df["nifty"] < df["nifty_sma13"])

    # S2 enhanced: ratio above SMA AND momentum positive
    s2_mid = (df["ratio_mid"] > df["ratio_mid_sma"]) & (df["ratio_roc13"] > 0) & df["mid_live"]

    # S3 / S4 unchanged
    s3 = df["gold"] > df["gold_sma"]
    s4 = df["eg_roc"] < -0.05

    # S5: gilt trending — use ETF if live, else always prefer gilt over cash
    s5_gilt = df["bond_live"] & (df["bond"] > df["bond_sma13"])
    s5_gilt = s5_gilt | ~df["bond_live"]   # synthetic: always choose gilt over cash

    raw = pd.Series(np.nan, index=df.index, dtype=object)
    raw[eq_regime  &  s2_mid]                     = "MID"
    raw[eq_regime  & ~s2_mid]                     = "NIFTY"
    raw[def_regime &  s3 & s4]                    = "GOLD"
    raw[def_regime & ~(s3 & s4) &  s5_gilt]       = "GILT"
    raw[def_regime & ~(s3 & s4) & ~s5_gilt]       = "CASH"

    df["signal"] = raw.ffill().fillna("CASH")

    # ── Returns ──────────────────────────────────────────────────────────────
    df["nifty_ret"] = df["nifty"].pct_change()
    df["mid_ret"]   = df["mid"].pct_change()
    df["gold_ret"]  = df["gold"].pct_change()
    df["bond_ret"]  = df["bond"].pct_change() if has_bond else BOND_WEEKLY_SYNTH
    # Where bond ETF is not yet live, fall back to synthetic
    if has_bond:
        df["bond_ret"] = df["bond_ret"].fillna(BOND_WEEKLY_SYNTH)

    df["active_pos"] = df["signal"].shift(1)
    df = df.iloc[1:].copy()

    df["strat_ret"] = np.select(
        [df["active_pos"] == "MID",
         df["active_pos"] == "NIFTY",
         df["active_pos"] == "GOLD",
         df["active_pos"] == "GILT"],
        [df["mid_ret"], df["nifty_ret"], df["gold_ret"], df["bond_ret"]],
        default=CASH_WEEKLY,
    )
    return df


# ── Strategy 4: Donchian Hybrid (Equity/Gold + Midcap) ────────────────────────
def run_donchian_mid(data: dict) -> pd.DataFrame:
    """
    Pure Donchian Hybrid — no moving averages anywhere.

    S1: NiftyBees/GoldBees 20w Donchian → EQUITY or GOLD regime
        Ratio breaks above 20w high → EQUITY
        Ratio breaks below 20w low  → GOLD
        In between                  → hold current regime

    S2: MID150BEES/NiftyBees 20w Donchian → MIDCAP or LARGECAP
        Only evaluated when S1 = EQUITY
        Ratio breaks above 20w high → MIDCAP
        Ratio breaks below 20w low  → LARGECAP
        In between                  → hold current mid/large allocation
        Pre-2019 (no mid data)      → always LARGECAP

    Always invested (MIDCAP / NIFTY / GOLD). No cash.
    """
    nb   = data["NIFTYBEES"]
    gold = data["GOLDBEES"]

    df = pd.concat([nb.rename("nifty"), gold.rename("gold")], axis=1, join="inner").dropna()

    if "MID150BEES" in data:
        df = df.join(data["MID150BEES"].rename("mid"), how="left")
    else:
        df["mid"] = np.nan
    df["mid_live"] = df["mid"].notna()
    df["mid"] = df["mid"].fillna(df["nifty"])

    # ── Donchian channels (shift(1) = strictly prior N bars, no look-ahead) ──
    # S1: equity/gold regime
    df["eg_ratio"]    = df["nifty"] / df["gold"]
    eg_shifted        = df["eg_ratio"].shift(1)
    df["eg_don_high"] = eg_shifted.rolling(DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max()
    df["eg_don_low"]  = eg_shifted.rolling(DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min()

    # S2: mid/large selector
    df["mid_ratio"]    = df["mid"] / df["nifty"]
    mid_shifted        = df["mid_ratio"].shift(1)
    df["mid_don_high"] = mid_shifted.rolling(DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max()
    df["mid_don_low"]  = mid_shifted.rolling(DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min()

    df = df.dropna(subset=["eg_don_high", "eg_don_low", "mid_don_high", "mid_don_low"]).copy()
    if df.empty:
        return df

    # ── S1 signal ─────────────────────────────────────────────────────────────
    s1_raw = pd.Series(np.nan, index=df.index, dtype=object)
    s1_raw[df["eg_ratio"] > df["eg_don_high"]] = "EQUITY"
    s1_raw[df["eg_ratio"] < df["eg_don_low"]]  = "GOLD"
    s1 = s1_raw.ffill().fillna("EQUITY")   # default: start in equity

    # ── S2 signal (independent state machine, applied only when S1=EQUITY) ────
    s2_raw = pd.Series(np.nan, index=df.index, dtype=object)
    s2_raw[df["mid_ratio"] > df["mid_don_high"]] = "MID"
    s2_raw[df["mid_ratio"] < df["mid_don_low"]]  = "NIFTY"
    s2_raw[~df["mid_live"]] = "NIFTY"    # pre-ETF: force largecap
    s2 = s2_raw.ffill().fillna("NIFTY")  # default: largecap

    # ── Combine: S1 gates S2 ─────────────────────────────────────────────────
    df["signal"] = np.where(s1 == "EQUITY", s2, "GOLD")

    # ── Returns ───────────────────────────────────────────────────────────────
    df["nifty_ret"] = df["nifty"].pct_change()
    df["mid_ret"]   = df["mid"].pct_change()
    df["gold_ret"]  = df["gold"].pct_change()

    df["active_pos"] = df["signal"].shift(1)
    df = df.iloc[1:].copy()

    df["strat_ret"] = np.select(
        [df["active_pos"] == "MID",
         df["active_pos"] == "NIFTY"],
        [df["mid_ret"], df["nifty_ret"]],
        default=df["gold_ret"],
    )
    return df


# ── Strategy 5: Donchian Pro (All 3 improvements combined) ────────────────────
def run_donchian_pro(data: dict) -> pd.DataFrame:
    """
    Donchian Pro — three combined improvements over the Donchian Hybrid:

    Improvement 1 — Asymmetric channels:
        Enter equity when EQ/Gold ratio breaks above 13w high (fast entry)
        Enter gold   when EQ/Gold ratio breaks below 26w low  (slow exit)
        Avoids missing equity rallies like 2012 and 2018.

    Improvement 2 — Absolute NiftyBees override:
        If NiftyBees makes a new 20w high → force EQUITY regardless of ratio.
        Catches bull runs where gold is also rallying (ratio stays stuck).

    Improvement 3 — Cash on conflict:
        If ratio says GOLD (below 26w low) but NiftyBees is above its 13w high
        → go to CASH (liquid fund). Avoids being in falling gold during
        equity strength (2015 problem).

    S2 (unchanged): MID150/NiftyBees 20w Donchian → MIDCAP or LARGECAP
    """
    nb   = data["NIFTYBEES"]
    gold = data["GOLDBEES"]

    df = pd.concat([nb.rename("nifty"), gold.rename("gold")], axis=1, join="inner").dropna()

    if "MID150BEES" in data:
        df = df.join(data["MID150BEES"].rename("mid"), how="left")
    else:
        df["mid"] = np.nan
    df["mid_live"] = df["mid"].notna()
    df["mid"] = df["mid"].fillna(df["nifty"])

    # ── Channels (all use shift(1) → strictly prior bars, no look-ahead) ────
    # S1: EQ/Gold asymmetric channel
    df["eg_ratio"]   = df["nifty"] / df["gold"]
    eg_sh            = df["eg_ratio"].shift(1)
    df["eg_13w_high"]= eg_sh.rolling(DONCHIAN_FAST, min_periods=DONCHIAN_FAST).max()
    df["eg_26w_low"] = eg_sh.rolling(DONCHIAN_SLOW, min_periods=DONCHIAN_SLOW).min()

    # S1 override: absolute NiftyBees channel
    nifty_sh            = df["nifty"].shift(1)
    df["nifty_20w_high"]= nifty_sh.rolling(DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max()
    df["nifty_13w_high"]= nifty_sh.rolling(DONCHIAN_FAST,   min_periods=DONCHIAN_FAST).max()

    # S2: MID/NIFTY 20w symmetric channel (unchanged)
    df["mid_ratio"]    = df["mid"] / df["nifty"]
    mid_sh             = df["mid_ratio"].shift(1)
    df["mid_don_high"] = mid_sh.rolling(DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max()
    df["mid_don_low"]  = mid_sh.rolling(DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min()

    df = df.dropna(subset=[
        "eg_13w_high", "eg_26w_low",
        "nifty_20w_high", "nifty_13w_high",
        "mid_don_high", "mid_don_low",
    ]).copy()
    if df.empty:
        return df

    # ── S1 regime signal ─────────────────────────────────────────────────────
    eg   = df["eg_ratio"]
    nif  = df["nifty"]

    # Priority (applied bottom-up, higher priority overwrites lower):
    # GOLD  → ratio below 26w low AND nifty NOT above 13w high
    # CASH  → ratio below 26w low AND nifty above 13w high (conflict)
    # EQUITY→ ratio above 13w high  OR  nifty above 20w high (absolute override)

    gold_cond = (eg  < df["eg_26w_low"])   & ~(nif > df["nifty_13w_high"])
    cash_cond = (eg  < df["eg_26w_low"])   &  (nif > df["nifty_13w_high"])
    eq_cond   = (eg  > df["eg_13w_high"])  |  (nif > df["nifty_20w_high"])

    s1_raw = pd.Series(np.nan, index=df.index, dtype=object)
    s1_raw[gold_cond] = "GOLD"
    s1_raw[cash_cond] = "CASH"
    s1_raw[eq_cond]   = "EQUITY"   # highest priority — overwrites GOLD and CASH

    s1 = s1_raw.ffill().fillna("EQUITY")

    # ── S2 mid/large signal ──────────────────────────────────────────────────
    s2_raw = pd.Series(np.nan, index=df.index, dtype=object)
    s2_raw[df["mid_ratio"] > df["mid_don_high"]] = "MID"
    s2_raw[df["mid_ratio"] < df["mid_don_low"]]  = "NIFTY"
    s2_raw[~df["mid_live"]] = "NIFTY"
    s2 = s2_raw.ffill().fillna("NIFTY")

    # ── Combine ──────────────────────────────────────────────────────────────
    df["signal"] = np.where(
        s1 == "EQUITY", s2,
        np.where(s1 == "CASH", "CASH", "GOLD")
    )

    # ── Returns ──────────────────────────────────────────────────────────────
    df["nifty_ret"] = df["nifty"].pct_change()
    df["mid_ret"]   = df["mid"].pct_change()
    df["gold_ret"]  = df["gold"].pct_change()

    df["active_pos"] = df["signal"].shift(1)
    df = df.iloc[1:].copy()

    df["strat_ret"] = np.select(
        [df["active_pos"] == "MID",
         df["active_pos"] == "NIFTY",
         df["active_pos"] == "GOLD"],
        [df["mid_ret"], df["nifty_ret"], df["gold_ret"]],
        default=CASH_WEEKLY,
    )
    return df


# ── Strategy 6: Donchian Pro v2 ────────────────────────────────────────────────
def run_donchian_pro2(data: dict) -> pd.DataFrame:
    """
    Donchian Pro v2 — two targeted fixes over Pro v1:

    Fix 1 — Smarter conflict resolution:
        Pro v1: ratio says GOLD + NiftyBees above 13w high → always CASH
        Pro v2: also check if GoldBees is above its 20w high
            If gold IS trending up → stay GOLD (gold supercycle case, fixes 2025)
            If gold is NOT trending → CASH (genuine conflict, neither is clear)

    Fix 2 — Dual weakness detector:
        When NiftyBees AND GoldBees are both below their 20w lows simultaneously
        → CASH regardless of any other signal.
        Targets 2013 and 2015 where both assets fell and no rotation helped.

    All other logic identical to Donchian Pro v1.
    """
    nb   = data["NIFTYBEES"]
    gold = data["GOLDBEES"]

    df = pd.concat([nb.rename("nifty"), gold.rename("gold")], axis=1, join="inner").dropna()

    if "MID150BEES" in data:
        df = df.join(data["MID150BEES"].rename("mid"), how="left")
    else:
        df["mid"] = np.nan
    df["mid_live"] = df["mid"].notna()
    df["mid"] = df["mid"].fillna(df["nifty"])

    # ── Channels ─────────────────────────────────────────────────────────────
    df["eg_ratio"]    = df["nifty"] / df["gold"]
    eg_sh             = df["eg_ratio"].shift(1)
    df["eg_13w_high"] = eg_sh.rolling(DONCHIAN_FAST,   min_periods=DONCHIAN_FAST).max()
    df["eg_26w_low"]  = eg_sh.rolling(DONCHIAN_SLOW,   min_periods=DONCHIAN_SLOW).min()

    nifty_sh             = df["nifty"].shift(1)
    df["nifty_20w_high"] = nifty_sh.rolling(DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max()
    df["nifty_13w_high"] = nifty_sh.rolling(DONCHIAN_FAST,   min_periods=DONCHIAN_FAST).max()
    df["nifty_20w_low"]  = nifty_sh.rolling(DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min()  # Fix 2

    gold_sh              = df["gold"].shift(1)
    df["gold_20w_high"]  = gold_sh.rolling(DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max()   # Fix 1
    df["gold_20w_low"]   = gold_sh.rolling(DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min()   # Fix 2

    df["mid_ratio"]    = df["mid"] / df["nifty"]
    mid_sh             = df["mid_ratio"].shift(1)
    df["mid_don_high"] = mid_sh.rolling(DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max()
    df["mid_don_low"]  = mid_sh.rolling(DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min()

    df = df.dropna(subset=[
        "eg_13w_high", "eg_26w_low",
        "nifty_20w_high", "nifty_13w_high", "nifty_20w_low",
        "gold_20w_high", "gold_20w_low",
        "mid_don_high", "mid_don_low",
    ]).copy()
    if df.empty:
        return df

    # ── S1 regime conditions ─────────────────────────────────────────────────
    eg   = df["eg_ratio"]
    nif  = df["nifty"]
    gld  = df["gold"]

    gold_trending  = gld  > df["gold_20w_high"]   # Fix 1: gold in uptrend
    ratio_defensive= eg   < df["eg_26w_low"]
    nifty_strong   = nif  > df["nifty_13w_high"]

    # Fix 2: both assets below 20w lows simultaneously
    dual_weak = (nif < df["nifty_20w_low"]) & (gld < df["gold_20w_low"])

    # Signal logic — applied in priority order (last assignment wins):
    # 1. GOLD  — ratio defensive, nifty weak
    # 2. CASH  — ratio defensive, nifty strong, gold NOT trending  (conflict)
    # 3. GOLD  — ratio defensive, nifty strong, gold IS trending   (Fix 1)
    # 4. EQUITY— ratio above 13w high OR nifty absolute new high
    # 5. CASH  — both assets in downtrend                          (Fix 2, highest priority)

    gold_base    = ratio_defensive & ~nifty_strong
    cash_conflict= ratio_defensive &  nifty_strong & ~gold_trending
    gold_override= ratio_defensive &  nifty_strong &  gold_trending
    eq_cond      = (eg > df["eg_13w_high"]) | (nif > df["nifty_20w_high"])

    s1_raw = pd.Series(np.nan, index=df.index, dtype=object)
    s1_raw[gold_base]     = "GOLD"
    s1_raw[cash_conflict] = "CASH"
    s1_raw[gold_override] = "GOLD"    # Fix 1: gold trend overrides cash
    s1_raw[eq_cond]       = "EQUITY"
    s1_raw[dual_weak]     = "CASH"    # Fix 2: highest priority

    s1 = s1_raw.ffill().fillna("EQUITY")

    # ── S2 mid/large Donchian (unchanged) ────────────────────────────────────
    s2_raw = pd.Series(np.nan, index=df.index, dtype=object)
    s2_raw[df["mid_ratio"] > df["mid_don_high"]] = "MID"
    s2_raw[df["mid_ratio"] < df["mid_don_low"]]  = "NIFTY"
    s2_raw[~df["mid_live"]] = "NIFTY"
    s2 = s2_raw.ffill().fillna("NIFTY")

    # ── Combine ──────────────────────────────────────────────────────────────
    df["signal"] = np.where(
        s1 == "EQUITY", s2,
        np.where(s1 == "CASH", "CASH", "GOLD")
    )

    # ── Returns ──────────────────────────────────────────────────────────────
    df["nifty_ret"] = df["nifty"].pct_change()
    df["mid_ret"]   = df["mid"].pct_change()
    df["gold_ret"]  = df["gold"].pct_change()

    df["active_pos"] = df["signal"].shift(1)
    df = df.iloc[1:].copy()

    df["strat_ret"] = np.select(
        [df["active_pos"] == "MID",
         df["active_pos"] == "NIFTY",
         df["active_pos"] == "GOLD"],
        [df["mid_ret"], df["nifty_ret"], df["gold_ret"]],
        default=CASH_WEEKLY,
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
_mkt = "SPY · MDY · GLD (US)" if IS_US else "NiftyBees · MID150BEES · GoldBees (India)"
st.markdown(f"""
<div style="padding:8px 0 4px 0">
  <span style="font-size:24px;font-weight:800;letter-spacing:-0.5px">📈 Backtest Analysis</span><br>
  <span style="font-size:13px;color:#555;font-family:'Space Mono',monospace">
    {_mkt} &nbsp;|&nbsp; Weekly · Yahoo Finance · No look-ahead bias
  </span>
</div>
""", unsafe_allow_html=True)

with st.spinner("Fetching full historical data from Yahoo Finance…"):
    raw, errors = fetch_all()
    for name, closes in raw.items():
        if isinstance(closes, pd.Series):
            st.toast(f"✅ {name}: {len(closes)} weeks ({closes.index[0].date()} → {closes.index[-1].date()})")

if errors:
    with st.expander("⚠️ Fetch warnings"):
        for e in errors: st.warning(e)

# Show splice info
splice_note = raw.pop("_splice_note", None)
if splice_note and not IS_US:
    st.info(f"📊 Midcap proxy: {splice_note}")

missing = [k for k in ["NIFTYBEES", "GOLDBEES"] if k not in raw]
if missing:
    st.error(f"Cannot run backtest — missing: {', '.join(missing)}")
    st.stop()

has_mid = "MID150BEES" in raw

if not has_mid:
    st.warning("MID150BEES data unavailable — 4-Signal strategy will use NIFTY for all equity phases (no midcap allocation).")

with st.spinner("Running backtests…"):
    try:
        df4 = run_4signal(raw)
    except Exception as e:
        st.error(f"4-Signal error: {e}"); st.exception(e); st.stop()
    try:
        dfD = run_donchian(raw)
    except Exception as e:
        st.error(f"Donchian error: {e}"); st.exception(e); st.stop()
    try:
        dfE = run_enhanced(raw)
    except Exception as e:
        st.error(f"Enhanced error: {e}"); st.exception(e); st.stop()
    try:
        dfH = run_donchian_mid(raw)
    except Exception as e:
        st.error(f"Donchian Hybrid error: {e}"); st.exception(e); st.stop()
    try:
        dfP = run_donchian_pro(raw)
    except Exception as e:
        st.error(f"Donchian Pro error: {e}"); st.exception(e); st.stop()
    try:
        dfP2 = run_donchian_pro2(raw)
    except Exception as e:
        st.error(f"Donchian Pro v2 error: {e}"); st.exception(e); st.stop()

# Guard against empty strategy output
for label, frame in [("4-Signal", df4), ("Donchian", dfD), ("Enhanced", dfE), ("Donchian Hybrid", dfH), ("Donchian Pro", dfP), ("Donchian Pro v2", dfP2)]:
    if frame.empty or "strat_ret" not in frame.columns:
        st.error(f"{label} strategy returned no usable data.")
        st.write(f"Shape: {frame.shape} | Columns: {list(frame.columns)}")
        if not frame.empty:
            st.write("Null counts:", frame.isnull().sum())
        st.stop()

# Align on common period
common_start = max(df4.index[0], dfD.index[0], dfE.index[0], dfH.index[0], dfP.index[0], dfP2.index[0])
common_end   = min(df4.index[-1], dfD.index[-1], dfE.index[-1], dfH.index[-1], dfP.index[-1], dfP2.index[-1])
df4 = df4[(df4.index >= common_start) & (df4.index <= common_end)]
dfD = dfD[(dfD.index >= common_start) & (dfD.index <= common_end)]
dfE = dfE[(dfE.index >= common_start) & (dfE.index <= common_end)]
dfH = dfH[(dfH.index >= common_start) & (dfH.index <= common_end)]
dfP  = dfP[ (dfP.index  >= common_start) & (dfP.index  <= common_end)]
dfP2 = dfP2[(dfP2.index >= common_start) & (dfP2.index <= common_end)]

# Benchmarks
nb_ret   = raw["NIFTYBEES"].pct_change().dropna()
gld_ret  = raw["GOLDBEES"].pct_change().dropna()
nb_ret   = nb_ret[(nb_ret.index >= common_start) & (nb_ret.index <= common_end)]
gld_ret  = gld_ret[(gld_ret.index >= common_start) & (gld_ret.index <= common_end)]

m4   = compute_metrics(df4["strat_ret"], "4-Signal App")
mD   = compute_metrics(dfD["strat_ret"], "Kiru Donchian")
mE   = compute_metrics(dfE["strat_ret"], "Enhanced 5-Signal")
mH   = compute_metrics(dfH["strat_ret"], "Donchian Hybrid")
mP   = compute_metrics(dfP["strat_ret"],  "Donchian Pro")
mP2  = compute_metrics(dfP2["strat_ret"], "Donchian Pro v2")
mNB  = compute_metrics(nb_ret,           "Nifty B&H")
mGLD = compute_metrics(gld_ret,          "Gold B&H")

start_yr = pd.to_datetime(common_start).year
end_yr   = pd.to_datetime(common_end).year

st.markdown(f"""
<div style="font-size:12px;color:#555;font-family:'Space Mono',monospace;margin:8px 0 16px 0">
  Backtest period: {start_yr}–{end_yr} &nbsp;·&nbsp; {len(df4)} weeks &nbsp;·&nbsp;
  Risk-free: {RISK_FREE_RATE*100:.1f}% p.a. &nbsp;·&nbsp; {'US: SPY/MDY/GLD' if IS_US else 'India: NSE ETFs'}
</div>
""", unsafe_allow_html=True)


# ── Summary Table ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📊 Summary Metrics</div>', unsafe_allow_html=True)

all_metrics = [m4, mD, mE, mH, mP, mP2, mNB, mGLD]
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

cols = st.columns(len(all_metrics))
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
yE   = yearly_table(dfE["strat_ret"])
yH   = yearly_table(dfH["strat_ret"])
yP   = yearly_table(dfP["strat_ret"])
yP2  = yearly_table(dfP2["strat_ret"])
yNB  = yearly_table(nb_ret)
yGLD = yearly_table(gld_ret)

yearly_df = pd.DataFrame({
    "4-Signal App":      y4,
    "Kiru Donchian":     yD,
    "Enhanced 5-Signal": yE,
    "Donchian Hybrid":   yH,
    "Donchian Pro":      yP,
    "Donchian Pro v2":   yP2,
    "Nifty B&H":         yNB,
    "Gold B&H":          yGLD,
}).dropna(how="all")

# Format with color
def style_pct(val):
    if pd.isna(val): return ""
    color = "#3CB371" if val >= 0 else "#E74C3C"
    prefix = "+" if val >= 0 else ""
    return f"color: {color}; font-weight: 600"

styled = yearly_df.style.map(style_pct).format(lambda x: f"+{x*100:.1f}%" if x >= 0 else f"{x*100:.1f}%", na_rep="-")
st.dataframe(styled, use_container_width=True)


# ── Equity Curve ──────────────────────────────────────────────────────────────
st.markdown(f'<div class="section-title">📈 Growth of {CURRENCY}{INITIAL_CAPITAL:,}</div>', unsafe_allow_html=True)

eq_df = pd.DataFrame({
    "4-Signal App":      m4["equity"],
    "Kiru Donchian":     mD["equity"],
    "Enhanced 5-Signal": mE["equity"],
    "Donchian Hybrid":   mH["equity"],
    "Donchian Pro":      mP["equity"],
    "Donchian Pro v2":   mP2["equity"],
    "Nifty B&H":         mNB["equity"],
    "Gold B&H":          mGLD["equity"],
}).dropna()

st.line_chart(eq_df, use_container_width=True)

# Final corpus
st.markdown('<div class="section-title">💰 Final Corpus</div>', unsafe_allow_html=True)
fc_cols = st.columns(len(all_metrics))
for i, m in enumerate(all_metrics):
    final_val = m["equity"].iloc[-1]
    multiple  = final_val / INITIAL_CAPITAL
    with fc_cols[i]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{m['label']}</div>
            <div class="metric-value" style="font-size:20px">{CURRENCY}{final_val:,.0f}</div>
            <div style="font-size:13px;color:#888;font-family:'Space Mono',monospace;margin-top:4px">{multiple:.1f}x</div>
        </div>
        """, unsafe_allow_html=True)


# ── Position Breakdown ─────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📦 Time Spent in Each Asset</div>', unsafe_allow_html=True)

pos4_counts = df4["active_pos"].value_counts(normalize=True).mul(100).round(1)
posD_counts = dfD["active_pos"].value_counts(normalize=True).mul(100).round(1)
posE_counts = dfE["active_pos"].value_counts(normalize=True).mul(100).round(1)
posH_counts = dfH["active_pos"].value_counts(normalize=True).mul(100).round(1)
posP_counts  = dfP["active_pos"].value_counts(normalize=True).mul(100).round(1)
posP2_counts = dfP2["active_pos"].value_counts(normalize=True).mul(100).round(1)

pc1, pc2, pc3, pc4, pc5, pc6 = st.columns(6)
asset_colors = {"MID": "#FF8C00", "NIFTY": "#1E90FF", "GOLD": "#DAA520",
                "GILT": "#9B59B6", "CASH": "#3CB371"}

def pos_bars(counts):
    for asset, pct in counts.items():
        clr   = asset_colors.get(asset, "#888")
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

with pc1:
    st.markdown("**4-Signal App**")
    pos_bars(pos4_counts)
with pc2:
    st.markdown("**Kiru Donchian**")
    pos_bars(posD_counts)
with pc3:
    st.markdown("**Enhanced 5-Signal**")
    pos_bars(posE_counts)
with pc4:
    st.markdown("**Donchian Hybrid**")
    pos_bars(posH_counts)
with pc5:
    st.markdown("**Donchian Pro**")
    pos_bars(posP_counts)
with pc6:
    st.markdown("**Donchian Pro v2**")
    pos_bars(posP2_counts)

# Number of switches
n_switches_4 = (df4["signal"] != df4["signal"].shift(1)).sum()
n_switches_D = (dfD["signal"] != dfD["signal"].shift(1)).sum()
n_switches_E = (dfE["signal"] != dfE["signal"].shift(1)).sum()
n_switches_H = (dfH["signal"] != dfH["signal"].shift(1)).sum()
n_switches_P  = (dfP["signal"]  != dfP["signal"].shift(1)).sum()
n_switches_P2 = (dfP2["signal"] != dfP2["signal"].shift(1)).sum()

st.markdown(f"""
<div style="font-size:13px;color:#666;font-family:'Space Mono',monospace;margin-top:8px">
  Switches — 4-Signal: {n_switches_4} &nbsp;|&nbsp; Donchian: {n_switches_D}
  &nbsp;|&nbsp; Enhanced: {n_switches_E} &nbsp;|&nbsp; Hybrid: {n_switches_H} &nbsp;|&nbsp; Pro: {n_switches_P} &nbsp;|&nbsp; Pro v2: {n_switches_P2}
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

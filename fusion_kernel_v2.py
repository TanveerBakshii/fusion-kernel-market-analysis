# fusion_kernel_v2.py
"""
Fusion Kernel v2 - Streamlit app (upgraded)
Features:
 - Async polling of DhanHQ (aiohttp)
 - Cache latest + short history in st.session_state
 - Vectorized SMA/EMA/RSI calculations
 - Polling controls (start/stop/interval)
 - Export CSV snapshot
 - Optional: commit snapshot to GitHub via a PAT (store in Streamlit Secrets)
 - HMAC helper for webhook testing
Note: set your Dhan API key in Streamlit secrets as: {"DHAN_API_KEY": "xxxx"}
Optional GH token: {"GITHUB_TOKEN": "ghp_xxx", "GITHUB_OWNER": "me", "GITHUB_REPO": "repo"}
"""
import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import aiohttp
import threading
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import json
import base64
import hashlib
import hmac
from typing import List, Dict, Any

# Optional GitHub features
try:
    from github import Github
    GITHUB_AVAILABLE = True
except Exception:
    GITHUB_AVAILABLE = False

st.set_page_config(page_title="Fusion Kernel v2", page_icon="🚀", layout="wide")

# ---------------------------
# Utilities & config
# ---------------------------
DHAN_BASE = st.secrets.get("DHAN_BASE", "https://api.dhan.co")  # override if needed
DHAN_API_KEY = st.secrets.get("DHAN_API_KEY", "")
DEFAULT_SYMBOLS = ["NSE:RELIANCE", "NSE:TCS", "NSE:INFY", "NSE:HDFCBANK"]

def now_iso(): return datetime.utcnow().isoformat() + "Z"

def hmac_sha256_hex(secret: str, payload: bytes) -> str:
    return hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()

# Initialize session state containers
if "cache" not in st.session_state:
    st.session_state.cache = {}  # symbol -> latest tick dict
if "history" not in st.session_state:
    st.session_state.history = {}  # symbol -> list[tick]
if "poller" not in st.session_state:
    st.session_state.poller = {"running": False, "interval": 5, "thread": None, "symbols": DEFAULT_SYMBOLS.copy()}
if "last_error" not in st.session_state:
    st.session_state.last_error = None

# ---------------------------
# Market data fetcher (async)
# ---------------------------
async def fetch_symbol(session: aiohttp.ClientSession, symbol: str) -> Dict[str,Any]:
    """
    Replace endpoint /marketdata with the exact Dhan endpoint you use.
    This function expects the Dhan API to accept Bearer token.
    """
    url = f"{DHAN_BASE}/marketdata/{symbol}"
    headers = {"Authorization": f"Bearer {DHAN_API_KEY}"}
    try:
        async with session.get(url, headers=headers, timeout=10) as resp:
            data = await resp.json()
            return {"symbol": symbol, "ok": True, "data": data}
    except Exception as e:
        return {"symbol": symbol, "ok": False, "error": str(e)}

async def fetch_all(symbols: List[str]) -> List[Dict[str,Any]]:
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_symbol(session, s) for s in symbols]
        return await asyncio.gather(*tasks)

def map_dhan_to_tick(symbol: str, raw: Dict[str,Any]) -> Dict[str,Any]:
    """
    Map Dhan response to canonical tick.
    Adapt the field names to your actual Dhan payload structure.
    """
    # Fallback mapping: try common keys
    last = raw.get("lastPrice") or raw.get("last_price") or raw.get("ltp") or raw.get("close")
    bid = raw.get("bestBid") or raw.get("bid")
    ask = raw.get("bestAsk") or raw.get("ask")
    volume = raw.get("volume") or raw.get("totalTradedQty") or raw.get("qty") or 0
    ts = raw.get("timestamp") or now_iso()
    tick = {
        "symbol": symbol,
        "timestamp": ts,
        "ingested_at": now_iso(),
        "last_price": float(last) if last is not None else None,
        "bid": float(bid) if bid is not None else None,
        "ask": float(ask) if ask is not None else None,
        "volume": float(volume),
        "raw": raw
    }
    return tick

# ---------------------------
# Polling background thread
# ---------------------------
def poller_loop(interval: float, symbols: List[str]):
    """
    Runs in a background thread; uses asyncio to fetch concurrently.
    Stores results into st.session_state.cache and history.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    while st.session_state.poller["running"]:
        start = time.time()
        try:
            results = loop.run_until_complete(fetch_all(symbols))
            for res in results:
                if res["ok"]:
                    tick = map_dhan_to_tick(res["symbol"], res["data"])
                    st.session_state.cache[res["symbol"]] = tick
                    hist = st.session_state.history.setdefault(res["symbol"], [])
                    hist.append(tick)
                    # cap history size
                    if len(hist) > 500:
                        hist.pop(0)
                else:
                    st.session_state.last_error = f"{now_iso()} | {res['symbol']} | {res.get('error')}"
        except Exception as e:
            st.session_state.last_error = f"{now_iso()} | poller-loop-exception | {e}"
        # sleep the remainder interval
        elapsed = time.time() - start
        to_sleep = max(0.1, interval - elapsed)
        time.sleep(to_sleep)

def start_poller(interval:int, symbols:List[str]):
    if st.session_state.poller.get("running"):
        return
    st.session_state.poller.update({"running": True, "interval": interval, "symbols": symbols})
    t = threading.Thread(target=poller_loop, args=(interval, symbols), daemon=True)
    st.session_state.poller["thread"] = t
    t.start()

def stop_poller():
    st.session_state.poller["running"] = False
    st.session_state.poller["thread"] = None

# ---------------------------
# Technical indicators (vectorized)
# ---------------------------
def compute_indicators_from_history(df: pd.DataFrame) -> Dict[str,Any]:
    out = {}
    close = df["close"].astype(float)
    out["sma_20"] = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else close.mean()
    out["sma_50"] = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else close.mean()
    out["ema_20"] = close.ewm(span=20).mean().iloc[-1]
    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = (100 - (100 / (1 + rs))).iloc[-1]
    out["rsi"] = float(rsi) if not np.isnan(rsi) else None
    out["current_price"] = float(close.iloc[-1])
    out["trend"] = "BULLISH" if out["sma_20"] > out["sma_50"] else "BEARISH"
    out["rsi_signal"] = "OVERSOLD" if out["rsi"] is not None and out["rsi"] < 30 else ("OVERBOUGHT" if out["rsi"] is not None and out["rsi"] > 70 else "NEUTRAL")
    return out

# ---------------------------
# GitHub snapshot commit (optional)
# ---------------------------
def commit_snapshot_to_github(owner:str, repo:str, path:str, content:bytes, message:str, token:str, branch:str="main"):
    if not GITHUB_AVAILABLE:
        raise RuntimeError("PyGithub not installed. pip install PyGithub")
    gh = Github(token)
    repository = gh.get_repo(f"{owner}/{repo}")
    try:
        # try get existing file
        existing = repository.get_contents(path, ref=branch)
        repository.update_file(path, message, content.decode("utf-8"), existing.sha, branch=branch)
    except Exception as e:
        # probably not exist -> create
        repository.create_file(path, message, content.decode("utf-8"), branch=branch)

# ---------------------------
# UI Layout
# ---------------------------
st.markdown("<h1 style='text-align:center;color:#1f77b4;'>🚀 Fusion Kernel - v2</h1>", unsafe_allow_html=True)
st.write("Advanced Market Analysis — DhanHQ-backed. Configure polling and interact with live cached ticks.")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    symbols_input = st.text_area("Symbols (comma-separated)", value=",".join(st.session_state.poller["symbols"]))
    interval = st.number_input("Poll interval (seconds)", min_value=1, max_value=300, value=int(st.session_state.poller["interval"]))
    start = st.button("Start Polling")
    stop = st.button("Stop Polling")
    snapshot_btn = st.button("Export snapshot (CSV)")
    commit_btn = st.button("Commit snapshot to GitHub (optional)")
    st.markdown("---")
    st.write("Secrets (use Streamlit secrets or env):")
    st.write(f"DHAN_API_KEY set: {'YES' if DHAN_API_KEY else 'NO'}")
    st.write(f"GitHub integration available: {'YES' if GITHUB_AVAILABLE else 'NO'}")
    st.markdown("---")
    st.write("Last poll error:")
    st.code(st.session_state.last_error or "no errors")

# Poller control actions
symbols = [s.strip() for s in symbols_input.split(",") if s.strip()]
if start:
    if not DHAN_API_KEY:
        st.sidebar.error("Set DHAN_API_KEY in Streamlit secrets to start polling.")
    else:
        start_poller(interval, symbols)
        st.sidebar.success("Poller started.")
if stop:
    stop_poller()
    st.sidebar.info("Poller stopped.")

# Main dashboard
col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Live Symbols")
    # show a table of cached ticks
    rows = []
    for s in symbols:
        tick = st.session_state.cache.get(s)
        if tick:
            rows.append({
                "symbol": s,
                "last_price": tick.get("last_price"),
                "bid": tick.get("bid"),
                "ask": tick.get("ask"),
                "volume": tick.get("volume"),
                "ts": tick.get("timestamp"),
                "ingested": tick.get("ingested_at")
            })
        else:
            rows.append({"symbol": s, "last_price": None, "bid": None, "ask": None, "volume": None, "ts": None, "ingested": None})
    df_live = pd.DataFrame(rows).set_index("symbol")
    st.dataframe(df_live, use_container_width=True)

    st.markdown("### Charts & Technicals")
    sel_symbol = st.selectbox("Select symbol for chart", symbols)
    days = st.slider("Days (history) for chart", min_value=10, max_value=365, value=60)

    hist = st.session_state.history.get(sel_symbol, [])
    if len(hist) < 2:
        st.info("Not enough history yet — start the poller and wait a few cycles.")
    else:
        # build a DataFrame for plotting (use recent N days; our hist stores ticks ordered oldest->newest)
        df_hist = pd.DataFrame(hist[-(days*1):])  # ticks are snapshot-like daily if Dhan returns daily; adapt as needed
        # if raw contains OHLC, prefer that, else synthesize
        if "raw" in df_hist.columns and isinstance(df_hist["raw"].iloc[0], dict) and "ohlc" in df_hist["raw"].iloc[0]:
            # implement OHLC extraction if available
            pass
        # create candlestick from 'last_price' as close and +/- variance for open/high/low for demo
        df_plot = pd.DataFrame({
            "date": pd.to_datetime(df_hist["timestamp"]),
            "close": df_hist["last_price"].astype(float),
        }).set_index("date").resample("D").last().ffill()
        # Synthesize open/high/low for visual quality (small variance)
        df_plot["open"] = df_plot["close"].shift(1).fillna(df_plot["close"])
        df_plot["high"] = df_plot[["open","close"]].max(axis=1) * 1.002
        df_plot["low"] = df_plot[["open","close"]].min(axis=1) * 0.998
        df_plot["volume"] = df_hist["volume"].astype(float).resample("D").last().fillna(0)

        # plot candlestick
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df_plot.index,
            open=df_plot["open"],
            high=df_plot["high"],
            low=df_plot["low"],
            close=df_plot["close"],
            name=sel_symbol
        ))
        # add SMA lines
        df_plot["sma20"] = df_plot["close"].rolling(20).mean()
        df_plot["sma50"] = df_plot["close"].rolling(50).mean()
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["sma20"], name="SMA20", mode="lines"))
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["sma50"], name="SMA50", mode="lines"))
        fig.update_layout(height=500, template="plotly_white", title=f"{sel_symbol} Price ({len(df_plot)} points)")
        st.plotly_chart(fig, use_container_width=True)

        # indicators
        indicators = compute_indicators_from_history(pd.DataFrame({"close": df_plot["close"]}))
        st.metric("Current Price", f"₹{indicators['current_price']:.2f}")
        st.metric("SMA(20)", f"₹{indicators['sma_20']:.2f}")
        st.metric("SMA(50)", f"₹{indicators['sma_50']:.2f}")
        st.metric("RSI(14)", f"{indicators['rsi']:.1f}" if indicators["rsi"] else "N/A")
        st.write("Trend:", indicators["trend"], "| RSI Signal:", indicators["rsi_signal"])

with col2:
    st.subheader("Portfolio Snapshot")
    # demo portfolio from live prices
    demo_portfolio = {
        "RELIANCE": {"qty": 10},
        "TCS": {"qty": 5},
        "INFY": {"qty": 8},
    }
    rows = []
    for s, info in demo_portfolio.items():
        price = st.session_state.cache.get(s, {}).get("last_price") or np.nan
        rows.append({"stock": s, "qty": info["qty"], "price": price, "value": info["qty"] * (price if not np.isnan(price) else 0)})
    pf_df = pd.DataFrame(rows)
    st.table(pf_df)
    st.markdown("---")
    st.subheader("Webhook / HMAC tester")
    webhook_secret = st.text_input("Webhook secret (for HMAC)", value="testsecret")
    payload_text = st.text_area("JSON payload", value=json.dumps({"type":"tick","symbol":sel_symbol,"ts":now_iso()}, indent=2))
    if st.button("Compute HMAC (sha256 hex)"):
        sig = hmac_sha256_hex(webhook_secret, payload_text.encode("utf-8"))
        st.code(sig)

# Snapshot export
if snapshot_btn:
    # build a consolidated snapshot json & csv
    snapshot = {"generated_at": now_iso(), "cache": st.session_state.cache, "history_counts": {k: len(v) for k,v in st.session_state.history.items()}}
    csv_buf = []
    for s, hist in st.session_state.history.items():
        for t in hist:
            csv_buf.append({
                "symbol": s,
                "timestamp": t.get("timestamp"),
                "ingested_at": t.get("ingested_at"),
                "last_price": t.get("last_price"),
                "bid": t.get("bid"),
                "ask": t.get("ask"),
                "volume": t.get("volume")
            })
    if csv_buf:
        df_csv = pd.DataFrame(csv_buf)
        csv_bytes = df_csv.to_csv(index=False).encode("utf-8")
        st.download_button("Download snapshot CSV", data=csv_bytes, file_name=f"fusion_snapshot_{int(time.time())}.csv", mime="text/csv")
    else:
        st.info("No history to export yet. Start poller and wait a few cycles.")

# Commit to GitHub
if commit_btn:
    if not GITHUB_AVAILABLE:
        st.error("PyGithub not installed in environment. Install it to enable commit feature.")
    else:
        gh_token = st.secrets.get("GITHUB_TOKEN", "")
        gh_owner = st.secrets.get("GITHUB_OWNER", "")
        gh_repo = st.secrets.get("GITHUB_REPO", "")
        if not all([gh_token, gh_owner, gh_repo]):
            st.error("Configure GITHUB_TOKEN, GITHUB_OWNER, GITHUB_REPO in Streamlit secrets to use this feature.")
        else:
            try:
                # create a compact json snapshot
                snapshot = {"generated_at": now_iso(), "cache": st.session_state.cache}
                content = json.dumps(snapshot, indent=2).encode("utf-8")
                path = f"data/fusion_snapshot_{int(time.time())}.json"
                commit_snapshot_to_github(gh_owner, gh_repo, path, content, f"chore: snapshot {now_iso()}", gh_token)
                st.success(f"Snapshot committed to {gh_owner}/{gh_repo}:{path}")
            except Exception as e:
                st.error(f"GitHub commit failed: {e}")

st.markdown("---")
st.caption("Fusion Kernel v2 — Data powered by your DhanHQ subscription. Keep DHAN_API_KEY in secrets for production use.")

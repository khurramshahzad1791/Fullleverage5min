# app.py
import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from threading import Thread, Event
import queue

st.set_page_config(
    page_title="BTC·ETH·XRP·SOL Trade Station",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for institutional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .signal-card {
        background: #1e2a3a;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 6px solid #3498db;
    }
    .trade-card {
        background: #2c3e50;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 6px solid #2ecc71;
    }
    .grade-AAA { color: #f1c40f; font-weight: bold; }
    .grade-AA { color: #9b59b6; font-weight: bold; }
    .grade-A { color: #3498db; font-weight: bold; }
    .grade-B { color: #2ecc71; font-weight: bold; }
    .grade-C { color: #e74c3c; font-weight: bold; }
    .profit { color: #2ecc71; }
    .loss { color: #e74c3c; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">📡 BTC · ETH · XRP · SOL Trade Station</h1>', unsafe_allow_html=True)
st.markdown("Real‑time scanning • Active trade management • Institutional‑grade")

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/clock--v1.png", width=80)
    st.markdown("## ⚙️ Settings")

    # Pakistan time
    now_pkt = datetime.utcnow() + timedelta(hours=5)
    st.info(f"**PKT:** {now_pkt.strftime('%H:%M:%S')}")

    # Risk parameters
    st.markdown("### 💰 Risk Management")
    account_balance = st.number_input("Account (USDT)", value=1000, step=100)
    risk_per_trade = st.slider("Risk per trade (%)", 0.1, 2.0, 0.5, 0.1) / 100
    max_leverage = st.selectbox("Max Leverage", [50, 100, 200, 300, 500], index=1)

    # Update frequency
    update_sec = st.slider("Refresh interval (sec)", 10, 120, 30, 5)

    st.markdown("---")
    if st.button("🔄 Reset All Trades"):
        for key in list(st.session_state.keys()):
            if key not in ['pairs']:
                del st.session_state[key]
        st.rerun()

# Fixed pairs
PAIRS = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "SOL/USDT"]

# -------------------- SESSION STATE --------------------
if 'signals' not in st.session_state:
    st.session_state.signals = {}          # latest signals per pair
if 'active_trades' not in st.session_state:
    st.session_state.active_trades = []    # list of dicts for taken trades
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}       # store OHLCV data per pair per timeframe

# -------------------- DATA FETCHING --------------------
def fetch_ohlcv(symbol, tf='5m', limit=200):
    """Fetch OHLCV from Binance (futures)"""
    try:
        exchange = ccxt.binanceusdm({'enableRateLimit': True, 'timeout': 10000})
        ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        st.warning(f"Data error {symbol}: {e}")
        return None

# -------------------- INDICATORS --------------------
def compute_indicators(df):
    """Add all indicators to dataframe."""
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # Bollinger Bands (20,2)
    df['bb_mid'] = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * bb_std
    df['bb_lower'] = df['bb_mid'] - 2 * bb_std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

    # RSI (14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Stochastic (14,3,3)
    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    df['stoch_k'] = 100 * (close - low_14) / (high_14 - low_14)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    # MACD (12,26,9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # ADX (14) for trend strength
    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr)
    minus_di = 100 * (minus_dm.abs().ewm(alpha=1/14).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['adx'] = dx.rolling(14).mean()

    # Volume surge
    df['vol_ma20'] = volume.rolling(20).mean()
    df['vol_surge'] = volume / df['vol_ma20']

    # ATR
    df['atr'] = atr

    return df

# -------------------- REGIME DETECTION --------------------
def detect_regime(df_trend):
    """Determine market regime (trending or consolidation)"""
    last_adx = df_trend['adx'].iloc[-1] if not pd.isna(df_trend['adx'].iloc[-1]) else 0
    last_bb_width = df_trend['bb_width'].iloc[-1] if not pd.isna(df_trend['bb_width'].iloc[-1]) else 0.1
    bb_width_ma = df_trend['bb_width'].rolling(20).mean().iloc[-1] or 0.1

    if last_adx > 25 and last_bb_width > bb_width_ma:
        return "STRONG_TREND"
    elif last_adx > 20:
        return "WEAK_TREND"
    else:
        return "CONSOLIDATION"

# -------------------- SIGNAL GENERATION --------------------
def generate_signal(pair, df_5m, df_1h):
    """Return signal dict or None"""
    last = df_5m.iloc[-1]
    prev = df_5m.iloc[-2]

    regime = detect_regime(df_1h)

    # Mean Reversion (for consolidation)
    if regime in ["CONSOLIDATION", "WEAK_TREND"]:
        bb_touch_long = last['low'] <= last['bb_lower'] and last['close'] > last['bb_lower']
        bb_touch_short = last['high'] >= last['bb_upper'] and last['close'] < last['bb_upper']
        rsi_ok_long = last['rsi'] < 35
        rsi_ok_short = last['rsi'] > 65
        stoch_ok_long = last['stoch_k'] < 25
        stoch_ok_short = last['stoch_k'] > 75
        macd_ok_long = last['macd_hist'] > prev['macd_hist'] and last['macd_hist'] < 0
        macd_ok_short = last['macd_hist'] < prev['macd_hist'] and last['macd_hist'] > 0
        vol_ok = last['vol_surge'] > 1.5

        long_score = sum([bb_touch_long, rsi_ok_long, stoch_ok_long, macd_ok_long, vol_ok])
        short_score = sum([bb_touch_short, rsi_ok_short, stoch_ok_short, macd_ok_short, vol_ok])

        if long_score >= 3 and bb_touch_long:
            direction = "LONG"
            confidence = 60 + long_score * 8
            atr = last['atr']
            entry = last['close']
            sl = entry - atr * 1.5
            tp1 = entry + atr * 3
            tp2 = entry + atr * 5
            strategy = "MeanRev"
        elif short_score >= 3 and bb_touch_short:
            direction = "SHORT"
            confidence = 60 + short_score * 8
            atr = last['atr']
            entry = last['close']
            sl = entry + atr * 1.5
            tp1 = entry - atr * 3
            tp2 = entry - atr * 5
            strategy = "MeanRev"
        else:
            return None

    # Momentum Breakout (for strong trend)
    elif regime == "STRONG_TREND":
        prev_high_20 = df_5m['high'].iloc[-20:-1].max()
        prev_low_20 = df_5m['low'].iloc[-20:-1].min()
        breakout_long = last['close'] > prev_high_20 and last['vol_surge'] > 1.8
        breakout_short = last['close'] < prev_low_20 and last['vol_surge'] > 1.8

        if breakout_long:
            direction = "LONG"
            confidence = 80 + (10 if last['vol_surge'] > 2.5 else 5)
            atr = last['atr']
            entry = last['close']
            sl = entry - atr * 1.2
            tp1 = entry + atr * 2.5
            tp2 = entry + atr * 4
            strategy = "Momentum"
        elif breakout_short:
            direction = "SHORT"
            confidence = 80 + (10 if last['vol_surge'] > 2.5 else 5)
            atr = last['atr']
            entry = last['close']
            sl = entry + atr * 1.2
            tp1 = entry - atr * 2.5
            tp2 = entry - atr * 4
            strategy = "Momentum"
        else:
            return None
    else:
        return None

    # Grade
    if confidence >= 90:
        grade = "A+"
    elif confidence >= 80:
        grade = "A"
    elif confidence >= 70:
        grade = "B+"
    elif confidence >= 60:
        grade = "B"
    else:
        grade = "C"

    return {
        'pair': pair,
        'regime': regime,
        'strategy': strategy,
        'direction': direction,
        'grade': grade,
        'confidence': confidence,
        'entry': entry,
        'sl': sl,
        'tp1': tp1,
        'tp2': tp2,
        'timestamp': datetime.now(),
        'price': entry
    }

# -------------------- UPDATE SIGNALS --------------------
def update_all_signals():
    """Fetch fresh data and generate signals for all pairs."""
    new_signals = {}
    for pair in PAIRS:
        df_5m = fetch_ohlcv(pair, '5m', 200)
        df_1h = fetch_ohlcv(pair, '1h', 200)
        if df_5m is not None and df_1h is not None:
            df_5m = compute_indicators(df_5m)
            df_1h = compute_indicators(df_1h)
            sig = generate_signal(pair, df_5m, df_1h)
            if sig:
                new_signals[pair] = sig
            # Store data for chart
            st.session_state.data_cache[pair] = (df_5m, df_1h)
    st.session_state.signals = new_signals
    st.session_state.last_update = datetime.now()

# -------------------- TRADE MANAGEMENT --------------------
def update_active_trades():
    """Check active trades against current price and suggest actions."""
    for trade in st.session_state.active_trades:
        pair = trade['pair']
        # Get current price from latest signal or fresh fetch
        if pair in st.session_state.data_cache:
            df_5m, _ = st.session_state.data_cache[pair]
            current_price = df_5m['close'].iloc[-1]
        else:
            # fallback fetch
            df = fetch_ohlcv(pair, '1m', 1)
            if df is not None:
                current_price = df['close'].iloc[-1]
            else:
                continue

        trade['current_price'] = current_price
        if trade['direction'] == 'LONG':
            trade['pnl_pct'] = (current_price - trade['entry']) / trade['entry'] * 100
            # Suggestions
            if current_price >= trade['tp1']:
                trade['suggestion'] = "✅ TP1 reached – take partial profits or move SL to breakeven"
            elif current_price >= trade['entry'] + (trade['tp1'] - trade['entry']) * 0.5:
                trade['suggestion'] = "📈 50% to TP1 – consider moving SL to breakeven"
            elif current_price <= trade['sl']:
                trade['suggestion'] = "❌ Stop hit – trade closed (update manually)"
            else:
                trade['suggestion'] = "⏳ Holding"
        else:  # SHORT
            trade['pnl_pct'] = (trade['entry'] - current_price) / trade['entry'] * 100
            if current_price <= trade['tp1']:
                trade['suggestion'] = "✅ TP1 reached – take partial profits or move SL to breakeven"
            elif current_price <= trade['entry'] - (trade['entry'] - trade['tp1']) * 0.5:
                trade['suggestion'] = "📈 50% to TP1 – consider moving SL to breakeven"
            elif current_price >= trade['sl']:
                trade['suggestion'] = "❌ Stop hit – trade closed (update manually)"
            else:
                trade['suggestion'] = "⏳ Holding"

# -------------------- BACKGROUND THREAD FOR AUTO-UPDATE --------------------
class DataUpdater(Thread):
    def __init__(self, interval):
        super().__init__()
        self.interval = interval
        self.stop_event = Event()

    def run(self):
        while not self.stop_event.is_set():
            update_all_signals()
            update_active_trades()
            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()

# Start background thread if not already running
if 'updater_thread' not in st.session_state:
    st.session_state.updater_thread = DataUpdater(update_sec)
    st.session_state.updater_thread.daemon = True
    st.session_state.updater_thread.start()

# -------------------- UI: MAIN AREA --------------------
# Metrics bar
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("BTC", "Signal" if "BTC/USDT" in st.session_state.signals else "—")
with col2:
    st.metric("ETH", "Signal" if "ETH/USDT" in st.session_state.signals else "—")
with col3:
    st.metric("XRP", "Signal" if "XRP/USDT" in st.session_state.signals else "—")
with col4:
    st.metric("SOL", "Signal" if "SOL/USDT" in st.session_state.signals else "—")
with col5:
    st.metric("Active Trades", len(st.session_state.active_trades))

if st.session_state.last_update:
    st.caption(f"Last update: {st.session_state.last_update.strftime('%H:%M:%S')}")

# -------------------- CURRENT SIGNALS --------------------
st.subheader("📡 Live Signals")
if st.session_state.signals:
    cols = st.columns(4)
    for i, (pair, sig) in enumerate(st.session_state.signals.items()):
        with cols[i % 4]:
            grade_color = {
                'A+': 'grade-AAA',
                'A': 'grade-AA',
                'B+': 'grade-A',
                'B': 'grade-B',
                'C': 'grade-C'
            }.get(sig['grade'], '')
            st.markdown(f"""
            <div class="signal-card">
                <h3>{pair.replace('/USDT','')}</h3>
                <p><span class="{grade_color}">{sig['grade']}</span> • {sig['direction']}</p>
                <p>Entry: ${sig['entry']:.2f}</p>
                <p>SL: ${sig['sl']:.2f} | TP1: ${sig['tp1']:.2f}</p>
                <p>{sig['strategy']} • {sig['regime']}</p>
            </div>
            """, unsafe_allow_html=True)

            # Take trade button
            if st.button(f"📥 Take {pair}", key=f"take_{pair}"):
                # Add to active trades
                new_trade = sig.copy()
                new_trade['taken_at'] = datetime.now()
                new_trade['current_price'] = sig['entry']
                new_trade['pnl_pct'] = 0.0
                new_trade['suggestion'] = "New trade – monitor"
                st.session_state.active_trades.append(new_trade)
                st.rerun()
else:
    st.info("No signals at the moment. Waiting for next update...")

# -------------------- ACTIVE TRADES --------------------
st.subheader("📊 Active Trades")
if st.session_state.active_trades:
    for i, trade in enumerate(st.session_state.active_trades):
        with st.container():
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            pair = trade['pair'].replace('/USDT','')
            pnl = trade['pnl_pct']
            pnl_class = "profit" if pnl >= 0 else "loss"
            with col1:
                st.markdown(f"**{pair}** {trade['direction']}")
            with col2:
                st.markdown(f"Entry: ${trade['entry']:.2f}")
            with col3:
                st.markdown(f"Now: ${trade['current_price']:.2f}")
            with col4:
                st.markdown(f"<span class='{pnl_class}'>{pnl:.2f}%</span>", unsafe_allow_html=True)
            with col5:
                st.markdown(f"SL: ${trade['sl']:.2f}")
            with col6:
                st.markdown(trade['suggestion'])

            # Action buttons
            cola, colb, colc, cold = st.columns(4)
            with cola:
                if st.button(f"🔒 Close {pair}", key=f"close_{i}"):
                    st.session_state.active_trades.pop(i)
                    st.rerun()
            with colb:
                if st.button(f"🎯 Move SL to Breakeven", key=f"be_{i}"):
                    trade['sl'] = trade['entry']
                    trade['suggestion'] = "SL moved to breakeven"
                    st.rerun()
            with colc:
                if st.button(f"💰 Take 50% Profit", key=f"tp50_{i}"):
                    # Simulate taking half profit – adjust entry for remaining half? 
                    # For simplicity, just note suggestion.
                    trade['suggestion'] = "Took 50% profit at current price"
                    st.rerun()
            with cold:
                if st.button(f"📈 Trail SL", key=f"trail_{i}"):
                    # Trail by 0.5 ATR? For demo, move to 50% of profit.
                    if trade['direction'] == 'LONG':
                        new_sl = trade['current_price'] - (trade['current_price'] - trade['entry']) * 0.3
                    else:
                        new_sl = trade['current_price'] + (trade['entry'] - trade['current_price']) * 0.3
                    trade['sl'] = new_sl
                    trade['suggestion'] = "Stop trailed"
                    st.rerun()
            st.markdown("---")
else:
    st.info("No active trades. Take a signal from above.")

# -------------------- CHART FOR SELECTED PAIR --------------------
st.subheader("📈 Chart")
selected_chart = st.selectbox("Select pair for chart", PAIRS)
if selected_chart in st.session_state.data_cache:
    df_5m, df_1h = st.session_state.data_cache[selected_chart]
    fig = go.Figure(data=[
        go.Candlestick(
            x=df_5m['timestamp'][-100:],
            open=df_5m['open'][-100:],
            high=df_5m['high'][-100:],
            low=df_5m['low'][-100:],
            close=df_5m['close'][-100:],
            name="5m"
        )
    ])
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df_5m['timestamp'][-100:], y=df_5m['bb_upper'][-100:],
                              line=dict(color='gray', dash='dash'), name='BB Upper'))
    fig.add_trace(go.Scatter(x=df_5m['timestamp'][-100:], y=df_5m['bb_lower'][-100:],
                              line=dict(color='gray', dash='dash'), name='BB Lower', fill='tonexty'))

    # If there's an active trade for this pair, show entry/SL/TP
    active = [t for t in st.session_state.active_trades if t['pair'] == selected_chart]
    if active:
        t = active[0]
        fig.add_hline(y=t['entry'], line_dash="dash", line_color="white", annotation_text="Entry")
        fig.add_hline(y=t['sl'], line_dash="dash", line_color="red", annotation_text="SL")
        fig.add_hline(y=t['tp1'], line_dash="dash", line_color="green", annotation_text="TP1")
        fig.add_hline(y=t['tp2'], line_dash="dash", line_color="lime", annotation_text="TP2")

    fig.update_layout(
        title=f"{selected_chart} – 5m",
        xaxis_rangeslider_visible=False,
        height=500,
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------- CLEANUP ON EXIT --------------------
# (Note: Streamlit doesn't have a direct on-exit hook, but we can ignore – thread will be killed when app stops)

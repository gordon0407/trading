import streamlit as st

# --- 頁面設定：一定要最早 ---
st.set_page_config(page_title="六大指標轉折分析儀", layout="wide")

try:
    import yfinance as yf
except ImportError:
    st.error("Please install yfinance: pip install yfinance")
    st.stop()

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Session state 初始化 ---
if "watchlist" not in st.session_state:
    st.session_state.watchlist = ["0388.HK", "MSFT"]

if "selected_stock" not in st.session_state:
    st.session_state.selected_stock = "0700.HK"

# --- 功能選單 ---
page = st.sidebar.radio("📂 功能選單", ["📈 技術分析", "⭐ 自選股"])


# --- 指標計算函數 ---
def calculate_indicators(df):
    df = df.copy()

    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["Signal_Line"]
    df["MACD5"] = df["MACD"].ewm(span=5, adjust=False).mean()

    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df["RSI"] = 100 - (100 / (1 + (gain / loss)))

    low_min = df["Low"].rolling(window=9).min()
    high_max = df["High"].rolling(window=9).max()
    rsv = (df["Close"] - low_min) / (high_max - low_min) * 100
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    df["J"] = 3 * k - 2 * d

    df["LWR"] = -100 * (
        (df["High"].rolling(window=14).max() - df["Close"])
        / (df["High"].rolling(window=14).max() - df["Low"].rolling(window=14).min())
    )

    df["BBI"] = (
        df["Close"].rolling(window=3).mean()
        + df["Close"].rolling(window=6).mean()
        + df["Close"].rolling(window=12).mean()
        + df["Close"].rolling(window=24).mean()
    ) / 4

    df["MTM"] = df["Close"] - df["Close"].shift(12)

    return df


def apply_switch_signals(df):
    cols = ["MACD", "J", "RSI", "LWR", "BBI", "MTM"]
    diffs = df[cols].diff()
    up_count = (diffs > 0).sum(axis=1)
    down_count = (diffs < 0).sum(axis=1)

    signals = ["Wait"] * len(df)
    current_state = "None"

    for i in range(len(df)):
        m_curr = df["MACD"].iloc[i]
        m5_curr = df["MACD5"].iloc[i]
        ups = up_count.iloc[i]
        downs = down_count.iloc[i]

        # 注意：原本 if 條件括號優先次序好易出事，呢度已經加括號避免誤判
        if (current_state != "Long") and ((m_curr >= m5_curr) or ((m5_curr - m_curr) >= 2)) and (downs >= 4):
            signals[i] = "BUY"
            current_state = "Long"
        elif (current_state != "Short") and ((m_curr =< m5_curr) or ((m_curr - m5_curr) >= 2)) and (ups >= 3):
            signals[i] = "SELL"
            current_state = "Short"

    df["Signal"] = signals

    trend_symbols = pd.DataFrame(index=df.index)
    for c in cols:
        trend_symbols[c] = diffs[c].apply(lambda x: "🟢↑" if x > 0 else ("🔴↓" if x < 0 else "⚪-"))

    return df, trend_symbols


# ===== 自選股頁 =====
if page == "⭐ 自選股":
    st.title("⭐ 我的自選股")

    new_stock = st.text_input("➕ 新增股票代號（例如 0700.HK / MSFT）")

    col_add, col_clear = st.columns([1, 1])
    if col_add.button("加入自選"):
        code = (new_stock or "").strip()
        if not code:
            st.warning("請輸入股票代號")
        elif code in st.session_state.watchlist:
            st.info("已經喺自選股入面")
        else:
            st.session_state.watchlist.append(code)
            st.success(f"{code} 已加入自選股")
            st.rerun()

    if col_clear.button("清空自選"):
        st.session_state.watchlist = []
        st.rerun()

    st.divider()

    if not st.session_state.watchlist:
        st.info("你仲未有自選股，先加一隻試下。")
    else:
        for stock in st.session_state.watchlist:
            col1, col2, col3 = st.columns([3, 1, 1])
            col1.write(stock)

            if col2.button("🔍 分析", key=f"analyze_{stock}"):
                st.session_state.selected_stock = stock
                st.rerun()

            if col3.button("❌ 刪除", key=f"delete_{stock}"):
                st.session_state.watchlist.remove(stock)
                st.rerun()


# ===== 技術分析頁 =====
if page == "📈 技術分析":
    st.title("📈 六大指標轉折分析儀")

    st.sidebar.header("🔍 股票設定")
    default_stock = st.session_state.get("selected_stock", "0700.HK")

    ticker_input = st.sidebar.text_input("輸入股票代號", value=default_stock)
    period = st.sidebar.selectbox("查看範圍", ["3mo", "6mo", "1y", "2y"], index=1)
    start_analysis = st.sidebar.button("分析數據")

    if start_analysis:
        try:
            df = yf.download(ticker_input, period=period, auto_adjust=True)

            if df is None or df.empty:
                st.error("下載唔到數據：請檢查股票代號或網絡狀態")
                st.stop()

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = calculate_indicators(df)
            df, trend_table = apply_switch_signals(df)
            df = df.dropna()

            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.07,
                subplot_titles=(f"{ticker_input} K線 (僅顯示轉折點)", "MACD 與 MACD5"),
                row_width=[0.3, 0.7],
            )

            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df["Open"],
                    high=df["High"],
                    low=df["Low"],
                    close=df["Close"],
                    name="K線",
                    opacity=0.4,
                ),
                row=1,
                col=1,
            )

            buy_pts = df[df["Signal"] == "BUY"]
            fig.add_trace(
                go.Scatter(
                    x=buy_pts.index,
                    y=buy_pts["Low"] * 0.98,
                    mode="markers+text",
                    name="首次買入",
                    marker=dict(symbol="star", size=15, color="#00FF00"),
                    text="BUY",
                    textposition="bottom center",
                ),
                row=1,
                col=1,
            )

            sell_pts = df[df["Signal"] == "SELL"]
            fig.add_trace(
                go.Scatter(
                    x=sell_pts.index,
                    y=sell_pts["High"] * 1.02,
                    mode="markers+text",
                    name="首次賣出",
                    marker=dict(symbol="x", size=12, color="#FF0000"),
                    text="SELL",
                    textposition="top center",
                ),
                row=1,
                col=1,
            )

            fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="#1f77b4")), row=2, col=1)
            fig.add_trace(
                go.Scatter(x=df.index, y=df["MACD5"], name="MACD5", line=dict(color="#ff7f0e", dash="dot")),
                row=2,
                col=1,
            )

            fig.update_layout(height=800, template="plotly_white", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📋 轉折點與趨勢記錄表")
            record_display = trend_table.copy()
            record_display["MACD > MACD5"] = (df["MACD"] > df["MACD5"]).apply(lambda x: "✅ 是" if x else "❌ 否")
            record_display["操作指令"] = df["Signal"].apply(
                lambda x: "🟢 買入訊號" if x == "BUY" else ("🔴 賣出訊號" if x == "SELL" else "-")
            )

            signal_only = record_display[record_display["操作指令"] != "-"]

            tab1, tab2 = st.tabs(["所有交易日記錄", "僅顯示訊號日"])
            with tab1:
                st.dataframe(record_display.iloc[::-1].head(30), use_container_width=True)
            with tab2:
                st.dataframe(signal_only.iloc[::-1], use_container_width=True)

        except Exception as e:
            st.error(f"分析出錯: {e}")

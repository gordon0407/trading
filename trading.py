import streamlit as st
try:
    import yfinance as yf
except ImportError:
    st.error("Please install yfinance: pip install yfinance")
    st.stop()
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 頁面設定 ---
st.set_page_config(page_title="六大指標轉折分析儀", layout="wide")

# --- 指標計算函數 ---
def calculate_indicators(df):
    df = df.copy()
    
    # 1. MACD (12, 26, 9)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2  
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean() 
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line'] 
    
    # MACD5 線 (MACD 的 5日均線)
    df['MACD5'] = df['MACD'].ewm(span=5, adjust=False).mean()
    
    # 2. RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))

    # 3. KDJ (9, 3, 3)
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    df['J'] = 3 * k - 2 * d

    # 4. LWR (Williams %R)
    df['LWR'] = -100 * ((df['High'].rolling(window=14).max() - df['Close']) / 
                        (df['High'].rolling(window=14).max() - df['Low'].rolling(window=14).min()))

    # 5. BBI
    df['BBI'] = (df['Close'].rolling(window=3).mean() + 
                 df['Close'].rolling(window=6).mean() + 
                 df['Close'].rolling(window=12).mean() + 
                 df['Close'].rolling(window=24).mean()) / 4

    # 6. MTM (12)
    df['MTM'] = df['Close'] - df['Close'].shift(12)
    
    return df

# --- 核心開關訊號邏輯 (更新版) ---
def apply_switch_signals(df):
    cols = ['MACD', 'J', 'RSI', 'LWR', 'BBI', 'MTM']
    diffs = df[cols].diff()
    up_count = (diffs > 0).sum(axis=1)
    down_count = (diffs < 0).sum(axis=1)
    
    # 準備存儲訊號的列表
    signals = ["Wait"] * len(df)
    current_state = "None" # 追蹤當前持倉狀態: "None", "Long", "Short"
    
    # 使用迴圈來判斷轉折點 (因為後一天的訊號依賴於前一天的狀態)
    for i in range(len(df)):
        m_curr = df['MACD'].iloc[i]
        m5_curr = df['MACD5'].iloc[i]
        ups = up_count.iloc[i]
        downs = down_count.iloc[i]
        
        # 買入觸發：當前不是多頭狀態 + 交叉向上 + 3個以上指標向上
        if current_state != "Long" and (m_curr > m5_curr) or ((m5_curr - m_curr)>2) and ups >= 4:
            signals[i] = "BUY"
            current_state = "Long"
            
        # 賣出觸發：當前不是空頭狀態 + 交叉向下 + 3個以上指標向下
        elif current_state != "Short" and m_curr < m5_curr and downs >= 3:
            signals[i] = "SELL"
            current_state = "Short"
    
    df['Signal'] = signals
    
    # 趨勢符號用於表格顯示
    trend_symbols = pd.DataFrame(index=df.index)
    for c in cols:
        trend_symbols[c] = diffs[c].apply(lambda x: "🟢↑" if x > 0 else ("🔴↓" if x < 0 else "⚪-"))
    
    return df, trend_symbols

# --- 介面呈現 ---
st.sidebar.header("🔍 股票設定")
ticker_input = st.sidebar.text_input("輸入股票代號", value="0700.HK")
period = st.sidebar.selectbox("查看範圍", ["3mo", "6mo", "1y", "2y"], index=1)
start_analysis = st.sidebar.button("分析數據")

if start_analysis:
    try:
        df = yf.download(ticker_input, period=period, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = calculate_indicators(df)
        df, trend_table = apply_switch_signals(df)
        df = df.dropna()

        # --- 雙層圖表 ---
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.07, 
                           subplot_titles=(f'{ticker_input} K線 (僅顯示轉折點)', 'MACD 與 MACD5'),
                           row_width=[0.3, 0.7])

        # 1. K線圖
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name='K線', opacity=0.4 # 調低 K 線亮度，讓訊號更顯眼
        ), row=1, col=1)

        # 標註買入訊號 (僅轉折點)
        buy_pts = df[df['Signal'] == 'BUY']
        fig.add_trace(go.Scatter(
            x=buy_pts.index, y=buy_pts['Low'] * 0.98,
            mode='markers+text', name='首次買入',
            marker=dict(symbol='star', size=15, color='#00FF00'),
            text="BUY", textposition="bottom center"
        ), row=1, col=1)

        # 標註賣出訊號 (僅轉折點)
        sell_pts = df[df['Signal'] == 'SELL']
        fig.add_trace(go.Scatter(
            x=sell_pts.index, y=sell_pts['High'] * 1.02,
            mode='markers+text', name='首次賣出',
            marker=dict(symbol='x', size=12, color='#FF0000'),
            text="SELL", textposition="top center"
        ), row=1, col=1)

        # 2. MACD 子圖
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='#1f77b4')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD5'], name='MACD5', line=dict(color='#ff7f0e', dash='dot')), row=2, col=1)

        fig.update_layout(height=800, template="plotly_white", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # 3. 歷史記錄表
        st.subheader("📋 轉折點與趨勢記錄表")
        record_display = trend_table.copy()
        record_display['MACD > MACD5'] = (df['MACD'] > df['MACD5']).apply(lambda x: "✅ 是" if x else "❌ 否")
        record_display['操作指令'] = df['Signal'].apply(
            lambda x: "🟢 買入訊號" if x == "BUY" else ("🔴 賣出訊號" if x == "SELL" else "-")
        )
        
        # 只過濾出有訊號的日子顯示
        signal_only = record_display[record_display['操作指令'] != "-"]
        
        col1, col2 = st.tabs(["所有交易日記錄", "僅顯示訊號日"])
        
        with col1:
            st.dataframe(record_display.iloc[::-1].head(30), use_container_width=True)
        with col2:
            st.dataframe(signal_only.iloc[::-1], use_container_width=True)

    except Exception as e:
        st.error(f"分析出錯: {e}")
import streamlit as st
import pandas as pd
import numpy as np
import glob
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import yfinance as yf

def fetch_stock_data(ticker):
    """Fetches 5y data for a ticker and saves it to CSV."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="5y")
        if df.empty:
            return False, "No data found for this ticker."
        
        filename = f"{ticker}_5y.csv"
        df.to_csv(filename)
        return True, filename
    except Exception as e:
        return False, str(e)

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df.set_index('Date', inplace=True)
    return df

def main():
    # Basic i18n translations for English and Chinese
    LANG = {
        'en': {
            'page_title': "Stock Analysis App",
            'title': "Interactive Stock Analysis",
            'select_existing': "Select Existing Data",
            'choose_file': "Choose a file:",
            'fetch_new': "Fetch New Data",
            'enter_ticker': "Enter Ticker Symbol (e.g., TSLA, AAPL):",
            'fetch_button': "Fetch Data",
            'fetching': "Fetching data for {ticker}...",
            'fetch_success': "Successfully fetched details for {ticker}!",
            'fetch_error': "Error: {msg}",
            'analysis_for': "Analysis for {ticker}",
            'key_stats': "Key Statistics",
            'latest_close': "Latest Close",
            'highest_5y': "Highest (5y)",
            'lowest_5y': "Lowest (5y)",
            'avg_volume': "Avg Volume",
            'price_volume_chart': "Price and Volume Chart",
            'select_window': "Select Trend Observation Window:",
            'window_1w': "1 Week (5 Days)",
            'window_1m': "1 Month (20 Days)",
            'window_1q': "1 Quarter (63 Days)",
            'price': "Price",
            'volume': "Volume",
            'show_raw': "Show Raw Data",
            'language_label': "Language / 语言",
        },
        'zh': {
            'page_title': "股票分析应用",
            'title': "交互式股票分析",
            'select_existing': "选择已有数据",
            'choose_file': "选择文件：",
            'fetch_new': "获取新数据",
            'enter_ticker': "输入股票代码（如 TSLA、AAPL）：",
            'fetch_button': "获取数据",
            'fetching': "正在为 {ticker} 获取数据...",
            'fetch_success': "成功获取 {ticker} 的数据！",
            'fetch_error': "错误：{msg}",
            'analysis_for': "{ticker} 的分析",
            'key_stats': "关键统计",
            'latest_close': "最新收盘价",
            'highest_5y': "5年最高",
            'lowest_5y': "5年最低",
            'avg_volume': "平均成交量",
            'price_volume_chart': "价格与成交量图",
            'select_window': "选择趋势观察窗口：",
            'window_1w': "1周（5 天）",
            'window_1m': "1月（20 天）",
            'window_1q': "1季（63 天）",
            'price': "价格",
            'volume': "成交量",
            'show_raw': "显示原始数据",
            'language_label': "Language / 语言",
        }
    }

    # Language selector
    lang_choice = st.sidebar.selectbox(LANG['en']['language_label'], ["English", "中文"], index=0)
    lang = 'en' if lang_choice == 'English' else 'zh'

    def t(key, **kwargs):
        """Translate helper."""
        txt = LANG[lang].get(key, key)
        try:
            return txt.format(**kwargs) if kwargs else txt
        except Exception:
            return txt

    st.set_page_config(page_title=LANG[lang]['page_title'], layout="wide")
    st.title(LANG[lang]['title'])

    # Sidebar for file selection
    st.sidebar.subheader(LANG[lang]['select_existing'])
    csv_files = glob.glob("*_5y.csv")
    selected_file = st.sidebar.selectbox(LANG[lang]['choose_file'], csv_files) if csv_files else None

    # Sidebar for new data fetch
    st.sidebar.markdown("---")
    st.sidebar.subheader(LANG[lang]['fetch_new'])
    new_ticker = st.sidebar.text_input(LANG[lang]['enter_ticker']).upper()
    
    if st.sidebar.button(LANG[lang]['fetch_button']):
        if not new_ticker:
            st.sidebar.error(LANG[lang]['fetch_error'].format(msg="Ticker empty"))
        else:
            with st.spinner(LANG[lang]['fetching'].format(ticker=new_ticker)):
                success, message = fetch_stock_data(new_ticker)
                if success:
                    st.success(LANG[lang]['fetch_success'].format(ticker=new_ticker))
                    st.rerun()  # Rerun the app to update the file list
                else:
                    st.error(LANG[lang]['fetch_error'].format(msg=message))

    if selected_file:
        ticker = selected_file.replace("_5y.csv", "")
        st.header(LANG[lang]['analysis_for'].format(ticker=ticker))
        
        # Load data
        df = load_data(selected_file)
        
        # Display Key Statistics
        st.subheader(LANG[lang]['key_stats'])
        col1, col2, col3, col4 = st.columns(4)
        # use safer access in case of empty dataframe
        latest_close = f"{df['Close'].iloc[-1]:.2f}" if not df['Close'].empty else "N/A"
        highest = f"{df['High'].max():.2f}" if not df['High'].empty else "N/A"
        lowest = f"{df['Low'].min():.2f}" if not df['Low'].empty else "N/A"
        avg_vol = f"{int(df['Volume'].mean()):,}" if not df['Volume'].empty else "N/A"

        col1.metric(LANG[lang]['latest_close'], latest_close)
        col2.metric(LANG[lang]['highest_5y'], highest)
        col3.metric(LANG[lang]['lowest_5y'], lowest)
        col4.metric(LANG[lang]['avg_volume'], avg_vol)

        # Interactive Chart
        st.subheader(LANG[lang]['price_volume_chart'])
        
        # Trend Window Selector
        window_options = {
            LANG[lang]['window_1w']: 5,
            LANG[lang]['window_1m']: 20,
            LANG[lang]['window_1q']: 63
        }
        selected_window_label = st.selectbox(LANG[lang]['select_window'], list(window_options.keys()), index=2)
        window_size = window_options[selected_window_label]

        # Calculate Volume Spikes and Forward Returns
        df['Vol_MA'] = df['Volume'].rolling(window=20).mean()
        df['Spike'] = df['Volume'] > (2 * df['Vol_MA'])
        df['Next_Close'] = df['Close'].shift(-window_size) 
        df['Return'] = (df['Next_Close'] - df['Close']) / df['Close']

        # Create subplots with shared x-axis
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.03, subplot_titles=(f"{ticker} {LANG[lang]['price']}", LANG[lang]['volume']), 
                            row_width=[0.2, 0.7])

        # Candlestick
        fig.add_trace(go.Candlestick(x=df.index,
                                     open=df['Open'],
                                     high=df['High'],
                                     low=df['Low'],
                                     close=df['Close'], name="OHLC"), 
                      row=1, col=1)

        # Annotate Spikes and link them to volume bars
        spikes_df = df[df['Spike']].copy()

        if not spikes_df.empty:
            # Prepare colors/symbols and marker scatter on price chart
            marker_colors = []
            marker_symbols = []
            for ret in spikes_df['Return']:
                if pd.isna(ret):
                    marker_colors.append('gray')
                    marker_symbols.append('circle')
                elif ret > 0:
                    marker_colors.append('#00FF00')
                    marker_symbols.append('triangle-up')
                else:
                    marker_colors.append('#FF0000')
                    marker_symbols.append('triangle-down')

            # Price markers with hover (show volume and forward return)
            fig.add_trace(
                go.Scatter(
                    x=spikes_df.index,
                    y=spikes_df['High'] * 1.01,  # place marker slightly above high
                    mode='markers',
                    marker=dict(symbol=marker_symbols, color=marker_colors, size=12),
                    customdata=np.stack([spikes_df['Volume'].astype(int), spikes_df['Return']], axis=-1),
                    hovertemplate='Date: %{x|%Y-%m-%d}<br>Vol: %{customdata[0]:,}<br>Return: %{customdata[1]:.2%}<extra></extra>'
                ),
                row=1, col=1
            )

            # Vertical guideline linking price marker to volume bar across subplots
            for i, date in enumerate(spikes_df.index):
                fig.add_shape(
                    type='line',
                    x0=date,
                    x1=date,
                    xref='x',
                    y0=0,
                    y1=1,
                    yref='paper',
                    line=dict(color=marker_colors[i], width=1, dash='dot'),
                    opacity=0.6
                )

        # Volume with enhanced hover and marked spike bars
        bar_colors = ['red' if row['Open'] - row['Close'] >= 0 else 'green' for index, row in df.iterrows()]
        marker_line_colors = ['yellow' if s else 'rgba(0,0,0,0)' for s in df['Spike']]
        marker_line_widths = [1 if s else 0 for s in df['Spike']]

        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                marker_color=bar_colors,
                marker_line_color=marker_line_colors,
                marker_line_width=marker_line_widths,
                customdata=np.stack([df['Volume'].astype(int), df['Return'], df['Spike']], axis=-1),
                hovertemplate='Volume: %{customdata[0]:,}<br>Return: %{customdata[1]:.2%}<br>Spike: %{customdata[2]}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add Moving Average to Volume to show spikes better
        fig.add_trace(go.Scatter(x=df.index, y=df['Vol_MA'], line=dict(color='orange', width=1), name="Vol MA(20)"), 
                      row=2, col=1)

        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=700,
            showlegend=False,
            template="plotly_dark",
            margin=dict(l=50, r=50, t=50, b=50) # Optimized margins
        )
        
        st.plotly_chart(fig, width="stretch")

        # Checkbox to show raw data
        if st.checkbox(LANG[lang]['show_raw']):
            st.dataframe(df)

if __name__ == "__main__":
    main()

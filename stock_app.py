import streamlit as st
import pandas as pd
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

        # Annotate Spikes
        spike_dates = df[df['Spike']].index
        
        for date in spike_dates:
            # Check return
            ret = df.loc[date, 'Return']
            if pd.isna(ret):
                color = "gray"
                symbol = "circle" # Wait
                opacity = 0.5
            elif ret > 0:
                color = "#00FF00" # Green for rise
                symbol = "triangle-up"
                opacity = 1.0
            else:
                color = "#FF0000" # Red for fall
                symbol = "triangle-down"
                opacity = 1.0
                
            fig.add_annotation(
                x=date, 
                y=df.loc[date, 'High'],
                text="", # Removed "V" text to reduce clutter, symbol is enough
                showarrow=True,
                arrowhead=2, # Optimized arrow head
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=color,
                opacity=opacity,
                ax=0,
                ay=-25, # Slightly higher offset
                row=1, col=1
            )

        # Volume
        colors = ['red' if row['Open'] - row['Close'] >= 0 
                  else 'green' for index, row in df.iterrows()]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name=LANG[lang]['volume']), 
                      row=2, col=1)
        
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

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


def detect_spikes(df: pd.DataFrame, rolling_window: int = 20, multiplier: float = 2.0) -> pd.DataFrame:
    """Return a copy of df with 'Vol_MA' and 'Spike' columns calculated.

    Spike definition: Volume > multiplier * Vol_MA (strictly greater), and Vol_MA must not be NaN.
    """
    df = df.copy()
    df['Vol_MA'] = df['Volume'].rolling(window=rolling_window).mean()
    df['Spike'] = (df['Volume'] > (multiplier * df['Vol_MA'])) & df['Vol_MA'].notna()
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
            'spike_settings': "Spike Detection Settings",
            'rolling_window': "Rolling Window (days)",
            'threshold_multiplier': "Threshold Multiplier",
            'show_markers': "Show Spike Markers",
            'show_guidelines': "Show Guideline Lines",
            'show_legend': "Show Legend",
            'export_chart': "Export Chart",
            'export_png': "Download PNG",
            'export_html': "Download HTML",
            'spike_marker': "Spike Marker",
            'export_png_warning': "PNG export requires kaleido; providing HTML export instead.",
            'spike_help': "Spike detection uses a rolling volume average and marks bars where Volume > multiplier * Vol MA. Adjust the rolling window and multiplier to tune sensitivity.",
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
            'spike_settings': "峰值检测设置",
            'rolling_window': "滚动窗口（天）",
            'threshold_multiplier': "阈值乘数",
            'show_markers': "显示峰值标记",
            'show_guidelines': "显示连线",
            'show_legend': "显示图例",
            'export_chart': "导出图表",
            'export_png': "下载 PNG",
            'export_html': "下载 HTML",
            'spike_marker': "峰值标记",
            'export_png_warning': "PNG 导出需要 kaleido；已改为提供 HTML 导出。",
            'spike_help': "峰值检测使用成交量的滚动平均，峰值定义为成交量 > 阈值乘以滚动平均。通过调整滚动窗口和阈值乘数来调节灵敏度。",
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

    # Use project brand icon as page favicon and show it in the top-right of the header
    st.set_page_config(page_title=LANG[lang]['page_title'], layout="wide", page_icon="assets/taotaoapp.jpg")

    # Header layout: title on the left, brand icon on the right (272px)
    header_col, icon_col = st.columns([8, 1])
    with header_col:
        st.title(LANG[lang]['title'])
    with icon_col:
        st.image("assets/taotaoapp.jpg", width=272)


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

    # Spike detection settings and display controls
    st.sidebar.markdown("---")
    st.sidebar.subheader(LANG[lang]['spike_settings'])
    rolling_window = st.sidebar.slider(LANG[lang]['rolling_window'], min_value=5, max_value=100, value=20, step=1)
    threshold = st.sidebar.slider(LANG[lang]['threshold_multiplier'], min_value=1.0, max_value=5.0, value=2.0, step=0.1, format="%.1f")
    show_markers = st.sidebar.checkbox(LANG[lang]['show_markers'], value=True)
    show_guidelines = st.sidebar.checkbox(LANG[lang]['show_guidelines'], value=True)
    show_legend = st.sidebar.checkbox(LANG[lang]['show_legend'], value=False)
    st.sidebar.caption(LANG[lang]['spike_help'])
    st.sidebar.markdown("---")
    st.sidebar.subheader(LANG[lang]['export_chart'])

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
        df = detect_spikes(df, rolling_window=rolling_window, multiplier=threshold)
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

        # Annotate Spikes and link them to volume bars (conditional on user controls)
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
            if show_markers:
                fig.add_trace(
                    go.Scatter(
                        x=spikes_df.index,
                        y=spikes_df['High'] * 1.01,  # place marker slightly above high
                        mode='markers',
                        marker=dict(symbol=marker_symbols, color=marker_colors, size=12),
                        customdata=np.stack([spikes_df['Volume'].astype(int), spikes_df['Return']], axis=-1),
                        hovertemplate='Date: %{x|%Y-%m-%d}<br>Vol: %{customdata[0]:,}<br>Return: %{customdata[1]:.2%}<extra></extra>',
                        name=LANG[lang]['spike_marker']
                    ),
                    row=1, col=1
                )

            # Vertical guideline linking price marker to volume bar across subplots
            if show_guidelines:
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
                hovertemplate='Volume: %{customdata[0]:,}<br>Return: %{customdata[1]:.2%}<br>Spike: %{customdata[2]}<extra></extra>',
                name=LANG[lang]['volume']
            ),
            row=2, col=1
        )

        # Add Moving Average to Volume to show spikes better
        fig.add_trace(go.Scatter(x=df.index, y=df['Vol_MA'], line=dict(color='orange', width=1), name=f"Vol MA({rolling_window})"), 
                      row=2, col=1)

        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=700,
            showlegend=show_legend,
            template="plotly_dark",
            margin=dict(l=50, r=50, t=50, b=50) # Optimized margins
        )

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

        # Export buttons
        try:
            png_bytes = fig.to_image(format="png", scale=2)
            st.download_button(LANG[lang]['export_png'], data=png_bytes, file_name=f"{ticker}_annotated.png", mime="image/png")
        except Exception:
            st.warning(LANG[lang]['export_png_warning'])

        html = fig.to_html(include_plotlyjs='cdn')
        st.download_button(LANG[lang]['export_html'], data=html, file_name=f"{ticker}_annotated.html", mime="text/html")

        # Checkbox to show raw data
        if st.checkbox(LANG[lang]['show_raw']):
            st.dataframe(df)

if __name__ == "__main__":
    main()

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


def preload_default_data():
    """Preload default data for GOOG if it doesn't exist."""
    default_file = "GOOG_5y.csv"
    if not os.path.exists(default_file):
        try:
            stock = yf.Ticker("GOOG")
            df = stock.history(period="5y")
            if not df.empty:
                df.to_csv(default_file)
                print(f"Preloaded default data: {default_file} ({len(df)} rows)")
        except Exception as e:
            print(f"Could not preload default data: {e}")

def detect_spikes(df: pd.DataFrame, rolling_window: int = 20, multiplier: float = 2.0) -> pd.DataFrame:
    """Return a copy of df with 'Vol_MA' and 'Spike' columns calculated.

    Spike definition: Volume > multiplier * Vol_MA (strictly greater), and Vol_MA must not be NaN.
    """
    df = df.copy()
    df['Vol_MA'] = df['Volume'].rolling(window=rolling_window).mean()
    df['Spike'] = (df['Volume'] > (multiplier * df['Vol_MA'])) & df['Vol_MA'].notna()
    return df


def predict_next_price(df: pd.DataFrame, method: str = 'momentum', lookback: int = 20, horizon: int = 5):
    """Simple prediction methods returning predicted price and expected return."""
    prices = df['Close'].dropna()
    if prices.empty:
        return None, None

    if method == 'momentum':
        # mean return over lookback days applied to last price
        returns = prices.pct_change().dropna()
        recent = returns.tail(lookback)
        mu = recent.mean() if not recent.empty else 0.0
        pred = prices.iloc[-1] * (1 + mu * horizon)
        exp_ret = (pred - prices.iloc[-1]) / prices.iloc[-1]
        return float(pred), float(exp_ret)
    elif method == 'linear':
        # fit linear trend to last lookback points and extrapolate
        y = prices.tail(lookback).values
        if len(y) < 2:
            return None, None
        x = np.arange(len(y))
        m, b = np.polyfit(x, y, 1)
        pred = m * (len(y) - 1 + horizon) + b
        exp_ret = (pred - prices.iloc[-1]) / prices.iloc[-1]
        return float(pred), float(exp_ret)
    elif method == 'ema':
        # EMA drift: difference between last close and EMA, carry forward
        ema = prices.ewm(span=lookback, adjust=False).mean()
        last_ema = ema.iloc[-1]
        drift = prices.iloc[-1] - last_ema
        pred = prices.iloc[-1] + drift * horizon / max(1, lookback/5)
        exp_ret = (pred - prices.iloc[-1]) / prices.iloc[-1]
        return float(pred), float(exp_ret)
    else:
        return None, None


def backtest_prediction(df: pd.DataFrame, method: str, lookback: int, horizon: int):
    """Backtest prediction over historical series and return MAE and direction accuracy."""
    prices = df['Close'].dropna()
    preds = []
    trues = []
    for i in range(lookback, len(prices) - horizon):
        window = prices.iloc[i - lookback:i]
        future = prices.iloc[i + horizon]
        # create a temp df-like series
        temp_df = window.to_frame(name='Close')
        pred, _ = predict_next_price(temp_df, method=method, lookback=lookback, horizon=horizon)
        if pred is None:
            continue
        preds.append(pred)
        trues.append(future)
    if not preds:
        return None, None
    preds = np.array(preds)
    trues = np.array(trues)
    mae = float(np.mean(np.abs(preds - trues)))
    dir_acc = float(np.mean((preds - trues.mean()) * (trues - trues.mean()) > 0)) if len(preds) > 0 else 0.0
    # better direction accuracy: compare sign of (pred - last) and (true - last)
    signs = []
    for i, p in enumerate(preds):
        last = (i + lookback - 1)
        last_price = prices.iloc[last]
        signs.append((p - last_price) * (trues[i] - last_price) > 0)
    dir_acc2 = float(np.mean(signs)) if signs else 0.0
    return mae, dir_acc2

def analyze_volume_price_correlation(df: pd.DataFrame, rolling_window: int = 20, multiplier: float = 2.0, forecast_days: int = 5) -> dict:
    """Analyze historical correlation between volume spikes and price movements.

    Args:
        df: DataFrame with stock data
        rolling_window: Number of days for volume moving average
        multiplier: Threshold multiplier for spike detection
        forecast_days: Number of days to look ahead for price movement

    Returns:
        Dictionary with correlation statistics
    """
    df = df.copy()
    df = detect_spikes(df, rolling_window, multiplier)

    # Calculate future returns
    df['Future_Close'] = df['Close'].shift(-forecast_days)
    df['Return'] = (df['Future_Close'] - df['Close']) / df['Close']

    # Get all spikes
    spikes = df[df['Spike']].copy()

    if len(spikes) == 0:
        return {
            'total_spikes': 0,
            'positive_returns': 0,
            'negative_returns': 0,
            'avg_return_after_spike': 0,
            'success_rate': 0,
            'correlation_strength': 0
        }

    # Count positive and negative returns after spikes
    positive_returns = len(spikes[spikes['Return'] > 0])
    negative_returns = len(spikes[spikes['Return'] < 0])
    avg_return = spikes['Return'].mean()
    success_rate = (positive_returns / len(spikes)) * 100 if len(spikes) > 0 else 0

    # Calculate correlation strength (simple correlation coefficient)
    correlation_strength = df['Spike'].astype(int).corr(df['Return'].fillna(0)) if len(df.dropna()) > 1 else 0

    return {
        'total_spikes': len(spikes),
        'positive_returns': positive_returns,
        'negative_returns': negative_returns,
        'avg_return_after_spike': avg_return,
        'success_rate': success_rate,
        'correlation_strength': correlation_strength
    }


def predict_from_recent_spikes(df: pd.DataFrame, rolling_window: int = 20, multiplier: float = 2.0, forecast_days: int = 5) -> dict:
    """Predict future price movement based on recent volume spikes and historical correlation.

    Args:
        df: DataFrame with stock data
        rolling_window: Number of days for volume moving average
        multiplier: Threshold multiplier for spike detection
        forecast_days: Number of days to predict into the future

    Returns:
        Dictionary with prediction details
    """
    df = df.copy()

    # Analyze historical correlation
    correlation_stats = analyze_volume_price_correlation(df, rolling_window, multiplier, forecast_days)

    # Apply spike detection to the entire dataset to ensure proper Vol_MA calculation
    df_with_spikes = detect_spikes(df, rolling_window, multiplier)

    # Get recent spikes (last 10 days) from the dataset with spikes already calculated
    recent_df = df_with_spikes.tail(10).copy()
    recent_spikes = recent_df[recent_df['Spike']]

    if len(recent_spikes) == 0:
        return {
            'prediction': 'neutral',
            'confidence': 0,
            'direction': 0,
            'forecast_days': forecast_days,
            'message': 'No recent volume spikes detected',
            'correlation_stats': correlation_stats
        }

    # Use historical correlation to make prediction
    avg_return = correlation_stats['avg_return_after_spike']
    success_rate_up = (correlation_stats['positive_returns'] / correlation_stats['total_spikes']) * 100 if correlation_stats['total_spikes'] > 0 else 0
    success_rate_down = (correlation_stats['negative_returns'] / correlation_stats['total_spikes']) * 100 if correlation_stats['total_spikes'] > 0 else 0
    correlation_strength = correlation_stats['correlation_strength']

    # Determine prediction direction based on historical correlation between spikes and upward price trends
    if correlation_strength > 0.1:  # Positive correlation with upward movement
        prediction = 'up'
        direction = 1
    elif correlation_strength < -0.1:  # Negative correlation with upward movement (i.e., correlation with downward movement)
        prediction = 'down'
        direction = -1
    else:
        # Weak correlation — apply fallback rules using avg_return and success rate margins
        # Prefer average return if effect size (abs) exceeds min_effect
        if abs(avg_return) >= min_effect and len(recent_spikes) >= min_spikes:
            prediction = 'up' if avg_return > 0 else 'down'
            direction = 1 if avg_return > 0 else -1
        # Otherwise, check success rate margin
        elif (success_rate_up >= success_rate_down + success_margin) and len(recent_spikes) >= min_spikes:
            prediction = 'up'
            direction = 1
        elif (success_rate_down >= success_rate_up + success_margin) and len(recent_spikes) >= min_spikes:
            prediction = 'down'
            direction = -1
        else:
            prediction = 'neutral'
            direction = 0

    # Calculate confidence based on correlation strength and success rates
    # Higher weight to correlation strength as it's a more direct measure of relationship
    abs_corr_strength = min(abs(correlation_strength) * 100, 100)
    confidence = min((abs_corr_strength * 0.7 + success_rate_up * 0.3), 100)

    return {
        'prediction': prediction,
        'confidence': confidence,
        'direction': direction,
        'forecast_days': forecast_days,
        'message': f'Based on {len(recent_spikes)} recent volume spike(s) and historical correlation',
        'correlation_stats': correlation_stats,
        'avg_return': avg_return,
        'success_rate_up': success_rate_up,
        'success_rate_down': success_rate_down,
        'recent_spikes_count': len(recent_spikes),
        'min_effect_used': min_effect,
        'success_margin_used': success_margin,
        'min_spikes_used': min_spikes
    }

def main():
    # Preload default data for GOOG
    preload_default_data()

    # Basic i18n translations for English and Chinese
    LANG = {
        'en': {
            'page_title': "Stock Analysis App",
            'title': "Interactive Stock Analysis/Prediction",
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
            'download_csv': "Download CSV Data",
            'volume_spike_prediction': "Volume Spike Prediction",
            'historical_correlation': "Historical Correlation",
            'total_spikes_analyzed': "Total Spikes Analyzed",
            'avg_return_after_spike': "Average Return After Spike",
            'success_rate': "Success Rate",
            'correlation_strength': "Correlation Strength",
            'recent_spike_prediction': "Future Price Prediction",
            'prediction_direction': "Prediction Direction",
            'confidence_level': "Confidence Level",
            'forecast_period': "Forecast Period (Days)",
            'prediction_message': "Prediction Message",
            'up': "Up",
            'down': "Down",
            'neutral': "Neutral",
            'days': "days",
            'spike_marker': "Spike Marker",
            'export_png_warning': "PNG export requires kaleido; providing HTML export instead.",
            'spike_help': "Spike detection uses a rolling volume average and marks bars where Volume > multiplier * Vol MA. Adjust the rolling window and multiplier to tune sensitivity.",
            'min_effect': "Min effect size (avg return)",
            'success_margin': "Success rate margin (%)",
            'min_spikes': "Min recent spikes to act",
            'diagnostics': "Diagnostics",
            'recent_spikes_count': "Recent Spikes Count",
            'success_rate_up': "Success Rate Up",
            'success_rate_down': "Success Rate Down",
            'min_effect_help': "If correlation is weak, fall back to average return or success-rate rules when effect size or margin exceed these thresholds.",
            'prediction': "Prediction",
            'prediction_method': "Method",
            'prediction_lookback': "Lookback (days)",
            'prediction_horizon': "Horizon (days)",
            'method_momentum': "Momentum (mean returns)",
            'method_linear': "Linear trend",
            'method_ema': "EMA drift",
            'predicted_price': "Predicted Price",
            'predicted_return': "Predicted Return",
            'pred_mae': "Backtest MAE",
            'pred_dir_acc': "Direction Accuracy",
            'pred_help': "Simple short-term prediction methods (no financial advice). Use backtest metrics to gauge effectiveness.",
        },
        'zh': {
            'page_title': "股票分析应用",
            'title': "交互式股票分析/预测",
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
            'download_csv': "下载 CSV 数据",
            'volume_spike_prediction': "成交量峰值预测",
            'historical_correlation': "历史相关性",
            'total_spikes_analyzed': "分析的峰值总数",
            'avg_return_after_spike': "峰值后平均回报率",
            'success_rate': "成功率",
            'correlation_strength': "相关强度",
            'recent_spike_prediction': "未来价格预测",
            'prediction_direction': "预测方向",
            'confidence_level': "置信水平",
            'forecast_period': "预测期（天）",
            'prediction_message': "预测信息",
            'up': "上涨",
            'down': "下跌",
            'neutral': "中性",
            'days': "天",
            'spike_marker': "峰值标记",
            'export_png_warning': "PNG 导出需要 kaleido；已改为提供 HTML 导出。",
            'spike_help': "峰值检测使用成交量的滚动平均，峰值定义为成交量 > 阈值乘以滚动平均。通过调整滚动窗口和阈值乘数来调节灵敏度。",
            'prediction': "预测",
            'prediction_method': "方法",
            'prediction_lookback': "回溯窗口（天）",
            'prediction_horizon': "预测期（天）",
            'method_momentum': "动量（平均收益）",
            'method_linear': "线性趋势",
            'method_ema': "EMA 漂移",
            'predicted_price': "预测价格",
            'predicted_return': "预测收益",
            'pred_mae': "回测平均绝对误差",
            'pred_dir_acc': "方向准确率",
            'pred_help': "简单的短期预测方法（非投资建议）。使用回测指标评估性能。",
            'min_effect': "最小效应大小（平均收益）",
            'success_margin': "成功率差距（%）",
            'min_spikes': "最小近期峰值数量",
            'diagnostics': "诊断信息",
            'recent_spikes_count': "近期峰值数量",
            'success_rate_up': "上涨成功率",
            'success_rate_down': "下跌成功率",
            'min_effect_help': "若相关性薄弱，则当平均回报或成功率差距超过阈值时采用回退规则。",
        }
    }

    # Language selector - default to Chinese
    lang_choice = st.sidebar.selectbox(LANG['en']['language_label'], ["English", "中文"], index=1)
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
        st.image("assets/taotaoapp.jpg", width=72)


    # Sidebar for file selection
    st.sidebar.subheader(LANG[lang]['select_existing'])
    csv_files = glob.glob("*_5y.csv")

    # Set default file to GOOG_5y.csv if it exists
    default_file = "GOOG_5y.csv" if "GOOG_5y.csv" in csv_files else None
    default_index = csv_files.index(default_file) if default_file and default_file in csv_files else 0

    selected_file = st.sidebar.selectbox(LANG[lang]['choose_file'], csv_files, index=default_index) if csv_files else None

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

    # Fallback rule parameters for weak correlation
    st.sidebar.markdown("---")
    st.sidebar.subheader(LANG[lang]['diagnostics'])
    min_effect = st.sidebar.slider(LANG[lang]['min_effect'], min_value=0.0, max_value=0.05, value=0.005, step=0.001, format="%.3f")
    success_margin = st.sidebar.slider(LANG[lang]['success_margin'], min_value=0, max_value=50, value=10, step=1)
    min_spikes = st.sidebar.slider(LANG[lang]['min_spikes'], min_value=1, max_value=10, value=1, step=1)
    st.sidebar.caption(LANG[lang]['min_effect_help'])

    # Prediction settings
    st.sidebar.markdown("---")
    st.sidebar.subheader(LANG[lang]['volume_spike_prediction'])
    forecast_days = st.sidebar.slider(LANG[lang]['forecast_period'], min_value=1, max_value=30, value=5, step=1)

    st.sidebar.markdown("---")
    st.sidebar.subheader(LANG[lang]['export_chart'])

    # Prediction controls
    st.sidebar.markdown('---')
    st.sidebar.subheader(LANG[lang]['prediction'])
    st.sidebar.caption(LANG[lang]['pred_help'])
    pred_method = st.sidebar.selectbox(LANG[lang]['prediction_method'], [LANG[lang]['method_momentum'], LANG[lang]['method_linear'], LANG[lang]['method_ema']])
    pred_lookback = st.sidebar.slider(LANG[lang]['prediction_lookback'], min_value=3, max_value=252, value=20, step=1)
    pred_horizon = st.sidebar.slider(LANG[lang]['prediction_horizon'], min_value=1, max_value=20, value=5, step=1)

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

        # Volume Spike Prediction Section
        st.subheader(LANG[lang]['volume_spike_prediction'])

        # Analyze historical correlation
        correlation_data = analyze_volume_price_correlation(df, rolling_window, threshold, forecast_days)

        # Show historical correlation metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(LANG[lang]['total_spikes_analyzed'], correlation_data['total_spikes'])
        with col2:
            st.metric(LANG[lang]['avg_return_after_spike'], f"{correlation_data['avg_return_after_spike']:.2%}")
        with col3:
            st.metric(LANG[lang]['success_rate'], f"{correlation_data['success_rate']:.1f}%")

        # Show correlation strength
        st.progress(min(max(correlation_data['correlation_strength'], -1), 1) / 2 + 0.5)
        st.caption(f"{LANG[lang]['correlation_strength']}: {correlation_data['correlation_strength']:.3f}")

        # Make prediction based on recent spikes
        prediction_data = predict_from_recent_spikes(df, rolling_window, threshold, forecast_days)

        # Show prediction
        st.subheader(LANG[lang]['recent_spike_prediction'])
        pred_col1, pred_col2, pred_col3 = st.columns(3)

        with pred_col1:
            # Determine color based on prediction
            if prediction_data['prediction'] == 'up':
                pred_color = "green"
                pred_text = LANG[lang]['up']
            elif prediction_data['prediction'] == 'down':
                pred_color = "red"
                pred_text = LANG[lang]['down']
            else:
                pred_color = "gray"
                pred_text = LANG[lang]['neutral']

            st.metric(LANG[lang]['prediction_direction'], f":{pred_color}[{pred_text}]")
        with pred_col2:
            st.metric(LANG[lang]['confidence_level'], f"{prediction_data['confidence']:.1f}%")
        with pred_col3:
            st.metric(LANG[lang]['forecast_period'], f"{prediction_data['forecast_days']} {LANG[lang]['days']}")

        st.info(prediction_data['message'])

        # Diagnostics display
        st.subheader(LANG[lang]['diagnostics'])
        dcol1, dcol2, dcol3 = st.columns(3)
        dcol1.metric(LANG[lang]['recent_spikes_count'], prediction_data.get('recent_spikes_count', 0))
        dcol2.metric(LANG[lang]['success_rate_up'], f"{prediction_data.get('success_rate_up', 0):.1f}%")
        dcol3.metric(LANG[lang]['success_rate_down'], f"{prediction_data.get('success_rate_down', 0):.1f}%")
        st.caption(f"avg_return: {prediction_data.get('avg_return', 0):.3%} | corr: {prediction_data['correlation_stats'].get('correlation_strength', 0):.3f}")

        # Short-term numeric forecasts using selected method
        method_map = {
            LANG[lang]['method_momentum']: 'momentum',
            LANG[lang]['method_linear']: 'linear',
            LANG[lang]['method_ema']: 'ema'
        }
        selected_method = method_map.get(pred_method, 'momentum')
        pred_price, pred_ret = predict_next_price(df, method=selected_method, lookback=pred_lookback, horizon=pred_horizon)
        mae, dir_acc = backtest_prediction(df, method=selected_method, lookback=pred_lookback, horizon=pred_horizon)

        s_col1, s_col2, s_col3 = st.columns(3)
        with s_col1:
            s_col1.metric(LANG[lang]['predicted_price'], f"{pred_price:.2f}" if pred_price is not None else "N/A")
        with s_col2:
            s_col2.metric(LANG[lang]['predicted_return'], f"{pred_ret*100:.2f}%" if pred_ret is not None else "N/A")
        with s_col3:
            s_col3.metric(LANG[lang]['pred_mae'], f"{mae:.2f}" if mae is not None else "N/A")
            st.caption(f"{LANG[lang]['pred_dir_acc']}: {dir_acc:.2f}" if dir_acc is not None else "")

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
        
        # Add predicted price marker (if available)
        try:
            if 'pred_price' in locals() and pred_price is not None:
                pred_x = df.index[-1] + pd.Timedelta(days=pred_horizon)
                fig.add_trace(go.Scatter(x=[pred_x], y=[pred_price], mode='markers+text', marker=dict(color='cyan', size=14, symbol='diamond'), text=[f"{pred_price:.2f}"], textposition='top center', name='Prediction'), row=1, col=1)
        except Exception:
            pass

        st.plotly_chart(fig, width="stretch")

        # Export buttons
        # Create three columns for the download buttons
        col1, col2, col3 = st.columns(3)

        # PNG Download Button
        try:
            png_bytes = fig.to_image(format="png", scale=2)
            with col1:
                st.download_button(
                    LANG[lang]['export_png'],
                    data=png_bytes,
                    file_name=f"{ticker}_annotated.png",
                    mime="image/png"
                )
        except Exception:
            with col1:
                st.warning(LANG[lang]['export_png_warning'])

        # HTML Download Button
        html = fig.to_html(include_plotlyjs='cdn')
        with col2:
            st.download_button(
                LANG[lang]['export_html'],
                data=html,
                file_name=f"{ticker}_annotated.html",
                mime="text/html"
            )

        # CSV Download Button
        csv = df.reset_index().to_csv(index=False)
        with col3:
            st.download_button(
                label=LANG[lang]['download_csv'],
                data=csv,
                file_name=f"{ticker}_5y.csv",
                mime='text/csv',
            )

        # Checkbox to show raw data
        if st.checkbox(LANG[lang]['show_raw']):
            st.dataframe(df)

if __name__ == "__main__":
    main()

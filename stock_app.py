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
    st.set_page_config(page_title="Stock Analysis App", layout="wide")
    st.title("Interactive Stock Analysis")

    # Sidebar for file selection
    st.sidebar.subheader("Select Existing Data")
    csv_files = glob.glob("*_5y.csv")
    selected_file = st.sidebar.selectbox("Choose a file:", csv_files) if csv_files else None

    # Sidebar for new data fetch
    st.sidebar.markdown("---")
    st.sidebar.subheader("Fetch New Data")
    new_ticker = st.sidebar.text_input("Enter Ticker Symbol (e.g., TSLA, AAPL):").upper()
    
    if st.sidebar.button("Fetch Data"):
        with st.spinner(f"Fetching data for {new_ticker}..."):
            success, message = fetch_stock_data(new_ticker)
            if success:
                st.success(f"Successfully fetched details for {new_ticker}!")
                st.rerun()  # Rerun the app to update the file list
            else:
                st.error(f"Error: {message}")

    if selected_file:
        ticker = selected_file.replace("_5y.csv", "")
        st.header(f"Analysis for {ticker}")
        
        # Load data
        df = load_data(selected_file)
        
        # Display Key Statistics
        st.subheader("Key Statistics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Latest Close", f"{df['Close'].iloc[-1]:.2f}")
        col2.metric("Highest (5y)", f"{df['High'].max():.2f}")
        col3.metric("Lowest (5y)", f"{df['Low'].min():.2f}")
        col4.metric("Avg Volume", f"{int(df['Volume'].mean()):,}")

        # Interactive Chart
        st.subheader("Price and Volume Chart")
        
        # Trend Window Selector
        window_options = {
            "1 Week (5 Days)": 5,
            "1 Month (20 Days)": 20,
            "1 Quarter (63 Days)": 63
        }
        selected_window_label = st.selectbox("Select Trend Observation Window:", list(window_options.keys()), index=2)
        window_size = window_options[selected_window_label]

        # Calculate Volume Spikes and Forward Returns
        df['Vol_MA'] = df['Volume'].rolling(window=20).mean()
        df['Spike'] = df['Volume'] > (2 * df['Vol_MA'])
        df['Next_Close'] = df['Close'].shift(-window_size) 
        df['Return'] = (df['Next_Close'] - df['Close']) / df['Close']

        # Create subplots with shared x-axis
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.03, subplot_titles=(f'{ticker} Price', 'Volume'), 
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
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name="Volume"), 
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
        
        st.plotly_chart(fig, use_container_width=True)

        # Checkbox to show raw data
        if st.checkbox("Show Raw Data"):
            st.dataframe(df)

if __name__ == "__main__":
    main()

import yfinance as yf
import pandas as pd
import os

def fetch_data():
    tickers = ["BATS.L", "INTC"]
    period = "5y"
    
    print(f"Fetching data for {tickers} over last {period}...")
    
    for ticker in tickers:
        try:
            # enable auto_adjust=True to get Adjusted Close prices which is usually better for analysis
            # usually user wants Open, High, Low, Close, Volume.
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            
            if df.empty:
                print(f"Warning: No data found for {ticker}")
                continue
                
            filename = f"{ticker}_5y.csv"
            df.to_csv(filename)
            print(f"Successfully saved {filename} ({len(df)} rows)")
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

if __name__ == "__main__":
    fetch_data()

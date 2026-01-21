import mplfinance as mpf
import pandas as pd
import os
import glob

def generate_charts():
    # Find all csv files ending with _5y.csv
    csv_files = glob.glob("*_5y.csv")
    
    if not csv_files:
        print("No stock data files found.")
        return

    print(f"Found {len(csv_files)} files: {csv_files}")

    for file in csv_files:
        try:
            print(f"Processing {file}...")
            # Read CSV and set index to DatetimeIndex
            df = pd.read_csv(file)
            df['Date'] = pd.to_datetime(df['Date'], utc=True)
            df.set_index('Date', inplace=True)
            
            # mplfinance expects index to be DatetimeIndex
            # It also expects columns "Open", "High", "Low", "Close", "Volume"
            # Our CSVs match this format.
            
            ticker_name = file.replace("_5y.csv", "")
            output_file = f"{ticker_name}_chart.png"
            
            # Create a plot with volume
            # type='candle' for candlestick chart
            # volume=True to show volume
            # style='yahoo' for familiar colors
            # title included to identify the stock
            # savefig to save to a file
            
            mpf.plot(
                df, 
                type='candle', 
                volume=True, 
                style='yahoo', 
                title=f"{ticker_name} - 5 Year History",
                ylabel='Price',
                ylabel_lower='Volume',
                savefig=output_file
            )
            
            print(f"Generated chart: {output_file}")
            
        except Exception as e:
            print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    generate_charts()

import pandas as pd
import glob
import numpy as np

def analyze_volume_patterns():
    csv_files = glob.glob("*_5y.csv")
    if not csv_files:
        print("No stock data files found.")
        return

    print(f"Analyzing {len(csv_files)} stock files for volume patterns...")
    print("-" * 60)

    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # Handle potential unnamed index col if read_csv doesn't pick it up automatically
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], utc=True)
                df.set_index('Date', inplace=True)
            
            # Key Logic: Define "Volume Spike"
            # We use 20-day Moving Average of Volume
            df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
            
            # Spike = Volume > 2 * Vol_MA20 (Strict definition)
            df['Spike'] = df['Volume'] > (2 * df['Vol_MA20'])
            
            # Calculate Future Returns (N days later)
            # We want to see what happens 5 trading days after the spike
            N = 5
            df['Future_Close'] = df['Close'].shift(-N)
            df['Return_5d'] = (df['Future_Close'] - df['Close']) / df['Close']
            
            # Filter for Spikes
            spikes = df[df['Spike']].copy()
            
            if len(spikes) == 0:
                print(f"No volume spikes found for {file}")
                continue

            # Hypothesis Check: "Share price intend to rise afterward"
            # We count how many times Return_5d > 0
            
            rise_count = len(spikes[spikes['Return_5d'] > 0])
            fall_count = len(spikes[spikes['Return_5d'] < 0])
            total_spikes = len(spikes)
            
            rise_prob = (rise_count / total_spikes) * 100
            
            print(f"Stock: {file.replace('_5y.csv', '')}")
            print(f"  Total Spikes Detected: {total_spikes}")
            print(f"  Cases where Price Rose after 5 days: {rise_count} ({rise_prob:.1f}%)")
            print(f"  Cases where Price Fell after 5 days: {fall_count}")
            
            avg_return = spikes['Return_5d'].mean() * 100
            print(f"  Average 5-day Return after Spike: {avg_return:.2f}%")
            
            # Find strongest negative signals (Volume Spike -> Big Drop)
            worst_drops = spikes.sort_values(by='Return_5d').head(3)
            print("  Top 3 Volume Spikes followed by biggest drops:")
            for date, row in worst_drops.iterrows():
                print(f"    Date: {date.date()} | Vol: {int(row['Volume']):,} | 5d Return: {row['Return_5d']*100:.2f}%")
            print("-" * 60)
            
        except Exception as e:
            print(f"Error analyzing {file}: {e}")

if __name__ == "__main__":
    analyze_volume_patterns()

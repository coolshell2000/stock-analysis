import sys
import os
import pandas as pd
import numpy as np
# Ensure project root is in path so tests can import stock_app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stock_app import detect_spikes


def make_df(vols):
    dates = pd.date_range("2020-01-01", periods=len(vols), freq='D')
    df = pd.DataFrame({
        'Date': dates,
        'Open': np.ones(len(vols)) * 10,
        'High': np.ones(len(vols)) * 11,
        'Low': np.ones(len(vols)) * 9,
        'Close': np.ones(len(vols)) * 10,
        'Volume': vols
    })
    df.set_index('Date', inplace=True)
    return df


def test_spike_detects_obvious_spike():
    vols = [100, 110, 120, 500, 130]
    df = make_df(vols)
    out = detect_spikes(df, rolling_window=3, multiplier=2.0)
    # only the 500 should be a spike (index 3)
    spikes = out['Spike'].values
    assert spikes.tolist() == [False, False, False, True, False]
    assert not np.isnan(out['Vol_MA'].iloc[3])


def test_no_spikes_when_window_too_large():
    vols = [100, 110, 120]
    df = make_df(vols)
    out = detect_spikes(df, rolling_window=10, multiplier=1.5)
    assert out['Spike'].sum() == 0


def test_strict_greater_condition():
    vols = [100, 100, 100, 200]
    df = make_df(vols)
    # rolling window 3: Vol_MA at index 3 is mean(100,100,100,200)? since current included -> 125
    # multiplier 1.6 -> 200 > 1.6*125 = 200 -> strict > means not a spike
    out = detect_spikes(df, rolling_window=3, multiplier=1.6)
    assert bool(out['Spike'].iloc[-1]) is False

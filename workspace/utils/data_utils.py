import os
from pathlib import Path

import pandas as pd

HOME_PATH = os.environ['HOME']
DATA_PATH = "/home/data/data/time_series/market_data/"

def get_files(dt_start, dt_end, data_type, exchange, instrument_type, symbol):
    """Load time series data files for a given date range and parameters"""
    span_days = pd.date_range(dt_start, dt_end, freq="D").strftime("%Y%m%d")
    files_for_range = []
    for day in span_days:
        file_path = Path(os.path.join(DATA_PATH, exchange, instrument_type, "latest", data_type + "_1min", day,
                                      symbol + "." + day + "." + data_type + ".1min.csv.gz"))

        if not file_path.exists():
            print(f"[FILE NOT FOUND] {file_path}")
            continue
        file_for_day = pd.read_csv(file_path)
        files_for_range.append(file_for_day)
    data = pd.concat(files_for_range)
    return data

def align_ts(*dfs):
    """Align multiple time series dataframes to the common time range"""
    start = max(df.index.min() for df in dfs)
    end   = min(df.index.max() for df in dfs)
    return (df.loc[start:end] for df in dfs), start, end
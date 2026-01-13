import os
from pathlib import Path

import pandas as pd
import numpy as np

HOME_PATH = os.environ['HOME']
DATA_PATH = "/home/data/data/time_series/market_data/"

def get_files(dt_start, dt_end, data_type, exchange, instrument_type, symbol, DATA_PATH):
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


def analyze_carryover(*dfs, names=None):
    """
    Analyze carryover flag distribution across dataframes.
    """
    if names is None:
        names = [f"df_{i}" for i in range(len(dfs))]
    
    stats = {}
    for name, df in zip(names, dfs):
        if "carryover" not in df.columns:
            stats[name] = {"status": "No carryover column"}
            continue
        
        total_rows = len(df)
        missing_original = (df["carryover"] != 0).sum()
        completely_missing = (df["carryover"] == -1).sum()
        carryover_counts = df["carryover"].value_counts().sort_index().to_dict()
        
        stats[name] = {
            "total_rows": total_rows,
            "missing_original": missing_original,
            "missing_original_pct": 100 * missing_original / total_rows,
            "completely_missing": completely_missing,
            "completely_missing_pct": 100 * completely_missing / total_rows,
            "carryover_counts": carryover_counts
        }
    
    return stats


def clean_carryover(*dfs, carryover_weight=0.5, remove_completely_missing=True):
    """
    Clean carryover data by filtering and tagging.
    Remove if missing, add is_carryover and give it less weights
    """
    cleaned_dfs = []
    
    for df in dfs:
        df = df.copy()
        
        if "carryover" not in df.columns:
            # No carryover column, add defaults
            df["is_carryover"] = 0
            df["sample_weight"] = 1.0
            cleaned_dfs.append(df)
            continue
        
        # Remove completely missing data
        if remove_completely_missing:
            df = df[df["carryover"] != -1]
        
        # Create quality flags and weights
        df["is_carryover"] = (df["carryover"] != 0).astype(int)
        df["sample_weight"] = np.where(df["is_carryover"], carryover_weight, 1.0)
        
        cleaned_dfs.append(df)
    
    return tuple(cleaned_dfs) if len(cleaned_dfs) > 1 else cleaned_dfs[0]
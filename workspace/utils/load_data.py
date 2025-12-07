import os
from pathlib import Path

import pandas as pd

HOME_PATH = os.environ['HOME']

# put your data in this folder or change forder name accordingly.
DATA_PATH = os.path.join(HOME_PATH, 'data', 'fordham')
# DATA_PATH = "/home/data/data/time_series/market_data/"

# this is where your output get written
# OUTPUT_PATH = os.path.join(HOME_PATH, 'research', 'fordham')
OUTPUT_PATH = os.path.join(HOME_PATH, 'volatility_prediction', 'feature_study')
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
    print(f"Created directory: {OUTPUT_PATH}")

# data_type could be: trade, level1, book, openinterest, refer to the file names in the data folder
def get_files(dt_start, dt_end, data_type, exchange, instrument_type, symbol):
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

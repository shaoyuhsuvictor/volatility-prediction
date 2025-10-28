import datetime as dt
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
import scipy.stats as stats

HOME_PATH = os.environ['HOME']

# put your data in this folder or change forder name accordingly.
# DATA_PATH = os.path.join(HOME_PATH, 'data', 'fordham')
DATA_PATH = "/home/data/data/time_series/market_data/"

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


def main():
    exchange = "binance"

    # start_date = "20250201"
    # end_date = "20250301"
    start_date = "20240201"
    end_date = "20240301"

    start_dt = dt.datetime.strptime(start_date, "%Y%m%d")
    end_dt = dt.datetime.strptime(end_date, "%Y%m%d")

    level1_data = get_files(start_dt, end_dt, "level1", exchange, "futures", "BTCUSDT")
    trade_data = get_files(start_dt, end_dt, "trade", exchange, "futures", "BTCUSDT")
    book_data = get_files(start_dt, end_dt, "book", exchange, "futures", "BTCUSDT")
    oi_data = get_files(start_dt, end_dt, "openinterest", exchange, "futures", "BTCUSDT")

    # rename to avoid automated renaming on merge
    trade_data = trade_data.rename(columns={"bin_id": "bin_id_trade", "carryover": "carryover_trade", "ramp_up": "ramp_up_trade"})
    book_data = book_data.rename(columns={"bin_id": "bin_id_book", "carryover": "carryover_book", "ramp_up": "ramp_up_book"})

    print("Read ", len(level1_data), len(trade_data), len(book_data), " rows")
    print("Level1 Keys:", level1_data.keys())
    print("Trade Keys:", trade_data.keys())
    print("Book Keys:", book_data.keys())

    data = level1_data.merge(trade_data, how='left', left_on='ts_end', right_on='ts_end')
    data = data.merge(book_data, how='left', left_on='ts_end', right_on='ts_end')
    print("Data Keys:", data.keys())

    # Plot midprice over time
    plt.figure(figsize=(20, 5))
    plt.plot(data["ts_end"], data["close_mid"])

    plt.xlabel("ts")
    plt.ylabel("price")
    plt.title("Mid Price")
    plt.grid(True)

    file_path = os.path.join(OUTPUT_PATH, 'close_mid.png')
    plt.savefig(file_path)
    print("Plot saved to ", file_path)
    plt.close()

    # calculating mid to mid return
    data["ret"] = ((data["close_mid"] - data["close_mid"].shift(1)) / data["close_mid"].shift(1)).ffill().fillna(0)
    data["future_ret_1"] = ((data["close_mid"].shift(-1) - data["close_mid"]) / data["close_mid"]).ffill().fillna(0)
    data["future_ret_5"] = ((data["close_mid"].shift(-5) - data["close_mid"]) / data["close_mid"]).ffill().fillna(0)

    # delay by 1 bin with respect to the signal. (Give strategy 1 min to enter position)
    data["future_ret_5_1"] = ((data["close_mid"].shift(-5) - data["close_mid"].shift(-1)) / data["close_mid"].shift(-1)).ffill().fillna(0)

    # return ema features
    data["ret_ema_5"] = (((data["close_mid"] - data["close_mid"].shift(5)) / data["close_mid"]).shift(5)).ffill().fillna(0).ewm(halflife=5,
                                                                                                                                adjust=False).mean()
    data["ret_ema_30"] = (((data["close_mid"] - data["close_mid"].shift(30)) / data["close_mid"]).shift(30)).ffill().fillna(0).ewm(halflife=30,
                                                                                                                                   adjust=False).mean()
    data["ret_ema_120"] = (((data["close_mid"] - data["close_mid"].shift(120)) / data["close_mid"]).shift(120)).ffill().fillna(0).ewm(halflife=120,
                                                                                                                                      adjust=False).mean()

    # trade imbalance features
    data["trade_imb_1"] = ((data["buy_volume"] - data["sell_volume"]).ewm(halflife=1, adjust=False).mean() /
                           (data["buy_volume"] + data["sell_volume"]).ewm(halflife=1, adjust=False).mean())
    data["trade_imb_10"] = ((data["buy_volume"] - data["sell_volume"]).ewm(halflife=10, adjust=False).mean() /
                            (data["buy_volume"] + data["sell_volume"]).ewm(halflife=10, adjust=False).mean())
    data["trade_imb_100"] = ((data["buy_volume"] - data["sell_volume"]).ewm(halflife=100, adjust=False).mean() /
                             (data["buy_volume"] + data["sell_volume"]).ewm(halflife=100, adjust=False).mean())

    # book imbalance features
    data["book_imb_10bps_10"] = ((data["bid_10bps_fill_size"] - data["ask_10bps_fill_size"]).ewm(halflife=10, adjust=False).mean() /
                                 (data["bid_10bps_fill_size"] + data["ask_10bps_fill_size"]).ewm(halflife=10, adjust=False).mean())
    data["book_imb_10bps_100"] = ((data["bid_10bps_fill_size"] - data["ask_10bps_fill_size"]).ewm(halflife=100, adjust=False).mean() /
                                  (data["bid_10bps_fill_size"] + data["ask_10bps_fill_size"]).ewm(halflife=100, adjust=False).mean())
    data["book_imb_100bps_100"] = ((data["bid_100bps_fill_size"] - data["ask_100bps_fill_size"]).ewm(halflife=100, adjust=False).mean() /
                                   (data["bid_100bps_fill_size"] + data["ask_100bps_fill_size"]).ewm(halflife=100, adjust=False).mean())

    # Print statistics
    print(f"Ret EMA 120min Mean: {data["ret_ema_120"].mean():.4f}")
    print(f"Ret EMA 120min STD: {data["ret_ema_120"].std():.4f}")
    print(f"Ret EMA 120min Skew: {stats.skew(data["ret_ema_120"]):.4f}")
    print(f"Ret EMA 120min Kurt: {stats.kurtosis(data["ret_ema_120"]):.4f}")

    #####################################################
    # Return EMA Histogram
    ######################################################
    plt.figure(figsize=(10, 10))
    plt.hist(data["ret"], bins=100, alpha=0.5, label="ret")
    plt.hist(data["ret_ema_5"], bins=100, alpha=0.5, label="ret_ema_5")
    plt.hist(data["ret_ema_30"], bins=100, alpha=0.5, label="ret_ema_30")
    plt.hist(data["ret_ema_120"], bins=100, alpha=0.5, label="ret_ema_120")

    plt.yscale("log")  # Set y-axis to logarithmic scale
    plt.xlabel("TradeImb")
    plt.ylabel("Frequency (Log Scale)")
    plt.title("Histogram of Price Returns")
    plt.grid(True)
    plt.legend()

    file_path = os.path.join(OUTPUT_PATH, 'mid_return_hist.png')
    plt.savefig(file_path)
    print("Plot saved to ", file_path)
    plt.close()

    ###############################################
    # Return EMA Time series
    ###############################################

    plt.figure(figsize=(20, 10))
    plt.plot(data["ret_ema_5"], label="ret_ema_5")
    plt.plot(data["ret_ema_30"], label="ret_ema_30")
    plt.plot(data["ret_ema_120"], label="ret_ema_120")

    plt.xlabel("Entry")
    plt.ylabel("Return EMA")
    plt.title("Return EMAs")
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 10000)

    file_path = os.path.join(OUTPUT_PATH, 'return_ema_plt.png')
    plt.savefig(file_path)
    print("Plot saved to ", file_path)
    plt.close()

    ###############################################
    # Trade Imbalance Histogram
    ###############################################

    plt.figure(figsize=(10, 10))
    plt.hist(data["trade_imb_1"], bins=100, alpha=0.5, label="trade_imb_1")
    plt.hist(data["trade_imb_10"], bins=100, alpha=0.5, label="trade_imb_10")
    plt.hist(data["trade_imb_100"], bins=100, alpha=0.5, label="trade_imb_100")

    plt.yscale("log")  # Set y-axis to logarithmic scale
    plt.xlabel("Trade Imbalance")
    plt.ylabel("Frequency (Log Scale)")
    plt.title("Histogram of Trade Imbalance EMAs")
    plt.grid(True)
    plt.legend()

    file_path = os.path.join(OUTPUT_PATH, 'trade_imbalance_hist.png')
    plt.savefig(file_path)
    print("Plot saved to ", file_path)
    plt.close()

    ###############################################
    # Trade Imbalance Time Series
    ###############################################

    plt.figure(figsize=(20, 10))
    plt.plot(data["trade_imb_1"], label="trade_imb_1")
    plt.plot(data["trade_imb_10"], label="trade_imb_10")
    plt.plot(data["trade_imb_100"], label="trade_imb_100")

    plt.xlabel("Entry")
    plt.ylabel("Trade Imbalance")
    plt.title("Trade Imbalance EMAs")
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 2000)

    file_path = os.path.join(OUTPUT_PATH, 'trade_imbalance_plt.png')
    plt.savefig(file_path)
    print("Plot saved to ", file_path)
    plt.close()

    ###############################################
    # Book Imbalance Histogram
    ###############################################

    plt.figure(figsize=(10, 10))
    plt.hist(data["book_imb_10bps_10"], bins=100, alpha=0.5, label="book_imb_10bps_10")
    plt.hist(data["book_imb_10bps_100"], bins=100, alpha=0.5, label="book_imb_10bps_100")
    plt.hist(data["book_imb_100bps_100"], bins=100, alpha=0.5, label="book_imb_100bps_100")

    plt.yscale("log")  # Set y-axis to logarithmic scale
    plt.xlabel("Book Imbalance")
    plt.ylabel("Frequency (Log Scale)")
    plt.title("Histogram of Book Imbalance EMAs")
    plt.grid(True)
    plt.legend()

    file_path = os.path.join(OUTPUT_PATH, 'book_imbalance_hist.png')
    plt.savefig(file_path)
    print("Plot saved to ", file_path)
    plt.close()

    ###############################################
    # Book Imbalance Time Series
    ###############################################
    plt.figure(figsize=(20, 10))
    plt.plot(data["book_imb_10bps_10"], label="book_imb_10bps_10")
    plt.plot(data["book_imb_10bps_100"], label="book_imb_10bps_100")
    plt.plot(data["book_imb_100bps_100"], label="book_imb_100bps_100")

    plt.xlabel("Entry")
    plt.ylabel("Book Imbalance")
    plt.title("Book Imbalance EMAs")
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 2000)

    file_path = os.path.join(OUTPUT_PATH, 'book_imbalance_plt.png')
    plt.savefig(file_path)
    print("Plot saved to ", file_path)
    plt.close()

    ####################################################
    # Correlation matrix
    #####################################################
    columns = ["ret_ema_5", "ret_ema_30", "ret_ema_120", "trade_imb_1", "trade_imb_10", "trade_imb_100", "future_ret_1", "future_ret_5",
               "future_ret_5_1"]
    data_subset = data[columns].dropna()

    # Calculate the Pearson correlation matrix
    correlation_matrix = data_subset.corr(method="pearson") * 100

    # Create heatmap with color scale
    plt.figure(figsize=(16, 12))
    plt.rcParams.update({'font.size': 14})  # Adjust the value as needed
    sns.heatmap(
        correlation_matrix,
        annot=True,  # Show correlation values
        cmap="coolwarm",  # Diverging color scale (blue for -1, red for +1)
        vmin=-100, vmax=100,  # Set range for correlations
        center=0,  # Center the color scale at 0
        fmt=".2f"  # Format numbers to 4 decimal places
    )
    plt.title("Correlation Matrix Heatmap")
    file_path = os.path.join(OUTPUT_PATH, 'correlation_matrix.png')
    plt.savefig(file_path)
    print("Plot saved to ", file_path)
    plt.close()
    print("Cross-Correlation Matrix:")
    print(correlation_matrix)

    ######################################################
    # Create autocorrelation plot
    ######################################################
    plt.figure(figsize=(10, 6))
    plot_acf(data["ret"], lags=20, title="Autocorrelation Plot of future_ret_1")
    # plot_acf(data["future_ret_5"], lags=20, title="Autocorrelation Plot of future_ret_5")

    # Customize plot
    plt.title("Autocorrelation Plot of future_ret_1", fontsize=16)
    plt.xlabel("Lag", fontsize=14)
    plt.ylabel("Autocorrelation", fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    # Set y-axis limits
    plt.ylim(-0.1, 1.0)
    plt.grid(True)

    file_path = os.path.join(OUTPUT_PATH, 'autocorrelation.png')
    plt.savefig(file_path)
    print("Plot saved to ", file_path)
    plt.close()

    ######################################################
    # backtest trade imbalance
    ######################################################
    plt.figure(figsize=(10, 6))

    data_sorted = data.sort_values("trade_imb_10").reset_index(drop=True)
    plt.plot(np.cumsum(data["trade_imb_10"] * data["future_ret_1"]), label="future_ret_1")
    plt.plot(np.cumsum(data["trade_imb_10"] * data["future_ret_5"]), label="future_ret_5")
    plt.plot(np.cumsum(data["trade_imb_10"] * data["future_ret_5_1"]), label="future_ret_5_1")

    # Customize plot
    plt.title("trade_imb_10 signal test", fontsize=16)
    plt.xlabel("time", fontsize=14)
    plt.ylabel("Cumulative return", fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    # Set y-axis limits
    # plt.ylim(-0.1, 1.0)
    plt.grid(True)
    plt.legend()

    file_path = os.path.join(OUTPUT_PATH, 'trade_imb_10_signal_backtest.png')
    plt.savefig(file_path)
    print("Plot saved to ", file_path)
    plt.close()

    ######################################################
    # backtest trade imbalance with filter
    ######################################################
    plt.figure(figsize=(10, 6))

    data_sorted = data[np.abs(data["trade_imb_10"]) > 0.1]
    plt.plot(np.cumsum(-np.sign(data["trade_imb_10"]) * data["future_ret_5_1"]), label="not filtered")
    plt.plot(np.cumsum(-np.sign(data_sorted["trade_imb_10"]) * data_sorted["future_ret_5_1"]), label="filtered")

    print("Lenght:", len(data), len(data_sorted))
    plt.title("trade_imb_10 signal test", fontsize=16)
    plt.xlabel("time", fontsize=14)
    plt.ylabel("Cumulative return", fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    # plt.ylim(-0.1, 1.0)
    plt.grid(True)
    plt.legend()

    file_path = os.path.join(OUTPUT_PATH, 'trade_imb_10_signal_backtest_filtered.png')
    plt.savefig(file_path)
    print("Plot saved to ", file_path)
    plt.close()

    ######################################################
    # Sharpe Demo Filtered vs unfiltered pnl
    ######################################################
    plt.figure(figsize=(10, 6))

    # data_sorted = data.sort_values("trade_imb_1").reset_index(drop=True)
    data_sorted = data[np.abs(data["trade_imb_10"]) > 0.1]
    data_sorted = data_sorted[np.abs(data_sorted["trade_imb_10"]) < 0.5]
    data["pnl"] = -data["trade_imb_10"] * data["future_ret_5_1"]
    data_sorted["pnl"] = -data_sorted["trade_imb_10"] * data_sorted["future_ret_5_1"]

    print("Count:", len(data))
    print("MeanPnl:", round(data["future_ret_5_1"].mean() * 10000, 3))
    print("Mean:", round((data["pnl"] * 10000).mean(), 3))
    print("STDev:", round((data["pnl"] * 10000).std(), 3))
    sharpe_unfiltered = round(data["pnl"].mean() / data["pnl"].std() * np.sqrt(len(data)), 3)
    print("Sharpe:", sharpe_unfiltered)

    print("Count:", len(data_sorted))
    print("MeanPnl:", round(data_sorted["future_ret_5_1"].mean() * 10000, 3))
    print("Mean:", round((data_sorted["pnl"] * 10000).mean(), 3))
    print("STDev:", round((data_sorted["pnl"] * 10000).std(), 3))
    sharpe_filtered = round(data_sorted["pnl"].mean() / data_sorted["pnl"].std() * np.sqrt(len(data_sorted)), 3)
    print("Sharpe:", sharpe_filtered)

    plt.hist(data["pnl"], bins=1000, alpha=0.5, label="unfiltered, sharpe="+str(sharpe_unfiltered) )
    plt.hist(data_sorted["pnl"], bins=1000, alpha=0.5, label="filtered, sharpe="+ str(sharpe_filtered))
    # Customize plot
    plt.title("trade_imb_10 signal pnl", fontsize=16)
    plt.xlabel("return", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    plt.xlim(-0.002, 0.002)
    plt.yscale("log")  # Set y-axis to logarithmic scale
    plt.grid(True)
    plt.legend()


    file_path = os.path.join(OUTPUT_PATH, 'trade_imb_10_sharpe.png')
    plt.savefig(file_path)
    print("Plot saved to ", file_path)
    plt.close()

    ######################################################
    # scatter
    ######################################################
    plt.figure(figsize=(10, 10))
    data_sorted = data.sort_values("trade_imb_1").reset_index(drop=True)

    plt.scatter(data["trade_imb_1"], data["future_ret_1"])

    # Customize plot
    plt.title("returns sorted by trade_imb_10", fontsize=16)
    plt.xlabel("trade imbalance", fontsize=14)
    plt.ylabel("return", fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    # Set y-axis limits
    plt.ylim(-0.005, 0.005)
    plt.grid(True)

    file_path = os.path.join(OUTPUT_PATH, 'trade_imb_1_scatter.png')
    plt.savefig(file_path)
    print("Plot saved to ", file_path)
    plt.close()

    ######################################################
    # backtest
    ######################################################
    plt.figure(figsize=(10, 6))

    data_sorted = data.sort_values("trade_imb_10").reset_index(drop=True)
    plt.plot(data_sorted["trade_imb_10"], np.cumsum(data_sorted["future_ret_1"]), label="future_ret_1")
    plt.plot(data_sorted["trade_imb_10"], np.cumsum(data_sorted["future_ret_5"]), label="future_ret_5")
    plt.plot(data_sorted["trade_imb_10"], np.cumsum(data_sorted["future_ret_5_1"]), label="future_ret_5_1")
    # plt.plot(data_sorted["trade_imb_10"],  np.cumsum(-data_sorted["trade_imb_10"] *data_sorted["future_ret_1"]), label = "future_ret_1")
    # plt.plot(data_sorted["trade_imb_10"], np.cumsum(-data_sorted["trade_imb_10"] * data_sorted["future_ret_5"]), label = "future_ret_5")
    # plt.plot(data_sorted["trade_imb_10"], np.cumsum(-data_sorted["trade_imb_10"] * data_sorted["future_ret_5_1"]), label = "future_ret_5_1")

    # plt.plot( np.cumsum(-data["trade_imb_1"] *data["future_ret_1"]))
    # plt.plot(np.cumsum(-data["trade_imb_1"] * data["future_ret_5"]))
    # plt.plot(np.cumsum(-data["trade_imb_1"] * data["future_ret_5_1"]))

    # Customize plot
    plt.title("returns sorted by trade_imb_10", fontsize=16)
    plt.xlabel("trade imbalance", fontsize=14)
    plt.ylabel("Cumulative return", fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    plt.legend()
    plt.grid(True)

    file_path = os.path.join(OUTPUT_PATH, 'trade_imb_10_sort.png')
    plt.savefig(file_path)
    print("Plot saved to ", file_path)
    plt.close()

    ######################################################
    # backtest
    ######################################################
    plt.figure(figsize=(10, 6))

    data_sorted = data.sort_values("trade_imb_10")
    data_sorted = data_sorted[np.abs(data_sorted["trade_imb_10"]) > 0.1]
    # plt.plot(data_sorted["trade_imb_1"], np.cumsum(data_sorted["future_ret_1"]))
    # plt.plot(data_sorted["trade_imb_1"], np.cumsum(data_sorted["future_ret_5"]))
    plt.plot(data_sorted["trade_imb_10"], np.cumsum(-data_sorted["trade_imb_10"] * data_sorted["future_ret_1"]), label="future_ret_1")
    plt.plot(data_sorted["trade_imb_10"], np.cumsum(-data_sorted["trade_imb_10"] * data_sorted["future_ret_5"]), label="future_ret_5")
    plt.plot(data_sorted["trade_imb_10"], np.cumsum(-data_sorted["trade_imb_10"] * data_sorted["future_ret_5_1"]), label="future_ret_5_1")

    # plt.plot( np.cumsum(-data["trade_imb_1"] *data["future_ret_1"]))
    # plt.plot(np.cumsum(-data["trade_imb_1"] * data["future_ret_5"]))
    # plt.plot(np.cumsum(-data["trade_imb_1"] * data["future_ret_5_1"]))

    # Customize plot
    plt.title("returns sorted by trade_imb_10", fontsize=16)
    plt.xlabel("trade imbalance", fontsize=14)
    plt.ylabel("Cumulative return", fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    # Set y-axis limits
    # plt.ylim(-0.1, 1.0)
    plt.legend()
    plt.grid(True)

    file_path = os.path.join(OUTPUT_PATH, 'trade_imb_10_sort_filtered.png')
    plt.savefig(file_path)
    print("Plot saved to ", file_path)
    plt.close()




if __name__ == "__main__":
    main()

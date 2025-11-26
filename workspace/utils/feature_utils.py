import pandas as pd
import numpy as np
import datetime as dt

min_per_year = 365 * 24 * 60

def sort_cols(cols):
    """Sort columns based on feature type first, then parameter"""
    # determine first appearance of each base
    base_order = {}
    for col in cols:
        parts = col.rsplit("_", 1)
        base = parts[0]
        if base not in base_order:
            base_order[base] = len(base_order)

    def get_suffix_num(col):
        parts = col.rsplit("_", 1)

        # base feature
        if len(parts) == 1:
            return 0

        suffix = parts[1]
        digits = ''.join(c for c in suffix if c.isdigit())

        if digits:
            return int(digits)

        # Non-numeric suffix
        return float("inf")

    return sorted(
        cols,
        key=lambda col: (
            base_order[col.rsplit("_", 1)[0]],
            get_suffix_num(col)
        )
    )



####### Target ######
def target_rv(time_idx, level1_data, horizons):
    """Compute future realized volatility targets"""
    df = pd.DataFrame(index=time_idx)

    close_mid = level1_data["close_mid"]
    log_ret_1m = np.log(close_mid / close_mid.shift(1))

    for h in horizons:
        df[f"target_rv_fwd{h}m"] = (
            log_ret_1m.pow(2)
                    .rolling(h)
                    .mean().
                    shift(-h)
                    .mul(min_per_year)
                    .pow(0.5)
        )
    return df

###### Features #######
def build_features(time_idx, level1_data, book_data, trade_data, taus):
    df_level1 = feature_level1(time_idx, level1_data, taus)
    df_book = feature_book(time_idx, book_data)
    df_trade = feature_trade(time_idx, trade_data, taus)
    df_others = feature_others(time_idx)

    df_features = pd.concat([df_level1, df_book, df_trade, df_others], axis=1)

    return df_features


### Level1 Features ###
def feature_level1(time_idx, level1_data, taus):
    """"""
    df_ret = feature_return(time_idx, level1_data, taus)
    df_rv = feature_rv(time_idx, level1_data, taus)
    df_shape = feature_shape(time_idx, level1_data, taus)
    df_tick_vol = feature_tick_vol(time_idx, level1_data, taus)
    df_price_range = feature_price_range(time_idx, level1_data, taus)
    df_spread = feature_spread(time_idx, level1_data)

    df_level1 = pd.concat([df_ret, df_rv, df_shape, df_tick_vol, df_price_range, df_spread], axis=1)
    df_level1 = df_level1[sort_cols(df_level1.columns)]

    return df_level1

# Return-based features
def feature_return(time_idx, level1_data, taus):
    """Compute return-based features"""
    df = pd.DataFrame(index=time_idx)

    close_mid = level1_data["close_mid"]
    for tau in taus:
        log_ret_tau = np.log(close_mid / close_mid.shift(tau))

        # log return
        df[f"log_ret_{tau}m"] = log_ret_tau

        # absolute return
        df[f"abs_ret_{tau}m"] = log_ret_tau.abs()

        # squared return
        df[f"sq_ret_{tau}m"] = log_ret_tau ** 2

    return df

# Realized volatility features
def feature_rv(time_idx, level1_data, taus):
    """Compute historical realized volatility features"""
    df = pd.DataFrame(index=time_idx)

    log_return = level1_data["log_return"]
    for tau in taus:
        if tau == 1:
            continue

        # rolling realized volatility (annualized)
        df[f"rv_{tau}m"] = log_return.rolling(tau).std() * np.sqrt(min_per_year)

    return df


# Distribution shape features
def feature_shape(time_idx, level1_data, taus):
    """Compute distribution shape features"""
    df = pd.DataFrame(index=time_idx)

    log_return = level1_data["log_return"]
    for tau in taus:
        if tau >= 3:
            # rolling skew
            df[f"skew_{tau}m"] = log_return.rolling(tau).skew()
            
            if tau >= 4:
                # rolling kurtosis
                df[f"kurtosis_{tau}m"] = log_return.rolling(tau).kurt()

    return df

# Tick volatility features
def feature_tick_vol(time_idx, level1_data, taus):
    """ Compute tick volatility features"""
    df = pd.DataFrame(index=time_idx)

    tick_volatility = level1_data["tick_volatility"]
    df["tick_vol"] = tick_volatility

    for tau in taus:
        df[f"tick_vol_ema_{tau}m"] = tick_volatility.ewm(span=tau, adjust=False).mean()

    return df

# Price range features
def feature_price_range(time_idx, level1_data, taus):
    """Compute price range features"""
    df = pd.DataFrame(index=time_idx)

    max_mid = level1_data["max_mid"]
    min_mid = level1_data["min_mid"]
    close_mid = level1_data["close_mid"]
    price_range = (max_mid - min_mid) / close_mid

    df["price_range"] = price_range
    df["log_price_range"] = np.log1p(price_range)

    for tau in taus:
        if tau > 1:
            roll_mean = price_range.rolling(tau).mean()
            roll_std = price_range.rolling(tau).std()

            df[f"price_range_mean_{tau}m"] = roll_mean
            df[f"price_range_std_{tau}m"] = roll_std

            df[f"price_range_z_{tau}m"] = (price_range - roll_mean) / roll_std

    return df
# Spread features
def feature_spread(time_idx, level1_data):
    """Compute spread features"""
    df = pd.DataFrame(index=time_idx)

    close_mid = level1_data["close_mid"]

    df["spread_mean"] = level1_data["mean_spread"]
    df["spread_median"] = level1_data["median_spread"]
    df["spread_min"] = level1_data["min_spread"]
    df["spread_max"] = level1_data["max_spread"]

    df["spread_mean_norm"] = level1_data["mean_spread"] / close_mid
    df["spread_median_norm"] = level1_data["median_spread"] / close_mid
    df["spread_min_norm"] = level1_data["min_spread"] / close_mid
    df["spread_max_norm"] = level1_data["max_spread"] / close_mid

    df["log_spread_mean_norm"] = np.log1p(df["spread_mean_norm"])
    df["log_spread_median_norm"] = np.log1p(df["spread_median_norm"])

    return df

### Book Features ###
def feature_book(time_idx, book_data):
    """"""
    df_obi = feature_obi(time_idx, book_data)
    df_ofi = feature_ofi(time_idx, book_data)

    df_book = pd.concat([df_obi, df_ofi], axis=1)
    df_book = df_book[sort_cols(df_book.columns)]

    return df_book

# Order-book imbalance features
def feature_obi(time_idx, book_data):
    """Compute order-book imbalance features within bps from mid"""
    df = pd.DataFrame(index=time_idx)

    df["obi"] = (
        (book_data["bid_size"] - book_data["ask_size"]) /
        (book_data["bid_size"] + book_data["ask_size"])
    )

    bps_levels = [1, 3, 10, 30, 100]

    for bps in bps_levels:
        bid_col = f"bid_{bps}bps_fill_size"
        ask_col = f"ask_{bps}bps_fill_size"
        obi_col = f"obi_{bps}bps"

        df[obi_col] = (
            (book_data[bid_col] - book_data[ask_col]) /
            (book_data[bid_col] + book_data[ask_col])
        )

    return df

# Order-flow imbalance features
def feature_ofi(time_idx, book_data):
    """Compute order-flow imbalance features within bps from mid"""
    df = pd.DataFrame(index=time_idx)

    bid_diff = book_data["bid_size"].diff()
    ask_diff = book_data["ask_size"].diff()

    df["ofi"] = (bid_diff - ask_diff) / (bid_diff + ask_diff)

    bps_levels = [1, 3, 10, 30, 100]

    for bps in bps_levels:
        bid_col = f"bid_{bps}bps_fill_size"
        ask_col = f"ask_{bps}bps_fill_size"
        ofi_col = f"ofi_{bps}bps"

        bid_diff = book_data[bid_col].diff()
        ask_diff = book_data[ask_col].diff()

        df[ofi_col] = (bid_diff - ask_diff) / (bid_diff + ask_diff)

    return df

### Trade Features ###
def feature_trade(time_idx, trade_data, taus):
    """"""
    df_trade_vol = feature_trade_vol(time_idx, trade_data, taus)
    df_trade_count = feature_trade_count(trade_data, taus)
    df_volumes = feature_volume(trade_data, taus)

    df_trade = pd.concat([df_trade_vol, df_trade_count, df_volumes], axis=1)
    df_trade = df_trade[sort_cols(df_trade.columns)]

    return df_trade

# Trade volatility features
def feature_trade_vol(time_idx, trade_data, taus):
    """Compute trade volatility features"""
    df = pd.DataFrame(index=time_idx)

    trade_volatility = trade_data["trade_volatility"]
    df["trade_vol"] = trade_volatility

    for tau in taus:
        df[f"trade_vol_ema_{tau}m"] = trade_volatility.ewm(span=tau, adjust=False).mean()

    return df

# Trade count features
def feature_trade_count(trade_data, taus):
    """Compute trade count features"""
    df = trade_data[[
        "trade_count", "buy_trade_count", "sell_trade_count", 
        "taker_trade_count", "taker_buy_trade_count", "taker_sell_trade_count"
    ]].copy()

    df["trade_count_imb"] = (
        (df["buy_trade_count"] - df["sell_trade_count"]) / 
        (df["buy_trade_count"] + df["sell_trade_count"])
    )

    df["taker_count_imb"] = (
        (df["taker_buy_trade_count"] - df["taker_sell_trade_count"]) / 
        (df["taker_buy_trade_count"] + df["taker_sell_trade_count"])
    )

    for tau in taus:
        df[f"trade_count_ema_{tau}m"] = df["trade_count"].ewm(tau).mean()
        df[f"taker_trade_count_ema_{tau}m"] = df["taker_trade_count"].ewm(tau).mean()

    return df

# Volume features
def feature_volume(trade_data, taus):
    """Compute volume features"""
    df = trade_data[[
        "volume", "buy_volume", "sell_volume",
        "vwap", "buy_vwap", "sell_vwap"
    ]].copy()
    
    df["volume_imb"] = (
        (df["buy_volume"] - df["sell_volume"]) / 
        (df["buy_volume"] + df["sell_volume"])
    )

    for tau in taus:
        df[f"volume_ema_{tau}m"] = df["volume"].ewm(span=tau, adjust=False).mean()
        df[f"buy_volume_ema_{tau}m"] = df["buy_volume"].ewm(span=tau, adjust=False).mean()
        df[f"sell_volume_ema_{tau}m"] = df["sell_volume"].ewm(span=tau, adjust=False).mean()

    return df

### Other Features ###
def feature_others(time_idx):
    """"""
    df_seas = feature_seasonality(time_idx)

    df_book = pd.concat([df_seas], axis=1)

    return df_book

# Seasonality features
def feature_seasonality(time_idx):
    """Compute seasonality features from datetime index"""
    df = pd.DataFrame(index=time_idx)

    # Hour of day
    df["hour"] = time_idx.hour
    df["sin_hour"] = np.sin(2 * np.pi * time_idx.hour / 24)
    df["cos_hour"] = np.cos(2 * np.pi * time_idx.hour / 24)
    
    # Minute of hour
    df["minute"] = time_idx.minute
    df["sin_minute"] = np.sin(2 * np.pi * time_idx.minute / 60)
    df["cos_minute"] = np.cos(2 * np.pi * time_idx.minute / 60)
    
    # Day of week
    df["day_of_week"] = time_idx.dayofweek
    df["sin_dow"] = np.sin(2 * np.pi * time_idx.dayofweek / 7)
    df["cos_dow"] = np.cos(2 * np.pi * time_idx.dayofweek / 7)
    
    # Weekday / weekend
    df["is_weekend"] = (time_idx.dayofweek >= 5).astype(int)
    
    # Daylight savings
    t_ny = time_idx.tz_localize("UTC").tz_convert("America/New_York")
    df["is_dst"] = t_ny.map(lambda x: x.dst() != dt.timedelta(0)).astype(int)

    # Hour start / end
    df["is_hour_start"] = (df["minute"] < 5).astype(int)
    df["is_hour_end"]   = (df["minute"] >= 55).astype(int)

    return df
import pandas as pd
import numpy as np
import datetime as dt

min_per_year = 365 * 24 * 60

def sort_cols(cols):
    """Sort columns based on feature type first, then parameter"""
    # determine first appearance of each base
    base_order = {}
    for col in cols:
        base = col.rsplit("_", 1)[0]
        if base not in base_order:
            base_order[base] = len(base_order)

    # return sorted result
    return sorted(
        cols,
        key=lambda col: (
            base_order[col.rsplit("_", 1)[0]],
            int(col.rsplit("_", 1)[1].replace("m", ""))
        )
    )

# Target
def target_rv(time_idx, close_mid, horizons):
    """Compute future realized volatility targets"""
    df = pd.DataFrame(index=time_idx)

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

### Level1 Features ###
def feature_level1(time_idx, close_mid, log_return, taus):
    """"""
    df_ret = feature_return(time_idx, close_mid, taus)
    df_rv = feature_rv(time_idx, log_return, taus)
    df_shape = feature_shape(time_idx, log_return, taus)

    df_level1 = pd.concat([df_ret, df_rv, df_shape], axis=1)
    df_level1 = df_level1[sort_cols(df_level1.columns)]

    return df_level1

# Return-based features
def feature_return(time_idx, close_mid, taus):
    """Compute return-based features"""
    df = pd.DataFrame(index=time_idx)

    for tau in taus:
        log_ret_tau = np.log(close_mid / close_mid.shift(tau))

        # log return
        df[f"log_ret_{tau}m"] = log_ret_tau

        # absolute return
        df[f"abs_ret_{tau}m"] = log_ret_tau.abs()

        # squared return
        df[f"sq_ret_{tau}m"] = log_ret_tau ** 2

    return df

# Realized volattility features
def feature_rv(time_idx, log_return, taus):
    """Compute historical realized volatility features"""
    df = pd.DataFrame(index=time_idx)

    for tau in taus:
        if tau == 1:
            continue

        # rolling realized volatility (annualized)
        df[f"rv_{tau}m"] = log_return.rolling(tau).std() * np.sqrt(min_per_year)

    return df


# Distribution shape features
def feature_shape(time_idx, log_return, taus):
    """Compute distribution shape features"""
    df = pd.DataFrame(index=time_idx)

    for tau in taus:
        if tau >= 3:
            # rolling skew
            df[f"skew_{tau}m"] = log_return.rolling(tau).skew()
            
            if tau >= 4:
                # rolling kurtosis
                df[f"kurtosis_{tau}m"] = log_return.rolling(tau).kurt()

    return df

### Book Features ###
def feature_book(time_idx, book_data):
    """"""
    df_obi = feature_obi(time_idx, book_data)

    df_book = pd.concat([df_obi], axis=1)

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

# def feature_ofi(time_idx, book_data):


### Trade Features ###


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
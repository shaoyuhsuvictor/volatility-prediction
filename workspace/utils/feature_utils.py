import pandas as pd
import numpy as np
import datetime as dt

min_per_year = 365 * 24 * 60

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

# Return-based features
def feature_return_based(time_idx, close_mid, taus):
    """Compute return-based features"""
    df = pd.DataFrame(index=time_idx)

    log_ret_1m = np.log(close_mid / close_mid.shift(1))

    for tau in taus:
        log_ret_tau = log_ret_1m if tau == 1 else np.log(close_mid / close_mid.shift(tau))

        # log return
        df[f"log_ret_{tau}m"] = log_ret_tau

        # absolute return
        df[f"abs_ret_{tau}m"] = log_ret_tau.abs()

        # squared return
        df[f"sq_ret_{tau}m"] = log_ret_tau ** 2

        if tau == 1:
            continue

        # rolling realized volatility (annualized)
        df[f"rv_{tau}m"] = log_ret_1m.rolling(tau).std() * np.sqrt(min_per_year)

        # rolling skew
        df[f"skew_{tau}m"] = log_ret_1m.rolling(tau).skew()

        # rolling kurtosis
        df[f"kurtosis_{tau}m"] = log_ret_1m.rolling(tau).kurt()

    return df

# Order-book imbalance features

# Historical realized volatility features

# Volume-based features

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
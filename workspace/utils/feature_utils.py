import pandas as pd
import numpy as np

def target_rv(time_idx, log_return, horizons, annualization=365*24*60):
    """Compute future realized volatility targets"""
    df = pd.DataFrame(index=time_idx)

    for h in horizons:
        col = f"target_rv_fwd{h}m"
        df[col] = (
            log_return.pow(2)
               .rolling(h)
               .mean()
               .shift(-h)
               .mul(annualization)
               .pow(0.5)
        )
    return df

def feature_log_ret(time_idx, close_mid, taus):
    """Compute log return features for given time lags"""
    df = pd.DataFrame(index=time_idx)

    for tau in taus:
        col = f'log_ret_{tau}m'
        df[col] = np.log(close_mid / close_mid.shift(tau))

    return df

def feature_abs_ret(time_idx, log_ret_df, taus):
    """Compute absolute log return features for given time lags"""
    df = pd.DataFrame(index=time_idx)

    for tau in taus:
        col = f'abs_ret_{tau}m'
        df[col] = log_ret_df[f'log_ret_{tau}m'].abs()

    return df

def feature_sq_ret(time_idx, log_ret_df, taus):
    """Compute sqaured log return features for given time lags"""
    df = pd.DataFrame(index=time_idx)

    for tau in taus:
        col = f'sq_ret_{tau}min'
        df[col] = log_ret_df[f'log_ret_{tau}m'] ** 2

    return df
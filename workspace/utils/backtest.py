import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def vol_managed_backtest(returns, pred_vol,
                               sigma_target=0.30,
                               w_min=0.5, w_max=2.0,
                               ann_factor=525600):

    # Target volatility scaling and capping
    w = (sigma_target / pred_vol).shift(1)
    w = w.clip(lower=w_min, upper=w_max)

    # Vol-Managed and Buy & Hold returns
    strat_ret = w * returns
    base_ret = returns

    # Sharpe
    sharpe_strat = np.sqrt(ann_factor) * strat_ret.mean() / strat_ret.std()
    sharpe_base  = np.sqrt(ann_factor) * base_ret.mean() / base_ret.std()

    # Sortino (downside deviation)
    def sortino(x):
        downside = x[x < 0]
        if len(downside) == 0:
            return np.nan
        dd = downside.std()
        return np.sqrt(ann_factor) * x.mean() / dd
    
    sortino_strat = sortino(strat_ret)
    sortino_base  = sortino(base_ret)

    # Max drawdown
    dd_strat = (strat_ret.cumsum() - strat_ret.cumsum().cummax()).min()
    dd_base  = (base_ret.cumsum() - base_ret.cumsum().cummax()).min()

    # Annualized volatility
    vol_strat = strat_ret.std() * np.sqrt(ann_factor)
    vol_base  = base_ret.std()  * np.sqrt(ann_factor)

    metrics = {
        "Cumulative Return (Vol-Managed)": strat_ret.cumsum().dropna().iloc[-1],
        "Cumulative Return (Buy & Hold)": base_ret.cumsum().dropna().iloc[-1],
        "Annualized Sharpe (Vol-Managed)": sharpe_strat,
        "Annualized Sharpe (Buy & Hold)"        : sharpe_base,
        "Annualized Sortino (Vol-Managed)": sortino_strat,
        "Annualized Sortino (Buy & Hold)"        : sortino_base,
        "Max DD (Vol-Managed)": dd_strat,
        "Max DD (Buy & Hold)"        : dd_base,
        "Annualized Volatility (Vol-Managed)"    : vol_strat,
        "Annualized Volatility (Buy & Hold)"    : vol_base
    }

    return strat_ret, base_ret, metrics, w

def plot_backtest(strat_ret, base_ret, weights=None, title="Volatility-Managed vs Buy & Hold"):
    """Plot cumulative returns of volatility-managed vs Buy & Hold, with optional weights subplot."""
    
    fig, axes = plt.subplots(2, 1, figsize=(14,6), sharex=True)
    
    # Cumulative returns
    base_ret.cumsum().plot(label="Buy & Hold",   alpha=0.9, ax=axes[0])
    strat_ret.cumsum().plot(label="Vol-Managed", alpha=0.9, ax=axes[0])
    
    axes[0].set_title(title)
    axes[0].set_ylabel("Cumulative Return")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: Weights
    if weights is not None:
        weights.plot(label="Volatility Scaling Weight", color="green", linewidth=0.7, alpha=0.8, ax=axes[1])
        axes[1].set_ylabel("Position Weight")
        axes[1].set_xlabel("Time")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].set_visible(False)
    
    plt.tight_layout()
    plt.show()



def metrics_to_table(metrics):
    """Convert metrics dict into a clean side-by-side comparison table."""
    df = pd.DataFrame(metrics, index=[0]).T
    df.columns = ["Value"]
    
    # Split into two columns: Vol-Managed vs Buy & Hold
    df["Metric"] = df.index
    df["Category"] = df["Metric"].apply(
        lambda x: "Buy & Hold" if "Buy & Hold" in x else "Vol-Managed"
    )
    
    df["Metric_Name"] = df["Metric"].apply(
        lambda x: x.replace(" (Buy & Hold)", "").replace(" (Vol-Managed)", "")
    )
    
    table = df.pivot(index="Metric_Name", columns="Category", values="Value")
    return table

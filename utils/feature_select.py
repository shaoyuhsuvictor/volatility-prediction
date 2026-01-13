import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_corr_heatmap(df, method="spearman", figsize=(10, 8), title="Correlation Heatmap"):
    """Plots a correlation heatmap for a given DataFrame"""
    corr = df.corr(method=method)

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
        square=True
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def select_ic(df, target, thresh=0.10):
    """Select features by Spearman IC vs target."""
    feats = [c for c in df.columns if c != target]
    ic = df[feats].corrwith(df[target], method="spearman")
    keep = ic[ic.abs() > thresh].index.tolist()
    return keep, ic

def prune_corr(df, feats, ic_series, thresh=0.70):
    """
    Prune features by Pearson collinearity using sequential IC-ranked selection.
    Keeps the highest-IC feature in each correlated cluster.
    """
    # Compute correlation matrix (abs)
    corr = df[feats].corr(method="pearson").abs()

    # Sort features by absolute IC (descending)
    ranked = sorted(feats, key=lambda f: abs(ic_series[f]), reverse=True)

    selected = []
    dropped = []

    for f in ranked:
        # Check correlation with already-selected features
        if all(corr.loc[f, s] < thresh for s in selected):
            selected.append(f)
        else:
            dropped.append(f)

    return selected, corr, dropped

def plot_ic(df, feats, target):
    """Plot Spearman IC heatmap."""
    corr = df[feats + [target]].corr(method="spearman")
    sns.heatmap(
        corr[[target]].T,
        annot=True,
        cmap="coolwarm",
        center=0,
    )
    plt.title("IC (Spearman)")
    plt.show()

def plot_corr(df, feats):
    """Plot Pearson feature correlation heatmap."""
    corr = df[feats].corr(method="pearson")
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
    )
    plt.title("Feature Correlation (Pearson)")
    plt.show()
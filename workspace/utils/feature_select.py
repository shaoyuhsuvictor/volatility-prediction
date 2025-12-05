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

def prune_corr(df, feats, thresh=0.70):
    """Prune features by Pearson collinearity."""
    corr = df[feats].corr(method="pearson").abs()
    upper = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))
    drop = [c for c in upper.columns if any(upper[c] > thresh)]
    final = [f for f in feats if f not in drop]
    return final, corr, drop

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
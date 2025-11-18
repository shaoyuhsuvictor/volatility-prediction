import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_corr_heatmap(df, figsize=(10, 8), title="Correlation Heatmap"):
    """Plots a correlation heatmap for a given DataFrame"""
    corr = df.corr()

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
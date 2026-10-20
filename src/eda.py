"""Basic exploratory data analysis plots (Week 1 EDA plan)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run_eda(df: pd.DataFrame, fig_dir: str) -> None:
    out = Path(fig_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Distribution of yield (raw and log scale to show skew).
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(df["yield"], bins=50)
    ax[0].set_title("Yield distribution")
    ax[0].set_xlabel("Yield")
    ax[1].hist(np.log1p(df["yield"]), bins=50)
    ax[1].set_title("log(1 + Yield)")
    fig.tight_layout()
    fig.savefig(out / "eda_yield_distribution.png", dpi=120)
    plt.close(fig)

    # 2. Average yield by crop (top 15 crops by number of records).
    top = df["crop"].value_counts().head(15).index
    avg = df[df["crop"].isin(top)].groupby("crop")["yield"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    avg.plot.barh(ax=ax)
    ax.set_title("Average yield by crop (15 most frequent crops)")
    fig.tight_layout()
    fig.savefig(out / "eda_avg_yield_by_crop.png", dpi=120)
    plt.close(fig)

    # 3. Yield trend over the years for the 5 most frequent crops.
    fig, ax = plt.subplots(figsize=(8, 5))
    for crop in df["crop"].value_counts().head(5).index:
        s = df[df["crop"] == crop].groupby("year")["yield"].mean()
        ax.plot(s.index, s.values, marker="o", label=crop)
    ax.set_title("Average yield per year (5 most frequent crops)")
    ax.set_xlabel("Year")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "eda_yield_trend.png", dpi=120)
    plt.close(fig)

    # 4. Correlation heatmap of numeric columns.
    cols = ["area", "annual_rainfall", "fertilizer", "pesticide", "yield"]
    corr = df[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im)
    ax.set_title("Spearman correlation")
    fig.tight_layout()
    fig.savefig(out / "eda_correlation.png", dpi=120)
    plt.close(fig)

"""
Matplotlib visualisations for SHAP relativities.

Bar charts for categorical features, line charts for continuous features.
Both show 95% confidence intervals by default.

Design choice: plots are deliberately plain - no corporate styling, no
seaborn dependency. Callers can apply their own style sheets on top.

Input is a Polars DataFrame from extract_relativities(). A thin conversion
to numpy arrays is the only pandas dependency here, and it doesn't exist -
Polars columns convert directly to numpy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    import matplotlib.axes


def plot_categorical(
    data: pl.DataFrame,
    feature: str,
    ax: "matplotlib.axes.Axes",
    show_ci: bool = True,
    color: str = "steelblue",
) -> None:
    """
    Bar chart of categorical relativities for a single feature.

    Args:
        data: Rows for this feature from extract_relativities() output.
            Required columns: level, relativity, lower_ci, upper_ci.
        feature: Feature name, used as the axis label.
        ax: Axes to draw on.
        show_ci: If True, draw error bars for confidence intervals.
        color: Bar color.
    """
    levels = data["level"].cast(pl.String).to_numpy()
    x = np.arange(len(levels))
    relativities = data["relativity"].to_numpy()

    if show_ci and "lower_ci" in data.columns and "upper_ci" in data.columns:
        lower_err = relativities - data["lower_ci"].to_numpy()
        upper_err = data["upper_ci"].to_numpy() - relativities
        yerr = np.array([lower_err, upper_err])
    else:
        yerr = None

    ax.bar(x, relativities, color=color, alpha=0.8, yerr=yerr,
           capsize=4, error_kw={"linewidth": 1})
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(levels, rotation=45, ha="right")
    ax.set_ylabel("Relativity")
    ax.set_title(feature)


def plot_continuous(
    data: pl.DataFrame,
    feature: str,
    ax: "matplotlib.axes.Axes",
    show_ci: bool = True,
    color: str = "steelblue",
) -> None:
    """
    Line chart of continuous relativities for a single feature.

    Args:
        data: Rows for this feature from extract_relativities() output.
            Required columns: level (numeric, stored as Utf8 string after
            extract_relativities), relativity, lower_ci, upper_ci.
        feature: Feature name, used as the axis label.
        ax: Axes to draw on.
        show_ci: If True, draw a shaded confidence band.
        color: Line color.
    """
    # P0-2 fix: the level column is Utf8 after extract_relativities() casts it.
    # Sorting strings lexicographically gives the wrong order for numeric values
    # (e.g. '10.0' < '2.0' as strings). Cast to Float64 before sorting.
    data_sorted = (
        data
        .with_columns(pl.col("level").cast(pl.Float64).alias("level_num"))
        .sort("level_num")
    )
    x = data_sorted["level_num"].to_numpy()
    y = data_sorted["relativity"].to_numpy()

    ax.plot(x, y, color=color, linewidth=1.5)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)

    if show_ci and "lower_ci" in data_sorted.columns and "upper_ci" in data_sorted.columns:
        ax.fill_between(
            x,
            data_sorted["lower_ci"].to_numpy(),
            data_sorted["upper_ci"].to_numpy(),
            alpha=0.2,
            color=color,
        )

    ax.set_xlabel(feature)
    ax.set_ylabel("Relativity")
    ax.set_title(feature)


def plot_relativities(
    relativities_df: pl.DataFrame,
    categorical_features: list[str],
    continuous_features: list[str],
    features: list[str] | None = None,
    show_ci: bool = True,
    figsize: tuple[int, int] = (12, 8),
) -> None:
    """
    Grid of relativity plots for all requested features.

    Categorical features get bar charts; continuous features get line charts.
    If features is None, all features in relativities_df are plotted.

    Args:
        relativities_df: Output from SHAPRelativities.extract_relativities().
        categorical_features: Feature names to treat as categorical (bar chart).
        continuous_features: Feature names to treat as continuous (line chart).
        features: Subset of features to plot. Defaults to all.
        show_ci: Whether to show confidence intervals.
        figsize: Overall figure size.
    """
    import matplotlib.pyplot as plt

    all_features = features or relativities_df["feature"].unique().to_list()
    n = len(all_features)
    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for i, feat in enumerate(all_features):
        ax = axes_flat[i]
        feat_data = relativities_df.filter(pl.col("feature") == feat)

        if feat in categorical_features:
            plot_categorical(feat_data, feat, ax, show_ci=show_ci)
        elif feat in continuous_features:
            plot_continuous(feat_data, feat, ax, show_ci=show_ci)
        else:
            # Fallback: treat as categorical
            plot_categorical(feat_data, feat, ax, show_ci=show_ci)

    # Hide unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    plt.show()

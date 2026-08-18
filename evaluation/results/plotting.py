"""Publication-oriented plots for the three core ARCEME analyses."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "model": "#176B87",
    "baseline": "#8C8C8C",
    "observed": "#2F7D32",
    "predicted": "#C7511F",
}


def _save(fig: plt.Figure, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _vertical_errorbar(ax, x, y, low, high, **kwargs) -> None:
    lower = np.maximum(0.0, np.asarray(y) - np.asarray(low))
    upper = np.maximum(0.0, np.asarray(high) - np.asarray(y))
    ax.errorbar(x, y, yerr=np.vstack([lower, upper]), **kwargs)


def _horizontal_errorbar(ax, x, y, low, high, **kwargs) -> None:
    lower = np.maximum(0.0, np.asarray(x) - np.asarray(low))
    upper = np.maximum(0.0, np.asarray(high) - np.asarray(x))
    ax.errorbar(x, y, xerr=np.vstack([lower, upper]), **kwargs)


def plot_cv_model_selection(summary: pd.DataFrame, path: str | Path) -> None:
    metrics = [
        ("fold_macro_nnse", "Fold-macro NNSE", True),
        ("mse_skill", "Persistence skill", True),
        ("mae", "MAE", False),
    ]
    configurations = list(dict.fromkeys(summary["configuration"].tolist()))
    fig, axes = plt.subplots(1, 3, figsize=(14, max(4.5, 0.62 * len(configurations))))
    for ax, (metric, label, higher_better) in zip(axes, metrics):
        data = summary[summary["metric"] == metric].set_index("configuration").reindex(
            configurations
        )
        y = np.arange(len(data))
        _horizontal_errorbar(
            ax,
            data["estimate"].to_numpy(),
            y,
            data["ci_low"].to_numpy(),
            data["ci_high"].to_numpy(),
            fmt="o",
            color=COLORS["model"],
            capsize=3,
        )
        ax.set_yticks(y, configurations if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.set_xlabel(label)
        ax.grid(axis="x", alpha=0.25)
        direction = "higher is better" if higher_better else "lower is better"
        ax.set_title(direction, fontsize=9)
        if metric == "mse_skill":
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    fig.suptitle("Cross-validated model selection (cube bootstrap)", fontweight="bold")
    fig.tight_layout()
    _save(fig, path)


def plot_factor_contrasts(contrasts: pd.DataFrame, path: str | Path) -> None:
    if contrasts.empty:
        return
    metrics = list(dict.fromkeys(contrasts["metric"].tolist()))
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.8), squeeze=False)
    for ax, metric in zip(axes.ravel(), metrics):
        data = contrasts[contrasts["metric"] == metric].reset_index(drop=True)
        y = np.arange(len(data))
        _horizontal_errorbar(
            ax,
            data["estimate"],
            y,
            data["ci_low"],
            data["ci_high"],
            fmt="o",
            color=COLORS["model"],
            capsize=3,
        )
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_yticks(y, data["contrast"] if ax is axes.ravel()[0] else [])
        ax.invert_yaxis()
        ax.set_xlabel(metric)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Paired 2×2 effects (positive follows contrast label)", fontweight="bold")
    fig.tight_layout()
    _save(fig, path)


def plot_holdout_lead_time(summary: pd.DataFrame, path: str | Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    for metric, label, color in (
        ("mae", "Model", COLORS["model"]),
        ("mae_base", "Persistence", COLORS["baseline"]),
    ):
        data = summary[summary["metric"] == metric].sort_values("lead_day")
        _vertical_errorbar(
            axes[0],
            data["lead_day"],
            data["estimate"],
            data["ci_low"],
            data["ci_high"],
            fmt="o-",
            label=label,
            color=color,
            capsize=2,
        )
    axes[0].set_ylabel("MAE (kNDVI)")
    axes[0].legend(frameon=False)

    data = summary[summary["metric"] == "mse_skill_ratio"].sort_values("lead_day")
    _vertical_errorbar(
        axes[1],
        data["lead_day"],
        data["estimate"],
        data["ci_low"],
        data["ci_high"],
        fmt="o-",
        color=COLORS["model"],
        capsize=2,
    )
    axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("MSE skill vs persistence")

    for metric, label, color in (
        ("observed_response", "Observed", COLORS["observed"]),
        ("predicted_response", "Predicted", COLORS["predicted"]),
    ):
        data = summary[summary["metric"] == metric].sort_values("lead_day")
        _vertical_errorbar(
            axes[2],
            data["lead_day"],
            data["estimate"],
            data["ci_low"],
            data["ci_high"],
            fmt="o-",
            label=label,
            color=color,
            capsize=2,
        )
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_ylabel("Mean change from event-end baseline")
    axes[2].legend(frameon=False)

    for ax in axes:
        ax.set_xlabel("Lead day")
        ax.set_xticks([5, 10, 15, 20, 25, 30])
        ax.grid(alpha=0.25)
    fig.suptitle("Independent holdout performance over forecast horizon", fontweight="bold")
    fig.tight_layout()
    _save(fig, path)


def plot_r30_scatter(cubes: pd.DataFrame, path: str | Path) -> None:
    data = cubes[["observed_r30", "predicted_r30", "mse_skill", "cube_id"]].dropna()
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    scatter = ax.scatter(
        data["observed_r30"],
        data["predicted_r30"],
        c=data["mse_skill"],
        cmap="RdYlBu",
        edgecolor="white",
        linewidth=0.5,
        s=48,
    )
    minimum = float(min(data["observed_r30"].min(), data["predicted_r30"].min()))
    maximum = float(max(data["observed_r30"].max(), data["predicted_r30"].max()))
    padding = max((maximum - minimum) * 0.05, 0.002)
    limits = (minimum - padding, maximum + padding)
    ax.plot(limits, limits, color="black", linestyle="--", linewidth=1)
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.axvline(0, color="grey", linewidth=0.6)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Observed day-30 response")
    ax.set_ylabel("Predicted day-30 response")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Cube MSE skill vs persistence")
    ax.set_title("Day-30 vegetation response")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    _save(fig, path)


def plot_environment_groups(summary: pd.DataFrame, path: str | Path) -> None:
    if summary.empty:
        return
    group_types = [value for value in ("climate", "landcover") if value in set(summary["group_type"])]
    if not group_types:
        return
    fig, axes = plt.subplots(1, len(group_types), figsize=(6 * len(group_types), 4.8), squeeze=False)
    for ax, group_type in zip(axes.ravel(), group_types):
        data = summary[
            (summary["group_type"] == group_type) & (summary["metric"] == "mae_gain")
        ].sort_values("estimate")
        y = np.arange(len(data))
        _horizontal_errorbar(
            ax,
            data["estimate"],
            y,
            data["ci_low"],
            data["ci_high"],
            fmt="o",
            color=COLORS["model"],
            capsize=3,
        )
        labels = [f"{group} (n={n})" for group, n in zip(data["group"], data["n_cubes"])]
        ax.set_yticks(y, labels)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("MAE gain over persistence (>0 better)")
        ax.set_title(group_type.capitalize())
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Cross-validated conditional forecast skill", fontweight="bold")
    fig.tight_layout()
    _save(fig, path)


def plot_event_tertiles(
    summary: pd.DataFrame, feature: str, path: str | Path
) -> None:
    if summary.empty:
        return
    order = ["low", "middle", "high"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    for ax, metric, ylabel in (
        (axes[0], "observed_r30", "Observed day-30 response"),
        (axes[1], "mae_gain", "MAE gain over persistence"),
    ):
        data = summary[summary["metric"] == metric].copy()
        data["order"] = data["group"].map({value: idx for idx, value in enumerate(order)})
        data = data.sort_values("order")
        _vertical_errorbar(
            ax,
            np.arange(len(data)),
            data["estimate"],
            data["ci_low"],
            data["ci_high"],
            fmt="o-",
            color=COLORS["model"],
            capsize=3,
        )
        labels = [f"{group}\n(n={n})" for group, n in zip(data["group"], data["n_cubes"])]
        ax.set_xticks(np.arange(len(data)), labels)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"Event conditioning: {feature}", fontweight="bold")
    fig.tight_layout()
    _save(fig, path)

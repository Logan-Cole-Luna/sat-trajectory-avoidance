import os
import math
from typing import Any, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_policy_plots(
    base_dir: str,
    episodes_df: Optional[Any],
    termination_df: Optional[Any],
    aggregates_df: Optional[Any],
    time_series_df: Optional[Any],
    time_series_stats: Optional[Any],
) -> None:
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    sns.set_theme(style="darkgrid")

    def _save_fig(fig: Any, path: str) -> None:
        try:
            fig.savefig(path, bbox_inches="tight")
        finally:
            plt.close(fig)

    has_episode_data = bool(episodes_df is not None and not getattr(episodes_df, "empty", True))
    has_time_series_stats = bool(time_series_stats is not None and not getattr(time_series_stats, "empty", True))

    # Reward histogram
    if has_episode_data and "reward" in episodes_df.columns:
        ax = sns.histplot(
            data=episodes_df,
            x="reward",
            hue="policy",
            bins=20,
            element="step",
            stat="density",
            common_norm=False,
            alpha=0.4,
        )
        ax.set_title("Episode Rewards by Policy")
        _save_fig(ax.get_figure(), os.path.join(plots_dir, "reward_distribution.png"))

    # Delta-v histogram
    if has_episode_data and "delta_v" in episodes_df.columns:
        ax = sns.histplot(
            data=episodes_df,
            x="delta_v",
            hue="policy",
            bins=20,
            element="step",
            stat="density",
            common_norm=False,
            alpha=0.4,
        )
        ax.set_title("Episode Delta-V by Policy")
        _save_fig(ax.get_figure(), os.path.join(plots_dir, "delta_v_distribution.png"))

    # Minimum distance histogram
    if has_episode_data and "min_distance" in episodes_df.columns:
        min_df = episodes_df.dropna(subset=["min_distance"])
        if not min_df.empty:
            ax = sns.histplot(
                data=min_df,
                x="min_distance",
                hue="policy",
                bins=20,
                element="step",
                stat="density",
                common_norm=False,
                alpha=0.4,
            )
            ax.set_title("Minimum Distance by Policy (m)")
            _save_fig(ax.get_figure(), os.path.join(plots_dir, "min_distance_distribution.png"))

    # Collision rate bar
    if has_episode_data and "collision" in episodes_df.columns:
        collision_series = episodes_df["collision"].apply(
            lambda x: np.nan if x is None or (isinstance(x, float) and math.isnan(x)) else float(bool(x))
        )
        rate_df = (
            pd.DataFrame({"policy": episodes_df["policy"], "collision_rate": collision_series})
            .dropna(subset=["collision_rate"])
            .groupby("policy")
            .mean()
            .reset_index()
        )
        if not rate_df.empty:
            ax = sns.barplot(data=rate_df, x="policy", y="collision_rate")
            ax.set_title("Collision Rate by Policy")
            ax.set_ylabel("Collision Rate")
            _save_fig(ax.get_figure(), os.path.join(plots_dir, "collision_rate_by_policy.png"))

    # Reward vs delta-v scatter
    if has_episode_data and {"reward", "delta_v"}.issubset(episodes_df.columns):
        scatter_df = episodes_df.dropna(subset=["reward", "delta_v"])
        if not scatter_df.empty:
            ax = sns.scatterplot(data=scatter_df, x="delta_v", y="reward", hue="policy")
            ax.set_title("Reward vs Delta-V")
            _save_fig(ax.get_figure(), os.path.join(plots_dir, "reward_vs_delta_v.png"))

    # Termination reasons
    if termination_df is not None and not termination_df.empty:
        ax = sns.barplot(data=termination_df, x="reason", y="count", hue="policy")
        ax.set_title("Termination Reasons by Policy")
        ax.set_ylabel("Count")
        _save_fig(ax.get_figure(), os.path.join(plots_dir, "termination_reasons.png"))

    # Aggregate metrics bars
    if aggregates_df is not None and not aggregates_df.empty:
        safe_df = aggregates_df.copy()
        metric_groups = [
            ("collision_success_rates.png", ["collision_rate", "success_rate"], "Collision / Success Rates"),
            (
                "core_metrics.png",
                ["avg_reward", "avg_delta_v", "avg_steps", "avg_fuel_consumed", "avg_min_distance"],
                "Aggregate Metrics",
            ),
        ]
        for fname, cols, title in metric_groups:
            cols_present = [c for c in cols if c in safe_df.columns]
            if not cols_present:
                continue
            melted = safe_df.melt(id_vars="policy", value_vars=cols_present, var_name="metric", value_name="value")
            if melted.empty:
                continue
            ax = sns.barplot(data=melted, x="metric", y="value", hue="policy")
            ax.set_title(title)
            ax.set_ylabel("Value")
            _save_fig(ax.get_figure(), os.path.join(plots_dir, fname))

    # Joint KDE: Reward vs Delta-V
    if has_episode_data and {"reward", "delta_v"}.issubset(episodes_df.columns):
        valid = episodes_df.dropna(subset=["reward", "delta_v"])
        if not valid.empty:
            sample = valid.sample(5000, random_state=42) if len(valid) > 5000 else valid
            g = sns.jointplot(data=sample, x="delta_v", y="reward", kind="kde", fill=True, cmap="mako")
            g.figure.suptitle("Joint KDE: Reward vs Delta-V")
            g.figure.tight_layout()
            g.figure.savefig(os.path.join(plots_dir, "reward_vs_delta_v_joint_kde.png"), bbox_inches="tight")
            plt.close(g.figure)

    # Termination reasons per episode (treated as runs)
    if has_episode_data and "termination_reason" in episodes_df.columns:
        term_ep = episodes_df.dropna(subset=["termination_reason"])
        if not term_ep.empty:
            term_counts = (
                term_ep.groupby(["policy", "episode", "termination_reason"], dropna=False)
                .size()
                .reset_index(name="count")
            )
            term_counts.rename(columns={"episode": "run"}, inplace=True)
            ax = sns.barplot(data=term_counts, x="run", y="count", hue="termination_reason")
            ax.set_title("Termination Reasons per Run")
            ax.set_ylabel("Count")
            _save_fig(ax.get_figure(), os.path.join(plots_dir, "termination_reasons_per_run.png"))

    # Time-series plots with mean/std shading
    if has_time_series_stats:
        metrics_meta = [
            ("min_distance", "Min Distance vs Time", "min_distance_vs_time.png", "Min Distance (m)"),
            ("cumulative_delta_v", "Cumulative Delta-V vs Time", "cumulative_delta_v_vs_time.png", "Cumulative Delta-V (m/s)"),
            ("delta_v_step", "Delta-V per Step vs Time", "delta_v_step_vs_time.png", "Delta-V per Step (m/s)"),
        ]
        for metric, title, fname, ylabel in metrics_meta:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            if mean_col not in time_series_stats.columns:
                continue
            fig, ax = plt.subplots()
            for policy, sub in time_series_stats.groupby("policy"):
                sub_sorted = sub.sort_values("time_bucket")
                x = sub_sorted["time_bucket"].to_numpy(dtype=float)
                y = sub_sorted[mean_col].to_numpy(dtype=float)
                y_std = sub_sorted[std_col].to_numpy(dtype=float) if std_col in sub_sorted else np.zeros_like(y)
                ax.plot(x, y, label=str(policy))
                ax.fill_between(x, y - y_std, y + y_std, alpha=0.2)
            ax.set_title(title)
            ax.set_xlabel("Elapsed Time (s)")
            ax.set_ylabel(ylabel)
            ax.legend()
            _save_fig(fig, os.path.join(plots_dir, fname))

    # Optional overlays of raw time-series points
    if has_time_series_stats and time_series_df is not None and not time_series_df.empty:
        sample = time_series_df.dropna(subset=["elapsed_time", "delta_v_step"])
        if not sample.empty:
            sample = sample.sample(min(len(sample), 5000), random_state=42) if len(sample) > 5000 else sample
            ax = sns.scatterplot(
                data=sample,
                x="elapsed_time",
                y="delta_v_step",
                hue="policy",
                s=12,
            )
            ax.set_title("Delta-V per Step (sampled)")
            ax.set_xlabel("Elapsed Time (s)")
            ax.set_ylabel("Delta-V per Step (m/s)")
            _save_fig(ax.get_figure(), os.path.join(plots_dir, "delta_v_step_scatter.png"))

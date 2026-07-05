from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ensure_output_dirs, load_config
from .environment import InspectionEnvironment
from .geometry import route_from_string


METHOD_COLORS = {
    "NearestNeighbor": "#7f7f7f",
    "ValuePerDistance": "#9467bd",
    "WindAwareGreedy": "#1f77b4",
    "ReserveOnlyPlanner": "#ff7f0e",
    "RAWA-RHP": "#2ca02c",
    "RAWA-NoRisk": "#d62728",
    "RAWA-NoWind": "#8c564b",
}
WIND_ORDER = ["calm", "moderate", "severe"]


def make_plots(output_root: Path) -> None:
    outputs = ensure_output_dirs(output_root)
    df = pd.read_csv(outputs["results"] / "episode_results.csv")
    make_scenario_example(outputs["figures"])
    make_trajectory_comparison(df, outputs["figures"])
    make_coverage_vs_wind(df, outputs["figures"])
    make_safety_vs_wind(df, outputs["figures"])
    make_pareto(df, outputs["figures"])
    make_ablation(df, outputs["figures"])


def make_scenario_example(fig_dir: Path) -> None:
    cfg = load_config()
    env = InspectionEnvironment(cfg, seed=0, wind_level="severe", obstacle_density="cluttered")
    fig, ax = plt.subplots(figsize=(6.6, 6.0), dpi=180)
    _draw_environment(ax, env)
    xs = np.linspace(8, 94, 10)
    ys = np.linspace(8, 94, 10)
    grid = np.array([(x, y) for y in ys for x in xs])
    w = env.wind.vector(grid)
    ax.quiver(grid[:, 0], grid[:, 1], w[:, 0], w[:, 1], color="#005f73", alpha=0.55, width=0.003)
    ax.set_title("Synthetic UAV inspection environment: severe wind, cluttered structures")
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "scenario_example.png")
    plt.close(fig)


def make_trajectory_comparison(df: pd.DataFrame, fig_dir: Path) -> None:
    subset = df[
        (df["seed"] == 0)
        & (df["wind_level"] == "severe")
        & (df["obstacle_density"] == "cluttered")
        & (df["planner"].isin(["NearestNeighbor", "ReserveOnlyPlanner", "RAWA-RHP"]))
    ]
    cfg = load_config()
    env = InspectionEnvironment(cfg, seed=0, wind_level="severe", obstacle_density="cluttered")
    fig, ax = plt.subplots(figsize=(6.8, 6.0), dpi=180)
    _draw_environment(ax, env)
    for _, row in subset.iterrows():
        pts = np.array(route_from_string(row["route"]))
        if len(pts) == 0:
            continue
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            marker="o",
            markersize=2.8,
            linewidth=1.7,
            color=METHOD_COLORS.get(row["planner"], None),
            label=f"{row['planner']} ({row['weighted_coverage']:.2f} coverage)",
        )
    ax.set_title("Trajectory comparison in a high-risk severe-wind case")
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.legend(loc="upper right", fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "trajectory_comparison.png")
    plt.close(fig)


def make_coverage_vs_wind(df: pd.DataFrame, fig_dir: Path) -> None:
    methods = _ordered_methods(df["planner"].unique())
    fig, ax = plt.subplots(figsize=(7.1, 4.2), dpi=180)
    x = np.arange(len(WIND_ORDER))
    for method in methods:
        means = []
        cis = []
        for wind in WIND_ORDER:
            vals = df[(df["wind_level"] == wind) & (df["planner"] == method)]["weighted_coverage"].to_numpy()
            means.append(np.mean(vals))
            cis.append(_ci95(vals))
        ax.errorbar(x, means, yerr=cis, marker="o", linewidth=1.7, capsize=3, label=method, color=METHOD_COLORS.get(method))
    ax.set_xticks(x, WIND_ORDER)
    ax.set_ylabel("Weighted inspection coverage")
    ax.set_xlabel("Wind condition")
    ax.set_ylim(0, max(0.12, min(1.0, ax.get_ylim()[1])))
    ax.set_title("Coverage degradation under increasing wind severity")
    ax.legend(ncol=2, fontsize=7.2, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "coverage_vs_wind.png")
    plt.close(fig)


def make_safety_vs_wind(df: pd.DataFrame, fig_dir: Path) -> None:
    methods = _ordered_methods(df["planner"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), dpi=180, sharex=True)
    x = np.arange(len(WIND_ORDER))
    for method in methods:
        b_rates = []
        c_rates = []
        for wind in WIND_ORDER:
            sub = df[(df["wind_level"] == wind) & (df["planner"] == method)]
            b_rates.append(sub["battery_violation"].mean())
            c_rates.append(sub["clearance_violation_rate"].mean())
        axes[0].plot(x, b_rates, marker="o", linewidth=1.6, color=METHOD_COLORS.get(method), label=method)
        axes[1].plot(x, c_rates, marker="o", linewidth=1.6, color=METHOD_COLORS.get(method), label=method)
    axes[0].set_ylabel("Battery violation rate")
    axes[1].set_ylabel("Expected clearance violation rate")
    for ax in axes:
        ax.set_xticks(x, WIND_ORDER)
        ax.set_xlabel("Wind condition")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_title("Return reliability")
    axes[1].set_title("Obstacle-clearance safety")
    axes[1].legend(ncol=1, fontsize=6.9, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(fig_dir / "safety_vs_wind.png")
    plt.close(fig)


def make_pareto(df: pd.DataFrame, fig_dir: Path) -> None:
    grouped = df.groupby("planner", sort=False).agg(
        weighted_coverage=("weighted_coverage", "mean"),
        clearance_violation_rate=("clearance_violation_rate", "mean"),
        battery_violation=("battery_violation", "mean"),
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=180)
    for method, row in grouped.iterrows():
        size = 80 + 260 * row["battery_violation"]
        ax.scatter(
            row["clearance_violation_rate"],
            row["weighted_coverage"],
            s=size,
            color=METHOD_COLORS.get(method),
            alpha=0.85,
            edgecolor="white",
            linewidth=0.8,
        )
        ax.text(row["clearance_violation_rate"] + 0.002, row["weighted_coverage"], method, fontsize=7)
    ax.set_xlabel("Expected clearance violation rate")
    ax.set_ylabel("Weighted inspection coverage")
    ax.set_title("Coverage-risk trade-off across planners")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "coverage_risk_pareto.png")
    plt.close(fig)


def make_ablation(df: pd.DataFrame, fig_dir: Path) -> None:
    methods = ["RAWA-RHP", "RAWA-NoRisk", "RAWA-NoWind"]
    sub = df[df["planner"].isin(methods)]
    grouped = sub.groupby("planner", sort=False).agg(
        weighted_coverage=("weighted_coverage", "mean"),
        clearance_violation_rate=("clearance_violation_rate", "mean"),
    )
    x = np.arange(len(methods))
    fig, ax1 = plt.subplots(figsize=(6.8, 4.0), dpi=180)
    vals = [grouped.loc[m, "weighted_coverage"] for m in methods if m in grouped.index]
    labels = [m for m in methods if m in grouped.index]
    x = np.arange(len(labels))
    ax1.bar(x - 0.18, vals, width=0.36, color="#2ca02c", label="Weighted coverage")
    ax2 = ax1.twinx()
    risks = [grouped.loc[m, "clearance_violation_rate"] for m in labels]
    ax2.bar(x + 0.18, risks, width=0.36, color="#d62728", label="Clearance violation")
    ax1.set_xticks(x, labels, rotation=15, ha="right")
    ax1.set_ylabel("Weighted inspection coverage")
    ax2.set_ylabel("Expected clearance violation rate")
    ax1.set_title("Contribution of risk and wind-awareness modules")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / "ablation_study.png")
    plt.close(fig)


def _draw_environment(ax, env: InspectionEnvironment) -> None:
    for rect in env.obstacles:
        patch = plt.Rectangle(
            (rect.xmin, rect.ymin),
            rect.width,
            rect.height,
            facecolor="#4d4d4d",
            edgecolor="#111111",
            alpha=0.82,
            label="structure/obstacle",
        )
        ax.add_patch(patch)
    pts = np.array([t.point for t in env.targets])
    ax.scatter(pts[:, 0], pts[:, 1], s=22, c="#f4a261", edgecolors="#5f3b00", linewidths=0.4, label="inspection target")
    ax.scatter([env.depot[0]], [env.depot[1]], marker="s", s=58, c="#2a9d8f", edgecolors="black", linewidths=0.5, label="depot")
    ax.set_xlim(0, env.cfg.world_size)
    ax.set_ylim(0, env.cfg.world_size)
    ax.set_aspect("equal", adjustable="box")
    handles, labels = ax.get_legend_handles_labels()
    dedup = {}
    for h, label in zip(handles, labels):
        dedup.setdefault(label, h)
    ax.legend(dedup.values(), dedup.keys(), fontsize=7.5, loc="upper right", frameon=False)


def _ordered_methods(methods: Iterable[str]) -> List[str]:
    order = list(METHOD_COLORS.keys())
    method_set = set(methods)
    return [m for m in order if m in method_set] + sorted(method_set - set(order))


def _ci95(values: np.ndarray) -> float:
    if len(values) <= 1:
        return 0.0
    return float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RAWA-RHP result figures.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_plots(args.output_root)
    print(f"Wrote figures to {args.output_root / 'figures'}")


if __name__ == "__main__":
    main()

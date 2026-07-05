from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


MAIN_PLANNERS = [
    "NearestNeighbor",
    "ValuePerDistance",
    "WindAwareGreedy",
    "ReserveOnlyPlanner",
    "FairRiskAwareGreedy",
    "FairALNSPlanner",
    "RAWA-RHP",
]
ABLATION_PLANNERS = [
    "RAWA-RHP",
    "RAWA-NoPacking",
    "RAWA-NoRisk",
    "RAWA-NoBeam",
    "RAWA-NoAdaptiveSearch",
    "RAWA-NoPacking-EqBudget",
    "RAWA-NoRisk-EqBudget",
    "RAWA-NoBeam-EqBudget",
    "RAWA-NoAdaptiveSearch-EqBudget",
    "RAWA-NoWind",
]
STRESS_PLANNERS = ["ReserveOnlyPlanner", "RAWA-RHP"]


def summarize_episode_results(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["wind_level", "obstacle_density", "planner"]
    continuous_cols = [
        "weighted_coverage",
        "safe_weighted_coverage",
        "coverage_ratio",
        "energy_used",
        "planned_policy_budget",
        "final_battery",
        "reserve_margin",
        "mission_time",
        "path_risk_accumulated",
        "clearance_violation_rate",
        "reserve_shortfall_probability",
        "runtime_seconds",
        "planner_replan_budget_seconds",
        "replan_count",
        "replan_latency_mean",
        "replan_latency_p95",
        "replan_latency_p99",
        "replan_latency_max",
        "candidate_expansions",
        "risk_evals",
        "unified_eval_count",
        "cache_hit_rate",
        "pruning_ratio",
        "packing_attempts",
        "packing_accepts",
        "packing_added_value",
        "packing_attempts_per_replan",
        "packing_accepts_per_replan",
        "packing_added_value_per_replan",
        "packing_added_value_per_total_value",
        "packing_accept_rate",
    ]
    tail_cols = {"runtime_seconds", "replan_latency_mean", "replan_latency_p95", "replan_latency_p99", "replan_latency_max"}
    rate_cols = [
        "return_success",
        "return_failure",
        "mission_success",
        "battery_violation",
        "reserve_shortfall",
        "emergency_abort",
        "clearance_violation",
    ]
    for keys, group in df.groupby(group_cols, sort=False):
        row = dict(zip(group_cols, keys))
        for col in continuous_cols:
            if col not in group.columns:
                continue
            vals = group[col].astype(float).to_numpy()
            row[f"{col}_mean"] = float(np.mean(vals))
            row[f"{col}_ci95"] = _ci95(vals)
            if col in tail_cols:
                row[f"{col}_q95"] = float(np.quantile(vals, 0.95))
                row[f"{col}_q99"] = float(np.quantile(vals, 0.99))
                row[f"{col}_max"] = float(np.max(vals))
        for col in rate_cols:
            if col not in group.columns:
                continue
            vals = group[col].astype(float).to_numpy()
            row[f"{col}_rate"] = float(np.mean(vals))
        row["n"] = int(len(group))
        rows.append(row)
    return pd.DataFrame(rows)


def write_tables(summary: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    main = summary[summary["planner"].isin(MAIN_PLANNERS)].copy()
    ablation = summary[summary["planner"].isin(ABLATION_PLANNERS)].copy()
    runtime = summary.copy()
    _write_markdown_table(
        main,
        out_dir / "main_results.md",
        [
            "wind_level",
            "obstacle_density",
            "planner",
            "weighted_coverage_mean",
            "safe_weighted_coverage_mean",
            "mission_success_rate",
            "return_success_rate",
            "battery_violation_rate",
            "clearance_violation_rate_mean",
            "reserve_shortfall_rate",
            "reserve_margin_mean",
        ],
    )
    _write_markdown_table(
        ablation,
        out_dir / "ablation_results.md",
        [
            "wind_level",
            "obstacle_density",
            "planner",
            "weighted_coverage_mean",
            "safe_weighted_coverage_mean",
            "mission_success_rate",
            "battery_violation_rate",
            "reserve_shortfall_rate",
            "clearance_violation_rate_mean",
            "path_risk_accumulated_mean",
        ],
    )
    _write_markdown_table(
        runtime,
        out_dir / "runtime_results.md",
        [
            "wind_level",
            "obstacle_density",
            "planner",
            "runtime_seconds_mean",
            "runtime_seconds_ci95",
            "runtime_seconds_q95",
            "mission_time_mean",
            "replan_latency_p95_q95",
            "replan_latency_p99_q95",
            "replan_latency_max_max",
        ],
    )
    write_constrained_ranking(summary, out_dir / "constrained_ranking.md")


def write_stress_tables(summary: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stress = summary[summary["planner"].isin(STRESS_PLANNERS)].copy()
    _write_markdown_table(
        stress,
        out_dir / "stress_results.md",
        [
            "wind_level",
            "obstacle_density",
            "planner",
            "weighted_coverage_mean",
            "safe_weighted_coverage_mean",
            "mission_success_rate",
            "battery_violation_rate",
            "reserve_shortfall_rate",
            "reserve_margin_mean",
            "clearance_violation_rate_mean",
        ],
    )


def write_constrained_ranking(summary: pd.DataFrame, path: Path) -> None:
    rows = []
    for (wind, density), group in summary.groupby(["wind_level", "obstacle_density"], sort=False):
        for threshold in [0.005, 0.01, 0.05]:
            masks = [
                group["battery_violation_rate"] <= 0.05,
                group["clearance_violation_rate_mean"] <= threshold,
            ]
            if "reserve_shortfall_rate" in group.columns:
                masks.append(group["reserve_shortfall_rate"] <= 0.05)
            eligible_mask = masks[0]
            for mask in masks[1:]:
                eligible_mask = eligible_mask & mask
            eligible = group[eligible_mask].copy()
            if len(eligible):
                eligible = eligible.sort_values("safe_weighted_coverage_mean", ascending=False).reset_index(drop=True)
                best_method = str(eligible.loc[0, "planner"])
                best_safe = float(eligible.loc[0, "safe_weighted_coverage_mean"])
                rank_lookup = {planner: i + 1 for i, planner in enumerate(eligible["planner"])}
            else:
                best_method = "none"
                best_safe = 0.0
                rank_lookup = {}
            rawa = group[group["planner"] == "RAWA-RHP"]
            reserve = group[group["planner"] == "ReserveOnlyPlanner"]
            if len(rawa):
                rawa_safe = float(rawa["safe_weighted_coverage_mean"].iloc[0])
                rawa_risk = float(rawa["clearance_violation_rate_mean"].iloc[0])
                rawa_rank = rank_lookup.get("RAWA-RHP", 0)
            else:
                rawa_safe = 0.0
                rawa_risk = 0.0
                rawa_rank = 0
            if len(reserve):
                reserve_risk = float(reserve["clearance_violation_rate_mean"].iloc[0])
                risk_reduction = (reserve_risk - rawa_risk) / max(reserve_risk, 1e-12)
            else:
                risk_reduction = 0.0
            rows.append(
                {
                    "wind_level": wind,
                    "obstacle_density": density,
                    "clearance_threshold": threshold,
                    "robust_constrained_best_method": best_method,
                    "RAWA_rank": rawa_rank,
                    "RAWA_safe_weighted_coverage": rawa_safe,
                    "coverage_gap_to_best": rawa_safe - best_safe,
                    "risk_reduction_vs_reserve": risk_reduction,
                }
            )
    _write_markdown_table(pd.DataFrame(rows), path, list(rows[0].keys()) if rows else [])


def _write_markdown_table(df: pd.DataFrame, path: Path, cols: List[str]) -> None:
    if not cols:
        path.write_text("No rows.\n")
        return
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for {path}: {missing}")
    view = df.loc[:, cols].copy()
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in view.iterrows():
        cells = [_format_cell(row[col]) for col in cols]
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def _format_cell(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _ci95(values: Iterable[float]) -> float:
    vals = np.asarray(list(values), dtype=float)
    if len(vals) <= 1:
        return 0.0
    return float(1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals)))

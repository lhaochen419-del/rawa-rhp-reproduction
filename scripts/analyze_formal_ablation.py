from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ABLATIONS = [
    "RAWA-NoPacking",
    "RAWA-NoPacking-EqTime",
    "RAWA-NoPacking-EqEval",
    "RAWA-NoBeam",
    "RAWA-NoBeam-EqTime",
    "RAWA-NoBeam-EqEval",
    "RAWA-NoRisk",
    "RAWA-NoRisk-EqTime",
    "RAWA-NoRisk-EqEval",
    "RAWA-BlindReserve",
    "RAWA-BlindReserve-EqTime",
    "RAWA-BlindReserve-EqEval",
    "RAWA-NoAdaptiveSearch",
    "RAWA-NoAdaptiveSearch-EqTime",
    "RAWA-NoAdaptiveSearch-EqEval",
]

DEFAULT_KEYS = [
    "scenario_seed",
    "seed",
    "wind_level",
    "obstacle_density",
    "actual_gust_multiplier",
    "scenario_profile",
    "ood_category",
]

SAFETY_COLS = [
    "battery_violation",
    "clearance_violation",
    "reserve_shortfall",
    "return_failure",
    "emergency_abort",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze paired formal RAWA-RHP ablations.")
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--main", default="RAWA-RHP")
    parser.add_argument("--ablations", default=",".join(DEFAULT_ABLATIONS))
    parser.add_argument("--metric", default="safe_weighted_coverage")
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--budget-tolerance", type=float, default=0.05)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--permutations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze_episodes(
        args.episodes,
        args.main,
        [item.strip() for item in args.ablations.split(",") if item.strip()],
        args.metric,
        args.threshold,
        args.budget_tolerance,
        args.bootstrap,
        args.permutations,
        args.seed,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(_format_markdown(report))
    if not report["formal_gate_pass"]:
        raise SystemExit("Formal ablation gate failed; see report for details.")


def analyze_episodes(
    episodes: Path,
    main: str = "RAWA-RHP",
    ablations: list[str] | None = None,
    metric: str = "safe_weighted_coverage",
    threshold: float = 0.01,
    budget_tolerance: float = 0.05,
    bootstrap: int = 10000,
    permutations: int = 20000,
    seed: int = 20260620,
) -> dict[str, object]:
    df = pd.read_csv(episodes)
    if ablations is None:
        ablations = list(DEFAULT_ABLATIONS)
    keys = _pairing_keys(df)
    if not keys:
        raise SystemExit("No pairing keys found in episode CSV.")
    rng = np.random.default_rng(seed)
    duplicate_rows = _duplicate_count(df, keys)
    expected = _expected_combinations(df, keys, [main] + ablations)
    rows = []
    for ablation in ablations:
        paired = _paired_diffs(df, keys, main, ablation, metric)
        if paired.empty:
            rows.append(_empty_row(ablation))
            continue
        diffs = paired["diff"].astype(float).to_numpy()
        cluster_col = "scenario_seed" if "scenario_seed" in paired.columns else None
        p_raw = _paired_signflip_pvalue(paired, permutations, rng, cluster_col)
        low, high = _bootstrap_ci(paired, bootstrap, rng, cluster_col)
        budget_check = _budget_check(df, keys, main, ablation, budget_tolerance)
        row = {
            "ablation": ablation,
            "n_pairs": int(len(diffs)),
            "mean_diff": float(np.mean(diffs)),
            "median_diff": float(np.median(diffs)),
            "ci95_low": float(low),
            "ci95_high": float(high),
            "p_raw": float(p_raw),
            "direction_id": _direction_mean(paired, "scenario_profile", "id"),
            "direction_correlated_gust": _direction_contains(paired, "ood_category", "correlated_gust"),
            "direction_narrow_passage": _direction_contains(paired, "ood_category", "narrow_passage"),
            "direction_cluttered": _direction_contains(paired, "obstacle_density", "cluttered"),
            **budget_check,
        }
        for col in SAFETY_COLS:
            if col in df.columns:
                row[f"{col}_delta_rate"] = _rate_delta(df, keys, main, ablation, col)
        rows.append(row)
    rows = _holm_correct(rows)
    for row in rows:
        row["passes"] = bool(
            row.get("mean_diff", 0.0) >= threshold
            and row.get("ci95_low", -1.0) > 0.0
            and row.get("p_holm", 1.0) < 0.05
            and row.get("direction_id", 0.0) > 0.0
            and row.get("direction_correlated_gust", 0.0) > 0.0
            and row.get("direction_narrow_passage", 0.0) > 0.0
            and row.get("direction_cluttered", 0.0) > 0.0
            and row.get("budget_check_pass", True)
            and all(row.get(f"{col}_delta_rate", 0.0) <= 0.0 for col in SAFETY_COLS)
        )
    actual_rows = int(len(df[df["planner"].isin([main] + ablations)]))
    safety_gate = _main_safety_gate(df, main)
    latency_gate = _latency_gate(df, main)
    completeness_gate = {
        "duplicate_rows": int(duplicate_rows),
        "expected_rows": int(expected),
        "actual_rows": int(actual_rows),
        "passes": bool(duplicate_rows == 0 and actual_rows == expected),
    }
    formal_gate_pass = bool(completeness_gate["passes"] and safety_gate["passes"] and latency_gate["passes"] and all(row.get("passes") for row in rows))
    report = {
        "episodes": str(episodes),
        "metric": metric,
        "threshold": threshold,
        "pairing_keys": keys,
        "cluster_key": "scenario_seed" if "scenario_seed" in df.columns else None,
        "budget_tolerance": budget_tolerance,
        "duplicate_rows": int(duplicate_rows),
        "expected_method_seed_scene_rows": int(expected),
        "actual_rows": int(actual_rows),
        "completeness_gate": completeness_gate,
        "main_safety_gate": safety_gate,
        "latency_gate": latency_gate,
        "formal_gate_pass": formal_gate_pass,
        "results": rows,
    }
    return report


def _duplicate_count(df: pd.DataFrame, keys: list[str]) -> int:
    subset = keys + ["planner"]
    return int(df.duplicated(subset=subset).sum())


def _expected_combinations(df: pd.DataFrame, keys: list[str], planners: list[str]) -> int:
    scenes = df[keys].drop_duplicates()
    return int(len(scenes) * len(planners))


def _pairing_keys(df: pd.DataFrame) -> list[str]:
    keys = [col for col in DEFAULT_KEYS if col in df.columns]
    if "scenario_seed" in keys and "seed" in keys:
        keys.remove("seed")
    return keys


def _paired_diffs(df: pd.DataFrame, keys: list[str], main: str, ablation: str, metric: str) -> pd.DataFrame:
    view = df[df["planner"].isin([main, ablation])].copy()
    pivot = view.pivot_table(index=keys, columns="planner", values=metric, aggfunc="mean")
    if main not in pivot or ablation not in pivot:
        return pd.DataFrame()
    paired = pivot[[main, ablation]].dropna().reset_index()
    paired["diff"] = paired[main] - paired[ablation]
    return paired


def _bootstrap_ci(paired: pd.DataFrame, n_boot: int, rng: np.random.Generator, cluster_col: str | None = None) -> tuple[float, float]:
    values = paired["diff"].astype(float).to_numpy()
    if len(values) == 0:
        return 0.0, 0.0
    if len(values) == 1:
        value = float(values[0])
        return value, value
    if cluster_col and cluster_col in paired.columns:
        cluster_stats = paired.groupby(cluster_col, sort=False)["diff"].agg(["sum", "count"]).reset_index()
        sums = cluster_stats["sum"].astype(float).to_numpy()
        counts = cluster_stats["count"].astype(float).to_numpy()
        idx = rng.integers(0, len(sums), size=(int(n_boot), len(sums)))
        means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    else:
        samples = rng.choice(values, size=(int(n_boot), len(values)), replace=True)
        means = np.mean(samples, axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _paired_signflip_pvalue(paired: pd.DataFrame, n_perm: int, rng: np.random.Generator, cluster_col: str | None = None) -> float:
    values = paired["diff"].astype(float).to_numpy()
    if len(values) == 0:
        return 1.0
    observed = float(np.mean(values))
    if np.isclose(observed, 0.0):
        return 1.0
    if cluster_col and cluster_col in paired.columns:
        clusters = paired[cluster_col].drop_duplicates().to_numpy()
        cluster_index = {cluster: idx for idx, cluster in enumerate(clusters)}
        row_cluster = paired[cluster_col].map(cluster_index).to_numpy()
        signs = rng.choice(np.array([-1.0, 1.0]), size=(int(n_perm), len(clusters)), replace=True)
        null = np.mean(signs[:, row_cluster] * values[None, :], axis=1)
    else:
        signs = rng.choice(np.array([-1.0, 1.0]), size=(int(n_perm), len(values)), replace=True)
        null = np.mean(signs * values, axis=1)
    return float((np.count_nonzero(np.abs(null) >= abs(observed)) + 1) / (len(null) + 1))


def _holm_correct(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    valid = [(idx, float(row.get("p_raw", 1.0))) for idx, row in enumerate(rows)]
    valid.sort(key=lambda item: item[1])
    m = len(valid)
    adjusted = [1.0] * len(rows)
    running = 0.0
    for rank, (idx, pvalue) in enumerate(valid):
        corrected = min(1.0, (m - rank) * pvalue)
        running = max(running, corrected)
        adjusted[idx] = running
    for idx, value in enumerate(adjusted):
        rows[idx]["p_holm"] = float(value)
    return rows


def _direction_mean(paired: pd.DataFrame, col: str, value: str) -> float:
    if col not in paired:
        return 0.0
    subset = paired[paired[col] == value]
    return float(subset["diff"].mean()) if len(subset) else 0.0


def _direction_contains(paired: pd.DataFrame, col: str, value: str) -> float:
    if col not in paired:
        return 0.0
    subset = paired[paired[col].astype(str).str.contains(value, regex=False)]
    return float(subset["diff"].mean()) if len(subset) else 0.0


def _rate_delta(df: pd.DataFrame, keys: list[str], main: str, ablation: str, col: str) -> float:
    paired = _paired_diffs(df, keys, main, ablation, col)
    if paired.empty:
        return 0.0
    return float(paired["diff"].mean())


def _main_safety_gate(df: pd.DataFrame, main: str) -> dict[str, object]:
    view = df[df["planner"] == main]
    rates = {}
    for col in SAFETY_COLS:
        if col in view.columns:
            rates[col] = float(view[col].astype(float).mean()) if len(view) else 1.0
    passes = bool(len(view) > 0 and all(value == 0.0 for value in rates.values()))
    return {"rates": rates, "passes": passes}


def _latency_gate(df: pd.DataFrame, main: str, p95_limit: float = 1.8, p99_limit: float = 2.2) -> dict[str, object]:
    view = df[df["planner"] == main]
    if view.empty or "replan_latency_p95" not in view.columns or "replan_latency_p99" not in view.columns:
        return {"p95_max": 0.0, "p99_max": 0.0, "p95_limit": p95_limit, "p99_limit": p99_limit, "passes": False}
    p95_max = float(view["replan_latency_p95"].astype(float).max())
    p99_max = float(view["replan_latency_p99"].astype(float).max())
    return {
        "p95_max": p95_max,
        "p99_max": p99_max,
        "p95_limit": p95_limit,
        "p99_limit": p99_limit,
        "passes": bool(p95_max <= p95_limit and p99_max <= p99_limit),
    }


def _budget_check(df: pd.DataFrame, keys: list[str], main: str, ablation: str, tolerance: float) -> dict[str, object]:
    if ablation.endswith("-EqTime"):
        metric = "replan_latency_p95"
    else:
        metric = ""
    if ablation.endswith("-EqEval"):
        return _eqeval_budget_check(df, keys, main, ablation, tolerance)
    if not metric:
        return {"budget_check_metric": "not_applicable", "budget_ratio": 1.0, "budget_check_pass": True}
    if metric not in df.columns:
        return {"budget_check_metric": metric, "budget_ratio": 0.0, "budget_check_pass": False}
    paired = _paired_values(df, keys, main, ablation, metric)
    if paired.empty:
        return {"budget_check_metric": metric, "budget_ratio": 0.0, "budget_check_pass": False}
    main_mean = float(paired[main].astype(float).mean())
    ablation_mean = float(paired[ablation].astype(float).mean())
    ratio = ablation_mean / max(main_mean, 1e-12)
    return {
        "budget_check_metric": metric,
        "budget_ratio": float(ratio),
        "budget_check_pass": bool((1.0 - tolerance) <= ratio <= (1.0 + tolerance)),
    }


def _eqeval_budget_check(df: pd.DataFrame, keys: list[str], main: str, ablation: str, tolerance: float) -> dict[str, object]:
    metrics = [field for field in ["unified_eval_count", "candidate_expansions", "risk_evals", "packing_attempts"] if field in df.columns]
    if not metrics:
        return {
            "budget_check_metric": "unified_or_components",
            "budget_ratio": 0.0,
            "budget_component_ratios": {},
            "budget_check_pass": False,
        }
    ratios: dict[str, float] = {}
    passes: dict[str, bool] = {}
    for metric in metrics:
        paired = _paired_values(df, keys, main, ablation, metric)
        if paired.empty:
            ratios[metric] = 0.0
            passes[metric] = False
            continue
        main_mean = float(paired[main].astype(float).mean())
        ablation_mean = float(paired[ablation].astype(float).mean())
        if main_mean == 0.0:
            ratio = 1.0 if ablation_mean == 0.0 else float("inf")
        else:
            ratio = ablation_mean / main_mean
        ratios[metric] = float(ratio)
        passes[metric] = bool((1.0 - tolerance) <= ratio <= (1.0 + tolerance))
    unified_pass = passes.get("unified_eval_count", False)
    component_metrics = [field for field in ["candidate_expansions", "risk_evals", "packing_attempts"] if field in passes]
    components_pass = bool(component_metrics and all(passes[field] for field in component_metrics))
    display_ratio = ratios.get("unified_eval_count")
    if display_ratio is None:
        display_ratio = float(np.mean([ratios[field] for field in component_metrics])) if component_metrics else 0.0
    return {
        "budget_check_metric": "unified_or_components",
        "budget_ratio": float(display_ratio),
        "budget_component_ratios": ratios,
        "budget_component_passes": passes,
        "budget_check_pass": bool(unified_pass or components_pass),
    }


def _paired_values(df: pd.DataFrame, keys: list[str], main: str, ablation: str, metric: str) -> pd.DataFrame:
    view = df[df["planner"].isin([main, ablation])].copy()
    pivot = view.pivot_table(index=keys, columns="planner", values=metric, aggfunc="mean")
    if main not in pivot or ablation not in pivot:
        return pd.DataFrame()
    return pivot[[main, ablation]].dropna().reset_index()


def _empty_row(ablation: str) -> dict[str, object]:
    return {
        "ablation": ablation,
        "n_pairs": 0,
        "mean_diff": 0.0,
        "median_diff": 0.0,
        "ci95_low": 0.0,
        "ci95_high": 0.0,
        "p_raw": 1.0,
        "p_holm": 1.0,
        "budget_check_metric": "not_applicable",
        "budget_ratio": 0.0,
        "budget_check_pass": False,
    }


def _format_markdown(report: dict[str, object]) -> str:
    rows = list(report["results"])
    lines = [
        "# Formal Ablation Analysis",
        "",
        f"- episodes: `{report['episodes']}`",
        f"- metric: `{report['metric']}`",
        f"- threshold: `{report['threshold']}`",
        f"- pairing keys: `{', '.join(report['pairing_keys'])}`",
        f"- cluster key: `{report.get('cluster_key')}`",
        f"- EqTime/EqEval tolerance: `{report.get('budget_tolerance')}`",
        f"- duplicate method-scene rows: `{report['duplicate_rows']}`",
        f"- expected rows: `{report['expected_method_seed_scene_rows']}`",
        f"- actual rows: `{report['actual_rows']}`",
        f"- completeness gate: `{'pass' if report.get('completeness_gate', {}).get('passes') else 'fail'}`",
        f"- RAWA hard safety gate: `{'pass' if report.get('main_safety_gate', {}).get('passes') else 'fail'}`",
        f"- RAWA latency gate: `{'pass' if report.get('latency_gate', {}).get('passes') else 'fail'}`",
        f"- formal gate: `{'pass' if report.get('formal_gate_pass') else 'fail'}`",
        "",
        "| ablation | n | mean diff | median diff | ci95 low | ci95 high | p holm | budget ratio | pass |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|:--|",
    ]
    for row in rows:
        lines.append(
            "| {ablation} | {n_pairs} | {mean_diff:.6f} | {median_diff:.6f} | {ci95_low:.6f} | {ci95_high:.6f} | {p_holm:.6f} | {budget_ratio:.6f} | {passes} |".format(
                **{**row, "passes": "yes" if row.get("passes") else "no"}
            )
        )
    lines.append("")
    lines.append("A module passes only if mean diff >= threshold, scenario-seed cluster bootstrap lower95 > 0, Holm-adjusted paired-test p < 0.05, directional means are positive in ID/correlated_gust/narrow_passage/cluttered subsets, EqTime/EqEval budget ratios are within tolerance, and safety-rate deltas are not positive.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_BASELINES = [
    "FairALNSPlanner",
    "FairRiskAwareGreedy",
    "ReserveOnlyPlanner",
]

DEFAULT_KEYS = [
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
    parser = argparse.ArgumentParser(description="Paired RAWA-RHP baseline comparison with clustered inference.")
    parser.add_argument("--episodes", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", default="", help="Optional comma labels matching --episodes.")
    parser.add_argument("--main", default="RAWA-RHP")
    parser.add_argument("--baselines", default=",".join(DEFAULT_BASELINES))
    parser.add_argument("--metric", default="safe_weighted_coverage")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--permutations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = _labels(args.labels, args.episodes)
    baselines = _split(args.baselines)
    rng = np.random.default_rng(args.seed)
    reports = []
    for label, path in zip(labels, args.episodes):
        df = _prepare(pd.read_csv(path))
        reports.append(_analyze_one(df, label, path, args.main, baselines, args.metric, args.bootstrap, args.permutations, rng))
    report = {
        "metric": args.metric,
        "main": args.main,
        "baselines": baselines,
        "reports": reports,
        "overall_pass": bool(all(item["passes"] for item in reports)),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(_format_markdown(report))
    if not report["overall_pass"]:
        raise SystemExit("Baseline comparison gate failed; see report for details.")


def _analyze_one(
    df: pd.DataFrame,
    label: str,
    path: Path,
    main: str,
    baselines: list[str],
    metric: str,
    bootstrap: int,
    permutations: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    keys = [key for key in DEFAULT_KEYS if key in df.columns]
    if not keys:
        raise SystemExit(f"No pairing keys found for {path}")
    rows = []
    for baseline in baselines:
        paired = _paired_diffs(df, keys, main, baseline, metric)
        if paired.empty:
            rows.append(_empty_row(baseline))
            continue
        cluster_col = "seed" if "seed" in paired.columns else None
        low, high = _bootstrap_ci(paired, bootstrap, rng, cluster_col)
        p_raw = _paired_signflip_pvalue(paired, permutations, rng, cluster_col)
        row = {
            "baseline": baseline,
            "n_pairs": int(len(paired)),
            "mean_diff": float(paired["diff"].mean()),
            "median_diff": float(paired["diff"].median()),
            "ci95_low": float(low),
            "ci95_high": float(high),
            "p_raw": float(p_raw),
        }
        for col in SAFETY_COLS:
            if col in df.columns:
                row[f"{col}_delta_rate"] = _rate_delta(df, keys, main, baseline, col)
        rows.append(row)
    rows = _holm_correct(rows)
    for row in rows:
        row["safety_not_worse"] = bool(all(row.get(f"{col}_delta_rate", 0.0) <= 0.0 for col in SAFETY_COLS))
        row["passes"] = bool(row.get("mean_diff", 0.0) > 0.0 and row.get("ci95_low", -1.0) > 0.0 and row.get("p_holm", 1.0) < 0.05 and row["safety_not_worse"])
    strongest = max(rows, key=lambda item: item.get("baseline_mean", -np.inf) if "baseline_mean" in item else item.get("mean_diff", -np.inf), default={})
    return {
        "label": label,
        "episodes": str(path),
        "pairing_keys": keys,
        "cluster_key": "seed" if "seed" in keys else None,
        "main_mean": float(df.loc[df["planner"] == main, metric].astype(float).mean()),
        "results": rows,
        "strongest_baseline_by_mean": _strongest_baseline(df, baselines, metric),
        "passes": bool(all(row.get("passes") for row in rows)),
    }


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "clearance_violation" not in out.columns:
        if "clearance_violation_rate" in out.columns:
            out["clearance_violation"] = out["clearance_violation_rate"].astype(float) > 0.05
        else:
            out["clearance_violation"] = False
    if "return_failure" not in out.columns:
        if "return_success" in out.columns:
            out["return_failure"] = ~out["return_success"].astype(bool)
        else:
            out["return_failure"] = False
    return out


def _paired_diffs(df: pd.DataFrame, keys: list[str], main: str, baseline: str, metric: str) -> pd.DataFrame:
    view = df[df["planner"].isin([main, baseline])].copy()
    pivot = view.pivot_table(index=keys, columns="planner", values=metric, aggfunc="mean")
    if main not in pivot or baseline not in pivot:
        return pd.DataFrame()
    paired = pivot[[main, baseline]].dropna().reset_index()
    paired["diff"] = paired[main] - paired[baseline]
    return paired


def _bootstrap_ci(paired: pd.DataFrame, n_boot: int, rng: np.random.Generator, cluster_col: str | None) -> tuple[float, float]:
    values = paired["diff"].astype(float).to_numpy()
    if len(values) == 0:
        return 0.0, 0.0
    if cluster_col and cluster_col in paired.columns:
        cluster_stats = paired.groupby(cluster_col, sort=False)["diff"].agg(["sum", "count"]).reset_index()
        sums = cluster_stats["sum"].astype(float).to_numpy()
        counts = cluster_stats["count"].astype(float).to_numpy()
        idx = rng.integers(0, len(sums), size=(int(n_boot), len(sums)))
        samples = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    else:
        samples = np.mean(rng.choice(values, size=(int(n_boot), len(values)), replace=True), axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _paired_signflip_pvalue(paired: pd.DataFrame, n_perm: int, rng: np.random.Generator, cluster_col: str | None) -> float:
    values = paired["diff"].astype(float).to_numpy()
    if len(values) == 0:
        return 1.0
    observed = float(np.mean(values))
    if np.isclose(observed, 0.0):
        return 1.0
    if cluster_col and cluster_col in paired.columns:
        clusters = paired[cluster_col].drop_duplicates().to_list()
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
    adjusted = [1.0] * len(rows)
    running = 0.0
    m = len(valid)
    for rank, (idx, pvalue) in enumerate(valid):
        corrected = min(1.0, (m - rank) * pvalue)
        running = max(running, corrected)
        adjusted[idx] = running
    for idx, value in enumerate(adjusted):
        rows[idx]["p_holm"] = float(value)
    return rows


def _rate_delta(df: pd.DataFrame, keys: list[str], main: str, baseline: str, col: str) -> float:
    paired = _paired_diffs(df, keys, main, baseline, col)
    if paired.empty:
        return 0.0
    return float(paired["diff"].mean())


def _strongest_baseline(df: pd.DataFrame, baselines: list[str], metric: str) -> dict[str, object]:
    means = df[df["planner"].isin(baselines)].groupby("planner")[metric].mean().sort_values(ascending=False)
    if means.empty:
        return {"planner": "", "mean": 0.0}
    return {"planner": str(means.index[0]), "mean": float(means.iloc[0])}


def _empty_row(baseline: str) -> dict[str, object]:
    return {
        "baseline": baseline,
        "n_pairs": 0,
        "mean_diff": 0.0,
        "median_diff": 0.0,
        "ci95_low": 0.0,
        "ci95_high": 0.0,
        "p_raw": 1.0,
        "p_holm": 1.0,
        "passes": False,
    }


def _format_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Baseline Paired Comparison",
        "",
        f"- metric: `{report['metric']}`",
        f"- main planner: `{report['main']}`",
        f"- overall gate: `{'pass' if report['overall_pass'] else 'fail'}`",
        "",
    ]
    for item in report["reports"]:
        strongest = item["strongest_baseline_by_mean"]
        lines.extend(
            [
                f"## {item['label']}",
                "",
                f"- main mean: `{item['main_mean']:.6f}`",
                f"- strongest baseline: `{strongest['planner']}` mean `{strongest['mean']:.6f}`",
                f"- gate: `{'pass' if item['passes'] else 'fail'}`",
                "",
                "| baseline | n | mean diff | median diff | ci95 low | ci95 high | Holm p | safety not worse | pass |",
                "|:--|--:|--:|--:|--:|--:|--:|:--|:--|",
            ]
        )
        for row in item["results"]:
            display = dict(row)
            display["safety"] = "yes" if row.get("safety_not_worse") else "no"
            display["passes_text"] = "yes" if row.get("passes") else "no"
            lines.append(
                "| {baseline} | {n_pairs} | {mean_diff:.6f} | {median_diff:.6f} | {ci95_low:.6f} | {ci95_high:.6f} | {p_holm:.6f} | {safety} | {passes_text} |".format(**display)
            )
        lines.append("")
    return "\n".join(lines)


def _labels(text: str, paths: list[Path]) -> list[str]:
    labels = _split(text)
    if labels:
        if len(labels) != len(paths):
            raise SystemExit("--labels must have the same length as --episodes.")
        return labels
    return [path.stem for path in paths]


def _split(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


if __name__ == "__main__":
    main()

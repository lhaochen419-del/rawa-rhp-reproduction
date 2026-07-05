from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate compact RAWA-RHP tables and figures from a reproduction package.")
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.package_root
    out = args.out_dir or (root / "generated")
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    _baseline(root, out)
    _ablation(root, out)
    _latency(root, out)
    _update_manifest(root)
    print(f"wrote regenerated tables/figures under {out}")


def _baseline(root: Path, out: Path) -> None:
    blind = pd.read_csv(root / "data" / "raw_results" / "baseline_blind100_episodes.csv")
    gust = pd.read_csv(root / "data" / "raw_results" / "baseline_gust3_episodes.csv")
    rows = []
    for label, df in [("Blind", blind), ("Gust x3", gust)]:
        for planner, sub in df.groupby("planner"):
            rows.append({"scenario": label, "planner": planner, "safe_weighted_coverage": float(sub["safe_weighted_coverage"].mean()), "n": int(len(sub))})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "tables" / "baseline_summary_regenerated.csv", index=False)
    planners = ["RAWA-RHP", "FairALNSPlanner", "FairRiskAwareGreedy", "ReserveOnlyPlanner"]
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    width = 0.18
    xs = [0, 1]
    for idx, planner in enumerate(planners):
        vals = []
        for scenario in ["Blind", "Gust x3"]:
            match = summary[(summary["scenario"] == scenario) & (summary["planner"] == planner)]
            vals.append(float(match["safe_weighted_coverage"].iloc[0]) if not match.empty else 0.0)
        ax.bar([x + (idx - 1.5) * width for x in xs], vals, width=width, label=planner)
    ax.set_xticks(xs, ["Blind", "Gust x3"])
    ax.set_ylabel("Safe weighted coverage")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "figures" / "baseline_comparison_regenerated.png", dpi=300)
    plt.close(fig)


def _ablation(root: Path, out: Path) -> None:
    data = json.loads((root / "data" / "processed_results" / "formal_ablation_analysis.json").read_text(encoding="utf-8"))
    df = pd.DataFrame(data["results"])
    df.to_csv(out / "tables" / "ablation_summary_regenerated.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    modules = ["NoPacking", "NoAdaptiveSearch", "BlindReserve", "NoRisk", "NoBeam"]
    budgets = ["Native", "EqTime", "EqEval"]
    width = 0.24
    for idx, budget in enumerate(budgets):
        vals = []
        for module in modules:
            label = f"RAWA-{module}" + ("" if budget == "Native" else f"-{budget}")
            vals.append(float(df.loc[df["ablation"] == label, "mean_diff"].iloc[0]))
        ax.bar([x + (idx - 1) * width for x in range(len(modules))], vals, width=width, label=budget)
    ax.axhline(0.01, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(range(len(modules)), modules, rotation=20, ha="right")
    ax.set_ylabel("RAWA-RHP improvement in SWC")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figures" / "ablation_contributions_regenerated.png", dpi=300)
    plt.close(fig)


def _latency(root: Path, out: Path) -> None:
    data = json.loads((root / "data" / "processed_results" / "latency_benchmark_summary.json").read_text(encoding="utf-8"))
    latency = pd.DataFrame([data["latency"]])
    latency.to_csv(out / "tables" / "latency_summary_regenerated.csv", index=False)
    labels = ["p50", "p95", "p99", "max"]
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.bar(labels, [data["latency"][key] for key in labels], color="#4c78a8")
    ax.axhline(data["p95_limit"], color="black", linestyle="--", linewidth=0.8, label="p95 limit")
    ax.axhline(data["p99_limit"], color="gray", linestyle=":", linewidth=0.8, label="p99 limit")
    ax.set_ylabel("Seconds")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figures" / "latency_benchmark_regenerated.png", dpi=300)
    plt.close(fig)


def _update_manifest(root: Path) -> None:
    manifest_path = root / "MANIFEST.json"
    if not manifest_path.exists():
        return
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())
    data["file_count"] = len(files)
    data["files"] = files
    data["python_after_regeneration"] = sys.version
    data["platform_after_regeneration"] = platform.platform()
    manifest_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

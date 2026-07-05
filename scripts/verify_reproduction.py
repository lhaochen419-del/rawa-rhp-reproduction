from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


EXPECTED = {
    "blind_gain": (0.1447430001652168, 2e-3),
    "gust3_gain": (0.201939, 2e-3),
    "latency_p95": (1.7548596779, 2e-3),
    "latency_p99": (1.75901256703, 2e-3),
}

SAFETY_COLS = ["battery_violation", "clearance_violation", "reserve_shortfall", "return_failure", "emergency_abort"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the RAWA-RHP reproduction package against manuscript key values.")
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.package_root
    checks: list[tuple[str, bool, str]] = []
    baseline_json = root / "data" / "processed_results" / "baseline_paired_comparison.json"
    ablation_json = root / "data" / "processed_results" / "formal_ablation_analysis.json"
    latency_json = root / "data" / "processed_results" / "latency_benchmark_summary.json"
    formal_episodes = root / "data" / "raw_results" / "formal_ablation_full_1_20_episodes.csv"
    for path in [baseline_json, ablation_json, latency_json, formal_episodes]:
        checks.append((f"exists:{path.relative_to(root)}", path.exists(), "file is present"))
    if baseline_json.exists():
        data = json.loads(baseline_json.read_text(encoding="utf-8"))
        blind = _baseline_gain(data, "blind100", "FairALNSPlanner")
        gust = _baseline_gain(data, "stress100_gust3", "FairALNSPlanner")
        if not np_isfinite(gust):
            gust = _baseline_gain(data, "stress100 gust 3.0", "FairALNSPlanner")
        checks.append(_close("blind_gain", blind))
        checks.append(_close("gust3_gain", gust))
    if ablation_json.exists():
        data = json.loads(ablation_json.read_text(encoding="utf-8"))
        checks.append(("formal_gate_pass", bool(data.get("formal_gate_pass")), "formal ablation gate"))
        for row in data.get("results", []):
            if str(row.get("ablation", "")).endswith("-EqEval"):
                ratio = float(row.get("budget_ratio", 0.0))
                checks.append((f"eqeval_budget:{row.get('ablation')}", 0.95 <= ratio <= 1.05, f"ratio={ratio:.6f}"))
    if latency_json.exists():
        data = json.loads(latency_json.read_text(encoding="utf-8"))
        checks.append(_close("latency_p95", float(data["latency"]["p95"])))
        checks.append(_close("latency_p99", float(data["latency"]["p99"])))
        checks.append(("latency_gate", bool(data.get("passes")), "strict latency benchmark gate"))
    if formal_episodes.exists():
        df = pd.read_csv(formal_episodes)
        main = df[df["planner"] == "RAWA-RHP"]
        for col in SAFETY_COLS:
            if col in main:
                rate = float(main[col].astype(bool).mean())
                checks.append((f"hard_failure_zero:{col}", rate == 0.0, f"rate={rate:.6f}"))
    failed = [item for item in checks if not item[1]]
    print("| check | status | detail |")
    print("|:--|:--|:--|")
    for name, ok, detail in checks:
        print(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")
    if failed:
        raise SystemExit(f"{len(failed)} reproduction checks failed")


def _baseline_gain(data: dict[str, object], label: str, baseline: str) -> float:
    for report in data.get("reports", []):
        if report.get("label") != label:
            continue
        for row in report.get("results", []):
            if row.get("baseline") == baseline:
                return float(row.get("mean_diff", 0.0))
    return float("nan")


def np_isfinite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


def _close(name: str, value: float) -> tuple[str, bool, str]:
    expected, tol = EXPECTED[name]
    ok = abs(float(value) - expected) <= tol
    return name, ok, f"value={value:.9f}, expected={expected:.9f}, tol={tol:g}"


if __name__ == "__main__":
    main()

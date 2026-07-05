from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import OBSTACLE_DENSITIES, WIND_LEVELS, stable_seed, load_config
from src.simulation import run_episode
from scripts.run_optimization_suite import _build_connected_env_edges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict RAWA-RHP latency benchmark on fixed scenario inputs.")
    parser.add_argument("--seeds", default="1-20")
    parser.add_argument("--planner", default="RAWA-RHP")
    parser.add_argument("--profiles", default="id,ood:correlated_gust,ood:narrow_passage")
    parser.add_argument("--wind-levels", default=",".join(WIND_LEVELS))
    parser.add_argument("--densities", default=",".join(OBSTACLE_DENSITIES))
    parser.add_argument("--gust", type=float, default=1.0)
    parser.add_argument("--warmup-seeds", default="901")
    parser.add_argument("--core", type=int, default=0)
    parser.add_argument("--deadline", type=float, default=1.8)
    parser.add_argument("--p95-limit", type=float, default=1.8)
    parser.add_argument("--p99-limit", type=float, default=2.2)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _pin_core(args.core)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    profiles = _split(args.profiles)
    wind_levels = _split(args.wind_levels)
    densities = _split(args.densities)
    warmup_specs = _specs(_parse_seeds(args.warmup_seeds), profiles, wind_levels, densities, args.gust)
    for spec in warmup_specs:
        _run_one(cfg, spec, args.planner, record_trace=False)
    rows = []
    traces = []
    checkpoint_path = args.out_dir / "latency_benchmark_episodes.checkpoint.csv"
    specs = _specs(_parse_seeds(args.seeds), profiles, wind_levels, densities, args.gust)
    start = time.perf_counter()
    for idx, spec in enumerate(specs, start=1):
        row = _run_one(cfg, spec, args.planner, record_trace=True)
        trace = _parse_trace(str(row.pop("replan_latency_trace", "")))
        traces.extend(trace)
        rows.append(row)
        if args.progress_every > 0 and (idx == 1 or idx % args.progress_every == 0 or idx == len(specs)):
            pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
            print(f"latency benchmark progress: {idx}/{len(specs)} episodes", flush=True)
    elapsed = time.perf_counter() - start
    df = pd.DataFrame(rows)
    latencies = np.asarray(traces, dtype=float)
    summary = _summary(df, latencies, elapsed, args.deadline, args.p95_limit, args.p99_limit)
    episodes_path = args.out_dir / "latency_benchmark_episodes.csv"
    summary_path = args.out_dir / "latency_benchmark_summary.json"
    report_path = args.out_dir / "latency_benchmark.md"
    df.to_csv(episodes_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    report_path.write_text(_format_markdown(summary))
    print(f"wrote {episodes_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")
    if not summary["passes"]:
        raise SystemExit("Latency benchmark gate failed.")


def _run_one(cfg, spec: tuple[int, str, str, float, str], planner: str, record_trace: bool) -> dict[str, object]:
    seed, profile, wind, density, gust = spec
    env, edges, retry_count, retry_reason = _build_connected_env_edges(cfg, int(seed), wind, density, float(gust), profile)
    episode_rng_seed = stable_seed(seed, env.seed, wind, density, gust, profile, "episode")
    algorithm_seed = stable_seed(seed, env.seed, wind, density, gust, profile, planner, "algorithm")
    row = run_episode(
        env,
        edges,
        planner,
        cfg,
        actual_gust_multiplier=gust,
        episode_rng_seed=episode_rng_seed,
        algorithm_seed=algorithm_seed,
        include_latency_trace=record_trace,
    )
    row["scenario_seed"] = int(seed)
    row["scenario_profile"] = profile
    row["edge_cache_retry_count"] = int(retry_count)
    row["edge_cache_retry_reason"] = retry_reason
    return row


def _summary(df: pd.DataFrame, latencies: np.ndarray, runtime_seconds: float, deadline: float, p95_limit: float, p99_limit: float) -> dict[str, object]:
    hard_safety = {}
    for col in ["battery_violation", "clearance_violation", "reserve_shortfall", "return_failure", "emergency_abort"]:
        if col in df.columns:
            hard_safety[col] = float(df[col].astype(float).mean()) if len(df) else 1.0
    if len(latencies):
        latency_summary = {
            "p50": float(np.quantile(latencies, 0.50)),
            "p95": float(np.quantile(latencies, 0.95)),
            "p99": float(np.quantile(latencies, 0.99)),
            "max": float(np.max(latencies)),
            "deadline_miss_rate": float(np.mean(latencies > deadline)),
            "n_replans": int(len(latencies)),
        }
    else:
        latency_summary = {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "deadline_miss_rate": 1.0, "n_replans": 0}
    passes = bool(
        len(df) > 0
        and latency_summary["p95"] <= p95_limit
        and latency_summary["p99"] <= p99_limit
        and all(value == 0.0 for value in hard_safety.values())
    )
    return {
        "episodes": int(len(df)),
        "runtime_seconds": float(runtime_seconds),
        "deadline": float(deadline),
        "p95_limit": float(p95_limit),
        "p99_limit": float(p99_limit),
        "latency": latency_summary,
        "hard_safety_rates": hard_safety,
        "passes": passes,
    }


def _format_markdown(summary: dict[str, object]) -> str:
    lat = summary["latency"]
    lines = [
        "# Strict Latency Benchmark",
        "",
        f"- episodes: `{summary['episodes']}`",
        f"- replan samples: `{lat['n_replans']}`",
        f"- p50: `{lat['p50']:.6f}` s",
        f"- p95: `{lat['p95']:.6f}` s",
        f"- p99: `{lat['p99']:.6f}` s",
        f"- max: `{lat['max']:.6f}` s",
        f"- deadline miss rate: `{lat['deadline_miss_rate']:.6f}`",
        f"- gate: `{'pass' if summary['passes'] else 'fail'}`",
        "",
        "| safety metric | rate |",
        "|:--|--:|",
    ]
    for key, value in summary["hard_safety_rates"].items():
        lines.append(f"| {key} | {value:.6f} |")
    return "\n".join(lines) + "\n"


def _pin_core(core: int) -> None:
    try:
        os.sched_setaffinity(0, {int(core)})
    except (AttributeError, OSError):
        return


def _specs(seeds: list[int], profiles: list[str], wind_levels: list[str], densities: list[str], gust: float) -> list[tuple[int, str, str, float, str]]:
    return [(seed, profile, wind, density, gust) for seed in seeds for profile in profiles for wind in wind_levels for density in densities]


def _parse_trace(text: str) -> list[float]:
    return [float(item) for item in text.split(";") if item]


def _parse_seeds(text: str) -> list[int]:
    if "-" in text:
        lo, hi = [int(part) for part in text.split("-", 1)]
        return list(range(lo, hi + 1))
    return [int(item) for item in _split(text)]


def _split(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


if __name__ == "__main__":
    main()

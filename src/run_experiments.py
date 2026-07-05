from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

from .config import (
    OBSTACLE_DENSITIES,
    WIND_LEVELS,
    ensure_output_dirs,
    load_config,
    profile_planners,
    profile_seeds,
)
from .environment import InspectionEnvironment
from .metrics import summarize_episode_results, write_stress_tables, write_tables
from .simulation import build_edge_cache, run_episode


STRESS_GUST_MULTIPLIER = 2.0


def run_profile(profile: str, output_root: Path) -> pd.DataFrame:
    cfg = load_config()
    outputs = ensure_output_dirs(output_root)
    planners = profile_planners(profile)
    seeds = profile_seeds(profile)
    rows: List[dict] = []
    wind_levels = ["moderate", "severe"] if profile in {"stress", "stress_holdout"} else WIND_LEVELS
    densities = OBSTACLE_DENSITIES
    actual_gust_multiplier = STRESS_GUST_MULTIPLIER if profile in {"stress", "stress_holdout"} else 1.0
    env_specs = [
        (seed, wind_level, density)
        for seed in seeds
        for wind_level in wind_levels
        for density in densities
    ]
    pbar = tqdm(env_specs, desc=f"{profile} environments")
    for seed, wind_level, density in pbar:
        env = InspectionEnvironment(cfg, seed=seed, wind_level=wind_level, obstacle_density=density)
        cache_start = time.perf_counter()
        edges = build_edge_cache(env, cfg)
        cache_seconds = time.perf_counter() - cache_start
        for planner_name in planners:
            start = time.perf_counter()
            row = run_episode(env, edges, planner_name, cfg, actual_gust_multiplier=actual_gust_multiplier)
            row["runtime_seconds"] = time.perf_counter() - start
            row["edge_cache_seconds"] = cache_seconds
            row["profile"] = profile
            rows.append(row)
    df = pd.DataFrame(rows)
    prefix = _output_prefix(profile)
    results_path = outputs["results"] / f"{prefix}episode_results.csv"
    summary_path = outputs["results"] / f"{prefix}summary_by_method.csv"
    df.to_csv(results_path, index=False)
    summary = summarize_episode_results(df)
    summary.to_csv(summary_path, index=False)
    if profile == "stress":
        write_stress_tables(summary, outputs["tables"])
    elif profile == "stress_holdout":
        write_stress_tables(summary, outputs["tables"] / "holdout")
    elif profile == "holdout":
        write_tables(summary, outputs["tables"] / "holdout")
    else:
        write_tables(summary, outputs["tables"])
    return df



def _output_prefix(profile: str) -> str:
    if profile == "stress":
        return "stress_"
    if profile == "holdout":
        return "holdout_"
    if profile == "stress_holdout":
        return "stress_holdout_"
    return ""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAWA-RHP UAV inspection experiments.")
    parser.add_argument("--profile", choices=["quick", "full", "stress", "holdout", "stress_holdout"], default="quick")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = run_profile(args.profile, args.output_root)
    prefix = _output_prefix(args.profile)
    out_path = args.output_root / "results" / (prefix + "episode_results.csv")
    print(f"Wrote {len(df)} episode rows to {out_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_optimization_suite import _build_connected_env_edges, _parse_seeds
from src.config import OBSTACLE_DENSITIES, WIND_LEVELS, SimulationConfig, load_config, stable_seed
from src.metrics import summarize_episode_results
from src.simulation import run_episode


PROFILES = ["id", "ood:correlated_gust", "ood:narrow_passage"]
SAFETY_COLS = ["battery_violation", "clearance_violation", "reserve_shortfall", "return_failure", "emergency_abort"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-factor RAWA-RHP parameter sensitivity on a stratified scenario subset.")
    parser.add_argument("--seeds", default="301-307")
    parser.add_argument("--profiles", default=",".join(PROFILES))
    parser.add_argument("--wind-levels", default=",".join(WIND_LEVELS))
    parser.add_argument("--densities", default=",".join(OBSTACLE_DENSITIES))
    parser.add_argument("--planner", default="RAWA-RHP")
    parser.add_argument("--version", default="parameter_sensitivity_s301_307_v1")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "results")
    parser.add_argument("--max-scenarios", type=int, default=0, help="Debug limit; 0 runs the full selected matrix.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel scenario workers.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output_root / args.version
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = _parse_seeds(args.seeds, default="301-307")
    profiles = _split(args.profiles)
    wind_levels = _split(args.wind_levels)
    densities = _split(args.densities)
    specs = [(seed, profile, wind, density) for seed in seeds for profile in profiles for wind in wind_levels for density in densities]
    if args.max_scenarios > 0:
        specs = specs[: args.max_scenarios]
    variants = _variants()
    rows: list[dict[str, object]] = []
    t0 = time.perf_counter()
    base_cfg = load_config()
    checkpoint = out_dir / "parameter_sensitivity_episodes.checkpoint.csv"
    if args.workers <= 1:
        for spec in tqdm(specs, desc="sensitivity scenarios"):
            rows.extend(_run_spec(spec, variants, base_cfg, args.planner))
            pd.DataFrame(rows).to_csv(checkpoint, index=False)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_run_spec, spec, variants, base_cfg, args.planner) for spec in specs]
            for future in tqdm(as_completed(futures), total=len(futures), desc="sensitivity scenarios"):
                rows.extend(future.result())
                pd.DataFrame(rows).to_csv(checkpoint, index=False)
    df = pd.DataFrame(rows)
    episodes = out_dir / "parameter_sensitivity_episodes.csv"
    summary = out_dir / "parameter_sensitivity_summary.csv"
    deltas = out_dir / "parameter_sensitivity_deltas.csv"
    matrix = out_dir / "sensitivity_matrix.csv"
    df.to_csv(episodes, index=False)
    _summary(df).to_csv(summary, index=False)
    _deltas(df).to_csv(deltas, index=False)
    _variant_matrix(variants).to_csv(matrix, index=False)
    (out_dir / "PARAMETER_SENSITIVITY.md").write_text(_markdown_report(df, variants, time.perf_counter() - t0), encoding="utf-8")
    print(f"wrote {episodes}")
    print(f"wrote {summary}")
    print(f"wrote {deltas}")


def _run_spec(
    spec: tuple[int, str, str, str],
    variants: list[dict[str, object]],
    base_cfg: SimulationConfig,
    planner: str,
) -> list[dict[str, object]]:
    seed, profile, wind, density = spec
    rows: list[dict[str, object]] = []
    cache: dict[str, tuple[object, object, int, str]] = {}
    for variant in variants:
        cfg = _cfg_for_variant(base_cfg, variant)
        cfg_key = json.dumps(dataclasses.asdict(cfg), sort_keys=True)
        if cfg_key not in cache:
            cache[cfg_key] = _build_connected_env_edges(cfg, int(seed), str(wind), str(density), 1.0, str(profile))
        env, edges, retry_count, retry_reason = cache[cfg_key]
        episode_rng_seed = stable_seed(seed, env.seed, wind, density, 1.0, profile, "episode")
        algorithm_seed = stable_seed(seed, env.seed, wind, density, 1.0, profile, planner, str(variant["variant"]), "algorithm")
        start = time.perf_counter()
        row = run_episode(
            env,
            edges,
            planner,
            cfg,
            actual_gust_multiplier=1.0,
            episode_rng_seed=episode_rng_seed,
            algorithm_seed=algorithm_seed,
            planner_overrides=_planner_overrides(variant),
        )
        row["runtime_seconds"] = time.perf_counter() - start
        row["scenario_seed"] = int(seed)
        row["scenario_id"] = f"{seed}|{profile}|{wind}|{density}|1.0"
        row["sensitivity_group"] = str(variant["group"])
        row["sensitivity_level"] = str(variant["level"])
        row["sensitivity_variant"] = str(variant["variant"])
        row["is_final_setting"] = bool(variant.get("is_final", False))
        row["edge_cache_retry_count"] = int(retry_count)
        row["edge_cache_retry_reason"] = str(retry_reason)
        rows.append(row)
    return rows


def _variants() -> list[dict[str, object]]:
    variants: list[dict[str, object]] = [{"group": "final", "level": "final", "variant": "final", "is_final": True}]
    for q in ["q80", "q95"]:
        variants.append({"group": "energy_quantile", "level": q, "variant": f"energy_quantile={q}", "planner": {"tail_quantile": q}})
    for value in [60.0, 100.0]:
        variants.append({"group": "reserve_floor", "level": value, "variant": f"reserve_floor={value:g}", "config": {"reserve_floor": value}})
    for value in [8.0, 24.0]:
        variants.append({"group": "reserve_buffer", "level": value, "variant": f"reserve_buffer={value:g}", "planner": {"reserve_buffer": value}})
    for value in [0.03, 0.08]:
        variants.append({"group": "clearance_threshold", "level": value, "variant": f"clearance_threshold={value:g}", "config": {"clearance_event_threshold": value}})
    for scale in [0.75, 1.25]:
        variants.append({"group": "risk_dispersion_scale", "level": scale, "variant": f"risk_dispersion_scale={scale:g}", "risk_scale": scale})
    for scale in [0.5, 1.5]:
        variants.append({"group": "beam_search_budget", "level": scale, "variant": f"beam_search_budget={scale:g}x", "search_scale": scale})
    for depth in [8, 10, 12]:
        variants.append({"group": "horizon_length", "level": depth, "variant": f"horizon_length={depth}", "planner": {"beam_depth": depth}})
    for scale in [0.75, 1.25]:
        variants.append({"group": "rr_cmp_weights", "level": scale, "variant": f"rr_cmp_weights={scale:g}x", "rr_scale": scale})
    for samples in [16, 64]:
        variants.append({"group": "gust_samples", "level": samples, "variant": f"gust_samples={samples}", "config": {"gust_samples": samples}})
    for cap in [1.0, 2.5]:
        variants.append({"group": "replan_cap", "level": cap, "variant": f"replan_cap={cap:g}s", "planner": {"max_replan_seconds": cap}})
    return variants


def _cfg_for_variant(base: SimulationConfig, variant: dict[str, object]) -> SimulationConfig:
    updates = dict(variant.get("config", {}) or {})
    if "risk_scale" in variant:
        scale = float(variant["risk_scale"])
        updates.update({"sigma0": base.sigma0 * scale, "k_cross": base.k_cross * scale, "k_gust": base.k_gust * scale})
    if updates:
        return dataclasses.replace(base, **updates)
    return base


def _planner_overrides(variant: dict[str, object]) -> dict[str, object]:
    overrides = dict(variant.get("planner", {}) or {})
    if "search_scale" in variant:
        scale = float(variant["search_scale"])
        overrides.update(
            {
                "beam_width": max(1, int(round(96 * scale))),
                "candidate_pool_size": max(4, int(round(16 * scale))),
                "repair_top_k": max(1, int(round(6 * scale))),
                "repair_max_insertions": max(1, int(round(10 * scale))),
                "packing_candidate_limit": max(1, int(round(18 * scale))),
                "max_replan_expansions": max(1, int(round(9000 * scale))),
            }
        )
    if "rr_scale" in variant:
        scale = float(variant["rr_scale"])
        overrides.update(
            {
                "packing_regret_weight": 0.38 * scale,
                "packing_unlock_weight": 0.24 * scale,
                "insertion_risk_weight": 18.0 * scale,
            }
        )
    return overrides


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (group, level, variant), sub in df.groupby(["sensitivity_group", "sensitivity_level", "sensitivity_variant"], sort=False):
        row = {
            "group": group,
            "level": level,
            "variant": variant,
            "n": int(len(sub)),
            "safe_weighted_coverage_mean": float(sub["safe_weighted_coverage"].mean()),
            "safe_weighted_coverage_sd": float(sub["safe_weighted_coverage"].std(ddof=1)) if len(sub) > 1 else 0.0,
            "latency_p50": float(np.quantile(sub["replan_latency_p95"].astype(float), 0.50)),
            "latency_p95": float(np.quantile(sub["replan_latency_p95"].astype(float), 0.95)),
            "latency_p99": float(np.quantile(sub["replan_latency_p95"].astype(float), 0.99)),
        }
        for col in SAFETY_COLS:
            if col in sub.columns:
                row[f"{col}_rate"] = float(sub[col].astype(bool).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _deltas(df: pd.DataFrame) -> pd.DataFrame:
    ref = df[df["is_final_setting"]].set_index("scenario_id")["safe_weighted_coverage"].astype(float)
    rows = []
    for (group, level, variant), sub in df[~df["is_final_setting"]].groupby(["sensitivity_group", "sensitivity_level", "sensitivity_variant"], sort=False):
        joined = sub.set_index("scenario_id").join(ref.rename("final_swc"), how="inner")
        delta = joined["safe_weighted_coverage"].astype(float) - joined["final_swc"].astype(float)
        rows.append(
            {
                "group": group,
                "level": level,
                "variant": variant,
                "n_pairs": int(len(delta)),
                "delta_swc_mean": float(delta.mean()),
                "delta_swc_median": float(delta.median()),
                "delta_swc_low": float(np.quantile(delta, 0.025)),
                "delta_swc_high": float(np.quantile(delta, 0.975)),
                "abs_delta_swc_mean": float(np.abs(delta).mean()),
            }
        )
    return pd.DataFrame(rows)


def _variant_matrix(variants: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for item in variants:
        rows.append(
            {
                "group": item["group"],
                "level": item["level"],
                "variant": item["variant"],
                "config_overrides": json.dumps(item.get("config", {}), sort_keys=True),
                "planner_overrides": json.dumps(_planner_overrides(item), sort_keys=True),
                "risk_scale": item.get("risk_scale", ""),
                "is_final_setting": bool(item.get("is_final", False)),
            }
        )
    return pd.DataFrame(rows)


def _markdown_report(df: pd.DataFrame, variants: list[dict[str, object]], runtime_seconds: float) -> str:
    summary = _summary(df)
    deltas = _deltas(df)
    worst = deltas.reindex(deltas["abs_delta_swc_mean"].sort_values(ascending=False).index).head(8)
    lines = [
        "# RAWA-RHP Parameter Sensitivity",
        "",
        f"- scenarios per variant: `{int(df['scenario_id'].nunique())}`",
        f"- variants including final reference: `{len(variants)}`",
        f"- episodes: `{len(df)}`",
        f"- runtime seconds: `{runtime_seconds:.1f}`",
        "",
        "The experiment is one-factor-at-a-time. Delta SWC is computed as variant minus the final reported configuration on the same paired scenario.",
        "",
        "## Largest absolute mean changes",
        "",
        "| group | level | mean delta SWC | mean abs delta SWC | n |",
        "|:--|:--|--:|--:|--:|",
    ]
    for _, row in worst.iterrows():
        lines.append(f"| {row['group']} | {row['level']} | {row['delta_swc_mean']:.6f} | {row['abs_delta_swc_mean']:.6f} | {int(row['n_pairs'])} |")
    lines.extend(["", "## Variant means", "", "| group | level | SWC mean | p95 latency | battery fail | clearance fail | reserve fail | return fail | emergency |", "|:--|:--|--:|--:|--:|--:|--:|--:|--:|"])
    for _, row in summary.iterrows():
        lines.append(
            "| {group} | {level} | {swc:.6f} | {lat:.6f} | {bat:.6f} | {clr:.6f} | {res:.6f} | {ret:.6f} | {emg:.6f} |".format(
                group=row["group"],
                level=row["level"],
                swc=float(row["safe_weighted_coverage_mean"]),
                lat=float(row["latency_p95"]),
                bat=float(row.get("battery_violation_rate", 0.0)),
                clr=float(row.get("clearance_violation_rate", 0.0)),
                res=float(row.get("reserve_shortfall_rate", 0.0)),
                ret=float(row.get("return_failure_rate", 0.0)),
                emg=float(row.get("emergency_abort_rate", 0.0)),
            )
        )
    return "\n".join(lines) + "\n"


def _split(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


EQTIME_PLANNERS = [
    "RAWA-NoPacking-EqTime",
    "RAWA-NoBeam-EqTime",
    "RAWA-NoRisk-EqTime",
    "RAWA-BlindReserve-EqTime",
    "RAWA-NoAdaptiveSearch-EqTime",
]

EQEVAL_PLANNERS = [
    "RAWA-NoPacking-EqEval",
    "RAWA-NoBeam-EqEval",
    "RAWA-NoRisk-EqEval",
    "RAWA-BlindReserve-EqEval",
    "RAWA-NoAdaptiveSearch-EqEval",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build EqTime/EqEval budget calibration from RAWA-RHP episode telemetry.")
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--source-planner", default="RAWA-RHP")
    parser.add_argument("--group-by", default="scenario_profile,ood_category,wind_level,obstacle_density,actual_gust_multiplier")
    parser.add_argument("--time-field", default="replan_latency_p95")
    parser.add_argument("--eval-fields", default="candidate_expansions,risk_evals,packing_attempts")
    parser.add_argument("--target-quantile", type=float, default=0.50)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--time-multipliers", default="", help="Comma-separated planner=multiplier overrides for EqTime planners.")
    parser.add_argument("--eval-multipliers", default="", help="Comma-separated planner=multiplier overrides for EqEval planners.")
    parser.add_argument("--unified-multipliers", default="", help="Comma-separated planner=multiplier overrides for EqEval unified eval targets.")
    parser.add_argument(
        "--eval-budget-scope",
        choices=["per_replan", "episode"],
        default="per_replan",
        help="Planner max_replan_expansions is per-replan; use per_replan unless reproducing legacy calibration.",
    )
    parser.add_argument("--extra-gust-multipliers", default="1.0,3.5", help="Also emit group coverage for these gust multipliers using matching observed groups.")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.episodes)
    source = df[df["planner"] == args.source_planner].copy()
    if source.empty:
        raise SystemExit(f"No rows found for source planner {args.source_planner}")
    group_by = _split(args.group_by)
    eval_fields = [field for field in _split(args.eval_fields) if field in source.columns]
    if args.time_field not in source.columns:
        raise SystemExit(f"Missing time field: {args.time_field}")
    if "candidate_expansions" not in eval_fields and "candidate_expansions" in source.columns:
        eval_fields.insert(0, "candidate_expansions")
    source = source.copy()
    if args.eval_budget_scope == "per_replan":
        if "replan_count" not in source.columns:
            raise SystemExit("per_replan eval budget requires replan_count in episode CSV.")
        source["candidate_expansions_per_replan"] = source["candidate_expansions"].astype(float) / source["replan_count"].clip(lower=1).astype(float)
        if "unified_eval_count" in source.columns:
            source["unified_eval_count_per_replan"] = source["unified_eval_count"].astype(float) / source["replan_count"].clip(lower=1).astype(float)
        expansion_target_field = "candidate_expansions_per_replan"
    else:
        expansion_target_field = "candidate_expansions"
    time_multipliers = _multiplier_map(args.time_multipliers)
    eval_multipliers = _multiplier_map(args.eval_multipliers)
    unified_multipliers = _multiplier_map(args.unified_multipliers)
    global_time = _quantile(source[args.time_field], args.target_quantile)
    global_expansions = int(max(1, round(_quantile(source[expansion_target_field], args.target_quantile)))) if expansion_target_field in source.columns else 1
    unified_target_field = "unified_eval_count_per_replan" if "unified_eval_count_per_replan" in source.columns else "unified_eval_count"
    global_unified = int(max(1, round(_quantile(source[unified_target_field], args.target_quantile)))) if unified_target_field in source.columns else int(2 * global_expansions)
    groups = _group_rows(source, group_by, args.time_field, eval_fields, args.target_quantile)
    groups = _extend_gust_coverage(groups, group_by, _floats(args.extra_gust_multipliers))
    planners: dict[str, dict[str, object]] = {}
    for planner in EQTIME_PLANNERS:
        multiplier = time_multipliers.get(planner, 1.0)
        target = float(global_time * multiplier)
        planners[planner] = {
            "min_replan_seconds": target,
            "max_replan_seconds": target,
        }
    for planner in EQEVAL_PLANNERS:
        multiplier = eval_multipliers.get(planner, 1.0)
        unified_multiplier = unified_multipliers.get(planner, 1.0)
        planners[planner] = {
            "max_replan_expansions": int(max(1, round(global_expansions * multiplier))),
            "fixed_replan_budget": True,
            "min_unified_eval_count": int(max(1, round(global_unified * unified_multiplier))),
        }
        planners[planner].update(_eqeval_search_overrides(planner))
    payload = {
        "source_episodes": str(args.episodes),
        "source_planner": args.source_planner,
        "group_by": group_by,
        "time_field": args.time_field,
        "eval_fields": eval_fields,
        "target_quantile": args.target_quantile,
        "tolerance": args.tolerance,
        "time_multipliers": time_multipliers,
        "eval_multipliers": eval_multipliers,
        "unified_multipliers": unified_multipliers,
        "eval_budget_scope": args.eval_budget_scope,
        "expansion_target_field": expansion_target_field,
        "global_targets": {
            "time": float(global_time),
            "candidate_expansions": int(global_expansions),
            "unified_eval_count": int(global_unified),
        },
        "planners": planners,
        "groups": groups,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"wrote {args.out}")
    print(f"groups: {len(groups)}")
    print(f"global time target ({args.time_field} q={args.target_quantile}): {global_time:.6f}")
    print(f"global {expansion_target_field} target: {global_expansions}")


def _group_rows(
    source: pd.DataFrame,
    group_by: list[str],
    time_field: str,
    eval_fields: list[str],
    quantile: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for keys, group in source.groupby(group_by, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key = {col: _json_value(value) for col, value in zip(group_by, keys)}
        targets = {"time": _quantile(group[time_field], quantile)}
        for field in eval_fields:
            targets[field] = _quantile(group[field], quantile)
        if "candidate_expansions" in targets:
            targets["max_replan_expansions"] = int(max(1, round(float(targets["candidate_expansions"]))))
        rows.append({"key": key, "n": int(len(group)), "targets": {k: _json_value(v) for k, v in targets.items()}})
    return rows


def _extend_gust_coverage(groups: list[dict[str, object]], group_by: list[str], gusts: list[float]) -> list[dict[str, object]]:
    if "actual_gust_multiplier" not in group_by or not gusts:
        return groups
    existing = {tuple(row["key"].get(col) for col in group_by) for row in groups}
    out = list(groups)
    for row in groups:
        base_key = dict(row["key"])
        for gust in gusts:
            trial_key = dict(base_key)
            trial_key["actual_gust_multiplier"] = float(gust)
            sig = tuple(trial_key.get(col) for col in group_by)
            if sig in existing:
                continue
            cloned = {"key": trial_key, "n": int(row["n"]), "targets": dict(row["targets"]), "synthetic_from_gust": base_key.get("actual_gust_multiplier")}
            out.append(cloned)
            existing.add(sig)
    return out


def _quantile(values: pd.Series, q: float) -> float:
    arr = values.astype(float).to_numpy()
    if len(arr) == 0:
        return 0.0
    return float(np.quantile(arr, q))


def _split(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _floats(text: str) -> list[float]:
    return [float(item) for item in _split(text)]


def _multiplier_map(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in _split(text):
        if "=" not in item:
            raise SystemExit(f"Multiplier must use planner=value syntax: {item}")
        key, value = item.split("=", 1)
        out[key.strip()] = float(value)
    return out


def _eqeval_search_overrides(planner: str) -> dict[str, int]:
    # EqEval must spend real search/evaluation work, not filler loops. These
    # overrides raise true beam/candidate/repair/packing limits while leaving
    # the ablated module semantics intact.
    if planner == "RAWA-NoPacking-EqEval":
        return {
            "beam_width": 760,
            "beam_depth": 16,
            "candidate_pool_size": 128,
            "repair_top_k": 80,
            "repair_max_insertions": 72,
            "packing_candidate_limit": 48,
        }
    if planner == "RAWA-NoBeam-EqEval":
        return {
            "beam_depth": 80,
            "candidate_pool_size": 128,
            "repair_top_k": 80,
            "repair_max_insertions": 72,
            "packing_candidate_limit": 128,
            "eqeval_probe_guard_multiplier": 512,
        }
    if planner == "RAWA-NoAdaptiveSearch-EqEval":
        return {
            "beam_width": 96,
            "beam_depth": 7,
            "candidate_pool_size": 16,
            "repair_top_k": 6,
            "repair_max_insertions": 10,
            "packing_candidate_limit": 18,
            "eqeval_probe_guard_multiplier": 512,
        }
    if planner in {"RAWA-NoRisk-EqEval", "RAWA-BlindReserve-EqEval"}:
        return {
            "beam_width": 760,
            "beam_depth": 16,
            "candidate_pool_size": 128,
            "repair_top_k": 80,
            "repair_max_insertions": 72,
            "packing_candidate_limit": 88,
        }
    return {
        "packing_candidate_limit": 48,
    }


def _json_value(value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return float(value)
    if isinstance(value, int):
        return int(value)
    return value


if __name__ == "__main__":
    main()

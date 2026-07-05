from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import random
import sys
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import OBSTACLE_DENSITIES, WIND_LEVELS, stable_seed, load_config
from src.environment import InspectionEnvironment
from src.metrics import summarize_episode_results
from src.simulation import build_edge_cache, run_episode
from src.wind import OOD_CATEGORIES
from scripts.analyze_formal_ablation import DEFAULT_ABLATIONS, analyze_episodes, _format_markdown


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
    "RAWA-NoRepair",
    "RAWA-OpenLoop",
]
FORMAL_FULL_PLANNERS = [
    "RAWA-RHP",
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
SMOKE_OOD_CATEGORIES = ["correlated_gust", "narrow_passage"]
STRESS_GUSTS = [2.0, 2.5, 3.0, 3.5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated RAWA-RHP optimization acceptance suites.")
    parser.add_argument("--version", default=None, help="Output version suffix. Defaults to next optimization_vN.")
    parser.add_argument("--mode", choices=["smoke", "blind", "stress", "ood", "ablation", "full"], default="smoke")
    parser.add_argument("--seeds", default=None, help="Comma list or A-B range. Defaults: smoke 160-161, blind/stress/ood 160-359.")
    parser.add_argument("--planners", default=None, help="Comma-separated planner override.")
    parser.add_argument("--gusts", default=None, help="Comma-separated stress gust multipliers.")
    parser.add_argument("--ood-categories", default=None, help="Comma-separated OOD category override.")
    parser.add_argument("--budget-mode", choices=["wall_clock", "evaluations", "anytime", "calibrated"], default="anytime")
    parser.add_argument("--optimizer-seeds", default="0-9", help="Optimizer seed list/range metadata for stochastic baselines.")
    parser.add_argument("--budget-calibration", type=Path, default=None, help="JSON or CSV mapping planner names to calibrated fixed budget overrides.")
    parser.add_argument("--randomize-planners", action="store_true", help="Use a stable random method order inside each shared scenario.")
    parser.add_argument("--skip-formal-analysis", action="store_true", help="Write raw full-mode outputs without running formal ablation gate.")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "results")
    parser.add_argument("--workers", type=int, default=1, help="Parallel environment workers. Default keeps deterministic serial execution.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "full" and args.version is None:
        args.version = "formal_full_ablation_s1_20_v1"
    out_dir = _version_dir(args.output_root, args.version)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = _parse_seeds(args.seeds, default="1-20" if args.mode == "full" else ("160-161" if args.mode == "smoke" else "161-360"))
    planner_override = args.planners is not None
    if planner_override:
        planners = _parse_list(args.planners)
    elif args.mode == "full":
        planners = FORMAL_FULL_PLANNERS
    else:
        planners = ABLATION_PLANNERS if args.mode == "ablation" else MAIN_PLANNERS
    gusts = [float(x) for x in _parse_list(args.gusts)] if args.gusts else ([1.0] if args.mode in {"smoke", "blind", "ood", "ablation", "full"} else STRESS_GUSTS)
    optimizer_seeds = _parse_seeds(args.optimizer_seeds, default="0-9")
    budget_overrides = _load_budget_overrides(args.budget_calibration)
    ood_categories = _parse_list(args.ood_categories) if args.ood_categories else OOD_CATEGORIES
    profiles = [f"ood:{cat}" for cat in ood_categories] if args.mode == "ood" else ["id"]
    wind_levels = ["moderate", "severe"] if args.mode == "stress" else WIND_LEVELS
    if args.mode == "full":
        profiles = ["id", "ood:correlated_gust", "ood:narrow_passage"]
        wind_levels = WIND_LEVELS
        gusts = [1.0]
    if args.mode == "smoke":
        gusts = [1.0, 3.5]
        wind_levels = ["severe"]
        if not planner_override:
            planners = planners[:2] + ["RAWA-RHP"] if "RAWA-RHP" not in planners[:3] else planners[:3]
        smoke_ood_categories = _parse_list(args.ood_categories) if args.ood_categories else SMOKE_OOD_CATEGORIES
        profiles = ["id"] + [f"ood:{cat}" for cat in smoke_ood_categories]
    if args.mode == "full":
        _validate_full_budget_calibration(planners, budget_overrides)
    stem = f"{args.mode}_{seeds[0]}_{seeds[-1]}"
    episode_path = out_dir / f"{stem}_episodes.csv"
    checkpoint_path = out_dir / f"{stem}_episodes.checkpoint.csv"
    df = run_suite(
        seeds,
        planners,
        gusts,
        wind_levels,
        OBSTACLE_DENSITIES,
        profiles,
        args.budget_mode,
        optimizer_seeds,
        budget_overrides=budget_overrides,
        workers=args.workers,
        randomize_planners=args.randomize_planners,
        checkpoint_path=checkpoint_path,
    )
    summary_path = out_dir / f"{stem}_summary.csv"
    df.to_csv(episode_path, index=False)
    summarize_episode_results(df).to_csv(summary_path, index=False)
    if args.mode == "full":
        budget_profile_path = out_dir / "budget_profile.csv"
        _write_budget_profile(df, budget_profile_path, args.budget_calibration, budget_overrides)
        manifest = out_dir / "manifest.md"
    else:
        budget_profile_path = None
        manifest = out_dir / f"{stem}_manifest.md"
    manifest.write_text(_manifest_text(args, seeds, planners, gusts, wind_levels, OBSTACLE_DENSITIES, profiles, optimizer_seeds, episode_path, summary_path, budget_profile_path))
    if args.mode == "full" and "RAWA-RHP" in planners and not args.skip_formal_analysis:
        formal_ablations = [planner for planner in DEFAULT_ABLATIONS if planner in planners]
        if formal_ablations:
            report = analyze_episodes(episode_path, ablations=formal_ablations)
            (out_dir / "formal_ablation_analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
            (out_dir / "formal_ablation_analysis.md").write_text(_format_markdown(report))
            if not report["formal_gate_pass"]:
                raise SystemExit("Formal full-mode analysis gate failed; reports were written.")
    print(f"wrote {episode_path}")
    print(f"wrote {summary_path}")
    if budget_profile_path is not None:
        print(f"wrote {budget_profile_path}")


def run_suite(
    seeds: list[int],
    planners: list[str],
    gusts: Iterable[float],
    wind_levels: Iterable[str],
    densities: Iterable[str],
    scenario_profiles: Iterable[str],
    budget_mode: str,
    optimizer_seeds: list[int],
    budget_overrides: dict[str, dict[str, object]] | None = None,
    workers: int = 1,
    randomize_planners: bool = False,
    checkpoint_path: Path | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specs = [
        (seed, wind, density, gust, profile)
        for seed in seeds
        for profile in scenario_profiles
        for gust in gusts
        for wind in wind_levels
        for density in densities
    ]
    if workers <= 1:
        for spec in tqdm(specs, desc="optimization envs"):
            rows.extend(_run_spec(spec, planners, budget_mode, optimizer_seeds, randomize_planners, budget_overrides or {}))
            _write_checkpoint(rows, checkpoint_path)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_run_spec, spec, planners, budget_mode, optimizer_seeds, randomize_planners, budget_overrides or {}) for spec in specs]
            for future in tqdm(as_completed(futures), total=len(futures), desc="optimization envs"):
                rows.extend(future.result())
                _write_checkpoint(rows, checkpoint_path)
    return pd.DataFrame(rows)


def _write_checkpoint(rows: list[dict[str, object]], checkpoint_path: Path | None) -> None:
    if checkpoint_path is None or not rows:
        return
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(tmp_path, index=False)
    tmp_path.replace(checkpoint_path)


def _run_spec(
    spec: tuple[int, str, str, float, str],
    planners: list[str],
    budget_mode: str,
    optimizer_seeds: list[int],
    randomize_planners: bool,
    budget_overrides: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    cfg = load_config()
    seed, wind, density, gust, profile = spec
    cache_start = time.perf_counter()
    env, edges, edge_cache_retry_count, edge_cache_retry_reason = _build_connected_env_edges(
        cfg,
        int(seed),
        str(wind),
        str(density),
        float(gust),
        str(profile),
    )
    env_seed = int(env.seed)
    cache_seconds = time.perf_counter() - cache_start
    rows: list[dict[str, object]] = []
    planner_order = list(planners)
    if randomize_planners:
        rng = random.Random(stable_seed(seed, wind, density, gust, profile, "planner_order"))
        rng.shuffle(planner_order)
    for run_order, planner_name in enumerate(planner_order):
        episode_rng_seed = stable_seed(seed, env_seed, wind, density, gust, profile, "episode")
        algorithm_seed = stable_seed(seed, env_seed, wind, density, gust, profile, planner_name, "algorithm")
        start = time.perf_counter()
        row = run_episode(
            env,
            edges,
            planner_name,
            env.cfg,
            actual_gust_multiplier=float(gust),
            episode_rng_seed=episode_rng_seed,
            algorithm_seed=algorithm_seed,
            planner_overrides=budget_overrides.get(planner_name),
        )
        row["runtime_seconds"] = time.perf_counter() - start
        row["edge_cache_seconds"] = cache_seconds
        row["seed"] = int(seed)
        row["scenario_seed"] = int(seed)
        row["env_seed"] = int(env_seed)
        row["episode_rng_seed"] = int(episode_rng_seed)
        row["algorithm_seed"] = int(algorithm_seed)
        row["scenario_profile"] = profile
        row["budget_mode"] = budget_mode
        row["budget_calibrated"] = bool(planner_name in budget_overrides)
        row["planner_run_order"] = int(run_order)
        row["planner_order_randomized"] = bool(randomize_planners)
        row["optimizer_seed_count"] = len(optimizer_seeds) if planner_name in {"FairALNSPlanner"} else 1
        row["edge_cache_retry_count"] = int(edge_cache_retry_count)
        row["edge_cache_retry_reason"] = edge_cache_retry_reason
        rows.append(row)
    return rows


def _build_connected_env_edges(
    cfg,
    seed: int,
    wind: str,
    density: str,
    gust: float,
    profile: str,
    max_attempts: int = 20,
) -> tuple[InspectionEnvironment, object, int, str]:
    last_error = ""
    for attempt in range(max_attempts):
        if attempt == 0:
            env_seed = int(seed) if profile == "id" else stable_seed(seed, profile)
        else:
            env_seed = stable_seed(seed, profile, wind, density, gust, "connected_retry", attempt)
        env = InspectionEnvironment(cfg, seed=int(env_seed), wind_level=wind, obstacle_density=density, scenario_profile=profile)
        try:
            return env, build_edge_cache(env, env.cfg), attempt, last_error
        except RuntimeError as exc:
            last_error = str(exc)
    raise RuntimeError(
        f"Could not build a connected edge cache after {max_attempts} attempts "
        f"for seed={seed}, profile={profile}, wind={wind}, density={density}, gust={gust}: {last_error}"
    )


def _parse_list(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _parse_seeds(text: str | None, default: str) -> list[int]:
    spec = text or default
    if "-" in spec:
        lo, hi = [int(x) for x in spec.split("-", 1)]
        return list(range(lo, hi + 1))
    return [int(x) for x in _parse_list(spec)]


def _version_dir(root: Path, version: str | None) -> Path:
    if version:
        return root / version
    existing = [p for p in root.glob("optimization_v*") if p.is_dir()]
    nums = []
    for path in existing:
        suffix = path.name.replace("optimization_v", "")
        if suffix.isdigit():
            nums.append(int(suffix))
    return root / f"optimization_v{(max(nums) + 1) if nums else 1}"


def _load_budget_overrides(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None:
        return {}
    if not path.exists():
        raise SystemExit(f"Budget calibration file not found: {path}")
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if "planners" in data and isinstance(data["planners"], dict):
            data = data["planners"]
        return {str(name): _clean_budget_values(values) for name, values in dict(data).items()}
    frame = pd.read_csv(path)
    if "planner" not in frame.columns:
        raise SystemExit("Budget calibration CSV must include a planner column.")
    out: dict[str, dict[str, object]] = {}
    for _, row in frame.iterrows():
        planner = str(row["planner"])
        values = {
            key: row[key]
            for key in ["max_replan_seconds", "min_replan_seconds", "max_replan_expansions", "beam_width", "beam_depth", "candidate_pool_size", "repair_top_k", "repair_max_insertions", "packing_candidate_limit", "fixed_replan_budget"]
            if key in frame.columns and pd.notna(row[key])
        }
        out[planner] = _clean_budget_values(values)
    return out


def _validate_full_budget_calibration(planners: list[str], budget_overrides: dict[str, dict[str, object]]) -> None:
    eq_planners = [planner for planner in planners if planner.endswith(("-EqTime", "-EqEval"))]
    if not eq_planners:
        return
    if not budget_overrides:
        raise SystemExit("Full formal EqTime/EqEval variants require --budget-calibration. Run a RAWA-RHP budget profile first, then provide calibrated fixed budgets.")
    missing = [planner for planner in eq_planners if planner not in budget_overrides]
    if missing:
        raise SystemExit(f"Budget calibration missing EqTime/EqEval planner entries: {', '.join(missing)}")


def _clean_budget_values(values: object) -> dict[str, object]:
    if not isinstance(values, dict):
        raise SystemExit("Budget calibration values must be objects keyed by planner.")
    out: dict[str, object] = {}
    int_keys = {"max_replan_expansions", "min_unified_eval_count", "eqeval_probe_guard_multiplier", "beam_width", "beam_depth", "candidate_pool_size", "repair_top_k", "repair_max_insertions", "packing_candidate_limit"}
    float_keys = {"max_replan_seconds", "min_replan_seconds"}
    bool_keys = {"fixed_replan_budget"}
    for key, value in values.items():
        if key in int_keys:
            out[key] = int(value)
        elif key in float_keys:
            out[key] = float(value)
        elif key in bool_keys:
            out[key] = bool(value)
    return out


def _write_budget_profile(
    df: pd.DataFrame,
    path: Path,
    calibration_path: Path | None,
    budget_overrides: dict[str, dict[str, object]],
) -> None:
    keys = [
        "scenario_seed",
        "env_seed",
        "wind_level",
        "obstacle_density",
        "actual_gust_multiplier",
        "scenario_profile",
        "ood_category",
    ]
    keys = [col for col in keys if col in df.columns]
    cols = keys + ["runtime_seconds", "candidate_expansions", "replan_latency_p95", "replan_latency_p99"]
    profile = df[df["planner"] == "RAWA-RHP"][cols].copy()
    profile["calibration_source"] = str(calibration_path) if calibration_path is not None else "none"
    profile["calibrated_planner_count"] = len(budget_overrides)
    profile.to_csv(path, index=False)


def _manifest_text(
    args: argparse.Namespace,
    seeds: list[int],
    planners: list[str],
    gusts: Iterable[float],
    wind_levels: Iterable[str],
    densities: Iterable[str],
    profiles: Iterable[str],
    optimizer_seeds: list[int],
    episode_path: Path,
    summary_path: Path,
    budget_profile_path: Path | None,
) -> str:
    lines = [
        f"# Optimization Suite {args.mode}",
        "",
        f"- seeds: {seeds[0]}-{seeds[-1]} ({len(seeds)})",
        f"- planners: {', '.join(planners)}",
        f"- gusts: {', '.join(f'{g:.1f}' for g in gusts)}",
        f"- wind levels: {', '.join(wind_levels)}",
        f"- obstacle densities: {', '.join(densities)}",
        f"- scenario profiles: {', '.join(profiles)}",
        f"- budget mode: {args.budget_mode}",
        f"- budget calibration: {args.budget_calibration if args.budget_calibration is not None else 'none'}",
        f"- randomize planners: {args.randomize_planners}",
        f"- workers: {args.workers}",
        f"- optimizer seeds: {optimizer_seeds[0]}-{optimizer_seeds[-1]} ({len(optimizer_seeds)})",
        f"- episodes: {episode_path}",
        f"- summary: {summary_path}",
    ]
    if budget_profile_path is not None:
        lines.append(f"- budget profile: {budget_profile_path}")
    if args.mode == "full":
        lines.extend(
            [
                "",
                "Formal full-mode matrix is fixed to seeds 1-20 unless overridden, ID/correlated_gust/narrow_passage profiles, all calm/moderate/severe wind levels, sparse/cluttered densities, and actual_gust_multiplier=1.0.",
                "Episode RNG seeds are shared across planners for each paired scenario; algorithm seeds remain planner-specific.",
            ]
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()

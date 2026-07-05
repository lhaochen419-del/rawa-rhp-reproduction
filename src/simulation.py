from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np

from .astar import astar_path
from .config import SimulationConfig, stable_seed
from .energy import EdgeMetrics, compute_edge_metrics
from .environment import InspectionEnvironment
from .geometry import Point, route_to_string
from .planners import EdgeCache, PlannerContext, make_planner


def build_edge_cache(env: InspectionEnvironment, cfg: SimulationConfig) -> EdgeCache:
    nodes = env.nodes()
    edges: EdgeCache = {}
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            path_ij = astar_path(env, nodes[i], nodes[j])
            rng_ij = np.random.default_rng(stable_seed(env.seed, env.wind_level, env.obstacle_density, i, j, "edge"))
            rng_ji = np.random.default_rng(stable_seed(env.seed, env.wind_level, env.obstacle_density, j, i, "edge"))
            edges[(i, j)] = compute_edge_metrics(i, j, path_ij, env, cfg, rng_ij)
            edges[(j, i)] = compute_edge_metrics(j, i, list(reversed(path_ij)), env, cfg, rng_ji)
    return edges


def run_episode(
    env: InspectionEnvironment,
    edges: EdgeCache,
    planner_name: str,
    cfg: SimulationConfig,
    actual_gust_multiplier: float = 1.0,
    episode_rng_seed: int | None = None,
    algorithm_seed: int | None = None,
    planner_overrides: Dict[str, object] | None = None,
    include_latency_trace: bool = False,
) -> Dict[str, object]:
    planner = make_planner(planner_name)
    if algorithm_seed is not None:
        setattr(planner, "algorithm_seed", int(algorithm_seed))
    if planner_overrides:
        _apply_planner_overrides(planner, planner_overrides)
    target_values = {target.idx + 1: target.value for target in env.targets}
    context = PlannerContext(cfg=cfg, edges=edges, target_values=target_values)
    planner.reset(context)
    if planner_overrides:
        _apply_planner_overrides(planner, planner_overrides)

    if episode_rng_seed is None:
        episode_rng_seed = stable_seed(env.seed, env.wind_level, env.obstacle_density, actual_gust_multiplier, "episode")
    if algorithm_seed is None:
        algorithm_seed = stable_seed(env.seed, env.wind_level, env.obstacle_density, planner_name, actual_gust_multiplier, "algorithm")
    current = 0
    unvisited = set(target_values.keys())
    visited: List[int] = []
    route_points: List[Point] = [env.depot]
    route_node_ids: List[int] = [0]
    battery = cfg.battery_capacity
    energy_used = 0.0
    planned_policy_budget = 0.0
    mission_time = 0.0
    path_risk_accumulated = 0.0
    expected_clearance_events = 0.0
    traversed_edges = 0
    min_clearance = float("inf")
    battery_violation = False
    emergency_abort = False
    replan_latencies: List[float] = []
    candidate_expansions = 0
    pruning_ratios: List[float] = []
    cache_hit_rates: List[float] = []
    anytime_stop_reasons: List[str] = []
    missed_single_feasible: List[float] = []
    missed_reject_reserve: List[float] = []
    missed_reject_avg_risk: List[float] = []
    missed_reject_edge_risk: List[float] = []
    missed_top_values: List[float] = []
    packing_attempts = 0
    packing_accepts = 0
    packing_added_values: List[float] = []
    packing_stop_reasons: List[str] = []

    while unvisited:
        replan_start = time.perf_counter()
        nxt = planner.choose_next(current, unvisited, battery)
        min_replan_seconds = float(getattr(planner, "min_replan_seconds", 0.0))
        elapsed = time.perf_counter() - replan_start
        if min_replan_seconds > elapsed:
            time.sleep(min_replan_seconds - elapsed)
        replan_latencies.append(time.perf_counter() - replan_start)
        diag = planner.diagnostics()
        candidate_expansions += int(diag.get("candidate_expansions", 0))
        pruning_ratios.append(float(diag.get("pruning_ratio", 0.0)))
        cache_hit_rates.append(float(diag.get("cache_hit_rate", 1.0)))
        anytime_stop_reasons.append(str(diag.get("anytime_stop_reason", "not_applicable")))
        missed_single_feasible.append(float(diag.get("missed_single_feasible", 0.0)))
        missed_reject_reserve.append(float(diag.get("missed_reject_reserve", 0.0)))
        missed_reject_avg_risk.append(float(diag.get("missed_reject_avg_risk", 0.0)))
        missed_reject_edge_risk.append(float(diag.get("missed_reject_edge_risk", 0.0)))
        missed_top_values.append(float(diag.get("missed_top_value", 0.0)))
        packing_attempts += int(diag.get("packing_attempts", 0))
        packing_accepts += int(diag.get("packing_accepts", 0))
        packing_added_values.append(float(diag.get("packing_added_value", 0.0)))
        packing_stop_reasons.append(str(diag.get("packing_stop_reason", "not_applicable")))
        if nxt is None:
            break
        edge = edges[(current, nxt)]
        edge_rng = np.random.default_rng(stable_seed(episode_rng_seed, current, nxt, "actual_edge"))
        actual_energy = planner.actual_edge_energy(edge, edge_rng, actual_gust_multiplier, include_photo=True)
        planned_policy_budget += planner.planned_edge_energy(edge, include_photo=True)
        battery -= actual_energy
        energy_used += actual_energy
        mission_time += edge.distance / cfg.v_ground + cfg.photo_time
        path_risk_accumulated = _combine_risk(path_risk_accumulated, edge.risk)
        expected_clearance_events += edge.risk
        traversed_edges += 1
        min_clearance = min(min_clearance, edge.min_clearance)
        if battery < 0.0:
            battery_violation = True
            emergency_abort = True
            break
        visited.append(nxt)
        unvisited.remove(nxt)
        current = nxt
        route_points.append(env.nodes()[nxt])
        route_node_ids.append(nxt)

    return_edge = edges[(current, 0)] if current != 0 else None
    if return_edge is not None:
        edge_rng = np.random.default_rng(stable_seed(episode_rng_seed, current, 0, "actual_edge"))
        actual_return = planner.actual_edge_energy(return_edge, edge_rng, actual_gust_multiplier, include_photo=False)
        planned_policy_budget += planner.planned_edge_energy(return_edge, include_photo=False)
        battery -= actual_return
        energy_used += actual_return
        mission_time += return_edge.distance / cfg.v_ground
        path_risk_accumulated = _combine_risk(path_risk_accumulated, return_edge.risk)
        expected_clearance_events += return_edge.risk
        traversed_edges += 1
        min_clearance = min(min_clearance, return_edge.min_clearance)
        route_points.append(env.depot)
        route_node_ids.append(0)
        if battery < 0.0:
            battery_violation = True
            emergency_abort = True

    collected_value = float(sum(target_values[i] for i in visited))
    total_value = env.total_target_value()
    weighted_coverage = collected_value / total_value
    clearance_violation_rate = expected_clearance_events / max(1, traversed_edges)
    clearance_threshold = float(getattr(cfg, "clearance_event_threshold", 0.05))
    mission_success = (not battery_violation) and clearance_violation_rate <= clearance_threshold
    safe_weighted_coverage = weighted_coverage if mission_success else 0.0
    reserve_margin = battery - cfg.reserve_floor
    reserve_shortfall = reserve_margin < 0.0
    reserve_shortfall_probability = float(1.0 / (1.0 + np.exp(max(-60.0, min(60.0, reserve_margin / 12.0)))))
    clearance_violation = clearance_violation_rate > clearance_threshold or min_clearance < (cfg.uav_radius + cfg.safety_buffer)
    return_success = (not battery_violation) and battery >= 0.0
    return_failure = not return_success
    row = {
        "seed": env.seed,
        "env_seed": env.seed,
        "episode_rng_seed": int(episode_rng_seed),
        "algorithm_seed": int(algorithm_seed),
        "wind_level": env.wind_level,
        "obstacle_density": env.obstacle_density,
        "planner": planner_name,
        "planner_replan_budget_seconds": float(getattr(planner, "max_replan_seconds", 0.0)),
        "n_targets": len(env.targets),
        "visited_targets": len(visited),
        "coverage_ratio": len(visited) / float(len(env.targets)),
        "weighted_coverage": weighted_coverage,
        "safe_weighted_coverage": safe_weighted_coverage,
        "mission_success": bool(mission_success),
        "collected_value": collected_value,
        "total_value": total_value,
        "energy_used": energy_used,
        "planned_policy_budget": planned_policy_budget,
        "final_battery": battery,
        "reserve_margin": reserve_margin,
        "reserve_shortfall": bool(reserve_shortfall),
        "reserve_shortfall_probability": reserve_shortfall_probability,
        "mission_time": mission_time,
        "return_success": bool(return_success),
        "return_failure": bool(return_failure),
        "battery_violation": bool(battery_violation),
        "path_risk_accumulated": path_risk_accumulated,
        "clearance_violation_rate": clearance_violation_rate,
        "min_clearance": min_clearance if np.isfinite(min_clearance) else 0.0,
        "clearance_violation": bool(clearance_violation),
        "emergency_abort": bool(emergency_abort),
        "actual_gust_multiplier": actual_gust_multiplier,
        "scenario_profile": env.scenario_profile,
        "ood_category": getattr(env, "ood_category", "id"),
        "replan_count": int(len(replan_latencies)),
        "replan_latency_mean": float(np.mean(replan_latencies)) if replan_latencies else 0.0,
        "replan_latency_p95": float(np.quantile(replan_latencies, 0.95)) if replan_latencies else 0.0,
        "replan_latency_p99": float(np.quantile(replan_latencies, 0.99)) if replan_latencies else 0.0,
        "replan_latency_max": float(np.max(replan_latencies)) if replan_latencies else 0.0,
        "candidate_expansions": int(candidate_expansions),
        "risk_evals": int(candidate_expansions),
        "unified_eval_count": int(candidate_expansions + candidate_expansions + packing_attempts),
        "cache_hit_rate": float(np.mean(cache_hit_rates)) if cache_hit_rates else 1.0,
        "pruning_ratio": float(np.mean(pruning_ratios)) if pruning_ratios else 0.0,
        "anytime_stop_reason": _dominant_stop_reason(anytime_stop_reasons),
        "packing_attempts": int(packing_attempts),
        "packing_accepts": int(packing_accepts),
        "packing_added_value": float(np.sum(packing_added_values)) if packing_added_values else 0.0,
        "packing_attempts_per_replan": float(packing_attempts / max(1, len(replan_latencies))),
        "packing_accepts_per_replan": float(packing_accepts / max(1, len(replan_latencies))),
        "packing_added_value_per_replan": float(np.sum(packing_added_values) / max(1, len(replan_latencies))) if packing_added_values else 0.0,
        "packing_added_value_per_total_value": float(np.sum(packing_added_values) / max(total_value, 1e-9)) if packing_added_values else 0.0,
        "packing_accept_rate": float(packing_accepts / max(1, packing_attempts)),
        "packing_stop_reason": _dominant_stop_reason(packing_stop_reasons),
        "missed_single_feasible_max": float(np.max(missed_single_feasible)) if missed_single_feasible else 0.0,
        "missed_reject_reserve_max": float(np.max(missed_reject_reserve)) if missed_reject_reserve else 0.0,
        "missed_reject_avg_risk_max": float(np.max(missed_reject_avg_risk)) if missed_reject_avg_risk else 0.0,
        "missed_reject_edge_risk_max": float(np.max(missed_reject_edge_risk)) if missed_reject_edge_risk else 0.0,
        "missed_top_value_max": float(np.max(missed_top_values)) if missed_top_values else 0.0,
        "route": route_to_string(route_points),
        "route_node_ids": "-".join(str(i) for i in route_node_ids),
    }
    if include_latency_trace:
        row["replan_latency_trace"] = ";".join(f"{value:.9f}" for value in replan_latencies)
    return row


def _combine_risk(a: float, b: float) -> float:
    return float(1.0 - (1.0 - a) * (1.0 - b))


def _apply_planner_overrides(planner, overrides: Dict[str, object]) -> None:
    for key, value in overrides.items():
        setattr(planner, key, value)
    base = getattr(planner, "_adaptive_budget_base", None)
    if base is not None:
        values = list(base)
        fields = [
            "beam_width",
            "beam_depth",
            "candidate_pool_size",
            "repair_top_k",
            "repair_max_insertions",
            "packing_candidate_limit",
            "max_replan_expansions",
        ]
        changed = False
        for idx, field in enumerate(fields):
            if field in overrides:
                values[idx] = int(getattr(planner, field))
                changed = True
        if changed:
            planner._adaptive_budget_base = tuple(values)


def _dominant_stop_reason(values: List[str]) -> str:
    if not values:
        return "none"
    counts: Dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(counts, key=counts.get)

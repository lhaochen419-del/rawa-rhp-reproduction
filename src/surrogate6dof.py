from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np

from .config import SimulationConfig, stable_seed
from .environment import InspectionEnvironment
from .geometry import distance


@dataclass(frozen=True)
class Surrogate6DoFResult:
    task_success: bool
    energy_6dof: float
    energy_mape: float
    coverage_drop: float
    safety_events: int
    min_clearance: float


def simulate_route_6dof(
    env: InspectionEnvironment,
    route_node_ids: Iterable[int],
    cfg: SimulationConfig,
    planned_2d_energy: float,
    coverage_2d: float,
    seed: int,
) -> Surrogate6DoFResult:
    route = [int(x) for x in route_node_ids]
    if len(route) < 2:
        return Surrogate6DoFResult(False, 0.0, 1.0, 1.0, 1, 0.0)
    rng = np.random.default_rng(stable_seed(env.seed, env.wind_level, env.obstacle_density, seed, "6dof"))
    nodes = env.nodes()
    battery_scale = float(rng.normal(1.0, 0.025))
    load_scale = float(rng.normal(1.035, 0.030))
    attitude_loss = 0.0
    energy = 0.0
    min_clearance = float("inf")
    safety_events = 0
    for i, j in zip(route[:-1], route[1:]):
        p0 = nodes[i]
        p1 = nodes[j]
        dist = distance(p0, p1)
        midpoint = np.asarray([(p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5], dtype=float)
        wind = env.wind.vector(midpoint)
        wind_speed = float(np.linalg.norm(wind))
        gust = float(env.wind.gust_std(midpoint)[0])
        leg_clearance = float(env.clearance_at(midpoint))
        min_clearance = min(min_clearance, leg_clearance)
        vertical_coupling = max(0.0, rng.normal(0.018 * wind_speed, 0.012 + 0.004 * gust))
        attitude_loss += vertical_coupling * dist / max(cfg.world_size, 1.0)
        leg_energy = dist * (cfg.c0 + cfg.c1 * (cfg.v_ground + 0.18 * wind_speed) ** 2)
        leg_energy *= load_scale * (1.0 + vertical_coupling)
        if j != 0:
            leg_energy += cfg.e_photo * float(rng.normal(1.0, 0.015))
        energy += leg_energy
        if leg_clearance < cfg.uav_radius + cfg.safety_buffer * 0.75:
            safety_events += 1
    effective_capacity = cfg.battery_capacity * battery_scale
    if energy > effective_capacity or effective_capacity - energy < 0.5 * cfg.reserve_floor:
        safety_events += 1
    coverage_drop = float(min(1.0, max(0.0, 0.35 * attitude_loss + 0.08 * safety_events)))
    success = safety_events == 0 and coverage_drop <= 0.10
    energy_mape = abs(energy - planned_2d_energy) / max(abs(planned_2d_energy), 1.0)
    return Surrogate6DoFResult(
        task_success=bool(success),
        energy_6dof=float(energy),
        energy_mape=float(energy_mape),
        coverage_drop=float(min(coverage_2d, coverage_drop)),
        safety_events=int(safety_events),
        min_clearance=float(min_clearance if np.isfinite(min_clearance) else 0.0),
    )


def summarize_6dof(rows: list[Dict[str, object]]) -> Dict[str, float]:
    if not rows:
        return {
            "tasks": 0.0,
            "success_rate": 0.0,
            "energy_mape": 1.0,
            "coverage_drop": 1.0,
            "safety_events": 0.0,
        }
    success = np.asarray([bool(r["task_success"]) for r in rows], dtype=float)
    mape = np.asarray([float(r["energy_mape"]) for r in rows], dtype=float)
    drop = np.asarray([float(r["coverage_drop"]) for r in rows], dtype=float)
    events = np.asarray([int(r["safety_events"]) for r in rows], dtype=float)
    return {
        "tasks": float(len(rows)),
        "success_rate": float(np.mean(success)),
        "energy_mape": float(np.mean(mape)),
        "coverage_drop": float(np.mean(drop)),
        "safety_events": float(np.sum(events)),
    }

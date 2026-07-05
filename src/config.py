from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "default.json"


@dataclass(frozen=True)
class SimulationConfig:
    world_size: float = 100.0
    grid_size: int = 50
    n_targets: int = 32
    battery_capacity: float = 1200.0
    reserve_floor: float = 80.0
    v_ground: float = 5.0
    v_air_max: float = 12.0
    c0: float = 1.0
    c1: float = 0.045
    c2: float = 0.070
    c_turn: float = 2.0
    e_photo: float = 5.0
    photo_time: float = 2.0
    sigma0: float = 0.25
    k_cross: float = 0.08
    k_gust: float = 0.05
    uav_radius: float = 0.35
    safety_buffer: float = 0.8
    clearance_event_threshold: float = 0.05
    p_max: float = 0.08
    gust_samples: int = 32
    alpha: float = 0.010
    beta: float = 0.006
    gamma: float = 8.0
    delta: float = 0.004

    @property
    def cell_size(self) -> float:
        return self.world_size / float(self.grid_size)


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> SimulationConfig:
    data = json.loads(path.read_text())
    return SimulationConfig(**data)


def profile_seeds(profile: str) -> List[int]:
    if profile == "quick":
        return list(range(5))
    if profile in {"full", "stress"}:
        return list(range(30))
    if profile in {"holdout", "stress_holdout"}:
        return list(range(30, 60))
    raise ValueError(f"Unknown profile: {profile}")


def profile_planners(profile: str) -> List[str]:
    main = [
        "NearestNeighbor",
        "ValuePerDistance",
        "WindAwareGreedy",
        "ReserveOnlyPlanner",
        "FairRiskAwareGreedy",
        "FairALNSPlanner",
        "RAWA-RHP",
    ]
    ablations = [
        "RAWA-NoPacking",
        "RAWA-NoRisk",
        "RAWA-NoBeam",
        "RAWA-NoAdaptiveSearch",
        "RAWA-NoPacking-EqBudget",
        "RAWA-NoRisk-EqBudget",
        "RAWA-NoBeam-EqBudget",
        "RAWA-NoAdaptiveSearch-EqBudget",
        "RAWA-NoWind",
    ]
    if profile == "quick":
        return main
    if profile in {"full", "holdout"}:
        return main + ablations
    if profile in {"stress", "stress_holdout"}:
        return ["ReserveOnlyPlanner", "RAWA-RHP"]
    raise ValueError(f"Unknown profile: {profile}")


WIND_LEVELS = ["calm", "moderate", "severe"]
OBSTACLE_DENSITIES = ["sparse", "cluttered"]


def stable_seed(*parts: object) -> int:
    text = "|".join(str(p) for p in parts)
    h = 2166136261
    for ch in text.encode("utf-8"):
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def ensure_output_dirs(root: Path = ROOT / "outputs") -> Dict[str, Path]:
    paths = {
        "results": root / "results",
        "figures": root / "figures",
        "tables": root / "tables",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.special import erfc

from .config import SimulationConfig
from .geometry import Point, segment_midpoints


def normal_survival(z: np.ndarray) -> np.ndarray:
    return 0.5 * erfc(z / math.sqrt(2.0))


def path_risk(
    path: Sequence[Point],
    wind,
    clearance_values: np.ndarray,
    cfg: SimulationConfig,
) -> float:
    mids, dirs, lengths = segment_midpoints(path)
    if len(mids) == 0:
        return 0.0
    w = wind.vector(mids)
    cross = np.abs(w[:, 0] * (-dirs[:, 1]) + w[:, 1] * dirs[:, 0])
    gust = wind.gust_std(mids)
    sigma = cfg.sigma0 + cfg.k_cross * cross + cfg.k_gust * gust
    margin = clearance_values - cfg.uav_radius - cfg.safety_buffer
    z = margin / np.maximum(sigma, 1e-6)
    local = normal_survival(z)
    local = np.clip(local * np.clip(lengths / 12.0, 0.0, 0.35), 0.0, 0.95)
    log_survival = np.sum(np.log1p(-local))
    return float(np.clip(1.0 - math.exp(log_survival), 0.0, 1.0))

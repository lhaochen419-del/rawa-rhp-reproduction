from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import SimulationConfig
from .geometry import Point, path_length, segment_midpoints, turn_angle_sum
from .risk import path_risk


@dataclass
class EdgeMetrics:
    source: int
    target: int
    path: Sequence[Point]
    distance: float
    energy_mean: float
    energy_q80: float
    energy_q90: float
    energy_q95: float
    energy_tail90: float
    energy_tail95: float
    energy_adaptive_mean: float
    energy_adaptive_q80: float
    energy_adaptive_q90: float
    energy_adaptive_q95: float
    energy_adaptive_tail90: float
    energy_adaptive_tail95: float
    no_wind_energy: float
    risk: float
    min_clearance: float
    samples: np.ndarray
    adaptive_samples: np.ndarray

    def sampled_energy(self, rng: np.random.Generator, gust_multiplier: float = 1.0, adaptive: bool = False) -> float:
        samples = self.adaptive_samples if adaptive else self.samples
        mean = self.energy_adaptive_mean if adaptive else self.energy_mean
        if len(samples) == 0:
            return mean
        sample = float(samples[int(rng.integers(0, len(samples)))])
        return float(max(0.0, mean + gust_multiplier * (sample - mean)))


def compute_edge_metrics(
    source: int,
    target: int,
    path: Sequence[Point],
    env,
    cfg: SimulationConfig,
    rng: np.random.Generator,
) -> EdgeMetrics:
    mids, dirs, lengths = segment_midpoints(path)
    if len(mids) == 0:
        samples = np.zeros(cfg.gust_samples, dtype=float)
        return EdgeMetrics(source, target, path, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.inf, samples, samples)

    wind_nom = env.wind.vector(mids)
    gust_std = env.wind.gust_std(mids)
    turn_energy = cfg.c_turn * turn_angle_sum(path)
    nominal = _energy_for_wind(wind_nom, dirs, lengths, cfg) + turn_energy

    samples = []
    for _ in range(cfg.gust_samples):
        if hasattr(env.wind, "sample_gust"):
            gust = env.wind.sample_gust(mids, rng, wind_nom.shape)
        else:
            gust = rng.normal(0.0, gust_std[:, None], size=wind_nom.shape)
        wind_sample = wind_nom + gust
        samples.append(_energy_for_wind(wind_sample, dirs, lengths, cfg) + turn_energy)
    sample_arr = np.asarray(samples, dtype=float)
    combined = np.concatenate([sample_arr, np.asarray([nominal])])
    q80 = _quantile(combined, 0.80)
    q90 = _quantile(combined, 0.90)
    q95 = _quantile(combined, 0.95)
    tail90 = _tail(combined, 0.90)
    tail95 = _tail(combined, 0.95)
    no_wind = cfg.c0 * float(np.sum(lengths)) + cfg.c1 * cfg.v_ground**2 * float(np.sum(lengths)) + turn_energy
    adaptive_arr = _adaptive_energy_samples(sample_arr, no_wind)
    adaptive_nominal = float(_adaptive_energy_samples(np.asarray([nominal], dtype=float), no_wind)[0])
    adaptive_combined = np.concatenate([adaptive_arr, np.asarray([adaptive_nominal])])
    adaptive_mean = float(np.mean(adaptive_combined))
    adaptive_q80 = _quantile(adaptive_combined, 0.80)
    adaptive_q90 = _quantile(adaptive_combined, 0.90)
    adaptive_q95 = _quantile(adaptive_combined, 0.95)
    adaptive_tail90 = _tail(adaptive_combined, 0.90)
    adaptive_tail95 = _tail(adaptive_combined, 0.95)
    clearance = env.clearance_at(mids)
    risk = path_risk(path, env.wind, clearance, cfg)
    return EdgeMetrics(
        source=source,
        target=target,
        path=list(path),
        distance=path_length(path),
        energy_mean=float(np.mean(combined)),
        energy_q80=float(q80),
        energy_q90=float(q90),
        energy_q95=float(q95),
        energy_tail90=float(tail90),
        energy_tail95=float(tail95),
        energy_adaptive_mean=float(adaptive_mean),
        energy_adaptive_q80=float(adaptive_q80),
        energy_adaptive_q90=float(adaptive_q90),
        energy_adaptive_q95=float(adaptive_q95),
        energy_adaptive_tail90=float(adaptive_tail90),
        energy_adaptive_tail95=float(adaptive_tail95),
        no_wind_energy=float(no_wind),
        risk=float(risk),
        min_clearance=float(np.min(clearance)),
        samples=sample_arr,
        adaptive_samples=adaptive_arr,
    )


def _energy_for_wind(
    wind: np.ndarray,
    dirs: np.ndarray,
    lengths: np.ndarray,
    cfg: SimulationConfig,
) -> float:
    ground_velocity = cfg.v_ground * dirs
    air_velocity = ground_velocity - wind
    air_speed = np.linalg.norm(air_velocity, axis=1)
    air_speed = np.minimum(air_speed, cfg.v_air_max)
    headwind = np.maximum(0.0, -np.sum(wind * dirs, axis=1))
    per_m = cfg.c0 + cfg.c1 * air_speed**2 + cfg.c2 * headwind**2
    return float(np.sum(lengths * per_m))


def _adaptive_energy_samples(values: np.ndarray, no_wind: float) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    wind_increment = vals - float(no_wind)
    reduced = np.where(wind_increment >= 0.0, 0.20 * wind_increment, 0.92 * wind_increment)
    return np.maximum(0.0, float(no_wind) + reduced)


def _tail(values: np.ndarray, q: float) -> float:
    values = np.sort(np.asarray(values, dtype=float))
    if len(values) == 0:
        return 0.0
    start = int(np.floor(q * (len(values) - 1)))
    return float(np.mean(values[start:]))


def _quantile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return 0.0
    return float(np.quantile(values, q))

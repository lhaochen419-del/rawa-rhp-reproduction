from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


WIND_BANDS = {
    "calm": (0.6, 1.8, 0.35),
    "moderate": (3.0, 5.0, 0.75),
    "severe": (6.0, 8.0, 1.20),
}

OOD_CATEGORIES = [
    "scale_target_count",
    "obstacle_count",
    "random_depot",
    "nonconvex_obstacles",
    "narrow_passage",
    "unseen_wind_direction",
    "correlated_gust",
    "mixed_ood",
]


@dataclass
class WindField:
    level: str
    seed: int
    world_size: float
    profile: str = "id"

    def __post_init__(self) -> None:
        self.category = _profile_category(self.profile)
        if self.level not in WIND_BANDS:
            raise ValueError(f"Unknown wind level {self.level}")
        lo, hi, gust = WIND_BANDS[self.level]
        rng = np.random.default_rng(self.seed + 7919)
        mag = float(rng.uniform(lo, hi))
        if self._uses_unseen_direction():
            unseen = np.asarray([math.pi / 8.0, 5.0 * math.pi / 8.0, 9.0 * math.pi / 8.0, 13.0 * math.pi / 8.0])
            theta = float(unseen[int(rng.integers(0, len(unseen)))] + rng.normal(0.0, math.pi / 48.0))
            mag *= float(rng.uniform(1.05, 1.22))
            gust *= float(rng.uniform(1.20, 1.45))
        else:
            theta = float(rng.uniform(0.0, 2.0 * math.pi))
        self.mean = np.array([mag * math.cos(theta), mag * math.sin(theta)], dtype=float)
        self.gust_base = gust
        vortex_count = 5 if self._is_ood() else 3
        self.vortex_centers = rng.uniform(0.12, 0.88, size=(vortex_count, 2)) * self.world_size
        scale = {"calm": 0.25, "moderate": 0.75, "severe": 1.15}[self.level]
        if self._is_ood():
            scale *= 1.35
        self.vortex_strength = rng.uniform(-1.0, 1.0, size=vortex_count) * scale
        self.phase = rng.uniform(0.0, 2.0 * math.pi, size=4)
        self.wave_amp = {"calm": 0.20, "moderate": 0.55, "severe": 0.95}[self.level]
        if self._is_ood():
            self.wave_amp *= 1.30

    def _is_ood(self) -> bool:
        return self.profile.startswith("ood")

    def _uses_unseen_direction(self) -> bool:
        return self.category in {"unseen_wind_direction", "mixed_ood"} or self.profile == "ood_extreme"

    def vector(self, xy: np.ndarray) -> np.ndarray:
        pts = np.asarray(xy, dtype=float)
        scalar = False
        if pts.ndim == 1:
            pts = pts[None, :]
            scalar = True
        out = np.repeat(self.mean[None, :], len(pts), axis=0)
        x = pts[:, 0] / self.world_size
        y = pts[:, 1] / self.world_size
        out[:, 0] += self.wave_amp * (
            np.sin(2.0 * math.pi * y + self.phase[0])
            + 0.45 * np.sin(4.0 * math.pi * x + self.phase[1])
        )
        out[:, 1] += self.wave_amp * (
            np.cos(2.0 * math.pi * x + self.phase[2])
            - 0.35 * np.sin(4.0 * math.pi * y + self.phase[3])
        )
        for center, strength in zip(self.vortex_centers, self.vortex_strength):
            rel = pts - center[None, :]
            r2 = np.sum(rel * rel, axis=1) + 16.0
            swirl = np.column_stack([-rel[:, 1], rel[:, 0]]) / np.sqrt(r2)[:, None]
            out += strength * np.exp(-r2 / (0.18 * self.world_size) ** 2)[:, None] * swirl
        if scalar:
            return out[0]
        return out

    def gust_std(self, xy: np.ndarray) -> np.ndarray:
        pts = np.asarray(xy, dtype=float)
        if pts.ndim == 1:
            pts = pts[None, :]
        x = pts[:, 0] / self.world_size
        y = pts[:, 1] / self.world_size
        modulation = 1.0 + 0.25 * np.sin(2.0 * math.pi * (x + y) + self.phase[0])
        return np.maximum(0.05, self.gust_base * modulation)

    def sample_gust(self, xy: np.ndarray, rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
        std = self.gust_std(xy)
        if self.category not in {"correlated_gust", "mixed_ood"} and self.profile != "ood_extreme" or len(std) == 0:
            return rng.normal(0.0, std[:, None], size=shape)
        shared = rng.normal(0.0, 1.0, size=(1, shape[1]))
        local = rng.normal(0.0, 1.0, size=shape)
        x = np.asarray(xy, dtype=float)[:, 0] / max(self.world_size, 1e-9)
        y = np.asarray(xy, dtype=float)[:, 1] / max(self.world_size, 1e-9)
        wave = np.sin(2.0 * math.pi * (0.7 * x + 0.3 * y) + self.phase[1])[:, None]
        correlated = 0.55 * shared + 0.45 * local + 0.30 * wave * rng.normal(0.0, 1.0, size=(1, shape[1]))
        return correlated * std[:, None]

    def summary(self) -> Tuple[float, float]:
        return float(np.linalg.norm(self.mean)), float(self.gust_base)


@dataclass
class OnlineWindBelief:
    """Small online bias model shared by planner and tests.

    It estimates a vector correction to a nominal wind field from observed
    local residuals. The update is intentionally conservative so final blind
    validation does not depend on unstable optimizer state.
    """

    nominal: WindField
    learning_rate: float = 0.35
    bias: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    residual_ema: float = 0.0
    observations: int = 0

    def vector(self, xy: np.ndarray) -> np.ndarray:
        base = self.nominal.vector(xy)
        return base + self.bias

    def gust_std(self, xy: np.ndarray) -> np.ndarray:
        scale = 1.0 + min(0.35, abs(self.residual_ema))
        return self.nominal.gust_std(xy) * scale

    def update(self, xy: np.ndarray, observed: np.ndarray) -> float:
        pts = np.asarray(xy, dtype=float)
        obs = np.asarray(observed, dtype=float)
        if pts.ndim == 1:
            pts = pts[None, :]
        if obs.ndim == 1:
            obs = obs[None, :]
        pred = self.nominal.vector(pts) + self.bias
        before = float(np.mean(np.linalg.norm(obs - pred, axis=1)))
        residual = np.mean(obs - pred, axis=0)
        rate = self.learning_rate / (1.0 + 0.08 * self.observations)
        self.bias = self.bias + rate * residual
        after_pred = self.nominal.vector(pts) + self.bias
        after = float(np.mean(np.linalg.norm(obs - after_pred, axis=1)))
        self.residual_ema = 0.85 * self.residual_ema + 0.15 * after
        self.observations += int(len(pts))
        return before - after


def _profile_category(profile: str) -> str:
    if profile.startswith("ood:"):
        category = profile.split(":", 1)[1]
        if category not in OOD_CATEGORIES:
            raise ValueError(f"Unknown OOD category {category}")
        return category
    if profile == "ood_extreme":
        return "mixed_ood"
    return "id"

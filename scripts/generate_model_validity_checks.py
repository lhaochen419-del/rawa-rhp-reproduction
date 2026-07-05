from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import SimulationConfig, load_config
from src.risk import normal_survival


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RAWA-RHP simulation-model sanity-check curves.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "results" / "model_validity_checks_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    energy = _energy_curves(cfg)
    risk = _risk_curves(cfg)
    scenarios = _scenario_validity_table()
    energy.to_csv(args.out_dir / "energy_model_sanity.csv", index=False)
    risk.to_csv(args.out_dir / "clearance_risk_sanity.csv", index=False)
    scenarios.to_csv(args.out_dir / "scenario_validity_summary.csv", index=False)
    _plot_energy(energy, args.out_dir)
    _plot_risk(risk, args.out_dir)
    (args.out_dir / "SIMULATION_MODEL_VALIDITY.md").write_text(_markdown(cfg, energy, risk, scenarios), encoding="utf-8")
    print(f"wrote {args.out_dir}")


def _energy_curves(cfg: SimulationConfig) -> pd.DataFrame:
    speeds = np.linspace(-8.0, 8.0, 65)
    rows = []
    length = 50.0
    turn = math.pi / 2.0
    for wind_component in speeds:
        for mode, wind in [
            ("head_tail", np.asarray([-wind_component, 0.0])),
            ("crosswind", np.asarray([0.0, wind_component])),
        ]:
            direction = np.asarray([1.0, 0.0])
            air = cfg.v_ground * direction - wind
            air_speed_uncapped = float(np.linalg.norm(air))
            air_speed = min(air_speed_uncapped, cfg.v_air_max)
            headwind = max(0.0, -float(np.dot(wind, direction)))
            per_m = cfg.c0 + cfg.c1 * air_speed**2 + cfg.c2 * headwind**2
            energy = length * per_m + cfg.c_turn * turn
            no_wind = length * (cfg.c0 + cfg.c1 * cfg.v_ground**2) + cfg.c_turn * turn
            adaptive_increment = energy - no_wind
            adaptive = max(0.0, no_wind + (0.20 if adaptive_increment >= 0.0 else 0.92) * adaptive_increment)
            rows.append(
                {
                    "mode": mode,
                    "wind_component": float(wind_component),
                    "air_speed_uncapped": air_speed_uncapped,
                    "air_speed_capped": air_speed,
                    "headwind_component": headwind,
                    "energy": float(energy),
                    "adaptive_energy": float(adaptive),
                    "no_wind_energy": float(no_wind),
                }
            )
    return pd.DataFrame(rows)


def _risk_curves(cfg: SimulationConfig) -> pd.DataFrame:
    rows = []
    clearances = np.linspace(cfg.uav_radius + cfg.safety_buffer, cfg.uav_radius + cfg.safety_buffer + 4.0, 81)
    crosswinds = [0.0, 2.5, 5.0, 7.5]
    gusts = [0.35, 0.75, 1.20, 1.80]
    length_scale = 0.35
    for cross in crosswinds:
        for gust in gusts:
            sigma = cfg.sigma0 + cfg.k_cross * cross + cfg.k_gust * gust
            for clearance in clearances:
                margin = clearance - cfg.uav_radius - cfg.safety_buffer
                local = float(normal_survival(np.asarray([margin / max(sigma, 1e-6)]))[0])
                event = float(np.clip(local * length_scale, 0.0, 0.95))
                rows.append(
                    {
                        "clearance": float(clearance),
                        "clearance_margin": float(margin),
                        "crosswind": float(cross),
                        "gust_std": float(gust),
                        "sigma": float(sigma),
                        "segment_event_probability": event,
                    }
                )
    return pd.DataFrame(rows)


def _scenario_validity_table() -> pd.DataFrame:
    rows = [
        {"scenario_family": "ID calm/open", "profile": "id", "wind": "calm", "obstacle_density": "sparse", "intended_role": "low-risk reference condition"},
        {"scenario_family": "ID routine", "profile": "id", "wind": "moderate", "obstacle_density": "sparse/cluttered", "intended_role": "nominal inspection condition"},
        {"scenario_family": "Stress", "profile": "id", "wind": "severe", "obstacle_density": "cluttered", "intended_role": "high reserve and clearance pressure"},
        {"scenario_family": "OOD correlated gust", "profile": "ood:correlated_gust", "wind": "all levels", "obstacle_density": "sparse/cluttered", "intended_role": "wind-uncertainty shift"},
        {"scenario_family": "OOD narrow passage", "profile": "ood:narrow_passage", "wind": "all levels", "obstacle_density": "sparse/cluttered", "intended_role": "geometric clearance shift"},
        {"scenario_family": "Gust x3", "profile": "id", "wind": "moderate/severe", "obstacle_density": "sparse/cluttered", "intended_role": "execution-time gust stress"},
    ]
    return pd.DataFrame(rows)


def _plot_energy(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for mode, label in [("head_tail", "head/tail component"), ("crosswind", "crosswind component")]:
        view = df[df["mode"] == mode]
        ax.plot(view["wind_component"], view["energy"], label=f"{label}: raw")
        ax.plot(view["wind_component"], view["adaptive_energy"], linestyle="--", label=f"{label}: adaptive")
    ax.set_xlabel("Wind component (simulation distance units/s)")
    ax.set_ylabel("Energy over 50 distance units")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"energy_model_sanity.{ext}", dpi=300)
    plt.close(fig)


def _plot_risk(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for cross in [0.0, 5.0, 7.5]:
        view = df[(df["crosswind"] == cross) & (df["gust_std"] == 1.20)]
        ax.plot(view["clearance_margin"], view["segment_event_probability"], label=f"crosswind={cross:g}")
    ax.axhline(0.05, color="black", linestyle="--", linewidth=0.8, label="event threshold reference")
    ax.set_xlabel("Clearance margin")
    ax.set_ylabel("Segment event probability")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"clearance_risk_sanity.{ext}", dpi=300)
    plt.close(fig)


def _markdown(cfg: SimulationConfig, energy: pd.DataFrame, risk: pd.DataFrame, scenarios: pd.DataFrame) -> str:
    energy_nonnegative = bool((energy["energy"] >= 0.0).all() and (energy["adaptive_energy"] >= 0.0).all())
    risk_monotone = []
    for (_, _), view in risk.groupby(["crosswind", "gust_std"], sort=False):
        ordered = view.sort_values("clearance_margin")
        risk_monotone.append(bool(np.all(np.diff(ordered["segment_event_probability"].to_numpy()) <= 1e-12)))
    lines = [
        "# Simulation Model Parameterisation and Validity Checks",
        "",
        "These checks support bounded simulation plausibility. They are not hardware flight validation and do not calibrate the model against real UAV logs.",
        "",
        "## Energy model sanity check",
        "",
        f"- Non-negative raw/adaptive energy: `{energy_nonnegative}`",
        "- Headwind and crosswind increase air-speed demand; tailwind benefit is bounded by the air-speed model and adaptive-energy transform.",
        "",
        "## Clearance-risk sanity check",
        "",
        f"- Risk is monotone non-increasing with clearance margin across sampled crosswind/gust settings: `{all(risk_monotone)}`",
        f"- Benchmark clearance event threshold: `{cfg.clearance_event_threshold}`",
        "",
        "## Scenario validity check",
        "",
        _markdown_table(scenarios),
        "",
        "The scenario families cover low-risk, nominal, pressure and OOD planning-layer conditions. They do not validate flight-control performance or real collision probability.",
    ]
    return "\n".join(lines) + "\n"


def _markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join([":--"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_KEYS = [
    "scenario_seed",
    "wind_level",
    "obstacle_density",
    "actual_gust_multiplier",
    "scenario_profile",
    "ood_category",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check paired EqTime/EqEval budget matching against a reference planner.")
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--reference", default="RAWA-RHP")
    parser.add_argument("--variants", required=True)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--time-field", default="replan_latency_p95")
    parser.add_argument("--eval-fields", default="unified_eval_count,candidate_expansions,risk_evals,packing_attempts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.episodes)
    variants = _split(args.variants)
    eval_fields = [field for field in _split(args.eval_fields) if field in df.columns]
    keys = [key for key in DEFAULT_KEYS if key in df.columns]
    rows = []
    variant_passes = []
    all_pass = True
    for variant in variants:
        if variant.endswith("-EqTime"):
            fields = [args.time_field]
        elif variant.endswith("-EqEval"):
            fields = eval_fields
        else:
            fields = [args.time_field]
        field_passes = {}
        for field in fields:
            if field not in df.columns:
                rows.append((variant, field, 0, 0.0, False, "missing field"))
                field_passes[field] = False
                continue
            paired = _paired(df, keys, args.reference, variant, field)
            if paired.empty:
                rows.append((variant, field, 0, 0.0, False, "missing pairs"))
                field_passes[field] = False
                continue
            ref_mean = float(paired[args.reference].astype(float).mean())
            var_mean = float(paired[variant].astype(float).mean())
            if ref_mean == 0.0:
                passed = var_mean == 0.0
                ratio = 1.0 if passed else float("inf")
            else:
                ratio = var_mean / ref_mean
                passed = (1.0 - args.tolerance) <= ratio <= (1.0 + args.tolerance)
            rows.append((variant, field, len(paired), ratio, passed, ""))
            field_passes[field] = bool(passed)
        if variant.endswith("-EqEval"):
            unified_pass = field_passes.get("unified_eval_count", False)
            component_fields = [field for field in ["candidate_expansions", "risk_evals", "packing_attempts"] if field in field_passes]
            components_pass = bool(component_fields and all(field_passes[field] for field in component_fields))
            variant_pass = bool(unified_pass or components_pass)
        else:
            variant_pass = all(field_passes.values()) if field_passes else False
        variant_passes.append((variant, variant_pass))
        all_pass = all_pass and variant_pass
    print("| variant | field | pairs | ratio | pass | note |")
    print("|:--|:--|--:|--:|:--|:--|")
    for variant, field, n_pairs, ratio, passed, note in rows:
        ratio_text = "inf" if ratio == float("inf") else f"{ratio:.6f}"
        print(f"| {variant} | {field} | {n_pairs} | {ratio_text} | {'yes' if passed else 'no'} | {note} |")
    print("")
    print("| variant | EqTime/EqEval overall pass |")
    print("|:--|:--|")
    for variant, passed in variant_passes:
        print(f"| {variant} | {'yes' if passed else 'no'} |")
    if not all_pass:
        raise SystemExit("Budget match check failed.")


def _paired(df: pd.DataFrame, keys: list[str], reference: str, variant: str, field: str) -> pd.DataFrame:
    view = df[df["planner"].isin([reference, variant])].copy()
    pivot = view.pivot_table(index=keys, columns="planner", values=field, aggfunc="mean")
    if reference not in pivot.columns or variant not in pivot.columns:
        return pd.DataFrame()
    return pivot[[reference, variant]].dropna().reset_index()


def _split(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


if __name__ == "__main__":
    main()

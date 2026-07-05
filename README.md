# RAWA-RHP Reproduction Package

This package contains the accepted evidence used for the RAWA-RHP simulation planning-layer manuscript.

## Contents

- `src/`: simulator, planners, wind/energy/risk models and metrics.
- `scripts/`: experiment, analysis, plotting, sensitivity and verification scripts.
- `configs/`: final simulator/planner configuration and scenario-suite descriptors.
- `data/raw_results/`: episode-level CSV files for baseline, ablation and latency evidence.
- `data/processed_results/`: paired statistics, ablation analysis and latency summaries.
- `data/figure_source_data/`: table and figure source data.
- `data/sensitivity/`: one-factor parameter sensitivity subset outputs, if generated.
- `data/model_validity/`: simulation-model sanity-check curves and source data.
- `docs/`: field dictionary, model-validity notes, sensitivity notes and archive checklist.

## Verification

```bash
python scripts/verify_reproduction.py --package-root .
python scripts/make_tables_figures.py --package-root .
```

## Evidence boundary

Formal manuscript evidence is limited to the files listed in `MANIFEST.json` under `formal_evidence_whitelist`. Smoke, development, budget-debug, old CVaR, NoRisk-Pure and tuning-run data are not manuscript evidence.

# Supplementary Reproduction and Sensitivity Material

## Reproduction package

The reproduction package contains the simulator, planner implementations, implemented baselines, analysis scripts, plotting scripts, accepted result files, figure source data, latency samples, parameter-sensitivity outputs and simulation-model validity checks. The package is prepared for public GitHub/Zenodo archival release after author confirmation of the final licence and repository metadata.

The formal manuscript evidence whitelist is restricted to:

- blind paired baseline episodes and paired statistics;
- gust x3 paired baseline episodes and paired statistics;
- formal full-mode ablation episodes and cluster-bootstrap/Holm analysis;
- strict latency benchmark episodes and summary;
- parameter-sensitivity subset outputs;
- simulation-model sanity-check source data.

Smoke, development, budget-debug, old CVaR, NoRisk-Pure and tuning-run data are not manuscript evidence.

## Verification script

The package includes `scripts/verify_reproduction.py`. It checks:

- blind paired SWC gain: 0.144743;
- gust x3 paired SWC gain: 0.201939;
- formal ablation gate pass;
- EqEval budget ratios within 5%;
- strict latency p95/p99: 1.754860 s / 1.759013 s;
- zero battery violation, clearance violation, reserve shortfall, return failure and emergency abort for RAWA-RHP in the formal simulation checks.

## Parameter sensitivity subset

The sensitivity subset uses seeds 301--307, three scenario profiles, three wind levels and two obstacle densities, giving 126 paired scenarios. The experiment is one-factor-at-a-time. Each variant changes one parameter family and is compared with the final setting on the same paired scenario.

The final setting reached mean SWC 0.875667 in this subset with zero evaluated hard failures. The largest mean changes were:

| Parameter family | Level | Mean delta SWC | Mean absolute delta SWC |
|---|---:|---:|---:|
| Beam/search budget | 0.5x | -0.023900 | 0.029825 |
| Reserve floor | 100 | -0.011881 | 0.019043 |
| Reserve floor | 60 | 0.012686 | 0.017460 |
| Risk dispersion scale | 0.75x | 0.000418 | 0.014185 |
| Risk dispersion scale | 1.25x | 0.001489 | 0.014101 |
| Gust samples | 16 | 0.001506 | 0.011981 |
| Gust samples | 64 | 0.003588 | 0.010818 |
| Energy quantile | q95 | 0.003691 | 0.009473 |

The interpretation is bounded. The final configuration is not a sharp single-seed optimum, but reserve conservatism and search budget are calibration-sensitive and should be rechecked when changing the simulator, mission scale or vehicle envelope.

## Simulation-model validity checks

The model-validity material reports plausibility checks rather than real-flight calibration.

- Energy sanity check: energy remains non-negative; adverse wind raises the energy curve; tailwind gains are bounded by the air-speed cap and adaptive-energy transform.
- Clearance-risk sanity check: risk decreases with clearance margin and increases with crosswind/gust dispersion.
- Scenario validity check: ID, severe, correlated-gust, narrow-passage and gust-stress scenarios cover low-risk, nominal, pressure and OOD planning-layer regimes.

These checks support planning-layer simulation plausibility only. They do not validate flight dynamics, true collision probabilities, hardware energy consumption or low-level collision avoidance.

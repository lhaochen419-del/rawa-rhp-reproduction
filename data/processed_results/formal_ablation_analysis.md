# Formal Ablation Analysis

- episodes: `outputs/results/formal_core_merged_s1_20_v2/full_1_20_episodes.csv`
- metric: `safe_weighted_coverage`
- threshold: `0.01`
- pairing keys: `scenario_seed, wind_level, obstacle_density, actual_gust_multiplier, scenario_profile, ood_category`
- cluster key: `scenario_seed`
- EqTime/EqEval tolerance: `0.05`
- duplicate method-scene rows: `0`
- expected rows: `5760`
- actual rows: `5760`
- completeness gate: `pass`
- RAWA hard safety gate: `pass`
- RAWA latency gate: `pass`
- formal gate: `pass`

| ablation | n | mean diff | median diff | ci95 low | ci95 high | p holm | budget ratio | pass |
|:--|--:|--:|--:|--:|--:|--:|--:|:--|
| RAWA-NoPacking | 360 | 0.163950 | 0.166667 | 0.158752 | 0.169033 | 0.000750 | 1.000000 | yes |
| RAWA-NoPacking-EqTime | 360 | 0.163950 | 0.166667 | 0.158887 | 0.169049 | 0.000750 | 1.014109 | yes |
| RAWA-NoPacking-EqEval | 360 | 0.144629 | 0.144000 | 0.139110 | 0.149929 | 0.000750 | 0.999548 | yes |
| RAWA-NoBeam | 360 | 0.018054 | 0.007937 | 0.014181 | 0.022173 | 0.000750 | 1.000000 | yes |
| RAWA-NoBeam-EqTime | 360 | 0.018054 | 0.007937 | 0.014151 | 0.022182 | 0.000750 | 1.014212 | yes |
| RAWA-NoBeam-EqEval | 360 | 0.017139 | 0.007663 | 0.013516 | 0.020799 | 0.000750 | 0.970686 | yes |
| RAWA-NoRisk | 360 | 0.033474 | 0.032922 | 0.031138 | 0.035675 | 0.000750 | 1.000000 | yes |
| RAWA-NoRisk-EqTime | 360 | 0.033589 | 0.033058 | 0.031286 | 0.035705 | 0.000750 | 1.016829 | yes |
| RAWA-NoRisk-EqEval | 360 | 0.033138 | 0.032654 | 0.030724 | 0.035398 | 0.000750 | 0.953818 | yes |
| RAWA-BlindReserve | 360 | 0.041053 | 0.038760 | 0.039000 | 0.043078 | 0.000750 | 1.000000 | yes |
| RAWA-BlindReserve-EqTime | 360 | 0.041053 | 0.038760 | 0.038982 | 0.043108 | 0.000750 | 1.015165 | yes |
| RAWA-BlindReserve-EqEval | 360 | 0.041828 | 0.038611 | 0.039172 | 0.044337 | 0.000750 | 0.975441 | yes |
| RAWA-NoAdaptiveSearch | 360 | 0.046174 | 0.038760 | 0.041649 | 0.051018 | 0.000750 | 1.000000 | yes |
| RAWA-NoAdaptiveSearch-EqTime | 360 | 0.046174 | 0.038760 | 0.041549 | 0.051018 | 0.000750 | 1.014107 | yes |
| RAWA-NoAdaptiveSearch-EqEval | 360 | 0.028184 | 0.022472 | 0.024293 | 0.031702 | 0.000750 | 0.981772 | yes |

A module passes only if mean diff >= threshold, scenario-seed cluster bootstrap lower95 > 0, Holm-adjusted paired-test p < 0.05, directional means are positive in ID/correlated_gust/narrow_passage/cluttered subsets, EqTime/EqEval budget ratios are within tolerance, and safety-rate deltas are not positive.

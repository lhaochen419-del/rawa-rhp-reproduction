# Baseline Paired Comparison

- metric: `safe_weighted_coverage`
- main planner: `RAWA-RHP`
- overall gate: `pass`

## blind100

- main mean: `0.843491`
- strongest baseline: `FairALNSPlanner` mean `0.698748`
- gate: `pass`

| baseline | n | mean diff | median diff | ci95 low | ci95 high | Holm p | safety not worse | pass |
|:--|--:|--:|--:|--:|--:|--:|:--|:--|
| FairALNSPlanner | 600 | 0.144743 | 0.085009 | 0.130698 | 0.159109 | 0.000150 | yes | yes |
| FairRiskAwareGreedy | 600 | 0.329982 | 0.322034 | 0.322147 | 0.338177 | 0.000150 | yes | yes |
| ReserveOnlyPlanner | 600 | 0.344981 | 0.343756 | 0.339113 | 0.351003 | 0.000150 | yes | yes |

## stress100_gust3

- main mean: `0.800568`
- strongest baseline: `FairALNSPlanner` mean `0.598629`
- gate: `pass`

| baseline | n | mean diff | median diff | ci95 low | ci95 high | Holm p | safety not worse | pass |
|:--|--:|--:|--:|--:|--:|--:|:--|:--|
| FairALNSPlanner | 400 | 0.201939 | 0.141146 | 0.181900 | 0.223108 | 0.000150 | yes | yes |
| FairRiskAwareGreedy | 400 | 0.335897 | 0.321739 | 0.325273 | 0.347078 | 0.000150 | yes | yes |
| ReserveOnlyPlanner | 400 | 0.406560 | 0.398473 | 0.399101 | 0.414153 | 0.000150 | yes | yes |

# Simulation Model Parameterisation and Validity Checks

These checks support bounded simulation plausibility. They are not hardware flight validation and do not calibrate the model against real UAV logs.

## Energy model sanity check

- Non-negative raw/adaptive energy: `True`
- Headwind and crosswind increase air-speed demand; tailwind benefit is bounded by the air-speed model and adaptive-energy transform.

## Clearance-risk sanity check

- Risk is monotone non-increasing with clearance margin across sampled crosswind/gust settings: `True`
- Benchmark clearance event threshold: `0.05`

## Scenario validity check

| scenario_family | profile | wind | obstacle_density | intended_role |
|:--|:--|:--|:--|:--|
| ID calm/open | id | calm | sparse | low-risk reference condition |
| ID routine | id | moderate | sparse/cluttered | nominal inspection condition |
| Stress | id | severe | cluttered | high reserve and clearance pressure |
| OOD correlated gust | ood:correlated_gust | all levels | sparse/cluttered | wind-uncertainty shift |
| OOD narrow passage | ood:narrow_passage | all levels | sparse/cluttered | geometric clearance shift |
| Gust x3 | id | moderate/severe | sparse/cluttered | execution-time gust stress |

The scenario families cover low-risk, nominal, pressure and OOD planning-layer conditions. They do not validate flight-control performance or real collision probability.

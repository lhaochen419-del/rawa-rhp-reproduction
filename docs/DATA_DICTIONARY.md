# Data Dictionary

## Episode-level fields

- `scenario_id` / `scenario_seed`: paired scenario identifier and seed.
- `seed`, `env_seed`, `episode_rng_seed`, `algorithm_seed`: deterministic random seeds.
- `planner`: planner name.
- `scenario_profile`, `ood_category`, `wind_level`, `obstacle_density`, `actual_gust_multiplier`: scenario descriptors.
- `n_targets`, `visited_targets`, `coverage_ratio`: target-count coverage.
- `collected_value`, `total_value`, `weighted_coverage`, `safe_weighted_coverage`: value collection and primary SWC metric.
- `battery_violation`, `clearance_violation`, `reserve_shortfall`, `return_success`, `return_failure`, `emergency_abort`: evaluated simulation hard-failure checks.
- `energy_used`, `planned_policy_budget`, `final_battery`, `reserve_margin`, `mission_time`: energy and mission telemetry in simulator units.
- `path_risk_accumulated`, `clearance_violation_rate`, `min_clearance`: clearance-risk telemetry.
- `replan_count`, `replan_latency_mean`, `replan_latency_p95`, `replan_latency_p99`, `replan_latency_max`, `runtime_seconds`: planning runtime telemetry.
- `candidate_expansions`, `risk_evals`, `packing_attempts`, `packing_accepts`, `unified_eval_count`: computation-budget telemetry.
- `route`, `route_node_ids`: executed route geometry and node sequence.

Boolean fields use `True`/`False`. Missing optional fields indicate that the source experiment was generated before that telemetry was added; downstream scripts derive conservative defaults only where documented.

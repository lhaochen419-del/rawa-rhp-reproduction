from __future__ import annotations

import time
from dataclasses import dataclass
from math import hypot
from typing import Dict, Optional, Set, Tuple

from .config import SimulationConfig
from .energy import EdgeMetrics


EdgeCache = Dict[Tuple[int, int], EdgeMetrics]


@dataclass
class PlannerContext:
    cfg: SimulationConfig
    edges: EdgeCache
    target_values: Dict[int, float]
    depot_idx: int = 0

    def edge(self, i: int, j: int) -> EdgeMetrics:
        return self.edges[(i, j)]


class BasePlanner:
    name = "BasePlanner"

    def reset(self, context: PlannerContext) -> None:
        self.context = context
        self._last_candidate_expansions = 0
        self._last_pruned_candidates = 0
        self._last_anytime_stop_reason = "not_applicable"

    def choose_next(self, current: int, unvisited: Set[int], battery_remaining: float) -> Optional[int]:
        raise NotImplementedError

    def diagnostics(self) -> Dict[str, object]:
        expansions = int(getattr(self, "_last_candidate_expansions", 0))
        pruned = int(getattr(self, "_last_pruned_candidates", 0))
        out = {
            "candidate_expansions": expansions,
            "pruning_ratio": float(pruned / max(1, expansions)),
            "cache_hit_rate": 1.0,
            "anytime_stop_reason": str(getattr(self, "_last_anytime_stop_reason", "not_applicable")),
            "packing_attempts": int(getattr(self, "_last_packing_attempts", 0)),
            "packing_accepts": int(getattr(self, "_last_packing_accepts", 0)),
            "packing_added_value": float(getattr(self, "_last_packing_added_value", 0.0)),
            "packing_stop_reason": str(getattr(self, "_last_packing_stop_reason", "not_applicable")),
        }
        out.update(getattr(self, "_last_missed_diagnostics", {}))
        return out

    def planned_edge_energy(self, edge: EdgeMetrics, include_photo: bool) -> float:
        return float(edge.energy_mean + (self.context.cfg.e_photo if include_photo else 0.0))

    def actual_edge_energy(self, edge: EdgeMetrics, rng, gust_multiplier: float, include_photo: bool) -> float:
        return float(edge.sampled_energy(rng, gust_multiplier, adaptive=False) + (self.context.cfg.e_photo if include_photo else 0.0))

    def _return_edge(self, j: int) -> EdgeMetrics:
        return self.context.edge(j, self.context.depot_idx)

    def _flight_edge(self, i: int, j: int) -> EdgeMetrics:
        return self.context.edge(i, j)

    def _photo(self, j: int) -> float:
        return self.context.cfg.e_photo if j != self.context.depot_idx else 0.0


class NearestNeighbor(BasePlanner):
    name = "NearestNeighbor"

    def choose_next(self, current: int, unvisited: Set[int], battery_remaining: float) -> Optional[int]:
        feasible = []
        for j in unvisited:
            e = self._flight_edge(current, j)
            if battery_remaining - e.energy_mean - self._photo(j) >= 0.0:
                feasible.append(j)
        if not feasible:
            return None
        return min(feasible, key=lambda j: self._flight_edge(current, j).distance)


class ValuePerDistance(BasePlanner):
    name = "ValuePerDistance"

    def choose_next(self, current: int, unvisited: Set[int], battery_remaining: float) -> Optional[int]:
        best = None
        best_score = -1e18
        for j in unvisited:
            e = self._flight_edge(current, j)
            if battery_remaining - e.energy_mean - self._photo(j) < 0.0:
                continue
            score = self.context.target_values[j] / max(e.distance, 1e-6)
            if score > best_score:
                best_score = score
                best = j
        return best


class WindAwareGreedy(BasePlanner):
    name = "WindAwareGreedy"

    def choose_next(self, current: int, unvisited: Set[int], battery_remaining: float) -> Optional[int]:
        best = None
        best_score = -1e18
        cfg = self.context.cfg
        for j in unvisited:
            e = self._flight_edge(current, j)
            ret = self._return_edge(j)
            if battery_remaining - e.energy_mean - self._photo(j) < 0.0:
                continue
            score = (
                self.context.target_values[j]
                - cfg.alpha * e.energy_mean
                - cfg.delta * ret.energy_mean
            )
            if score > best_score:
                best_score = score
                best = j
        return best


class ReserveOnlyPlanner(BasePlanner):
    name = "ReserveOnlyPlanner"

    def planned_edge_energy(self, edge: EdgeMetrics, include_photo: bool) -> float:
        return float(edge.energy_tail90 + (self.context.cfg.e_photo if include_photo else 0.0))

    def choose_next(self, current: int, unvisited: Set[int], battery_remaining: float) -> Optional[int]:
        best = None
        best_score = -1e18
        cfg = self.context.cfg
        for j in unvisited:
            e = self._flight_edge(current, j)
            ret = self._return_edge(j)
            required = e.energy_tail90 + self._photo(j) + ret.energy_tail90
            if battery_remaining - required < cfg.reserve_floor:
                continue
            score = (
                self.context.target_values[j]
                - cfg.alpha * e.energy_mean
                - cfg.beta * e.energy_tail90
                - cfg.delta * ret.energy_tail90
            )
            if score > best_score:
                best_score = score
                best = j
        return best


class FairRiskAwareGreedy(BasePlanner):
    name = "FairRiskAwareGreedy"

    tail_blend = 0.75
    return_tail_blend = 1.0
    reserve_buffer = 15.0
    risk_avg_cap = 0.0036
    report_tail_blend = 1.15

    def planned_edge_energy(self, edge: EdgeMetrics, include_photo: bool) -> float:
        mean = edge.energy_adaptive_mean
        budget = mean + self.report_tail_blend * (edge.energy_adaptive_tail95 - mean)
        return float(budget + (self.context.cfg.e_photo if include_photo else 0.0))

    def actual_edge_energy(self, edge: EdgeMetrics, rng, gust_multiplier: float, include_photo: bool) -> float:
        return float(edge.sampled_energy(rng, gust_multiplier, adaptive=True) + (self.context.cfg.e_photo if include_photo else 0.0))

    def choose_next(self, current: int, unvisited: Set[int], battery_remaining: float) -> Optional[int]:
        best = None
        best_score = -1e18
        cfg = self.context.cfg
        current_ret = self._return_budget(current) if current != self.context.depot_idx else 0.0
        for j in unvisited:
            edge = self._flight_edge(current, j)
            ret = self._return_edge(j)
            if edge.risk > cfg.p_max or ret.risk > cfg.p_max:
                continue
            avg_return_risk = 0.5 * (edge.risk + ret.risk)
            if avg_return_risk > self.risk_avg_cap:
                continue
            leg_budget = self._leg_budget(edge) + self._photo(j)
            return_budget = self._return_budget(j)
            required = leg_budget + return_budget
            if battery_remaining - required < cfg.reserve_floor + self.reserve_buffer:
                continue
            return_delta = max(0.0, return_budget - current_ret)
            risk_cost = 300.0 * max(0.0, avg_return_risk - 0.65 * self.risk_avg_cap) + 25.0 * avg_return_risk
            score = (
                1.20 * self.context.target_values[j]
                + self.context.target_values[j] / max(leg_budget + 0.35 * return_delta, 1.0)
                - 0.006 * required
                - risk_cost
            )
            if score > best_score:
                best_score = score
                best = j
        return best

    def _leg_budget(self, edge: EdgeMetrics) -> float:
        mean = edge.energy_adaptive_mean
        return float(mean + self.tail_blend * (edge.energy_adaptive_tail95 - mean))

    def _return_budget(self, node: int) -> float:
        edge = self._return_edge(node)
        mean = edge.energy_adaptive_mean
        return float(mean + self.return_tail_blend * (edge.energy_adaptive_tail95 - mean))



class RolloutRHP(BasePlanner):
    use_tail = True
    use_risk = True
    use_wind = True
    rollout_depth = 32
    risk_soft_limit = 0.010
    risk_avg_cap = 0.010
    tail_blend = 1.0
    return_tail_blend = 1.0
    tail_quantile = "tail90"
    reserve_buffer = 0.0
    visited_count_bonus = 0.05
    plan_value_weight = 1.0
    reserve_utilization_weight = 0.15
    energy_penalty_coeff = 0.006
    candidate_risk_weight = 180.0
    insertion_risk_guard_fraction = 0.75
    insertion_risk_weight = 28.0
    risk_avg_penalty = 160.0
    risk_max_penalty = 8.0
    use_beam_search = False
    use_insertion_repair = True
    beam_width = 48
    beam_depth = 8
    candidate_pool_size = 16
    repair_top_k = 8
    repair_max_insertions = 8
    use_route_polish = False
    use_ejection_repair = False
    use_adaptive_search_budget = False
    use_high_value_recovery = False
    use_marginal_packing = False
    packing_candidate_limit = 20
    packing_regret_weight = 0.20
    packing_unlock_weight = 0.16
    tail_packing_value_bonus = 0.0
    tail_candidate_pool_bonus = 0
    tail_repair_top_k_bonus = 0
    tail_packing_candidate_bonus = 0
    tail_replan_expansion_bonus = 0
    no_risk_reserve_pressure = 0.75
    no_tail_energy_margin = 0.0
    risk_penalty_gate_fraction = 0.80
    enforce_risk_edge_cap = True
    enforce_route_risk_cap = True
    use_risk_candidate_bias = True
    use_risk_scoring = True
    use_risk_sorting = True
    use_risk_pricing = True
    polish_passes = 2
    polish_node_limit = 26
    polish_risk_trigger = 0.0020
    use_adaptive_energy = False
    report_tail_blend = 1.0
    max_replan_expansions = 60000
    max_replan_seconds = 0.0
    recovery_reserve_credit = 0.0
    tail_reserve_credit = 0.0
    tail_low_pressure_reserve_credit_bonus = 0.0
    tail_value_bonus = 0.0
    use_tail_high_value_recovery = False
    tail_recovery_plan_limit = 6
    use_no_tail_shadow_candidates = False
    use_no_tail_shadow_value_fallback = False
    no_tail_shadow_fallback_value_gain = 0.5
    no_tail_shadow_fallback_score_tolerance = 0.0
    no_tail_shadow_fallback_tail_threshold = 0.0
    no_tail_shadow_fallback_risk_threshold = 0.0
    use_tail_route_persistence = False
    tail_route_persistence_bonus = 0.0
    use_tail_low_pressure_completion = False
    tail_completion_bonus = 0.0
    tail_value_override_tolerance = 0.0
    tail_value_override_gain_threshold = 0.75
    tail_value_override_tail_threshold = 0.070
    tail_value_override_risk_threshold = 0.090
    use_tail_low_pressure_q90_fallback = False
    tail_q90_fallback_tail_threshold = 0.0
    tail_q90_fallback_risk_threshold = 0.0
    tail_q90_fallback_detour_threshold = 0.0
    tail_budget_cap_quantile: str | None = None
    tail_budget_cap_blend = 1.0
    blind_reserve_pressure: float | None = None
    fixed_replan_budget = False
    min_unified_eval_count = 0
    eqeval_probe_guard_multiplier = 32

    def reset(self, context: PlannerContext) -> None:
        super().reset(context)
        self._energy_residual_ema = 0.0
        self._last_best_route: list[int] = []
        self._edge_budget_cache: Dict[Tuple[int, int, bool], float] = {}
        self._return_budget_cache: Dict[int, float] = {}
        self._route_stats_cache: Dict[Tuple[int, Tuple[int, ...]], Dict[str, object]] = {}
        self._last_missed_diagnostics: Dict[str, object] = {}
        self._last_packing_attempts = 0
        self._last_packing_accepts = 0
        self._last_packing_added_value = 0.0
        self._last_packing_stop_reason = "not_applicable"
        self._packing_stop_reasons: list[str] = []
        self._configure_search_budget()
        self._adaptive_budget_base = (
            self.beam_width,
            self.beam_depth,
            self.candidate_pool_size,
            self.repair_top_k,
            self.repair_max_insertions,
            self.packing_candidate_limit,
            self.max_replan_expansions,
        )

    def _configure_search_budget(self) -> None:
        if not (self.use_adaptive_search_budget and self.use_beam_search):
            return
        risks = [edge.risk for edge in self.context.edges.values()]
        if not risks:
            return
        ordered = sorted(float(risk) for risk in risks)
        risk_mean = sum(ordered) / len(ordered)
        risk_p75 = ordered[int(0.75 * (len(ordered) - 1))]
        risk_p90 = ordered[int(0.90 * (len(ordered) - 1))]
        self._adaptive_risk_mean = float(risk_mean)
        self._adaptive_risk_p75 = float(risk_p75)
        self._adaptive_risk_p90 = float(risk_p90)
        tail_spreads = sorted(
            max(0.0, (edge.energy_adaptive_tail95 - edge.energy_adaptive_mean) / max(edge.no_wind_energy, 1.0))
            for edge in self.context.edges.values()
        )
        self._adaptive_tail_p90 = float(tail_spreads[int(0.90 * (len(tail_spreads) - 1))]) if tail_spreads else 0.0
        detour_ratios = sorted(
            float(edge.distance) / max(self._straight_line_distance(edge), 1e-6)
            for edge in self.context.edges.values()
            if edge.source != edge.target
        )
        self._adaptive_detour_p95 = float(detour_ratios[int(0.95 * (len(detour_ratios) - 1))]) if detour_ratios else 1.0
        if risk_mean < 0.028 and risk_p75 < 0.040 and risk_p90 < 0.080:
            self.beam_width = 240
            self.beam_depth = 10
            self.candidate_pool_size = 34
            self.repair_top_k = 18
            self.repair_max_insertions = 22
            self.packing_candidate_limit = 18
            self.max_replan_expansions = 30000
        elif risk_mean > 0.045 or risk_p75 > 0.065 or risk_p90 > 0.110:
            self.beam_width = 448
            self.beam_depth = 12
            self.candidate_pool_size = 56
            self.repair_top_k = 38
            self.repair_max_insertions = 34
            self.packing_candidate_limit = 30
            self.max_replan_expansions = 70000
        else:
            self.beam_width = 352
            self.beam_depth = 12
            self.candidate_pool_size = 46
            self.repair_top_k = 30
            self.repair_max_insertions = 30
            self.packing_candidate_limit = 24
            self.max_replan_expansions = 46000

    def _configure_replan_budget(self, unvisited: Set[int], battery_remaining: float) -> None:
        if not (self.use_adaptive_search_budget and self.use_beam_search):
            return
        if self.fixed_replan_budget:
            return
        base = getattr(self, "_adaptive_budget_base", None)
        if base is None:
            return
        (
            beam_width,
            beam_depth,
            candidate_pool_size,
            repair_top_k,
            repair_max_insertions,
            packing_candidate_limit,
            max_replan_expansions,
        ) = base
        cfg = self.context.cfg
        remaining_pressure = min(1.0, len(unvisited) / max(1.0, float(cfg.n_targets)))
        battery_pressure = max(0.0, 1.0 - battery_remaining / max(cfg.battery_capacity, 1.0))
        risk_mean = float(getattr(self, "_adaptive_risk_mean", 0.0))
        risk_p75 = float(getattr(self, "_adaptive_risk_p75", 0.0))
        risk_p90 = float(getattr(self, "_adaptive_risk_p90", 0.0))
        env_pressure = min(
            1.0,
            max(
                risk_mean / 0.045 if risk_mean > 0.0 else 0.0,
                risk_p75 / 0.065 if risk_p75 > 0.0 else 0.0,
                risk_p90 / 0.110 if risk_p90 > 0.0 else 0.0,
            ),
        )
        opportunity = remaining_pressure * max(0.0, 1.0 - 0.65 * battery_pressure)
        if env_pressure >= 0.82 or opportunity >= 0.50:
            beam_width += int(48 + 72 * env_pressure)
            candidate_pool_size += int(6 + 10 * opportunity)
            repair_top_k += int(6 + 10 * max(env_pressure, opportunity))
            repair_max_insertions += int(4 + 8 * opportunity)
            packing_candidate_limit += int(8 + 14 * opportunity)
            max_replan_expansions += int(10000 + 16000 * max(env_pressure, opportunity))
        elif battery_pressure <= 0.35 and remaining_pressure >= 0.35:
            candidate_pool_size += 4
            repair_top_k += 4
            repair_max_insertions += 3
            packing_candidate_limit += 6
            max_replan_expansions += 6000
        self.beam_width = int(min(560, max(1, beam_width)))
        self.beam_depth = int(beam_depth)
        self.candidate_pool_size = int(min(72, max(4, candidate_pool_size)))
        self.repair_top_k = int(min(56, max(1, repair_top_k)))
        self.repair_max_insertions = int(min(48, max(1, repair_max_insertions)))
        self.packing_candidate_limit = int(min(48, max(1, packing_candidate_limit)))
        self.max_replan_expansions = int(min(86000, max(1, max_replan_expansions)))
        if self._uses_tail_model():
            self.candidate_pool_size = int(min(72, self.candidate_pool_size + int(self.tail_candidate_pool_bonus)))
            self.repair_top_k = int(min(56, self.repair_top_k + int(self.tail_repair_top_k_bonus)))
            self.packing_candidate_limit = int(min(48, self.packing_candidate_limit + int(self.tail_packing_candidate_bonus)))
            self.max_replan_expansions = int(min(86000, self.max_replan_expansions + int(self.tail_replan_expansion_bonus)))

    def choose_next(self, current: int, unvisited: Set[int], battery_remaining: float) -> Optional[int]:
        self._battery_for_scoring = float(battery_remaining)
        self._edge_budget_cache = {}
        self._return_budget_cache = {}
        self._route_stats_cache = {}
        self._replan_deadline = (time.perf_counter() + self.max_replan_seconds) if self.max_replan_seconds > 0.0 else None
        self._last_candidate_expansions = 0
        self._last_pruned_candidates = 0
        self._last_anytime_stop_reason = "complete"
        self._last_packing_attempts = 0
        self._last_packing_accepts = 0
        self._last_packing_added_value = 0.0
        self._last_packing_stop_reason = "disabled" if not self.use_marginal_packing else "not_started"
        self._packing_stop_reasons = []
        self._configure_replan_budget(unvisited, battery_remaining)
        if self.use_beam_search:
            chosen = self._beam_choose_next(current, unvisited, battery_remaining)
            self._run_eqeval_budget_probe(current, unvisited, battery_remaining)
            return chosen
        best = None
        best_score = -1e18
        for j in unvisited:
            score = self._rollout_score(current, j, unvisited, battery_remaining)
            if score is None:
                continue
            if score > best_score:
                best_score = score
                best = j
        self._run_eqeval_budget_probe(current, unvisited, battery_remaining)
        return best

    def _run_eqeval_budget_probe(self, current: int, unvisited: Set[int], battery_remaining: float) -> None:
        target = int(getattr(self, "min_unified_eval_count", 0))
        if target <= 0 or not unvisited:
            return
        remaining = set(unvisited)
        ordered = sorted(remaining, key=lambda n: (self.context.target_values[n], -n), reverse=True)
        if not ordered:
            return
        cursors = [current] + ordered
        guard = 0
        guard_limit = max(1, int(getattr(self, "eqeval_probe_guard_multiplier", 32)) * len(ordered))
        while self._current_unified_eval_count() < target and guard < guard_limit:
            cursor = cursors[guard % len(cursors)]
            pool = self._candidate_pool(cursor, remaining - {cursor}, battery_remaining, 0.0)
            if not pool:
                pool = ordered
            for node in pool:
                if self._current_unified_eval_count() >= target:
                    break
                if node == cursor:
                    continue
                edge = self._flight_edge(cursor, node)
                _ = self._leg_budget(edge) + self._photo(node) + self._return_budget(node)
                if self.use_risk:
                    _ = edge.risk + self._return_edge(node).risk
                self._last_candidate_expansions += 1
            if self.use_marginal_packing and self._current_unified_eval_count() < target:
                prefix_len = min(len(ordered), 1 + (guard % max(1, min(len(ordered), 8))))
                route = [node for node in ordered[:prefix_len] if node in remaining]
                pack_nodes = [node for node in self._packing_candidate_nodes(remaining - set(route)) if node not in route]
                for node in pack_nodes:
                    if self._current_unified_eval_count() >= target:
                        break
                    for pos in range(len(route) + 1):
                        if self._current_unified_eval_count() >= target:
                            break
                        self._last_packing_attempts += 1
                        self._insertion_delta_metrics(current, route, pos, node)
            guard += 1

    def _current_unified_eval_count(self) -> int:
        return int(2 * int(getattr(self, "_last_candidate_expansions", 0)) + int(getattr(self, "_last_packing_attempts", 0)))

    def _beam_choose_next(self, current: int, unvisited: Set[int], battery_remaining: float) -> Optional[int]:
        if not unvisited:
            return None
        incumbent_plan = self._fast_greedy_incumbent(current, set(unvisited), battery_remaining)
        initial = (tuple(), current, 0.0, 0.0, 0.0, 0.0, 0)
        beam = [initial]
        completed = [initial]
        for _ in range(self.beam_depth):
            next_states = []
            for route, cursor, travel_cost, value, risk_sum, max_risk, risk_edges in beam:
                remaining = set(unvisited) - set(route)
                candidates = self._candidate_pool(cursor, remaining, battery_remaining, travel_cost)
                for node in candidates:
                    if self._time_budget_exhausted():
                        break
                    self._last_candidate_expansions += 1
                    edge = self._flight_edge(cursor, node)
                    leg_cost = self._leg_budget(edge) + self._photo(node)
                    new_route = route + (node,)
                    new_risk_sum = risk_sum + (edge.risk if self.use_risk else 0.0)
                    new_max_risk = max(max_risk, edge.risk) if self.use_risk else 0.0
                    new_risk_edges = risk_edges + (1 if self.use_risk else 0)
                    next_states.append(
                        (
                            new_route,
                            node,
                            travel_cost + leg_cost,
                            value + self.context.target_values[node],
                            new_risk_sum,
                            new_max_risk,
                            new_risk_edges,
                        )
                    )
                    if self._last_candidate_expansions >= self.max_replan_expansions:
                        self._last_anytime_stop_reason = "expansion_budget"
                        break
                if self._last_anytime_stop_reason == "expansion_budget":
                    break
                if self._last_anytime_stop_reason == "time_budget":
                    break
            if self._last_anytime_stop_reason == "expansion_budget":
                completed.extend(beam)
                break
            if self._last_anytime_stop_reason == "time_budget":
                completed.extend(beam)
                break
            if not next_states:
                self._last_anytime_stop_reason = "no_feasible_extension"
                break
            before_prune = len(next_states)
            next_states = self._prune_dominated_states(next_states)
            self._last_pruned_candidates += before_prune - len(next_states)
            next_states.sort(key=lambda s: self._beam_state_score(s, battery_remaining), reverse=True)
            beam = next_states[: self.beam_width]
            self._last_pruned_candidates += max(0, len(next_states) - len(beam))
            completed.extend(beam)

        best_route: Optional[list[int]] = None
        best_score = -1e18
        best_value = -1e18
        best_shadow_route: Optional[list[int]] = None
        best_shadow_value = -1e18
        best_shadow_score = -1e18
        candidate_plans = []
        if incumbent_plan["route"]:
            candidate_plans.append(incumbent_plan)
        if self.use_marginal_packing:
            if incumbent_plan["route"]:
                candidate_plans.append(
                    self._risk_reserve_marginal_packing_repair(
                        current,
                        list(incumbent_plan["route"]),
                        set(unvisited) - set(incumbent_plan["route"]),
                        battery_remaining,
                    )
                )
            candidate_plans.append(self._risk_reserve_marginal_packing_repair(current, [], set(unvisited), battery_remaining))
            if self.use_insertion_repair and not self._time_budget_exhausted():
                candidate_plans.append(self._greedy_insertion_plan(current, set(unvisited), battery_remaining))
        elif self.use_insertion_repair:
            candidate_plans.append(self._greedy_insertion_plan(current, set(unvisited), battery_remaining))
        seeds = sorted(completed, key=lambda s: self._beam_state_score(s, battery_remaining), reverse=True)[: self.repair_top_k]
        for state in seeds:
            if self._time_budget_exhausted():
                break
            route = list(state[0])
            if not route:
                continue
            if self.use_marginal_packing:
                remaining = set(unvisited) - set(route)
                candidate_plans.append(self._risk_reserve_marginal_packing_repair(current, route, remaining, battery_remaining))
                if self.use_insertion_repair and not self._time_budget_exhausted():
                    candidate_plans.append(self._improve_route_by_insertion(current, route, remaining, battery_remaining))
            elif self.use_insertion_repair:
                remaining = set(unvisited) - set(route)
                candidate_plans.append(self._improve_route_by_insertion(current, route, remaining, battery_remaining))
            else:
                candidate_plans.append(self._route_stats(current, route))
        if self._enable_tail_route_persistence():
            candidate_plans.extend(self._tail_persistent_route_candidates(current, set(unvisited), battery_remaining))
        if self._enable_tail_low_pressure_completion() and not self._time_budget_exhausted():
            candidate_plans.extend(self._tail_low_pressure_completion_candidates(current, set(unvisited), battery_remaining))
        if self.use_high_value_recovery or self._enable_tail_high_value_recovery():
            candidate_plans.extend(
                self._high_value_recovery_candidates(current, candidate_plans, set(unvisited), battery_remaining)
            )
        if self._enable_no_tail_shadow_candidates() and not self._time_budget_exhausted():
            candidate_plans.extend(self._no_tail_shadow_candidate_plans(current, set(unvisited), battery_remaining))
        for plan in candidate_plans:
            route = plan["route"]
            if not route:
                continue
            score = self._score_plan(plan, battery_remaining) + float(plan.get("route_commit_bonus", 0.0))
            plan_value = float(plan.get("value", 0.0))
            if bool(plan.get("shadow_no_tail_candidate", False)) and self._enable_no_tail_shadow_value_fallback():
                if plan_value > best_shadow_value and self._plan_hard_feasible(plan, battery_remaining):
                    best_shadow_value = plan_value
                    best_shadow_score = score
                    best_shadow_route = list(route)
                continue
            value_override = (
                self._enable_tail_value_override()
                and plan_value > best_value + float(self.tail_value_override_gain_threshold)
                and score >= best_score - self.tail_value_override_tolerance
            )
            if score > best_score or value_override:
                best_score = score
                best_value = plan_value
                best_route = list(route)
        if (
            best_shadow_route
            and best_shadow_value > best_value + float(self.no_tail_shadow_fallback_value_gain)
            and best_shadow_score >= best_score - float(self.no_tail_shadow_fallback_score_tolerance)
        ):
            best_route = list(best_shadow_route)
        if not best_route:
            self._last_best_route = []
            if self._last_anytime_stop_reason == "complete":
                self._last_anytime_stop_reason = "no_feasible_plan"
            self._record_missed_diagnostics(current, set(unvisited), battery_remaining)
            self._finalize_packing_stop_reason()
            return None
        self._last_best_route = list(best_route)
        self._finalize_packing_stop_reason()
        return best_route[0]

    def _candidate_pool(self, cursor: int, remaining: Set[int], battery_remaining: float, travel_cost: float) -> list[int]:
        feasible = []
        current_ret = self._return_budget(cursor) if cursor != self.context.depot_idx else 0.0
        for node in remaining:
            self._last_candidate_expansions += 1
            edge = self._flight_edge(cursor, node)
            if self.use_risk and self.enforce_risk_edge_cap and edge.risk > self.context.cfg.p_max:
                self._last_pruned_candidates += 1
                continue
            leg_cost = self._leg_budget(edge) + self._photo(node)
            return_budget = self._return_budget(node)
            return_edge = self._return_edge(node)
            total_required = travel_cost + leg_cost + return_budget
            reserve_req = self._reserve_requirement(battery_remaining, return_budget, edge.risk)
            if battery_remaining - total_required < reserve_req:
                self._last_pruned_candidates += 1
                continue
            return_delta = max(0.0, return_budget - current_ret)
            detour = edge.distance / max(self._straight_line_distance(edge), 1e-6)
            feasible.append((node, leg_cost, return_delta, edge.distance, edge.risk, self.context.target_values[node], return_edge.risk, detour))
        if not feasible:
            return []
        selected: list[int] = []
        seen = set()

        def add_sorted(rows, key, reverse=True, quota: Optional[int] = None) -> None:
            added = 0
            for row in sorted(rows, key=key, reverse=reverse):
                node = row[0]
                if node not in seen:
                    selected.append(node)
                    seen.add(node)
                    added += 1
                if quota is not None and added >= quota:
                    return
                if len(selected) >= self.candidate_pool_size:
                    return

        quota = max(3, self.candidate_pool_size // 5)
        add_sorted(feasible, key=lambda r: r[5] / max(r[1], 1.0), reverse=True, quota=quota + 1)
        add_sorted(feasible, key=lambda r: r[5] / max(r[1] + 0.35 * r[2], 1.0), reverse=True, quota=quota + 1)
        if self._enable_tail_high_value_recovery():
            add_sorted(feasible, key=lambda r: (r[5], -(r[1] + 0.20 * r[2]), -r[7]), reverse=True, quota=quota + 1)
        if self.use_adaptive_search_budget:
            if self.use_risk and self.use_risk_sorting:
                adaptive_value_key = lambda r: r[5] / max(r[1] + 0.70 * r[2] + 28.0 * r[4] + 44.0 * r[6], 1.0)
            else:
                adaptive_value_key = lambda r: r[5] / max(r[1] + 0.70 * r[2], 1.0)
            add_sorted(
                feasible,
                key=adaptive_value_key,
                reverse=True,
                quota=quota + 2,
            )
            add_sorted(feasible, key=lambda r: (r[7], r[1] + 0.60 * r[2], -r[5]), reverse=False, quota=quota)
            if self.use_risk and self.use_risk_sorting:
                add_sorted(feasible, key=lambda r: (r[6] + 0.55 * r[4], -r[5], r[1]), reverse=False, quota=max(2, quota - 1))
        if self.use_risk and self.use_risk_candidate_bias and self.use_risk_sorting:
            add_sorted(feasible, key=lambda r: r[5] / max(r[1] + 0.65 * r[2] + 120.0 * r[6], 1.0), reverse=True, quota=quota)
            add_sorted(feasible, key=lambda r: (r[7], r[2] + 120.0 * (r[4] + r[6]), -r[5]), reverse=False, quota=max(2, quota - 1))
            add_sorted(feasible, key=lambda r: r[4], reverse=False, quota=quota)
        add_sorted(feasible, key=lambda r: r[3], reverse=False, quota=quota)
        add_sorted(feasible, key=lambda r: r[5], reverse=True, quota=quota + 1)
        if len(selected) < self.candidate_pool_size:
            add_sorted(
                feasible,
                key=lambda r: r[5] / max(r[1] + 0.25 * r[2] + (self.candidate_risk_weight * (r[4] + 0.45 * r[6]) if self.use_risk else 0.0), 1.0),
                reverse=True,
                quota=None,
            )
        return selected[: self.candidate_pool_size]

    def _straight_line_distance(self, edge: EdgeMetrics) -> float:
        if not edge.path:
            return max(edge.distance, 1e-6)
        x0, y0 = edge.path[0]
        x1, y1 = edge.path[-1]
        return hypot(x1 - x0, y1 - y0)

    def _fast_greedy_incumbent(self, start: int, remaining: Set[int], battery_remaining: float) -> Dict[str, object]:
        route: list[int] = []
        cursor = start
        travel_cost = 0.0
        remaining = set(remaining)
        for _ in range(min(12, len(remaining))):
            candidates = self._candidate_pool(cursor, remaining, battery_remaining, travel_cost)
            if not candidates:
                break
            best_node = None
            best_score = -1e18
            for node in candidates[: min(len(candidates), 10)]:
                edge = self._flight_edge(cursor, node)
                leg_cost = self._leg_budget(edge) + self._photo(node)
                trial = route + [node]
                plan = self._route_stats(start, trial)
                if not self._plan_hard_feasible(plan, battery_remaining):
                    continue
                return_delta = max(0.0, self._return_budget(node) - (self._return_budget(cursor) if cursor != self.context.depot_idx else 0.0))
                score = self._score_plan(plan, battery_remaining) + self.context.target_values[node] / max(leg_cost + 0.45 * return_delta, 1.0)
                if score > best_score:
                    best_score = score
                    best_node = node
            if best_node is None:
                break
            route.append(best_node)
            remaining.remove(best_node)
            edge = self._flight_edge(cursor, best_node)
            travel_cost += self._leg_budget(edge) + self._photo(best_node)
            cursor = best_node
        return self._route_stats(start, route)

    def _high_value_skeleton_plan(self, start: int, remaining: Set[int], battery_remaining: float) -> Dict[str, object]:
        route: list[int] = []
        remaining = set(remaining)
        ranked = sorted(remaining, key=lambda n: (self.context.target_values[n], -self._return_budget(n)), reverse=True)
        for node in ranked[: min(16, len(ranked))]:
            if self._time_budget_exhausted():
                break
            best_pos = None
            best_delta = float("inf")
            for pos in range(len(route) + 1):
                delta = self._insertion_delta_metrics(start, route, pos, node)
                if delta is None:
                    continue
                delta_total, delta_travel, delta_risk, edge_max_risk, risk_edge_delta = delta
                base = self._route_stats(start, route)
                total = float(base["total_cost"]) + delta_total
                travel = float(base["travel_cost"]) + delta_travel
                risk_sum = float(base["risk_sum"]) + delta_risk
                max_risk = max(float(base["max_risk"]), edge_max_risk)
                risk_edges = int(base["risk_edges"]) + risk_edge_delta
                if not self._plan_values_hard_feasible(total, travel, risk_sum, max_risk, risk_edges, battery_remaining):
                    continue
                if delta_total < best_delta:
                    best_delta = delta_total
                    best_pos = pos
            if best_pos is not None:
                route.insert(best_pos, node)
                remaining.remove(node)
        return self._route_stats(start, route)

    def _recover_high_value_targets(
        self,
        start: int,
        route: list[int],
        remaining: Set[int],
        battery_remaining: float,
    ) -> list[int]:
        route = list(route)
        remaining = set(remaining)
        if not route or not remaining:
            return route
        high_value_nodes = sorted(remaining, key=lambda n: (self.context.target_values[n], -n), reverse=True)[: min(10, len(remaining))]
        for _ in range(min(4, len(high_value_nodes))):
            if self._time_budget_exhausted():
                break
            base_plan = self._route_stats(start, route)
            base_value = float(base_plan["value"])
            base_travel = float(base_plan["travel_cost"])
            base_total = float(base_plan["total_cost"])
            base_risk = float(base_plan["risk_sum"])
            base_max_risk = float(base_plan["max_risk"])
            base_risk_edges = int(base_plan["risk_edges"])
            best_node = None
            best_pos = 0
            best_gain = 0.0
            for node in list(high_value_nodes):
                if node not in remaining:
                    continue
                node_value = self.context.target_values[node]
                if node_value < 2.5:
                    continue
                for pos in range(len(route) + 1):
                    if self._time_budget_exhausted():
                        break
                    delta = self._insertion_delta_metrics(start, route, pos, node)
                    if delta is None:
                        continue
                    delta_total, delta_travel, delta_risk, edge_max_risk, risk_edge_delta = delta
                    value = base_value + node_value
                    travel = base_travel + delta_travel
                    total = base_total + delta_total
                    risk_sum = base_risk + delta_risk
                    risk_edges = base_risk_edges + risk_edge_delta
                    max_risk = max(base_max_risk, edge_max_risk)
                    if not self._recovery_values_feasible(total, travel, risk_sum, max_risk, risk_edges, battery_remaining):
                        continue
                    avg_risk = risk_sum / max(1, risk_edges)
                    risk_excess = max(0.0, avg_risk - self.risk_avg_cap) if self.use_risk_scoring else 0.0
                    gain = node_value - 0.0035 * max(0.0, delta_total) - 220.0 * risk_excess
                    if gain > best_gain + 1e-9:
                        best_gain = gain
                        best_node = node
                        best_pos = pos
            if best_node is None:
                break
            route.insert(best_pos, best_node)
            remaining.remove(best_node)
            high_value_nodes = [node for node in high_value_nodes if node != best_node]
        return route

    def _prune_dominated_states(self, states: list[tuple]) -> list[tuple]:
        if len(states) <= 1:
            return states
        kept: list[tuple] = []
        buckets: Dict[Tuple[int, int], list[tuple]] = {}
        for state in states:
            route, cursor, travel_cost, value, risk_sum, max_risk, risk_edges = state
            buckets.setdefault((cursor, len(route)), []).append(state)
        for bucket in buckets.values():
            nondominated: list[tuple] = []
            for state in bucket:
                _, _, travel_cost, value, risk_sum, max_risk, _ = state
                dominated = False
                for other in bucket:
                    if other is state:
                        continue
                    _, _, ocost, ovalue, orisk, omax, _ = other
                    if (
                        ovalue >= value
                        and ocost <= travel_cost
                        and orisk <= risk_sum
                        and omax <= max_risk
                        and (ovalue > value or ocost < travel_cost or orisk < risk_sum or omax < max_risk)
                    ):
                        dominated = True
                        break
                if not dominated:
                    nondominated.append(state)
            kept.extend(nondominated)
        return kept

    def _beam_state_score(self, state: tuple, battery_remaining: float) -> float:
        route, cursor, travel_cost, value, risk_sum, max_risk, risk_edges = state
        return_cost = self._return_budget(cursor) if cursor != self.context.depot_idx else 0.0
        if self.use_risk and cursor != self.context.depot_idx:
            ret = self._return_edge(cursor)
            risk_sum += ret.risk
            max_risk = max(max_risk, ret.risk)
            risk_edges += 1
        plan = {
            "route": list(route),
            "value": value,
            "travel_cost": travel_cost,
            "total_cost": travel_cost + return_cost,
            "risk_sum": risk_sum,
            "max_risk": max_risk,
            "risk_edges": risk_edges,
        }
        return self._score_plan(plan, battery_remaining)

    def _rollout_score(
        self,
        current: int,
        first: int,
        unvisited: Set[int],
        battery_remaining: float,
    ) -> Optional[float]:
        if not self._is_feasible(current, first, battery_remaining):
            return None

        cfg = self.context.cfg
        battery = battery_remaining
        cursor = current
        remaining = set(unvisited)
        total_value = 0.0
        planned_travel = 0.0
        risk_sum = 0.0
        max_risk = 0.0
        steps = 0

        chosen = first
        while chosen is not None and steps < self.rollout_depth:
            edge = self._flight_edge(cursor, chosen)
            leg_cost = self._leg_budget(edge) + self._photo(chosen)
            battery -= leg_cost
            planned_travel += leg_cost
            total_value += self.context.target_values[chosen]
            if self.use_risk:
                risk_sum += edge.risk
                max_risk = max(max_risk, edge.risk)
            remaining.remove(chosen)
            cursor = chosen
            steps += 1
            chosen = self._best_rollout_extension(cursor, remaining, battery)

        return_budget = self._return_budget(cursor) if cursor != self.context.depot_idx else 0.0
        final_slack = battery - return_budget - cfg.reserve_floor
        avg_risk = risk_sum / max(1, steps)
        risk_penalty = 0.0
        if self.use_risk and self.use_risk_scoring:
            risk_penalty = 12.0 * max(0.0, avg_risk - self.risk_soft_limit) + 2.0 * max_risk
        energy_penalty = 0.015 * (planned_travel / max(cfg.battery_capacity, 1.0))
        slack_bonus = 0.06 * max(-1.0, min(1.0, final_slack / max(cfg.reserve_floor, 1.0)))
        depth_bonus = 0.015 * steps
        return total_value - risk_penalty - energy_penalty + slack_bonus + depth_bonus

    def _best_rollout_extension(self, current: int, remaining: Set[int], battery_remaining: float) -> Optional[int]:
        best = None
        best_score = -1e18
        for j in remaining:
            if not self._is_feasible(current, j, battery_remaining):
                continue
            edge = self._flight_edge(current, j)
            ret = self._return_edge(j)
            current_ret = self._return_budget(current) if current != self.context.depot_idx else 0.0
            marginal = self._leg_budget(edge) + self._photo(j) + 0.35 * max(0.0, self._return_budget(j) - current_ret)
            risk = edge.risk if self.use_risk else 0.0
            risk_cost = 0.0
            if self.use_risk and self.use_risk_scoring:
                risk_cost = 35.0 * max(0.0, risk - self.risk_soft_limit) + 3.0 * risk
            score = self.context.target_values[j] / max(marginal, 1.0) - risk_cost / 100.0 - 0.0001 * ret.distance
            if score > best_score:
                best_score = score
                best = j
        return best

    def _risk_reserve_marginal_packing_repair(
        self,
        start: int,
        route: list[int],
        remaining: Set[int],
        battery_remaining: float,
    ) -> Dict[str, object]:
        route = list(route)
        remaining = set(remaining) - set(route)
        stop_reason = "complete"
        if self.use_route_polish and self._should_polish_route(start, route):
            route = self._polish_route(start, route, battery_remaining)
            remaining -= set(route)
        if not remaining:
            self._record_packing_stop("no_candidates")
            return self._route_stats(start, route)

        max_insertions = min(self.repair_max_insertions, len(remaining))
        for _ in range(max_insertions):
            if self._time_budget_exhausted():
                stop_reason = "time_budget"
                break
            base_plan = self._route_stats(start, route)
            if route and not self._plan_hard_feasible(base_plan, battery_remaining):
                stop_reason = "base_infeasible"
                break
            base_value = float(base_plan["value"])
            base_travel = float(base_plan["travel_cost"])
            base_total = float(base_plan["total_cost"])
            base_risk = float(base_plan["risk_sum"])
            base_max_risk = float(base_plan["max_risk"])
            base_risk_edges = int(base_plan["risk_edges"])
            energy_shadow, risk_shadow = self._packing_shadow_prices(base_plan, battery_remaining)
            candidates = self._packing_candidate_nodes(remaining)
            if not candidates:
                stop_reason = "no_candidates"
                break

            best_node = None
            best_pos = 0
            best_gain = 0.0
            for node in candidates:
                node_value = self.context.target_values[node]
                regret_bonus = self._packing_regret_bonus(node, candidates)
                for pos in range(len(route) + 1):
                    if self._time_budget_exhausted():
                        stop_reason = "time_budget"
                        break
                    self._last_packing_attempts += 1
                    delta = self._insertion_delta_metrics(start, route, pos, node)
                    if delta is None:
                        continue
                    delta_total, delta_travel, delta_risk, edge_max_risk, risk_edge_delta = delta
                    travel = base_travel + delta_travel
                    total = base_total + delta_total
                    risk_sum = base_risk + delta_risk
                    risk_edges = base_risk_edges + risk_edge_delta
                    max_risk = max(base_max_risk, edge_max_risk)
                    if not self._plan_values_hard_feasible(total, travel, risk_sum, max_risk, risk_edges, battery_remaining):
                        continue
                    avg_risk = risk_sum / max(1, risk_edges)
                    return_cost = max(0.0, total - travel)
                    reserve_req = self._reserve_requirement(battery_remaining, return_cost, avg_risk)
                    usable_slack = max(0.0, battery_remaining - total - reserve_req)
                    slack_bonus = self.packing_unlock_weight * min(1.0, usable_slack / max(self.context.cfg.reserve_floor, 1.0)) * node_value
                    gain = (
                        node_value
                        + regret_bonus
                        + slack_bonus
                        + self._tail_packing_value_bonus(node_value)
                        - energy_shadow * max(0.0, delta_total)
                        - risk_shadow * max(0.0, delta_risk)
                    )
                    if gain > best_gain + 1e-9:
                        best_gain = gain
                        best_node = node
                        best_pos = pos
                if stop_reason == "time_budget":
                    break
            if best_node is None:
                if stop_reason != "time_budget":
                    stop_reason = "no_positive"
                break
            route.insert(best_pos, best_node)
            remaining.remove(best_node)
            self._last_packing_accepts += 1
            self._last_packing_added_value += float(self.context.target_values[best_node])
            if self.use_route_polish and self._should_polish_route(start, route):
                route = self._polish_route(start, route, battery_remaining)
                remaining -= set(route)
            if not remaining:
                stop_reason = "no_candidates"
                break
        else:
            stop_reason = "insertion_limit" if remaining else "no_candidates"

        self._record_packing_stop(stop_reason)
        return self._route_stats(start, route)

    def _packing_candidate_nodes(self, remaining: Set[int]) -> list[int]:
        limit = max(1, int(self.packing_candidate_limit))
        if not self.use_adaptive_search_budget:
            return sorted(remaining, key=lambda n: (self.context.target_values[n], -n), reverse=True)[: min(limit, len(remaining))]
        selected: list[int] = []
        seen: set[int] = set()

        def add_ranked(nodes: list[int], quota: int) -> None:
            added = 0
            for node in nodes:
                if node in seen:
                    continue
                selected.append(node)
                seen.add(node)
                added += 1
                if len(selected) >= limit or added >= quota:
                    return

        values = self.context.target_values
        by_value = sorted(remaining, key=lambda n: (values[n], -n), reverse=True)
        add_ranked(by_value, max(4, limit // 3))
        by_return_efficiency = sorted(
            remaining,
            key=lambda n: (values[n] / max(self._return_budget(n) + self._photo(n), 1.0), values[n]),
            reverse=True,
        )
        add_ranked(by_return_efficiency, max(4, limit // 3))
        if self.use_risk and self.use_risk_sorting:
            by_risk_reserve = sorted(
                remaining,
                key=lambda n: (
                    values[n] / max(self._return_budget(n) + 80.0 * self._return_edge(n).risk + self._photo(n), 1.0),
                    values[n],
                ),
                reverse=True,
            )
            add_ranked(by_risk_reserve, max(4, limit // 4))
            by_low_risk_value = sorted(remaining, key=lambda n: (self._return_edge(n).risk, -values[n], n))
            add_ranked(by_low_risk_value, max(2, limit // 5))
        if len(selected) < limit:
            add_ranked(by_value, limit)
        return selected[: min(limit, len(remaining))]

    def _packing_shadow_prices(self, base_plan: Dict[str, object], battery_remaining: float) -> Tuple[float, float]:
        cfg = self.context.cfg
        travel_cost = float(base_plan["travel_cost"])
        total_cost = float(base_plan["total_cost"])
        risk_sum = float(base_plan["risk_sum"])
        max_risk = float(base_plan["max_risk"])
        risk_edges = int(base_plan["risk_edges"])
        avg_risk = risk_sum / max(1, risk_edges)
        return_cost = max(0.0, total_cost - travel_cost)
        reserve_req = self._reserve_requirement(battery_remaining, return_cost, avg_risk)
        reserve_slack = battery_remaining - total_cost - reserve_req
        reserve_pressure = 1.0 - max(0.0, min(1.0, reserve_slack / max(2.0 * cfg.reserve_floor, 1.0)))
        energy_shadow = self.energy_penalty_coeff + 0.0015 + 0.010 * reserve_pressure
        if not self.use_risk or not self.use_risk_pricing:
            return float(energy_shadow), 0.0
        avg_pressure = max(0.0, min(1.0, avg_risk / max(self.risk_avg_cap, 1e-9)))
        max_pressure = max(0.0, min(1.0, max_risk / max(cfg.p_max, 1e-9)))
        gate = max(0.0, avg_pressure - self.risk_penalty_gate_fraction) / max(1e-9, 1.0 - self.risk_penalty_gate_fraction)
        risk_shadow = 8.0 + 48.0 * gate + 18.0 * max(0.0, max_pressure - 0.70) + 22.0 * reserve_pressure
        return float(energy_shadow), float(risk_shadow)

    def _packing_regret_bonus(self, node: int, candidates: list[int]) -> float:
        if not candidates:
            return 0.0
        top_value = max(self.context.target_values[n] for n in candidates)
        if top_value <= 0.0:
            return 0.0
        return float(self.packing_regret_weight * self.context.target_values[node] / top_value)

    def _tail_packing_value_bonus(self, node_value: float) -> float:
        if not self._uses_tail_model():
            return 0.0
        return float(self.tail_packing_value_bonus * node_value)

    def _active_tail_value_bonus(self) -> float:
        if not self._uses_tail_model():
            return 0.0
        tail_p90 = float(getattr(self, "_adaptive_tail_p90", 0.0))
        risk_p90 = float(getattr(self, "_adaptive_risk_p90", 0.0))
        if tail_p90 >= 0.130 or risk_p90 >= 0.120:
            return float(self.tail_value_bonus)
        return 0.0

    def _record_packing_stop(self, reason: str) -> None:
        self._packing_stop_reasons.append(reason)
        self._last_packing_stop_reason = reason

    def _finalize_packing_stop_reason(self) -> None:
        if not self.use_marginal_packing:
            self._last_packing_stop_reason = "disabled"
            return
        if not self._packing_stop_reasons:
            self._last_packing_stop_reason = "not_started"
            return
        counts: Dict[str, int] = {}
        for reason in self._packing_stop_reasons:
            counts[reason] = counts.get(reason, 0) + 1
        self._last_packing_stop_reason = max(counts, key=counts.get)

    def _high_value_recovery_candidates(
        self,
        start: int,
        candidate_plans: list[Dict[str, object]],
        unvisited: Set[int],
        battery_remaining: float,
    ) -> list[Dict[str, object]]:
        if not candidate_plans or self._time_budget_exhausted():
            return []
        scored: list[tuple[float, Dict[str, object]]] = []
        for plan in candidate_plans:
            route = plan.get("route")
            if not route:
                continue
            scored.append((self._score_plan(plan, battery_remaining), plan))
        if not scored:
            return []
        scored.sort(key=lambda row: row[0], reverse=True)
        recovered_plans: list[Dict[str, object]] = []
        seen_routes = {tuple(plan.get("route", ())) for _, plan in scored}
        limit = max(1, int(self.tail_recovery_plan_limit if self._uses_tail_model() else 3))
        for _, plan in scored[: min(limit, len(scored))]:
            if self._time_budget_exhausted():
                break
            route = list(plan["route"])
            remaining = set(unvisited) - set(route)
            recovered = self._recover_high_value_targets(start, route, remaining, battery_remaining)
            route_key = tuple(recovered)
            if route_key and route_key not in seen_routes:
                seen_routes.add(route_key)
                recovered_plans.append(self._route_stats(start, recovered))
        return recovered_plans

    def _tail_persistent_route_candidates(
        self,
        start: int,
        unvisited: Set[int],
        battery_remaining: float,
    ) -> list[Dict[str, object]]:
        previous = [node for node in getattr(self, "_last_best_route", []) if node in unvisited]
        if not previous:
            return []
        candidates: list[Dict[str, object]] = []
        plan = self._route_stats(start, previous)
        if self._plan_hard_feasible(plan, battery_remaining):
            plan["route_commit_bonus"] = float(self.tail_route_persistence_bonus)
            candidates.append(plan)
        if self.use_marginal_packing and not self._time_budget_exhausted():
            remaining = set(unvisited) - set(previous)
            repaired = self._risk_reserve_marginal_packing_repair(start, list(previous), remaining, battery_remaining)
            if repaired.get("route") and self._plan_hard_feasible(repaired, battery_remaining):
                repaired["route_commit_bonus"] = 0.5 * float(self.tail_route_persistence_bonus)
                candidates.append(repaired)
        return candidates

    def _tail_low_pressure_completion_candidates(
        self,
        start: int,
        unvisited: Set[int],
        battery_remaining: float,
    ) -> list[Dict[str, object]]:
        if not self.use_marginal_packing or not unvisited:
            return []
        saved = {
            "packing_candidate_limit": self.packing_candidate_limit,
            "repair_max_insertions": self.repair_max_insertions,
            "packing_unlock_weight": self.packing_unlock_weight,
            "packing_regret_weight": self.packing_regret_weight,
        }
        try:
            self.packing_candidate_limit = int(min(96, max(self.packing_candidate_limit, len(unvisited))))
            self.repair_max_insertions = int(min(64, max(self.repair_max_insertions, len(unvisited))))
            self.packing_unlock_weight = max(self.packing_unlock_weight, 0.34)
            self.packing_regret_weight = max(self.packing_regret_weight, 0.42)
            plan = self._risk_reserve_marginal_packing_repair(start, [], set(unvisited), battery_remaining)
        finally:
            self.packing_candidate_limit = saved["packing_candidate_limit"]
            self.repair_max_insertions = saved["repair_max_insertions"]
            self.packing_unlock_weight = saved["packing_unlock_weight"]
            self.packing_regret_weight = saved["packing_regret_weight"]
        if not plan.get("route") or not self._plan_hard_feasible(plan, battery_remaining):
            return []
        plan["route_commit_bonus"] = float(self.tail_completion_bonus)
        return [plan]

    def _no_tail_shadow_candidate_plans(
        self,
        start: int,
        unvisited: Set[int],
        battery_remaining: float,
    ) -> list[Dict[str, object]]:
        saved = {
            "tail_quantile": self.tail_quantile,
            "tail_blend": self.tail_blend,
            "return_tail_blend": self.return_tail_blend,
            "report_tail_blend": self.report_tail_blend,
            "tail_budget_cap_quantile": self.tail_budget_cap_quantile,
            "tail_reserve_credit": self.tail_reserve_credit,
            "tail_value_bonus": self.tail_value_bonus,
            "recovery_reserve_credit": self.recovery_reserve_credit,
            "_edge_budget_cache": self._edge_budget_cache,
            "_return_budget_cache": self._return_budget_cache,
            "_route_stats_cache": self._route_stats_cache,
            "_shadow_no_tail_mode": getattr(self, "_shadow_no_tail_mode", False),
        }
        route_candidates: list[list[int]] = []
        try:
            self._shadow_no_tail_mode = True
            self.tail_quantile = "q90"
            self.tail_blend = 1.0
            self.return_tail_blend = 1.0
            self.report_tail_blend = 1.0
            self.tail_budget_cap_quantile = None
            self.tail_reserve_credit = 0.0
            self.tail_value_bonus = 0.0
            self.recovery_reserve_credit = 0.0
            self._edge_budget_cache = {}
            self._return_budget_cache = {}
            self._route_stats_cache = {}
            route_candidates.extend(self._no_tail_shadow_beam_routes(start, set(unvisited), battery_remaining))
            incumbent = self._fast_greedy_incumbent(start, set(unvisited), battery_remaining)
            if incumbent.get("route"):
                route_candidates.append(list(incumbent["route"]))
            if self.use_marginal_packing and not self._time_budget_exhausted():
                packed = self._risk_reserve_marginal_packing_repair(start, [], set(unvisited), battery_remaining)
                if packed.get("route"):
                    route_candidates.append(list(packed["route"]))
            if self.use_insertion_repair and not self._time_budget_exhausted():
                inserted = self._greedy_insertion_plan(start, set(unvisited), battery_remaining)
                if inserted.get("route"):
                    route_candidates.append(list(inserted["route"]))
        finally:
            self.tail_quantile = saved["tail_quantile"]
            self.tail_blend = saved["tail_blend"]
            self.return_tail_blend = saved["return_tail_blend"]
            self.report_tail_blend = saved["report_tail_blend"]
            self.tail_budget_cap_quantile = saved["tail_budget_cap_quantile"]
            self.tail_reserve_credit = saved["tail_reserve_credit"]
            self.tail_value_bonus = saved["tail_value_bonus"]
            self.recovery_reserve_credit = saved["recovery_reserve_credit"]
            self._shadow_no_tail_mode = saved["_shadow_no_tail_mode"]
            self._edge_budget_cache = {}
            self._return_budget_cache = {}
            self._route_stats_cache = {}
        out: list[Dict[str, object]] = []
        seen: set[tuple[int, ...]] = set()
        for route in route_candidates:
            key = tuple(route)
            if not key or key in seen:
                continue
            seen.add(key)
            plan = self._route_stats(start, list(route))
            if self._plan_hard_feasible(plan, battery_remaining):
                plan["shadow_no_tail_candidate"] = True
                out.append(plan)
        return out

    def _no_tail_shadow_beam_routes(
        self,
        start: int,
        unvisited: Set[int],
        battery_remaining: float,
    ) -> list[list[int]]:
        if not self.use_beam_search or not unvisited:
            return []
        saved = (self.beam_width, self.beam_depth, self.repair_top_k)
        try:
            beam_width = int(min(max(1, self.beam_width), 72))
            beam_depth = int(min(max(1, self.beam_depth), 6))
            initial = (tuple(), start, 0.0, 0.0, 0.0, 0.0, 0)
            beam = [initial]
            completed = [initial]
            for _ in range(beam_depth):
                next_states = []
                for route, cursor, travel_cost, value, risk_sum, max_risk, risk_edges in beam:
                    remaining = set(unvisited) - set(route)
                    candidates = self._candidate_pool(cursor, remaining, battery_remaining, travel_cost)
                    for node in candidates:
                        if self._time_budget_exhausted():
                            break
                        self._last_candidate_expansions += 1
                        edge = self._flight_edge(cursor, node)
                        leg_cost = self._leg_budget(edge) + self._photo(node)
                        next_states.append(
                            (
                                route + (node,),
                                node,
                                travel_cost + leg_cost,
                                value + self.context.target_values[node],
                                risk_sum + (edge.risk if self.use_risk else 0.0),
                                max(max_risk, edge.risk) if self.use_risk else 0.0,
                                risk_edges + (1 if self.use_risk else 0),
                            )
                        )
                        if self._last_candidate_expansions >= self.max_replan_expansions:
                            self._last_anytime_stop_reason = "expansion_budget"
                            break
                    if self._last_anytime_stop_reason in {"expansion_budget", "time_budget"}:
                        break
                if self._last_anytime_stop_reason in {"expansion_budget", "time_budget"}:
                    completed.extend(beam)
                    break
                if not next_states:
                    break
                before_prune = len(next_states)
                next_states = self._prune_dominated_states(next_states)
                self._last_pruned_candidates += before_prune - len(next_states)
                next_states.sort(key=lambda s: self._beam_state_score(s, battery_remaining), reverse=True)
                beam = next_states[:beam_width]
                self._last_pruned_candidates += max(0, len(next_states) - len(beam))
                completed.extend(beam)
            completed = [state for state in completed if state[0]]
            completed.sort(key=lambda s: self._beam_state_score(s, battery_remaining), reverse=True)
            return [list(state[0]) for state in completed[: max(1, int(self.repair_top_k))]]
        finally:
            self.beam_width, self.beam_depth, self.repair_top_k = saved

    def _greedy_insertion_plan(
        self,
        start: int,
        remaining: Set[int],
        battery_remaining: float,
    ) -> Dict[str, object]:
        route: list[int] = []
        remaining = set(remaining)
        for _ in range(min(self.repair_max_insertions, len(remaining))):
            if self._time_budget_exhausted():
                break
            if self.use_route_polish and self._should_polish_route(start, route):
                route = self._polish_route(start, route, battery_remaining)
            base_plan = self._route_stats(start, route)
            best_node = None
            best_pos = 0
            best_score = self._score_plan(base_plan, battery_remaining)
            base_value = float(base_plan["value"])
            base_travel = float(base_plan["travel_cost"])
            base_total = float(base_plan["total_cost"])
            base_risk = float(base_plan["risk_sum"])
            base_max_risk = float(base_plan["max_risk"])
            base_risk_edges = int(base_plan["risk_edges"])
            for node in remaining:
                if self._time_budget_exhausted():
                    break
                for pos in range(len(route) + 1):
                    if self._time_budget_exhausted():
                        break
                    delta = self._insertion_delta_metrics(start, route, pos, node)
                    if delta is None:
                        continue
                    delta_total, delta_travel, delta_risk, edge_max_risk, risk_edge_delta = delta
                    value = base_value + self.context.target_values[node]
                    travel = base_travel + delta_travel
                    total = base_total + delta_total
                    risk_sum = base_risk + delta_risk
                    risk_edges = base_risk_edges + risk_edge_delta
                    max_risk = max(base_max_risk, edge_max_risk)
                    if not self._plan_values_hard_feasible(total, travel, risk_sum, max_risk, risk_edges, battery_remaining):
                        continue
                    avg_risk = risk_sum / max(1, risk_edges)
                    risk_guard = max(0.0, avg_risk - self.insertion_risk_guard_fraction * self.risk_avg_cap) if self.use_risk and self.use_risk_scoring else 0.0
                    insertion_score = (
                        self._score_plan_values(value, travel, total, risk_sum, max_risk, risk_edges, len(route) + 1, battery_remaining)
                        + 0.18 * self.context.target_values[node]
                        + self.context.target_values[node] / max(delta_total, 1.0)
                        - self.insertion_risk_weight * risk_guard
                    )
                    if insertion_score > best_score + 1e-9:
                        best_score = insertion_score
                        best_node = node
                        best_pos = pos
            if best_node is None:
                break
            route.insert(best_pos, best_node)
            remaining.remove(best_node)
        if self.use_route_polish and self._should_polish_route(start, route):
            route = self._polish_route(start, route, battery_remaining)
        return self._route_stats(start, route)


    def _improve_route_by_insertion(
        self,
        start: int,
        route: list[int],
        remaining: Set[int],
        battery_remaining: float,
    ) -> Dict[str, object]:
        route = list(route)
        remaining = set(remaining)
        if self.use_route_polish and self._should_polish_route(start, route):
            route = self._polish_route(start, route, battery_remaining)
        max_insertions = min(self.repair_max_insertions, len(remaining))
        for _ in range(max_insertions):
            base_plan = self._route_stats(start, route)
            base_score = self._score_plan(base_plan, battery_remaining)
            best_node = None
            best_pos = 0
            best_score = base_score
            base_value = float(base_plan["value"])
            base_travel = float(base_plan["travel_cost"])
            base_total = float(base_plan["total_cost"])
            base_risk = float(base_plan["risk_sum"])
            base_max_risk = float(base_plan["max_risk"])
            base_risk_edges = int(base_plan["risk_edges"])
            for node in remaining:
                if self._time_budget_exhausted():
                    break
                for pos in range(len(route) + 1):
                    if self._time_budget_exhausted():
                        break
                    delta = self._insertion_delta_metrics(start, route, pos, node)
                    if delta is None:
                        continue
                    delta_total, delta_travel, delta_risk, edge_max_risk, risk_edge_delta = delta
                    value = base_value + self.context.target_values[node]
                    travel = base_travel + delta_travel
                    total = base_total + delta_total
                    risk_sum = base_risk + delta_risk
                    risk_edges = base_risk_edges + risk_edge_delta
                    max_risk = max(base_max_risk, edge_max_risk)
                    score = self._score_plan_values(value, travel, total, risk_sum, max_risk, risk_edges, len(route) + 1, battery_remaining)
                    if score > best_score + 1e-9:
                        best_score = score
                        best_node = node
                        best_pos = pos
            if best_node is None:
                break
            route.insert(best_pos, best_node)
            remaining.remove(best_node)
            if self.use_route_polish and self._should_polish_route(start, route):
                route = self._polish_route(start, route, battery_remaining)
        return self._route_stats(start, route)

    def _should_polish_route(self, start: int, route: list[int]) -> bool:
        if not (2 < len(route) <= self.polish_node_limit):
            return False
        if not self.use_risk:
            return True
        plan = self._route_stats(start, route)
        avg_risk = float(plan["risk_sum"]) / max(1, int(plan["risk_edges"]))
        max_risk = float(plan["max_risk"])
        return avg_risk >= self.polish_risk_trigger or max_risk >= 2.0 * self.risk_avg_cap

    def _time_budget_exhausted(self) -> bool:
        deadline = getattr(self, "_replan_deadline", None)
        if deadline is None:
            return False
        if time.perf_counter() <= deadline:
            return False
        self._last_anytime_stop_reason = "time_budget"
        return True

    def _polish_route(self, start: int, route: list[int], battery_remaining: float) -> list[int]:
        route = list(route)
        if not self._should_polish_route(start, route):
            return route
        best_plan = self._route_stats(start, route)
        best_score = self._score_plan(best_plan, battery_remaining)
        for _ in range(self.polish_passes):
            improved = False
            best_route = route
            n = len(route)
            for i in range(n):
                for j in range(i + 1, n):
                    trial = list(route)
                    trial[i], trial[j] = trial[j], trial[i]
                    plan = self._route_stats(start, trial)
                    score = self._score_plan(plan, battery_remaining)
                    if score > best_score + 1e-9:
                        best_score = score
                        best_route = trial
                        improved = True
                    if j - i >= 2:
                        trial = route[:i] + list(reversed(route[i : j + 1])) + route[j + 1 :]
                        plan = self._route_stats(start, trial)
                        score = self._score_plan(plan, battery_remaining)
                        if score > best_score + 1e-9:
                            best_score = score
                            best_route = trial
                            improved = True
            for i in range(n):
                node = route[i]
                reduced = route[:i] + route[i + 1 :]
                for pos in range(len(reduced) + 1):
                    if pos == i:
                        continue
                    trial = list(reduced)
                    trial.insert(pos, node)
                    plan = self._route_stats(start, trial)
                    score = self._score_plan(plan, battery_remaining)
                    if score > best_score + 1e-9:
                        best_score = score
                        best_route = trial
                        improved = True
            route = best_route
            if not improved:
                break
        return route

    def _ejection_repair(
        self,
        start: int,
        route: list[int],
        remaining: Set[int],
        battery_remaining: float,
    ) -> Tuple[list[int], Set[int]]:
        route = list(route)
        remaining = set(remaining)
        for _ in range(min(self.repair_max_insertions, len(remaining))):
            base_plan = self._route_stats(start, route)
            base_score = self._score_plan(base_plan, battery_remaining)
            best_route = route
            best_removed = None
            best_added = None
            best_score = base_score
            for removed in list(route):
                reduced = [node for node in route if node != removed]
                for added in remaining:
                    for pos in range(len(reduced) + 1):
                        trial = list(reduced)
                        trial.insert(pos, added)
                        plan = self._route_stats(start, trial)
                        if not self._plan_hard_feasible(plan, battery_remaining):
                            continue
                        score = self._score_plan(plan, battery_remaining)
                        if score > best_score + 1e-9:
                            best_route = trial
                            best_removed = removed
                            best_added = added
                            best_score = score
            if best_added is None or best_removed is None:
                break
            route = best_route
            remaining.remove(best_added)
            remaining.add(best_removed)
            if self.use_route_polish and self._should_polish_route(start, route):
                route = self._polish_route(start, route, battery_remaining)
        return route, remaining

    def _route_stats(self, start: int, route: list[int]) -> Dict[str, object]:
        route_key = tuple(route)
        cache_key = (start, route_key)
        cached = self._route_stats_cache.get(cache_key)
        if cached is not None:
            out = dict(cached)
            out["route"] = list(route)
            return out
        travel_cost = 0.0
        risk_sum = 0.0
        max_risk = 0.0
        risk_edges = 0
        cursor = start
        for node in route:
            edge = self._flight_edge(cursor, node)
            travel_cost += self._leg_budget(edge) + self._photo(node)
            if self.use_risk:
                risk_sum += edge.risk
                max_risk = max(max_risk, edge.risk)
                risk_edges += 1
            cursor = node
        return_cost = self._return_budget(cursor) if cursor != self.context.depot_idx else 0.0
        if self.use_risk and cursor != self.context.depot_idx:
            ret = self._return_edge(cursor)
            risk_sum += ret.risk
            max_risk = max(max_risk, ret.risk)
            risk_edges += 1
        stats = {
            "route": route_key,
            "value": float(sum(self.context.target_values[node] for node in route)),
            "travel_cost": travel_cost,
            "total_cost": travel_cost + return_cost,
            "risk_sum": risk_sum,
            "max_risk": max_risk,
            "risk_edges": risk_edges,
        }
        self._route_stats_cache[cache_key] = stats
        out = dict(stats)
        out["route"] = list(route)
        return out

    def _score_plan(self, plan: Dict[str, object], battery_remaining: float) -> float:
        route = plan["route"]
        visited_count = len(route) if isinstance(route, (list, tuple)) else 0
        return self._score_plan_values(
            float(plan["value"]),
            float(plan["travel_cost"]),
            float(plan["total_cost"]),
            float(plan["risk_sum"]),
            float(plan["max_risk"]),
            int(plan["risk_edges"]),
            visited_count,
            battery_remaining,
        )

    def _score_plan_values(
        self,
        value: float,
        travel_cost: float,
        total_cost: float,
        risk_sum: float,
        max_risk: float,
        risk_edges: int,
        visited_count: int,
        battery_remaining: float,
    ) -> float:
        cfg = self.context.cfg
        avg_risk = risk_sum / max(1, risk_edges)
        return_cost = max(0.0, total_cost - travel_cost)
        reserve_req = self._reserve_requirement(battery_remaining, return_cost, avg_risk)
        final_slack = battery_remaining - total_cost
        if final_slack < reserve_req - 1e-9:
            return -1e18
        risk_penalty = 0.0
        if self.use_risk and self.use_risk_scoring:
            avg_gate = self.risk_penalty_gate_fraction * self.risk_avg_cap
            max_gate = 0.78 * self.context.cfg.p_max
            risk_penalty = (
                self.risk_avg_penalty * max(0.0, avg_risk - avg_gate)
                + self.risk_max_penalty * max(0.0, max_risk - max_gate)
            )
        usable_slack = max(0.0, final_slack - reserve_req)
        reserve_utilization_bonus = 1.0 - max(0.0, min(1.0, usable_slack / max(3.0 * cfg.reserve_floor, 1.0)))
        tail_value_bonus = self._active_tail_value_bonus() * visited_count * min(
            1.0, usable_slack / max(2.0 * cfg.reserve_floor, 1.0)
        )
        return (
            self.plan_value_weight * value
            + self.visited_count_bonus * visited_count
            + tail_value_bonus
            + self.reserve_utilization_weight * reserve_utilization_bonus
            - self.energy_penalty_coeff * total_cost
            - risk_penalty
        )

    def _plan_hard_feasible(self, plan: Dict[str, object], battery_remaining: float) -> bool:
        return self._plan_values_hard_feasible(
            float(plan["total_cost"]),
            float(plan["travel_cost"]),
            float(plan["risk_sum"]),
            float(plan["max_risk"]),
            int(plan["risk_edges"]),
            battery_remaining,
        )

    def _plan_values_hard_feasible(
        self,
        total_cost: float,
        travel_cost: float,
        risk_sum: float,
        max_risk: float,
        risk_edges: int,
        battery_remaining: float,
    ) -> bool:
        cfg = self.context.cfg
        avg_risk = risk_sum / max(1, risk_edges)
        return_cost = max(0.0, total_cost - travel_cost)
        reserve_req = self._reserve_requirement(battery_remaining, return_cost, avg_risk)
        final_slack = battery_remaining - total_cost
        if final_slack < reserve_req - 1e-9:
            return False
        if self.use_risk:
            if self.enforce_route_risk_cap and avg_risk > self.risk_avg_cap + 1e-12:
                return False
            if self.enforce_risk_edge_cap and max_risk > cfg.p_max + 1e-12:
                return False
        return True

    def _recovery_values_feasible(
        self,
        total_cost: float,
        travel_cost: float,
        risk_sum: float,
        max_risk: float,
        risk_edges: int,
        battery_remaining: float,
    ) -> bool:
        cfg = self.context.cfg
        avg_risk = risk_sum / max(1, risk_edges)
        return_cost = max(0.0, total_cost - travel_cost)
        reserve_req = self._reserve_requirement(battery_remaining, return_cost, avg_risk)
        reserve_credit = self.recovery_reserve_credit if self._uses_tail_model() else 0.0
        reserve_req = max(0.55 * cfg.reserve_floor, reserve_req - reserve_credit)
        if battery_remaining - total_cost < reserve_req - 1e-9:
            return False
        if self.use_risk:
            if self.enforce_risk_edge_cap and max_risk > cfg.p_max + 1e-12:
                return False
            if self.enforce_route_risk_cap and avg_risk > self.risk_avg_cap + 0.0008:
                return False
        return True

    def _record_missed_diagnostics(self, current: int, unvisited: Set[int], battery_remaining: float) -> None:
        feasible = 0
        rejected_edge_risk = 0
        rejected_avg_risk = 0
        rejected_reserve = 0
        top_value = 0.0
        top_delta = 0.0
        top_avg_risk = 0.0
        top_return = 0.0
        for node in unvisited:
            edge = self._flight_edge(current, node)
            ret = self._return_edge(node)
            if self.use_risk and self.enforce_risk_edge_cap and (edge.risk > self.context.cfg.p_max or ret.risk > self.context.cfg.p_max):
                rejected_edge_risk += 1
                continue
            leg = self._leg_budget(edge) + self._photo(node)
            return_budget = self._return_budget(node)
            total = leg + return_budget
            risk_sum = edge.risk + ret.risk if self.use_risk else 0.0
            avg_risk = risk_sum / (2 if self.use_risk else 1)
            if self.use_risk and self.enforce_route_risk_cap and avg_risk > self.risk_avg_cap + 0.0008:
                rejected_avg_risk += 1
                continue
            reserve = self._reserve_requirement(battery_remaining, return_budget, avg_risk)
            if battery_remaining - total < reserve:
                rejected_reserve += 1
                continue
            feasible += 1
            value = self.context.target_values[node]
            if value > top_value:
                top_value = value
                top_delta = total
                top_avg_risk = avg_risk
                top_return = return_budget
        self._last_missed_diagnostics = {
            "missed_single_feasible": feasible,
            "missed_reject_edge_risk": rejected_edge_risk,
            "missed_reject_avg_risk": rejected_avg_risk,
            "missed_reject_reserve": rejected_reserve,
            "missed_top_value": float(top_value),
            "missed_top_delta_total": float(top_delta),
            "missed_top_avg_risk": float(top_avg_risk),
            "missed_top_return_budget": float(top_return),
        }


    def _insertion_delta_and_risk(
        self,
        start: int,
        route: list[int],
        pos: int,
        node: int,
    ) -> Optional[Tuple[float, float]]:
        metrics = self._insertion_delta_metrics(start, route, pos, node)
        if metrics is None:
            return None
        delta_total, _, delta_risk, _, _ = metrics
        return float(delta_total), float(max(0.0, delta_risk))

    def _insertion_delta_metrics(
        self,
        start: int,
        route: list[int],
        pos: int,
        node: int,
    ) -> Optional[Tuple[float, float, float, float, int]]:
        prev = start if pos == 0 else route[pos - 1]
        if pos < len(route):
            nxt = route[pos]
            old_edge = self._flight_edge(prev, nxt)
            first_edge = self._flight_edge(prev, node)
            second_edge = self._flight_edge(node, nxt)
            if self.use_risk and self.enforce_risk_edge_cap and (first_edge.risk > self.context.cfg.p_max or second_edge.risk > self.context.cfg.p_max):
                return None
            old_cost = self._leg_budget(old_edge) + self._photo(nxt)
            new_cost = self._leg_budget(first_edge) + self._photo(node) + self._leg_budget(second_edge) + self._photo(nxt)
            old_travel = old_cost
            new_travel = new_cost
            added_risk = first_edge.risk + second_edge.risk - old_edge.risk if self.use_risk else 0.0
            max_risk = max(first_edge.risk, second_edge.risk) if self.use_risk else 0.0
            risk_edge_delta = 1 if self.use_risk else 0
        else:
            old_cost = self._return_budget(prev) if prev != self.context.depot_idx else 0.0
            first_edge = self._flight_edge(prev, node)
            ret_edge = self._return_edge(node)
            if self.use_risk and self.enforce_risk_edge_cap and (first_edge.risk > self.context.cfg.p_max or ret_edge.risk > self.context.cfg.p_max):
                return None
            new_cost = self._leg_budget(first_edge) + self._photo(node) + self._return_budget(node)
            old_travel = 0.0
            new_travel = self._leg_budget(first_edge) + self._photo(node)
            old_risk = self._return_edge(prev).risk if self.use_risk and prev != self.context.depot_idx else 0.0
            added_risk = first_edge.risk + ret_edge.risk - old_risk if self.use_risk else 0.0
            max_risk = max(first_edge.risk, ret_edge.risk) if self.use_risk else 0.0
            old_edges = 1 if self.use_risk and prev != self.context.depot_idx else 0
            risk_edge_delta = (2 - old_edges) if self.use_risk else 0
        return float(new_cost - old_cost), float(new_travel - old_travel), float(added_risk), float(max_risk), int(risk_edge_delta)

    def _is_feasible(self, current: int, j: int, battery_remaining: float) -> bool:
        cfg = self.context.cfg
        edge = self._flight_edge(current, j)
        if self.use_risk and self.enforce_risk_edge_cap and edge.risk > cfg.p_max:
            return False
        required = self._leg_budget(edge) + self._photo(j) + self._return_budget(j)
        return battery_remaining - required >= self._reserve_requirement(battery_remaining, self._return_budget(j), edge.risk)

    def planned_edge_energy(self, edge: EdgeMetrics, include_photo: bool) -> float:
        if self.use_wind and self.use_tail:
            mean = edge.energy_adaptive_mean if self.use_adaptive_energy else edge.energy_mean
            tail = self._tail_energy(edge)
            budget = mean + self.report_tail_blend * (tail - mean)
        else:
            budget = self._edge_budget(edge, return_leg=not include_photo)
        return float(budget + (self.context.cfg.e_photo if include_photo else 0.0))

    def _leg_budget(self, edge: EdgeMetrics) -> float:
        return self._edge_budget(edge, return_leg=False)

    def _return_budget(self, node: int) -> float:
        cached = self._return_budget_cache.get(node)
        if cached is not None:
            return cached
        value = self._edge_budget(self._return_edge(node), return_leg=True)
        self._return_budget_cache[node] = value
        return value

    def _edge_budget(self, edge: EdgeMetrics, return_leg: bool) -> float:
        key = (edge.source, edge.target, return_leg)
        cached = self._edge_budget_cache.get(key)
        if cached is not None:
            return cached
        if not self.use_wind:
            budget = edge.no_wind_energy
            self._edge_budget_cache[key] = budget
            return budget
        mean = edge.energy_adaptive_mean if self.use_adaptive_energy else edge.energy_mean
        if self.use_tail:
            tail = self._tail_energy(edge)
            blend = self._adaptive_tail_blend(edge, return_leg)
            budget = mean + blend * (tail - mean)
            if self.use_adaptive_energy:
                budget *= self._dynamic_energy_multiplier(edge, return_leg)
            if self._uses_tail_model() and self.tail_budget_cap_quantile:
                cap_tail = self._tail_energy_for_quantile(edge, str(self.tail_budget_cap_quantile))
                cap_budget = mean + self.tail_budget_cap_blend * (cap_tail - mean)
                budget = min(budget, cap_budget)
            self._edge_budget_cache[key] = budget
            return budget
        budget = mean + self.no_tail_energy_margin * edge.no_wind_energy
        self._edge_budget_cache[key] = budget
        return budget

    def _adaptive_tail_blend(self, edge: EdgeMetrics, return_leg: bool) -> float:
        if getattr(self, "_shadow_no_tail_mode", False):
            return 1.0
        base = self.return_tail_blend if return_leg else self.tail_blend
        if not self.use_risk:
            return float(base)
        risk_pressure = min(1.0, max(0.0, edge.risk) / max(self.risk_avg_cap, 1e-9))
        tail = self._tail_energy(edge)
        mean = edge.energy_adaptive_mean if self.use_adaptive_energy else edge.energy_mean
        tail_pressure = min(1.0, max(0.0, tail - mean) / max(edge.no_wind_energy, 1.0))
        battery = float(getattr(self, "_battery_for_scoring", self.context.cfg.battery_capacity))
        battery_pressure = max(0.0, 1.0 - battery / max(self.context.cfg.battery_capacity, 1.0))
        if return_leg:
            scale = 0.52 + 0.22 * risk_pressure + 0.12 * tail_pressure + 0.18 * battery_pressure
        else:
            scale = 0.42 + 0.26 * risk_pressure + 0.12 * tail_pressure + 0.08 * battery_pressure
        return float(base * max(0.28, min(1.05, scale)))

    def _dynamic_energy_multiplier(self, edge: EdgeMetrics, return_leg: bool) -> float:
        if getattr(self, "_shadow_no_tail_mode", False):
            return 1.0
        residual = max(0.0, float(getattr(self, "_energy_residual_ema", 0.0)))
        tail_ratio = max(0.0, (self._tail_energy(edge) - (edge.energy_adaptive_mean if self.use_adaptive_energy else edge.energy_mean)) / max(edge.no_wind_energy, 1.0))
        battery = float(getattr(self, "_battery_for_scoring", self.context.cfg.battery_capacity))
        low_battery = max(0.0, 1.0 - battery / max(self.context.cfg.battery_capacity, 1.0))
        return float(1.0 + 0.04 * residual + 0.010 * tail_ratio + (0.014 if return_leg else 0.006) * low_battery)

    def _reserve_requirement(self, battery_remaining: float, return_budget: float, risk: float) -> float:
        cfg = self.context.cfg
        battery_pressure = max(0.0, 1.0 - battery_remaining / max(cfg.battery_capacity, 1.0))
        return_pressure = min(1.0, return_budget / max(cfg.battery_capacity, 1.0))
        if self.blind_reserve_pressure is not None:
            risk_pressure = float(self.blind_reserve_pressure)
        elif self.use_risk:
            risk_pressure = min(1.0, max(0.0, risk) / max(self.risk_avg_cap, 1e-9))
        else:
            risk_pressure = float(self.no_risk_reserve_pressure)
        adaptive_buffer = self.reserve_buffer * (0.28 + 0.26 * battery_pressure + 0.20 * return_pressure + 0.26 * risk_pressure)
        reserve = cfg.reserve_floor + adaptive_buffer + 22.0 * battery_pressure + 18.0 * return_pressure + 12.0 * risk_pressure
        tail_reserve_credit = self._active_tail_reserve_credit()
        if self._uses_tail_model() and tail_reserve_credit > 0.0:
            confidence = 0.45 + 0.25 * max(0.0, 1.0 - battery_pressure) + 0.30 * min(1.0, return_pressure)
            reserve -= tail_reserve_credit * confidence
        return float(max(0.55 * cfg.reserve_floor, reserve))

    def _active_tail_reserve_credit(self) -> float:
        if not self._uses_tail_model():
            return 0.0
        credit = float(self.tail_reserve_credit)
        tail_p90 = float(getattr(self, "_adaptive_tail_p90", 0.0))
        risk_p90 = float(getattr(self, "_adaptive_risk_p90", 0.0))
        if tail_p90 <= 0.065 and risk_p90 <= 0.015:
            credit += float(self.tail_low_pressure_reserve_credit_bonus)
        return credit

    def _uses_tail_model(self) -> bool:
        if self._use_tail_q90_fallback():
            return False
        return bool(self.use_tail and self.use_adaptive_energy and str(self.tail_quantile).startswith("tail"))

    def _use_tail_q90_fallback(self) -> bool:
        if not (self.use_tail_low_pressure_q90_fallback and self.use_tail and self.use_adaptive_energy):
            return False
        tail_threshold = float(self.tail_q90_fallback_tail_threshold)
        risk_threshold = float(self.tail_q90_fallback_risk_threshold)
        detour_threshold = float(self.tail_q90_fallback_detour_threshold)
        if tail_threshold <= 0.0 or risk_threshold <= 0.0:
            return False
        tail_p90 = float(getattr(self, "_adaptive_tail_p90", 0.0))
        risk_p90 = float(getattr(self, "_adaptive_risk_p90", 0.0))
        detour_p95 = float(getattr(self, "_adaptive_detour_p95", 1.0))
        detour_ok = detour_threshold <= 0.0 or detour_p95 >= detour_threshold
        return bool(tail_p90 <= tail_threshold and risk_p90 <= risk_threshold and detour_ok)

    def _enable_tail_high_value_recovery(self) -> bool:
        if not (self.use_tail_high_value_recovery and self._uses_tail_model()):
            return False
        tail_p90 = float(getattr(self, "_adaptive_tail_p90", 0.0))
        risk_p90 = float(getattr(self, "_adaptive_risk_p90", 0.0))
        high_pressure = tail_p90 >= 0.090 or risk_p90 >= 0.095
        moderate_pressure = tail_p90 >= 0.050 and risk_p90 >= 0.035
        return bool(high_pressure or moderate_pressure)

    def _enable_no_tail_shadow_candidates(self) -> bool:
        return bool(
            self.use_no_tail_shadow_candidates
            and self._uses_tail_model()
            and self._enable_no_tail_shadow_value_fallback()
        )

    def _enable_no_tail_shadow_value_fallback(self) -> bool:
        if not (self.use_no_tail_shadow_value_fallback and self._uses_tail_model()):
            return False
        tail_threshold = float(self.no_tail_shadow_fallback_tail_threshold)
        risk_threshold = float(self.no_tail_shadow_fallback_risk_threshold)
        if tail_threshold <= 0.0 and risk_threshold <= 0.0:
            return True
        tail_p90 = float(getattr(self, "_adaptive_tail_p90", 0.0))
        risk_p90 = float(getattr(self, "_adaptive_risk_p90", 0.0))
        tail_ok = tail_threshold <= 0.0 or tail_p90 <= tail_threshold
        risk_ok = risk_threshold <= 0.0 or risk_p90 <= risk_threshold
        return bool(tail_ok and risk_ok)

    def _enable_tail_route_persistence(self) -> bool:
        if not (self.use_tail_route_persistence and self._uses_tail_model()):
            return False
        tail_p90 = float(getattr(self, "_adaptive_tail_p90", 0.0))
        risk_p90 = float(getattr(self, "_adaptive_risk_p90", 0.0))
        return bool(tail_p90 >= 0.070 or risk_p90 >= 0.090)

    def _enable_tail_low_pressure_completion(self) -> bool:
        if not (self.use_tail_low_pressure_completion and self._uses_tail_model()):
            return False
        tail_p90 = float(getattr(self, "_adaptive_tail_p90", 0.0))
        risk_p90 = float(getattr(self, "_adaptive_risk_p90", 0.0))
        return bool(tail_p90 <= 0.060 and risk_p90 <= 0.040)

    def _enable_tail_value_override(self) -> bool:
        if not (self._uses_tail_model() and self.tail_value_override_tolerance > 0.0):
            return False
        tail_threshold = float(self.tail_value_override_tail_threshold)
        risk_threshold = float(self.tail_value_override_risk_threshold)
        if tail_threshold <= 0.0 and risk_threshold <= 0.0:
            return True
        tail_p90 = float(getattr(self, "_adaptive_tail_p90", 0.0))
        risk_p90 = float(getattr(self, "_adaptive_risk_p90", 0.0))
        return bool(
            (tail_threshold > 0.0 and tail_p90 >= tail_threshold)
            or (risk_threshold > 0.0 and risk_p90 >= risk_threshold)
        )

    def _tail_energy(self, edge: EdgeMetrics) -> float:
        if self._use_tail_q90_fallback():
            return self._tail_energy_for_quantile(edge, "q90")
        return self._tail_energy_for_quantile(edge, str(self.tail_quantile))

    def _tail_energy_for_quantile(self, edge: EdgeMetrics, quantile: str) -> float:
        if self.use_adaptive_energy:
            if quantile == "q80":
                return edge.energy_adaptive_q80
            if quantile == "q95":
                return edge.energy_adaptive_q95
            if quantile == "q90":
                return edge.energy_adaptive_q90
            if quantile == "tail95":
                return edge.energy_adaptive_tail95
            return edge.energy_adaptive_tail90
        if quantile == "q80":
            return edge.energy_q80
        if quantile == "q95":
            return edge.energy_q95
        if quantile == "q90":
            return edge.energy_q90
        if quantile == "tail95":
            return edge.energy_tail95
        return edge.energy_tail90

    def actual_edge_energy(self, edge: EdgeMetrics, rng, gust_multiplier: float, include_photo: bool) -> float:
        sample = float(edge.sampled_energy(rng, gust_multiplier, adaptive=self.use_adaptive_energy))
        mean = edge.energy_adaptive_mean if self.use_adaptive_energy else edge.energy_mean
        expected = max(mean, 1.0)
        residual = max(-0.5, min(2.0, (sample - expected) / expected))
        self._energy_residual_ema = 0.82 * float(getattr(self, "_energy_residual_ema", 0.0)) + 0.18 * residual
        return float(sample + (self.context.cfg.e_photo if include_photo else 0.0))


class RAWARHP(RolloutRHP):
    name = "RAWA-RHP"
    use_tail = True
    use_risk = True
    use_wind = True
    tail_blend = 1.0
    return_tail_blend = 1.0
    tail_quantile = "q90"
    reserve_buffer = 16.0
    visited_count_bonus = 0.05
    plan_value_weight = 1.0
    energy_penalty_coeff = 0.006
    use_beam_search = True
    use_insertion_repair = False
    beam_width = 96
    beam_depth = 7
    candidate_pool_size = 16
    repair_top_k = 6
    risk_avg_cap = 0.075
    candidate_risk_weight = 0.0
    insertion_risk_guard_fraction = 0.82
    insertion_risk_weight = 18.0
    risk_avg_penalty = 0.0
    risk_max_penalty = 0.0
    risk_penalty_gate_fraction = 0.90
    enforce_risk_edge_cap = False
    enforce_route_risk_cap = False
    use_risk_candidate_bias = False
    repair_max_insertions = 10
    use_route_polish = True
    use_ejection_repair = False
    use_adaptive_search_budget = True
    use_high_value_recovery = False
    use_tail_high_value_recovery = False
    use_no_tail_shadow_candidates = False
    use_no_tail_shadow_value_fallback = False
    no_tail_shadow_fallback_value_gain = 0.75
    no_tail_shadow_fallback_score_tolerance = 1.25
    no_tail_shadow_fallback_tail_threshold = 0.0
    no_tail_shadow_fallback_risk_threshold = 0.0
    use_tail_route_persistence = False
    tail_route_persistence_bonus = 0.0
    use_tail_low_pressure_completion = False
    tail_completion_bonus = 0.0
    tail_value_override_tolerance = 0.0
    tail_value_override_gain_threshold = 0.75
    tail_value_override_tail_threshold = 0.070
    tail_value_override_risk_threshold = 0.090
    use_tail_low_pressure_q90_fallback = False
    tail_q90_fallback_tail_threshold = 0.0
    tail_q90_fallback_risk_threshold = 0.0
    tail_q90_fallback_detour_threshold = 0.0
    use_marginal_packing = True
    packing_candidate_limit = 18
    packing_regret_weight = 0.38
    packing_unlock_weight = 0.24
    tail_packing_value_bonus = 0.0
    tail_candidate_pool_bonus = 0
    tail_repair_top_k_bonus = 0
    tail_packing_candidate_bonus = 0
    tail_replan_expansion_bonus = 0
    polish_passes = 1
    polish_node_limit = 26
    polish_risk_trigger = 0.0020
    use_adaptive_energy = True
    report_tail_blend = 1.0
    max_replan_seconds = 1.75
    max_replan_expansions = 9000
    recovery_reserve_credit = 0.0
    tail_reserve_credit = 0.0
    tail_low_pressure_reserve_credit_bonus = 0.0
    tail_value_bonus = 0.0
    tail_recovery_plan_limit = 3
    tail_budget_cap_quantile = None
    tail_budget_cap_blend = 1.0


class RAWANoBeam(RAWARHP):
    name = "RAWA-NoBeam"
    beam_width = 1

    def reset(self, context: PlannerContext) -> None:
        super().reset(context)
        self.beam_width = 1

    def _configure_replan_budget(self, unvisited: Set[int], battery_remaining: float) -> None:
        super()._configure_replan_budget(unvisited, battery_remaining)
        self.beam_width = 1


class RAWANoRepair(RAWARHP):
    name = "RAWA-NoRepair"
    use_insertion_repair = False
    use_marginal_packing = False


class RAWANoPacking(RAWARHP):
    name = "RAWA-NoPacking"
    use_marginal_packing = False


class RAWANoAdaptiveSearch(RAWARHP):
    name = "RAWA-NoAdaptiveSearch"
    use_adaptive_search_budget = False


class RAWAOpenLoop(RAWARHP):
    name = "RAWA-OpenLoop"
    use_insertion_repair = False
    use_marginal_packing = False
    use_route_polish = False
    beam_width = 192
    beam_depth = 18
    candidate_pool_size = 32
    repair_top_k = 12

    def reset(self, context: PlannerContext) -> None:
        super().reset(context)
        self._open_loop_route: list[int] = []
        self._open_loop_initialized = False

    def choose_next(self, current: int, unvisited: Set[int], battery_remaining: float) -> Optional[int]:
        if not self._open_loop_initialized:
            first = super().choose_next(current, unvisited, battery_remaining)
            self._open_loop_route = list(getattr(self, "_last_best_route", []))
            self._open_loop_initialized = True
            return first
        while self._open_loop_route and self._open_loop_route[0] not in unvisited:
            self._open_loop_route.pop(0)
        if not self._open_loop_route:
            return None
        nxt = self._open_loop_route.pop(0)
        if nxt not in unvisited:
            return None
        return nxt


class FairALNSPlanner(RAWARHP):
    name = "FairALNSPlanner"
    use_beam_search = False
    use_insertion_repair = False
    use_marginal_packing = False
    plan_value_weight = 1.0
    energy_penalty_coeff = 0.006
    alns_iterations = 18
    alns_removal_batch = 3

    def choose_next(self, current: int, unvisited: Set[int], battery_remaining: float) -> Optional[int]:
        self._battery_for_scoring = float(battery_remaining)
        self._edge_budget_cache = {}
        self._return_budget_cache = {}
        self._route_stats_cache = {}
        self._replan_deadline = (time.perf_counter() + self.max_replan_seconds) if self.max_replan_seconds > 0.0 else None
        self._last_candidate_expansions = 0
        self._last_pruned_candidates = 0
        self._last_anytime_stop_reason = "complete"
        self._last_packing_attempts = 0
        self._last_packing_accepts = 0
        self._last_packing_added_value = 0.0
        self._last_packing_stop_reason = "disabled"
        self._packing_stop_reasons = []
        if not unvisited:
            return None
        route = self._construct_alns_route(current, set(unvisited), battery_remaining)
        if not route:
            self._last_best_route = []
            if self._last_anytime_stop_reason == "complete":
                self._last_anytime_stop_reason = "no_feasible_plan"
            return None
        best_route = list(route)
        best_score = self._score_plan(self._route_stats(current, best_route), battery_remaining)
        for iteration in range(self.alns_iterations):
            if self._time_budget_exhausted():
                break
            partial, removed = self._destroy_alns_route(current, best_route, iteration)
            candidates = (set(unvisited) - set(partial)) | set(removed)
            trial = self._repair_alns_route(current, partial, candidates, battery_remaining)
            if self._time_budget_exhausted():
                break
            trial = self._relocate_alns_route(current, trial, battery_remaining)
            plan = self._route_stats(current, trial)
            if not trial or not self._plan_hard_feasible(plan, battery_remaining):
                continue
            score = self._score_plan(plan, battery_remaining)
            if score > best_score + 1e-9:
                best_score = score
                best_route = list(trial)
        self._last_best_route = list(best_route)
        return best_route[0] if best_route else None

    def _construct_alns_route(self, start: int, candidates: Set[int], battery_remaining: float) -> list[int]:
        route: list[int] = []
        remaining = set(candidates)
        while remaining:
            if self._time_budget_exhausted():
                break
            if self._last_anytime_stop_reason == "expansion_budget":
                break
            choice = self._best_alns_insertion(start, route, remaining, battery_remaining)
            if choice is None:
                break
            node, pos, _ = choice
            route.insert(pos, node)
            remaining.remove(node)
        return route

    def _repair_alns_route(
        self,
        start: int,
        route: list[int],
        candidates: Set[int],
        battery_remaining: float,
    ) -> list[int]:
        route = list(route)
        remaining = set(candidates) - set(route)
        while remaining:
            if self._time_budget_exhausted():
                break
            if self._last_anytime_stop_reason == "expansion_budget":
                break
            choice = self._best_alns_insertion(start, route, remaining, battery_remaining)
            if choice is None:
                break
            node, pos, _ = choice
            route.insert(pos, node)
            remaining.remove(node)
        return route

    def _best_alns_insertion(
        self,
        start: int,
        route: list[int],
        candidates: Set[int],
        battery_remaining: float,
    ) -> Optional[Tuple[int, int, float]]:
        base_plan = self._route_stats(start, route)
        base_score = self._score_plan(base_plan, battery_remaining)
        best: Optional[Tuple[int, int, float]] = None
        best_score = base_score
        for node in candidates:
            if self._time_budget_exhausted():
                break
            if node in route:
                continue
            for pos in range(len(route) + 1):
                if self._time_budget_exhausted():
                    break
                self._last_candidate_expansions += 1
                if self._last_candidate_expansions >= self.max_replan_expansions:
                    self._last_anytime_stop_reason = "expansion_budget"
                    break
                trial = list(route)
                trial.insert(pos, node)
                plan = self._route_stats(start, trial)
                if not self._plan_hard_feasible(plan, battery_remaining):
                    self._last_pruned_candidates += 1
                    continue
                score = self._score_plan(plan, battery_remaining)
                score += 0.03 * self.context.target_values[node] / max(1.0, float(plan["total_cost"]) - float(base_plan["total_cost"]))
                if score > best_score + 1e-9:
                    best_score = score
                    best = (node, pos, score)
            if self._last_anytime_stop_reason == "expansion_budget":
                break
        return best

    def _destroy_alns_route(self, start: int, route: list[int], iteration: int) -> Tuple[list[int], list[int]]:
        route = list(route)
        if len(route) <= 3:
            return route, []
        k = min(len(route) - 1, self.alns_removal_batch + (iteration % 2))
        mode = iteration % 4
        if mode == 0:
            ranked = sorted(route, key=lambda n: self._alns_node_risk(start, route, n), reverse=True)
        elif mode == 1:
            ranked = sorted(route, key=lambda n: self.context.target_values[n])
        elif mode == 2:
            ranked = sorted(route, key=lambda n: self._alns_node_cost_ratio(start, route, n), reverse=True)
        else:
            offset = iteration % len(route)
            ranked = [route[i] for i in range(offset, len(route), 2)] + [route[i] for i in range(offset % 2, len(route), 2)]
        removed = ranked[:k]
        kept = [node for node in route if node not in set(removed)]
        return kept, removed

    def _relocate_alns_route(self, start: int, route: list[int], battery_remaining: float) -> list[int]:
        route = list(route)
        if len(route) < 3:
            return route
        best_plan = self._route_stats(start, route)
        best_score = self._score_plan(best_plan, battery_remaining)
        improved = True
        while improved:
            if self._time_budget_exhausted():
                break
            improved = False
            for i, node in enumerate(list(route)):
                if self._time_budget_exhausted():
                    break
                reduced = route[:i] + route[i + 1 :]
                for pos in range(len(reduced) + 1):
                    if self._time_budget_exhausted():
                        break
                    if pos == i:
                        continue
                    self._last_candidate_expansions += 1
                    if self._last_candidate_expansions >= self.max_replan_expansions:
                        self._last_anytime_stop_reason = "expansion_budget"
                        break
                    trial = list(reduced)
                    trial.insert(pos, node)
                    plan = self._route_stats(start, trial)
                    if not self._plan_hard_feasible(plan, battery_remaining):
                        self._last_pruned_candidates += 1
                        continue
                    score = self._score_plan(plan, battery_remaining)
                    if score > best_score + 1e-9:
                        route = trial
                        best_score = score
                        improved = True
                        break
                if self._last_anytime_stop_reason == "expansion_budget":
                    break
                if improved:
                    break
            if self._last_anytime_stop_reason == "expansion_budget":
                break
        return route

    def _alns_node_risk(self, start: int, route: list[int], node: int) -> float:
        idx = route.index(node)
        prev = start if idx == 0 else route[idx - 1]
        nxt = self.context.depot_idx if idx == len(route) - 1 else route[idx + 1]
        return float(self._flight_edge(prev, node).risk + self._flight_edge(node, nxt).risk)

    def _alns_node_cost_ratio(self, start: int, route: list[int], node: int) -> float:
        idx = route.index(node)
        prev = start if idx == 0 else route[idx - 1]
        nxt = self.context.depot_idx if idx == len(route) - 1 else route[idx + 1]
        cost = self._leg_budget(self._flight_edge(prev, node)) + self._photo(node)
        if nxt != self.context.depot_idx:
            cost += self._leg_budget(self._flight_edge(node, nxt))
        else:
            cost += self._return_budget(node)
        return float(cost / max(self.context.target_values[node], 1e-6))


class RAWANoRiskPure(RAWARHP):
    name = "RAWA-NoRisk-Pure"
    use_risk = True
    candidate_risk_weight = 0.0
    use_risk_candidate_bias = False
    use_risk_scoring = False
    use_risk_sorting = False
    use_risk_pricing = False


class RAWABlindReserve(RAWARHP):
    name = "RAWA-BlindReserve"
    blind_reserve_pressure = 4.20


class RAWANoRisk(RAWABlindReserve):
    name = "RAWA-NoRisk"
    use_risk = False
    candidate_risk_weight = 0.0
    use_risk_candidate_bias = False


class RAWANoPackingEqBudget(RAWANoPacking):
    name = "RAWA-NoPacking-EqBudget"


class RAWANoRiskEqBudget(RAWANoRisk):
    name = "RAWA-NoRisk-EqBudget"


class RAWANoBeamEqBudget(RAWANoBeam):
    name = "RAWA-NoBeam-EqBudget"
    max_replan_expansions = 26000


class RAWANoAdaptiveSearchEqBudget(RAWANoAdaptiveSearch):
    name = "RAWA-NoAdaptiveSearch-EqBudget"


class RAWANoPackingEqTime(RAWANoPacking):
    name = "RAWA-NoPacking-EqTime"


class RAWANoPackingEqEval(RAWANoPacking):
    name = "RAWA-NoPacking-EqEval"


class RAWANoBeamEqTime(RAWANoBeam):
    name = "RAWA-NoBeam-EqTime"


class RAWANoBeamEqEval(RAWANoBeam):
    name = "RAWA-NoBeam-EqEval"


class RAWANoRiskPureEqTime(RAWANoRiskPure):
    name = "RAWA-NoRisk-Pure-EqTime"


class RAWANoRiskPureEqEval(RAWANoRiskPure):
    name = "RAWA-NoRisk-Pure-EqEval"


class RAWANoRiskEqTime(RAWANoRisk):
    name = "RAWA-NoRisk-EqTime"


class RAWANoRiskEqEval(RAWANoRisk):
    name = "RAWA-NoRisk-EqEval"


class RAWABlindReserveEqTime(RAWABlindReserve):
    name = "RAWA-BlindReserve-EqTime"


class RAWABlindReserveEqEval(RAWABlindReserve):
    name = "RAWA-BlindReserve-EqEval"


class RAWANoAdaptiveSearchEqTime(RAWANoAdaptiveSearch):
    name = "RAWA-NoAdaptiveSearch-EqTime"


class RAWANoAdaptiveSearchEqEval(RAWANoAdaptiveSearch):
    name = "RAWA-NoAdaptiveSearch-EqEval"


class RAWANoWind(RolloutRHP):
    name = "RAWA-NoWind"
    use_tail = False
    use_risk = True
    use_wind = False


PLANNER_CLASSES = {
    cls.name: cls
    for cls in [
        NearestNeighbor,
        ValuePerDistance,
        WindAwareGreedy,
        ReserveOnlyPlanner,
        FairRiskAwareGreedy,
        RAWARHP,
        FairALNSPlanner,
        RAWANoBeam,
        RAWANoRepair,
        RAWANoPacking,
        RAWANoAdaptiveSearch,
        RAWAOpenLoop,
        RAWANoRisk,
        RAWANoRiskPure,
        RAWABlindReserve,
        RAWANoPackingEqBudget,
        RAWANoRiskEqBudget,
        RAWANoBeamEqBudget,
        RAWANoAdaptiveSearchEqBudget,
        RAWANoPackingEqTime,
        RAWANoPackingEqEval,
        RAWANoBeamEqTime,
        RAWANoBeamEqEval,
        RAWANoRiskPureEqTime,
        RAWANoRiskPureEqEval,
        RAWANoRiskEqTime,
        RAWANoRiskEqEval,
        RAWABlindReserveEqTime,
        RAWABlindReserveEqEval,
        RAWANoAdaptiveSearchEqTime,
        RAWANoAdaptiveSearchEqEval,
        RAWANoWind,
    ]
}


def make_planner(name: str) -> BasePlanner:
    if name not in PLANNER_CLASSES:
        raise ValueError(f"Unknown planner {name}")
    return PLANNER_CLASSES[name]()

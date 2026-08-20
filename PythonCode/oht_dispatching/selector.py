"""Job-to-OHT candidate construction and dispatch selection."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from .config import (
    DISPATCH_COST,
    DISPATCH_FIRST_MATCH,
    DISPATCH_MODES,
)


QUEUED_JOB_STATE = 1
IDLE_OHT_STATE = 0
PICKUP_ROUTE_WINDOW = 16


class OHTDispatcher:
    """Select available OHTs and retain the latest routing-cost snapshot."""

    def __init__(self, mode: str = DISPATCH_FIRST_MATCH):
        if mode not in DISPATCH_MODES:
            raise ValueError(f"dispatch mode must be one of {DISPATCH_MODES}")
        self.mode = mode
        self.reset_episode()

    def reset_episode(self) -> None:
        self.rail_costs: dict[int, float] = {}
        self.cost_snapshot_step: int | None = None
        self.costs_ready = False
        self._job_count = 0
        self._eligible_job_count = 0
        self._zero_candidate_count = 0
        self._candidate_total = 0
        self._selected_count = 0
        self._selected_hops_total = 0
        self._cost_attempt_count = 0
        self._path_cost_count = 0
        self._path_cost_total = 0.0
        self._first_match_path_cost_total = 0.0
        self._first_match_hops_total = 0
        self._cost_saving_total = 0.0
        self._relative_cost_saving_total = 0.0
        self._candidate_cost_spread_total = 0.0
        self._multi_candidate_count = 0
        self._changed_count = 0
        self._strict_cost_improvement_count = 0
        self._margin_count = 0
        self._margin_total = 0.0
        self._cost_snapshot_fallback_count = 0
        self._invalid_cost_count = 0

    def capture_cost_snapshot(
        self,
        rail_ids,
        costs,
        step: int,
    ) -> None:
        """Freeze command-0 rail costs for a later command-6 decision."""
        if self.mode != DISPATCH_COST:
            return
        rail_ids = np.asarray(rail_ids)
        costs = np.asarray(costs, dtype=np.float64)
        if costs.shape != rail_ids.shape:
            raise RuntimeError(
                "dispatch cost snapshot shape does not match topology"
            )
        self.rail_costs = {
            int(rail_id): float(costs[row])
            for row, rail_id in enumerate(rail_ids)
        }
        self.cost_snapshot_step = int(step)
        self.costs_ready = True

    def assign(
        self,
        ohts: Mapping[int, Any],
        jobs: Iterable[Any],
    ) -> dict[int, dict[str, Any]]:
        """Assign each eligible job to one compatible, currently free OHT."""
        assignments = {}
        used = set()
        jobs = list(jobs)
        owned_oht_ids = {
            int(job.OHTId)
            for job in jobs
            if int(job.OHTId) != 0
        }
        pickup_candidates = self._index_pickup_candidates(
            ohts,
            owned_oht_ids,
        )

        for job in jobs:
            self._job_count += 1
            if job.State != QUEUED_JOB_STATE or job.OHTId != 0:
                continue
            self._eligible_job_count += 1

            candidates = self._compatible_candidates(
                job,
                pickup_candidates.get(job.FromNode, ()),
                used,
            )
            self._candidate_total += len(candidates)
            if not candidates:
                self._zero_candidate_count += 1
                continue

            selected = candidates[0]
            if self.mode == DISPATCH_COST:
                selected = self._select_lowest_cost(candidates)

            assignments[job.ID] = {
                "oht_id": selected["oht_id"],
                "pickup_path": list(selected["pickup_path"]),
            }
            used.add(selected["oht_id"])
            self._selected_count += 1
            self._selected_hops_total += selected["pickup_hops"]

        return assignments

    @staticmethod
    def _index_pickup_candidates(ohts, owned_oht_ids):
        candidates_by_rail = defaultdict(list)
        for iteration_index, (oht_id, oht) in enumerate(ohts.items()):
            if (
                oht.State != IDLE_OHT_STATE
                or oht.JobID != 0
                or oht.DispatchedCommand != 0
                or int(oht_id) in owned_oht_ids
            ):
                continue
            route_window = tuple(list(oht.RouteList)[:PICKUP_ROUTE_WINDOW])
            seen_rails = set()
            for pickup_index, rail_id in enumerate(route_window):
                if rail_id in seen_rails:
                    continue
                seen_rails.add(rail_id)
                candidates_by_rail[rail_id].append((
                    iteration_index,
                    oht_id,
                    oht,
                    route_window[: pickup_index + 1],
                    pickup_index + 1,
                ))
        return candidates_by_rail

    @staticmethod
    def _compatible_candidates(job, indexed_candidates, used):
        candidates = []
        for (
            iteration_index,
            oht_id,
            oht,
            pickup_path,
            pickup_hops,
        ) in indexed_candidates:
            carrier_ok = any(
                value in job.CarrierTypes for value in oht.CarrierTypes
            )
            area_ok = (
                oht.RunningAreaType in job.RunningAreaTyes
                or oht.RunningAreaType == 0
                or job.RunningAreaTyes[0] == 0
            )
            if not carrier_ok or not area_ok or oht_id in used:
                continue
            candidates.append({
                "oht_id": oht_id,
                "pickup_path": pickup_path,
                "pickup_hops": pickup_hops,
                "iteration_index": iteration_index,
            })
        return candidates

    def _select_lowest_cost(self, candidates):
        self._cost_attempt_count += 1
        if not self.costs_ready:
            self._cost_snapshot_fallback_count += 1
            return candidates[0]

        scored = []
        for candidate in candidates:
            score = 0.0
            for rail_id in candidate["pickup_path"]:
                cost = self.rail_costs.get(int(rail_id))
                if cost is None or not np.isfinite(cost) or cost < 0.0:
                    self._invalid_cost_count += 1
                    raise ValueError(
                        "invalid dispatch rail cost: "
                        f"mode={self.mode}, rail_id={rail_id}, cost={cost}"
                    )
                score += cost
            scored.append((score, candidate))

        first_match_score = scored[0][0]
        first_match_hops = candidates[0]["pickup_hops"]
        candidate_scores = [item[0] for item in scored]
        candidate_cost_spread = max(candidate_scores) - min(candidate_scores)
        scored.sort(key=lambda item: (
            item[0],
            item[1]["pickup_hops"],
            int(item[1]["oht_id"]),
        ))
        selected_score, selected = scored[0]
        cost_saving = max(0.0, first_match_score - selected_score)
        relative_saving = (
            cost_saving / first_match_score
            if first_match_score > 1e-12
            else 0.0
        )
        changed = selected["oht_id"] != candidates[0]["oht_id"]
        strict_improvement = cost_saving > 1e-9

        self._path_cost_count += 1
        self._path_cost_total += selected_score
        self._first_match_path_cost_total += first_match_score
        self._first_match_hops_total += first_match_hops
        self._cost_saving_total += cost_saving
        self._relative_cost_saving_total += relative_saving
        self._candidate_cost_spread_total += candidate_cost_spread
        self._changed_count += int(changed)
        self._strict_cost_improvement_count += int(strict_improvement)
        if len(scored) > 1:
            self._multi_candidate_count += 1
            self._margin_count += 1
            self._margin_total += scored[1][0] - scored[0][0]
        return selected

    @staticmethod
    def _safe_ratio(numerator, denominator):
        return (
            float(numerator) / float(denominator)
            if denominator > 0
            else 0.0
        )

    def diagnostics(self, current_step: int) -> dict[str, float]:
        eligible = self._eligible_job_count
        selected = self._selected_count
        attempts = self._cost_attempt_count
        decisions = self._path_cost_count
        snapshot_age = (
            -1.0
            if self.cost_snapshot_step is None
            else float(max(0, int(current_step) - self.cost_snapshot_step))
        )
        changed_ratio = self._safe_ratio(self._changed_count, decisions)
        selection_margin = self._safe_ratio(
            self._margin_total,
            self._margin_count,
        )
        return {
            "dispatch/mode_first_match": float(
                self.mode == DISPATCH_FIRST_MATCH
            ),
            "dispatch/mode_neutral_path_cost": 0.0,
            "dispatch/mode_live_td7_path_cost": float(
                self.mode == DISPATCH_COST
            ),
            "dispatch/cost_mode_active": float(self.mode == DISPATCH_COST),
            "dispatch/cost_snapshot_ready": float(self.costs_ready),
            "dispatch/cost_snapshot_ready_ratio": self._safe_ratio(
                decisions,
                attempts,
            ),
            "dispatch/cost_snapshot_age_steps": snapshot_age,
            "dispatch/eligible_job_count_total": float(eligible),
            "dispatch/candidate_count_mean": self._safe_ratio(
                self._candidate_total,
                self._job_count,
            ),
            "dispatch/candidate_count_mean_per_eligible_job": self._safe_ratio(
                self._candidate_total,
                eligible,
            ),
            "dispatch/zero_candidate_ratio_per_eligible_job": self._safe_ratio(
                self._zero_candidate_count,
                eligible,
            ),
            "dispatch/selected_ratio_per_eligible_job": self._safe_ratio(
                selected,
                eligible,
            ),
            "dispatch/selected_pickup_hops_mean": self._safe_ratio(
                self._selected_hops_total,
                selected,
            ),
            "dispatch/first_match_pickup_hops_mean": self._safe_ratio(
                self._first_match_hops_total,
                decisions,
            ),
            "dispatch/first_match_path_cost_mean": self._safe_ratio(
                self._first_match_path_cost_total,
                decisions,
            ),
            "dispatch/selected_path_cost_mean": self._safe_ratio(
                self._path_cost_total,
                decisions,
            ),
            "dispatch/cost_saving_vs_first_mean": self._safe_ratio(
                self._cost_saving_total,
                decisions,
            ),
            "dispatch/cost_saving_vs_first_ratio_mean": self._safe_ratio(
                self._relative_cost_saving_total,
                decisions,
            ),
            "dispatch/candidate_cost_spread_mean": self._safe_ratio(
                self._candidate_cost_spread_total,
                decisions,
            ),
            "dispatch/multi_candidate_ratio": self._safe_ratio(
                self._multi_candidate_count,
                decisions,
            ),
            "dispatch/changed_from_first_ratio": changed_ratio,
            "dispatch/strict_cost_improvement_ratio": self._safe_ratio(
                self._strict_cost_improvement_count,
                decisions,
            ),
            "dispatch/cost_snapshot_fallback_ratio": self._safe_ratio(
                self._cost_snapshot_fallback_count,
                attempts,
            ),
            "dispatch/selection_margin_mean": selection_margin,
            # Compatibility aliases for existing dashboards.
            "dispatch/selection_changed_from_first_match_ratio": changed_ratio,
            "dispatch/best_second_margin_mean": selection_margin,
            "dispatch/cost_snapshot_fallback_count": float(
                self._cost_snapshot_fallback_count
            ),
            "dispatch/invalid_cost_count": float(self._invalid_cost_count),
        }

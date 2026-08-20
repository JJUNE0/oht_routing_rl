"""Controlled-center reward construction for contextual rail control."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from simulator.oht import OHTState
from oht_routing.mdp.reward.rail_cycle import (
    ContextualRailRewardMixin,
    OHTCycleTracker,
    RailPassTemporalTracker,
)
from oht_routing.mdp.action import ACTION_MODES, REGION_B_RL
from oht_routing.utils.reward_diagnostic import RewardDiagnosticWriter
from oht_routing.mdp.topology import ContextualTopology
from oht_routing.mdp.reward.config import (
    RAIL_REWARD_FREE_FLOW_NEUTRAL_2,
    REWARD_VERSION,
    TAT_PENALTY_START,
    RewardContract,
    canonical_reward_version,
    reward_contract,
)


class ContextualRewardError(RuntimeError):
    """Raised when reward inputs violate the controlled-center contract."""


@dataclass(frozen=True)
class ContextualRewardConfig:
    reward_version: str = REWARD_VERSION
    global_alpha: float = 0.5
    local_alpha: float = 0.5
    rail_tat_weight: float = 30.0
    rail_free_flow_neutral_ratio: float = 2.0
    action_mode: str = REGION_B_RL
    smooth_b_rl_weight: float = 0.25
    smooth_exp_residual_weight: float = 0.5
    tat_weight: float = 11.0
    op_weight: float = 4.0
    backlog_weight: float = 0.0004
    backlog_growth_enabled: bool = True
    backlog_growth_horizon: int = 300
    backlog_growth_scale: float = 30.0
    backlog_growth_weight: float = 0.16
    idle_reserve_target: float = 200.0
    idle_reserve_scale: float = 50.0
    idle_reserve_weight: float = 0.20
    local_oht_weight: float = 0.3
    local_predicted_oht_weight: float = 0.075
    local_stop_weight: float = 0.3
    local_idle_weight: float = 0.0
    local_capacity_weight: float = 0.1
    use_tat: bool = True
    use_op: bool = True
    use_backlog: bool = True
    tat_reference: float = 165.0
    op_reference: float = 0.80
    local_reward_scale: float = 2.0
    rail_tat_clip: float | None = 1.0

    @classmethod
    def for_version(
        cls,
        reward_version: str = REWARD_VERSION,
        *,
        action_mode: str = REGION_B_RL,
    ) -> "ContextualRewardConfig":
        """Build the single immutable Reward N profile."""
        canonical_reward_version(reward_version)
        return cls(action_mode=action_mode)

    @property
    def contract(self) -> RewardContract:
        return reward_contract(self.reward_version)

    @property
    def rail_reward_mode(self) -> str:
        """Constant diagnostic label; Reward N has no rail-mode selector."""
        return RAIL_REWARD_FREE_FLOW_NEUTRAL_2

    def __post_init__(self):
        canonical_version = canonical_reward_version(self.reward_version)
        object.__setattr__(self, "reward_version", canonical_version)
        numeric = (
            self.global_alpha, self.local_alpha, self.rail_tat_weight,
            self.smooth_b_rl_weight, self.smooth_exp_residual_weight,
            self.tat_weight, self.op_weight,
            self.backlog_weight, self.tat_reference, self.op_reference,
            self.backlog_growth_scale, self.backlog_growth_weight,
            self.idle_reserve_target, self.idle_reserve_scale,
            self.idle_reserve_weight,
            self.local_reward_scale, self.local_oht_weight,
            self.local_predicted_oht_weight, self.local_stop_weight,
            self.local_idle_weight, self.local_capacity_weight,
        )
        if not np.isfinite(numeric).all():
            raise ValueError("reward config contains NaN or Inf")
        if self.tat_reference <= 0:
            raise ValueError("tat_reference must be positive")
        if self.tat_weight < 0:
            raise ValueError("tat_weight must be non-negative")
        if self.local_reward_scale <= 0:
            raise ValueError("local_reward_scale must be positive")
        if self.backlog_weight < 0:
            raise ValueError("backlog_weight must be non-negative")
        if (
            isinstance(self.backlog_growth_horizon, bool)
            or not isinstance(self.backlog_growth_horizon, (int, np.integer))
            or self.backlog_growth_horizon <= 0
        ):
            raise ValueError("backlog_growth_horizon must be a positive integer")
        if self.backlog_growth_scale <= 0:
            raise ValueError("backlog_growth_scale must be positive")
        if self.backlog_growth_weight < 0:
            raise ValueError("backlog_growth_weight must be non-negative")
        if self.idle_reserve_scale <= 0:
            raise ValueError("idle_reserve_scale must be positive")
        if self.idle_reserve_target < 0 or self.idle_reserve_weight < 0:
            raise ValueError("idle reserve target/weight must be non-negative")
        if min(
            self.local_oht_weight,
            self.local_predicted_oht_weight,
            self.local_stop_weight,
            self.local_idle_weight,
            self.local_capacity_weight,
        ) < 0:
            raise ValueError("local reward weights must be non-negative")
        if self.rail_tat_weight < 0:
            raise ValueError("rail_tat_weight must be non-negative")
        if (
            not np.isfinite(self.rail_free_flow_neutral_ratio)
            or self.rail_free_flow_neutral_ratio <= 0
        ):
            raise ValueError(
                "rail_free_flow_neutral_ratio must be finite and positive"
            )
        if self.rail_tat_clip is not None and (
            not np.isfinite(self.rail_tat_clip) or self.rail_tat_clip <= 0
        ):
            raise ValueError("rail_tat_clip must be positive or None")
        if not 0.0 <= self.op_reference <= 1.0:
            raise ValueError("op_reference must be in [0, 1]")
        if self.action_mode not in ACTION_MODES:
            raise ValueError(f"action_mode must be one of {ACTION_MODES}")
        if (
            self.smooth_b_rl_weight < 0
            or self.smooth_exp_residual_weight < 0
        ):
            raise ValueError("smooth weights must be non-negative")


@dataclass(frozen=True)
class ControlledRewardBatch:
    total: np.ndarray
    global_raw: float
    global_normalized: float
    global_component: float
    local_raw: np.ndarray
    local_normalized: np.ndarray
    local_component: np.ndarray
    rail_reward_raw: np.ndarray
    rail_reward_weighted_preclip: np.ndarray
    rail_reward_postclip: np.ndarray
    rail_tat_event_count: int
    smooth_control_delta: np.ndarray
    smooth_penalty: np.ndarray
    controlled_rail_ids: np.ndarray
    total_tat_level: float
    tat_signal_available: float
    tat_error: float
    tat_raw: float
    completed_episode: float
    completed_delta: float
    op_rate: float
    op_reference: float
    op_error: float
    op_delta: float
    op_raw: float
    backlog: float
    backlog_raw: float
    backlog_growth_signal: float
    backlog_growth_raw: float
    idle_oht_count: float
    idle_reserve_signal: float
    idle_reserve_raw: float
    local_oht_raw: np.ndarray
    local_predicted_raw: np.ndarray
    local_stop_raw: np.ndarray
    local_idle_raw: np.ndarray
    local_capacity_raw: np.ndarray
    waiting: float
    queued: float
    idle_oht_observation: np.ndarray
    smooth_weight_effective: float
    env_step: int
    episode_id: int
    terminal_penalty: float = 0.0

    @property
    def local_scaled(self):
        """Compatibility alias for pre-Reward-S fixed-scale consumers."""
        return self.local_normalized

    @property
    def rail_tat_raw(self):
        """Deprecated penalty-direction alias."""
        return -self.rail_reward_raw

    @property
    def rail_tat_weighted_preclip(self):
        """Deprecated penalty-direction alias."""
        return -self.rail_reward_weighted_preclip

    @property
    def rail_tat_penalty(self):
        """Deprecated penalty-direction alias."""
        return -self.rail_reward_postclip


class ContextualRewardBuilder(ContextualRailRewardMixin):
    """Build one reward row per controlled rail from the post-action snapshot."""

    def __init__(
        self,
        topology: ContextualTopology,
        config: ContextualRewardConfig | None = None,
        *,
        completion_diagnostic_path: str | Path | None = None,
        global_step_provider: Callable[[], int] | None = None,
        completion_diagnostic_max_global_step: int | None = None,
        reward_diagnostic_writer: RewardDiagnosticWriter | None = None,
    ):
        self.topology = topology
        self.config = config or ContextualRewardConfig()
        self._controlled_rail_id_set = frozenset(
            int(value) for value in self.topology.controlled_rail_ids
        )
        self.completion_diagnostic_path = (
            Path(completion_diagnostic_path)
            if completion_diagnostic_path is not None
            else None
        )
        self._global_step_provider = global_step_provider
        self.reward_diagnostic_writer = reward_diagnostic_writer
        self.completion_diagnostic_max_global_step = (
            int(completion_diagnostic_max_global_step)
            if completion_diagnostic_max_global_step is not None
            else None
        )
        if (
            self.completion_diagnostic_max_global_step is not None
            and self.completion_diagnostic_max_global_step < 0
        ):
            raise ValueError(
                "completion_diagnostic_max_global_step must be non-negative"
            )
        self.reward_steps = 0
        self._oht_cycle_trackers: dict[int, OHTCycleTracker] = {}
        self._rail_pass_trackers: dict[int, RailPassTemporalTracker] = {}
        self._reset_temporal()

    @property
    def reward_version(self) -> str:
        return self.config.contract.version

    def _reset_temporal(self) -> None:
        self._total_completed_jobs = 0.0
        self._prev_op_rate: float | None = None
        self._backlog_history = deque(
            maxlen=int(self.config.backlog_growth_horizon) + 1
        )
        self._recent_route_ratios = deque(maxlen=1_000)
        self._recent_route_negative = deque(maxlen=1_000)
        self._route_ratio_sample_added = False
        self._last_global_terms: dict[str, float] = {}
        self._last_local_terms: dict[str, np.ndarray] = {}
        self._last_rail_tat_event_count = 0
        self._last_rail_tat_cycle_count = 0
        self._last_rail_tat_controlled_assignment_count = 0
        self._last_rail_tat_uncontrolled_assignment_count = 0
        self._oht_cycle_trackers.clear()
        self._rail_pass_trackers.clear()

    def reset_episode(self) -> None:
        """Clear only episode-temporal state; training statistics persist."""
        self._reset_temporal()

    def _current_global_step(self) -> int:
        return (
            int(self._global_step_provider())
            if self._global_step_provider is not None else self.reward_steps
        )

    def _cycle_diagnostic_active(self) -> bool:
        return bool(
            self.reward_diagnostic_writer is not None
            and self.reward_diagnostic_writer.window_name(
                self._current_global_step()
            ) is not None
        )

    @staticmethod
    def _job_backlog(pclient) -> tuple[float, float]:
        waiting = float(getattr(pclient, "WaitingCommandCount", 0) or 0)
        queued = float(getattr(pclient, "QueuedCommandCount", 0) or 0)
        return waiting, queued

    def _global_raw(self, pclient) -> float:
        """Compute the locked Reward N global term from total TAT."""
        cfg = self.config
        cur_tat = float(getattr(pclient, "TotalTat"))
        cur_op = float(getattr(pclient, "TotalOhtOperationRate"))
        completed_delta = float(
            getattr(pclient, "CompletedCommandCount", 0) or 0
        )
        waiting, queued = self._job_backlog(pclient)
        if not np.isfinite((
            cur_tat, cur_op, completed_delta, waiting, queued
        )).all():
            raise ContextualRewardError(
                "global reward input contains NaN or Inf"
            )
        self._total_completed_jobs += completed_delta
        tat_signal_available = cur_tat > 0.0
        tat_excess = (
            max(0.0, cur_tat - TAT_PENALTY_START)
            if tat_signal_available else 0.0
        )
        tat_error = -tat_excess / cfg.tat_reference
        tat_raw = cfg.tat_weight * tat_error if cfg.use_tat else 0.0
        op_delta = (
            float(self._prev_op_rate) - cur_op
            if self._prev_op_rate is not None else 0.0
        )
        op_error = cfg.op_reference - cur_op
        op_raw = cfg.op_weight * op_error if cfg.use_op else 0.0
        backlog = waiting + queued
        backlog_raw = -cfg.backlog_weight * backlog if cfg.use_backlog else 0.0
        if (
            cfg.backlog_growth_enabled
            and len(self._backlog_history) >= cfg.backlog_growth_horizon
        ):
            backlog_delta = (
                backlog - self._backlog_history[-cfg.backlog_growth_horizon]
            )
            backlog_growth_signal = float(np.clip(
                max(0.0, backlog_delta) / cfg.backlog_growth_scale,
                0.0,
                1.0,
            ))
        else:
            backlog_delta = 0.0
            backlog_growth_signal = 0.0
        self._backlog_history.append(float(backlog))
        backlog_growth_raw = (
            -cfg.backlog_growth_weight * backlog_growth_signal
            if cfg.backlog_growth_enabled else 0.0
        )
        idle_oht_count = 0.0
        for oht in getattr(pclient, "OHT_DIC", {}).values():
            try:
                state = int(getattr(oht, "State", OHTState.NULL))
            except (TypeError, ValueError, OverflowError) as error:
                raise ContextualRewardError(
                    "idle OHT state contains NaN or Inf"
                ) from error
            idle_oht_count += float(state == int(OHTState.IDLE))
        idle_reserve_signal = float(np.clip(
            max(0.0, cfg.idle_reserve_target - idle_oht_count)
            / cfg.idle_reserve_scale,
            0.0,
            1.0,
        ))
        idle_reserve_raw = -cfg.idle_reserve_weight * idle_reserve_signal
        raw = (
            tat_raw
            + op_raw
            + backlog_raw
            + backlog_growth_raw
            + idle_reserve_raw
        )
        self._last_global_terms = {
            "tat_error": tat_error,
            "tat_signal_available": float(tat_signal_available),
            "tat_raw": tat_raw,
            "total_tat": cur_tat,
            "completed_episode": self._total_completed_jobs,
            "completed_delta": completed_delta,
            "op_rate": cur_op,
            "op_reference": cfg.op_reference,
            "op_error": op_error,
            "op_delta": op_delta,
            "op_raw": op_raw,
            "backlog": backlog,
            "backlog_raw": backlog_raw,
            "backlog_delta": backlog_delta,
            "backlog_growth_signal": backlog_growth_signal,
            "backlog_growth_raw": backlog_growth_raw,
            "idle_oht_count": idle_oht_count,
            "idle_reserve_signal": idle_reserve_signal,
            "idle_reserve_raw": idle_reserve_raw,
            "waiting": waiting,
            "queued": queued,
        }
        self._prev_op_rate = cur_op
        return float(raw)

    def _local_raw(self, pclient) -> np.ndarray:
        result = np.empty(len(self.topology.controlled_rail_ids), dtype=np.float64)
        local_values = {
            name: np.empty(len(self.topology.controlled_rail_ids), dtype=np.float64)
            for name in ("oht", "predicted", "stop", "idle", "capacity")
        }
        idle_observation = np.empty(
            len(self.topology.controlled_rail_ids), dtype=np.float64
        )
        rails = getattr(pclient, "RAILLINE_DIC", {})
        ohts = getattr(pclient, "OHT_DIC", {})
        for row, rail_id_value in enumerate(self.topology.controlled_rail_ids):
            rail_id = int(rail_id_value)
            if rail_id not in rails:
                raise ContextualRewardError(f"missing controlled rail {rail_id}")
            rail = rails[rail_id]
            rail_ohts = [
                ohts[oht_id] for oht_id in getattr(rail, "OhtList")
                if oht_id in ohts
            ]
            avg_stop = (
                float(np.mean([float(getattr(oht, "StopTime")) for oht in rail_ohts]))
                if rail_ohts else 0.0
            )
            oht_count = len(getattr(rail, "OhtList"))
            capacity = oht_count / max(1, int(getattr(rail, "PortCount")) + 1)
            result[row] = 0.0
            local_values["oht"][row] = -self.config.local_oht_weight * oht_count
            local_values["predicted"][row] = (
                -self.config.local_predicted_oht_weight
                * float(getattr(rail, "PredictedOHTCount"))
            )
            local_values["stop"][row] = -self.config.local_stop_weight * avg_stop
            idle_observation[row] = float(getattr(rail, "IdleOHTCount"))
            local_values["idle"][row] = (
                -self.config.local_idle_weight * idle_observation[row]
            )
            local_values["capacity"][row] = (
                -self.config.local_capacity_weight * capacity
            )
            result[row] = sum(values[row] for values in local_values.values())
        if not np.isfinite(result).all():
            raise ContextualRewardError("local reward input contains NaN or Inf")
        self._last_local_terms = local_values
        self._last_idle_oht_observation = idle_observation
        return result

    def build(
        self,
        pclient,
        *,
        applied_action,
        previous_applied_action,
        env_step: int,
        episode_id: int,
    ) -> ControlledRewardBatch:
        count = len(self.topology.controlled_rail_ids)
        action = np.asarray(applied_action, dtype=np.float64).reshape(-1)
        if action.shape != (count,) or not np.isfinite(action).all():
            raise ContextualRewardError("applied_action must be finite [controlled]")
        if previous_applied_action is None:
            previous = action
        else:
            previous = np.asarray(previous_applied_action, dtype=np.float64).reshape(-1)
            if previous.shape != (count,) or not np.isfinite(previous).all():
                raise ContextualRewardError(
                    "previous_applied_action must be finite [controlled]"
                )

        global_raw = self._global_raw(pclient)
        local_raw = self._local_raw(pclient)
        global_normalized = float(global_raw)
        local_normalized = local_raw / self.config.local_reward_scale
        self.reward_steps += 1
        global_component = self.config.global_alpha * global_normalized
        local_component = self.config.local_alpha * local_normalized
        rail_reward_raw = self._rail_reward_raw(
            pclient,
            env_step=env_step,
            episode_id=episode_id,
        )
        (
            rail_reward_weighted_preclip,
            rail_reward_postclip,
        ) = self._scale_rail_reward(rail_reward_raw)
        if self.config.action_mode == REGION_B_RL:
            control_delta = np.abs(
                (0.5 + 0.5 * action) - (0.5 + 0.5 * previous)
            )
            smooth_weight = self.config.smooth_b_rl_weight
        else:
            control_delta = np.abs(action - previous)
            smooth_weight = self.config.smooth_exp_residual_weight
        smooth = smooth_weight * control_delta
        total = (
            global_component + local_component + rail_reward_postclip - smooth
        )
        arrays = {
            "total": total, "local_raw": local_raw,
            "local_normalized": local_normalized,
            "local_component": local_component,
            "rail_reward_raw": rail_reward_raw,
            "rail_reward_weighted_preclip": rail_reward_weighted_preclip,
            "rail_reward_postclip": rail_reward_postclip,
            "smooth_control_delta": control_delta,
            "smooth_penalty": smooth,
            "controlled_rail_ids": self.topology.controlled_rail_ids,
            "local_oht_raw": self._last_local_terms["oht"],
            "local_predicted_raw": self._last_local_terms["predicted"],
            "local_stop_raw": self._last_local_terms["stop"],
            "local_idle_raw": self._last_local_terms["idle"],
            "local_capacity_raw": self._last_local_terms["capacity"],
            "idle_oht_observation": self._last_idle_oht_observation,
        }
        frozen_arrays = {}
        for name, value in arrays.items():
            array = np.ascontiguousarray(np.asarray(value).copy())
            if array.shape != (count,) or not np.isfinite(array).all():
                raise ContextualRewardError(f"{name} is not finite [{count}]")
            array.setflags(write=False)
            frozen_arrays[name] = array
        return ControlledRewardBatch(
            **frozen_arrays,
            rail_tat_event_count=int(self._last_rail_tat_event_count),
            global_raw=float(global_raw),
            global_normalized=float(global_normalized),
            global_component=float(global_component),
            total_tat_level=float(self._last_global_terms["total_tat"]),
            tat_signal_available=float(
                self._last_global_terms["tat_signal_available"]
            ),
            tat_error=float(self._last_global_terms["tat_error"]),
            tat_raw=float(self._last_global_terms["tat_raw"]),
            completed_episode=float(
                self._last_global_terms["completed_episode"]
            ),
            completed_delta=float(self._last_global_terms["completed_delta"]),
            op_rate=float(self._last_global_terms["op_rate"]),
            op_reference=float(self._last_global_terms["op_reference"]),
            op_error=float(self._last_global_terms["op_error"]),
            op_delta=float(self._last_global_terms["op_delta"]),
            op_raw=float(self._last_global_terms["op_raw"]),
            backlog=float(self._last_global_terms["backlog"]),
            backlog_raw=float(self._last_global_terms["backlog_raw"]),
            backlog_growth_signal=float(
                self._last_global_terms["backlog_growth_signal"]
            ),
            backlog_growth_raw=float(
                self._last_global_terms["backlog_growth_raw"]
            ),
            idle_oht_count=float(self._last_global_terms["idle_oht_count"]),
            idle_reserve_signal=float(
                self._last_global_terms["idle_reserve_signal"]
            ),
            idle_reserve_raw=float(
                self._last_global_terms["idle_reserve_raw"]
            ),
            waiting=float(self._last_global_terms["waiting"]),
            queued=float(self._last_global_terms["queued"]),
            smooth_weight_effective=float(smooth_weight),
            env_step=int(env_step),
            episode_id=int(episode_id),
        )

    def route_ratio_diagnostics(self) -> dict[str, float]:
        """Return a bounded rolling tail summary of completed route cycles."""
        prefix = "lead/route_ratio/"
        values = np.asarray(self._recent_route_ratios, dtype=np.float64)
        available = bool(values.size and self._route_ratio_sample_added)
        result = {
            prefix + "available": float(available),
            prefix + "mean": float(values.mean()) if available else 0.0,
            prefix + "max": float(values.max()) if available else 0.0,
            prefix + "ratio_gt_2": float(np.mean(values > 2.0))
            if available else 0.0,
            prefix + "negative_reward_cycle_ratio": float(np.mean(
                self._recent_route_negative
            )) if available else 0.0,
        }
        for percentile in (50, 75, 90, 95, 99):
            result[prefix + f"p{percentile}"] = (
                float(np.percentile(values, percentile))
                if available else 0.0
            )
        return result

    def diagnostics(self, batch: ControlledRewardBatch) -> dict[str, float]:
        rail_tat_sum = float(batch.rail_tat_penalty.sum())
        rail_tat_vector_mean = float(batch.rail_tat_penalty.mean())
        rail_tat_event_mean = (
            rail_tat_sum / batch.rail_tat_event_count
            if batch.rail_tat_event_count > 0 else 0.0
        )
        global_contribution = np.full_like(
            batch.local_component, batch.global_component
        )
        rail_tat_contribution = -batch.rail_tat_penalty
        smooth_contribution = -batch.smooth_penalty
        contributions = (
            global_contribution,
            batch.local_component,
            rail_tat_contribution,
            smooth_contribution,
        )
        abs_means = np.asarray(
            [float(np.mean(np.abs(value))) for value in contributions],
            dtype=np.float64,
        )
        abs_denominator = float(abs_means.sum()) + np.finfo(np.float64).eps
        abs_shares = abs_means / abs_denominator
        rail_nonzero = np.abs(batch.rail_reward_postclip[
            ~np.isclose(batch.rail_reward_postclip, 0.0)
        ])
        rail_active_representative = (
            float(np.median(rail_nonzero)) if rail_nonzero.size else 0.0
        )
        global_representative = float(abs(batch.global_component))
        local_representative = float(np.mean(np.abs(batch.local_component)))
        scale_values = np.asarray([
            global_representative,
            local_representative,
            rail_active_representative,
        ])
        positive_scales = scale_values[scale_values > 0.0]
        scale_mean = float(positive_scales.mean()) if positive_scales.size else 0.0
        balance_error = (
            float(np.mean(np.abs(positive_scales - scale_mean)) / scale_mean)
            if scale_mean > 0.0 else 0.0
        )
        tat_contribution = self.config.global_alpha * batch.tat_raw
        backlog_contribution = (
            self.config.global_alpha * batch.backlog_raw
        )
        budget_abs = {
            "tat_raw": abs(batch.tat_raw),
            "backlog_raw": abs(batch.backlog_raw),
            "tat": abs(tat_contribution),
            "backlog": abs(backlog_contribution),
            "global": float(abs(batch.global_component)),
            "local": float(np.mean(np.abs(batch.local_component))),
            "rail": float(np.mean(np.abs(batch.rail_reward_postclip))),
            "smooth": float(np.mean(np.abs(batch.smooth_penalty))),
        }
        final_budget = {
            name: budget_abs[name]
            for name in ("tat", "backlog", "local", "rail", "smooth")
        }
        budget_total = sum(final_budget.values())
        budget_shares = {
            name: value / budget_total if budget_total > 0.0 else 0.0
            for name, value in final_budget.items()
        }
        result = {
            "reward/global_raw": batch.global_raw,
            "reward/global/total_tat": batch.total_tat_level,
            "reward/global/tat_reference": self.config.tat_reference,
            "reward/global/tat_error": batch.tat_error,
            "reward/global/tat_weight": self.config.tat_weight,
            "reward/global/tat_signal_available": (
                batch.tat_signal_available
            ),
            "reward/global/tat_raw": batch.tat_raw,
            "reward/global/tat_component": batch.tat_raw,
            "reward/global/tat_component_raw": batch.tat_raw,
            "reward/global/completed_episode": batch.completed_episode,
            "reward/global/completed_delta": batch.completed_delta,
            "reward/global/op_rate": batch.op_rate,
            "reward/global/op_reference": batch.op_reference,
            "reward/global/op_error": batch.op_error,
            "reward/global/op_delta": batch.op_delta,
            "reward/global/op_weight": self.config.op_weight,
            "reward/global/op_raw": batch.op_raw,
            "reward/global/backlog": batch.backlog,
            "reward/global/backlog_weight": self.config.backlog_weight,
            "reward/global/backlog_raw": batch.backlog_raw,
            "reward/global/backlog_component_raw": batch.backlog_raw,
            "reward/global/backlog_growth_enabled": float(
                self.config.backlog_growth_enabled
            ),
            "reward/global/backlog_growth_horizon": float(
                self.config.backlog_growth_horizon
            ),
            "reward/global/backlog_growth_scale": (
                self.config.backlog_growth_scale
            ),
            "reward/global/backlog_growth_weight": (
                self.config.backlog_growth_weight
            ),
            "reward/global/backlog_growth_signal": (
                batch.backlog_growth_signal
            ),
            "reward/global/backlog_growth_raw": batch.backlog_growth_raw,
            "reward/global/backlog_growth_component": (
                self.config.global_alpha * batch.backlog_growth_raw
            ),
            "reward/global/idle_oht_count": batch.idle_oht_count,
            "reward/global/idle_reserve_target": (
                self.config.idle_reserve_target
            ),
            "reward/global/idle_reserve_scale": (
                self.config.idle_reserve_scale
            ),
            "reward/global/idle_reserve_weight": (
                self.config.idle_reserve_weight
            ),
            "reward/global/idle_reserve_signal": batch.idle_reserve_signal,
            "reward/global/idle_reserve_raw": batch.idle_reserve_raw,
            "reward/global/idle_reserve_component": (
                self.config.global_alpha * batch.idle_reserve_raw
            ),
            "reward/global/backlog_component": batch.backlog_raw,
            "reward/global/op_component": batch.op_raw,
            "reward/global/global_component": batch.global_component,
            "reward/global/raw": batch.global_raw,
            "reward/global/normalized": batch.global_normalized,
            "reward/global/component": batch.global_component,
            "reward/global/raw_sum": batch.global_raw,
            "reward/global/raw_decomposition_error": abs(
                batch.global_raw
                - batch.tat_raw
                - batch.op_raw
                - batch.backlog_raw
                - batch.backlog_growth_raw
                - batch.idle_reserve_raw
            ),
            "reward/global_component": batch.global_component,
            "reward/total_tat_level": batch.total_tat_level,
            "reward/local_raw_mean": float(batch.local_raw.mean()),
            "reward/local_raw_std": float(batch.local_raw.std()),
            "reward/local_scaled_mean": float(batch.local_normalized.mean()),
            "reward/local_scaled_std": float(batch.local_normalized.std()),
            "reward/local_component_mean": float(batch.local_component.mean()),
            "reward/local_component_std": float(batch.local_component.std()),
            "reward/local/raw_mean": float(batch.local_raw.mean()),
            "reward/local/raw_std": float(batch.local_raw.std()),
            "reward/local/normalized_mean": float(
                batch.local_normalized.mean()
            ),
            "reward/local/normalized_std": float(
                batch.local_normalized.std()
            ),
            "reward/local/component_mean": float(batch.local_component.mean()),
            "reward/local/component_std": float(batch.local_component.std()),
            "reward/local/predicted_oht_weight": (
                self.config.local_predicted_oht_weight
            ),
            "reward/local/predicted_oht_component_abs_mean": float(
                np.mean(np.abs(batch.local_predicted_raw))
            ),
            "reward/local/predicted_raw_abs_mean": float(
                np.mean(np.abs(batch.local_predicted_raw))
            ),
            "reward/local/local_reward_scale": self.config.local_reward_scale,
            "reward/local/local_component_abs_mean": local_representative,
            "reward/rail_tat_raw_mean": float(batch.rail_tat_raw.mean()),
            "reward/rail_tat_weighted_preclip_mean": float(
                batch.rail_tat_weighted_preclip.mean()
            ),
            "reward/rail_tat_penalty_mean": rail_tat_vector_mean,
            "reward/rail_tat_mean": float(batch.rail_reward_postclip.mean()),
            "reward/rail_tat_penalty_max": float(batch.rail_tat_penalty.max()),
            "reward/rail_tat_event_count": float(
                batch.rail_tat_event_count
            ),
            "reward/rail_tat_sum": rail_tat_sum,
            "reward/rail_tat_vector_mean": rail_tat_vector_mean,
            "reward/rail_tat_event_mean": float(rail_tat_event_mean),
            "reward/rail_tat_cycle_count": float(
                self._last_rail_tat_cycle_count
            ),
            "reward/rail_tat_controlled_assignment_count": float(
                self._last_rail_tat_controlled_assignment_count
            ),
            "reward/rail_tat_uncontrolled_assignment_count": float(
                self._last_rail_tat_uncontrolled_assignment_count
            ),
            "reward/smooth_penalty_mean": float(batch.smooth_penalty.mean()),
            "reward/smooth_penalty_max": float(batch.smooth_penalty.max()),
            "reward/smooth_control_delta_mean": float(
                batch.smooth_control_delta.mean()
            ),
            "reward/smooth_control_delta_max": float(
                batch.smooth_control_delta.max()
            ),
            "reward/total_mean": float(batch.total.mean()),
            "reward/total_std": float(batch.total.std()),
            "reward/total_min": float(batch.total.min()),
            "reward/total_max": float(batch.total.max()),
            "reward/terminal_penalty": float(batch.terminal_penalty),
            "reward/finite_ratio": float(np.isfinite(batch.total).mean()),
            "reward/local/raw_decomposition_error_max": float(np.max(np.abs(
                batch.local_raw
                - batch.local_oht_raw
                - batch.local_predicted_raw
                - batch.local_stop_raw
                - batch.local_idle_raw
                - batch.local_capacity_raw
            ))),
            "reward/contribution/global_mean": float(
                global_contribution.mean()
            ),
            "reward/contribution/tat_abs": budget_abs["tat"],
            "reward/contribution/backlog_abs": budget_abs["backlog"],
            "reward/contribution/local_abs": budget_abs["local"],
            "reward/contribution/local_mean": float(
                batch.local_component.mean()
            ),
            "reward/contribution/rail_tat_mean": float(
                rail_tat_contribution.mean()
            ),
            "reward/contribution/smooth_mean": float(
                smooth_contribution.mean()
            ),
            "reward/contribution/total_mean": float(batch.total.mean()),
            "reward/contribution/sum_error": abs(float(
                batch.total.mean() - batch.terminal_penalty
                - sum(float(value.mean()) for value in contributions)
            )),
            "reward/scale/global_abs_mean": float(abs_means[0]),
            "reward/scale/local_abs_mean": float(abs_means[1]),
            "reward/scale/rail_tat_abs_mean": float(abs_means[2]),
            "reward/scale/smooth_abs_mean": float(abs_means[3]),
            "reward/scale/total_abs_mean": float(
                np.mean(np.abs(batch.total))
            ),
            "reward/scale/global_abs_share": float(abs_shares[0]),
            "reward/scale/local_abs_share": float(abs_shares[1]),
            "reward/scale/rail_tat_abs_share": float(abs_shares[2]),
            "reward/scale/smooth_abs_share": float(abs_shares[3]),
            "reward/scale/abs_share_sum_error": abs(
                1.0 - float(abs_shares.sum())
            ),
            "reward/scale/global_representative": global_representative,
            "reward/scale/local_representative": local_representative,
            "reward/scale/rail_active_representative": (
                rail_active_representative
            ),
            "reward/scale/global_to_local": (
                global_representative / local_representative
                if local_representative > 0.0 else 0.0
            ),
            "reward/scale/global_to_rail": (
                global_representative / rail_active_representative
                if rail_active_representative > 0.0 else 0.0
            ),
            "reward/scale/local_to_rail": (
                local_representative / rail_active_representative
                if rail_active_representative > 0.0 else 0.0
            ),
            "reward/scale/main_balance_error": balance_error,
            "reward/rail/neutral_ratio": (
                self.config.rail_free_flow_neutral_ratio
            ),
            "reward/rail/weight": self.config.rail_tat_weight,
            "reward/rail/clip": float(self.config.rail_tat_clip or 0.0),
            "reward/rail/nonzero_raw_abs_mean": float(np.mean(np.abs(
                batch.rail_reward_raw[~np.isclose(batch.rail_reward_raw, 0.0)]
            ))) if np.any(~np.isclose(batch.rail_reward_raw, 0.0)) else 0.0,
            "reward/rail/nonzero_weighted_abs_mean": float(np.mean(np.abs(
                batch.rail_reward_weighted_preclip[
                    ~np.isclose(batch.rail_reward_weighted_preclip, 0.0)
                ]
            ))) if np.any(~np.isclose(
                batch.rail_reward_weighted_preclip, 0.0
            )) else 0.0,
            "reward/rail/nonzero_postclip_abs_mean": (
                float(rail_nonzero.mean()) if rail_nonzero.size else 0.0
            ),
            "reward/config/global_alpha": self.config.global_alpha,
            "reward/config/local_alpha": self.config.local_alpha,
            "reward/config/local_reward_scale": self.config.local_reward_scale,
            "reward/config/rail_tat_weight": self.config.rail_tat_weight,
            "reward/config/rail_tat_clip": float(
                self.config.rail_tat_clip
                if self.config.rail_tat_clip is not None else 0.0
            ),
            "reward/config/smooth_weight_effective": (
                batch.smooth_weight_effective
            ),
            "reward/budget/tat_raw_abs": budget_abs["tat_raw"],
            "reward/budget/backlog_raw_abs": budget_abs["backlog_raw"],
            "reward/budget/tat_abs": budget_abs["tat"],
            "reward/budget/backlog_abs": budget_abs["backlog"],
            "reward/budget/global_abs": budget_abs["global"],
            "reward/budget/local_abs": budget_abs["local"],
            "reward/budget/rail_abs": budget_abs["rail"],
            "reward/budget/smooth_abs": budget_abs["smooth"],
            "reward/budget/tat_share": budget_shares["tat"],
            "reward/budget/backlog_share": budget_shares["backlog"],
            "reward/budget/local_share": budget_shares["local"],
            "reward/budget/rail_share": budget_shares["rail"],
            "reward/budget/smooth_share": budget_shares["smooth"],
            "reward/budget/share_sum_error": abs(
                1.0 - sum(budget_shares.values())
            ) if budget_total > 0.0 else 0.0,
        }
        for name, values in (
            ("oht", batch.local_oht_raw),
            ("predicted", batch.local_predicted_raw),
            ("stop", batch.local_stop_raw),
            ("idle", batch.local_idle_raw),
            ("capacity", batch.local_capacity_raw),
        ):
            result[f"reward/local/{name}_raw_mean"] = float(values.mean())
            result[f"reward/local/{name}_raw_std"] = float(values.std())
            result[f"reward/local/{name}_raw_abs_mean"] = float(
                np.abs(values).mean()
            )
            # Compact Reward N metrics use coefficient-applied subterms,
            # not the unweighted input features.
            compact_name = "pred" if name == "predicted" else name
            result[f"local/{compact_name}_abs_mean"] = float(
                np.abs(values).mean()
            )
            result[f"local/{compact_name}_std"] = float(values.std())
        local_term_abs = {
            name: result[f"reward/local/{name}_raw_abs_mean"]
            for name in ("oht", "predicted", "stop", "idle", "capacity")
        }
        local_term_denominator = (
            sum(local_term_abs.values()) + np.finfo(np.float64).eps
        )
        for name, output in (
            ("predicted", "predicted_abs_share"),
            ("oht", "oht_abs_share"),
            ("stop", "stop_abs_share"),
            ("idle", "idle_abs_share"),
            ("capacity", "capacity_abs_share"),
        ):
            result[f"reward/local/{output}"] = (
                local_term_abs[name] / local_term_denominator
            )
        if not np.isfinite(tuple(result.values())).all():
            raise ContextualRewardError("reward diagnostics contain NaN or Inf")
        return result

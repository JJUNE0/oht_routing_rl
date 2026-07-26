"""Controlled-center reward construction for contextual rail control."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from contextual_action import ACTION_MODES, EXP_RESIDUAL, REGION_B_RL
from contextual_observation import RunningFeatureNormalizer
from contextual_topology import ContextualTopology

REWARD_VERSION = "contextual_controlled_reward_v4_balanced_global_local"


class ContextualRewardError(RuntimeError):
    """Raised when reward inputs violate the controlled-center contract."""


@dataclass(frozen=True)
class ContextualRewardConfig:
    global_alpha: float = 0.5
    local_alpha: float = 0.5
    rail_tat_weight: float = 1.0
    action_mode: str = REGION_B_RL
    smooth_b_rl_weight: float = 0.5
    smooth_exp_residual_weight: float = 0.5
    tat_weight: float = 9.2
    op_weight: float = 5.0
    backlog_weight: float = 0.01
    use_tat: bool = True
    use_op: bool = False
    use_backlog: bool = True
    tat_reference: float = 2.90706 * 60.0
    tat_ema_beta: float = 0.05
    freeze_after_env_steps: int = 30_000
    normalizer_epsilon: float = 1e-6
    global_clip: float | None = 5.0
    local_clip: float | None = None

    def __post_init__(self):
        numeric = (
            self.global_alpha, self.local_alpha, self.rail_tat_weight,
            self.smooth_b_rl_weight, self.smooth_exp_residual_weight,
            self.tat_weight, self.op_weight,
            self.backlog_weight, self.tat_reference, self.tat_ema_beta,
            self.normalizer_epsilon,
        )
        if not np.isfinite(numeric).all():
            raise ValueError("reward config contains NaN or Inf")
        if self.tat_reference <= 0 or self.normalizer_epsilon <= 0:
            raise ValueError("tat_reference and normalizer_epsilon must be positive")
        if self.freeze_after_env_steps < 0:
            raise ValueError("freeze_after_env_steps must be non-negative")
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
    rail_tat_penalty: np.ndarray
    smooth_control_delta: np.ndarray
    smooth_penalty: np.ndarray
    controlled_rail_ids: np.ndarray
    marginal_tat_ema: float
    tat_signal_available: float
    env_step: int
    episode_id: int


class ContextualRewardBuilder:
    """Build one reward row per controlled rail from the post-action snapshot."""

    def __init__(
        self,
        topology: ContextualTopology,
        config: ContextualRewardConfig | None = None,
    ):
        self.topology = topology
        self.config = config or ContextualRewardConfig()
        self.local_normalizer = RunningFeatureNormalizer(
            1, epsilon=self.config.normalizer_epsilon, clip=self.config.local_clip
        )
        self.global_normalizer = RunningFeatureNormalizer(
            1, epsilon=self.config.normalizer_epsilon, clip=self.config.global_clip
        )
        self.reward_steps = 0
        self._oht_route_accum: dict[int, list[tuple[int, float]]] = {}
        self._reset_temporal()

    def _reset_temporal(self) -> None:
        self._total_completed_jobs = 0.0
        self._prev_completed: float | None = None
        self._prev_tat_sum = 0.0
        self._prev_op_rate: float | None = None
        self._tat_ema = 0.0
        self._oht_route_accum.clear()

    def reset_episode(self) -> None:
        """Clear only episode-temporal state; training statistics persist."""
        self._reset_temporal()

    @staticmethod
    def _job_backlog(pclient) -> tuple[float, float]:
        waiting = float(getattr(pclient, "WaitingCommandCount", 0) or 0)
        queued = float(getattr(pclient, "QueuedCommandCount", 0) or 0)
        return waiting, queued

    def _global_raw(self, pclient) -> float:
        cfg = self.config
        cur_tat = float(getattr(pclient, "TotalTat"))
        cur_op = float(getattr(pclient, "TotalOhtOperationRate"))
        completed_delta = float(
            getattr(pclient, "CompletedCommandCount", 0) or 0
        )
        if not np.isfinite((cur_tat, cur_op, completed_delta)).all():
            raise ContextualRewardError("global reward input contains NaN or Inf")
        self._total_completed_jobs += completed_delta
        completed = self._total_completed_jobs
        cur_tat_sum = cur_tat * completed
        waiting, queued = self._job_backlog(pclient)

        if self._prev_completed is None:
            raw = 0.0
        else:
            delta_completed = completed - self._prev_completed
            if delta_completed > 0:
                marginal_tat = (
                    cur_tat_sum - self._prev_tat_sum
                ) / delta_completed
                beta = cfg.tat_ema_beta
                self._tat_ema = (
                    marginal_tat if self._tat_ema <= 0
                    else (1 - beta) * self._tat_ema + beta * marginal_tat
                )
            # No completed command means there is no TAT observation. Treating
            # the initial zero EMA as a real zero-second TAT awarded the maximum
            # positive TAT reward while the factory state was unchanged.
            tat_term = (
                (cfg.tat_reference - self._tat_ema) / cfg.tat_reference
                if self._tat_ema > 0
                else 0.0
            )
            op_delta = float(self._prev_op_rate) - cur_op
            raw = (
                (cfg.tat_weight * tat_term if cfg.use_tat else 0.0)
                + (cfg.op_weight * op_delta if cfg.use_op else 0.0)
                - (cfg.backlog_weight * (waiting + queued)
                   if cfg.use_backlog else 0.0)
            )
        self._prev_completed = completed
        self._prev_tat_sum = cur_tat_sum
        self._prev_op_rate = cur_op
        return float(raw)

    def _local_raw(self, pclient) -> np.ndarray:
        result = np.empty(len(self.topology.controlled_rail_ids), dtype=np.float64)
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
            result[row] = -(
                0.3 * oht_count
                + 0.2 * float(getattr(rail, "PredictedOHTCount"))
                + 0.3 * avg_stop
                + 0.1 * float(getattr(rail, "IdleOHTCount"))
                + 0.1 * capacity
            )
        if not np.isfinite(result).all():
            raise ContextualRewardError("local reward input contains NaN or Inf")
        return result

    def _rail_tat(self, pclient) -> np.ndarray:
        credit: dict[int, float] = {}
        for oht_id, oht in getattr(pclient, "OHT_DIC", {}).items():
            route = self._oht_route_accum.setdefault(int(oht_id), [])
            for pass_time in (getattr(oht, "PassTimes", None) or []):
                rail_id = int(getattr(pass_time, "ID"))
                elapsed = float(getattr(pass_time, "PassTime"))
                if not np.isfinite(elapsed):
                    raise ContextualRewardError("rail pass time contains NaN or Inf")
                route.append((rail_id, elapsed))
            completions = getattr(oht, "CmdCompleteTat", None)
            if not completions:
                continue
            total_time = sum(elapsed for _, elapsed in route)
            values = completions.values() if hasattr(completions, "values") else completions
            for completion in values:
                cmd_tat = float(getattr(completion, "CmdTat", 0) or 0)
                if cmd_tat <= 0:
                    continue
                excess = (cmd_tat - self.config.tat_reference) / self.config.tat_reference
                if route and total_time > 0:
                    for rail_id, elapsed in route:
                        credit[rail_id] = credit.get(rail_id, 0.0) + excess * (
                            elapsed / total_time
                        )
                else:
                    fallback = (
                        getattr(oht, "RouteList")[0]
                        if getattr(oht, "RouteList", None) else None
                    )
                    if fallback is not None:
                        fallback = int(fallback)
                        credit[fallback] = credit.get(fallback, 0.0) + excess
            route.clear()
        result = np.asarray(
            [credit.get(int(rail_id), 0.0)
             for rail_id in self.topology.controlled_rail_ids],
            dtype=np.float64,
        )
        return result * self.config.rail_tat_weight

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
        global_norm = float(
            self.global_normalizer.normalize([[global_raw]], name="global_reward")[0, 0]
        )
        local_norm = self.local_normalizer.normalize(
            local_raw[:, None], name="local_reward"
        )[:, 0]
        # Contractual normalize-before-update; exactly one batch update each.
        self.global_normalizer.update([[global_raw]], name="global_reward")
        self.local_normalizer.update(local_raw[:, None], name="local_reward")
        self.reward_steps += 1
        if self.reward_steps >= self.config.freeze_after_env_steps:
            self.global_normalizer.freeze()
            self.local_normalizer.freeze()

        global_component = self.config.global_alpha * global_norm
        local_component = self.config.local_alpha * local_norm
        rail_tat = self._rail_tat(pclient)
        if self.config.action_mode == REGION_B_RL:
            control_delta = np.abs(
                (0.5 + 0.5 * action) - (0.5 + 0.5 * previous)
            )
            smooth_weight = self.config.smooth_b_rl_weight
        else:
            control_delta = np.abs(action - previous)
            smooth_weight = self.config.smooth_exp_residual_weight
        smooth = smooth_weight * control_delta
        total = global_component + local_component - rail_tat - smooth
        arrays = {
            "total": total, "local_raw": local_raw,
            "local_normalized": local_norm, "local_component": local_component,
            "rail_tat_penalty": rail_tat,
            "smooth_control_delta": control_delta,
            "smooth_penalty": smooth,
            "controlled_rail_ids": self.topology.controlled_rail_ids,
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
            global_raw=float(global_raw),
            global_normalized=float(global_norm),
            global_component=float(global_component),
            marginal_tat_ema=float(self._tat_ema),
            tat_signal_available=float(self._tat_ema > 0),
            env_step=int(env_step),
            episode_id=int(episode_id),
        )

    def diagnostics(self, batch: ControlledRewardBatch) -> dict[str, float]:
        return {
            "reward/global_raw": batch.global_raw,
            "reward/global_normalized": batch.global_normalized,
            "reward/global_component": batch.global_component,
            "reward/marginal_tat_ema": batch.marginal_tat_ema,
            "reward/tat_signal_available": batch.tat_signal_available,
            "reward/local_raw_mean": float(batch.local_raw.mean()),
            "reward/local_raw_std": float(batch.local_raw.std()),
            "reward/local_normalized_mean": float(batch.local_normalized.mean()),
            "reward/local_normalized_std": float(batch.local_normalized.std()),
            "reward/local_component_mean": float(batch.local_component.mean()),
            "reward/local_component_std": float(batch.local_component.std()),
            "reward/rail_tat_penalty_mean": float(batch.rail_tat_penalty.mean()),
            "reward/rail_tat_penalty_max": float(batch.rail_tat_penalty.max()),
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
            "reward/finite_ratio": float(np.isfinite(batch.total).mean()),
            "normalizer/local_reward_update_calls": float(
                self.local_normalizer.update_calls
            ),
            "normalizer/local_reward_sample_count": float(
                self.local_normalizer.count
            ),
            "normalizer/global_reward_update_calls": float(
                self.global_normalizer.update_calls
            ),
            "normalizer/global_reward_sample_count": float(
                self.global_normalizer.count
            ),
        }

    def save_normalizers(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        local, global_state = (
            self.local_normalizer.state_dict(),
            self.global_normalizer.state_dict(),
        )
        np.savez_compressed(
            target,
            topology_hash=np.asarray(self.topology.topology_hash),
            mapping_hash=np.asarray(self.topology.mapping_hash),
            reward_steps=np.asarray(self.reward_steps),
            local_mean=local["mean"], local_m2=local["m2"],
            local_count=np.asarray(local["count"]),
            local_update_calls=np.asarray(local["update_calls"]),
            local_frozen=np.asarray(local["frozen"]),
            global_mean=global_state["mean"], global_m2=global_state["m2"],
            global_count=np.asarray(global_state["count"]),
            global_update_calls=np.asarray(global_state["update_calls"]),
            global_frozen=np.asarray(global_state["frozen"]),
        )
        return target

    def load_normalizers(self, path: str | Path) -> None:
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["topology_hash"].item()) != self.topology.topology_hash:
                raise ContextualRewardError("reward normalizer topology hash mismatch")
            if str(saved["mapping_hash"].item()) != self.topology.mapping_hash:
                raise ContextualRewardError("reward normalizer mapping hash mismatch")
            for prefix, normalizer in (
                ("local", self.local_normalizer),
                ("global", self.global_normalizer),
            ):
                normalizer.load_state_dict({
                    "dim": 1,
                    "mean": saved[f"{prefix}_mean"],
                    "m2": saved[f"{prefix}_m2"],
                    "count": int(saved[f"{prefix}_count"].item()),
                    "update_calls": int(saved[f"{prefix}_update_calls"].item()),
                    "frozen": bool(saved[f"{prefix}_frozen"].item()),
                })
            self.reward_steps = int(saved["reward_steps"].item())

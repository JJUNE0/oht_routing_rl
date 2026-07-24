import os
import time
from collections import deque
from datetime import datetime

import numpy as np
import torch

import wandb
from cocel_rl.algorithms.token_td7 import TokenTD7Learner
from cocel_rl.algorithms.token_td7.numerics import assert_finite_arrays, assert_finite_tree
from ClientAlgorithm import (
    ClientAlgorithm as PerRailClientAlgorithm,
    RunningRewardNormalizer,
    StateNormalizer,
    train_config as per_rail_train_config,
)


EXP_META = {
    "cost_structure": "b_rl",
    "action_range": "a[-1,1]_b[0,1]",
    "reward_version": "J",
    "centering": False,
    "note": "td7_full_fix",
    "description": (
        "[Region TD7 actor/critic normalization hot-fix, fresh training] "
        "Bounded single-projection attention과 SALE zs-to-zsa 연결을 유지한다. TD7의 "
        "actor/critic AvgL1Norm 경로를 복원하고, downstream context normalization, "
        "masked pre-tanh penalty, mean token-Q aggregation을 적용한다."
    ),
}


def _make_run_name(exp_meta):
    stamp = datetime.now().strftime("%m%d_%H%M")
    return (
        f"run_{stamp}_{exp_meta['cost_structure']}_"
        f"{exp_meta['action_range']}_{exp_meta['reward_version']}_{exp_meta['note']}"
    )


def train_config():
    config = per_rail_train_config()
    config["resume"] = False
    config["resume_ckpt_dir"] = ""
    config["save_dir"] = "./checkpoints/region_td7"
    config["buffer_capacity"] = 500_000
    config["model_type"] = "region_td7_bounded_attention_sale_actor_critic_v2"
    config["algorithm"]["name"] = "TD7"
    config["algorithm"]["encoder_grad_clip"] = 1.0
    config["algorithm"]["encoder_grad_abort_norm"] = 1_000.0
    config["algorithm"]["failure_artifact_topk"] = 16
    config["numeric_artifact_dir"] = "./results/numeric_failures"
    config["token_td7"] = {
        "embed_dim": 128,
        "num_heads": 4,
    }
    config["region"] = {
        "enabled": True,
        "target_size": 50,
        "min_size": 1,
        "context_global_dim": 6,
    }
    config["curriculum"]["scale_end"] = 1.0
    config["exp_meta"] = dict(EXP_META)
    return config


class ClientAlgorithm(PerRailClientAlgorithm):
    parameterDw = {}
    parameterPassTimes = {}
    parameterC = {}

    def __init__(self):
        print("ClientAlgorithm_region start")

        self.obs_dim = 14  # 6 global + 8 rail (ClientAlgorithm._build_raw_obs 참고)
        self.act_dim = 1
        self.action_bound = [-1.0, 1.0]
        self.config = train_config()
        self.global_obs_dim = int(self.config["region"].get("context_global_dim", 6))
        self.context_dim = self.global_obs_dim + self.obs_dim * 2

        print(f"obs_dim : {self.obs_dim}")
        print(f"act_dim : {self.act_dim}")
        print(f"action_bound : {self.action_bound}")
        print(f"context_dim : {self.context_dim}")
        print("[Algo] region-token TD7 bounded attention + normalized actor/critic")

        self.learner = TokenTD7Learner(
            rail_dim=self.obs_dim,
            action_dim=self.act_dim,
            action_bound=self.action_bound,
            context_dim=self.context_dim,
            config=self.config,
        )

        if self.config["save_model"]:
            os.makedirs(self.config["save_dir"], exist_ok=True)
        self._checkpoint_root = None
        self._checkpoint_no = 0

        self.last_batched_states = None
        self.last_actions = None
        self.last_raw_obs = None
        self.last_region_views = None
        self.last_region_actions = None
        self.last_sorted_rail_ids = None
        self._last_step_reward = 0.0
        self.total_steps = 0

        self.parameterDw = {}
        self.parameterPassTimes = {}
        self.parameterC = {}

        self.state_normalizer = StateNormalizer(dim=self.obs_dim)
        self.global_reward_normalizer = RunningRewardNormalizer()
        self.local_reward_normalizer = RunningRewardNormalizer()
        self.delta_normalizer = RunningRewardNormalizer()
        self.baseline_normalizer = RunningRewardNormalizer()

        self.prev_tat = None
        self.prev_op_rate = None
        self.prev_completed = None
        self.prev_jp = None
        self.prev_tat_sum = None
        self.total_completed_jobs = 0  # CompletedCommandCount(delta) 누적, 에피소드 시작부터
        self._tat_ema = 0.0
        self.delta_normalizer_frozen = False
        self.pending_smooth = None

        self._oht_route_accum = {}
        self._rail_tat_credit = {}
        self._last_rail_tat_penalty = 0.0

        self.sub_step_count = 0
        self.sub_episode_count = 0

        self._episode_return = 0.0
        self._best_episode_return = -float("inf")
        self._best_episode = -1
        self._best_step = 0
        self._episode_failed = False

        self._region_members = None
        self._region_rail_to_region = None
        self._region_cache_key = None
        self._region_stats = {}
        self._timings = {}
        self._timing_steps = 0

        # Checkpoints do not currently contain the replay buffer. After a
        # successful resume, refill it with the restored policy before any
        # optimizer update is allowed.
        self._resume_loaded = False
        self._resume_learning_started = False
        self._resume_buffer_refill_active = False
        self._resume_buffer_refill_start_step = None
        self._resume_buffer_refill_completed_step = None

        self.reward_log_path = "./region_reward_log.csv"
        if not os.path.exists(self.reward_log_path):
            with open(self.reward_log_path, "w") as f:
                f.write("step,sub_episode,tat,op_rate,sub_r,delta_r,baseline_r\n")

        if self.config["resume"]:
            self._resume_loaded = self.resume_checkpoint()
            if self._resume_loaded:
                self.config["warmup_episodes"] = 0
                self.config["warmup_steps"] = 0

        self.wandb_ok = False
        try:
            wandb_config = dict(self.config)
            wandb_config["exp_meta"] = dict(EXP_META)
            wandb.init(
                project="oht-routing-rl-td7-region",
                config=wandb_config,
                name=_make_run_name(EXP_META),
                notes=EXP_META.get("description", ""),
                resume="allow",
                mode=os.environ.get("WANDB_MODE", "online"),
            )
            self.wandb_ok = True
        except Exception as e:
            print(f"[wandb] init failed ({e}); training continues without wandb.")

    def on_new_connection(self):
        print(
            f"[Reconnect] keep learner state. total_steps={self.total_steps}, "
            f"buffer={self.learner.buffer.size}"
        )
        self.last_batched_states = None
        self.last_actions = None
        self.last_raw_obs = None
        self.last_region_views = None
        self.last_region_actions = None
        self.last_sorted_rail_ids = None
        self._last_step_reward = 0.0
        self.prev_tat = None
        self.prev_op_rate = None
        self.prev_completed = None
        self.prev_jp = None
        self.sub_start_tat = None
        self.sub_start_op = None
        self.prev_tat_sum = None
        self._tat_ema = 0.0
        self.pending_smooth = None

    def Reset(self, pclient):
        self._maybe_save_best_checkpoint(pclient)
        super().Reset(pclient)
        self._episode_return = 0.0
        self._episode_failed = False
        self.last_region_views = None
        self.last_region_actions = None
        self.last_sorted_rail_ids = None
        self._last_step_reward = 0.0

    def _t(self, key, dt):
        self._timings[key] = self._timings.get(key, 0.0) + float(dt)

    def _finish_timing_step(self, step_t0):
        self._t("step_total", time.perf_counter() - step_t0)
        self._timing_steps += 1

    def _build_timing_log(self):
        n = max(1, int(self._timing_steps))
        return {
            f"timing/{key}_ms": float(value * 1000.0 / n)
            for key, value in sorted(self._timings.items())
        }

    def _build_region_cache(self, pclient, sorted_rail_ids):
        rail_set = set(sorted_rail_ids)
        neighbors = {rid: set() for rid in sorted_rail_ids}
        for rid in sorted_rail_ids:
            rail = pclient.RAILLINE_DIC[rid]
            for nxt in getattr(rail, "DivergingLineIDList", []):
                if nxt in rail_set:
                    neighbors[rid].add(nxt)
                    neighbors[nxt].add(rid)

        target_size = max(1, int(self.config["region"].get("target_size", 30)))
        assigned = set()
        groups = []
        for seed in sorted_rail_ids:
            if seed in assigned:
                continue
            queue = deque([seed])
            queued = {seed}
            group = []
            while queue and len(group) < target_size:
                cur = queue.popleft()
                if cur in assigned:
                    continue
                assigned.add(cur)
                group.append(cur)
                if len(group) >= target_size:
                    break
                for nxt in sorted(neighbors.get(cur, [])):
                    if nxt not in assigned and nxt not in queued:
                        queue.append(nxt)
                        queued.add(nxt)
            if group:
                groups.append(group)

        flat = [rid for g in groups for rid in g]
        unique = set(flat)
        unassigned = len(rail_set - unique)
        duplicate = len(flat) - len(unique)
        sizes = np.array([len(g) for g in groups], dtype=np.float32)
        if len(sizes) == 0:
            # 시뮬레이터 초기화/재연결 구간에는 RAILLINE_DIC가 잠시 비어 있을 수 있다.
            # 이는 비정상 partition이 아니라 아직 action을 적용할 rail이 없는 상태다.
            # 빈 캐시를 남겨 두면 다음 통신 구간에서 rail 목록이 생겼을 때 key 변경으로
            # 정상적으로 partition을 다시 생성한다.
            self._region_members = []
            self._region_rail_to_region = {}
            self._region_cache_key = tuple(sorted_rail_ids)
            self._region_stats = {
                "region/num_regions": 0.0,
                "region/size_min": 0.0,
                "region/size_mean": 0.0,
                "region/size_max": 0.0,
                "region/size_std": 0.0,
                "region/unassigned_count": 0.0,
                "region/duplicate_count": 0.0,
            }
            print("[Region] no rail lines available; skipping region action for this step.")
            return

        self._region_members = [np.array(g, dtype=np.int64) for g in groups]
        self._region_rail_to_region = {
            int(rid): int(region_id)
            for region_id, group in enumerate(self._region_members)
            for rid in group
        }
        self._region_cache_key = tuple(sorted_rail_ids)
        self._region_stats = {
            "region/num_regions": float(len(groups)),
            "region/size_min": float(sizes.min()),
            "region/size_mean": float(sizes.mean()),
            "region/size_max": float(sizes.max()),
            "region/size_std": float(sizes.std()),
            "region/unassigned_count": float(unassigned),
            "region/duplicate_count": float(duplicate),
        }
        print(
            "[Region] built partition: "
            f"num={len(groups)} size_min={sizes.min():.0f} "
            f"size_mean={sizes.mean():.1f} size_max={sizes.max():.0f} "
            f"unassigned={unassigned} duplicate={duplicate}"
        )

    def _ensure_region_cache(self, pclient, sorted_rail_ids):
        key = tuple(sorted_rail_ids)
        if self._region_cache_key != key:
            self._build_region_cache(pclient, sorted_rail_ids)

    def _make_context(self, rail_feat):
        if len(rail_feat) == 0:
            return np.zeros(self.context_dim, dtype=np.float32)
        if rail_feat.shape[1] >= self.global_obs_dim:
            global_obs = rail_feat[0, : self.global_obs_dim]
        else:
            global_obs = np.zeros(self.global_obs_dim, dtype=np.float32)
        mean_feat = np.mean(rail_feat, axis=0)
        max_feat = np.max(rail_feat, axis=0)
        return np.concatenate([global_obs, mean_feat, max_feat]).astype(np.float32)

    def _build_region_views(self, obs_arr, sorted_rail_ids):
        index_by_rail = {int(rid): i for i, rid in enumerate(sorted_rail_ids)}
        views = []
        for region_id, members in enumerate(self._region_members):
            idxs = [index_by_rail[int(rid)] for rid in members if int(rid) in index_by_rail]
            if not idxs:
                continue
            rail_feat = np.asarray(obs_arr[idxs], dtype=np.float32)
            mask = np.ones((len(idxs),), dtype=bool)
            views.append(
                {
                    "region_id": region_id,
                    "rail_ids": np.asarray([sorted_rail_ids[i] for i in idxs], dtype=np.int64),
                    "rail_feat": rail_feat,
                    "context": self._make_context(rail_feat),
                    "mask": mask,
                }
            )
        return views

    def _select_region_actions(self, views, eval=False, apply_scale=True):
        if not views:
            return [], 1.0
        scale = self._action_scale() if apply_scale else 1.0

        lengths = [view["rail_feat"].shape[0] for view in views]
        max_len = max(lengths)
        num_views = len(views)

        # 모든 region을 max_len으로 패딩해 한 번의 forward로 처리
        # (region별 개별 호출은 매번 GPU sync를 일으켜 region 수만큼 latency가 누적됨)
        batched_rail = np.zeros((num_views, max_len, self.obs_dim), dtype=np.float32)
        batched_mask = np.zeros((num_views, max_len), dtype=bool)
        batched_context = np.zeros((num_views, self.context_dim), dtype=np.float32)
        for i, (view, n) in enumerate(zip(views, lengths)):
            batched_rail[i, :n] = view["rail_feat"]
            batched_mask[i, :n] = view["mask"]
            batched_context[i] = view["context"]

        assert_finite_arrays(
            "action/input",
            (("rail_feat", batched_rail), ("context", batched_context)),
            context=f"env_step={self.total_steps}",
        )

        action, _ = self.learner.actor.get_action(
            batched_rail,
            batched_context,
            batched_mask,
            eval=eval,
        )
        action = np.asarray(action, dtype=np.float32) * scale

        finite_action_ratio = float(np.isfinite(action).mean()) if action.size else 1.0
        assert_finite_arrays(
            "action/output",
            (("action", action), ("curriculum_scale", np.asarray(scale, dtype=np.float32))),
            context=(
                f"env_step={self.total_steps}, finite_action_ratio={finite_action_ratio:.6f}"
            ),
        )
        action = np.clip(action, self.action_bound[0], self.action_bound[1])
        action = action * batched_mask[:, :, None].astype(np.float32)

        region_actions = [action[i, :n].astype(np.float32) for i, n in enumerate(lengths)]
        return region_actions, finite_action_ratio

    def _region_actions_to_rail_array(self, views, region_actions, sorted_rail_ids):
        index_by_rail = {int(rid): i for i, rid in enumerate(sorted_rail_ids)}
        rail_actions = np.zeros((len(sorted_rail_ids), self.act_dim), dtype=np.float32)
        for view, action in zip(views, region_actions):
            for local_i, rid in enumerate(view["rail_ids"]):
                rail_actions[index_by_rail[int(rid)]] = action[local_i]
        return rail_actions

    def _apply_costs(self, pclient, sorted_rail_ids, rail_actions):
        costs = []
        ratios = []
        for i, rail_id in enumerate(sorted_rail_ids):
            base = float(pclient.RAILLINE_DIC[rail_id].DistancePerVelocity)
            w = float(self.parameterDw.get(rail_id, 1))
            c = float(self.parameterC.get(rail_id, 0))
            a_i = float(rail_actions[i, 0])
            b_rl = 0.5 + 0.5 * a_i
            cost = max(1e-6, base + (w * c * b_rl))
            neutral = max(1e-6, base + (w * c * 0.5))
            pclient.RAILLINECOST_DIC[rail_id].FRailLineCost = cost
            costs.append(cost)
            ratios.append(cost / neutral)
        costs = np.asarray(costs, dtype=np.float32)
        ratios = np.asarray(ratios, dtype=np.float32)
        return {
            "cost/final_mean": float(costs.mean()) if len(costs) else 0.0,
            "cost/final_max": float(costs.max()) if len(costs) else 0.0,
            "cost/ratio_mean": float(ratios.mean()) if len(ratios) else 0.0,
            "cost/ratio_std": float(ratios.std()) if len(ratios) else 0.0,
            "cost/ratio_p50": float(np.percentile(ratios, 50)) if len(ratios) else 0.0,
            "cost/ratio_p95": float(np.percentile(ratios, 95)) if len(ratios) else 0.0,
            "cost/changed_ratio_5pct": float(np.mean(np.abs(ratios - 1.0) > 0.05)) if len(ratios) else 0.0,
        }

    def _compute_rail_rewards(self, pclient, sorted_rail_ids, is_sub_boundary, sub_terminal_r):
        g = self.get_global_reward(pclient)
        self._last_g = g
        self._update_rail_tat_credit(pclient)
        w_smooth = float(self.config["reward_weights"].get("smooth", 0.0))
        rewards = np.zeros((len(sorted_rail_ids),), dtype=np.float32)
        local_rewards = np.zeros((len(sorted_rail_ids),), dtype=np.float32)
        for i, rail_id in enumerate(sorted_rail_ids):
            rail = pclient.RAILLINE_DIC[rail_id]
            reward = self.get_step_reward(rail, pclient)
            local_rewards[i] = self._last_local_norm
            if is_sub_boundary:
                reward += sub_terminal_r / max(1, len(sorted_rail_ids))
            if (
                w_smooth > 0.0
                and self.pending_smooth is not None
                and i < len(self.pending_smooth)
            ):
                reward -= w_smooth * float(self.pending_smooth[i])
            rewards[i] = float(reward)
        self._last_step_reward = float(rewards[-1]) if len(rewards) else 0.0
        self._last_local_rewards = local_rewards
        self._episode_return += float(rewards.mean()) if len(rewards) else 0.0
        return rewards

    def _push_region_transitions(self, current_views, sorted_rail_ids, rail_rewards, done):
        index_by_rail = {int(rid): i for i, rid in enumerate(sorted_rail_ids)}
        local_rewards = getattr(self, "_last_local_rewards", None)
        region_rewards = []
        region_internal_stds = []
        region_local_stds = []
        next_by_region = {int(v["region_id"]): v for v in current_views}
        for prev_view, prev_action in zip(self.last_region_views, self.last_region_actions):
            region_id = int(prev_view["region_id"])
            if region_id not in next_by_region:
                continue
            reward_idxs = [index_by_rail[int(rid)] for rid in prev_view["rail_ids"] if int(rid) in index_by_rail]
            if not reward_idxs:
                continue
            region_rail_rewards = rail_rewards[reward_idxs]
            region_reward = float(np.mean(region_rail_rewards))
            region_internal_stds.append(float(np.std(region_rail_rewards)))
            if local_rewards is not None:
                region_local_stds.append(float(np.std(local_rewards[reward_idxs])))
            next_view = next_by_region[region_id]
            self.learner.buffer.push(
                prev_view["rail_feat"],
                prev_action,
                region_reward,
                next_view["rail_feat"],
                float(done),
                prev_view["mask"],
                next_view["mask"],
                prev_view["context"],
                next_view["context"],
            )
            region_rewards.append(region_reward)
        self._last_region_internal_std_mean = float(np.mean(region_internal_stds)) if region_internal_stds else 0.0
        self._last_region_internal_std_max = float(np.max(region_internal_stds)) if region_internal_stds else 0.0
        self._last_region_local_std_mean = float(np.mean(region_local_stds)) if region_local_stds else 0.0
        return np.asarray(region_rewards, dtype=np.float32)

    def _save_checkpoint(self):
        next_checkpoint_no = self._checkpoint_no + 1
        payload = {
            "learner": self.learner.get_params(),
            "meta_data": {
                "algorithm": "TD7",
                "model_type": self.config["model_type"],
                "exp_meta": dict(EXP_META),
                "timestep": self.total_steps,
                "sub_step_count": self.sub_step_count,
                "sub_episode_count": self.sub_episode_count,
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "context_dim": self.context_dim,
                "buffer_saved": False,
                "buffer_size_at_save": self.learner.buffer.size,
                "state_normalizer": {
                    "mean": self.state_normalizer.mean.tolist(),
                    "var": self.state_normalizer.var.tolist(),
                    "count": self.state_normalizer.count,
                    "is_fixed": self.state_normalizer.is_fixed,
                },
                "delta_normalizer": {
                    "mean": self.delta_normalizer.mean,
                    "var": self.delta_normalizer.var,
                    "count": self.delta_normalizer.count,
                    "frozen": self.delta_normalizer_frozen,
                },
                "parameterDw": dict(self.parameterDw),
            },
        }
        assert_finite_tree(
            "checkpoint/save_full",
            payload,
            context=f"env_step={self.total_steps}, checkpoint={next_checkpoint_no}",
        )

        checkpoint_root = self._checkpoint_root
        if checkpoint_root is None:
            run_dir = datetime.now().strftime("%Y%m%d-%H%M%S")
            checkpoint_root = os.path.join(self.config["save_dir"], run_dir)
        ckpt_dir = os.path.join(checkpoint_root, f"checkpoint_{next_checkpoint_no}")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(payload, os.path.join(ckpt_dir, "checkpoint.pt"))
        torch.save(self.learner.actor.state_dict(), os.path.join(ckpt_dir, "policy.pt"))
        self._checkpoint_root = checkpoint_root
        self._checkpoint_no = next_checkpoint_no
        print(f"[Checkpoint] region TD7 saved: {ckpt_dir}")

    _BEST_MIN_STEPS = 500  # 에피소드 최소 스텝 수 (이보다 짧으면 best 저장 안 함)

    def _maybe_save_best_checkpoint(self, pclient):
        """에피소드 return(매 스텝 reward 합산) 기준으로 best.pt 갱신."""
        if self._resume_loaded and not self._resume_learning_started:
            return
        if self.total_steps < self.config["warmup_steps"]:
            return
        if self._episode_failed:
            return
        if self.sub_step_count < self._BEST_MIN_STEPS:
            return
        if self._episode_return <= self._best_episode_return:
            return
        ep_idx = getattr(pclient, "episode_index", self.sub_episode_count)
        self._best_episode_return = self._episode_return
        self._best_episode = ep_idx
        self._best_step = self.total_steps
        self._save_best_checkpoint(pclient)

    def _save_best_checkpoint(self, pclient):
        """best/ 고정 경로에 덮어쓰기 저장."""
        best_tat = float(getattr(pclient, "TotalTat", 0.0))
        best_op_rate = float(getattr(pclient, "TotalOhtOperationRate", 0.0))
        best_completed = int(self.total_completed_jobs)
        best_queued = int(getattr(pclient, "QueuedCommandCount", 0))

        payload = {
            "learner": self.learner.get_params(),
            "meta_data": {
                "algorithm": "TD7",
                "model_type": self.config["model_type"],
                "exp_meta": dict(EXP_META),
                "timestep": self.total_steps,
                "sub_step_count": self.sub_step_count,
                "sub_episode_count": self.sub_episode_count,
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "context_dim": self.context_dim,
                "best_episode_return": self._best_episode_return,
                "best_episode": self._best_episode,
                "best_step": self._best_step,
                "best_tat": best_tat,
                "best_op_rate": best_op_rate,
                "best_completed": best_completed,
                "best_queued": best_queued,
                "buffer_size_at_save": self.learner.buffer.size,
                "state_normalizer": {
                    "mean": self.state_normalizer.mean.tolist(),
                    "var": self.state_normalizer.var.tolist(),
                    "count": self.state_normalizer.count,
                    "is_fixed": self.state_normalizer.is_fixed,
                },
                "delta_normalizer": {
                    "mean": self.delta_normalizer.mean,
                    "var": self.delta_normalizer.var,
                    "count": self.delta_normalizer.count,
                    "frozen": self.delta_normalizer_frozen,
                },
                "parameterDw": dict(self.parameterDw),
            },
        }
        assert_finite_tree(
            "checkpoint/save_best_full",
            payload,
            context=f"env_step={self.total_steps}, episode={self._best_episode}",
        )

        checkpoint_root = self._checkpoint_root
        if checkpoint_root is None:
            run_dir = datetime.now().strftime("%Y%m%d-%H%M%S")
            checkpoint_root = os.path.join(self.config["save_dir"], run_dir)
        best_dir = os.path.join(checkpoint_root, "best")
        os.makedirs(best_dir, exist_ok=True)
        torch.save(payload, os.path.join(best_dir, "checkpoint.pt"))
        torch.save(self.learner.actor.state_dict(), os.path.join(best_dir, "policy.pt"))
        self._checkpoint_root = checkpoint_root
        print(
            f"[Best Checkpoint] ep={self._best_episode} | return={self._best_episode_return:.4f} "
            f"| TAT={best_tat:.3f} | completed={best_completed} | queued={best_queued} "
            f"| step={self._best_step} → {best_dir}"
        )
        self._wlog(
            {
                "checkpoint/best_return": self._best_episode_return,
                "checkpoint/best_episode": float(self._best_episode),
                "checkpoint/best_step": float(self._best_step),
                "checkpoint/best_tat": best_tat,
                "checkpoint/best_op_rate": best_op_rate,
                "checkpoint/best_completed": float(best_completed),
                "checkpoint/best_queued": float(best_queued),
            },
            step=self.total_steps,
        )

    def resume_checkpoint(self):
        ckpt_path = os.path.join(self.config["resume_ckpt_dir"], "checkpoint.pt")
        if not os.path.exists(ckpt_path):
            print(f"[Resume] checkpoint not found at {ckpt_path}. starting fresh.")
            return False

        ckpt = torch.load(ckpt_path, map_location=self.config["device"], weights_only=False)
        assert_finite_tree(
            "checkpoint/load_full",
            ckpt,
            context=f"path={ckpt_path}",
        )
        self.learner.load_params(ckpt["learner"])

        meta = ckpt.get("meta_data", {})
        self.total_steps = meta.get("timestep", 0)
        self.sub_step_count = meta.get("sub_step_count", 0)
        self.sub_episode_count = meta.get("sub_episode_count", 0)
        self.parameterDw = dict(meta.get("parameterDw", {}))

        if "state_normalizer" in meta:
            sn = meta["state_normalizer"]
            self.state_normalizer.mean = np.array(sn["mean"], dtype=np.float32)
            self.state_normalizer.var = np.array(sn["var"], dtype=np.float32)
            self.state_normalizer.count = sn["count"]
            self.state_normalizer.is_fixed = sn["is_fixed"]

        if "delta_normalizer" in meta:
            dn = meta["delta_normalizer"]
            self.delta_normalizer.mean = dn["mean"]
            self.delta_normalizer.var = dn["var"]
            self.delta_normalizer.count = dn["count"]
            self.delta_normalizer_frozen = bool(dn.get("frozen", False))

        if not meta.get("buffer_saved", False):
            print("[Resume] replay buffer was not saved; resuming with an empty buffer.")

        buffer_capacity = int(self.learner.buffer.capacity)
        self._resume_buffer_refill_active = self.learner.buffer.size < buffer_capacity
        self._resume_buffer_refill_start_step = self.total_steps
        self._resume_buffer_refill_completed_step = None
        if self._resume_buffer_refill_active:
            print(
                "[Resume] replay refill mode enabled: "
                f"size={self.learner.buffer.size}/{buffer_capacity}. "
                "Optimizer updates are paused until the buffer is full; "
                "policy rollout and transition collection continue."
            )

        print(
            f"[Resume] loaded {ckpt_path} | total_steps={self.total_steps} "
            f"sub_step_count={self.sub_step_count} sub_episode_count={self.sub_episode_count}"
        )
        return True

    def _resume_buffer_fill_ratio(self):
        capacity = max(1, int(self.learner.buffer.capacity))
        return min(1.0, float(self.learner.buffer.size) / float(capacity))

    def _update_resume_buffer_refill_state(self):
        """Return True on the step that completes a resumed-buffer refill."""
        if not self._resume_buffer_refill_active:
            return False

        size = int(self.learner.buffer.size)
        capacity = int(self.learner.buffer.capacity)
        if size < capacity:
            if self.total_steps % 100 == 0:
                print(
                    "[Resume] replay refill in progress: "
                    f"{size}/{capacity} ({100.0 * self._resume_buffer_fill_ratio():.1f}%). "
                    "Inference/collection only; learner update paused."
                )
            return False

        self._resume_buffer_refill_active = False
        self._resume_buffer_refill_completed_step = self.total_steps
        start_step = self._resume_buffer_refill_start_step
        collected_steps = (
            self.total_steps - start_step if start_step is not None else 0
        )
        print(
            "[Resume] replay buffer refill complete: "
            f"{size}/{capacity} after {collected_steps} environment steps. "
            "Learner updates will resume on the next step."
        )
        self._wlog(
            {
                "resume/replay_refill_complete": 1.0,
                "resume/replay_refill_steps": float(collected_steps),
                "buffer/size": float(size),
                "buffer/fill_ratio": 1.0,
            },
            step=self.total_steps,
        )
        return True

    def _log_region_diagnostics(
        self,
        pclient,
        sorted_rail_ids,
        views,
        region_actions,
        rail_actions,
        region_rewards,
        cost_stats,
        finite_action_ratio,
        done,
    ):
        diag_t0 = time.perf_counter()
        actions = rail_actions.reshape(-1)
        b_vals = 0.5 + 0.5 * actions
        within_stds = [float(np.std(a.reshape(-1))) for a in region_actions if len(a)]
        region_action_stds = np.asarray(within_stds, dtype=np.float32)
        region_action_means = np.asarray(
            [float(np.mean(a.reshape(-1))) for a in region_actions if len(a)],
            dtype=np.float32,
        )
        region_action_ranges = np.asarray(
            [float(np.max(a.reshape(-1)) - np.min(a.reshape(-1))) for a in region_actions if len(a)],
            dtype=np.float32,
        )
        active_region_actions = []
        inactive_region_actions = []
        active_region_stds = []
        inactive_region_stds = []
        active_region_ranges = []
        inactive_region_ranges = []
        for view, action in zip(views, region_actions):
            if len(action) == 0:
                continue
            flat_action = action.reshape(-1)
            c_vals = np.asarray(
                [float(self.parameterC.get(int(rid), 0.0)) for rid in view["rail_ids"]],
                dtype=np.float32,
            )
            is_active_region = bool(np.any(c_vals > 0.0))
            target_actions = active_region_actions if is_active_region else inactive_region_actions
            target_stds = active_region_stds if is_active_region else inactive_region_stds
            target_ranges = active_region_ranges if is_active_region else inactive_region_ranges
            target_actions.extend([float(x) for x in flat_action])
            target_stds.append(float(np.std(flat_action)))
            target_ranges.append(float(np.max(flat_action) - np.min(flat_action)))

        active_region_actions = np.asarray(active_region_actions, dtype=np.float32)
        inactive_region_actions = np.asarray(inactive_region_actions, dtype=np.float32)
        active_region_stds = np.asarray(active_region_stds, dtype=np.float32)
        inactive_region_stds = np.asarray(inactive_region_stds, dtype=np.float32)
        active_region_ranges = np.asarray(active_region_ranges, dtype=np.float32)
        inactive_region_ranges = np.asarray(inactive_region_ranges, dtype=np.float32)
        eval_region_actions, eval_finite = self._select_region_actions(views, eval=True, apply_scale=True)
        eval_rail_actions = self._region_actions_to_rail_array(views, eval_region_actions, sorted_rail_ids)
        eval_actions = eval_rail_actions.reshape(-1)
        q_probe = {}
        pretanh_probe = {}
        layer_scale_probe = {}
        if views and region_actions:
            q_probe = self.learner.probe_transition(
                views[0]["rail_feat"],
                region_actions[0],
                views[0]["context"],
                views[0]["mask"],
            )
            pretanh_probe = self.learner.probe_pretanh(
                views[0]["rail_feat"],
                views[0]["context"],
                views[0]["mask"],
            )
            layer_scale_probe = self.learner.probe_layer_scale(
                views[0]["rail_feat"],
                views[0]["context"],
                views[0]["mask"],
            )
        weight_probe = self.learner.probe_weight_norms()
        bias_probe = self.learner.probe_bias_norms()

        oht_all = list(pclient.OHT_DIC.values())
        oht_states_list = [o.State for o in oht_all]
        js = self._job_stats(pclient)
        waiting = float(getattr(pclient, "WaitingCommandCount", 0) or 0)
        transfer = float(getattr(pclient, "TransferCommandCount", 0) or 0)
        diag_elapsed_ms = float((time.perf_counter() - diag_t0) * 1000.0)
        log_data = {
            "step": self.total_steps,
            "episode": getattr(pclient, "episode_index", 0),
            "sub_episode": self.sub_episode_count,
            "phase/random": 0.0,
            "phase/resume_buffer_refill": float(self._resume_buffer_refill_active),
            "buffer/size": self.learner.buffer.size,
            "buffer/fill_ratio": self._resume_buffer_fill_ratio(),
            "learner/update_enabled": float(
                not self._resume_buffer_refill_active
                and self.total_steps != self._resume_buffer_refill_completed_step
            ),
            "reward/step_reward": float(getattr(self, "_last_step_reward", 0.0)),
            "reward/global_norm": float(getattr(self, "_last_g", 0.0)),
            "reward/local_norm": float(getattr(self, "_last_local_norm", 0.0)),
            "reward/rail_tat": float(getattr(self, "_last_rail_tat_penalty", 0.0)),
            **self._rail_tat_diag(),
            "reward/alpha": float(self.config.get("reward_alpha", 0.5)),
            "reward/region_internal_std_mean": float(getattr(self, "_last_region_internal_std_mean", 0.0)),
            "reward/region_internal_std_max": float(getattr(self, "_last_region_internal_std_max", 0.0)),
            "reward/region_local_std_mean": float(getattr(self, "_last_region_local_std_mean", 0.0)),
            "reward/marg_tat": float(getattr(self, "_last_marg_tat", 0.0)),
            "reward/backlog": float(getattr(self, "_last_backlog", 0.0)),
            "reward/smooth_mean": (
                float(np.mean(self.pending_smooth)) if self.pending_smooth is not None else 0.0
            ),
            "curriculum/action_scale": float(self._action_scale()),
            "global/tat": float(pclient.TotalTat),
            "global/op_rate": float(pclient.TotalOhtOperationRate),
            "global/queued": float(pclient.QueuedCommandCount),
            "global/queued_jobs": float(pclient.QueuedCommandCount),
            "global/waiting": waiting,
            "global/transfer": transfer,
            "global/completed": float(getattr(pclient, "CompletedCommandCount", 0) or 0),
            "job/mean_wait_priority": js["mean_wait_pri"],
            "job/mean_reassign": js["mean_reassign"],
            "job/queued": js["queued"],
            "oht/idle_count": float(oht_states_list.count(0)),
            "oht/move_to_load": float(oht_states_list.count(2)),
            "oht/move_to_unload": float(oht_states_list.count(4)),
            "oht/loading": float(oht_states_list.count(3)),
            "oht/unloading": float(oht_states_list.count(5)),
            "action/mean": float(actions.mean()) if len(actions) else 0.0,
            "action/std": float(actions.std()) if len(actions) else 0.0,
            "action/min": float(actions.min()) if len(actions) else 0.0,
            "action/max": float(actions.max()) if len(actions) else 0.0,
            "action/eval_mean": float(eval_actions.mean()) if len(eval_actions) else 0.0,
            "action/eval_std": float(eval_actions.std()) if len(eval_actions) else 0.0,
            "b_rl/mean": float(b_vals.mean()) if len(b_vals) else 0.0,
            "b_rl/std": float(b_vals.std()) if len(b_vals) else 0.0,
            "token_action/mean": float(actions.mean()) if len(actions) else 0.0,
            "token_action/std": float(actions.std()) if len(actions) else 0.0,
            "token_action/eval_mean": float(eval_actions.mean()) if len(eval_actions) else 0.0,
            "token_action/eval_std": float(eval_actions.std()) if len(eval_actions) else 0.0,
            "token_action/saturation": float(np.mean(np.abs(actions) > 0.98)) if len(actions) else 0.0,
            "token_action/eval_saturation": float(np.mean(np.abs(eval_actions) > 0.98)) if len(eval_actions) else 0.0,
            "token_b/mean": float(b_vals.mean()) if len(b_vals) else 0.0,
            "token_b/std": float(b_vals.std()) if len(b_vals) else 0.0,
            "token_b/eval_mean": float((0.5 + 0.5 * eval_actions).mean()) if len(eval_actions) else 0.0,
            "token_b/eval_std": float((0.5 + 0.5 * eval_actions).std()) if len(eval_actions) else 0.0,
            "region_reward/mean": float(region_rewards.mean()) if len(region_rewards) else 0.0,
            "region_reward/std": float(region_rewards.std()) if len(region_rewards) else 0.0,
            "region_reward/min": float(region_rewards.min()) if len(region_rewards) else 0.0,
            "region_reward/max": float(region_rewards.max()) if len(region_rewards) else 0.0,
            "region_action/mean": float(actions.mean()) if len(actions) else 0.0,
            "region_action/std": float(actions.std()) if len(actions) else 0.0,
            "region_action/abs_mean": float(np.abs(actions).mean()) if len(actions) else 0.0,
            "region_action/within_region_std_mean": float(np.mean(within_stds)) if within_stds else 0.0,
            "in_region_per_rail_action_std/mean": float(region_action_stds.mean()) if len(region_action_stds) else 0.0,
            "in_region_per_rail_action_std/max": float(region_action_stds.max()) if len(region_action_stds) else 0.0,
            "in_region_per_rail_action_std/min": float(region_action_stds.min()) if len(region_action_stds) else 0.0,
            "region_action/region_mean_min": float(region_action_means.min()) if len(region_action_means) else 0.0,
            "region_action/region_mean_max": float(region_action_means.max()) if len(region_action_means) else 0.0,
            "region_action/region_mean_range": (
                float(region_action_means.max() - region_action_means.min())
                if len(region_action_means)
                else 0.0
            ),
            "region_action/in_region_range_mean": float(region_action_ranges.mean()) if len(region_action_ranges) else 0.0,
            "region_action/in_region_range_max": float(region_action_ranges.max()) if len(region_action_ranges) else 0.0,
            "region_action/in_region_range_min": float(region_action_ranges.min()) if len(region_action_ranges) else 0.0,
            "active_region/count": float(len(active_region_stds)),
            "inactive_region/count": float(len(inactive_region_stds)),
            "active_region/ratio": float(len(active_region_stds) / max(1, len(region_actions))),
            "active_region/action_mean": float(active_region_actions.mean()) if len(active_region_actions) else 0.0,
            "active_region/action_std": float(active_region_actions.std()) if len(active_region_actions) else 0.0,
            "active_region/action_abs_mean": float(np.abs(active_region_actions).mean()) if len(active_region_actions) else 0.0,
            "inactive_region/action_mean": float(inactive_region_actions.mean()) if len(inactive_region_actions) else 0.0,
            "inactive_region/action_std": float(inactive_region_actions.std()) if len(inactive_region_actions) else 0.0,
            "inactive_region/action_abs_mean": float(np.abs(inactive_region_actions).mean()) if len(inactive_region_actions) else 0.0,
            "active_region/in_region_per_rail_action_std_mean": float(active_region_stds.mean()) if len(active_region_stds) else 0.0,
            "active_region/in_region_per_rail_action_std_max": float(active_region_stds.max()) if len(active_region_stds) else 0.0,
            "active_region/in_region_per_rail_action_std_min": float(active_region_stds.min()) if len(active_region_stds) else 0.0,
            "inactive_region/in_region_per_rail_action_std_mean": float(inactive_region_stds.mean()) if len(inactive_region_stds) else 0.0,
            "inactive_region/in_region_per_rail_action_std_max": float(inactive_region_stds.max()) if len(inactive_region_stds) else 0.0,
            "inactive_region/in_region_per_rail_action_std_min": float(inactive_region_stds.min()) if len(inactive_region_stds) else 0.0,
            "active_region/in_region_action_range_mean": float(active_region_ranges.mean()) if len(active_region_ranges) else 0.0,
            "active_region/in_region_action_range_max": float(active_region_ranges.max()) if len(active_region_ranges) else 0.0,
            "inactive_region/in_region_action_range_mean": float(inactive_region_ranges.mean()) if len(inactive_region_ranges) else 0.0,
            "inactive_region/in_region_action_range_max": float(inactive_region_ranges.max()) if len(inactive_region_ranges) else 0.0,
            "region_mask/valid_tokens_mean": float(
                np.mean([len(v["rail_ids"]) for v in views]) / max(1.0, self._region_stats.get("region/size_max", 1.0))
            ),
            "numeric/action_finite_ratio": float(finite_action_ratio),
            "numeric/eval_action_finite_ratio": float(eval_finite),
            "termination/done": float(done),
            "termination/by_queue": float(pclient.QueuedCommandCount > 500),
            "termination/by_tat": float(float(pclient.TotalTat) > 500 and self.sub_step_count > 100),
            "timing/log_diag_ms": diag_elapsed_ms,
            "loss/q1": self.learner.total_losses.get("loss/q1", self.learner.total_losses.get("q1", 0)),
            "loss/q2": self.learner.total_losses.get("loss/q2", self.learner.total_losses.get("q2", 0)),
            "loss/critic": self.learner.total_losses.get("loss/critic", self.learner.total_losses.get("critic", 0)),
            "loss/actor": self.learner.total_losses.get("loss/actor", self.learner.total_losses.get("actor", 0)),
            "loss/encoder": self.learner.total_losses.get("loss/encoder", self.learner.total_losses.get("encoder", 0)),
            **self._region_stats,
            **cost_stats,
            **self._build_timing_log(),
            **self.learner.total_losses,
            **q_probe,
            **pretanh_probe,
            **layer_scale_probe,
            **weight_probe,
            **bias_probe,
        }
        self._wlog(log_data, step=self.total_steps)

    def _algorithm_impl(self, pclient):
        step_t0 = time.perf_counter()
        section_t0 = step_t0
        print(
            f"episode index : {getattr(pclient, 'episode_index', 0)} | sub_step: {self.sub_step_count}"
        )

        done = False
        tat_threshold = 500
        if pclient.QueuedCommandCount > 500:
            print("Queued Command >= 500. Ending simulation episode.")
            pclient.SendIsEnd(1)
            done = True
            self._episode_failed = True
        elif float(pclient.TotalTat) > tat_threshold and self.sub_step_count > 100:
            print(f"[Early Stop] TAT={pclient.TotalTat:.1f} > {tat_threshold:.1f}.")
            with open(self.reward_log_path, "a") as f:
                f.write(
                    f"{self.total_steps},{self.sub_episode_count},"
                    f"{pclient.TotalTat:.4f},{pclient.TotalOhtOperationRate:.4f},EARLY_STOP\n"
                )
            self._wlog(
                {
                    "early_stop/tat": float(pclient.TotalTat),
                    "early_stop/op_rate": float(pclient.TotalOhtOperationRate),
                    "early_stop/step": self.sub_step_count,
                    "early_stop/episode": getattr(pclient, "episode_index", 0),
                },
                step=self.total_steps,
            )
            pclient.SendIsEnd(1)
            done = True
            self._episode_failed = True
        else:
            pclient.SendIsEnd(0)

        print("\n" + "=" * 60)
        self._t("setup", time.perf_counter() - section_t0)

        is_warmup = self.total_steps < self.config["warmup_steps"]
        if is_warmup:
            section_t0 = time.perf_counter()
            current_obs, sorted_rail_ids = self.get_local_observations(pclient)
            self._t("obs", time.perf_counter() - section_t0)
            section_t0 = time.perf_counter()
            self._ensure_region_cache(pclient, sorted_rail_ids)
            self._t("region_cache", time.perf_counter() - section_t0)
            section_t0 = time.perf_counter()
            self.compute_baseline_params(pclient)
            self._t("baseline", time.perf_counter() - section_t0)
            section_t0 = time.perf_counter()
            for rail_id in sorted_rail_ids:
                base = float(pclient.RAILLINE_DIC[rail_id].DistancePerVelocity)
                w = float(self.parameterDw.get(rail_id, 1))
                c = float(self.parameterC.get(rail_id, 0))
                cost = max(1e-6, base + (w * c * 0.5))
                pclient.RAILLINECOST_DIC[rail_id].FRailLineCost = cost
            self._t("set_cost", time.perf_counter() - section_t0)
            self.total_steps += 1
            print(f"total step : {self.total_steps} (Buffer: {self.learner.buffer.size})")
            self._finish_timing_step(step_t0)
            if self.total_steps == self.config["warmup_steps"]:
                self._timings = {}
                self._timing_steps = 0
            return

        section_t0 = time.perf_counter()
        current_obs, sorted_rail_ids = self.get_local_observations(pclient)
        self._t("obs", time.perf_counter() - section_t0)
        section_t0 = time.perf_counter()
        self._ensure_region_cache(pclient, sorted_rail_ids)
        current_views = self._build_region_views(current_obs, sorted_rail_ids)
        self._t("region_cache", time.perf_counter() - section_t0)
        section_t0 = time.perf_counter()
        self.compute_baseline_params(pclient)
        self._t("baseline", time.perf_counter() - section_t0)

        region_rewards = np.zeros((0,), dtype=np.float32)
        did_push_transition = False
        if self.last_region_views is not None and self.last_region_actions is not None:
            self.sub_step_count += 1
            is_sub_boundary = self.sub_step_count % self.config["sub_episode_len"] == 0
            sub_terminal_r = 0.0
            if is_sub_boundary:
                sub_terminal_r = self.get_sub_terminal_reward(pclient)
                self.sub_episode_count += 1
                print(f"[Sub-episode {self.sub_episode_count} done] Step {self.sub_step_count}")

            section_t0 = time.perf_counter()
            rail_rewards = self._compute_rail_rewards(
                pclient,
                sorted_rail_ids,
                is_sub_boundary,
                sub_terminal_r,
            )
            self._t("reward", time.perf_counter() - section_t0)
            section_t0 = time.perf_counter()
            region_rewards = self._push_region_transitions(
                current_views,
                sorted_rail_ids,
                rail_rewards,
                done,
            )
            self._t("push", time.perf_counter() - section_t0)
            did_push_transition = True

            self.total_steps += 1
            print(f"total step : {self.total_steps} (Buffer: {self.learner.buffer.size})")

            refill_completed_this_step = self._update_resume_buffer_refill_state()
            learner_update_allowed = (
                self.total_steps > self.config["update_after"]
                and not self._resume_buffer_refill_active
                and not refill_completed_this_step
            )
            if learner_update_allowed:
                section_t0 = time.perf_counter()
                self.learner.set_curriculum_scale(self._action_scale())
                self.learner.learn()
                self._resume_learning_started = True
                self._t("learn", time.perf_counter() - section_t0)
                if self.total_steps % 2 == 0:
                    print(
                        f"Step: {self.total_steps} | region_R: "
                        f"{(region_rewards.mean() if len(region_rewards) else 0.0):.4f} | "
                        f"Q1: {self.learner.total_losses.get('loss/q1', 0):.4f}"
                    )

                if self.config["save_model"] and (
                    self.total_steps % self.config["model_checkpoint_freq"] == 0
                ):
                    self._save_checkpoint()

            if is_sub_boundary:
                self.prev_tat = None
                self.prev_op_rate = None
                self.prev_completed = None
                self.prev_jp = None

        if done:
            self.last_batched_states = None
            self.last_actions = None
            self.last_raw_obs = None
            self.last_region_views = None
            self.last_region_actions = None
            self.last_sorted_rail_ids = None
            self._finish_timing_step(step_t0)
            return

        section_t0 = time.perf_counter()
        region_actions, finite_action_ratio = self._select_region_actions(
            current_views,
            eval=False,
            apply_scale=True,
        )
        rail_actions = self._region_actions_to_rail_array(
            current_views,
            region_actions,
            sorted_rail_ids,
        )

        if self.last_actions is not None and len(self.last_actions) == len(rail_actions):
            diff = np.asarray(rail_actions, dtype=np.float32) - np.asarray(self.last_actions, dtype=np.float32)
            self.pending_smooth = np.abs(diff).reshape(len(rail_actions), -1).mean(axis=1)
        else:
            self.pending_smooth = None

        self._t("action", time.perf_counter() - section_t0)
        section_t0 = time.perf_counter()
        cost_stats = self._apply_costs(pclient, sorted_rail_ids, rail_actions)
        self._t("set_cost", time.perf_counter() - section_t0)

        self.last_batched_states = current_obs
        self.last_actions = rail_actions
        self.last_raw_obs = None
        self.last_region_views = current_views
        self.last_region_actions = region_actions
        self.last_sorted_rail_ids = list(sorted_rail_ids)

        if (
            did_push_transition
            and self.total_steps > self.config["update_after"]
            and self.total_steps % 100 == 0
        ):
            section_t0 = time.perf_counter()
            try:
                self._log_region_diagnostics(
                    pclient=pclient,
                    sorted_rail_ids=sorted_rail_ids,
                    views=current_views,
                    region_actions=region_actions,
                    rail_actions=rail_actions,
                    region_rewards=region_rewards,
                    cost_stats=cost_stats,
                    finite_action_ratio=finite_action_ratio,
                    done=done,
                )
            except FloatingPointError:
                raise
            except Exception as e:
                # 진단은 관측용이다. 로깅 실패가 action 적용/학습 step을 무효화하면 안 된다.
                print(f"[Region diagnostics] logging skipped: {e}")
            self._t("log", time.perf_counter() - section_t0)

        self._finish_timing_step(step_t0)

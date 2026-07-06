import argparse
import math
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
from cocel_rl.algorithms import TD3, TD7
from cocel_rl.buffers.off_policy_buffer import OffPolicyBuffer
from cocel_rl.core.learner import Learner
from cocel_rl.core.logger import OffPolicyLogger
from eval import evaluate
from torch.nn.modules.module import T

import wandb


def train_config():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.abspath(os.path.join(base_dir, "..", "results"))

    warmup = 10000  # warmup_steps + 커리큘럼 start_step 동시 제어. (StateNormalizer freeze는 ep1 끝이라 사실상 무관, 미세 노브)
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "actor_lr": 3e-4,
        "critic_lr": 3e-4,
        "adam_eps": 1e-8,
        "buffer_capacity": 5000000,
        "batch_size": 1024,
        "gamma": 0.99,
        "tau": 0.005,
        "max_rollout": 1,
        "update_after": 100,
        "actor_hidden_dims": [512, 512, 512],
        "critic_hidden_dims": [512, 512, 512],
        "activation_fc": "relu",
        "save_model": True,
        "save_dir": "./checkpoints/",
        "resume": False,  # 전달본 기본: 처음부터 학습. 이어 학습하려면 True로 바꾸고 아래 경로 지정.
        "resume_ckpt_dir": "./checkpoints/td7_final",  # (예시) resume=True일 때 불러올 체크포인트 경로 (첨부된 최종 TD7 모델)
        "model_checkpoint_freq": 10000,
        "eval_db_dir": results_dir,
        "warmup_episodes": 2,
        "warmup_steps": warmup,
        "reward_alpha": 0.5,
        # ── Action-scale 커리큘럼 ──────────────────────────────
        #  RL 시작 직후 액션을 작게 damping(→ Dijkstra baseline), step 진행에 따라 1.0까지 키움.
        #  a=0 ↔ b_rl=0.5 (warmup 중립값)이라, scale 작으면 사실상 순수 Dijkstra.
        "curriculum": {
            "enabled": True,
            "start_step": warmup,  # = warmup_steps (RL 시작 시점)과 일치시켜야 함
            "end_step": 40000,  # 이 시점에 scale=1.0 도달
            "scale_start": 0.05,
            "scale_end": 1.0,  # 1.0→0.4: 풀권한이 불안정 원인(11ep 분석, ep3 피크후 드리프트).
            #          Dijkstra(b_rl=0.5) 근처 미세조정으로 안정화.
            "shape": "geometric",  # 지수 보간 (초반 오래 작게). 'linear'도 가능
        },
        # Sub-episode 설정
        "sub_episode_len": 1000,
        # Baseline 값 (pclient 단위, warm-up에서 측정)
        "tat_base": 2.90706 * 60,  # eval.py DB 분석 기준 baseline(분→초 환산, ≈174.42). 초반 10000스텝 직접측정값(176.5)보다 더 정확한 기준으로 교체.
        "op_base": 0.810,
        # ── Reward 가중치 (2026-06-24 재보정: 실측 raw std 기준, 정규화기 제거판) ──
        #  tat    : windowed marginal TAT (baseline 대비 상대개선). 주신호.
        #           실측 tat_term std≈0.0602, backlog(0.01*backlog) std≈0.1113 기준,
        #           tat:backlog 기여도 비율 5:1을 맞추려면 w_tat = 5*0.1113/0.0602 ≈ 9.2.
        #  op     : 가동률 감소분. 정밀 std 미실측(coarse 로그로는 per-step d_op 복원 불가) → 기존값 유지.
        #  backlog: Waiting+Queued 수준 페널티 (혼잡 standing, -).
        #  smooth : 라인별 액션 step간 변화 페널티 (route thrash 억제, -).
        #  (제거됨: throughput=죽은신호(완료수 항상 ≈175299), job_pressure=죽은신호(pri=2 고정))
        "reward_weights": {
            "tat": 9.2,       # get_global_reward 내 TAT term 가중치
            "op": 5.0,        # get_global_reward 내 op term 가중치
            "backlog": 0.01,  # get_global_reward 내 backlog term 가중치
            "smooth": 0.5,    # 액션 스무딩 페널티
            "rail_tat": 1.0,  # step_reward 내 rail_tat_penalty 가중치
            "global_alpha": 0.3, # step_reward 내 global(g) 가중치
            "local_alpha": 0.0,     # step_reward 내 local(l) 가중치
        },
        "reward_components": {
            "use_global_reward": True,   # g를 step_reward에 포함 (dense signal)
            "use_local_reward": False,   # l을 step_reward에 포함
            "use_rail_tat": True,        # rail_tat_penalty를 step_reward에 포함
            "global": {
                "use_tat": True,         # global 내 marginal TAT term
                "use_op": False,          # global 내 op rate term
                "use_backlog": True,     # global 내 backlog penalty
            },
            "local": {
                "use_oht_count": True,   # local 내 OHT 재차 수
                "use_predicted": True,   # local 내 예상 OHT 수
                "use_stop": True,        # local 내 평균 정지시간
                "use_idle": True,        # local 내 idle blocking
                "use_capacity": True,    # local 내 capacity ratio
            },
        },
        "tat_ema_beta": 0.05,  # marginal TAT EMA 평활(≈20스텝 메모리)
        # 정규화기 비교 실험(버전 G): 정규화 방식은 원래(C 이전)의 온라인 적응형
        # delta_normalizer + freeze + clip으로 복원. 리워드 *공식*(total_completed_jobs
        # 누적, tat_base 보정, tat/backlog 재보정, rail_tat 항)만 새 버전을 그대로 사용해서
        # "정규화 방식은 고정, 리워드만 바뀐" 깨끗한 비교가 되게 함.
        "reward_freeze_after": 20000,  # warmup_steps 이후 이만큼 더 지나면 리워드 정규화 통계 고정(비정상성 차단)
        "global_reward_clip": 5.0,  # 정규화된 global reward 절대값 상한
        "use_state_normalizer": True,  # False면 관측값을 정규화 없이 그대로 사용
        "algorithm": {
            "name": "TD7",  # 'TD3' 또는 'TD7' (2026-06-13: 시뮬 desync 해결 후 상위 알고리즘 시도)
            "noise_scale": 0.1,
            "target_noise_scale": 0.2,
            "target_noise_clip": 0.5,
            "policy_update_delay": 2,
            "use_onnx": False,
            # --- TD7 전용 (name='TD7'일 때만 사용, TD3엔 무해) ---
            "encoder_lr": 3e-4,  # SALE 인코더 학습률
            "zs_dim": 256,  # 임베딩 차원
            "hdim": 256,  # TD7 은닉층 폭
            "target_update_rate": 250,  # 하드 타깃/인코더 스냅샷 주기
            "lap_alpha": 0.4,  # LAP 우선순위 지수
            "min_priority": 1.0,  # LAP Huber 경계
        },
    }
    return config


class DummyEnv:
    def __init__(self, obs_dim, act_dim, action_bound):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.action_bound = action_bound
        print(f"obs_dim : {self.obs_dim}")
        print(f"act_dim : {self.act_dim}")
        print(f"action_bound : {self.action_bound}")


class StateNormalizer:
    def __init__(self, dim):
        self.dim = dim
        self.mean = np.zeros(dim, dtype=np.float32)
        self.var = np.ones(dim, dtype=np.float32)
        self.count = 0
        self.is_fixed = False

    def update(self, x):
        if self.is_fixed:
            return
        self.count += 1
        old_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.count
        self.var += (x - old_mean) * (x - self.mean)

    def fix(self):
        self.is_fixed = True
        print(f"[StateNormalizer] 통계 고정! (샘플 수: {self.count})")
        print(f"  Mean: {self.mean}")
        print(f"  Std:  {self.get_std()}")

    def get_std(self):
        return np.sqrt(self.var / max(1, self.count)) + 1e-8

    def normalize(self, x):
        return (x - self.mean) / self.get_std()


class RunningRewardNormalizer:
    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0

    def update_and_normalize(self, reward):
        self.count += 1
        old_mean = self.mean
        self.mean += (reward - self.mean) / self.count
        self.var += (reward - old_mean) * (reward - self.mean)
        std = np.sqrt(self.var / max(1, self.count)) + 1e-8
        return (reward - self.mean) / std


class ClientAlgorithm:
    parameterDw = {}
    parameterPassTimes = {}
    parameterC = {}

    def __init__(self):
        print("ClientAlgorithm 시작")

        # obs = 10 global (job 3개 추가) + 8 rail = 18
        self.obs_dim = 18
        self.act_dim = 1
        self.action_bound = [-1.0, 1.0]

        env = DummyEnv(self.obs_dim, self.act_dim, self.action_bound)
        self.config = train_config()
        algo_name = str(self.config["algorithm"]["name"]).upper()
        algo_cls = {"TD3": TD3, "TD7": TD7}.get(algo_name)
        if algo_cls is None:
            raise ValueError(f"지원하지 않는 algorithm: {algo_name} (TD3/TD7)")
        print(f"[Algo] {algo_name} 사용")
        self.algo = algo_cls(env, self.config)
        self.learner = Learner(self.algo, self.config)

        if self.config["save_model"] and not os.path.exists(self.config["save_dir"]):
            os.makedirs(self.config["save_dir"])
        self.logger = OffPolicyLogger(self.config)

        self.last_batched_states = None
        self.last_actions = None
        self.last_raw_obs = None
        self.total_steps = 0

        self.parameterDw = {}
        self.parameterPassTimes = {}
        self.parameterC = {}

        # State 정규화
        self.state_normalizer = StateNormalizer(dim=self.obs_dim)

        # Reward 정규화
        self.global_reward_normalizer = RunningRewardNormalizer()
        self.local_reward_normalizer = RunningRewardNormalizer()
        self.delta_normalizer = RunningRewardNormalizer()
        self.baseline_normalizer = RunningRewardNormalizer()

        # Global reward delta용
        self.prev_tat = None
        self.prev_op_rate = None
        self.prev_completed = None
        self.prev_jp = None
        self.prev_tat_sum = None  # windowed marginal TAT용 (TotalTat × 완료수)
        self.total_completed_jobs = 0  # CompletedCommandCount(delta) 누적, 에피소드 시작부터
        self._tat_ema = 0.0  # marginal TAT EMA 상태
        self.delta_normalizer_frozen = False  # warmup 후 리워드 정규화 통계 고정 플래그
        self.pending_smooth = None  # |a_t - a_{t-1}| per-rail, 다음 push에 반영

        # rail-단위 TAT 귀속용: OHT가 현재 job 동안 지나온 (rail_id, 머문시간) 누적,
        # job 완료 시(CmdCompleteTat) 머문시간 비중대로 그 rail들에 TAT 초과분을 분배.
        self._oht_route_accum = {}
        self._rail_tat_credit = {}
        self._last_rail_tat_penalty = 0.0
        self._last_g = 0.0

        # Sub-episode 카운터
        self.sub_step_count = 0
        self.sub_episode_count = 0

        self.reward_log_path = "./reward_log.csv"
        if not os.path.exists(self.reward_log_path):
            with open(self.reward_log_path, "w") as f:
                f.write("step,sub_episode,tat,op_rate,sub_r,delta_r,baseline_r\n")

        if self.config["resume"]:
            self.resume_checkpoint()
            self.config["warmup_episodes"] = 0
            self.config["warmup_steps"] = 0

        self.wandb_ok = False
        try:
            wandb.init(
                project="oht-routing-rl",
                config=self.config,
                name=f"run_{datetime.now().strftime('%m%d_%H%M')}",
                resume="allow",
                mode=os.environ.get("WANDB_MODE", "online"),
            )
            self.wandb_ok = True
        except Exception as e:
            print(f"[wandb] init 실패 ({e}); 로깅 비활성화하고 학습 계속.")

    def on_new_connection(self):
        """시뮬레이터 재연결 시 호출. 학습 상태(buffer/모델/total_steps/normalizer)는
        유지하되, 끊김 구간을 가로지르는 transition이 생기지 않도록 step 경계 상태만 초기화."""
        print(
            f"[Reconnect] 학습 상태 유지하며 재연결. total_steps={self.total_steps}, buffer={self.learner.buffer.size}"
        )
        self.last_batched_states = None
        self.last_actions = None
        self.last_raw_obs = None
        self.prev_tat = None
        self.prev_op_rate = None
        self.prev_completed = None
        self.prev_jp = None
        self.sub_start_tat = None
        self.sub_start_op = None
        self.prev_tat_sum = None
        self._tat_ema = 0.0
        self.pending_smooth = None

    def _wlog(self, data, step=None):
        if not self.wandb_ok:
            return
        try:
            wandb.log(data, step=step)
        except Exception:
            pass

    def _action_scale(self):
        """커리큘럼 액션 스케일. total_steps 기준, start~end 구간에서 scale_start→scale_end.
        a=0 ↔ b_rl=0.5(중립값)이라 scale 작을수록 Dijkstra prior 지배."""
        cur = self.config.get("curriculum", {})
        if not cur.get("enabled", False):
            return 1.0
        s0, s1 = cur["scale_start"], cur["scale_end"]
        start, end = cur["start_step"], cur["end_step"]
        t = self.total_steps
        if t >= end:
            return s1
        if t <= start:
            return s0
        progress = (t - start) / max(1, (end - start))
        if cur.get("shape") == "linear":
            return float(s0 + (s1 - s0) * progress)
        return float(s0 * (s1 / s0) ** progress)  # geometric

    def resume_checkpoint(self):
        ckpt_path = os.path.join(self.config["resume_ckpt_dir"], "checkpoint.pt")
        if not os.path.exists(ckpt_path):
            print("[Resume] 체크포인트 없음. 처음부터 시작.")
            return

        try:
            ckpt = torch.load(
                ckpt_path, map_location=self.config["device"], weights_only=False
            )

            self.learner.actor.load_state_dict(ckpt["policy_state_dict"])
            self.learner.critic.load_state_dict(ckpt["critic_state_dict"])
            self.learner.actor_optimizer.load_state_dict(
                ckpt["policy_optimizer_state_dict"]
            )
            self.learner.critic_optimizer.load_state_dict(
                ckpt["critic_optimizer_state_dict"]
            )

            if "buffer" in ckpt and ckpt["buffer"] is not None:
                self.learner.buffer = ckpt["buffer"]

            meta = ckpt.get("meta_data", {})
            self.total_steps = meta.get("timestep", 0)
            self.sub_step_count = meta.get("sub_step_count", 0)
            self.sub_episode_count = meta.get("sub_episode_count", 0)

            if "delta_normalizer" in meta:
                dn = meta["delta_normalizer"]
                self.delta_normalizer.mean = dn["mean"]
                self.delta_normalizer.var = dn["var"]
                self.delta_normalizer.count = dn["count"]
                self.delta_normalizer_frozen = bool(dn.get("frozen", False))

            if "state_normalizer" in meta:
                sn = meta["state_normalizer"]
                self.state_normalizer.mean = np.array(sn["mean"], dtype=np.float32)
                self.state_normalizer.var = np.array(sn["var"], dtype=np.float32)
                self.state_normalizer.count = sn["count"]
                self.state_normalizer.is_fixed = sn["is_fixed"]
                print(
                    f"[Resume] StateNormalizer 복원 (count: {sn['count']}, fixed: {sn['is_fixed']})"
                )

            if "parameterDw" in meta:
                self.parameterDw = meta["parameterDw"]

            print(f"[Resume] 체크포인트 로드 완료! Step: {self.total_steps}")

        except Exception as e:
            print(f"[Resume] 로드 실패: {e}. 처음부터 시작.")

    # ============================================================
    # Job 집계 헬퍼 (새 시뮬레이터가 주는 JOB_DIC 활용)
    # Job state: 1=QUEUED, 2=RESERVED, 3=WAITING, 5=TRANSFERRING, 7=COMPLETED
    # ============================================================
    def _job_stats(self, pclient):
        jobs = list(getattr(pclient, "JOB_DIC", {}).values())
        queued = float(getattr(pclient, "QueuedCommandCount", 0) or 0)
        if not jobs:
            return {
                "queued": queued,
                "mean_wait_pri": 0.0,
                "mean_reassign": 0.0,
                "sum_wait_pri": 0.0,
                "sum_reassign": 0.0,
            }
        wait_pri = [
            max(0.0, float(getattr(j, "Priority", 0) or 0))
            for j in jobs
            if getattr(j, "State", 0) in (1, 3)
        ]
        reassign = [max(0.0, float(getattr(j, "ReAssignCount", 0) or 0)) for j in jobs]
        return {
            "queued": queued,
            "mean_wait_pri": float(np.mean(wait_pri)) if wait_pri else 0.0,
            "mean_reassign": float(np.mean(reassign)) if reassign else 0.0,
            "sum_wait_pri": float(np.sum(wait_pri)) if wait_pri else 0.0,
            "sum_reassign": float(np.sum(reassign)) if reassign else 0.0,
        }

    def _job_pressure(self, pclient):
        """우선순위-가중 대기 + 재할당 churn. 클수록 나쁨. delta로 사용."""
        s = self._job_stats(pclient)
        return 0.1 * s["mean_wait_pri"] + 1.0 * s["mean_reassign"]

    # ============================================================
    # Raw observation (정규화 전)
    # ============================================================
    def _build_raw_obs(self, pclient):
        sorted_rail_ids = sorted(pclient.RAILLINE_DIC.keys())
        raw_obs_list = []

        oht_all = list(pclient.OHT_DIC.values())
        oht_states_list = [o.State for o in oht_all]

        js = self._job_stats(pclient)

        global_features = [
            # --- 글로벌 시스템 지표 ---
            float(pclient.TotalTat),  # 전체 평균 TAT
            float(pclient.TotalOhtOperationRate),  # 전체 OHT 가동률 (낮을수록 좋음)
            # --- 글로벌 OHT 상태 분포 ---
            float(oht_states_list.count(0)),  # IDLE
            float(oht_states_list.count(2)),  # MOVE_TO_LOAD
            float(oht_states_list.count(3)),  # LOADING
            float(oht_states_list.count(4)),  # MOVE_TO_UNLOAD
            float(oht_states_list.count(5)),  # UNLOADING
            # --- 글로벌 Job 상태 (새 시뮬레이터 추가 정보) ---
            float(js["queued"]),  # 미할당 Job 수 (backlog)
            float(js["mean_wait_pri"]),  # 대기 Job 평균 우선순위
            float(js["mean_reassign"]),  # 활성 Job 평균 재할당 횟수
        ]

        for rail_id in sorted_rail_ids:
            rail = pclient.RAILLINE_DIC[rail_id]

            rail_features = [
                # --- 레일 혼잡도 (동적) ---
                float(rail.PredictedOHTCount),  # 진입 예정 OHT
                float(self.parameterDw.get(rail_id, 1)),  # 과거 지연 가중치
                float(self.parameterC.get(rail_id, 0)),  # 미래 OHT 수
                # --- 레일 물리 속성 (고정) ---
                float(rail.DistancePerVelocity),  # 빈 레일 통과 시간
                float(rail.Distance),  # 레일 길이
                # --- 레일 구조 정보 (고정) ---
                float(rail.PortCount),  # Port 수
                float(rail.DivergingLineCount),  # 분기 Line 수
                float(rail.Level2JoiningLineCount),  # 합류 Line 수
            ]
            raw_obs_list.append(
                np.array(global_features + rail_features, dtype=np.float32)
            )

        return raw_obs_list, sorted_rail_ids

    # ============================================================
    # Normalized observation (네트워크 입력용)
    # ============================================================
    def get_local_observations(self, pclient):
        raw_obs_list, sorted_rail_ids = self._build_raw_obs(pclient)
        if not self.config.get("use_state_normalizer", True):
            return np.array(raw_obs_list, dtype=np.float32), sorted_rail_ids
        normalized_list = []
        for obs in raw_obs_list:
            self.state_normalizer.update(obs)
            normalized_list.append(self.state_normalizer.normalize(obs))
        return np.array(normalized_list, dtype=np.float32), sorted_rail_ids

    # ============================================================
    # Local Reward (학습용)
    # ============================================================
    def get_local_reward(self, rail, pclient):
        lc = self.config.get("reward_components", {}).get("local", {})
        oht_count = len(rail.OhtList)
        predicted = rail.PredictedOHTCount
        idle_blocking = rail.IdleOHTCount
        rail_ohts = [pclient.OHT_DIC[o] for o in rail.OhtList if o in pclient.OHT_DIC]
        avg_stop = np.mean([o.StopTime for o in rail_ohts]) if rail_ohts else 0.0
        capacity_ratio = oht_count / max(1, rail.PortCount + 1)
        local_r = -(
            (0.3 * oht_count        if lc.get("use_oht_count", True) else 0.0)
            + (0.2 * predicted      if lc.get("use_predicted", True) else 0.0)
            + (0.3 * avg_stop       if lc.get("use_stop",      True) else 0.0)
            + (0.1 * idle_blocking  if lc.get("use_idle",      True) else 0.0)
            + (0.1 * capacity_ratio if lc.get("use_capacity",  True) else 0.0)
        )
        return local_r

    # ============================================================
    # Rail 단위 TAT 귀속 (job 완료시, 지나온 rail들에 머문시간 비중대로 분배)
    # ============================================================
    def _update_rail_tat_credit(self, pclient):
        """매 스텝 1회 호출. OHT.PassTimes(직전 통신구간 동안 지나온 rail+소요시간)를
        OHT별로 누적해두고, OHT.CmdCompleteTat(방금 완료된 job의 실제 TAT, 시뮬레이터
        제공)이 뜨면 그 누적 경로의 머문시간 비중대로 TAT 초과분을 rail별로 분배한다.
        분배 후 그 OHT의 누적 경로는 비우고 다음 job을 위해 새로 쌓기 시작한다."""
        self._rail_tat_credit = {}
        tat_ref = float(self.config.get("tat_base", 2.90706 * 60))
        for oht_id, oht in pclient.OHT_DIC.items():
            pass_times = getattr(oht, "PassTimes", None)
            if pass_times:
                route = self._oht_route_accum.setdefault(oht_id, [])
                for pt in pass_times:
                    route.append((pt.ID, float(pt.PassTime)))

            cct = getattr(oht, "CmdCompleteTat", None)
            if not cct:
                continue
            route = self._oht_route_accum.get(oht_id, [])
            total_time = sum(t for _, t in route)
            for c in cct.values():
                cmd_tat = float(getattr(c, "CmdTat", 0.0))
                if cmd_tat <= 0.0:
                    continue
                rel_excess = (cmd_tat - tat_ref) / tat_ref
                if route and total_time > 0:
                    for rail_id, t in route:
                        self._rail_tat_credit[rail_id] = (
                            self._rail_tat_credit.get(rail_id, 0.0) + rel_excess * (t / total_time)
                        )
                else:
                    # 누적된 경로가 없으면(예: job이 한 통신구간 안에 즉시 끝난 경우)
                    # 최소한 현재 rail 1개에는 귀속시킨다.
                    fallback = oht.RouteList[0] if getattr(oht, "RouteList", None) else None
                    if fallback is not None:
                        self._rail_tat_credit[fallback] = self._rail_tat_credit.get(fallback, 0.0) + rel_excess
            self._oht_route_accum[oht_id] = []

    def _rail_tat_penalty(self, rail):
        return float(self._rail_tat_credit.get(rail.ID, 0.0))

    def _rail_tat_diag(self):
        """rail_tat은 sparse event(대부분 rail이 0)라서 마지막 rail 값 하나만 보면
        거의 항상 0으로 보임. 이번 스텝에 credit을 받은 rail 전체에 대한 집계를 따로 로깅."""
        vals = list(self._rail_tat_credit.values())
        return {
            "reward/rail_tat_event_count": float(len(vals)),
            "reward/rail_tat_mean": float(np.mean(vals)) if vals else 0.0,
            "reward/rail_tat_max": float(np.max(vals)) if vals else 0.0,
            "reward/rail_tat_sum": float(np.sum(vals)) if vals else 0.0,
        }

    # ============================================================
    # Global Reward (delta 기반, job 정보 반영)
    # ============================================================
    def get_global_reward(self, pclient):
        """Windowed marginal TAT(주신호) + op 보조 + backlog 페널티.
        누적 TotalTat의 staleness(에피소드 중반 이후 1-step delta≈0)를 피하려고
        '이번 스텝에 완료된 job들의 평균 TAT'를 EMA로 추정해 baseline 대비 상대개선으로
        보상. 정규화 방식은 원래(버전 C 이전)의 온라인 delta_normalizer + warmup 후 freeze
        + clip(±5)으로 복원(버전 G) — 리워드 공식(total_completed_jobs 누적, tat_base 보정,
        tat/backlog 재보정 weight)만 새 버전이고 정규화 방식은 고정해서 비교 가능하게 함."""
        w = self.config["reward_weights"]
        cur_tat = float(pclient.TotalTat)
        cur_op = float(pclient.TotalOhtOperationRate)
        # CompletedCommandCount는 누적이 아니라 "이번 스텝에 새로 완료된 job 수"(delta).
        # TotalTat(누적평균)과 짝을 맞추려면 직접 누적해서 총 완료수를 만들어야 함.
        completed_delta = float(getattr(pclient, "CompletedCommandCount", 0) or 0)
        self.total_completed_jobs += completed_delta
        cur_completed = self.total_completed_jobs
        cur_waiting = float(getattr(pclient, "WaitingCommandCount", 0) or 0)
        cur_queued = float(getattr(pclient, "QueuedCommandCount", 0) or 0)
        tat_ref = float(self.config.get("tat_base", 2.90706 * 60))

        if self.prev_completed is None:
            self.prev_tat = cur_tat
            self.prev_op_rate = cur_op
            self.prev_completed = cur_completed
            self.prev_tat_sum = cur_tat * cur_completed
            return 0.0

        # (1) windowed marginal TAT — 누적평균(TotalTat)의 staleness 제거.
        #     누적합 = TotalTat × 완료수. 두 스텝 차이를 새 완료수로 나누면 그 구간 평균 TAT.
        cur_tat_sum = cur_tat * cur_completed
        dN = cur_completed - self.prev_completed
        if dN > 0:
            marg_tat = (cur_tat_sum - self.prev_tat_sum) / dN
            beta = float(self.config.get("tat_ema_beta", 0.05))
            self._tat_ema = (
                marg_tat
                if self._tat_ema <= 0.0
                else (1.0 - beta) * self._tat_ema + beta * marg_tat
            )
        tat_term = (tat_ref - self._tat_ema) / tat_ref  # 무차원, baseline보다 빠르면 +
        self._last_marg_tat = self._tat_ema

        # (2) op 감소분 (보조). throughput 고정이라 op는 TAT 보조 프록시 → 작은 가중치.
        d_op = self.prev_op_rate - cur_op

        # (3) backlog 수준 페널티 (혼잡 standing). Waiting+Queued 둘 다 live로 수신.
        backlog = cur_waiting + cur_queued
        self._last_backlog = backlog

        gc = self.config.get("reward_components", {}).get("global", {})
        raw = (
            (w["tat"] * tat_term    if gc.get("use_tat",     True) else 0.0)
            + (w["op"] * d_op       if gc.get("use_op",      True) else 0.0)
            - (w["backlog"] * backlog if gc.get("use_backlog", True) else 0.0)
        )

        self.prev_tat = cur_tat
        self.prev_op_rate = cur_op
        self.prev_completed = cur_completed
        self.prev_tat_sum = cur_tat_sum

        # 정규화: warmup 후 일정 step 지나면 통계 고정(에피소드 간 비정상성/드리프트 차단)
        freeze_at = self.config["warmup_steps"] + int(
            self.config.get("reward_freeze_after", 20000)
        )
        if (not self.delta_normalizer_frozen) and self.total_steps >= freeze_at:
            self.delta_normalizer_frozen = True
            print(
                f"[RewardNorm] 정규화 통계 고정 (step={self.total_steps}, "
                f"mean={self.delta_normalizer.mean:.4f}, count={self.delta_normalizer.count})"
            )
        if self.delta_normalizer_frozen:
            std = (
                np.sqrt(self.delta_normalizer.var / max(1, self.delta_normalizer.count))
                + 1e-8
            )
            g = (raw - self.delta_normalizer.mean) / std
        else:
            g = self.delta_normalizer.update_and_normalize(raw)
        clip = float(self.config.get("global_reward_clip", 5.0))
        return float(np.clip(g, -clip, clip))

    def get_step_reward(self, rail, pclient):
        """reward_components config로 항별 on/off 제어.
        로깅용 계산은 항상 수행, step_reward 반영은 use_* 플래그로 결정.
        0704 재현: use_global+use_local=True, tat_dense=alpha, local=1-alpha.
        reward_version J: use_global=True, use_local=False, tat_dense=0.3.
        reward_version I: use_global=False, use_local=False (rail_tat only)."""
        rc = self.config.get("reward_components", {})
        w = self.config.get("reward_weights", {})

        # 로깅용 (항상 계산)
        l_raw = self.get_local_reward(rail, pclient)
        l = self.local_reward_normalizer.update_and_normalize(l_raw)
        self._last_local_norm = l
        rail_tat_penalty = self._rail_tat_penalty(rail)
        self._last_rail_tat_penalty = rail_tat_penalty

        reward = 0.0
        if rc.get("use_rail_tat", True):
            reward -= float(w.get("rail_tat", 1.0)) * rail_tat_penalty
        if rc.get("use_global_reward", True):
            reward += float(w.get("global_alpha", 0.3)) * self._last_g
        if rc.get("use_local_reward", False):
            reward += float(w.get("local_alpha", 0.0)) * l
        return reward

    # ============================================================
    # Sub-terminal Reward (baseline 비교, 로깅)
    # ============================================================
    def get_sub_terminal_reward(self, pclient):
        current_tat = float(pclient.TotalTat)
        current_op = float(pclient.TotalOhtOperationRate)

        if not hasattr(self, "sub_start_tat") or self.sub_start_tat is None:
            self.sub_start_tat = current_tat
            self.sub_start_op = current_op
            return 0.0

        delta_tat = self.sub_start_tat - current_tat
        delta_op = self.sub_start_op - current_op
        sub_r = delta_tat + 100 * delta_op

        self.sub_start_tat = current_tat
        self.sub_start_op = current_op

        with open(self.reward_log_path, "a") as f:
            f.write(
                f"{self.total_steps},{self.sub_episode_count},"
                f"{current_tat:.4f},{current_op:.4f},{sub_r:.4f}\n"
            )

        print(
            f"[Sub-Terminal] delta_TAT: {delta_tat:.3f}, "
            f"delta_Op: {delta_op:.4f}, R: {sub_r:.3f}"
        )

        self._wlog(
            {
                "sub_terminal/tat": current_tat,
                "sub_terminal/op_rate": current_op,
                "sub_terminal/sub_r": sub_r,
                "sub_terminal/episode": self.sub_episode_count,
            },
            step=self.total_steps,
        )

        return 0

    # ============================================================
    # Terminal Reward — DB에서 계산, 로깅 전용
    # ============================================================
    def get_terminal_reward_log(self, episode_idx):
        db_filename = f"ep_0512_{episode_idx}.db"
        args = argparse.Namespace(
            save_dir=self.config["eval_db_dir"], db_filename=db_filename
        )
        try:
            result = evaluate(args)
            print(
                f"[Eval DB] TAT={result['tat']:.3f}, OpRate={result['operation_rate']:.3f}"
            )
        except Exception as e:
            print(f"[Eval DB] DB 로드 실패: {e}")

    # ============================================================
    # Baseline cost (warm-up 전용)
    # ============================================================
    def compute_baseline_params(self, pclient):
        parameterA = 0.1
        imaxCount = 10
        parameterC = {}

        if len(self.parameterDw) == 0:
            for id in pclient.RAILLINECOST_DIC:
                self.parameterDw[id] = 1
                self.parameterPassTimes[id] = []

        for id in pclient.RAILLINECOST_DIC:
            parameterC[id] = 0

        for id in pclient.OHT_DIC:
            oht = pclient.OHT_DIC[id]
            limitCount = len(oht.RouteList) - 1
            if imaxCount < len(oht.RouteList) and imaxCount > 0:
                limitCount = imaxCount
            if len(oht.RouteList) > 0:
                for i in range(1, min(limitCount + 1, len(oht.RouteList))):
                    if oht.RouteList[i] in parameterC:
                        parameterC[oht.RouteList[i]] += 1

        for id in pclient.OHT_DIC:
            oht = pclient.OHT_DIC[id]
            for passt in oht.PassTimes:
                lineid = passt.ID
                if lineid not in self.parameterPassTimes:
                    self.parameterPassTimes[lineid] = []
                if lineid not in pclient.RAILLINE_DIC:
                    continue
                ohtcount = len(oht.FrontOhts.get(lineid, []))
                if ohtcount == 0:
                    ohtcount = 1
                passtime = (
                    passt.PassTime - pclient.RAILLINE_DIC[lineid].DistancePerVelocity
                ) / ohtcount
                self.parameterPassTimes[lineid].append(passtime)

        for id in self.parameterPassTimes:
            if id not in self.parameterDw:
                self.parameterDw[id] = 1
            for i in range(len(self.parameterPassTimes[id])):
                self.parameterDw[id] += parameterA * (
                    self.parameterPassTimes[id][i] - self.parameterDw[id]
                )
            self.parameterPassTimes[id] = []

        self.parameterC = parameterC

    # ============================================================
    # Reset (에피소드 재시작, v==2)
    #   NOTE: episode_index 는 PClient(isEnd==1)에서 증가시키므로 여기선 안 함.
    # ============================================================
    def Reset(self, pclient):
        print("\n" + "=" * 50)
        print("[RESET] Episode Reset!")

        trcnt_total_new = self.total_completed_jobs
        tat_total_new = getattr(pclient, "TotalTat", 0.0)
        oht_operation_rate_new = getattr(pclient, "TotalOhtOperationRate", 0.0)
        trcnt_total_base = 175299
        tat_total_base = float(self.config.get("tat_base", 2.90706 * 60))  # pclient.TotalTat과 동일 단위(초)
        oht_operation_rate_base = 0.79292

        try:
            print("-" * 70)
            print(
                f"> Transfer Count : {trcnt_total_new} (Improvement: {(trcnt_total_new - trcnt_total_base) / trcnt_total_base * 100:.2f}%)"
            )
            print(
                f"> Total TAT      : {tat_total_new:.3f} (Improvement: {(tat_total_base - tat_total_new) / tat_total_base * 100:.2f}%)"
            )
            print(
                f"> Operation Rate : {oht_operation_rate_new:.3f} (Difference: {oht_operation_rate_base - oht_operation_rate_new:.3f})"
            )
            print("-" * 70)
        except ZeroDivisionError:
            print("Base metrics are zero.")

        self._wlog(
            {
                "episode/final_tat": float(tat_total_new),
                "episode/final_op_rate": float(oht_operation_rate_new),
                "episode/final_completed": float(trcnt_total_new),
                "episode/index": getattr(pclient, "episode_index", 0),
            },
            step=self.total_steps,
        )

        if (
            self.config.get("use_state_normalizer", True)
            and self.total_steps >= self.config["warmup_steps"]
            and not self.state_normalizer.is_fixed
        ):
            self.state_normalizer.fix()

        # 상태 초기화
        self.last_batched_states = None
        self.last_actions = None
        self.last_raw_obs = None
        self.prev_tat = None
        self.prev_op_rate = None
        self.prev_completed = None
        self.prev_jp = None
        self.sub_start_tat = None
        self.sub_start_op = None
        self.prev_tat_sum = None
        self.total_completed_jobs = 0  # 새 에피소드 시작, 누적 완료수 리셋
        self._tat_ema = 0.0
        self.pending_smooth = None
        self._oht_route_accum = {}  # 새 에피소드 시작, OHT 경로 누적 리셋
        self._rail_tat_credit = {}
        self.sub_step_count = 0
        self.sub_episode_count = 0

        print(
            f"[RESET] Episode {getattr(pclient, 'episode_index', 0)} 시작. Sub-episode 카운터 리셋."
        )
        print("=" * 70 + "\n")

    # ============================================================
    # Algorithm (메인 루프, v==0)
    # ============================================================
    def Algorithm(self, pclient):
        try:
            self._algorithm_impl(pclient)
        except Exception as e:
            # /loop 안정성: 한 step의 예외가 전체 학습을 죽이지 않도록 방어.
            import traceback

            print(f"[Algorithm] step 예외 무시하고 계속: {e}")
            traceback.print_exc()
            # 안전한 기본 cost 라도 보내서 시뮬레이터와 동기 유지
            try:
                for rail_id in pclient.RAILLINECOST_DIC:
                    base = pclient.RAILLINE_DIC[rail_id].DistancePerVelocity
                    pclient.RAILLINECOST_DIC[rail_id].FRailLineCost = base
            except Exception:
                pass

    def _algorithm_impl(self, pclient):
        print(
            f"episode index : {getattr(pclient, 'episode_index', 0)} | sub_step: {self.sub_step_count}"
        )

        done = False
        tat_threshold = 500
        if pclient.QueuedCommandCount > 500:
            print("Queued Command가 500건 이상입니다. Sim을 종료합니다.")
            pclient.SendIsEnd(1)
            done = True
        elif float(pclient.TotalTat) > tat_threshold and self.sub_step_count > 100:
            print(
                f"[Early Stop] TAT={pclient.TotalTat:.1f} > {tat_threshold:.1f}. 다음 에피소드로."
            )
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
        else:
            pclient.SendIsEnd(0)

        print("\n" + "=" * 60)

        # Warm-up
        is_warmup = self.total_steps < self.config["warmup_steps"]
        if is_warmup:
            current_batched_obs, sorted_rail_ids = self.get_local_observations(pclient)
            self.compute_baseline_params(pclient)

            for rail_id in sorted_rail_ids:
                base = pclient.RAILLINE_DIC[rail_id].DistancePerVelocity
                w = self.parameterDw.get(rail_id, 1)
                c = self.parameterC.get(rail_id, 0)
                cost = base + (w * c * 0.5)
                pclient.RAILLINECOST_DIC[rail_id].FRailLineCost = cost

            self.total_steps += 1
            print(
                f"total step : {self.total_steps} (Buffer: {self.learner.buffer.size})"
            )
            return

        # ============================================================
        # RL MODE
        # ============================================================
        current_batched_obs, sorted_rail_ids = self.get_local_observations(pclient)

        self.compute_baseline_params(pclient)

        if self.last_batched_states is not None:
            self.sub_step_count += 1
            is_sub_boundary = self.sub_step_count % self.config["sub_episode_len"] == 0

            sub_terminal_r = 0.0
            if is_sub_boundary:
                sub_terminal_r = self.get_sub_terminal_reward(pclient)
                self.sub_episode_count += 1
                print(
                    f"[Sub-episode {self.sub_episode_count} 완료] Step {self.sub_step_count}"
                )

            # 글로벌 보상은 글로벌 양(TAT/가동률/완료 등)이라 스텝당 1회만 계산.
            # (라인마다 호출하면 prev_*가 갱신돼 2번째 라인부터 delta=0이 되는 버그 방지)
            g = self.get_global_reward(pclient)
            self._last_g = g
            self._update_rail_tat_credit(pclient)

            w_smooth = float(self.config["reward_weights"].get("smooth", 0.0))
            reward = 0.0
            for i in range(len(sorted_rail_ids)):
                rail = pclient.RAILLINE_DIC[sorted_rail_ids[i]]
                reward = self.get_step_reward(rail, pclient)
                if is_sub_boundary:
                    reward += sub_terminal_r / len(sorted_rail_ids)
                # 액션 스무딩: 직전 실행 액션의 step간 변화 페널티(route thrash 억제)
                if (
                    w_smooth > 0.0
                    and self.pending_smooth is not None
                    and i < len(self.pending_smooth)
                ):
                    reward -= w_smooth * float(self.pending_smooth[i])
                self.learner.buffer.push(
                    self.last_batched_states[i],
                    self.last_actions[i],
                    reward,
                    current_batched_obs[i],
                    float(done),
                )

            self.total_steps += 1
            print(
                f"total step : {self.total_steps} (Buffer: {self.learner.buffer.size})"
            )

            if self.total_steps > self.config["update_after"]:
                self.learner.learn()
                if self.total_steps % 2 == 0:
                    print(
                        f"Step: {self.total_steps} | R: {reward:.4f} | "
                        f"Q1 Loss: {self.learner.total_losses.get('q1', 0):.4f}"
                    )

                if self.config["save_model"] and (
                    self.total_steps % self.config["model_checkpoint_freq"] == 0
                ):
                    (
                        policy,
                        critic,
                        pol_opt,
                        crit_opt,
                        enc_opt,
                        log_alpha,
                        alpha_opt,
                        _,
                    ) = self.learner.get_params()
                    meta_data = {
                        "algorithm": str(self.config["algorithm"]["name"]),
                        "timestep": self.total_steps,
                        "log_datetime": datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
                        "critic_hidden_dims": list(self.config["critic_hidden_dims"]),
                        "actor_hidden_dims": list(self.config["actor_hidden_dims"]),
                        "sub_episode_count": self.sub_episode_count,
                        "delta_normalizer": {
                            "mean": self.delta_normalizer.mean,
                            "var": self.delta_normalizer.var,
                            "count": self.delta_normalizer.count,
                            "frozen": self.delta_normalizer_frozen,
                        },
                        "state_normalizer": {
                            "mean": self.state_normalizer.mean.tolist(),
                            "var": self.state_normalizer.var.tolist(),
                            "count": self.state_normalizer.count,
                            "is_fixed": self.state_normalizer.is_fixed,
                        },
                        "parameterDw": dict(self.parameterDw),
                    }
                    self.logger.log(
                        policy,
                        critic,
                        pol_opt,
                        crit_opt,
                        enc_opt,
                        meta_data,
                        log_alpha,
                        alpha_opt,
                        None,
                        best_update_flag=False,
                        cur_update_flag=False,
                    )
                    print(
                        f"\n{'-' * 70}\n Checkpoint Saved! (Step: {self.total_steps})\n{'-' * 70}\n"
                    )

                if self.total_steps % 100 == 0:
                    actions_np = np.array(
                        [
                            float(self.last_actions[i])
                            for i in range(len(sorted_rail_ids))
                        ]
                    )
                    oht_all = list(pclient.OHT_DIC.values())
                    oht_states_list = [o.State for o in oht_all]
                    js = self._job_stats(pclient)
                    self._wlog(
                        {
                            "step": self.total_steps,
                            "episode": getattr(pclient, "episode_index", 0),
                            "sub_episode": self.sub_episode_count,
                            "reward/step_reward": reward,
                            "reward/global_norm": float(getattr(self, "_last_g", 0.0)),
                            "reward/local_norm": float(
                                getattr(self, "_last_local_norm", 0.0)
                            ),
                            "reward/rail_tat": float(
                                getattr(self, "_last_rail_tat_penalty", 0.0)
                            ),
                            **self._rail_tat_diag(),
                            "reward/alpha": float(self.config.get("reward_alpha", 0.5)),
                            "reward/marg_tat": float(
                                getattr(self, "_last_marg_tat", 0.0)
                            ),
                            "reward/backlog": float(
                                getattr(self, "_last_backlog", 0.0)
                            ),
                            "reward/smooth_mean": (
                                float(np.mean(self.pending_smooth))
                                if self.pending_smooth is not None
                                else 0.0
                            ),
                            "curriculum/action_scale": float(self._action_scale()),
                            "system/tat": float(pclient.TotalTat),
                            "system/op_rate": float(pclient.TotalOhtOperationRate),
                            "system/queued_jobs": float(pclient.QueuedCommandCount),
                            "system/completed": float(
                                getattr(pclient, "CompletedCommandCount", 0)
                            ),
                            # Job 지표 (새 정보)
                            "job/mean_wait_priority": js["mean_wait_pri"],
                            "job/mean_reassign": js["mean_reassign"],
                            "job/queued": js["queued"],
                            "oht/idle_count": float(oht_states_list.count(0)),
                            "oht/move_to_load": float(oht_states_list.count(2)),
                            "oht/move_to_unload": float(oht_states_list.count(4)),
                            "oht/loading": float(oht_states_list.count(3)),
                            "oht/unloading": float(oht_states_list.count(5)),
                            "action/mean": float(np.mean(actions_np)),
                            "action/std": float(np.std(actions_np)),
                            "action/min": float(np.min(actions_np)),
                            "action/max": float(np.max(actions_np)),
                            "b_rl/mean": float(np.mean(0.5 + 0.5 * actions_np)),
                            "b_rl/std": float(np.std(0.5 + 0.5 * actions_np)),
                            # TD3: critic/actor/q1/q2  ·  TD7: + encoder (없는 키는 0)
                            "loss/q1": self.learner.total_losses.get("q1", 0),
                            "loss/q2": self.learner.total_losses.get("q2", 0),
                            "loss/critic": self.learner.total_losses.get("critic", 0),
                            "loss/actor": self.learner.total_losses.get("actor", 0),
                            "loss/encoder": self.learner.total_losses.get("encoder", 0),
                            "buffer/size": self.learner.buffer.size,
                        },
                        step=self.total_steps,
                    )

                    if self.total_steps % 1000 == 0 and self.wandb_ok:
                        self._wlog(
                            {
                                "hist/action_distribution": wandb.Histogram(actions_np),
                                "hist/b_rl_distribution": wandb.Histogram(
                                    0.5 + 0.5 * actions_np
                                ),
                            },
                            step=self.total_steps,
                        )

            if is_sub_boundary:
                self.prev_tat = None
                self.prev_op_rate = None
                self.prev_completed = None
                self.prev_jp = None

        if done:
            self.last_batched_states = None
            self.last_actions = None
            self.last_raw_obs = None
            return

        # 모델 추론
        batched_actions, _ = self.learner.actor.get_action(
            current_batched_obs, eval=False
        )

        # ── Curriculum: 초반엔 액션을 작게(→ Dijkstra baseline), 점차 키움 ──
        scale = self._action_scale()
        batched_actions = batched_actions * scale  # a=0 ↔ b_rl=0.5 (중립값)

        # 액션 스무딩용: 이번 액션(a_t)과 직전 실행 액션(a_{t-1})의 라인별 변화량.
        # 다음 스텝 push에서 a_t의 결과 보상에 -smooth·|Δa|로 반영(thrash 억제).
        if self.last_actions is not None and len(self.last_actions) == len(
            batched_actions
        ):
            _da = np.asarray(batched_actions, dtype=np.float32) - np.asarray(
                self.last_actions, dtype=np.float32
            )
            self.pending_smooth = (
                np.abs(_da).reshape(len(batched_actions), -1).mean(axis=1)
            )
        else:
            self.pending_smooth = None

        self.last_batched_states = current_batched_obs
        self.last_actions = batched_actions  # 실제 실행된(damped) 액션을 버퍼에 저장
        self.last_raw_obs = None

        for i, rail_id in enumerate(sorted_rail_ids):
            base = pclient.RAILLINE_DIC[rail_id].DistancePerVelocity
            w = self.parameterDw.get(rail_id, 1)
            c = self.parameterC.get(rail_id, 0)

            a_i = float(batched_actions[i])
            b_rl = 0.5 + 0.5 * a_i  # [0, 1]
            cost = base + (w * c * b_rl)
            pclient.RAILLINECOST_DIC[rail_id].FRailLineCost = cost

    # ============================================================
    # 기존 함수들
    # ============================================================
    def AlgorithmAfter(self, pclient):
        for id in pclient.RAILLINECOST_DIC:
            railLine = pclient.RAILLINE_DIC[id]
            if len(railLine.OhtList) > 3 and railLine.IdleOHTCount < 1:
                pass

    def UpdateDatas(self, pclient):
        pass

    def UpdateOHTRoute(self, pclient, ohtId, curLine, destination, route_list):
        import heapq
        from collections import deque

        v = defaultdict(lambda: float("inf"))
        parent = defaultdict(int)
        v[curLine] = 0
        parent[curLine] = -1
        heap = [(0, curLine)]
        while heap:
            weight, cur_line = heapq.heappop(heap)
            for next_id in pclient.RAILLINE_DIC[cur_line].DivergingLineIDList:
                nw = pclient.RAILLINE_DIC[next_id].Distance + weight
                if nw < v[next_id] and nw < v[destination]:
                    v[next_id] = nw
                    parent[next_id] = cur_line
                    if next_id != destination:
                        heapq.heappush(heap, (nw, next_id))
        nxt = destination
        route = deque([destination])
        while nxt != -1:
            cur = nxt
            nxt = parent[cur]
            if nxt != -1:
                route.appendleft(nxt)
        pclient.OHT_DIC[ohtId].RouteList = route_list

    def DijkStraRoute(self, pclient, curLine, destination):
        import heapq
        from collections import deque

        v = defaultdict(lambda: float("inf"))
        parent = defaultdict(int)
        v[curLine] = 0
        parent[curLine] = -1
        heap = [(0, curLine)]
        while heap:
            weight, cur_line = heapq.heappop(heap)
            for next_id in pclient.RAILLINE_DIC[cur_line].DivergingLineIDList:
                nw = pclient.RAILLINE_DIC[next_id].Distance + weight
                if nw < v[next_id] and nw < v[destination]:
                    v[next_id] = nw
                    parent[next_id] = cur_line
                    if next_id != destination:
                        heapq.heappush(heap, (nw, next_id))
        nxt = destination
        route = deque([destination])
        while nxt != -1:
            cur = nxt
            nxt = parent[cur]
            if nxt != -1:
                route.appendleft(nxt)
        return route

    def ReRoute(self, pclient, oht_list, from_nodes, to_nodes):
        route_dict = {}
        for idx, oht_id in enumerate(oht_list):
            route_dict[oht_id] = self.DijkStraRoute(
                pclient, from_nodes[idx], to_nodes[idx]
            )
        return route_dict

    def Assign(self, pclient, job_list):
        oht_dic = defaultdict(int)
        v = set()
        for job in job_list:
            for ohtId, oht in pclient.OHT_DIC.items():
                flag = any(ct in job.CarrierTypes for ct in oht.CarrierTypes)
                if not flag:
                    continue
                flag = oht.RunningAreaType in job.RunningAreaTyes
                flag2 = flag or oht.RunningAreaType == 0 or job.RunningAreaTyes[0] == 0
                if oht.State != 3 and flag and flag2 and ohtId not in v:
                    for index, r in enumerate(oht.RouteList):
                        if index > 15:
                            break
                        if index > 5:
                            continue
                        if r == job.FromNode:
                            oht_dic[job.ID] = ohtId
                            v.add(ohtId)
                            break
                    else:
                        continue
                    break
        return oht_dic

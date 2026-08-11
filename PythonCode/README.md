# Contextual TD7 for OHT Routing

반도체 FAB OHT의 rail cost를 학습해 혼잡 구간을 우회시키는 contextual TD7 구현입니다.
기본 학습 구성은 **Directional Context Encoder + TD7 + SALE + LAP**입니다.

## 핵심 구조

- 제어 단위: boundary를 제외한 rail 4,996개
- 관측: center rail, incoming 10개, outgoing 10개, global state
- Actor 출력: rail별 raw action `a ∈ [-1, 1]`
- Critic/SALE 입력: 실제 환경에 적용된 `applied_action`
- Replay: 한 simulator step을 snapshot으로 저장하고 rail 단위로 sampling
- TCP: Python server가 simulator와 state/cost를 교환

주요 파일:

```text
main_contextual.py              실행 및 TCP loop
ClientAlgorithm_contextual.py   action, reward, replay, 학습 runtime
contextual_*.py                 topology, observation, action, reward, W&B
cocel_rl/algorithms/contextual_td7/
                                encoder, actor, critic, SALE, LAP, checkpoint
tests/                          contextual 회귀 테스트
```

## Action 계약

기본값은 `region_b_rl`입니다.

```text
policy_action      = actor(state)
exploratory_action = clip(policy_action + noise, -1, 1)
applied_action     = curriculum_scale × exploratory_action
b_rl               = 0.5 + 0.5 × applied_action
cost               = base + w × c × b_rl
```

- warm-up: `applied_action=0`, `b_rl=0.5`
- step 10,000부터 scale `0.05`
- step 40,000까지 geometric 방식으로 scale `1.0` 도달
- `c=0`인 rail은 action에 의해 cost가 변하지 않음

기존 residual 방식은 ablation으로 유지합니다.

```powershell
--action-mode exp_residual --action-scale 0.05
```

## 학습 실행

저장소 루트에서:

```powershell
conda activate aicc

python .\PythonCode\main_contextual.py `
  --mode training `
  --action-enabled `
  --action-mode region_b_rl `
  --sale `
  --lap `
  --critic-loss-mode auto `
  --seed 0 `
  --exploration-noise-std 0.10 `
  --exploration-noise-final-std 0.02 `
  --exploration-noise-anneal-steps 100000 `
  --exploration-noise-clip 0.20 `
  --warmup-steps 10000 `
  --normalizer-freeze-steps 10000 `
  --rail-free-flow-reference-config .\diagnostics\rail_free_flow_reference.json `
  --reward-diagnostic-dir .\diagnostics\reward_j `
  --curriculum-end-step 20000 `
  --curriculum-scale-start 0.05 `
  --curriculum-scale-end 1.0 `
  --curriculum-shape geometric `
  --smooth-b-rl-weight 0.25 `
  --minimum-replay-env-steps 100 `
  --minimum-action-enabled-env-steps 100 `
  --replay-capacity-env-steps 10000 `
  --batch-size 1024 `
  --updates-per-env-step 1 `
  --learn-every-env-steps 1 `
  --sim-end-time 45000 `
  --wandb `
  --smoke-report .\contextual_training_noiseanneal_seed0.json `
  --checkpoint-root .\checkpoints\contextual_noiseanneal_seed0
```

The Reward I predecessor uses the fixed-scale contextual reward directly; reward normalizers
are inactive and there is no reward-normalizer CLI switch. Observation/state
normalization is unchanged. `--reward-diagnostic-dir` is optional; when it is
omitted, no Reward I JSONL records or free-flow occurrence diagnostics are
created. The default diagnostic windows are
`0:1000,10000:11000,20000:21000` and can be overridden with
`--reward-diagnostic-windows`.

Its fixed reward for controlled rail `i` is
`0.5 * global_raw + 0.5 * (local_raw / 3) - clip(100 * rail_tat_raw, -0.5, 0.5) - 0.25 * abs(delta_b_rl)`.

## Reward J Phase 1: free-flow neutral-2 rail reward

Reward J replaces only the rail term with
`2.0 - route_time / route_free_flow_time`. A ratio below 2 is rewarded, 2 is
neutral, and a ratio above 2 is penalized. Phase 1 uses `rail_tat_weight=1.0`
and per-rail clipping to `[-1.0, 1.0]`.
The global raw, local `/3`, idle-off, and `0.25 * abs(delta_b_rl)` terms from
Reward I are unchanged.

No offline baseline calibration or reference file is required. The neutral
ratio can be overridden with `--rail-free-flow-neutral-ratio`, but defaults to
`2.0`. Existing Reward I checkpoints and replay are incompatible with Reward J.

## Reward K Phase 2 provisional first run

The first balanced-scale run keeps `tat_reference=174.4236`, neutral ratio
`2.0`, both global/local alpha values at `0.5`, and smooth weight at `0.25`.
Its provisional coefficients are `tat_weight=18.4`,
`backlog_weight=0.0005`, `local_predicted_oht_weight=0.10`,
`local_reward_scale=2.0`, `rail_tat_weight=50.0`, and
`rail_tat_clip=1.0`. These are first-run values, not validated calibration
results. W&B and optional reward JSONL record representative global, local,
active-rail scales, good/bad reward signs, clipping, and Q-mean deltas for the
next calibration pass. Reward I/J checkpoints and replay are incompatible.

## Reward Q: Reward N pressure ablation without idle reserve

Reward Q preserves the reward function used by W&B run `9qokskqq`: the global
TAT raw term is the one-sided, unbounded
`-11 * max(0, TotalTat - 160) / 165`, with no `tat_raw_clip`. For
`B_t = Queued_t + Waiting_t`, its global raw reward is
`tat_raw + 4*(0.8-OP) - 0.0008*B_t`
`- 0.24*clip(max(0, B_t-B_{t-300})/30, 0, 1)`. The idle-reserve term is
disabled with `idle_reserve_weight=0.0`; global alpha remains `0.5`.

The ablation uses `local_predicted_oht_weight=0.10`, `rail_tat_weight=40`,
`rail_tat_clip=1`, local divisor `2`, and smooth weight `0.25`. It restores the
actual `9qokskqq` termination configuration: 10,000-step episode grace,
`TotalTat >= 200` for 300 consecutive steps, then one replayed terminal
penalty of `-20`. Reward Q requires fresh critic, target critic, replay, and
reward-version state.

실행 후 simulator GUI에서 Python 연동을 활성화하고 Run을 시작합니다.

## Command 6 dispatch mode

기본 배차는 기존과 동일한 first-match입니다.

```powershell
python .\PythonCode\main_contextual.py --dispatch-mode first-match
```

command 0에서 실제 적용한 rail cost의 pickup 경로 누적합이 가장 작은
후보 OHT를 선택하려면:

```powershell
python .\PythonCode\main_contextual.py --dispatch-mode cost
```

두 모드 모두 기존 eligibility 조건과 `RouteList[:16]` 후보 범위를
그대로 사용합니다.

## Checkpoint와 재개

기본 저장 위치:

```text
checkpoints/<algorithm_variant>_<action_version>/
├─ latest/checkpoint.pt
├─ periodic/step_*.pt
└─ crash/checkpoint.pt
```

재개:

```powershell
--resume-checkpoint .\checkpoints\<variant>\latest\checkpoint.pt
```

- action/reward/network/config 버전이 정확히 같아야 로드됨
- replay buffer는 checkpoint에 저장되지 않음
- 재시작 후 로드한 policy로 replay를 다시 채운 뒤 학습 재개
- action mode 변경 시 기존 checkpoint 사용 금지
- crash checkpoint는 진단용이며 재개 불가

## Logging

`--wandb` 사용 시 project `oht-routing-contextual-td7`에 기록합니다.

- `action/*`, `b_rl/*`: policy, exploration, applied action
- `reward/*`: global/local/TAT/smooth reward
- `replay/*`, `learner/*`, `critic/*`: buffer와 학습 상태
- `global/*`, `job/*`, `oht/*`: simulator 상태
- `runtime/*`: observation, inference, update, send 시간

run 이름과 config는 `contextual_wandb.py`의 `EXP_META`에서 생성됩니다.

## Contextual twin-critic diagnostics

`critic/q_abs_diff_mean` and `critic/q_abs_diff_max` are the mean and maximum
absolute differences between Q1 and Q2 outputs for the current replay batch.
`critic/q1_loss` and `critic/q2_loss` are the separate losses whose sum is
`learner/critic_loss`. `critic/q1_grad_norm` and
`critic/q2_grad_norm` are the finite, pre-clip L2 gradient norms of each
Q-exclusive head.

`critic/parameter_l2_distance` and
`critic/parameter_max_abs_diff` compare corresponding trainable parameters in
the Q1- and Q2-exclusive heads. They deliberately exclude the shared SALE
state-action projection and the separately optimized contextual encoder.

Actor metrics are stateful to avoid logging-interval/policy-delay aliasing.
`learner/actor_updated_this_step` is the only current-step 0/1 flag.
`learner/actor_loss_last`, `learner/actor_grad_norm_last`, and
`learner/actor_last_update_step` retain the most recent real actor update, while
`learner/actor_updates_total` is cumulative.

## 테스트

```powershell
$env:PYTHONPATH="$PWD;$PWD\PythonCode;$PWD\PythonCode\tests"
python -m unittest discover -s .\PythonCode\tests -p "test_contextual*.py"
```

현재 contextual 회귀 테스트: **157개**.


## Ablation study

```powershell

#1 
python .\PythonCode\main_contextual.py `
  --mode training `
  --action-enabled `
  --no-lap `
  --replay-buffer-rail `
  --batch-size 1024 `
  --wandb

#2 
python .\PythonCode\main_contextual.py `
  --mode training `
  --action-enabled `
  --no-sale `
  --replay-buffer-rail `
  --batch-size 1024 `
  --wandb

#3 
python .\PythonCode\main_contextual.py `
  --mode training `
  --action-enabled `
  --no-sale `
  --no-lap `
  --replay-buffer-rail `
  --batch-size 1024 `
  --wandb

#4
python main_contextual.py \
  --mode training \
  --action-enabled \
  --replay-buffer-snapshot  \
  --batch-size 4 \
  --no-lap \
  --wandb

python .\PythonCode\main_contextual.py `
  --mode training `
  --action-enabled `
  --no-lap `
  --replay-buffer-rail `
  --batch-size 1024 `
  --dispatch-mode cost `
  --wandb


```

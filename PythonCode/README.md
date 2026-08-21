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
  --reward-version E `
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
  --terminate-on-warmup-complete `
  --normalizer-freeze-steps 10000 `
  --save-state-normalizer .\normalizers\contextual_state_e_seed0.npz `
  --save-reward-normalizer `
  --reward-diagnostic-dir .\diagnostics\reward_e_fixedscale1 `
  --curriculum-end-step 20000 `
  --curriculum-scale-start 1.0 `
  --curriculum-scale-end 1.0 `
  --curriculum-shape geometric `
  --smooth-b-rl-weight 0.05 `
  --minimum-replay-env-steps 100 `
  --minimum-action-enabled-env-steps 100 `
  --replay-capacity-env-steps 100000 `
  --batch-size 1024 `
  --updates-per-env-step 1 `
  --learn-every-env-steps 1 `
  --sim-end-time 45000 `
  --tat-termination-policy episode2_tat180 `
  --wandb `
  --smoke-report .\contextual_reward_e_fixedscale1_seed0.json `
  --checkpoint-root .\checkpoints\contextual_reward_e_fixedscale1_seed0
```

### Reusing only the state normalizer

The collection run above atomically saves the local/global observation
normalizer once both statistics reach `--normalizer-freeze-steps` and freeze.
At the next terminal observation after the 10,000 warm-up transitions, the
runtime sends `SendIsEnd(1)`, stores the final warm-up reward transition, and
forces a latest checkpoint. The actor is not evaluated on that boundary tick;
after the simulator reset, policy control starts in the next clean episode.
For a later run with a fresh actor, critic, replay, and checkpoint, replace the
state save option and the automatic reward save option with:

```powershell
--load-state-normalizer .\normalizers\contextual_state_e_seed0.npz
--load-reward-normalizer
```

A successful load automatically changes the effective action warm-up from the
configured `10000` to `0`; do not also pass `--warmup-steps 0`. Loading is
also an exact bypass of the warm-up episode boundary, so it does not force an
extra reset before first-episode policy control. Loading is
fail-fast unless the observation version, local/global/relation feature names
and order, dimensions, topology hash, mapping hash, epsilon, clip, and frozen
state all match. This restores observation/state statistics only. It does not
restore actor/critic weights, replay, checkpoints, or Reward E's separate
reward normalizers; `--load-reward-normalizer` restores those separately.
The replay-size and action-enabled-transition gates still
apply before the first learner update. With the current `episode2_tat180`
policy, a loaded state normalizer also changes the effective TAT-termination
start to episode 1; a collection run without a loaded snapshot retains the
episode-1 exemption and starts TAT termination from episode 2.

`--save-reward-normalizer` with no path atomically saves both Reward E local
and global normalizers in one file under `.\normalizers`. The generated name
contains the reward version, exact reward-profile fingerprint, topology and
mapping hashes, and seed. The console prints the resolved path. An explicit
path is also accepted:

```powershell
--save-reward-normalizer .\normalizers\reward_e_seed0.npz
```

Use `--load-reward-normalizer` with no path to resolve the same automatic
filename, or pass the explicit saved path. Loading rejects snapshots unless
the snapshot format, reward/contract/TAT/normalization versions, every reward
term and coefficient, enabled local/global normalizers, epsilon/clipping,
topology, mapping, populated counts, and frozen state all match. State and
reward normalizer loads are standalone from actor, critic, replay, and
checkpoint state. For a true zero-warm-up fresh-agent run, load both state and
reward snapshots.

The Reward I predecessor uses the fixed-scale contextual reward directly, so
its reward normalizers are inactive even though the generic snapshot CLI is
available. Observation/state
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

## Reward R: backlog pressure rebalance

Reward R keeps Reward Q's actor, critic, SALE, replay, TAT/OP terms,
termination contract, parameterDw episode reset, and all learner
hyperparameters. It changes only `backlog_weight=0.0008 -> 0.008` and
`backlog_growth_weight=0.24 -> 0.16`. Its global raw backlog terms are therefore
`-0.008*B_t - 0.16*clip(max(0, B_t-B_{t-300})/30, 0, 1)` before the unchanged
global alpha `0.5`. Reward Q checkpoints, replay, and reward-normalizer state
are incompatible; start Reward R fresh.

## Reward S: Reward-E structure with signed TotalTat

Reward S (`contextual_controlled_reward_v21_e_structure_signed_total_tat`)
restores Reward E's training-wide normalize-before-update global/local reward
structure. Its raw global reward is
`9.2*(165-TotalTat)/165 - 0.01*(Waiting+Queued)`; this TAT signal is signed,
unclipped, and independent of completion-count or previous-TAT state. OP,
idle-reserve, backlog-growth, and marginal-TAT reward terms are disabled.

The local raw reward is
`-(0.3*OHT + 0.2*PredictedOHT + 0.3*Stop + 0.1*Idle + 0.1*Capacity)`.
There is no fixed local divisor. The final per-rail reward is
`0.5*Norm(global_raw) + 0.5*Norm(local_raw) + rail_reward - smooth_penalty`.
Rail reward uses Reward E's fixed-TAT, actual-elapsed-time attribution with
weight `1`, and b_rl smoothing uses weight `0.05`. Reward normalizers freeze
after 30,000 reward steps and persist across episode reset; observation
normalizers retain their separate lifecycle. Start with fresh checkpoint,
replay, and reward-normalizer state.

## Reward T: global-normalization ablation

Reward T (`contextual_controlled_reward_v22_global_raw_local_running_norm`)
removes the global running normalizer while retaining Reward S's local,
rail-reward, and smoothing paths. The final global component is exactly
`0.5*(2.3*(165-TotalTat)/165 - 0.0025*(Waiting+Queued))`. When
`TotalTat <= 0`, the TAT subterm is zero and only the backlog subterm remains.

The local raw reward remains
`-(0.3*OHT + 0.2*PredictedOHT + 0.3*Stop + 0.1*Idle + 0.1*Capacity)` and still
uses normalize-before-update running statistics with the 30,000-step freeze.
W&B records coefficient-applied rail-wise subterm magnitude and dispersion as
`local/{oht,pred,stop,idle,capacity}_{abs_mean,std}`. Reward T requires a fresh
checkpoint, replay, and reward-normalizer state.

Reward T's W&B budget applies `global_alpha` before comparing terms. It logs
`reward/contribution/{tat,backlog,local}_abs` and computes
`reward/budget/{tat,backlog,local,rail,smooth}_share` from the five magnitudes
that enter the final reward. Variance/covariance shares are intentionally not
included in this schema.

## Reward U: actual completion-CmdTat ablation

Reward U (`contextual_controlled_reward_v23_completion_tat_global_raw_local_running_norm`)
changes only Reward T's TAT signal. On each reward tick it consumes each newly
completed command once and computes
`completion_raw = mean((165-CmdTat)/165)`, or zero when no command completed.
The global component is
`0.5*(2.3*completion_raw - 0.0025*(Waiting+Queued))`. `TotalTat` remains an
environment/termination diagnostic and is not used to calculate this reward.

Completion is detected at the existing OHT `UNLOADING -> IDLE/new-cycle`
boundary. Its value comes from `PClient.OHT_DIC[*].CmdCompleteTat[*].CmdTat`;
`OHTTat` is never substituted. Episode-local command-ID deduplication is reset
between episodes, while the local running normalizer retains Reward T's
training-wide normalize-before-update and 30,000-step freeze contract. Global
normalization remains off, and local, rail, smooth, learner, replay, action,
dispatch, and termination behavior are unchanged.

For W&B inspection, each episode automatically selects the first valid command
found in `CmdCompleteTat` and keeps that command ID fixed until completion.
`trace/command/{cmd_tat,oht_tat,oht_id,oht_state,available,completed}` follows
the command even if it moves to another OHT. The completion tick retains the
last valid TAT values with `completed=1`; later ticks omit the trace until the
next episode selects a new command.

## Selecting historical Reward E through U

All named reward contracts and their complete parameter profiles are managed
in `contextual_reward_version_cfg.py`. Add or revise a reward version there;
`contextual_reward.py` contains the generic immutable config schema and reward
calculation/runtime state only. Existing imports from `contextual_reward` are
re-exported for compatibility.

The current CLI experiment defaults to `E` and accepts these unique locked
reward profiles:

`E`, `F_RAMP`, `F_NO_RAMP`, `G`, `H`, `I`, `J`, `K`, `L`, `M`, `N`,
`O`, `P`, `Q`, `R`, `S_REBALANCE`, `S_EHYBRID`, `T`, and `U`.

```powershell
python .\PythonCode\main_contextual.py --reward-version E
python .\PythonCode\main_contextual.py --reward-version F_RAMP
python .\PythonCode\main_contextual.py --reward-version S_REBALANCE
python .\PythonCode\main_contextual.py --reward-version T
python .\PythonCode\main_contextual.py --reward-version U
```

This selects a complete locked profile, not only a TAT formula. The profile
also selects global/local coefficients, local raw terms, backlog, rail and
smooth rewards, global/local reward-normalizer enablement, normalization
ordering/freeze, clipping, TAT confidence, and TAT termination/terminal
penalty defaults for historical reproduction. Named runtime termination
policies `episode2_tat175` and `episode2_tat180` override only the environment
`done` condition; they do not change Reward E's formula. Select the current
TAT-180 experiment with
`--tat-termination-policy episode2_tat180`. Reward
E-G restore the historical marginal-TAT EMA path; I through
`S_REBALANCE` use fixed reward scale; `S_EHYBRID` restores both running reward
normalizers; T keeps only the local normalizer; U uses newly completed
commands' actual `CmdTat`. Replay, checkpoints, standalone reward-normalizer
files, W&B metadata, and checkpoint directories carry the unique profile key
and reject cross-profile state.

Historical aliases are accepted: `F` resolves to `F_RAMP`, while `S` resolves
to the later `S_EHYBRID`. Use `F_NO_RAMP` and `S_REBALANCE` explicitly for the
other same-letter contracts. Hyphenated spellings such as `F-NO-RAMP` are also
canonicalized.

The older individual reward CLI knobs remain parseable for compatibility, but
a value that differs from the selected named profile is rejected. A changed
coefficient, termination, or normalization rule must be introduced as a new
unique reward profile rather than reusing an existing identity.

## Previous applied action input

The decision-time state is action-augmented without mixing controller memory
into the physical observation normalizer. For transition `t`, the actor uses
`pi(observation_t, applied_action_(t-1))` and the critic uses
`Q(observation_t, applied_action_(t-1), candidate_action_t)`. The critic keeps
the traditional current candidate-action input; the previous action is state
memory needed to evaluate the existing action-delta smoothing reward.

For stacked input, every selected observation frame carries the simulator-
applied action immediately preceding that frame. Thus a stack with offsets
`[0, I, 2I]` pairs observations `[s_t, s_(t-I), s_(t-2I)]` with previous
actions `[a_(t-1), a_(t-I-1), a_(t-2I-1)]`. The next state at `t+1` is paired
with `a_t`. Episode reset starts with a zero previous action. This observation,
network, replay, stack, and checkpoint contract requires a fresh run.

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
- `--batch-size`는 resume 시 안전하게 덮어쓸 수 있음
- `--resume-inference-until-replay-full`을 주면 새 replay가
  `--replay-capacity-env-steps`에 도달할 때까지 actor+exploration 추론과
  transition 수집만 하고, 다음 tick부터 learner update를 재개함
- `--resume-deterministic-first-episode`을 주면 첫 resume 에피소드는
  action noise와 learner update 없이 deterministic actor로 끝까지 수집함.
  해당 replay를 유지하고 다음 에피소드부터 저장된 exploration schedule과
  learner update를 재개하며, 첫 에피소드가 100 env steps보다 짧으면 다음
  에피소드에서 일반 minimum replay gate를 채운 뒤 학습함
- full-refill 도중 프로세스를 재시작하면 replay는 저장되지 않으므로 다시
  0부터 채워야 함
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

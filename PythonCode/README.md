# Contextual TD7 v6 for OHT routing

## Checkpoint actor evaluation

Run a deterministic actor evaluation without exploration, replay collection,
learner updates, or checkpoint writes:

```powershell
python .\PythonCode\main.py `
  --mode actor_inference `
  --action-enabled `
  --resume-checkpoint ".\checkpoints\ctx_td7_v6_actor_tat\periodic\step_40000.pt"
```

병목 분석용 원본 환경 snapshot도 함께 저장하려면 `--save_data`를 추가합니다.
actor 평가 시작 후 첫 2,000 tick의 모든 rail/OHT/active job과 action/cost를
`results/environment_capture/` 아래의 압축 JSONL로 저장합니다.

```powershell
python .\PythonCode\main.py `
  --mode actor_inference `
  --action-enabled `
  --save_data `
  --resume-checkpoint ".\checkpoints\ctx_td7_v6_actor_tat\periodic\step_40000.pt"
```

The checkpoint is loaded after simulator topology initialization. Confirm the
`[checkpoint-loaded]` block reports the expected environment step, action
scale, and restored normalizers before using evaluation results.

반도체 FAB OHT의 rail cost를 학습해 혼잡 구간을 우회시키는 contextual TD7
구현입니다. v6.2 runtime은 Reward P만 실행하며, 이전 Reward
E~U 구현과 실험 기록은 `contextual-region-brl-v1` 브랜치와 루트의
`EXPERIMENTS.md`에 보존합니다.

## 실행 진입점

저장소 루트에서 실행합니다.

```powershell
conda activate aicc

# baseline
python .\PythonCode\main.py

# Reward P 학습
python .\PythonCode\main.py `
  --mode training `
  --action-enabled `
  --no-lap `
  --save-state-normalizer .\PythonCode\normalizers\contextual_v6_actor_tat_seed0.npz `
  --checkpoint-root .\PythonCode\checkpoints\ctx_td7_reward_p
```

Stage 1의 2,000초 단기 학습은 CLI alias로 실행할 수 있습니다.

```powershell
python .\PythonCode\main.py `
  --mode training `
  --action-enabled `
  --stage 1
```

## V6 neighbor aggregation

V6 keeps the learned rail embedding and the existing shared center, neighbor,
and global feature encoders. By default, each direction's 15 encoded neighbor
tokens are flattened in deterministic topology-rank order and projected to one
64-dimensional context. No attention module is created or trained.

```powershell
# Default: rail embedding + directional flat projection, no attention
python .\PythonCode\main.py `
  --mode training `
  --action-enabled `
  --stage 1 `
  --periodic-checkpoint-interval 2000 `
  --checkpoint-root .\checkpoints\ctx_td7_v6_flat

# Opt-in: the previous directional cross-attention aggregation
python .\PythonCode\main.py `
  --mode training `
  --action-enabled `
  --stage 1 `
  --use-attention `
  --periodic-checkpoint-interval 2000 `
  --checkpoint-root .\checkpoints\ctx_td7_v6_attention
```

The encoder mode is part of the checkpoint network contract. Add
`--use-attention` whenever loading a V6 attention checkpoint; omit it for a V6
flat checkpoint. V5 및 V6.0/V6.1 model checkpoint와 state-normalizer는
6-feature actor-global 계약과 호환되지 않습니다. V6.2 Stage 1 policy와
normalizer를 새로 생성해야 합니다.

`--stage 1`은 `sim_end_time=2000`인 단기 학습 계약입니다. `--stage 2`는
`sim_end_time=45000`으로 실행되며, 매 episode의 첫 2,000 tick에는 별도로
불러온 frozen Stage 1 policy를 사용하고 2,001번째 tick부터 Stage 2 policy를
사용합니다. `--stage`와 `--sim-end-time`은 함께 사용할 수 없습니다. 두 옵션을
모두 생략하면 runtime 기본값 45,000이 적용되며 checkpoint에 저장된 종료 시간은
복원하지 않습니다.

## Stage 2 frozen-prefix 학습

V6.2 actor-TAT flat Stage 1 checkpoint에서 새 Stage 2 learner를
시작하는 예시는 다음과
같습니다.

```powershell
python .\PythonCode\main.py `
  --mode training `
  --action-enabled `
  --stage 2 `
  --load_stage1_policy ".\checkpoints\ctx_td7_v6_actor_tat_stage1\periodic\step_34000.pt" `
  --exploration-noise-std 0.05 `
  --exploration-noise-final-std 0.05 `
  --periodic-checkpoint-interval 2000 `
  --checkpoint-root ".\checkpoints\ctx_td7_v6_4_actor_tat_stage2_from_s1_34000_noise005"
```

`--load_stage1_policy`와 `--load-stage1-policy`는 같은 옵션입니다. 이 로드는
Stage 1 checkpoint의 online encoder, actor, frozen SALE state encoder,
frozen observation normalizer, 저장 당시 applied-action scale을 prefix용으로
복원합니다. Fresh Stage 2에서는 encoder, actor, policy target, SALE 경로를
이 값으로 한 번만 초기화하며 episode reset이나 Stage 2 checkpoint resume
시에는 다시 덮어쓰지 않습니다. Stage 1 critic, optimizer, replay, update
counter, reward state, Python/NumPy/Torch RNG는 Stage 2 learner에 복원하지
않습니다.

Fresh Stage 2 CLI 실행에서 noise 두 옵션을 모두 생략해도 기본값은
`0.05 -> 0.05`로 해석됩니다. 위처럼 두 값을 명시하면 실행 명령 자체에도
실험 조건이 남습니다. 한쪽이라도 명시한 경우에는 일반 explicit override
규칙을 따르며, Stage 2 checkpoint resume은 저장된 noise 설정을 복원합니다.

각 episode에서 1~2,000번째 tick은 Stage 1 deterministic policy만 실행합니다.
이 구간에는 exploration, transition staging/replay insertion, learner sampling,
learner update가 모두 없습니다. 2,001번째 tick에서 reset 없이 Stage 2 policy로
전환하고, 그 action의 첫 transition은 다음 observation이 들어오는 2,002번째
tick에 replay에 저장됩니다. Stage 1 policy module은 `eval()` 상태이며 모든
parameter가 `requires_grad=False`인 Stage 2 learner와 완전히 분리된 복사본입니다.

Stage 2의 curriculum, exploration annealing, learner cadence, latest/periodic
checkpoint cadence는 누적 Stage 2 tick을 0부터 세어 진행합니다. 따라서 위
명령의 첫 periodic artifact는 Stage 2가 실제로 2,000 tick 진행된 뒤
`periodic/step_02000.pt`로 저장됩니다. 전체 simulator tick은 checkpoint
metadata의 `runtime_env_step`, Stage 2 tick은 `stage2_env_steps`에 각각
기록됩니다.

`--reward-version`은 체크포인트/실행 계약을 명시하기 위한 호환 옵션이며
`P`만 허용합니다. simulator GUI에서 Python 연동을 활성화한 뒤 Run을
시작해야 TCP 세션이 연결됩니다.

공식 진입점은 `main.py` 하나입니다. 삭제된 region-token 경로는 더 이상
별도 wrapper나 진입점을 제공하지 않습니다.

Runtime 기본값의 단일 원본은
`oht_routing/runtime/config.py`의 `ContextualRuntimeConfig`입니다. CLI는
사용자가 명시한 옵션만 override하며 별도의 기본값을 두지 않습니다.
현재 region-B curriculum 기본은 geometric `0.05 -> 1.0`, global step
20,000입니다. 따라서 위 명령에 curriculum 옵션을 반복할 필요가 없으며,
필요할 때만 `--curriculum-scale-start` 같은 CLI override를 사용합니다.
`--no-lap`과 `--no-sale`도 최종 runtime config에 직접 반영됩니다.

## Reward P 계약

controlled rail `i`의 최종 reward는 다음과 같습니다.

```text
r_i = 0.5 * global_raw
    + 0.5 * (local_raw_i / 2)
    + clip(30 * rail_raw_i, -1, 1)
    - 0.25 * abs(delta_b_rl_i)
```

global 항은 아래의 합입니다.

```text
total_tat          = pclient.TotalTat
tat_raw            = -4.3 * max(total_tat - 160, 0) / 165
op_raw             = 0
backlog_raw        = -0.0007 * (waiting + queued)
backlog_growth_raw = -0.17 * clip(max(0, B_t - B_(t-300)) / 30, 0, 1)
idle_reserve_raw   = -0.09 * clip((200 - idle_oht_count) / 50, 0, 1)

predicted_oht_raw_i = -0.05 * predicted_oht_count_i
stop_time_raw_i     = -0.12 * sum(StopTime of OHTs on rail i)
```

Reward P reads cumulative `pclient.TotalTat` directly. `TotalTat == 0` is an
unavailable/reset sentinel and its reward contribution is `0`; a negative or
non-finite value is invalid. `0 < TotalTat <= 160` also contributes `0`, and
only values above 160 seconds receive the unbounded one-sided penalty above.
The recent-300-second completion TAT remains diagnostic-only and does not
affect Reward P. `TotalTat` is also the early-termination signal, the actor's
first normalized global feature, and the critic's separately normalized direct
scalar.
Reward 계산은 이 normalized critic 입력이 아니라 simulator raw 값을 직접
사용합니다.
The rail-local StopTime sum is deliberately unclipped, so worsening congestion
continues to increase the penalty.

rail 항은 완료된 cycle의
`2 - route_time / route_free_flow_time`이며 rail별로 `[-1, 1]`에
clip됩니다. reward running normalizer는 사용하지 않습니다. 관측값의
state normalizer는 별도 계약으로 유지합니다.

TAT 종료 조건은 10,000 environment-step grace 이후
`TotalTat >= 200`이 300회 연속 발생하는 경우입니다. 종료 transition에는
`-20` terminal penalty를 한 번 broadcast합니다.

계약의 단일 소스는
`oht_routing/mdp/reward/config.py`의 `REWARD_P_CONTRACT`와
`REWARD_P_PROFILE`입니다. Reward O 정의는 과거 계약 확인용으로만
보존됩니다.

## V6.2 observation 계약

V6.2는 compact schema를 유지하면서 normalized cumulative
`total_tat_s`를 actor-global 입력의 첫 feature로 복원합니다. Critic에는
기존의 direct TotalTat scalar도 계속 전달합니다.

```text
center_local            [N, 14]
incoming_local          [N, 15, 14]
outgoing_local          [N, 15, 14]
incoming_relation       [N, 15, 2]
outgoing_relation       [N, 15, 2]
actor_global_state      [6]
previous_applied_action [N, 1]
critic_total_tat        [1]  # direct critic copy
```

local feature는 `free_flow_time_s`, `port_count`, `incoming_degree`,
`outgoing_degree`, `predicted_oht_count`, `reservation_port_count`, 6개 OHT
state count, `stop_time_sum`, `stopped_oht_count` 순서입니다.
`oht_density`와 `next_10_route_oht_count`는 RL 입력에서 제거됐습니다.
actor global feature는 `total_tat_s`, `operation_rate`, `queued_ratio`,
`waiting_ratio`, `transferring_ratio`, `mean_reassign` 순서입니다.
rail identity embedding(8),
15-in/15-out topology, relation feature 2개, previous applied action은
유지합니다.

Actor와 actor-side SALE/target actor는 normalized actor-global TotalTat를
사용합니다. Replay는 현재·다음 state의 actor-global TotalTat와 direct-critic
TotalTat를 함께 보존하며, target actor와 target critic에는 각각 다음 state
값을 전달합니다. OP는 actor state로 남지만 Reward P의 `op_weight=0.0`,
`use_op=False` 계약은 유지됩니다.

## 코드 구조

```text
main.py
└─ 얇은 실행 진입점
oht_routing/
├─ algorithms/rl/contextual_td7/   TD7, SALE, LAP, replay, checkpoint
├─ mdp/                            action, observation, transition, termination
│  └─ reward/                      Reward P 설정, 조합, rail-cycle 추적
├─ runtime/                        client, CLI/config, protocol, TCP server
└─ utils/                          W&B, reward 진단, topology/분석 도구
oht_dispatching/                   job-to-OHT 후보 생성과 dispatch 선택
simulator/                         TCP protocol과 simulator entity 모델
tests/                             contextual 회귀 테스트
```

## 유틸리티

`PythonCode/oht_routing/utils` 아래 도구는 저장소 루트에서 직접
실행합니다.

```powershell
python .\PythonCode\oht_routing\utils\check_rail_topology.py --help
python .\PythonCode\oht_routing\utils\download_wandb_run.py --help
python .\PythonCode\oht_routing\utils\export_wandb_run.py --help
python .\PythonCode\simulator\evaluate.py --help
```

## State normalizer

관측 통계만 수집하는 run에서는 다음처럼 frozen snapshot을 저장합니다.

```powershell
python .\PythonCode\main.py `
  --mode training `
  --action-enabled `
  --save-state-normalizer .\PythonCode\normalizers\contextual_state_n_seed0.npz
```

새 learner로 시작하면서 같은 V6.2 artifact 계약의 관측 통계만 재사용하려면
`--load-state-normalizer <path>`를 사용합니다. 유효한 snapshot을
불러오면 effective warm-up은 0이 됩니다. actor, critic, replay, reward
상태는 복원하지 않습니다. actor-global feature와 direct-critic TotalTat는
분리된 normalizer state를 사용합니다. V5 및 V6.0/V6.1 normalizer는
actor-global 차원이 다르므로 호환되지 않습니다.

## Replay 메모리

기본 replay capacity는 100,000 environment step입니다. V5는 매 state의
정적 rail feature를 한 번만 저장하고, protocol 범위가 보장되는 compact local
count를 `uint8`/`uint16`으로 lossless packing합니다. policy/applied/previous
action은 Q15 `int16`으로
저장하며 최대 절대 복원 오차는 약 `1.53e-5`입니다. reward는 `float32`, LAP
priority는 `float16`을 사용합니다. current/next actor-global 및 direct-critic
TotalTat도 replay transition에 포함됩니다.

실제 설정에 따른 예상치는 시작 요약의 `estimated full replay RAM`과 W&B
`replay/storage_bytes`에서 확인할 수 있습니다.

LAP은 기본 활성입니다. 균등 rail sampling으로 학습하려면 `--no-lap`을
명시합니다. resume에서도 현재 CLI의 LAP 설정을 유지하며, 체크포인트가 다른
LAP 사용 계약으로 저장됐다면 호환성 검사에서 명시적으로 거부합니다.

## Checkpoint와 resume

통합 runtime/checkpoint 버전은 `v6.4.0`입니다. 같은 major라도 저장된
network/observation 계약이 정확히 일치해야 호환될 수 있습니다.
checkpoint 로드는
통합 버전, topology/mapping hash, network config, action mode,
SALE/LAP 사용 여부, Reward P만 호환성으로 검사합니다. replay payload는
저장하지 않으므로 training resume 후에는 replay를 다시 채워야 합니다.
V6는 neighbor aggregation의 기본값을 attention에서 flat projection으로
바꾼 계열입니다. V6.2에서 actor-global dimension이 5에서 6으로 바뀌었으므로
V5 및 V6.0/V6.1 model checkpoint, Stage 1 policy, state normalizer는 모두
거부됩니다.

```powershell
python .\PythonCode\main.py `
  --mode training `
  --action-enabled `
  --resume-checkpoint .\PythonCode\checkpoints\ctx_td7_v6_flat\step_220000.pt `
  --resume-deterministic-first-episode
```

immutable periodic checkpoint의 기본 저장 주기는 5,000 global environment
step입니다. Resume 실행에서 주기를 바꾸려면
`--periodic-checkpoint-interval`을 명시합니다. 예를 들어 V6 flat
40,000-step checkpoint에서 재학습하며 2,000 step마다 별도 저장하려면
다음처럼 새 checkpoint root를 사용합니다.

단, `--stage 2`에서는 frozen Stage 1 prefix를 제외한 누적
`stage2_env_steps`가 latest/periodic 저장 주기와 periodic 파일명의 기준입니다.

```powershell
python .\PythonCode\main.py `
  --mode training `
  --action-enabled `
  --stage 1 `
  --resume-warmstart-steps 100 `
  --periodic-checkpoint-interval 2000 `
  --checkpoint-root .\checkpoints\ctx_td7_v6_flat_resume40k_warm100_p2000 `
  --resume-checkpoint .\checkpoints\ctx_td7_v6_flat\periodic\step_40000.pt
```

Attention으로 저장한 V6 checkpoint를 resume할 때는 위 명령에
`--use-attention`을 반드시 추가합니다. 기존 V5 `step_40000.pt`는 V6
flat/attention 어느 쪽으로도 resume할 수 없습니다.

이 경우 resume 직후 100 tick은 복원한 deterministic actor로
시뮬레이터만 진행합니다. Exploration, replay insertion, learner
update를 모두 끄고 100번째 tick에 episode 종료를 요청합니다.
다음 reset episode부터 정상 resume 학습을 시작합니다. Global step은
이 throwaway tick도 환경 step으로 계속 카운트하므로 periodic artifact는
`step_42000.pt`, `step_44000.pt`, ...로
저장됩니다. `latest/checkpoint.pt`는 기존대로 1,000 step마다 같은 파일을
교체하며, 원본 실행 폴더를 `--checkpoint-root`로 재사용하면 이미 존재하는
후속 periodic checkpoint를 덮어쓸 수 있으므로 별도 root를 권장합니다.

Resume 시작 구간 제어의 차이는 다음과 같습니다.

- `--resume-warmstart-steps N`: resume 직후 첫 짧은 episode를 N
  tick에 종료하고 그 구간을 replay에서 완전히 제외합니다.
- `--resume-deterministic-first-episode`: 첫 episode 전체의 exploration과
  learner update를 끄지만 transition은 replay에 보존합니다.

`--resume-warmstart-steps`와 `--resume-deterministic-first-episode`는 의미가
겹치므로 동시에 사용할 수 없습니다.
Stage 2의 매-episode frozen Stage 1 prefix와도 역할이 겹치므로
`--resume-warmstart-steps`는 `--stage 2`와 함께 사용할 수 없습니다.

`--resume-deterministic-first-episode`는 첫 resume episode에서 actor
exploration과 learner update를 끄고 transition만 수집한 뒤, 다음
episode부터 저장된 exploration schedule과 update를 재개합니다.
`--resume-inference-until-replay-full`은 replay capacity가 찰 때까지
학습을 더 오래 보류하는 별도 옵션입니다.

PyTorch가 memory-efficient attention backward의 비결정적 CUDA 경로를
경고할 수 있습니다. 현재 runtime은 deterministic 알고리즘을
`warn_only=True`로 요청하므로 이 경고는 실행 중단이 아니라 재현성
주의사항입니다.

## W&B와 실험 기록

모든 새 실험 전에 `oht_routing/utils/wandb_logging.py`의 `EXP_META`를
먼저
갱신합니다. run 이름은 `_make_run_name`이 만들며 직접 짓지 않습니다.
`wandb.init`은 runtime config와 `EXP_META`를 함께 기록하고
`EXP_META["description"]`을 notes로 전달합니다.

현재 v6 변경과 실험 결과는 루트의 `EXPERIMENTS_v6.md`에
기록합니다. 모든 변경은 `oht_routing/version.py`의 단일
`vMAJOR.MINOR.PATCH` 버전으로 분리하고 호환성 영향을 함께 기록합니다.
MAJOR가 바뀌면 루트에 `EXPERIMENTS_v{new_major}.md`를 새로 만들고,
이전 major의 기록 파일은 수정하지 않고 이력으로 보존합니다.
W&B metric을 바꾸면
`oht_routing/utils/export_wandb_run.py`도 같이 수정합니다.

## 테스트

```powershell
$env:PYTHONPATH="$PWD;$PWD\PythonCode;$PWD\PythonCode\tests"
& 'C:\Users\bjy66\miniconda3\envs\aicc\python.exe' `
  -m unittest discover -s .\PythonCode\tests -p 'test_contextual*.py'
```

## V6.3 multi-simulator Stage 2

The distributed path keeps the packed replay in central RAM and gives CUDA,
the online learner, the frozen Stage 1 policy, checkpoints, and W&B to one
central owner. Each simulator connection retains independent observation,
reward, transition, episode, previous-action, dispatcher, and exploration-RNG
state.

```powershell
python .\PythonCode\main.py `
  --mode training `
  --action-enabled `
  --stage 2 `
  --reward-version P `
  --no-lap `
  --seed 0 `
  --load-stage1-policy ".\PythonCode\checkpoints\ctx_td7_v6_2_actor_tat_stage1_seed0\periodic\step_60000.pt" `
  --exploration-noise-std 0.05 `
  --exploration-noise-final-std 0.05 `
  --num-sim 4 `
  --port 9100 9101 9102 9103 `
  --batch-size 512 `
  --periodic-checkpoint-interval 2000 `
  --checkpoint-root ".\PythonCode\checkpoints\ctx_td7_v6_4_dist4_stage2_from_s1_60000_noise005_seed0"
```

`--num-sim` is the number of simulator connections expected by the Python
runtime; it does not start simulator executables. Configure one simulator
instance for each port. The port count must equal `--num-sim`, ports must be
unique, and every listener is bound before collection starts. Omitting both
options preserves the existing single-port `wpconfig.json` behavior.

Every worker runs its own 2,000-tick frozen Stage 1 prefix. Stage 2 curriculum
and checkpoint clocks count aggregate learner-active worker ticks. V6.3 does
not support distributed full-state resume because a central artifact does not
contain every collector's live episode/reward/RNG state; start a fresh
distributed learner from the frozen V6.2 Stage 1 policy.

Reward P 식의 고정값 검증은
`tests/test_contextual_reward_n_contract.py`에 있습니다.

# Contextual TD7 v5 for OHT routing

## Checkpoint actor evaluation

Run a deterministic actor evaluation without exploration, replay collection,
learner updates, or checkpoint writes:

```powershell
python .\PythonCode\main.py `
  --mode actor_inference `
  --action-enabled `
  --resume-checkpoint ".\checkpoints\ctx_td7_reward_p\periodic\step_400000.pt"
```

병목 분석용 원본 환경 snapshot도 함께 저장하려면 `--save_data`를 추가합니다.
actor 평가 시작 후 첫 2,000 tick의 모든 rail/OHT/active job과 action/cost를
`results/environment_capture/` 아래의 압축 JSONL로 저장합니다.

```powershell
python .\PythonCode\main.py `
  --mode actor_inference `
  --action-enabled `
  --save_data `
  --resume-checkpoint ".\checkpoints\ctx_td7_reward_p\periodic\step_400000.pt"
```

The checkpoint is loaded after simulator topology initialization. Confirm the
`[checkpoint-loaded]` block reports the expected environment step, action
scale, and restored normalizers before using evaluation results.

반도체 FAB OHT의 rail cost를 학습해 혼잡 구간을 우회시키는 contextual TD7
구현입니다. v5 runtime은 Reward P만 실행하며, 이전 Reward
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
  --save-state-normalizer .\PythonCode\normalizers\contextual_v5_seed0.npz `
  --checkpoint-root .\PythonCode\checkpoints\ctx_td7_reward_p
```

Stage 1의 2,000초 단기 학습은 CLI alias로 실행할 수 있습니다.

```powershell
python .\PythonCode\main.py `
  --mode training `
  --action-enabled `
  --stage 1
```

`--stage 1`은 `sim_end_time=2000`만 설정하며 `--sim-end-time`과 함께 사용할
수 없습니다. Resume에서도 명시한 `--stage 1`은 2,000을 유지합니다. 두 옵션을
모두 생략하면 현재 runtime 기본값 45,000이 적용되며 checkpoint에 저장된 종료
시간은 복원하지 않습니다.

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
affect Reward P. `TotalTat` is also the early-termination signal and is exposed
only to the critic through a separately normalized scalar.
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

## V5 observation 계약

V5는 actor-visible 입력을 compact schema로 줄이고, cumulative
`total_tat_s`를 critic에만 추가하는 asymmetric actor-critic 구조입니다.

```text
center_local            [N, 14]
incoming_local          [N, 15, 14]
outgoing_local          [N, 15, 14]
incoming_relation       [N, 15, 2]
outgoing_relation       [N, 15, 2]
actor_global_state      [5]
previous_applied_action [N, 1]
critic_total_tat        [1]  # critic only
```

local feature는 `free_flow_time_s`, `port_count`, `incoming_degree`,
`outgoing_degree`, `predicted_oht_count`, `reservation_port_count`, 6개 OHT
state count, `stop_time_sum`, `stopped_oht_count` 순서입니다.
`oht_density`와 `next_10_route_oht_count`는 RL 입력에서 제거됐습니다.
actor global feature는 `operation_rate`, `queued_ratio`, `waiting_ratio`,
`transferring_ratio`, `mean_reassign` 순서입니다. rail identity embedding(8),
15-in/15-out topology, relation feature 2개, previous applied action은
유지합니다.

Actor와 actor-side SALE/target actor에는 어떤 형태의 TotalTat도 전달하지
않습니다. Replay는 현재·다음 state의 critic-only TotalTat를 함께 보존하며,
target critic에는 반드시 다음 state 값을 전달합니다. OP는 actor state로는
남지만 Reward P의 `op_weight=0.0`, `use_op=False` 계약은 유지됩니다.

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

새 learner로 시작하면서 같은 V5 artifact 계약의 관측 통계만 재사용하려면
`--load-state-normalizer <path>`를 사용합니다. 유효한 snapshot을
불러오면 effective warm-up은 0이 됩니다. actor, critic, replay, reward
상태는 복원하지 않습니다. actor-visible feature와 critic-only TotalTat는
분리된 normalizer state를 사용합니다. V4 이하 이전 major의 normalizer는
호환되지 않습니다.

## Replay 메모리

기본 replay capacity는 100,000 environment step입니다. V5는 매 state의
정적 rail feature를 한 번만 저장하고, protocol 범위가 보장되는 compact local
count를 `uint8`/`uint16`으로 lossless packing합니다. policy/applied/previous
action은 Q15 `int16`으로
저장하며 최대 절대 복원 오차는 약 `1.53e-5`입니다. reward는 `float32`, LAP
priority는 `float16`을 사용합니다. current/next critic-only TotalTat도 replay
transition에 포함됩니다.

실제 설정에 따른 예상치는 시작 요약의 `estimated full replay RAM`과 W&B
`replay/storage_bytes`에서 확인할 수 있습니다.

LAP은 기본 활성입니다. 균등 rail sampling으로 학습하려면 `--no-lap`을
명시합니다. resume에서도 현재 CLI의 LAP 설정을 유지하며, 체크포인트가 다른
LAP 사용 계약으로 저장됐다면 호환성 검사에서 명시적으로 거부합니다.

## Checkpoint와 resume

통합 runtime/checkpoint 버전은 `v5.1.0`입니다. 같은 major의 이전
버전 artifact만 현재 runtime보다 새 버전이 아닌 경우 호환될 수 있습니다.
checkpoint 로드는
통합 버전, topology/mapping hash, network config, action mode,
SALE/LAP 사용 여부, Reward P만 호환성으로 검사합니다. replay payload는
저장하지 않으므로 training resume 후에는 replay를 다시 채워야 합니다.
V5는 compact asymmetric observation과 Reward P의 one-sided cumulative
TotalTat penalty를 함께 도입한 breaking version입니다. 기존 v2/v3/v4
checkpoint, replay, normalizer는 거부됩니다.

```powershell
python .\PythonCode\main.py `
  --mode training `
  --action-enabled `
  --resume-checkpoint .\PythonCode\checkpoints\ctx_td7_reward_p\step_220000.pt `
  --resume-deterministic-first-episode
```

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

현재 v5 변경과 실험 결과는 루트의 `EXPERIMENTS_v5.md`에
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

Reward P 식의 고정값 검증은
`tests/test_contextual_reward_n_contract.py`에 있습니다.

# Contextual TD7 v9.0.0 for OHT routing

반도체 FAB OHT의 rail cost를 학습해 혼잡 구간을 우회시키는 contextual TD7
구현입니다. 실행 가능한 reward 계수는 **Q**(기본)와 **N**(2026-08-20 run
`1y9sx4a5` 복원) 두 가지이며 `--reward-version`으로 선택합니다. O/P를 포함한
이전 구현과 실험 기록은 루트의 `EXPERIMENTS.md`, `EXPERIMENTS_v2.md` ~
`EXPERIMENTS_v8.md`에 보존합니다. v9 기록은 `EXPERIMENTS_v9.md`입니다.

공식 진입점은 `main.py` 하나입니다. Runtime 기본값의 단일 원본은
`oht_routing/runtime/config.py`의 `ContextualRuntimeConfig`이며, CLI는
사용자가 명시한 옵션만 override합니다. simulator GUI에서 Python 연동을
활성화한 뒤 Run을 시작해야 TCP 세션이 연결됩니다.

## 실행

저장소 루트에서 실행합니다. 모든 예시는 PowerShell 기준이며, 줄바꿈은
백틱(`` ` ``)입니다.

```powershell
conda activate aicc

# baseline
python .\PythonCode\main.py

# Stage 1 학습 (episode 2,000 tick)
python .\PythonCode\main.py --mode training --action-enabled --stage 1
```

### Stage 2 학습 (frozen Stage 1 prefix)

`v9.0.0_stage1_policy.pt`를 Stage 1 prefix로 불러와 Stage 2부터 시작합니다.

```powershell
python .\PythonCode\main.py `
  --mode training `
  --action-enabled `
  --stage 2 `
  --load-stage1-policy ".\PythonCode\v9.0.0_stage1_policy.pt" `
  --replay-eviction-mode random
```

Reward N 계수로 돌리려면 `--reward-version N`만 추가합니다. exploration은
0.05 고정을 권장합니다(아래 참고).

```powershell
python .\PythonCode\main.py `
  --mode training `
  --action-enabled `
  --stage 2 `
  --reward-version N `
  --load-stage1-policy ".\PythonCode\v9.0.0_stage1_policy.pt" `
  --replay-eviction-mode random `
  --exploration-noise-std 0.05 `
  --exploration-noise-final-std 0.05
```

다른 장비에서 실행할 때 필요한 것은 저장소와
**`PythonCode/v9.0.0_stage1_policy.pt` (23 MB) 하나**입니다. 이 파일은 Reward
Q로 학습됐지만 frozen prefix로만 쓰이므로 `--reward-version N`에서도 그대로
로드됩니다. 다만 topology는 분리되지 않으므로 simulator 레이아웃이 같아야
합니다(`topology_hash`/`mapping_hash` 검사).

매 episode의 1~2,000 tick은 frozen Stage 1 policy가 운전하며 exploration,
replay insertion, learner update가 모두 없습니다. 2,001 tick에서 reset 없이
Stage 2 policy로 전환합니다. **Stage 2 learner는 기본적으로 처음부터
학습합니다** — Stage 1 가중치를 물려받으려면 `--stage1-policy-warm-start`를
명시해야 하고, 그때만 Stage 1 artifact의 reward version이 일치해야 합니다.
Stage 2의 curriculum과 checkpoint 주기는 prefix를 제외한 `stage2_env_steps`
기준입니다.

#### exploration noise

`0.05` 고정을 권장합니다. 동일 조건에서 노이즈만 바꾼 두 run
(`jki8xzcv` 0.05 vs `s31qyeko` 0.10, 그 외 설정과 Stage 1 artifact 동일):

| | 0.05 | 0.10 |
| --- | ---: | ---: |
| TAT 초반 -> 후반 | 175.2 -> **172.3** | 174.7 -> **188.0** |
| backlog | 170.4 | 237.1 |
| action clip 비율 | 1.3% | **3.9%** |
| policy 자체 std | 0.453 | **0.353** |

0.10은 액션을 경계 밖으로 밀어 clipping을 3배로 늘리고, 정책 자체의 다양성을
오히려 줄였습니다. 참고로 노이즈가 설명하는 행동 분산은 0.05에서 1.1%뿐이며,
나머지는 정책이 상태에 반응해 만들어냅니다.

### Actor inference 평가

exploration, replay, learner update, checkpoint 저장 없이 deterministic actor만
실행합니다.

```powershell
python .\PythonCode\main.py `
  --mode actor_inference `
  --action-enabled `
  --stage 1 `
  --resume-checkpoint ".\PythonCode\v9.0.0_stage1_policy.pt" `
  --checkpoint-root ".\checkpoints\_inference_v9"
```

`--save_data`를 추가하면 첫 2,000 tick의 rail/OHT/job/action/cost를
`results/environment_capture/`에 압축 JSONL로 저장합니다 (약 1.4 GB).
reward term share 재검증에 사용합니다 —
`results/reward_diagnostics/reward_q_calibration/` 참고.

`[checkpoint-loaded]` 블록의 environment step, action scale, 복원된
normalizer를 확인한 뒤 결과를 사용하세요.

## Reward 계약

기본은 Q입니다. `--reward-version N`은 아래 계수만 바꾸며 구조는 동일합니다.

| 파라미터 | Q | N |
| --- | ---: | ---: |
| `tat_weight` | 4.0 | 11.0 |
| `op_weight` / `use_op` | 0.0 / off | 4.0 / on |
| `backlog_weight` | 0.0007 | 0.0004 |
| `backlog_growth_weight` | 0.17 | 0.16 |
| `idle_reserve_weight` | 0.09 | 0.2 |
| `local_oht_weight` | 0.0 | 0.3 |
| `local_predicted_oht_weight` | 0.01 | 0.075 |
| `local_stop_weight` | 0.30 | 0.3 |
| `local_density_weight` | 5.5 | 0.0 |
| `local_capacity_weight` | 0.0 | 0.1 |
| `rail_tat_weight` / clip / neutral | 660 / 22 / 1.70 | 30 / 1.0 / 2.0 |

아래 식은 Q 기준입니다.

controlled rail `i`의 최종 reward입니다.

```text
r_i = 0.5 * global_raw
    + 0.5 * (local_raw_i / 2)
    + clip(660 * rail_raw_i, -22, 22)
    - 0.25 * abs(delta_b_rl_i)
```

```text
# global (모든 rail 공통)
tat_raw            = -4.0 * max(TotalTat - 160, 0) / 165
backlog_raw        = -0.0007 * (waiting + queued)
backlog_growth_raw = -0.17 * clip(max(0, B_t - B_(t-300)) / 30, 0, 1)
idle_reserve_raw   = -0.09 * clip((200 - idle_oht_count) / 50, 0, 1)
op_raw             = 0

# local (rail별)
predicted_oht_raw_i = -0.01 * predicted_oht_count_i
stop_time_raw_i     = -0.30 * sum(StopTime of OHTs on rail i)
density_raw_i       = -5.5  * (oht_count_i / (Distance_i / 1000))

# rail-cycle (완료된 cycle을 체류시간 비례로 배분)
rail_raw = 1.70 - route_time / route_free_flow_time
```

`TotalTat`은 simulator raw 값을 직접 읽습니다. `0`은 unavailable sentinel,
`0 < TotalTat <= 160`도 기여가 `0`이며, 160 초과에만 one-sided penalty가
붙습니다. 음수/비유한 값은 fail-fast입니다. recent-300초 completion TAT는
진단 전용입니다. StopTime 합은 의도적으로 clip하지 않습니다.

Stage 2 기준 예상 term share는 rail-cycle 30.5%, TAT 29.9%, density 18.9%,
backlog 14.6%, predicted 2.6%입니다. TAT 항이 비활성인 Stage 1 구간에서는
rail-cycle 38%, density 28%, backlog 26%가 됩니다.

TAT 종료 조건은 10,000 step grace 이후 `TotalTat >= 200`이 300회 연속
발생하는 경우이며, 종료 transition에 `-20` terminal penalty를 broadcast합니다.

계약의 단일 소스는 `oht_routing/mdp/reward/config.py`의
`REWARD_Q_CONTRACT`/`REWARD_Q_PROFILE`와 `REWARD_N_CONTRACT`/
`REWARD_N_PROFILE`입니다. Reward O/P 정의는 과거 계약 확인용으로만 보존되며
실행할 수 없습니다. 고정값 검증은
`tests/test_contextual_reward_n_contract.py`에 있습니다.

## Action 계약

기본 `action_mode`는 `region_b_rl`입니다.

```text
b_rl_i = 0.5 + 0.5 * applied_action_i
C_i    = t_ff_i + d_w_i * (c_i + 1) * b_rl_i
```

`+1` offset은 `region_b_rl`에서만 적용되며, `free_flow_residual`과
`exp_residual`은 ablation으로 남아 있고 `d_w * c` 기준선을 씁니다. region-B
curriculum 기본값은 geometric `0.05 -> 1.0`, global step 20,000입니다.

## Observation 계약

```text
center_local            [N, 14]
incoming_local          [N, 15, 14]
outgoing_local          [N, 15, 14]
incoming_relation       [N, 15, 2]
outgoing_relation       [N, 15, 2]
actor_global_state      [6]
previous_applied_action [N, 1]
critic_total_tat        [1]   # critic 전용 direct scalar
```

local feature 순서는 `free_flow_time_s`, `port_count`, `incoming_degree`,
`outgoing_degree`, `predicted_oht_count`, `reservation_port_count`, 6개 OHT
state count, `stop_time_sum`, `stopped_oht_count`입니다. actor global은
`total_tat_s`, `operation_rate`, `queued_ratio`, `waiting_ratio`,
`transferring_ratio`, `mean_reassign` 순입니다. rail identity embedding(8),
15-in/15-out topology, relation feature 2개는 유지합니다.

기본 encoder는 방향별 15개 neighbor token을 topology-rank 순으로 flatten해
투영합니다. `--use-attention`으로 이전 directional cross-attention을 선택할 수
있으며, 이는 checkpoint network 계약의 일부이므로 로드 시 저장 당시와 같은
설정을 써야 합니다.

## Checkpoint와 resume

통합 버전은 `oht_routing/version.py`의 `CONTEXTUAL_VERSION` = **v9.0.0**
하나입니다. 로드 시 통합 버전, topology/mapping hash, network config, action
mode, SALE/LAP 사용 여부, reward version을 검사합니다. 단
`--load-stage1-policy`는 frozen prefix 전용이므로 reward version을 검사하지
않습니다. replay payload는 저장하지
않으므로 resume 후 다시 채웁니다. **v8 이하의 checkpoint, Stage 1 policy,
state normalizer는 reward 의미가 달라 모두 거부됩니다.**

Checkpoint root를 생략하면 `checkpoints/ctx_td7_{version}_...` 슬러그가 자동
생성되어 실행끼리 섞이지 않습니다. periodic 기본 주기는 5,000 step,
`latest/checkpoint.pt`는 1,000 step마다 교체입니다.

```powershell
python .\PythonCode\main.py `
  --mode training --action-enabled --stage 1 `
  --resume-checkpoint ".\checkpoints\<run>\periodic\step_40000.pt" `
  --periodic-checkpoint-interval 2000 `
  --checkpoint-root ".\checkpoints\<new-run>"
```

Resume 시작 구간 제어(동시 사용 불가):

- `--resume-warmstart-steps N`: 첫 짧은 episode를 N tick에 종료하고 그 구간을
  replay에서 제외합니다. `--stage 2`와 함께 쓸 수 없습니다.
- `--resume-deterministic-first-episode`: 첫 episode의 exploration과 learner
  update만 끄고 transition은 replay에 보존합니다.
- `--resume-inference-until-replay-full`: replay가 찰 때까지 학습을 보류합니다.

## Replay 메모리

기본 capacity는 100,000 environment step입니다. 정적 rail feature는 state당
한 번만 저장하고, local count는 `uint8`/`uint16`으로 lossless packing하며,
action은 Q15 `int16`(최대 오차 약 `1.53e-5`), reward는 `float32`, LAP priority는
`float16`입니다. 예상 사용량은 시작 요약의 `estimated full replay RAM`과 W&B
`replay/storage_bytes`에서 확인합니다.

`--replay-eviction-mode`는 `fifo`(기본)와 `random`입니다. `random`은 capacity
도달 후 무작위 슬롯을 교체하며 `num_stacks=1`에서만 허용됩니다. LAP은 기본
활성이고 `--no-lap`으로 균등 rail sampling으로 바꿉니다.

## State normalizer

```powershell
# 관측 통계만 수집
python .\PythonCode\main.py --mode training --action-enabled `
  --save-state-normalizer .\PythonCode\normalizers\contextual_v9_seed0.npz
```

`--load-state-normalizer <path>`로 같은 계약의 통계만 재사용하면 effective
warm-up이 0이 됩니다. actor-global feature와 direct-critic TotalTat는 분리된
normalizer state를 씁니다.

## Multi-simulator Stage 2

중앙 owner가 CUDA, learner, frozen Stage 1 policy, checkpoint, W&B를 소유하고,
simulator 연결마다 observation/reward/transition/episode/exploration-RNG를
독립적으로 유지합니다.

```powershell
python .\PythonCode\main.py `
  --mode training --action-enabled --stage 2 --seed 0 `
  --load-stage1-policy ".\PythonCode\v9.0.0_stage1_policy.pt" `
  --num-sim 4 --port 9100 9101 9102 9103 `
  --batch-size 512 `
  --checkpoint-root ".\checkpoints\ctx_td7_v9_dist4_stage2"
```

`--num-sim`은 Python runtime이 기대하는 연결 수이며 simulator를 직접 띄우지는
않습니다. 포트 수는 `--num-sim`과 같아야 하고 중복될 수 없습니다. 분산 실행은
full-state resume을 지원하지 않으므로 frozen Stage 1 policy에서 새로
시작합니다.

## 코드 구조

```text
main.py                              얇은 실행 진입점
oht_routing/
├─ algorithms/rl/contextual_td7/     TD7, SALE, LAP, replay, checkpoint
├─ mdp/                              action, observation, transition, termination
│  └─ reward/                        Reward Q 설정, 조합, rail-cycle 추적
├─ runtime/                          client, CLI/config, protocol, TCP server
└─ utils/                            W&B, reward 진단, topology/분석 도구
oht_dispatching/                     job-to-OHT 후보 생성과 dispatch 선택
simulator/                           TCP protocol과 simulator entity 모델
tests/                               contextual 회귀 테스트
```

## 유틸리티

```powershell
python .\PythonCode\oht_routing\utils\check_rail_topology.py --help
python .\PythonCode\oht_routing\utils\download_wandb_run.py --help
python .\PythonCode\oht_routing\utils\export_wandb_run.py --help
python .\PythonCode\simulator\evaluate.py --help
```

## 테스트

```powershell
cd PythonCode
python -m unittest discover -s tests -p 'test_contextual*.py'
```

## W&B와 실험 기록

모든 새 실험 전에 `oht_routing/utils/wandb_logging.py`의 `EXP_META`를 먼저
갱신합니다. run 이름은 `_make_run_name`이 생성하므로 직접 짓지 않습니다.
`wandb.init`은 runtime config와 `EXP_META`를 함께 기록하고
`EXP_META["description"]`을 notes로 전달합니다.

모든 변경은 `oht_routing/version.py`의 단일 `vMAJOR.MINOR.PATCH` 버전으로
분리하고 현재 major의 `EXPERIMENTS_v9.md`에 호환성 영향과 함께 기록합니다.
MAJOR가 바뀌면 루트에 `EXPERIMENTS_v{new_major}.md`를 먼저 만들고 이전 major
파일은 이력으로 보존합니다. W&B metric을 바꾸면
`oht_routing/utils/export_wandb_run.py`도 같이 수정합니다.

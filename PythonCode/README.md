# Contextual TD7 v2 for OHT routing

반도체 FAB OHT의 rail cost를 학습해 혼잡 구간을 우회시키는 contextual TD7
구현입니다. `contextual-td7-v2` 브랜치는 Reward N만 실행하며, 이전 Reward
E~U 구현과 실험 기록은 `contextual-region-brl-v1` 브랜치와 루트의
`EXPERIMENTS.md`에 보존합니다.

## 실행 진입점

저장소 루트에서 실행합니다.

```powershell
conda activate aicc

# baseline
python .\PythonCode\main_contextual.py

# Reward N 학습
python .\PythonCode\main_contextual.py `
  --mode training `
  --action-enabled `
  --reward-version N `
  --replay-buffer-rail `
  --batch-size 1024 `
  --warmup-steps 10000 `
  --wandb `
  --checkpoint-root .\PythonCode\checkpoints\ctx_td7_reward_n
```

`--reward-version`은 체크포인트/실행 계약을 명시하기 위한 호환 옵션이며
`N`만 허용합니다. simulator GUI에서 Python 연동을 활성화한 뒤 Run을
시작해야 TCP 세션이 연결됩니다.

공식 진입점은 `main_contextual.py` 하나입니다. `main.py`는 기존 실행기를
위한 얇은 wrapper이며, 삭제된 region-token 경로인 `--region`은 거부합니다.

## Reward N 계약

controlled rail `i`의 최종 reward는 다음과 같습니다.

```text
r_i = 0.5 * global_raw
    + 0.5 * (local_raw_i / 2)
    + clip(30 * rail_raw_i, -1, 1)
    - 0.25 * abs(delta_b_rl_i)
```

global 항은 아래의 합입니다.

```text
tat_raw            = -11 * max(0, TotalTat - 160) / 165
op_raw             = 4 * (0.8 - operation_rate)
backlog_raw        = -0.0004 * (waiting + queued)
backlog_growth_raw = -0.16 * clip(max(0, B_t - B_(t-300)) / 30, 0, 1)
idle_reserve_raw   = -0.20 * clip((200 - idle_oht_count) / 50, 0, 1)
```

rail 항은 완료된 cycle의
`2 - route_time / route_free_flow_time`이며 rail별로 `[-1, 1]`에
clip됩니다. reward running normalizer는 사용하지 않습니다. 관측값의
state normalizer는 별도 계약으로 유지합니다.

TAT 종료 조건은 10,000 environment-step grace 이후
`TotalTat >= 200`이 300회 연속 발생하는 경우입니다. 종료 transition에는
`-20` terminal penalty를 한 번 broadcast합니다.

계약의 단일 소스는
`oht_routing/mdp/reward/config.py`의 `REWARD_N_CONTRACT`와
`REWARD_N_PROFILE`입니다.

## 코드 구조

```text
main_contextual.py
└─ 얇은 실행 진입점
oht_routing/
├─ algorithms/rl/contextual_td7/   TD7, SALE, LAP, replay, checkpoint
├─ mdp/                            action, observation, transition, termination
│  └─ reward/                      Reward N 설정, 조합, rail-cycle 추적
├─ routing/dispatch.py             simulator dispatch 선택
├─ runtime/                        client, CLI, bootstrap, protocol, TCP server
└─ telemetry/                      W&B와 reward 진단
oht_dispatching/                   job-to-OHT dispatching 전용 패키지
tests/                             contextual 회귀 테스트
```

## State normalizer

관측 통계만 수집하는 run에서는 다음처럼 frozen snapshot을 저장합니다.

```powershell
python .\PythonCode\main_contextual.py `
  --mode training `
  --action-enabled `
  --save-state-normalizer .\PythonCode\normalizers\contextual_state_n_seed0.npz
```

새 learner로 시작하면서 같은 관측 통계만 재사용하려면
`--load-state-normalizer <path>`를 사용합니다. 유효한 snapshot을
불러오면 effective warm-up은 0이 됩니다. actor, critic, replay, reward
상태는 복원하지 않습니다.

## Checkpoint와 resume

새 checkpoint 형식은
`contextual_td7_checkpoint_v8_reward_n_only`입니다. replay payload는
저장하지 않으므로 resume 후 replay를 다시 채워야 합니다. v7 Reward N
checkpoint는 명시적인 호환 경로로 읽으며, reward identity나 topology,
observation, network 계약이 다르면 fail-fast합니다.

```powershell
python .\PythonCode\main_contextual.py `
  --mode training `
  --action-enabled `
  --resume-checkpoint .\PythonCode\checkpoints\ctx_td7_reward_version_n\step_220000.pt `
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

모든 새 실험 전에 `oht_routing/telemetry/wandb.py`의 `EXP_META`를 먼저
갱신합니다. run 이름은 `_make_run_name`이 만들며 직접 짓지 않습니다.
`wandb.init`은 runtime config와 `EXP_META`를 함께 기록하고
`EXP_META["description"]`을 notes로 전달합니다.

v2 변경과 실험 결과는 루트의 `EXPERIMENTS_v2.md`에 기록합니다.
reward 공식 변경 시 reward version을 올리고, action/observation/replay/
checkpoint 계약 변경 시 해당 버전과 호환성 영향을 함께 기록합니다.
W&B metric을 바꾸면 루트와 `PythonCode`의 `export_wandb_run.py`도
같이 수정합니다.

## 테스트

```powershell
$env:PYTHONPATH="$PWD;$PWD\PythonCode;$PWD\PythonCode\tests"
& 'C:\Users\bjy66\miniconda3\envs\aicc\python.exe' `
  -m unittest discover -s .\PythonCode\tests -p 'test_contextual*.py'
```

Reward N 식의 고정값 검증은
`tests/test_contextual_reward_n_contract.py`에 있습니다.

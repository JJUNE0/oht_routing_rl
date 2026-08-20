# Contextual TD7 v2 실험 기록

이 파일은 `contextual-td7-v2` 브랜치의 코드 변경과 실험 결과만 기록한다.
이전 Reward E~U 구현과 실험 이력은 `contextual-region-brl-v1` 브랜치와
`EXPERIMENTS.md`에 보존한다.

## 기록 원칙

- 실제 실행 전에 `PythonCode/contextual_wandb.py`의 `EXP_META`를 먼저
  갱신한다.
- run 이름은 `_make_run_name`으로만 생성한다.
- reward 공식을 바꾸면 `reward_version`을 올리고 새 식과 호환성 영향을
  기록한다.
- action, observation, replay, checkpoint 계약을 바꾸면 해당 버전과 기존
  checkpoint 호환 여부를 기록한다.
- W&B metric을 바꾸면 루트와 `PythonCode`의
  `export_wandb_run.py`를 동시에 갱신한다.
- simulator/W&B를 실제로 실행하지 않은 코드 변경은 실험 결과처럼 쓰지
  않고, 검증한 테스트와 호환성만 기록한다.

## 2026-08-20 - Reward N 단일화 및 v2 구조 정리

### 목적

- v2 runtime에서 Reward N만 남긴다.
- `main_contextual.py`와 reward/runtime 책임을 분리한다.
- 이후 v2 실험 기록이 자동으로 이 파일을 향하도록 하네스를 전환한다.
- 기존 Reward N v7 checkpoint의 resume 경로는 유지한다.

### Reward 계약

- `reward_version=N`
- `contract=contextual_controlled_reward_v16_tat_one_sided_unbounded`
- TAT: `-11 * max(0, TotalTat - 160) / 165`
- OP: `4 * (0.8 - operation_rate)`
- backlog: `-0.0004 * (waiting + queued)`
- backlog growth: horizon 300, scale 30, weight `0.16`
- idle reserve: target 200, scale 50, weight `0.20`
- local divisor `2`, predicted OHT weight `0.075`
- rail: `clip(30 * (2 - route_time/free_flow_time), -1, 1)`
- smooth: `-0.25 * abs(delta_b_rl)`
- global/local alpha: 각각 `0.5`
- reward running normalizer: 사용하지 않음
- 종료: 10,000-step grace 이후 `TotalTat >= 200` 300회 연속
- terminal penalty: `-20` 1회 broadcast

기존 `EXPERIMENTS.md`의 Reward N 설명 중 threshold 170은 현재 저장된
Reward N run/checkpoint와 맞지 않는 오래된 기록이다. v2 기준값은 200이다.

### 코드 및 하네스 변경

- historical Reward E~U profile, completion-TAT, marginal-TAT, reward running
  normalizer 분기를 v2 실행 경로에서 제거했다.
- `main_contextual.py`는 57줄 진입점으로 줄이고 CLI, bootstrap, protocol,
  server, runtime summary 모듈로 분리했다.
- reward core와 OHT cycle/rail reward 추적을
  `contextual_reward.py` / `contextual_rail_reward.py`로 분리했다.
- reward 진단/W&B tick을 `contextual_runtime_diagnostics.py`로 분리했다.
- runtime 설정 계약과 검증을 `contextual_runtime_config.py`로 분리했다.
- checkpoint 형식을
  `contextual_td7_checkpoint_v8_reward_n_only`로 올리고 새 checkpoint에서
  reward normalizer payload를 제거했다.
- v7 `contextual_td7_checkpoint_v7_locked_reward_profile`은 Reward N
  identity 검증 후 호환 로드한다.
- W&B compact schema는 `contextual_wandb_compact_v10_n_only`,
  reward diagnostic schema는 `contextual_reward_diagnostic_v24_n_only`다.
- `AGENTS.md`, `CLAUDE.md`, README, `.gitignore`를 v2 기록 흐름에
  맞췄고 두 `export_wandb_run.py` metric 목록을 동기화했다.

### 호환성 및 실행 여부

- Reward N 공식 자체는 바뀌지 않았으므로 reward version은 N을 유지한다.
- non-N checkpoint와 CLI reward 선택은 fail-fast한다.
- replay는 checkpoint에 저장하지 않으므로 resume 후 다시 수집한다.
- 이 정리 작업 중 simulator와 W&B run은 시작하지 않았다.
- 실제 v7 Reward N `step_220000.pt`를 v2 runtime으로 전체 로드해
  model/optimizer, frozen state normalizer, `reward_steps=219995`,
  `runtime_env_step=220000`, `learner_updates=209899` 복원을 확인했다.
- `python -m unittest discover -s tests -p 'test_contextual*.py'`:
  249 tests 통과.

## 2026-08-20 - legacy runtime 제거

- `ClientAlgorithm.py`, `ClientAlgorithm_region.py`,
  `ClientAlgorithm_0704.py`를 삭제했다.
- 중복 TCP loop를 갖고 있던 `main.py`는 `main_contextual.py`를 호출하는
  호환 wrapper로 축소했다. 삭제된 region-token 경로인 `--region`은
  명시적으로 거부한다.
- 공식 실행 진입점은 `PythonCode/main_contextual.py` 하나다.
- 기존 `token_td7` 패키지는 import되지 않는 라이브러리 코드로 남지만,
  삭제된 runtime 클래스 이름을 가리키던 주석은 제거했다.
- Reward N, observation/action, learner, replay, checkpoint 계약은 변경하지
  않았다. simulator와 W&B run은 실행하지 않았다.
- wrapper compile/help/`--region` 거부 동작을 확인했고,
  `test_contextual*.py` 249개가 통과했다.

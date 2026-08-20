# Contextual TD7 v2 실험 기록

이 파일은 `contextual-td7-v2` 브랜치의 코드 변경과 실험 결과만 기록한다.
이전 Reward E~U 구현과 실험 이력은 `contextual-region-brl-v1` 브랜치와
`EXPERIMENTS.md`에 보존한다.

## 기록 원칙

- 실제 실행 전에 `PythonCode/oht_routing/utils/wandb_logging.py`의
  `EXP_META`를 먼저
  갱신한다.
- run 이름은 `_make_run_name`으로만 생성한다.
- reward 공식을 바꾸면 `reward_version`을 올리고 새 식과 호환성 영향을
  기록한다.
- action, observation, replay, checkpoint 계약을 바꾸면 해당 버전과 기존
  checkpoint 호환 여부를 기록한다.
- W&B metric을 바꾸면
  `PythonCode/oht_routing/utils/export_wandb_run.py`를 동시에 갱신한다.
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

## 2026-08-20 - `oht_routing` 패키지로 재배치

- 활성 contextual TD7 구현을
  `PythonCode/oht_routing/algorithms/rl/contextual_td7`로 옮겼다.
- MDP 계약은 `oht_routing/mdp`, simulator 수명주기는
  `oht_routing/runtime`, dispatch는 `oht_routing/routing`, W&B와
  reward 진단은 `oht_routing/utils`로 분리했다.
- 사용되지 않던 TD3, TD7, token-TD7, generic buffer/config를 포함한
  `cocel_rl` 패키지와 빈 `PythonCode/core` 디렉터리를 삭제했다.
- 실행 진입점 `PythonCode/main_contextual.py`와 `main.py` 호환
  wrapper는 유지했다.
- import 경로만 변경했다. Reward N 공식, action/observation/replay,
  checkpoint, W&B metric/schema 계약은 바꾸지 않았으며 simulator와
  W&B run도 시작하지 않았다.
- 새 패키지 compile/import smoke와 `test_contextual*.py` 249개 회귀
  테스트가 통과했다.

## 2026-08-20 - `oht_dispatching` 패키지 경계 추가

- `PythonCode/oht_dispatching` 패키지를 추가했다.
- `oht_routing`은 rail/path 비용과 경로 제어, `oht_dispatching`은
  job-to-OHT 할당과 dispatch 정책을 소유하도록 책임 경계를 정했다.
- 기존 command 6 선택 코드는 동작 변경 없이
  `oht_routing/routing/dispatch.py`에 유지했다.
- 실행 코드와 Reward N 계약은 변경하지 않았으며 simulator와 W&B run은
  시작하지 않았다.

## 2026-08-20 - telemetry와 보조 스크립트를 `utils`로 통합

- `oht_routing/telemetry`를 `oht_routing/utils`로 이름을 바꾸고
  runtime 및 테스트 import를 새 경로로 갱신했다.
- topology audit, W&B download/export/package, contextual/region plot,
  SQLite 결과 평가 스크립트를 `oht_routing/utils`로 이동했다.
- 중복되던 두 `export_wandb_run.py`는
  `oht_routing/utils/export_wandb_run.py` 하나로 합쳤다.
- simulator 데이터 모델과 공식 실행 진입점은 이동하지 않았다.
- Reward N, checkpoint, W&B metric/schema 계약은 변경하지 않았으며
  simulator와 W&B run은 시작하지 않았다.
- 새 패키지 compile/import와 topology/W&B/evaluation 도구의 CLI를
  확인했고, `test_contextual*.py` 249개가 통과했다.
- plot 도구가 사용하는 `matplotlib`을 `requirements.txt`에 명시했다.

## 2026-08-20 - 실행 진입점을 `main.py`로 단일화

- 호환 wrapper였던 기존 `PythonCode/main.py`를 제거하고
  `main_contextual.py`의 실제 contextual runtime 진입점을
  `PythonCode/main.py`로 옮겼다.
- `main_contextual.py`는 삭제했으며 공식 실행 명령과 테스트 import를
  `main.py`로 갱신했다.
- runtime, Reward N, checkpoint 및 W&B 계약은 변경하지 않았고
  simulator와 W&B run은 시작하지 않았다.
- 새 `main.py` compile/help와 `main_contextual.py` 제거를 확인했고,
  `test_contextual*.py` 249개가 통과했다.

## 2026-08-20 - simulator protocol/model 패키지 분리

- 최상위 `PClient.py`, `Job.py`, `Oht.py`, `RailLine.py`와 관련
  payload 모델을 `PythonCode/simulator` 패키지로 옮겼다.
- 파일명은 snake_case로 정리하고 기존 wire-facing 클래스명과 simulator
  protocol 동작은 유지했다.
- runtime, topology audit, reward 및 protocol 테스트 import를
  `simulator` 패키지 경로로 갱신했다.
- `simulator`는 저수준 TCP/wire 계약, `oht_routing`과
  `oht_dispatching`은 정책을 소유하도록 경계를 분리했다.
- runtime console prefix를 파일명 기반 `[main-contextual]`에서
  `[contextual-runtime]`으로 바꿨다.
- Reward N, checkpoint 및 W&B 계약은 변경하지 않았으며 simulator와
  W&B run은 시작하지 않았다.
- simulator package compile/import, `main.py`와 topology utility CLI를
  확인했고 `test_contextual*.py` 249개가 통과했다.

## 2026-08-21 - simulator DB 평가 도구 이동

- `oht_routing/utils/evaluate_simulation.py`를
  `PythonCode/simulator/evaluate.py`로 옮겼다.
- simulator 결과 DB 평가는 routing utility가 아니라 simulator 도구가
  소유하도록 경계를 바로잡았다.
- 평가 공식과 기본 결과 경로는 변경하지 않았다.

## 2026-08-21 - runtime config 조립 통합

- `runtime/bootstrap.py`의 CLI 설정 조립, checkpoint runtime 설정 복원,
  process seed 설정을 `runtime/config.py`로 합치고 `bootstrap.py`를
  삭제했다.
- `runtime_config_from_args()`는 dataclass 필드와 CLI의 같은 이름을
  자동 매핑하고, 이름이 다른 다섯 필드만 alias로 관리하도록 줄였다.
- seed 적용은 config 객체 생성의 부작용이 되지 않도록 `main.py`에서
  `seed_everything(config.seed)`를 명시적으로 호출하는 방식을 유지했다.
- `ContextualRuntimeConfig`가 runtime 설정의 최상위 원본이며,
  `ClientAlgorithm`이 이를 learner/network/reward 설정으로 변환한다.
- Reward N, checkpoint 및 W&B 계약은 변경하지 않았으며 simulator와
  W&B run은 시작하지 않았다.
- 기본 CLI config build/seed와 `main.py` CLI를 확인했고,
  `test_contextual*.py` 249개가 통과했다.

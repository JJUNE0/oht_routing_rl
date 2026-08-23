# Contextual TD7 v3 실험 기록

이 파일은 `CONTEXTUAL_VERSION` v3.x의 코드 변경과 실험 결과만
기록한다. v2.x 이력은 `EXPERIMENTS_v2.md`, 이전 Reward E~U 이력은
`EXPERIMENTS.md`에 보존한다. 실험 작성 규칙은 루트 `AGENTS.md`를 따른다.

## 2026-08-23 - v3.1.2 major-scoped experiment logs

- `EXPERIMENTS_v2.md`에 잘못 섞여 있던 v3.0.0, v3.1.0, v3.1.1
  기록을 이 파일로 옮겨 major별 이력을 분리했다.
- `AGENTS.md`와 `CLAUDE.md`는 현재 major의
  `EXPERIMENTS_v{MAJOR}.md`에 버전 섹션을 작성하도록 통일했다.
- MAJOR 버전을 올릴 때 새 `EXPERIMENTS_v{new_major}.md`를 먼저
  만들고, 이전 major 파일은 archive로 보존하는 규칙을 추가했다.
- 루트 `.gitignore`는 현재 및 향후의 `EXPERIMENTS_v*.md`를
  자동으로 추적한다.
- runtime, observation, action, replay, checkpoint, Reward N 계약은
  v3.1.1과 같다.

## 2026-08-21 - v3.1.1 canonical runtime config

- 모든 runtime 기본값의 단일 원본을 `ContextualRuntimeConfig`로 통합했다.
  CLI는 독자적인 기본값을 갖지 않고, 사용자가 명시한 인자만
  config에 override한다.
- `main.py`, server, protocol, startup summary는 모두 해석이 완료된
  `client.config`만 사용하도록 변경했다. simulator end time과 console
  log interval도 단일 config에서 관리한다.
- 기본 region-B action curriculum은 기존 최고 성능 설정인
  `0.05 -> 1.0`, global step 20,000, geometric을 유지한다.
- CLI의 `--no-lap`/`--no-sale`이 parser에서는 보이지만 runtime config에
  반영되지 않던 destination 불일치 버그를 수정했다. 이제 신규 학습과
  resume 모두 명시한 LAP/SALE 선택을 그대로 유지한다.
- W&B의 mode-aware 기본값(training/actor inference on, baseline off)도
  config 내부에서 해석하므로 CLI와 programmatic runtime이 같은 계약을 쓴다.
- observation/network/replay/checkpoint/Reward N 계약은 v3.1.0과 같다.
- 전체 unittest 287개가 통과했다. simulator/W&B 실험 run은 아직
  시작하지 않았다.

## 2026-08-21 - v3.1.0 packed 100k replay

- 기본 replay capacity를 100,000 environment step으로 복원했다.
- replay payload는 checkpoint에 저장되지 않으므로 capacity는 resume 시 저장된
  값으로 덮어쓰지 않고 현재 CLI 값(기본 100,000)을 유지한다.
- LAP 사용 여부도 현재 실행 인자를 유지한다. 따라서 새 학습은 물론 resume
  설정에서도 `--no-lap`이 저장된 runtime config에 의해 조용히 무시되지 않는다.
- local physical state의 정적 4개 feature는 topology별 한 번만 저장하고,
  동적 count는 simulator wire 범위에 맞춰 `uint8`/`uint16`으로 lossless
  packing한다. `oht_density`는 상태별 OHT count와 정적 rail Distance에서
  float32 원래 값으로 복원한다.
- policy/applied/previous-applied action은 `[-1, 1]` Q15 `int16`으로 저장한다.
  최대 절대 복원 오차는 `1 / (2 * 32767)`, 약 `1.53e-5`다.
- Reward N 학습 신호는 `float32`를 유지하고 LAP priority만 `float16`으로
  저장한다. priority 합과 제곱합은 실제 저장된 표현값으로 계산한다.
- 최악의 잦은 episode 종료에서도 100,000 transition을 보존하기 위해 state
  ring의 `2C` 계약은 유지한다. 단순 `C+1` 축소는 적용하지 않았다.
- 4,999 physical rail, 4,996 controlled rail 기준 numeric replay 예상치는
  LAP 활성 18.64 GiB, `--no-lap` 17.71 GiB다. 기존 LAP 활성 약 70.78 GiB
  대비 약 73.7% 감소한다.
- 동일한 `(episode_id, env_step)` transition의 중복 삽입은 state를 쓰기 전에
  거부하고, replay에서 완전히 빠진 episode의 history metadata도 제거한다.
- observation/network/checkpoint/Reward N 의미는 v3.0.0과 같고 v3.0.0
  checkpoint는 같은 major 호환 규칙으로 로드할 수 있다. replay 자체는
  checkpoint에 저장하지 않는다.
- 전체 unittest 279개가 통과했다.
- simulator/W&B 실험 run은 아직 시작하지 않았다.

## 2026-08-21 - v3.0.0 bottleneck-aware observation

- Reward 공식은 N으로 유지하고 observation/network/topology 계약만 breaking
  변경했다.
- 모든 center/incoming/outgoing rail은 동일한 physical 16개 feature를 쓴다:
  free-flow time, port/incoming/outgoing degree, OHT density, predicted OHT,
  next-10 route OHT, reservation, OHT state 6종 count, StopTime 합과 stopped
  OHT count.
- rail identity는 local normalizer에 섞지 않고 physical rail row를 입력으로
  받는 공유 `nn.Embedding(num_rails, 8)`로 인코딩한다. main encoder와 SALE
  encoder는 각자 embedding을 소유한다.
- global state는 성능·workload·fleet composition·congestion·strict 60초 trend
  17개 feature로 교체했다. 60초 history가 차기 전 trend는 0이고 episode
  reset/reconnect에서 history를 지운다.
- 저장된 actor-inference 병목 데이터를 비교해 directional context를 10에서
  incoming 15 + outgoing 15로 늘렸다. held-out future StopTime 설명력은 10보다
  15에서 개선됐고 16~20의 추가 이득은 작았다.
- topology hash는 graph/free-flow time 외에 density와 static local feature에
  영향을 주는 rail `Distance`와 `PortCount`까지 검증한다.
- `predicted_oht_count`와 `next_10_route_oht_count`의 전체 rail 및 union-nonzero
  상관/MAE/availability를 W&B와 CSV export에 기록한다.
- local physical 16/global 17 state ring의 메모리 증가를 반영해 CLI replay
  capacity 기본값을 100,000에서 10,000 environment step으로 낮추고 estimator의
  neighbor/LAP 위치 인자 버그를 수정했다.
- v2 checkpoint, normalizer snapshot, 10-neighbor topology cache는 호환되지
  않는다. 과거 `step_400000.pt`의 fingerprint 승격 예외도 제거했다.
- 실제 4,999-rail capture로 v3 cache를 재생성해 runtime load와 hash를
  검증했고, 전체 unittest 270개가 통과했다. simulator/W&B 실험 run은 아직
  시작하지 않았다.

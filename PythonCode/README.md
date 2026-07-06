# OHT 라우팅 강화학습 (Dynamic Link Weight Control)

반도체 FAB의 OHT(천장 반송차, Overhead Hoist Transport) 혼잡을 완화하기 위한
**강화학습(RL) 기반 동적 링크 가중치 제어** 코드입니다.

매 의사결정 시점에 RL 에이전트가 각 RailLine의 **cost(가중치)** 를 산출하고,
시뮬레이터는 이 cost를 반영한 최단경로(Dijkstra) 라우팅으로 OHT를 운행합니다.
가중치를 동적으로 조정해 혼잡 구간을 우회시키는 것이 목표이며,
핵심 성능지표는 **평균 TAT(반송 소요시간)** 입니다.

---

## 1. 시스템 구성

```
 ┌─────────────────────────┐         TCP (127.0.0.1:9100)        ┌──────────────────────────┐
 │  Python (본 코드)        │  ◀───────────────────────────────▶ │  Pinokio.Simulator.exe   │
 │  = TCP 서버              │   상태(state) / cost(action) 교환    │  = TCP 클라이언트 (GUI)  │
 │  main.py / PClient /     │                                     │  (별도 제공)             │
 │  ClientAlgorithm(RL)     │                                     │                          │
 └─────────────────────────┘                                     └──────────────────────────┘
```

- **Python 쪽이 TCP 서버**입니다. 먼저 `main.py`를 실행해 서버를 띄운 뒤,
  시뮬레이터(GUI)에서 Python 연동을 켜고 **Run** 하면 시뮬레이터가 클라이언트로 접속합니다.
- 시뮬레이터는 매 스텝 명령코드 `v`를 보내고, Python은 `v` 종류에 따라 상태를 수신하거나
  행동(RailLine cost)을 응답합니다(아래 §3 프로토콜 참고).
- 시뮬레이터 실행파일/모델 DB는 본 코드 패키지에 포함되지 않습니다(별도 전달).

---

## 2. 파일 구조

```
PythonCode/
├── main.py                     # 진입점. TCP 서버 루프 + v-code 분기 + 재접속 회복
├── PClient.py                  # 시뮬레이터 통신 계층(바이너리 프로토콜 송수신/파싱)
├── ClientAlgorithm.py          # per-rail TD7 본체(상태·행동·보상 설계, 학습 루프)
├── ClientAlgorithm_region.py   # region-token TD7 (ClientAlgorithm 상속, 어텐션 기반)
├── ClientAlgorithm_0704.py     # 참조용 — global+local 보상 전부 사용, 현재 최고 성능 기록
├── eval.py                     # 결과 DB(COMMAND_LOG) 기반 성능평가(TAT/가동률/반송건수)
├── requirements.txt            # 파이썬 의존 패키지
│
├── cocel_rl/                   # RL 라이브러리(알고리즘·버퍼·학습기·로거)
│   ├── algorithms/
│   │   ├── td3/                #   TD3 구현
│   │   ├── td7/                #   TD7 구현
│   │   └── token_td7/          #   Token-TD7 (리전·레일 어텐션 기반 확장 아키텍처)
│   ├── buffers/                #   off-policy 리플레이 버퍼
│   ├── core/                   #   learner(학습 루프), logger
│   ├── configs/                #   알고리즘 하이퍼파라미터(YAML)
│   └── utils/
│
└── (시뮬레이터 데이터 모델 — PClient가 수신 데이터를 담는 클래스)
    ├── RailLine.py             # 레일 라인 정보(거리/속도/분기·합류 등)
    ├── RailLineCost.py         # 라인별 cost(= RL 행동이 채우는 출력값)
    ├── Oht.py                  # OHT 상태(위치/적재/목적지/정체시간 등)
    ├── Job.py                  # 반송 작업(상태/우선순위/경로 등)
    ├── LinePassTime.py         # 라인 통과시간
    ├── ohtLinePassTime.py      # OHT별 라인 통과시간
    ├── ohtPos.py               # OHT 위치
    └── OHTCommandTime.py       # 명령 수행 시간
```

> 학습을 새로 실행하면 `checkpoints/<타임스탬프>/`(모델),
> `reward_log.csv`(스텝별 보상 로그), `log/`(통신 로그)가 추가로 생성됩니다.

---

## 3. 통신 프로토콜 (명령코드 `v`)

시뮬레이터가 매 스텝 보내는 `v` 값에 따라 Python이 처리합니다 (`main.py`).

| `v` | 의미 | Python 처리 |
|----|------------------------|--------------------------------------------------|
| 0 | ActiveData (상태→행동) | 상태 수신 → **RL이 RailLine별 cost 산출** → 전송 |
| 1 | 에피소드 종료 | 종료 신호 기록(루프 계속) |
| 2 | 재시작(Reset) | 에피소드 상태 초기화 |
| 3 | Snapshot | 스냅샷 데이터 수신·갱신 |
| 4 | 단일 OHT 경로 요청 | 해당 OHT 경로 계산 후 응답 |
| 5 | Reroute | 재라우팅 대상 OHT 경로 재계산 후 응답 |
| 6 | Assign | 작업 할당 처리 후 응답 |

> **스트림 동기화(desync) 회복**: 위 집합(`{0..6}`) 밖의 `v`가 오면 TCP 스트림 정렬이
> 깨진 것으로 보고, 가비지를 파싱하지 않고 소켓을 닫은 뒤 재접속해 회복합니다.
> 재접속해도 학습 상태(리플레이 버퍼/네트워크)는 유지됩니다(`client`는 루프 밖에서 1회 생성).

바이너리 프로토콜(빅엔디언, 가변길이 블록)의 인코딩/디코딩은 모두 `PClient.py`에 있습니다.

---

## 4. 강화학습 설계

### 4-1. per-rail TD7 (`ClientAlgorithm.py`)

| 항목 | 내용 |
|------|------|
| **상태 (obs, 18차원)** | 글로벌 10 (혼잡/가동률/작업 통계 등) + RailLine 8 |
| **행동 (act)** | RailLine별 cost 스케일 1차원, 범위 `[-1, 1]` (중립=Dijkstra 기본 가중치) |
| **알고리즘** | `TD7`(기본) 또는 `TD3` — `train_config()`에서 선택 |
| **보상** | `reward_components` config로 항별 on/off 제어 (§4-3 참고) |
| **워밍업/커리큘럼** | 초기 `warmup_steps` 동안 데이터 수집 후 RL 시작. 행동 스케일을 작게(≈Dijkstra) 출발해 점진 확대 |

하이퍼파라미터는 모두 **`train_config()`** 한 곳에 모여 있습니다
(학습률·버퍼·배치·`reward_weights`·`curriculum`·`reward_components`·알고리즘 설정 등).

- **이어 학습(resume)**: 기본값은 `resume=False`. 이어가려면 `resume=True`, `resume_ckpt_dir` 지정.

### 4-2. region-token TD7 (`ClientAlgorithm_region.py` + `cocel_rl/algorithms/token_td7/`)

`ClientAlgorithm`을 상속한 확장 아키텍처. 레일 전체를 **인접 레일 기반 리전(region)** 으로 묶고,
글로벌 컨텍스트와 리전 내 레일 토큰을 **교차 어텐션**으로 융합해 리전 단위 액션을 산출합니다.

| 항목 | 내용 |
|------|------|
| **입력** | 글로벌 컨텍스트 10차원 + 레일 토큰 시퀀스 (레일 obs 18차원 × 레일 수) |
| **어텐션** | `ContextConcatRailAttention` — 글로벌·레일 임베딩 concat → Q/K 프로젝션 → Multi-Head Attention |
| **출력** | 리전별 1차원 액션 → 해당 리전 내 모든 레일에 동일 cost 적용 |
| **리전 구성** | `region.target_size=50` (레일 수 기준), 인접 레일 클러스터링 |
| **설정** | `config["token_td7"]` — `embed_dim`, `num_heads` |

> **상속 관계**: `get_step_reward`, `get_global_reward`, `get_local_reward` 등 보상 로직은
> `ClientAlgorithm.py`에서 그대로 상속 → 부모 코드 수정이 region에도 자동 반영됨.

### 4-3. 보상 설계 (`reward_components` config)

`train_config()`의 `reward_components`로 각 보상 항을 코드 수정 없이 on/off 할 수 있습니다.

```python
"reward_components": {
    "use_global_reward": True,   # 글로벌 TAT 밀도 신호 (w_tat_dense * g)
    "use_local_reward":  False,  # 레일별 로컬 보상 (혼잡/대기/용량 등)
    "use_rail_tat":      True,   # rail-level TAT 페널티 (primary signal)
    "global": {
        "use_tat":     True,
        "use_op":      False,    # 가동률 항
        "use_backlog": True,     # 적체 페널티
    },
    "local": {
        "use_oht_count":  True,
        "use_predicted":  True,
        "use_stop":       True,
        "use_idle":       True,
        "use_capacity":   True,
    },
}
```

| reward_version | 설명 | `use_global` | `use_local` | `use_rail_tat` |
|:-:|---|:-:|:-:|:-:|
| **–** (0704, 최고 성능) | global + local 전부 사용 | ✅ | ✅ | – |
| **I** | rail_tat_penalty 단독 | ❌ | ❌ | ✅ |
| **J** | rail_tat + dense global 보조 | ✅ | ❌ | ✅ |

---

## 5. 실행 방법

### 5-1. 의존성 설치
```bash
pip install -r requirements.txt
```
> `torch`는 환경(CUDA 유무)에 맞는 빌드를 설치하세요. GPU가 없으면 자동으로 CPU로 동작합니다.

### 5-2. 학습

**per-rail TD7** (기본)
```bash
set PYTHONUTF8=1
set WANDB_MODE=offline        # wandb 미사용/오프라인 (online 쓰려면 생략)
python -u main.py
```

**region-token TD7** (`--region` 플래그)
```bash
set PYTHONUTF8=1
set WANDB_MODE=offline
python -u main.py --region
```

`--region`을 붙이면 `main.py`가 `ClientAlgorithm_region`을 로드해
리전 어텐션 아키텍처로 실행됩니다. 플래그 없이 실행하면 per-rail TD7이 기본입니다.

실행 후 시뮬레이터(GUI)에서 모델 DB를 열고 Python TCP 연동을 켠 뒤 **Run**.
학습이 진행되며 `checkpoints/<타임스탬프>/`에 모델이 주기적으로 저장됩니다.

### 5-3. 평가
시뮬레이션 결과 DB(`COMMAND_LOG` 테이블 포함)에 대해 TAT/가동률/반송건수를 산출합니다.
```bash
python eval.py --save_dir <결과DB 폴더> --db_filename <파일명.db>
```
- 평가 구간/기준값(baseline)은 `eval.py` 상단 상수에 정의되어 있습니다.
  - 평가 윈도우: `09:00 ~ 19:00`
  - baseline: TAT `2.90706`분, 가동률 `0.79292`, 반송건수 `175299` (Dijkstra 기준)
- 개선율은 `(baseline − 신규) / baseline` 으로 출력됩니다.

---

## 6. 실험 로드맵 (TODO)

### ✅ 완료

- [x] **TD7 per-rail — global + local reward 전부 사용** (`ClientAlgorithm_0704.py`)
  - 현재 최고 성능. TAT 평균 **+1.0% 개선**, 최대 ep4 **+1.34%**
  - global: TAT marginal + backlog penalty / local: oht_count, predicted, stop, idle, capacity
- [x] **reward_components config 추가** — 보상 항 코드 수정 없이 on/off
- [x] **region-token TD7 초기 학습 진행** (`ClientAlgorithm_region.py`)
  - 리전 단위 어텐션 아키텍처 동작 확인, 학습 진행 중

---

### 🔲 진행 예정

#### 보상 ablation (per-rail TD7 기반)

- [ ] **reward_version J 검증** — rail_tat_penalty + w_tat_dense × g (actor gradient vanishing 방지)
  - reward_version I(rail_tat only)에서 NaN explosion 재현됨 → dense signal 추가로 해결 여부 확인
- [ ] **local reward 제거 효과 확인** — `use_local_reward=False` 단독 실험
- [ ] **global reward 항별 제거** — `use_op=False` (가동률), `use_backlog=False` (적체) 효과 분리
- [ ] **최소 보상 조합 탐색** — rail_tat + 가장 기여도 높은 항만 남기는 ablation 완성

#### 아키텍처 실험

- [ ] **region-token TD7 충분한 학습 및 per-rail 대비 성능 비교**
  - 리전 어텐션이 per-rail 대비 TAT 개선에 실제 기여하는지 확인
- [ ] **region-token TD7 + reward_version J 조합**

#### 기타

- [ ] `use_state_normalizer=False` 효과 확인 (관측 정규화 제거)
- [ ] 하이퍼파라미터 튜닝 (embed_dim, num_heads, buffer_capacity for region)

---

## 7. 성능 평가 결과 (현재 최고 — per-rail TD7, global + local reward)

- **wandb run**: `ozl3xyft`
- **설정**: per-rail TD7, `reward_alpha=0.7`, global reward (TAT marginal + backlog) + local reward 전부 사용
- **평가 방법**: `eval.py` — 결과 DB(`COMMAND_LOG`)에서 TAT·가동률·반송건수 산출
- **평가 파일**: `eval_summary_ep_0629_all.db.csv`

**Baseline (Dijkstra)**: TAT **2.9071분 (174.4초)** / 반송건수 **175,299** / 가동률 **0.793**

> ep1은 워밍업(반송건수 미달, 미완성 에피소드), ep2~21은 sub-episode(시뮬 재시작 전 단편)로 평가 제외.  
> ep22부터 반송건수 175,299건의 풀에피소드.

| 에피소드 | TAT (분) | TAT (초) | TAT 개선율 | 가동률 | 반송건수 |
|:--:|:--:|:--:|:--:|:--:|:--:|
| ep22 | 2.9216 | 175.3 | -0.50% | 0.796 | 175,299 |
| ep23 | 2.9421 | 176.5 | -1.21% | 0.795 | 175,285 |
| ep24 | 2.8620 | 171.7 | +1.55% | 0.781 | 175,299 |
| ep25 | 2.8740 | 172.4 | +1.14% | 0.783 | 175,299 |
| ep26 | 2.8495 | 171.0 | +1.98% | 0.778 | 175,299 |
| ep27 | 2.8625 | 171.8 | +1.53% | 0.781 | 175,299 |
| ep28 | 2.8726 | 172.4 | +1.18% | 0.784 | 175,300 |
| ep29 | 2.8589 | 171.5 | +1.66% | 0.781 | 175,299 |
| ep30 | 2.8340 | 170.0 | +2.51% | 0.775 | 175,298 |
| ep31 | 2.8444 | 170.7 | +2.16% | 0.777 | 175,300 |
| ep32 | 2.8438 | 170.6 | +2.17% | 0.777 | 175,298 |
| **ep33** | **2.8249** | **169.5** | **+2.83%** | 0.773 | 175,299 |
| ep34 | 2.8283 | 169.7 | +2.71% | 0.773 | 175,297 |
| ep35 | 2.8277 | 169.7 | +2.73% | 0.773 | 175,298 |
| ep36 | 2.8274 | 169.6 | +2.74% | 0.773 | 175,299 |

**요약**

- **최고 성능: ep33, TAT 2.8249분 (169.5초), +2.83% 개선**
- 수렴 구간(ep30~36): TAT **169.5 ~ 170.7초**, 개선율 **+2.16% ~ +2.83%** 안정
- 반송건수 175,299건 유지 — 처리량 동일, 소요시간만 단축
- 가동률 0.773~0.784 (baseline 0.793 대비 -1.1%p) — 혼잡 구간 우회 중 일부 구간 공차 증가에 따른 자연스러운 트레이드오프

---

## 8. 참고

- **인코딩**: Windows(cp949) 콘솔에서의 한글/이모지 출력 오류를 막기 위해 `main.py`가
  stdout/stderr를 UTF-8로 재설정합니다. 실행 시 `PYTHONUTF8=1` 사용을 권장합니다.
- **도메인 enum**
  - OHT 상태: `0`=IDLE, `1`=STAGE, `2`=MOVE_TO_LOAD, `3`=LOADING, `4`=MOVE_TO_UNLOAD, `5`=UNLOADING
  - Job 상태: `1`=QUEUED, `2`=RESERVED, `3`=WAITING, `5`=TRANSFERRING, `7`=COMPLETED

# Headless 검토와 최종 TAT 5% 개선 방향

v10.6.4 문서 기준. .NET 대상 지정 누락으로 유휴 차량 동점 정렬이 달라지는 원인을
찾아 수정했다. [GUI/headless 수정 보고서](HEADLESS_PARITY_FIX.md) 참조.
아래 v10.4~10.5 headless baseline 및 정책 점수는 수정 전 런타임 결과다.
수정 후 동일 평가 구간으로 baseline과 후보 정책을 다시 평가해야 한다.

v10.6.0 후속 확인: 회의 기준은 **12시간 평가 + 30분 완료 여유**다. 아래의
09~19시/174.4236초 계산은 legacy 10시간 비교로 구분한다. 새로운 CSV에는
07~19시 `meeting_eval_*`를 별도로 기록하며 같은 시간창의 baseline 없이는
개선율을 계산하지 않는다. [회의 기준과 12시간 재평가](C:/Users/junheemike/Documents/SKH/AICC0601_2612_반출용/oht_routing_rl/INPUT_TIME_AUDIT.md).

2026-09-11 · 문서 기준 v10.6.4 · 학습 런 v10.6.3 · 45,000초 baseline(v10.4.2) 완료

입력 DB를 직접 조회한 결과 실제 데이터는 약 **20시간**이다. 45,000초는
12시간 30분의 실행 길이이고, 아래 09~19시는 현재 evaluate.py의 별도 평가 구간이다.
[시간 범위 실측과 GUI 표시 이유](C:/Users/junheemike/Documents/SKH/AICC0601_2612_반출용/oht_routing_rl/INPUT_TIME_AUDIT.md)를 참고한다.

## 현재 판단

Headless 실행을 유지한다. 원본 엔진·모델 DLL의 이동·배차·경로 계산은 변경하지 않았다.
별도 비교에서 GUI 기본 초기화, TCP 연결 순서, private TCP 대기 패치,
모델 로딩 함수의 차이가 초기 궤적을 바꾼다는 증거는 나오지 않았다.

아래 v10.4~10.5 수치는 수정 전 headless 런타임의 역사 기록이다. .NET 4.8 대상 지정
수정 후 고정 episode54의 2,000초 GUI/headless TAT는 모두 167.7초이며, 공통 cutoff
07:32:59의 8,921개 작업과 43개 필드가 일치한다. 전체 45,000초 GUI 동등성은 검증하지 않았다.

현재 v10.6.3 fresh 학습은 `--episodes 50`으로 기동되었으며, CLI 옵션을 생략할 때의
기본값 100과 구분한다. 실행 결과는
`results/headless/run_0911_1740_v10.6.3_b_rl_0.0-1.0_Q_headless`, episode별 best
체크포인트는 `checkpoints/run_0911_1740_v10.6.3_b_rl_0.0-1.0_Q_episodebest`이다.
W&B는 [vp1o1994](https://wandb.ai/offpolicy-postech/oht-routing-contextual-td7/runs/vp1o1994)이다.
50회 종료 후 검증 inference를 실행하도록 finalizer에 연결했고 기동을 확인했다.
학습 완료 여부는 이번 문서 작업에서 조회하지 않았다. 최종 재검증은
[HEADLESS_REVALIDATION.md](HEADLESS_REVALIDATION.md)에 기록한다.

## 1. 서로 다른 TAT를 분리해야 한다

`PythonCode/simulator/evaluate.py`의 실제 기준은 다음과 같다.

- **ACTIVATED_TIME > 2026-02-16 09:00:00, < 19:00:00**인 명령을 선택한다.
- TAT는 `COMPLETED_TIME - ACTIVATED_TIME`이다. `TOTAL_TIME` 열의 평균과 다를 수 있다.
- 19시 이전에 발생해 19시 이후 완료된 명령도 평가에 포함된다.
- baseline은 **2.90706분 = 174.4236초**, 5% 목표는 **165.70242초**다.
- 현재 개선율이 정확히 4.2%라면 TAT는 167.0978088초이며, 추가로 **1.3953888초**를 낮춰야 한다.

앞서 비교한 164.4/168.2초는 **07:00부터 첫 2,000초**의 누적 TAT다.
최종 평가 대상 시간에 도달하기 전이므로 이 수치에 174.4236 baseline을 직접 대입해
5% 달성 여부를 판정하면 안 된다.

새 `evaluation.csv`에는 기존 episode `tat_s`와 별도로 `eval_tat_s`,
`eval_improvement_pct`, `eval_window_complete`, `eval_tat_target_met`, 명령 수·coverage를 기록한다.
짧거나 중단된 실행에는 최종 목표 달성 판정을 내리지 않는다.
명령 수가 감소해 평균만 좋아지는지도 함께 확인해야 한다.

기존 native 결과 `results/0909/ud7_st2_3.db`에서 원래 evaluate.py의 `tat()`와 새 계산을 비교했다.
둘 다 **168.8554917911214초**, 차이 **0.0초**였다. 같은 DB의 `TOTAL_TIME` 평균은
169.25429925612386초였다. 이 DB는 나중에 덮어쓴 학습 결과이며 o9s1ssl5 원본이나
사용자가 말한 4.2% inference 결과로 간주하지 않았다.

## 2. GUI 차이의 원인을 분리한 결과

모든 아래 실험은 기존 v10.0.0 Stage 1 파일의 같은 복사본을 사용했다.
SHA-256: `3dfd5fd8a308540a065934fc9a46abf382252c01f79af8408b1e69e80eeaa0d3`.
학습·탐색·replay 수집 없이 CUDA, action scale 1.0으로 실행했다.

| 비교 | 범위 | 결과 |
| --- | --- | --- |
| 기존 headless 반복 + 모든 tick 진단 | 2,000초 | 완료 9,030개, native TAT 168.2010899224806초로 재현 |
| GUI AI20 기본 파라미터 복사/적용, 초기화 순서, animation 간격 | 2,001개 관측 | TAT·queue·waiting·policy mean·route 상관 값 모두 기존 headless와 같음 |
| TCP 연결을 모델 로드 앞으로 이동 | 301개 관측 | 위 지표 차이 없음 |
| GUI 원본 TCP DLL 사용, supervisor sleep 제거 | 301개 관측 | 위 지표 차이 없음 |
| 설치된 GUI의 `MainFrame.LoadSimModel`을 직접 호출 | 301개 관측 | 위 지표 차이 없음 |

GUI 로더 직접 호출은 **모델 로딩 함수 비교**다. 렌더링·GUI 전체 RunCycle을 실행한 검증은 아니다.
실험용 GUI 객체의 필수 입력 필드와 수명을 보완한 최종 실행만 위 표에 포함했다.
이 진단 코드는 production 실행기에 넣지 않았다.

GUI와 다른 첫 기록은 route 예측 진단에서 sim 39초, policy mean에서 59초,
waiting/transferring에서 89초, TAT에서 109초다. 따라서 마지막 한 tick을 빼면
차이가 해결된다는 설명은 성립하지 않는다. 워밍업 native 명령 비교에서도 같은 경로의
이동 시간부터 차이가 나타난다. 가중치 파일을 잘못 읽었거나 TCP sleep 때문에 생긴
차이라는 가설은 현재 실험으로 뒷받침되지 않는다.

근거 파일은 `results/gui_compare/`의 `diagnostic_result.json`, `gui_defaults_result.json`,
`protocol_ab_result.json`, `gui_loader_result.json`, `warmup_command_comparison.json`에 있다.
과거 GUI와의 상세 비교는
[GUI_HEADLESS_COMPARISON.md](C:/Users/junheemike/Documents/SKH/AICC0601_2612_반출용/oht_routing_rl/GUI_HEADLESS_COMPARISON.md)에 보존했다.

## 3. 에피소드마다 복사할 필요는 없다

실행 파일 4개는 합계 **118,295 bytes**, 복사 10회 측정 중앙값 **2.696ms**였다.
native 모델 로딩은 약 2초, 2,000초 inference는 약 500초였다.
파일 복사는 속도의 주된 병목은 아니지만, 중복을 없애고 실행 도중 바이너리가 바뀌지 않게 수정했다.

현재 동작은 다음과 같다.

1. 실행별 `runtime/`에 실행 파일을 **한 번만 빌드**하고 SHA-256을 기록한다.
2. Python 서버·학습기·replay는 계속 유지한다.
3. 각 에피소드는 같은 실행 파일로 새 native 프로세스를 시작한다.
4. 입력 DB 하나를 계속 읽을 수 있다. 에피소드 수만큼 입력 파일을 복제할 필요가 없다.
5. 평가 체크포인트·Stage 1 prefix도 실행 시작에 한 번만 고정한다.
6. 결과 DB와 요청한 에피소드별 체크포인트는 각각 저장한다.

native 프로세스 재시작은 유지했다. 엔진의 singleton, 이벤트 큐, 차량·배차 상태를
초기화하기 위한 것이다. GUI도 에피소드 사이 `RemoveModel`/`AllClear`/재로딩을 한다.
한 에피소드 안에서는 45,000초까지 중간 재시작 없이 계속 롤아웃한다. 입력은
07:00~다음 날 03:00의 20시간이고, 45,000초는 12시간 30분 실행 계약이다.
약 45.9MB 입력의 임시 복호화본은 성공 후 정리한다.

검증: 30초 baseline **2개 에피소드 완료**, 실행 파일 **1개**, Python episode ID **1 → 2**,
별도 결과 DB 2개, runtime 준비 **0.781초**, 짧은 실행의 최종 TAT 판정은 비어 있었다.
상세: `results/gui_compare/reuse_result.json`.

```powershell
# 입력 파일 하나로 자동 반복. input 폴더에 100개 복사본을 만들 필요가 없다.
.\.venv\Scripts\python.exe PythonCode\run_headless.py --mode stage1 --input "..\db\base\AICC_Input_260403.db" --episodes 100 --port 9100 --name 'stage1_{episode:04d}'
```

위 명령은 새 학습을 시작하는 예시다. 현재 진행 중인 학습을 위해 다시 실행할 필요는 없다.

GPU actor 계산은 이미 사용하고 있다. 동시 학습 중 측정된 Python tick 평균은 약 125.3ms,
그중 observation build 약 29.8ms, encoder/actor 약 28.4ms였다. 파일 복사보다
관측 생성·native 이벤트·네트워크 왕복을 프로파일링하는 편이 속도 개선 효과가 크다.
엔진을 GPU로 옮기는 작업은 단순 실행 인자 변경으로 해결되지 않는다.

## 4. 5% 개선을 위한 우선순위

### 먼저 같은 조건으로 모델을 선택한다

학습 중 TAT 최저와 고정 checkpoint 최저는 다르다. 이번에도 episode 47의 학습 TAT는
162.5초였으나 고정 inference는 172.8초였다. 모든 checkpoint를 평가하기보다는
좋은 구간의 후보 몇 개를 같은 입력·prefix·horizon·무탐색 조건으로 평가한 뒤
**09:00–19:00 점수**로 순위를 정하는 것이 우선이다.

22만 step 후보로 `PythonCode/v9.0.0_stage2_policy.pt`를 찾았다.
내부의 Stage 2 step/저장 schedule은 **220,000**, global step은 **232,000**이다.
함께 있는 `v9.0.0_stage1_policy.pt`의 해시가 저장 당시 prefix 해시와 정확히 일치한다.
하지만 이것은 v9 TD7 모델이며 현재 v10 UD7 5-critic 모델과 구분해야 한다.
**사용자의 4.2% 실행과 같은 파일인지 아직 확인되지 않았다.**
파일명·버전을 바꿔 v10에 강제로 로드하지 않는다.

v10 Stage 2 평가에는 다음 두 파일이 필요하다. 새 실행기는 prefix 누락을 거절하며,
실제 로더는 Stage 2에 기록된 prefix SHA-256과 일치하는지도 검사한다.

```powershell
# 경로를 실제 v10 Stage 2 모델과 그 모델의 원래 prefix로 바꿔 사용한다.
.\.venv\Scripts\python.exe PythonCode\run_headless.py --mode inference --checkpoint '<Stage2 checkpoint.pt>' --stage1-policy '<matching Stage1.pt>' --end-time 45000 --episodes 1 --port 9102
```

### 재학습 전 적용 강도를 작게 비교한다

저장된 `rl_cost_lambda`가 0.5인 모델이라면 **0.45 / 0.50 / 0.55**처럼 좁은 범위가
첫 실험 후보이다. 재학습 없이 비교할 수 있으며, 필요한 평균 개선 폭은 약 1.40초다.
`--rl-cost-lambda 0.45`를 inference 명령에 추가할 수 있다. 기본값은 override하지 않는다.
이 옵션은 prefix를 포함한 전체 rollout의 비용 강도를 바꾸는 별도 실험이므로,
일반적인 checkpoint 비교에는 옵션을 생략한다. 개선 효과는 아직 측정하지 않았다.

### 평균보다 지연 명령의 원인을 좁힌다

보유한 후속 GUI 학습 DB를 참고 분석하면, 평균 이송 시간은 126.34초,
queue 10.18초, waiting 17.57초였다. 가장 느린 5%의 평균 TAT는 380.82초이며,
전체 TAT 합계의 11.28%를 차지했다. 이 5%에서만 명령당 약 27.9초를 줄여도
전체 평균 약 1.40초 감소에 해당한다.

reroute가 있었던 명령은 27.16%이고 평균 243.41초, 없었던 명령은 141.06초였다.
이는 **혼잡·반복 reroute·정지 구간을 진단할 후보**라는 뜻이다. reroute가 지연을
유발했다거나 끄면 좋아진다는 인과 결론은 아니다. 해당 표본은 4.2% inference가
아니므로, 실제 최선 모델의 결과 DB에서도 다시 확인해야 한다.

재학습을 한다면 이런 지연 구간과 09:00–19:00 부하를 Stage 2에서 충분히 경험하는지,
actor 행동이 한 값으로 몰리는지, critic 간 차이와 TD error가 함께 악화되는지를
먼저 본다. 과거 v9 기록의 `corr(TAT, route_ratio) ≈ 0`만으로 경로 최적화 여지가
없다고 결론내릴 수는 없다. 부하와 정책 반응이 함께 섞인 상관관계이기 때문이다.

## 5. 완료된 45,000초 baseline과 검증 범위

완료된 baseline 실행 폴더:

[run_0911_1023_v10.4.2_b_rl_0.0-1.0_Q_headless](C:/Users/junheemike/Documents/SKH/AICC0601_2612_반출용/oht_routing_rl/results/baseline_full/run_0911_1023_v10.4.2_b_rl_0.0-1.0_Q_headless/run.json)

45,000초를 완주했고 evaluate.py의 09~19시 점수는 **176.10184197466072초**,
175,301건이다. 실제 소요 9,871.11초(약 2시간 45분).
기존 174.4236초보다 1.67824초, 0.9622% 높다. 같은 조건으로 새 GUI baseline을
측정한 결과가 아니므로 원인을 headless로 단정하지 않는다. 동등성이 확인됐다고도
할 수 없다. 입력 전체 20시간을 완료한 결과가 아니라는 점도 구분한다.

코드 검증: 관련 테스트 **91 passed, 66 subtests**, 기존 SyntaxWarning 1개.
실제 2-episode 연결·저장 검증, 원래 evaluate.py와의 수치 일치, checkpoint 설정 복원 시
명시적 lambda 0.45 유지 및 UD7 5-critic 설정 유지도 확인했다.

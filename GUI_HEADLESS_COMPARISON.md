# GUI와 headless 비교 및 기존 UD7 모델 재평가

작성일 2026-09-11 · 원 평가 runtime v10.3.3 · 후속 검토 v10.6.3

후속 통제 실험과 최종 평가 기준은
[최신 headless·5% 검토](C:/Users/junheemike/Documents/SKH/AICC0601_2612_반출용/oht_routing_rl/HEADLESS_REVIEW_5_PERCENT.md)에 있다.
아래 164.4/168.2초는 수정 전 런타임의 역사적 첫 2,000초 지표이며, evaluate.py의 09:00–19:00 baseline
174.4236초와 직접 비교해 5% 달성을 판정할 수 없다.

## 문서 상태

이 문서는 수정 전 headless 실행의 역사 기록이다. .NET 4.8 `TargetFrameworkAttribute`
누락으로 동점 차량 정렬이 달랐던 원인은 v10.6.2에서 수정됐다. 수정 후 고정 episode54의
2,000초 GUI/headless TAT는 모두 167.7초이며, 공통 cutoff 07:32:59의 8,921개 작업과
43개 필드가 일치한다. 전체 45,000초 GUI 동등성은 별도로 검증하지 않았다.

## 결론

기록된 주요 학습 설정과 TAT 분포는 유사하다. 그러나 **GUI와 headless가 완전히 동일한 궤적을 만든다고 판정할 수는 없다.** 학습 전 워밍업에서도 차이가 있고, 에피소드 경계의 관측 수가 다르다.

사용자가 기억한 약 164초의 근거도 확인했다. 같은 SHA-256의 기존 모델을 사용한 GUI Stage 2 실행의 **첫 2,000초 고정 Stage 1 구간에서 TAT 164.4초**가 기록됐다. 이 구간은 Stage 2 학습이 시작되기 전이므로, 같은 모델의 headless 재평가와 비교할 수 있는 자료다. 단, 과거 GUI 기록과 현재 실행의 사후 비교이며 모든 설정·입력 상태를 통제한 새 A/B 실험은 아니다.

### 재평가 결과

| 고정 모델 평가 | TAT |
| --- | ---: |
| 기존 v10.0.0 모델 / 과거 GUI Stage 1 고정 구간 | **164.4초** (1999초 표본) |
| 같은 v10.0.0 모델 / 이번 headless | **168.2초** (2000초 완주) |
| 새 학습 episode 47 모델 / headless | **172.8초** (2000초 완주) |

이번 실행은 9101 포트, CUDA, action scale 1.0, 탐색·학습·replay 수집 비활성으로
완주했다. 실제 시간은 **502.65초(약 8분 23초)**였다. native DB의 완료 command
**9,030개**, 평균 TOTAL_TIME **168.201089922초**가 Python TAT와 일치하고,
DB quick_check, 원본·복사본·입력 해시, 정상 종료를 확인했다. 평가 후 9101 포트는 해제됐다.

기존 모델은 새 47회차 모델보다 **4.6초 낮지만**, 같은 모델의 GUI 기록보다
**3.8초(+2.31%) 높다**. 따라서 172.8초를 전부 “종료 checkpoint의 품질 차이”로만
설명할 수는 없다. 구현·초기화·실행 조건 차이에 대한 추가 원인 분리가 필요하다.
이후 같은 모델의 headless 진단 재실행에서 완료 명령 수와 native TAT가 정확히 재현됐다.
과거 GUI를 같은 조건으로 다시 실행한 반복 실험은 아니므로, GUI 차이의 분산을 추정한 것은 아니다.

동일 모델의 곡선도 끝에서만 차이나는 것은 아니다.

| simulation time | GUI TAT | headless native 누적 TAT | 차이 |
| --- | ---: | ---: | ---: |
| 499초 | 139.8 | 140.227 | +0.427 |
| 999초 | 157.8 | 158.532 | +0.732 |
| 1499초 | 162.7 | 165.219 | +2.519 |
| 1999초 | 164.4 | 168.201 | +3.801 |

native COMPLETED_TIME은 초 단위로 저장되므로 재구성 곡선에는 최대 한 초의 경계
모호성이 있다. 위 비교는 정밀한 패킷별 equality 검사가 아니다. 그 제한을 고려하더라도
마지막 한 틱만 비교에서 빼서 전체 차이가 해결됐다고 주장할 근거는 없다.

![GUI/headless 학습 및 고정 모델 TAT 비교](C:/Users/junheemike/Documents/SKH/AICC0601_2612_반출용/oht_routing_rl/results/gui_compare/comparison.png)

[평가 검증 JSON](C:/Users/junheemike/Documents/SKH/AICC0601_2612_반출용/oht_routing_rl/results/gui_compare/result.json) ·
[평가 CSV](C:/Users/junheemike/Documents/SKH/AICC0601_2612_반출용/oht_routing_rl/results/eval_original/run_0911_0923_v10.3.3_b_rl_0.0-1.0_Q_headless/evaluation.csv) ·
[native 결과 DB](C:/Users/junheemike/Documents/SKH/AICC0601_2612_반출용/oht_routing_rl/results/eval_original/run_0911_0923_v10.3.3_b_rl_0.0-1.0_Q_headless/best_stage1_inference.db) ·
[곡선 비교 원자료](C:/Users/junheemike/Documents/SKH/AICC0601_2612_반출용/oht_routing_rl/results/gui_compare/same_policy_curve_comparison.json)

학습 config 차이와 현재 버전 호환 검증을 포함한 checkpoint·variant 테스트는
**59개 및 40개 subtest 통과**했다. 기존 regex escape 경고 1개가 있었으며,
시뮬레이터/정책 동작은 이번 비교를 위해 변경하지 않았다.


## 1. 비교 자료와 출처

| 자료 | 성격 | 사용 범위 |
| --- | --- | --- |
| [GUI szclabdk](https://wandb.ai/offpolicy-postech/oht-routing-contextual-td7/runs/szclabdk) | v10.0.0 Stage 1 **학습** | 전체 history 11,732행, 설정, 에피소드별 마지막 표본 |
| [기존 headless 60p73gf8](https://wandb.ai/offpolicy-postech/oht-routing-contextual-td7/runs/60p73gf8) | v10.2.0 Stage 1 **학습** | 이미 끝난 1~50회차 결과 및 첫 워밍업 기록만 조회 |
| [GUI o9s1ssl5](https://wandb.ai/offpolicy-postech/oht-routing-contextual-td7/runs/o9s1ssl5) | 기존 파일을 사용한 Stage 2 실행 | 첫 에피소드의 **고정 Stage 1 구간만** 조회 |
| [47회차 평가 결과](C:/Users/junheemike/Documents/SKH/AICC0601_2612_반출용/oht_routing_rl/results/episode47/result.json) | 새 학습의 47회차 종료 모델 고정 평가 | 2,000초, TAT 172.8초 |
| [기존 파일 출처](C:/Users/junheemike/Documents/SKH/AICC0601_2612_반출용/oht_routing_rl/results/gui_compare/selection.json) | 사용자가 지정한 v10.0.0 모델 | 원본을 별도 복사하고 SHA-256 확인 |

W&B 기록은 인증된 읽기 API로 내려받았다. 기존 학습을 종료하거나 재시작하지 않았고, 100회까지 반복 모니터링도 다시 켜지 않았다.

GUI native 로그에서도 `MainFrame_PythonRun.cs`의 `RunSimulation_Cycle` 호출을 확인했다. 학습 기록의 `simPeriodTime`은 2,000초, 기존 파일의 Stage 2 기록은 45,000초다. 따라서 두 GUI 기록 모두 Python 배치 경로를 사용했다는 근거가 있다.

## 2. 설정과 학습 결과 비교

두 학습 run의 기록된 runtime config 차이는 `sim_ports`, `checkpoint_root`, `episode_summary_path` 세 항목이다. EXP_META에서는 버전·포트·실험 설명/이름이 다르다. UD7 critic 5개, UBOC, reward Q, SALE/LAP, batch 1024, replay 100000/random, seed 0, CUDA, Stage 1=2000, action curriculum 및 normalizer warmup 설정은 일치했다.

근거: [설정 차이 JSON](C:/Users/junheemike/Documents/SKH/AICC0601_2612_반출용/oht_routing_rl/results/gui_compare/config_differences.json).

같은 6~50회차 범위를 비교했다. 1~5회차는 워밍업 경계가 포함되어 제외했다.

| TAT 통계, 초 | GUI | headless |
| --- | ---: | ---: |
| 에피소드 수 | 45 | 45 |
| 최저 | 162.5 | 162.5 |
| 중앙값 | 166.9 | 166.7 |
| 평균 | 169.496 | 168.944 |
| 최고 | 205.3 | 219.8 |

GUI 값은 각 에피소드의 마지막 W&B 표본으로 대부분 simulation time 1999초다. headless 값은 종료 요약으로 2000초다. 통계가 비슷하다는 비교에는 쓸 수 있지만, 같은 시각의 terminal metric을 완전히 동일하게 측정한 자료는 아니다. 학습 후반 TAT 급등은 GUI 기록에도 있어 headless에서만 발생한 현상이라고 볼 수 없다.

GUI의 마지막 summary TAT 161.4초는 **59회차 1319초에서 중단된 값**이다. 2,000초 완료 TAT와 비교하면 안 된다. W&B run state는 finished지만 summary의 `run/status`는 interrupted이며, 마지막 학습 step은 117,320이다.

[에피소드별 GUI 표본](C:/Users/junheemike/Documents/SKH/AICC0601_2612_반출용/oht_routing_rl/results/gui_compare/gui_episodes.csv) ·
[분석 JSON](C:/Users/junheemike/Documents/SKH/AICC0601_2612_반출용/oht_routing_rl/results/gui_compare/comparison_analysis.json)

## 3. 동일 동작이 아닌 구체적인 증거

첫 워밍업 에피소드에서 같은 simulation time으로 맞춘 200개 표본을 비교했다.

- action 평균과 curriculum scale은 모두 일치했다. 학습 업데이트도 아직 없는 구간이다.
- 최초의 관측 지표 차이는 99초에서 queue/waiting 각각 1개 차이였다.
- 같은 시각 TAT의 최대 차이는 1.7초였다.
- 1999초 TAT는 GUI 169.6초, headless 170.8초였다.

학습 전에 차이가 나타났으므로 모든 차이를 “학습 노이즈 때문”이라고 설명할 수 없다. 입력/GUI 설정, 초기화 상태, 동일 시각 이벤트 처리 순서 등의 원인을 더 좁히려면 초기 command·action·상태를 함께 수집하는 대조가 필요하다. 이번 기록만으로 원인 하나를 확정하지 않았다.

에피소드 경계도 다르다. GUI의 다음 에피소드 global step 증가 패턴은 2,000이며, headless 종료 요약은 정상 에피소드당 2,001개 관측을 기록한다. 예를 들어 두 번째 에피소드의 global step 2010에서 GUI simulation time은 9초, headless는 8초다. 이 차이는 누적되어 같은 global step의 학습 샘플이 서로 다른 상태를 가리키게 한다. **같은 seed·설정만으로 같은 학습 결과를 기대하기 어려운 구체적인 차이**다. 다만 이 한 틱 차이만으로 172.8초의 원인을 전부 설명한 것은 아니다.

로컬 `db/base`의 DB 256개는 현재 모두 같은 SHA-256이며, 상위 `db/AICC_Input_260403.db`도 같은 파일 내용이었다. 과거 GUI run이 실제로 연 파일의 해시는 W&B에 남아 있지 않아, 이를 과거 입력 동일성의 확정 증거로 사용하지 않았다. [현재 입력 해시 목록](C:/Users/junheemike/Documents/SKH/AICC0601_2612_반출용/oht_routing_rl/results/gui_compare/base_input_hashes.json).

## 4. “기존 모델로 164 정도”의 확인

관련 로컬 로그를 검색해 기존 파일을 사용한 GUI run `o9s1ssl5`를 찾았다. 로드 로그의 해시가 현재 파일과 일치한다.

`3dfd5fd8a308540a065934fc9a46abf382252c01f79af8408b1e69e80eeaa0d3`

[GUI 모델 로드 로그 발췌](C:/Users/junheemike/Documents/SKH/AICC0601_2612_반출용/oht_routing_rl/results/gui_compare/gui_original_policy_load_excerpt.txt)

| GUI 첫 에피소드 simulation time | TAT, 초 | Stage 1 고정 정책 사용 | learner updates |
| --- | ---: | --- | ---: |
| 1979 | 164.3 | 예 | 0 |
| 1989 | 164.4 | 예 | 0 |
| 1999 | **164.4** | 예 | 0 |

action scale도 1.0이다. 이후 Stage 2로 전환된 구간의 지표와 섞지 않았다. 해당 run은 나중에 crashed 상태가 되었지만, 위 초기 구간의 기록과 로드 해시를 확인할 수 있다.

[GUI 고정 정책 구간 원자료](C:/Users/junheemike/Documents/SKH/AICC0601_2612_반출용/oht_routing_rl/results/gui_compare/gui_original_policy_prefix.json)

지정 파일 자체는 다음 상태다.

| 항목 | 기존 v10.0.0 파일 | 새 학습 episode 47 |
| --- | ---: | ---: |
| global step | 117,000 | 94,043 |
| learner updates | 106,898 | 83,941 |
| episode metadata | 59 | 47 |
| 저장 종류 | latest, 에피소드 중간 | episode_end |
| 저장 당시 GUI simulation time | 999초 | 해당 없음 |
| action scale | 1.0 | 1.0 |
| normalizer | 고정·저장됨 | 고정·저장됨 |

기존 파일을 “59회차 최종 모델”이라고 부르는 것은 부정확하다. GUI 학습 history의 step 117,000은 59회차 999초이고 당시 TAT는 156.4초였다. 그 값 역시 전체 2,000초 고정 모델 평가 TAT가 아니다.

## 5. 높은 TAT를 해석하는 기준

현재까지 확인된 사실은 다음과 같다.

1. 새 학습의 episode 47 training TAT 162.5초와 그 종료 모델의 inference TAT 172.8초는 다른 실험이다. 전자는 계속 갱신되는 가중치·탐색 행동의 누적 결과이고, 후자는 마지막 가중치를 처음부터 고정한 결과다.
2. 기존 v10.0.0 파일은 episode 47 모델과 가중치 및 normalizer가 다른 별도 학습 결과다. 같은 UD7 구조라고 성능까지 같지는 않다.
3. 학습 에피소드 TAT만 보고 고른 최저 모델이 고정 평가에서도 최저라는 보장은 없다. Stage 2에 넘길 모델의 성능을 판단하려면 동일 입력·horizon의 고정 평가 TAT를 함께 봐야 한다.
4. headless와 GUI 사이의 작은 상태 및 경계 차이는 실제로 관측됐다. 그러므로 모든 성능 차이를 모델 탓으로 확정하거나, 반대로 headless 엔진이 크게 잘못됐다고 확정하지 않았다.

이번 실험에서는 원래 파일을 수정하지 않고 별도 복사본을 CUDA·무탐색·학습 비활성으로 실행했다. 기존 100회 학습과 자동 후처리는 유지했다. [headless 구현 보고서](C:/Users/junheemike/Documents/SKH/AICC0601_2612_반출용/oht_routing_rl/HEADLESS_IMPLEMENTATION_REPORT.md)와 함께 보면 구현 변경 범위와 이번 수치 비교를 구분할 수 있다.

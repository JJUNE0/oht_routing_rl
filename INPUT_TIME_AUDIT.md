# 입력 DB의 실제 시간 범위와 에피소드 길이

2026-09-11 · 문서 기준 v10.6.4 (회의 기준 및 현재 headless 상태 반영)

현재 `db/base/AICC_Input_260403.db`는 **12시간이 아니라 약 20시간의 운송 명령을 담고 있다.**
암호화 파일 크기로 추정한 것이 아니라, 설치된 시뮬레이터의 라이선스 확인과
기존 `LogManager.DecryptFile` 로더를 거쳐 SQLite를 읽기 전용으로 조회했다.
원본 DB와 기존 학습 설정은 변경하지 않았다.

## 실제 조회 결과

| 구분 | 시작 | 종료 | 길이 |
|---|---|---|---|
| DB `PARAMETER`의 시뮬레이션 설정 | 2026-02-16 07:00:00 | 2026-02-17 03:00:00 | 72,000초, 20시간 |
| `COMMAND.CRT_TM` 실제 최소·최대 | 2026-02-16 07:00:00 | 2026-02-17 03:00:01 | 72,001초 |
| Python Stage 2의 실행 범위 | 2026-02-16 07:00:00 | 2026-02-16 19:30:00 | 45,000초, 12시간 30분 |
| 현재 `evaluate.py`의 평가 범위 | 2026-02-16 09:00:00 초과 | 2026-02-16 19:00:00 미만 | 10시간 |

운송 명령은 총 **347,308건**이고 서로 다른 생성 시각은 72,002개다.
07~18시에는 시간당 17,218~17,943건, 19~다음 날 02시에도
시간당 16,951~17,158건이 있다. 마지막 03:00:00~03:00:01에는 9건이다.
즉 12시간 뒤가 빈 데이터인 것도, GUI에 표시된 종료 시각만 긴 것도 아니다.
다른 시간 기반 테이블인 STOCKER_COMMAND, LIFTER_COMMAND, EQP_HISTORY는 모두 0건이다.

`CRT_TM`을 시작 이상·종료 미만으로 세면:

- 첫 12시간(07:00~19:00): **210,808건**.
- 첫 45,000초(07:00~19:30): **219,389건**.
- 19:30 이후 남은 입력: **127,919건**, 전체의 약 36.8%.

이는 입력 생성 시각 기준 집계다. 결과 DB의 `ACTIVATED_TIME`과 완료 건수는
실제 시뮬레이터 처리를 거친 값이므로 동일하다고 가정하지 않는다.
끝의 1초 초과 행은 실제 DB에 있다. 그 행을 넣은 이유는 DB만으로 알 수 없다.

입력 SHA-256: `dbe98bbf761a7f90ef924f0d552df9e2b1337a04837f9fc7f8771431bdad406d`.
이 결론은 현재 파일에 관한 것이며 과거 설명 당시 다른 파일을 썼는지는 확인하지 못했다.
전체 시간별 건수·입력 식별자는
[조회 원자료](C:/Users/junheemike/Documents/SKH/AICC0601_2612_반출용/oht_routing_rl/results/input_time_audit/input_time_coverage.json)에 있다.

## GUI가 다음 날 03시를 표시하는 이유

호출 경로는 `PARAMETER → SpecSimModel → Fab → MainFrame.SetSimTime`이다.
`ModelManager.SetSpecSimModel`은 DB의 시작·종료 시각을 Fab으로 복사하고,
GUI의 `SetSimTime`은 Fab의 두 값을 화면 시작·종료 컨트롤에 그대로 넣는다.
화면이 운송 명령 수나 학습 에피소드 길이를 계산해서 표시하는 방식이 아니다.

Python TCP 연결 후에는 별도 처리가 있다. `PServer_Python.GetEndTime`이
Python 응답의 초 수를 읽어 `engine.EndDateTime = StartDateTime + seconds`로
덮어쓴다. 이 함수는 GUI의 날짜 컨트롤을 갱신하지 않는다.
따라서 화면에 DB의 20시간 범위가 남아 있어도 엔진은 Python이 지정한
45,000초에 끝날 수 있다. Headless도 같은 TCP 협상을 사용한다.

근거: [DB→Fab](C:/Users/junheemike/Documents/SKH/AICC0601_2612_반출용/oht_routing_rl/tools/pinokio/ModelManager.decompiled.cs:1642),
[Fab→GUI](C:/Users/junheemike/Documents/SKH/AICC0601_2612_반출용/oht_routing_rl/tools/pinokio/MainFrame.decompiled.cs:5179),
[Python의 종료 시각 재설정](C:/Users/junheemike/Documents/SKH/AICC0601_2612_반출용/oht_routing_rl/tools/pinokio/PythonTCP.decompiled.cs:1713).

## 현재 설정에 대한 판단

사용자가 제공한 회의 설명으로 **12시간 평가 + 마지막 Job 완료를 위한 30분**이라는
45,000초의 의도가 확인됐다. 현재 입력 시작이 07:00이므로 해당 12시간은
07:00~19:00, 시뮬레이션 종료는 19:30이다. 기존 문서에서 추측했던
"평가 전 2시간"을 회의 의도로 간주하지 않는다. 45,000초는 유지한다.

현재 v10.6.3 fresh Stage 1은 50에피소드로 기동된 설정이며, CLI 생략 시 기본값 100과
구분한다. 학습 완료 여부는 이 문서에서 조회하지 않는다.

회의 자료의 일반적인 "24시간 원천 데이터" 설명과 현재 파일의 실측 20시간은
별개다. 제공된 현재 파일의 명령은 다음 날03시까지지만, 그 전체를 평가해야 한다는 뜻은 아니다.

기존 evaluate.py의 09~19시와 174.4236초는 **legacy 10시간 비교**로 명시한다.
새 실행 CSV에는 기존 `eval_*`와 별도로 **`meeting_eval_*`에 07~19시 12시간
TAT**를 기록한다. 새로운 시간창에 legacy baseline을 재사용하지 않는다.
12시간 개선율은 같은 시간창으로 평가한 baseline DB를 명시해야 계산한다.

완료된 동일 headless baseline을 12시간으로 다시 계산한 결과는 **176.73236276351494초**,
210,804건, 가동률 0.8001745333480891이다. 이 값은 이번 headless 측정이며
과거 GUI의 공식 baseline으로 대체하지 않는다.
[12시간 원자료](C:/Users/junheemike/Documents/SKH/AICC0601_2612_반출용/oht_routing_rl/results/input_time_audit/meeting_12h_baseline.json)

```powershell
# 프로젝트 루트에서 실행. baseline DB를 생략하면 TAT만 계산하고 개선율은 판정하지 않는다.
.\.venv\Scripts\python.exe PythonCode\simulator\evaluate.py --save_dir '<완료 결과 폴더>' --db_filename '<결과.db>' --start-time '2026-02-16 07:00:00' --end-time '2026-02-16 19:00:00' --baseline-db '<같은 조건의 완료 baseline.db>'
```

시각 경계는 기존 평가와 같은 `ACTIVATED_TIME > start`, `< end`를 유지하며,
평가 종료 뒤 완료된 Job도 발생 시각이 평가 안쪽이면 포함한다.

완료된 45,000초 headless baseline은 legacy 09~19시 구간에서 **176.10184197466072초**,
175,301건이다. 하드코딩된 기준 174.4236초보다 **1.67824초(0.9622%) 높다**.
이는 같은 조건의 GUI와 headless 비교 실험이 아니므로 차이를 headless 탓으로
확정할 수 없지만, GUI 동등성을 검증 완료했다고 할 수도 없다. 전체 20시간의
점수도 아니다. 기준값을 조용히 바꾸지 않고 두 값을 별도로 보관한다.

[완료 결과](C:/Users/junheemike/Documents/SKH/AICC0601_2612_반출용/oht_routing_rl/results/baseline_full/run_0911_1023_v10.4.2_b_rl_0.0-1.0_Q_headless/evaluation.csv)

## 반영한 변경과 검증

- 새 headless 로그에 DB 모델의 시작·종료·초 수와 TCP 협상 후 실제 실행 범위를
  따로 출력한다. 원본 모델 및 엔진의 계산은 그대로다.
- 관측 생성의 스칼라 검사만 `np.isfinite(float)`에서 `math.isfinite(float)`로
  변경했다. 실 topology와 합성 상태에서 관측 배열이 모두 일치했고 생성 시간은
  27.29→16.45ms였다. 전체 rollout 속도 개선량은 아직 측정하지 않았다.
- 관련 테스트 104개와 58개 subtest 통과. 기존 SyntaxWarning 1개.
  native 새 빌드와 실제 load-only에서 모델 길이 72,000초 로그를 확인했다.
- GPU 후보는 조사만 했다. CPU gather 제거까지 포함할 때 약 4.1ms/tick 절감,
  기존 전체 실행의 약 1.6%에 해당하는 추정치다. GPU 전체 시뮬레이터는 구현하지 않았다.

[GPU 가속 조사 보고서](C:/Users/junheemike/Documents/SKH/AICC0601_2612_반출용/oht_routing_rl/results/gpu_audit/GPU_ACCELERATION_AUDIT.md)

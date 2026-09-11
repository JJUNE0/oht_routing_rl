# Headless 최종 재검증 기록

2026-09-11 · 문서/재검증 v10.6.4 · 시뮬레이션·학습 로직 변경 없음

**새로운 120초 실행과 기존 2,000초 결과의 재대조가 통과했다.**
이번 작업에서 GUI를 새로 실행하거나 2,000초 시뮬레이션을 다시 돌린 것은 아니다.
실제 GUI에서 확보한 고정 입력·정책의 패킷과 원본 결과 DB를 기준으로 재검증했다.

디컴파일 도구와 단계별 구현 과정은
[headless 구현 및 분석 절차](HEADLESS_IMPLEMENTATION_REPORT.md),
.NET 대상 지정 누락의 원인은 [parity 수정 보고서](HEADLESS_PARITY_FIX.md)를 참조한다.

## 검증 항목과 결과

| 항목 | 이번 확인 결과 |
|---|---|
| 현재 소스의 새 production host 빌드 | GUI와 동일한 `.NETFramework,Version=v4.8` 확인 |
| 대상 지정이 없는 이전 EXE의 부정 검사 | 검증 도구가 종료 코드 1로 거절 |
| 새 120초 고정 정책 inference | 완주, 마지막 TAT 74.4초, wall time 32.15초 |
| 새 실행 ↔ 실제 GUI capture | 비교 가능한 0~119초의 120개 frame 모두 일치 |
| 새 실행 ↔ 이전 수정 headless capture | 0~120초의 121개 frame 모두 일치 |
| phase별 capture 무결성 | 읽을 수 있는 모든 payload의 SHA-256이 index와 일치 |
| 저장된 GUI/수정 headless 2,000초 DB | 기존 SHA-256 유지, quick_check 정상 |
| 공통 완료 시각의 COMMAND_LOG | 8,921개 작업 × 43개 열 전부 일치, schema도 동일 |
| 원본 GUI·엔진·모델·input·checkpoint·기존 수정 host | 이전 검증의 6개 해시와 일치 |
| 엄격 모드 관리형 모듈 디컴파일 | 4개 성공, 출력 파일도 이전 엄격 결과와 같은 SHA-256 |
| 원본 ↔ private TCP 메서드 본문 | 131개 중 기존 연결 감시 루프 한 곳만 변경 |
| Python headless 회귀 테스트 | 13개 통과 |
| 문서 재현 명령·링크 | 디컴파일 도구 컴파일 예시 통과, 문서 11개의 로컬 링크 62개 존재 확인 |

새 실행은 `results/parity_recheck/rollouts/run_0911_1825_v10.6.4_b_rl_0.0-1.0_Q_headless`다.
episode54/step108050의 고정 checkpoint를 사용했고, 학습·탐험·normalizer 갱신 없이
9101 포트에서 CUDA inference를 수행했다. 현재 Stage 1 학습의 상태를 조회하거나 변경하지 않았다.

원본 GUI capture의 마지막 120초 비용 송신 payload는 기존 파일에서 일부가 빠져 있다.
따라서 GUI와의 전체 frame 비교는 0~119초다. 새 capture와 이전 수정 headless capture는
120초까지 완전해서 121개 frame을 비교했다. 비교에서 제외하는 데이터는 매 8,912바이트
비용 송신 패킷 앞의 실제 시계 값 4바이트뿐이며, 물리 상태·경로·비용·padding을 숨기지 않았다.

## 2,000초 결과의 재확인

GUI 마지막 저장 시각인 **2026-02-16 07:32:59**로 맞추면 양쪽 모두 다음과 같다.

- 완료 작업 8,921개, 평균 `TOTAL_TIME` 167.71560721892166초.
- 재경로 횟수 합 5,099회.
- 작업 이름, 차량 배정, 경로, 거리, 완료 시각 등 43개 열에 값 차이 없음.
- 한쪽에만 존재하는 작업 0개, schema 차이 없음.

수정 headless의 전체 DB에는 종료 시 flush한 작업 95개가 더 있다.
전체 9,016개 평균은 167.67754314551908초이며 마지막 완료 시각은 07:33:19다.
이 파일 전체 평균을 GUI의 더 짧은 저장 구간과 혼동하지 않는다.
이전 전체 2,000초 실행의 마지막 packet TAT는 GUI와 수정 headless 모두 167.7초였다.

## 실제 도구 재확인

| 도구 | 설치 파일에서 확인한 버전 | 용도 |
|---|---|---|
| ICSharpCode.Decompiler / ILSpy 배포 | 7.2.0.6844, product `7.2.0.6844-104407a3` | .NET 어셈블리의 C# 표현 복원 |
| Mono.Cecil | 0.11.4.0 | IL·대상 프레임워크 메타데이터 검사 및 private TCP 패치 |
| `Framework64/v4.0.30319/csc.exe` | 4.8.9232.0 | C# host/보조 도구 컴파일 |
| Python unittest, sqlite3, hashlib | 프로젝트 `.venv` | 회귀 테스트·읽기 전용 DB 대조·해시 검사 |

`csc.exe` 경로의 `v4.0.30319`는 host의 대상 프레임워크가 4.0이라는 뜻이 아니다.
이번에 검사한 실행 파일은 4.8 대상 특성을 명시한다.
`DecompileAudit.exe`는 `ThrowOnAssemblyResolveErrors=true`와 framework 참조 탐색 경로를 사용했다.
엄격 모드 출력은 `results/parity_recheck/decompiled/`에 보존했다.

## 재현과 원자료

프로젝트 루트에서 이미 완료된 새 실행을 재대조한다. 학습이나 시뮬레이터를 시작하지 않는다.

```powershell
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe results\parity_recheck\verify_evidence.py --rollout results\parity_recheck\rollouts\run_0911_1825_v10.6.4_b_rl_0.0-1.0_Q_headless
```

아래 파일은 이번 검증의 실제 원자료다.

- [통합 검증 JSON](results/parity_recheck/revalidation.json): 범위·도구·소스 해시·결과.
- [DB 대조](results/parity_recheck/db_comparison.json): 전체/공통 구간 통계와 모든 필드 비교.
- [새 실행과 GUI wire](results/parity_recheck/fresh_gui_wire_comparison.json).
- [새 실행과 이전 수정 wire](results/parity_recheck/fresh_prior_wire_comparison.json).
- [capture index 무결성](results/parity_recheck/wire_index_integrity.json).
- [테스트 로그](results/parity_recheck/test_headless.log), [도구 버전·해시](results/parity_recheck/tool_versions.json).

`summarize.py`는 이미 수행한 native 도구 검사 결과와 저장된 증거를 종합하는 기록 도구다.
그 스크립트 단독 실행이 native 도구 검사를 다시 수행하는 것은 아니다.

## 검증 범위와 문서 역할

45,000초 전체 평가의 GUI 동등성, native C++ 소스 복구, Linux/GPU 시뮬레이터 이식은
아직 완료 범위에 포함하지 않는다. 12시간 평가와 legacy 10시간 평가의 baseline을 구분한다.
수정 전 headless 점수는 해당 런타임의 역사적 결과로 보존하며 새 평가에 그대로 합치지 않는다.

기존 문서에는 최종 결론과 역사 기록을 구분해 반영했다. 실행은 `HEADLESS.md`와
`UD7_START.md`, 시간 기준은 `INPUT_TIME_AUDIT.md`, 성능 실험 방향은
`HEADLESS_REVIEW_5_PERCENT.md`, 서버 이식은 `LINUX_SIMULATION_PLAN.md`를 기준으로 읽는다.

문서 작업은 기술 절차 정리에 **gpt-5.6-sol**, 기존 문서 일관성 정리에
**gpt-5.6-luna**를 배정했다. 주 에이전트가 실제 재검증과 최종 사실 검토를 맡았다.
실제 토큰·요금 절감량을 측정한 것은 아니므로 절감률은 제시하지 않는다.

# GUI/headless 결과 차이 원인과 수정

2026-09-11 · v10.6.2 · 원본 GUI 및 시뮬레이션 DLL은 수정하지 않음

v10.6.4에서 현재 소스의 새 120초 실행과 저장된 2,000초 DB를 다시 검증했다.
GUI와의 120개 완전 frame, 이전 수정 실행과의 121개 frame, 공통 구간의
8,921개 작업 × 43개 열이 다시 일치했다. 4개 관리형 모듈 엄격 디컴파일도 재성공했다.
이번에 새로 수행한 범위와 원자료는 [최종 재검증 기록](HEADLESS_REVALIDATION.md)에 있다.

## 확인된 원인

headless 콘솔 실행 파일에 `TargetFrameworkAttribute`가 없었다. 원본 GUI는
`.NETFramework,Version=v4.8`을 지정한다. 같은 .NET 설치와 DLL을 사용해도
진입 실행 파일의 대상 버전이 없으면 일부 프레임워크 코드가 구버전 호환 동작을 한다.
실행 설정 파일의 `supportedRuntime ... sku=v4.8`만 복사해서는 충분하지 않았다.

이 차이가 `Scheduler.GetFastsOHTNPath()`의 동점 정렬 순서를 바꿨다.
원본 비교 함수는 경로 길이가 같으면 0을 반환하며 차량 ID로 동점을 해소하지 않는다.
구버전 QuickSort와 4.8의 Introsort가 동점 원소를 다른 순서로 반환했고,
유휴 차량을 다른 Bay로 옮기는 선택이 달라졌다.

Microsoft의 [호환성 설명](https://devblogs.microsoft.com/dotnet/introduction-to-net-framework-compatibility/)은
진입 어셈블리의 대상 버전으로 호환 동작을 결정함을 설명한다.
[ArraySortHelper 원본](https://github.com/microsoft/referencesource/blob/main/mscorlib/system/collections/generic/arraysorthelper.cs)의
`BinaryCompatibility.TargetsAtLeast_Desktop_V4_5` 분기는 이를 정렬에 적용한다.

### 실제 최초 분기

- 같은 input DB, 고정 episode54/step108050 체크포인트, CUDA 무노이즈 inference.
- 0~28초: 레일·작업·차량 수신 패킷과 Python 비용 응답이 모두 일치.
- 28.3994059321392초: `RailLineNode/A1_Line_5196_5145/OHT_CHECK_OUT` 이벤트에서
  유휴 차량의 Bay 재배치 선택이 갈림.
- 29초: GUI는 OHT 53137을 목적 레일 2516으로, 기존 headless는 OHT 53135를 그곳으로 보냄.
  나머지 차량은 목적 레일 2497 유지. 이 레일 ID는 Python wire 매핑 기준이다.
- 29초: 변경된 입력을 받은 정책 비용 응답에도 첫 차이 발생. 작업 패킷의 첫 차이는 63초.
- 해당 사건은 재경로 탐색의 병렬 실행 경쟁 때문이라는 초기 가설로 설명할 필요가 없다.
  프레임워크 대상 지정 한 가지로 관측된 차이가 사라졌다.

`results/parity_fix/SortCompatibility.cs`는 원본과 같은 동점 비교 함수를 사용한 최소 재현이다.
대상 지정 없음: `53137,53135`; 4.8 지정: `53135,53137` 순서를 실제 확인했다.
이 작은 예제의 초기 배열은 정렬 차이를 설명하기 위한 것이며 전체 native 후보 배열을 덤프한 것은 아니다.

## 적용한 변경과 검증

- `tools/pinokio/HeadlessProgram.cs`: 원본 GUI와 같은 4.8 대상 어셈블리 특성 추가.
- 실행 시 유효 대상 버전을 로그에 남기고, 기대 버전이 아니면 즉시 거절.
- `tools/pinokio/VerifyFramework.cs`, `build.ps1`: 원본 GUI와 새 EXE의 대상 메타데이터 비교.
  기존 targetless EXE 거절, 수정 EXE 통과를 확인.
- 원본 `Simulation.Model.dll`, `Simulation.Engine.dll`, C++ 경로·배차 DLL은 그대로 사용.
  경로 비용, 보상, 정책 가중치, 동점 비교 함수를 수정한 것이 아니다.
- 관련 Python headless 테스트 13개 통과.

| 비교 | 수정 전 | 대상 버전 수정 후 |
|---|---|---|
| GUI와 raw wire 일치 | 0~28초 | 0~119초의 120개 완전 frame 모두 일치 |
| 120초 마지막 TAT | headless 75.0초 | headless 74.4초 |
| episode54 2,000초 | GUI 167.7초, headless 163.0초 | GUI/headless 모두 167.7초 |

수정된 production host의 2,000초 실행이 497.34초에 완주했다.
완료 시각을 GUI 마지막 저장 시각인 **07:32:59**로 맞추면 양쪽 모두
**8,921개 작업**, 평균 TAT **167.71560721892166초**, 재경로 횟수 합 **5,099**다.
작업 이름으로 대조한 `COMMAND_LOG` **43개 열이 모두 정확히 일치**한다.
누락·추가 작업 0개, 값이 다른 작업 0개다. 차량 배정, 이동 경로, 시간 필드도 포함한다.
근거는 `results/parity_fix/final_comparison.json`과 재현 코드 `compare_final.py`다.

headless는 종료 시 잔여 완료 작업까지 저장하므로 전체 DB는 9,016개 작업,
평균 167.67754314551908초, 마지막 완료 07:33:19다. GUI보다 95개가 더 저장된
차이는 기존에 확인한 마지막 flush 동작이다. 같은 완료 구간의 결과는 완전히 같다.
따라서 이 checkpoint/input의 2,000초에서 관측된 GUI/headless 결과 차이는 해결됐다.

원본 GUI capture는 0~120초 121개 frame의 index가 있다. 마지막 송신 payload 일부가
종료 전 버퍼에 남아 120초 frame 전체의 바이트 비교에서는 제외했다.
비교 시 매 8,912바이트 비용 패킷의 첫 4바이트 실제 시계 값만 제외한다.
물리 상태·비용·padding을 그 외에 임의로 제외하지 않았다.

근거: `results/parity_fix/framework_comparison.json`, `wire_comparison.json`,
`oht_differences.json`, `events_runs/`, `framework_runs/`, `production_fixed/`.
수정된 빌드는 새로운 headless 실행부터 사용된다. 이미 시작된 실행의 runtime 복사본은
그대로이므로 실행 중인 학습의 과거 결과를 수정된 GUI 호환 결과로 소급 해석하지 않는다.
2,000초 일치와 12시간 최종 평가의 일치는 별개의 검증 범위다.

## 디컴파일 실패 여부

참조 누락을 숨기던 기존 도구 설정을 피하고, framework 참조 경로를 지정한 엄격 모드로
설치된 `Simulation.Model.dll`, `Simulation.Engine.dll`, `Pinokio.Simulator.exe`,
private `Pinokio.TCP.IP.dll` 전체를 새로 디컴파일했다. 이 네 결과에서
실패/미지원 코드 표시는 발견되지 않았다. 결과는 `results/parity_fix/*.full.cs`에 있다.
처음에는 `System.IO.Compression` 참조 해석 실패가 있었고, 참조 경로 추가 후 성공했다.

headless는 디컴파일한 전체 엔진을 다시 컴파일한 구현이 아니다. 기존 DLL을 호출하는
별도 host이므로, 이 문제는 디컴파일에서 이동 로직을 잃은 문제가 아니라 host의 대상 지정 누락이었다.
TCP 131개 메서드 본문 비교에서도 변경은 연결 감시 루프의 `Sleep(5)` 한 곳뿐이다.

`DispatchCpp.dll`, `Shortestpath.dll`은 C++ 바이너리다. 위 C# 디컴파일 성공 범위에
포함되지 않으며, C++ 전체 소스까지 복구·검증했다고 주장하지 않는다.

## Dijkstra와 A*

현재 기본 경로는 **Dijkstra**다. `PathFinder.FindPath()`는 `IsUseAstar`가 참인
특정 조건에서만 A*를 호출하며, 이번 기본 설정은 거짓이다. A* 네트워크를 초기화하는
코드가 있다는 사실은 A*를 기본 알고리즘으로 사용한다는 뜻이 아니다.
유휴 차량 선택에서 문제를 찾은 `GetFastsOHTNPath()`도 Dijkstra 경로 길이를 사용한다.

Linux 서버에서 여러 파라미터를 평가하려는 목적의 구체적인 의존성과 이식 순서는
[Linux/GPU 병렬 실행 검토](LINUX_SIMULATION_PLAN.md)에 정리한다.

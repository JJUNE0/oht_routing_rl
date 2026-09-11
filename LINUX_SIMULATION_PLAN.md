# Linux 서버에서 다수 파라미터를 평가하기 위한 이식 검토

2026-09-11 · 문서 기준 v10.6.4 · 현재는 Windows headless 구현이며 Linux 실행은 미구현

목적은 서버에서 수십 개 독립 파라미터 조합을 동시에 평가하는 것이다.
한 시뮬레이션의 이벤트 전체를 GPU로 옮기는 것과, 독립 시뮬레이션 여러 개의 처리량을
늘리는 것은 별개의 작업이다. 우선 목표는 **독립 CPU 시뮬레이션 worker + GPU 정책 batch**다.

## 확인한 이식 장애물

| 구성 | 현재 코드에서 확인한 내용 | Linux에 필요한 작업 |
|---|---|---|
| 콘솔 host | .NET Framework 4.8, `kernel32.SetDllDirectory`, 원본 DLL 참조 | 지원되는 cross-platform .NET 프로젝트로 분리; 네이티브 로더 교체 |
| 엔진/모델 | `Simulation.Model`에 `System.Windows.Forms` 의존; 기존 GUI·모델 패키지 결합 | UI·렌더링·메시지박스 의존 제거, 모델 로드와 이벤트 코어만 라이브러리로 빌드 |
| 경로 계산 | `Shortestpath.dll`의 Cdecl 함수 호출 | 동일 ABI의 Linux `.so` 빌드 또는 동등한 구현과 경로 비교 |
| 배차 가속 | `DispatchCpp.dll`의 Cdecl 함수 호출 | Linux `.so` 빌드, 입력/출력 배열·메모리 해제 계약 보존 |
| 입력/결과 DB | 원본 복호화 및 SQLite 계층 사용 | DB·복호화·native SQLite를 Linux에서 검증 |
| 라이선스 | host가 설치된 엔진의 라이선스 검사를 실행 | 서버에서 사용할 정식 라이선스 및 Linux 지원 경로 확인 |
| Python 연결 | 현재 host와 Python server 모두 127.0.0.1 기반 | 같은 서버 내 worker별 포트, 또는 명시적인 원격 coordinator 프로토콜 |

Microsoft의 [이식 지침](https://learn.microsoft.com/en-us/dotnet/core/porting/framework-overview)은
Windows 전용 API 의존을 제거해야 다른 OS에서 실행할 수 있음을 설명한다.
[네이티브 라이브러리 로딩](https://learn.microsoft.com/en-us/dotnet/standard/native-interop/native-library-loading)도
OS별 라이브러리를 전제로 한다. Windows DLL 파일명을 `.so`로 바꾸는 작업으로 해결되지 않는다.

현재 workspace에서 빌드 가능한 `AMHS.OSS.Logic` C# 프로젝트는 찾았지만,
시뮬레이터 코어 전체와 두 C++ 라이브러리의 원본 소스는 찾지 못했다.
엄격 C# 디컴파일은 성공했으나, 그것만으로 Linux에서 링크·실행 가능한 프로젝트나
C++ 소스가 확보된 것은 아니다. 지금 Linux 서버로 복사해서 실행할 수 있다고 안내하면 부정확하다.

## 권장 구현 순서

1. 이번에 수정한 Windows GUI/headless 4.8 실행을 기준 구현으로 고정한다.
   input·checkpoint·native DLL 해시와 실제 wire capture를 비교 fixture로 보존한다.
2. GUI에서 분리된 모델 로더/이벤트 코어를 cross-platform .NET으로 빌드하고,
   C++ 경로·배차 함수를 Linux에서 제공한다. 라이선스·DB 경로도 함께 검증한다.
3. 같은 seed/입력/action으로 짧은 wire 비교 → 2,000초 command 단위 비교 →
   45,000초와 동일 평가 구간 TAT·완료 건수 비교 순서로 검증한다.
   현재 .NET 4.8의 동점 정렬과 event 동시 시각 처리 순서를 명시적으로 보존한다.
   플랫폼 기본 `Sort`나 GPU 최단경로 함수로 단순 교체하면 같은 최단거리여도 경로가 바뀔 수 있다.
4. worker마다 native 상태·출력·seed·checkpoint를 독립 소유하고, CPU/RAM 한도 내에서
   여러 프로세스를 실행한다. GPU owner는 정책 요청을 작은 시간 제한으로 묶어 처리한다.
   파라미터마다 모델이 다르면 같은 모델끼리 batch하거나 별도 모델 인스턴스를 관리한다.
5. wall time과 동시 처리량을 프로파일한 뒤 병목 커널만 GPU로 이식한다.
   현재 `QueueDijkstra`는 C++ 호출에 threadCount=8을 고정 전달한다.
   32 worker라면 해당 구간에 최대 256개 worker thread 요청이 겹칠 수 있어,
   실제 코어 수에 맞춘 worker 수와 내부 thread 수 조정이 먼저다.

현재 `run_headless.py`는 두 개 이상의 독립 실행에서 다른 포트·결과 폴더를 사용할 수 있다.
이는 Windows에서의 실행 기능이다. 여러 worker를 단일 GPU owner에 연결하는 서버 구조와
Linux 시뮬레이터 이식이 이 문서 작성으로 구현·검증된 것은 아니다.
기존 `runtime/distributed/` 코드는 재사용 후보이며, native rollout 여러 개를 연결한 처리량과
정책 동일성을 검증한 뒤 실제 sweep 실행 경로에 통합해야 한다.

## GPU 후보의 범위

이전 [GPU 전처리 벤치마크](results/gpu_audit/GPU_ACCELERATION_AUDIT.md)는
합성 관측의 CPU/GPU gather 비교였다. 약 4.1ms/tick의 부분 절감 후보를 확인했지만
전체 simulator 가속이나 Linux 포팅을 입증하지 않았다.
학습/inference tensor 연산은 이미 CUDA를 사용할 수 있다.
독립 rollout batch는 GPU 활용률을 높일 수 있으나, 이벤트 시간축 내부의 의존성을 없애지 않는다.
TAT를 낮추는 정책 품질 개선과, 같은 평가를 더 빨리 반복하는 실행 속도 개선은 별도로 측정한다.

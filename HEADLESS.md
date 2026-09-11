# Pinokio GUI 없는 실행

현재 설치된 시뮬레이터의 .NET 엔진·모델 DLL을 직접 호출하는 콘솔 실행기를
추가했습니다. Python TCP/IP on, 입력 DB 로드, 결과 폴더·이름 지정,
실행과 에피소드 반복을 명령으로 처리합니다. GUI 창이나 자동 클릭은 없습니다.
기존 `start_ud7_stage1.cmd`와 `start_ud7_stage2_best.cmd`도 headless 실행으로 연결했습니다.

## 새 checkout 준비

Windows x64와 .NET Framework 4.8, 정상 라이선스가 있는 원본 시뮬레이터가
필요합니다. 저장소와 같은 상위 폴더의 `Simulator/`에 원본 EXE, config,
managed/native DLL을 포함한 설치 파일 전체를 둡니다. 기본 입력 위치는
`../db/base/AICC_Input_260403.db`이며 `--input`으로 다른 export DB를 지정할 수 있습니다.
이 저장소는 원본 시뮬레이터, DB, 체크포인트 및 디컴파일한 vendor 소스를 배포하지 않습니다.

저장소 루트에 `.venv`를 만든 뒤 `PythonCode/requirements.txt`를 설치하고,
학습에는 환경에 맞는 CUDA PyTorch를 설치합니다. 검증 호스트의 패키지와
설치 명령은 [requirements-gpu.txt](PythonCode/requirements-gpu.txt)에 있습니다.
`run_headless.py`는 실행 시 `tools/pinokio/build.ps1`로 host를 빌드하며,
처음에는 SHA-256으로 고정한 ILSpy 의존성을 내려받습니다.
실행 전 `start_ud7_stage1_headless.cmd --check --port 9120`으로 인자를 확인할 수 있습니다.
보고서의 `results/` 경로는 검증 당시 로컬 증거를 가리키며 Git에 포함되지 않습니다.

GUI 호출과의 대응, 변경한 부분, 실제 검증 결과와 미검증 범위는
[headless 구현·동등성 검토 보고서](HEADLESS_IMPLEMENTATION_REPORT.md)에 정리했습니다.
현재 evaluate.py의 09:00–19:00 TAT 기준, GUI 차이 진단, 파일 재사용 및 5% 개선 실험 방향은
[최신 검토 보고서](C:/Users/junheemike/Documents/SKH/AICC0601_2612_반출용/oht_routing_rl/HEADLESS_REVIEW_5_PERCENT.md)에 있습니다.
회의 기준은 **12시간 평가 + 30분 완료 여유**로 확인됐습니다. 새 CSV의
`meeting_eval_*`는 07~19시 12시간 점수이며, 기존 `eval_*`는 legacy 09~19시
10시간 점수입니다. 174.4236초 baseline은 legacy 구간에만 적용합니다.
12시간 비교 명령은 [시간 범위 보고서](C:/Users/junheemike/Documents/SKH/AICC0601_2612_반출용/oht_routing_rl/INPUT_TIME_AUDIT.md)에 있습니다.

프로젝트 폴더에서 PowerShell로 실행합니다:

```powershell
cd "C:\Users\junheemike\Documents\SKH\AICC0601_2612_반출용\oht_routing_rl"

# 새 Stage 1 학습: 현재 fresh 실행은 50에피소드, CUDA (CLI 기본값은 100)
.\start_ud7_stage1_headless.cmd --episodes 50 --port 9100

# 위 실행이 끝난 뒤, 새 Stage 1의 lowest TAT 모델을 무노이즈 inference
.\.venv\Scripts\python.exe PythonCode\run_headless.py --mode inference --episodes 1 --port 9100

# 새 Stage 1 lowest TAT를 고정한 뒤 Stage 2 학습
.\start_ud7_stage2_headless.cmd --episodes 100 --port 9100
```

Stage 1은 매번 **처음부터** 시작합니다. 에피소드 종료마다 체크포인트를 저장하며,
학습 후 온전한 에피소드의 마지막 TAT가 최저이면 `best_tat/checkpoint.pt`를 갱신합니다.
warmup, 조기 종료, normalizer 미고정 모델은 best가 되지 않습니다.
Stage 2는 새 Stage 1 실행을 가리키는 `ud7_stage1_current.json`의 best를 복사·검증하며,
에피소드 첫 2,000틱을 frozen Stage 1 inference로 실행합니다.
Stage 1=2,000초, Stage 2=45,000초 계약을 유지합니다.
입력 DB 자체는 07:00~다음 날 03:00의 **20시간 데이터**입니다. Stage 2는
그중 07:00~당일 19:30까지 실행합니다. GUI의 입력 기간 표시와 Python이
협상한 실행 기간이 다른 이유와 직접 조회한 건수는
[시간 범위 보고서](C:/Users/junheemike/Documents/SKH/AICC0601_2612_반출용/oht_routing_rl/INPUT_TIME_AUDIT.md)에 있습니다.
새 native 로그는 입력 기간과 실제 실행 기간을 따로 출력합니다.
체크포인트 inference는 저장된 action 강도를 고정해 사용하며 평가 중 curriculum을 진행하지 않습니다.
현재 pointer는 가장 최근 시작한 새 Stage 1입니다. 테스트 직후처럼 warmup만
진행한 경우 best가 없어 inference/Stage 2를 시작하지 않습니다.

Stage 1 기본 입력은 다음 파일입니다:

`C:\Users\junheemike\Documents\SKH\AICC0601_2612_반출용\db\base\AICC_Input_260403.db`

```powershell
# input 폴더의 모든 .db를 정렬해서 차례로 실행; episodes가 더 많으면 순환
.\start_ud7_stage1_headless.cmd --input-dir "..\db\base" --episodes 100 --port 9120 --output-dir "results\headless" --name "ud7_{input}_{episode:04d}"

# 특정 Stage 1 체크포인트로 평가
.\.venv\Scripts\python.exe PythonCode\run_headless.py --mode inference --checkpoint "checkpoints\선택한실행\best_tat\checkpoint.pt" --end-time 2000 --episodes 1 --port 9120

# Python baseline 정책을 연결해 짧은 동작 확인
.\.venv\Scripts\python.exe PythonCode\run_headless.py --mode baseline --end-time 120 --episodes 2 --port 9120

# Python 없이 시뮬레이터 자체 경로 정책만 실행
.\.venv\Scripts\python.exe PythonCode\run_headless.py --mode simulator --end-time 120 --episodes 1

# 경로·포트·인자 검사만 수행 (학습, 시뮬레이터, 새 실행 폴더 생성 안 함)
.\start_ud7_stage1_headless.cmd --check --port 9120
```

`--input`으로 파일 하나를 지정할 수도 있습니다. 입력은 시뮬레이터에서 export한
암호화 DB여야 합니다. `--name`은 `{input}`과 `{episode:04d}`를 지원하고,
중복 결과명은 거절합니다. 사용 중인 포트도 실행 전에 거절합니다.
`--no-wandb`는 W&B를 끄며, `--device cpu`는 inference 장치를 선택합니다.
Stage 2 checkpoint 평가에는 `--stage1-policy`로 학습 당시의 prefix를 함께 지정합니다.
이 경우 기본 horizon은 45,000초이며, prefix가 빠진 Stage 2 평가는 실행 전에 거절합니다.
`--rl-cost-lambda 0.45`처럼 비용 강도를 바꾸는 inference 실험도 가능합니다.
지정하지 않으면 checkpoint의 값을 유지하며, 지정하면 prefix를 포함한 전체 실행에 적용합니다.
학습 preset은 CUDA를 사용합니다. `--timeout`은 에피소드당 최대 실제 초이며 기본 21,600초입니다.
Ctrl+C는 이 실행기가 시작한 자식 프로세스만 종료하고, 완료된 결과를 남깁니다.

각 호출은 `--output-dir` 아래 자동 run 이름의 새 폴더를 만듭니다.

- `<결과명>.db`: native 시뮬레이터 결과. 완료 command와 마지막 집계 테이블 저장 포함.
- `<결과명>.log`, `python.log`: 시뮬레이터와 Python 로그.
- `evaluation.csv`: 입력·결과 DB·실행 시간·에피소드 TAT·완주/조기 종료 여부,
  evaluate.py의 최종 구간 TAT·개선율·명령 coverage.
- `episodes.jsonl`: Python이 종료 신호를 받은 뒤 기록한 지표. 체크포인트 저장 후 기록됩니다.
- `run.json`: EXP_META, 실행 인자, 완료 에피소드 수, complete/failed/interrupted 상태.
- `runtime/`: 실행 시작에 한 번 준비하고 모든 에피소드가 재사용하는 실행 파일과 native 로그.
- `policy.pt`, `prefix.pt`: inference에서 한 번 고정한 모델. run.json에 원본 경로와 SHA-256 기록.

CSV의 TAT는 종료 직전 마지막 active packet의 `TotalTat`입니다.
종료 신호 자체에는 최종 관측이 없습니다. `full_horizon`과 `early_termination`을
함께 확인해야 하며, 학습 중 lowest TAT가 무노이즈 평가에서도 최저임을 보장하지 않습니다.

v10.7.0부터 W&B에는 다음 두 지표도 기록합니다.

- `eval/final_tat`: 정상 완주한 에피소드의 마지막 active packet TAT(초).
  종료 때 한 번만 기록하며, 중간 tick에는 값을 채우지 않습니다.
  이 값은 위의 에피소드 TAT이며 `evaluate.py`의 12시간 구간 점수와 구분합니다.
- `runtime/acceleration_rate`: 에피소드 첫 active packet 이후 진행한 시뮬레이션
  초를 실제 경과 초로 나눈 배수입니다. `10`이면 실제 1초 동안 시뮬레이션이
  10초 진행했다는 뜻입니다. 기존 W&B 주기로 기록하고 에피소드마다 초기화합니다.
  GUI 연결·DB 로딩 대기는 제외하고 실행 중 Python 연산·TCP 대기는 포함합니다.

이미 실행 중인 Python에는 코드가 자동 반영되지 않으며, 새 실행부터 적용됩니다.
CSV export는 terminal 지표가 없는 중간 행도 보존하고 해당 열을 빈칸으로 둡니다.

최종 목표 비교에는 `eval_tat_s`를 사용합니다. baseline은 174.4236초, 5% 목표는
165.70242초이며, Stage 1처럼 평가 구간 이전에 끝난 실행에는 최종 달성 판정이 없습니다.
입력 DB는 하나만 지정해도 `--episodes` 횟수만큼 반복합니다. 에피소드마다 Python은
유지하고 native 프로세스만 초기화합니다. 실행 파일·모델을 매 에피소드 복사하지 않습니다.
임시 복호화 DB는 성공 후 지웁니다. 디버깅 시 `--keep-work`로 보관합니다.
에러가 기록됐거나 DB 무결성 검사/종료 요약이 실패하면 다음 에피소드로 넘어가지 않습니다.

## 현재 학습 종료 후 자동 평가

현재 사용자 지시로 기동한 50에피소드 학습에는 다음 후처리 실행기를 연결합니다.
`results/ud7_stage1_goal.json`에 기록된 프로세스의 PID와 시작 시각을 확인하고
학습 종료를 기다리므로, 새 학습을 시작하거나 기존 학습을 재시작하지 않습니다.

```powershell
.\.venv\Scripts\python.exe PythonCode\finalize_ud7_stage1.py --goal-state results\ud7_stage1_goal.json
```

학습 완료 후 모든 에피소드의 체크포인트 해시·TAT·시뮬레이터 결과를 대조하고,
최저 유효 TAT 모델을 복사해 고정한 뒤 같은 입력 DB로 2,000초 inference를 실행합니다.
중복 실행은 잠금으로 거절합니다. 실행 상태는 학습 결과 폴더의 `finalizer.json`에,
최종 모델·순위·학습 TAT와 평가 TAT 비교는 `final_stage1/`에 저장합니다.
이미 완료된 학습만 검증하려면 `--audit-only`를 추가합니다.
native 평가 결과는 경로 길이 제한을 피하도록 `results/eval_<학습경로해시>/`에
저장하고, 최종 보고서와 `result.json`에서 이 경로를 연결합니다.
설치된 .NET 실행기는 긴 실행 파일/설정 경로에서 시작하지 못할 수 있습니다.
260자에 도달하는 입력·결과·실행 설정 경로는 Python 시작 전에 거절하므로,
오류가 표시되면 `--output-dir`과 결과 이름을 짧게 지정하세요.

## 빌드와 성능 범위

실행기는 `tools/pinokio/build.ps1`로 현재 소스를 빌드합니다.
.NET Framework 4.8의 csc를 사용하므로 별도 .NET SDK는 필요 없습니다.
첫 빌드는 공식 ILSpy 7.2 배포에서 Mono.Cecil을 받아 사용하며 archive SHA-256을 검사합니다.
분석 도구와 빌드 산출물은 Git에서 제외되어 있습니다.

`HeadlessProgram.cs`와 `HeadlessLoader.cs`는 설치된 GUI의 모델 초기화 순서와
이벤트 실행 경로를 따라 기존 엔진을 호출합니다. 기본 Euclidean/penalty-off,
AI2.0/rerouting, C++ 배차 가속 설정을 사용합니다. `IsUseAccelearionVer=true`는
배차 C++ 가속 경로를 뜻하며 물리 시뮬레이션 속도를 바꾸는 설정이 아닙니다. AI2.0은 기존 시뮬레이터의 내부 모드 이름이며,
Python batch가 자동으로 설정합니다. Rerouting/Route Selection 스위치는 이 배포 GUI에서
숨겨져 있으므로 사용자가 수동으로 확인할 옵션이 아닙니다. GUI에서 별도로 바꾼 임의 설정을
자동으로 읽어오는 기능은 없습니다. 설치된 엔진의 라이선스 검사도 그대로 실행됩니다.
전체 GUI/엔진 소스가 없어 별도 host를 추가한 형태이며, 원래 EXE에 인자를 추가한 것은 아닙니다.

이전 v10.6.1 episode54 비교에서는 같은 2,000초 고정 정책에도 GUI 167.7초,
headless 163.0초가 나왔습니다. 이는 수정 전 런타임의 역사 수치입니다.
v10.6.2에서 원인을 확인했습니다. headless EXE에 GUI와 같은 .NET 4.8 대상 지정이
없어 동점 경로 길이의 차량 정렬 순서가 달랐습니다. 대상 지정과 빌드/실행 검사를
추가했으며, 수정 후 GUI의 0~119초 raw 패킷과 정책 응답이 모두 일치했습니다.
2,000초 TAT도 167.7초로 같고, 같은 완료 시각의 8,921개 작업은 43개 필드 전부
일치했습니다. 검증 범위와 근거는 [수정 보고서](HEADLESS_PARITY_FIX.md)와
[최종 재검증 기록](HEADLESS_REVALIDATION.md)에 있습니다.
이미 시작된 실행의 runtime은 변경하지 않으므로 새 실행부터 적용됩니다. 현재 v10.6.3
fresh 50-episode 학습은 `results/headless/run_0911_1740_v10.6.3_b_rl_0.0-1.0_Q_headless`에서,
episode별 best 후보는 `checkpoints/run_0911_1740_v10.6.3_b_rl_0.0-1.0_Q_episodebest`에서 관리합니다.
W&B는 [vp1o1994](https://wandb.ai/offpolicy-postech/oht-routing-contextual-td7/runs/vp1o1994)입니다.
학습과 best-TAT inference finalizer 연결은 기동된 설정이며 학습 완료 여부는 조회하지 않습니다.

`Pinokio.TCP.IP.dll`의 연결 감시 루프가 연결 중 계속 busy-spin하는 것을 확인하여,
**headless 전용 복사본**에만 5ms 대기를 추가했습니다. 시뮬레이션 틱마다 지연시키는
코드는 아닙니다. 원본 DLL의 SHA-256이 달라지면 패치를 거절하므로 업데이트 시 재검토해야 합니다.
원본 GUI EXE와 DLL은 변경하지 않습니다.

시뮬레이터 이벤트 계산은 여전히 CPU입니다. RL 학습/inference는 CUDA를 사용할 수 있지만,
이벤트 엔진 자체를 GPU로 옮기는 기능이나 다중 시뮬레이터 병렬 학습은 이 실행기에 없습니다.
Linux 서버에서 다수 파라미터를 실행할 목적의 의존성 조사와 구현 순서는
[Linux/GPU 병렬 실행 검토](LINUX_SIMULATION_PLAN.md)에 정리했습니다.

독립 실행 두 개는 가능합니다. 각 실행에 서로 다른 `--port`와 결과 폴더를 지정하면
학습과 inference를 동시에 실행할 수 있습니다. 실제로 학습 9100 + 평가 9101 연결을
확인했습니다. 각 실행은 자기 Python 서버와 native 프로세스를 사용하며, 하나의
learner에 여러 시뮬레이터를 합치는 방식은 아닙니다. CPU/GPU는 공유하므로 개별
실행 시간은 늘어날 수 있습니다.

GPU 사용만 켜서 모든 계산이 빨라졌다고 해석하면 안 됩니다. GUI 실행과 동일 조건의
장시간 수치 비교는 별도로 필요합니다.

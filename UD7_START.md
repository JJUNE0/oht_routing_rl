# UD7 Stage 1 재학습 실행

**GUI 없이 실행할 수 있습니다.** `start_ud7_stage1_headless.cmd`를 실행하면
Python 서버, 입력 DB 로드, TCP 연결, 시뮬레이션, 결과 저장, 다음 에피소드까지
자동 진행합니다. CLI 기본값은 100에피소드이며, 현재 사용자 지시로 기동한 fresh
Stage 1은 `--episodes 50`으로 설정되어 있습니다.
포트·파일·이름·inference 명령은 [HEADLESS.md](HEADLESS.md)를 참고하세요.
기존 `start_ud7_stage1.cmd`와 `start_ud7_stage2_best.cmd`도 이제 같은 headless
실행기를 사용합니다. 아래는 GUI를 직접 사용할 때의 Python 단독 실행 방법입니다.

1. 기존 Python 서버를 종료하고 Pinokio 시뮬레이션을 초기화합니다.
2. `.venv\Scripts\python.exe PythonCode\run_ud7_stage1.py`를 실행합니다.
3. `waiting on 127.0.0.1:9100`이 표시되면 Pinokio의 Python TCP/IP를 같은 포트로 연결합니다.
4. Python TCP/IP 탭의 **Open Files**에서 입력 DB가 들어 있는 **폴더**를 선택하고
   결과 폴더와 새 이름을 지정합니다. 이 경로에서 Python batch가 자동으로 시작되므로
   일반 Run 버튼을 따로 누르지 않습니다. 자세한 GUI 절차는
   [episode54 안내](results/episode54/README.md)를 참고하세요.

95,000 모델이나 이전 normalizer를 불러오지 않는 새 학습입니다.
UD7 critic 5개, reward Q, SALE/LAP, random replay 100,000,
초기 warmup 10,000스텝, stage 1 에피소드 길이 2,000을 사용합니다.

매 실행마다 새 체크포인트 폴더를 생성하고 `ud7_stage1_current.json`에
현재 폴더 위치를 남깁니다. 각 에피소드 종료 시 다음 파일을 저장합니다.

- `episodes/episode_XXXXXX_step_XXXXXXXXX.pt`: 해당 종료 시점 모델
- 같은 이름의 `.json`: 에피소드, 스텝, TAT, 종료 시각, 최저 후보 여부, SHA-256
- `best_tat/checkpoint.pt`: 후보 중 TAT가 가장 낮은 모델의 동일 복사본
- `best_tat/selection.json`: 최저 모델의 선택 근거

기존 latest/periodic 저장도 유지합니다. warmup 및 조기 종료 에피소드는
보관하지만 best 후보에서는 제외합니다. 완전한 에피소드 전체가 warmup 이후여야
하며 normalizer가 채워져 고정되고 learner 업데이트가 있어야 합니다.
동일 TAT는 먼저 저장된 best를 유지합니다.

프로토콜의 종료 신호에는 새 TAT가 없으므로 **종료 직전 마지막 active packet의
누적 TotalTat**를 사용합니다. 첫 실행은 SimTime=2000일 수 있고 후속 episode는
틱 수 차이가 있을 수 있으므로 `sim_time`과 `full_horizon`을 함께 확인합니다.
이는 학습 중 측정치이며 별도 무노이즈 inference 성능을 보장하지 않습니다.
정상 종료는 마지막 SimTime이 종료 시간-1 이상이고 강제 종료 플래그가 없는 경우입니다.

Stage 1 학습을 충분히 진행한 뒤 Python 서버를 종료하고 시뮬레이터를 초기화한 다음
`.venv\Scripts\python.exe PythonCode\run_ud7_best_stage2.py`를 실행합니다. 최신 Stage 1 실행의 best 모델을 자동으로
선택하고 실행별로 복사·SHA-256 검증합니다. best가 없으면 실행하지 않습니다.
첫 2,000스텝의 frozen inference 후 stage 2 학습으로 이어집니다.

설정만 확인하려면 프로젝트 폴더의 PowerShell에서:

```powershell
& ".\.venv\Scripts\python.exe" ".\PythonCode\run_ud7_stage1.py" --check
```

이 확인 명령은 새 학습 폴더나 현재 실행 포인터를 만들지 않습니다.

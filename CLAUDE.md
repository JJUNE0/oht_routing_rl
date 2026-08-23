## 실험 관리 규칙

모든 실험 전 `EXP_META` 먼저 채울 것. run 이름·wandb config는 여기서 자동 생성.

```python
EXP_META = {
    'version':        CONTEXTUAL_VERSION,  # oht_routing/version.py의 단일 버전
    'cost_structure': 'b_rl',       # 'b_rl' | 'residual' | 'direct'
    'action_range':   '0.5-1.5',    # 문자열로 명시
    'reward_version': 'N',          # 현재 고정 reward profile
    'centering':      False,
    'note':           'diag',       # 한 단어 가설/목적
    'description':    '...',        # 이번 실험에서 뭐가 바뀌었는지 한두 문장
}
```

run 이름 자동 생성 (`_make_run_name` 사용, 직접 짓지 말 것):
```
run_{MMDD_HHMM}_{version}_{cost_structure}_{action_range}_{reward_version}_{note}
```

`wandb.init` 시 반드시:
- `config=self.config` 에 `EXP_META` 포함
- `notes=EXP_META["description"]` 전달

## 변경 기록 규칙

- 프로젝트 버전은 `PythonCode/oht_routing/version.py`의
  `CONTEXTUAL_VERSION` 하나만 사용한다. 하네스에 현재 버전 문자열을
  별도로 복사하지 않는다.
- 모든 코드·실험 변경은 `vMAJOR.MINOR.PATCH` 버전으로
  분리하고 현재 major와 같은 `EXPERIMENTS_v{MAJOR}.md`에
  해당 버전의 새 `##` 섹션을 만들어 기록한다. 현재 기록 파일은
  `EXPERIMENTS_v3.md`다.
- `CONTEXTUAL_VERSION`의 MAJOR를 올릴 때는 다른 변경을 기록하기 전에
  루트에 `EXPERIMENTS_v{new_major}.md`를 새로 만들고, 새 major의 모든
  기록은 그 파일에만 작성한다. 이전 major 파일은 이력으로 보존한다.
- MAJOR: 기존 checkpoint와 호환되지 않는 network, topology mapping,
  action, observation, replay, SALE/LAP, reward 의미 변경.
- MINOR: checkpoint 호환성을 유지하는 기능·설정·메트릭 추가.
- PATCH: 동작 계약을 바꾸지 않는 버그 수정·리팩터링·문서 변경.
- reward 공식을 바꾸면 `reward_version` 정책과 통합 버전을
  함께 검토하고 현재 `EXPERIMENTS_v{MAJOR}.md`에 호환성 영향을
  기록한다.
- `PythonCode/oht_routing/runtime/client.py` 또는
  `PythonCode/oht_routing/utils/wandb_logging.py`에서 W&B 메트릭
  추가/변경 → `PythonCode/oht_routing/utils/export_wandb_run.py`
  동시 수정

## 실행

```bash
python PythonCode/main.py
python PythonCode/main.py --mode training --action-enabled --reward-version N
```

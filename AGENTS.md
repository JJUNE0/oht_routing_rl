## 실험 관리 규칙

모든 실험 전 `EXP_META` 먼저 채울 것. run 이름·wandb config는 여기서 자동 생성.

```python
EXP_META = {
    'cost_structure': 'b_rl',       # 'b_rl' | 'residual' | 'direct'
    'action_range':   '0.5-1.5',    # 문자열로 명시
    'reward_version': 'A',          # reward 공식 바꿀 때마다 A→B→C
    'centering':      False,
    'note':           'diag',       # 한 단어 가설/목적
    'description':    '...',        # 이번 실험에서 뭐가 바뀌었는지 한두 문장
}
```

run 이름 자동 생성 (`_make_run_name` 사용, 직접 짓지 말 것):
```
run_{MMDD_HHMM}_{cost_structure}_{action_range}_{reward_version}_{note}
```

`wandb.init` 시 반드시:
- `config=self.config` 에 `EXP_META` 포함
- `notes=EXP_META["description"]` 전달

## 변경 기록 규칙

- reward 공식 변경 → `reward_version` 알파벳 올리기 + `EXPERIMENTS_v2.md` 한 줄 기록
- action 방식·obs 구조 등 코드 교체 → 버전 명시 + `EXPERIMENTS_v2.md` 기록
- `ClientAlgorithm.py` 에 wandb 메트릭 추가/변경 → `export_wandb_run.py` 동시 수정

## 실행

```bash
python main.py           # per-rail TD7
python main.py --region  # region-token TD7
```

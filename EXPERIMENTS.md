# OHT Routing RL Experiment Log

## 2026-07-19 — W&B `5ytouc3j` 붕괴 분석 및 `checkpoint_9` 재학습 준비

### 목적

- `PythonCode/ClientAlgorithm_region.py`로 수행한 region-token TD7 학습이 약 100k step 이후 붕괴한 원인을 규명한다.
- 마지막 정상 체크포인트에서 안전하게 이어 학습할 수 있도록 resume 시 replay buffer 복구 절차를 정리하고 구현한다.

### 원본 실행

| 항목 | 값 |
|---|---|
| W&B run | `5ytouc3j` |
| Run name | `run_0716_1836_b_rl_a[-1,1]_b[0,1]_J_obs_fix` |
| Model | `region_td7_concat_attention` |
| Algorithm | Token TD7 |
| Observation | rail obs 14차원, context 34차원 |
| Region | 230개, target size 50 |
| Batch / replay | 1,024 / 500,000 region transitions |
| Learning rate | actor/critic/encoder 모두 `3e-4` |
| Target update | learner update 250회마다 hard copy |
| Warm-up / curriculum | 10k / 10k→40k |
| 원본 실행 resume | `false` |
| 최종 상태 | 524k까지 실행 후 수동 중단(`KeyboardInterrupt`, W&B state `killed`) |

근거 자료:

- `PythonCode/results/wandb/5ytouc3j/run_info.json`
- `PythonCode/results/wandb/5ytouc3j/history.csv`
- `PythonCode/results/wandb/5ytouc3j/history.jsonl`
- `wandb/run-20260716_183636-5ytouc3j/files/output.log`
- `checkpoints/region_td7/20260716-203024/`

### 확정된 붕괴 타임라인

| Step | 상태 | 판정 |
|---:|---|---|
| 100,000 | `checkpoint_9`의 모델·target·optimizer tensor 전체 finite | 마지막으로 저장된 정상 상태 |
| 106,000 | encoder loss `0.0010103`, encoder grad norm `0.068464`; actor/critic/action도 finite | 마지막 완전 정상 로그 |
| 106,100 | `loss/encoder`, `grad/encoder_norm`만 최초 NaN | online encoder가 먼저 오염 |
| 106,200 | encoder만 NaN, actor/critic/action은 여전히 finite | acting용 fixed encoder는 아직 정상 |
| 106,250 | Q1 `-38.8086` 출력 직후 최초 non-finite action 경고 | 250-update hard sync에서 NaN online encoder가 fixed encoder로 복사됨 |
| 106,252 | Q1도 NaN | critic 오염 시작 |
| 106,300 | actor/critic/encoder 관련 47개 지표 NaN, action finite ratio `0` | 전체 학습 붕괴 |
| 110,000 | `checkpoint_10`의 critic 전체와 actor/optimizer 광범위 NaN | 체크포인트 오염 확정 |

106,300 이후 action은 학습된 값이 아니다. `ClientAlgorithm_region.py`가 non-finite action을 `0`으로 치환하여 `b_rl=0.5`, cost ratio `1`인 중립 baseline으로 계속 실행한 것이다. 이후 지표가 다시 좋아 보이는 구간은 모델 회복이 아니라 zero-action fallback 결과다.

### 직접 원인

확정 가능한 직접 원인은 다음과 같다.

```text
106001~106100 사이 encoder forward/loss 또는 gradient non-finite
→ finite 검사가 없는 clip_grad_norm_ 및 encoder_optimizer.step()
→ online encoder와 Adam moments 오염
→ learner update 250회 경계인 step 106250에서 fixed encoder로 hard copy
→ action NaN
→ critic, actor, target 및 optimizer 전체 오염
```

현재 encoder grad clip은 이미 `1.0`이다. 그러나 `clip_grad_norm_`의 기본 `error_if_nonfinite=False` 동작에서는 gradient 하나가 NaN이면 global norm과 나머지 gradient도 NaN이 될 수 있다. 따라서 clip 값을 단순히 낮추는 것만으로는 해결되지 않는다.

Encoder transition loss는 reward/critic과 분리되어 있으므로 reward 또는 Q loss가 online encoder를 직접 역전파해 먼저 깨뜨린 것은 아니다. 로그에서도 106,200까지 actor/critic/action은 정상이다.

### 현재 자료로 확정할 수 없는 부분

최초 non-finite를 만든 정확한 batch, feature 또는 연산은 보존되지 않았다.

- 체크포인트의 `buffer_saved=false`로 replay sample을 복원할 수 없다.
- learner 지표는 100-step 간격이다.
- sampled state/context, encoder 중간 출력, 개별 gradient가 기록되지 않았다.
- 가능한 상류 원인은 transient replay 입력 이상, 큰 finite/OOD context, MHA forward/backward overflow 등이다.

반면 다음 가설은 붕괴 시점과 일치하지 않는다.

- Replay buffer 포화: step 12,200에 이미 포화
- Reward normalizer 변경: step 30,001에 고정
- Action curriculum 변경: step 40,000에 종료
- State normalizer 변경: 첫 episode 종료인 약 step 45,000에 고정
- 이전 체크포인트 오염: 원본 run은 `resume=false`
- Simulator reset/reconnect/network 오류: 붕괴 구간에 관련 이벤트 없음

### 체크포인트 판정

| 체크포인트 | 판정 | 비고 |
|---|---|---|
| `checkpoint_9` | 사용 가능 | timestep 100,000, learner training steps 90,000, 모든 tensor finite |
| `checkpoint_10` 이후 | 사용 금지 | 모델 및 optimizer non-finite |
| `best` | 사용 금지 | 붕괴 후 step 269,995에서 zero-action 성능으로 덮어씀 |

`checkpoint_9`에는 저장 당시 buffer size 500,000이 metadata로만 기록되어 있고 실제 replay buffer 내용은 저장되지 않았다.

### 2026-07-19 resume/refill 코드 변경

대상: `PythonCode/ClientAlgorithm_region.py`

- `resume=true` 및 `checkpoints/region_td7/20260716-203024/checkpoint_9` 지정
- 고정된 `update_after=102500` 같은 step 기준은 사용하지 않음
- 체크포인트가 정상적으로 로드되고 실제 `buffer.size < buffer.capacity`이면 자동으로 replay refill mode 진입
- Refill 동안 복원된 actor로 action을 추론하고 transition만 수집
- Action 선택은 기존 `eval=false`를 유지하므로 Gaussian exploration noise는 포함됨
- Refill 동안 `learner.learn()`과 모든 optimizer update 차단
- Buffer가 capacity에 도달한 완료 step도 학습하지 않고, 다음 실제 transition step부터 학습 재개
- Reconnect/Reset 후에도 buffer와 refill 상태 유지
- 첫 optimizer update 전에는 새 run의 best checkpoint 저장 차단
- 체크포인트 경로가 없으면 resume 실패로 처리하고 기존 fresh warm-up 설정 유지

추가 W&B/진단 지표:

- `phase/resume_buffer_refill`
- `buffer/fill_ratio`
- `learner/update_enabled`
- `resume/replay_refill_complete`
- `resume/replay_refill_steps`

현재 region 수가 약 230개이므로 빈 buffer 500,000개를 채우는 데 약 2,174 environment steps가 예상되지만, 코드에서는 예상 step이 아니라 실제 buffer capacity 도달 여부만 사용한다.

### 구현 검증

- UTF-8 AST parse 통과
- 실제 `checkpoint_9/checkpoint.pt` CPU load 통과
- `total_steps=100000`, `training_steps=90000` 복원 확인
- Resume 직후 buffer `0/500000`, refill active 확인
- Buffer `499999 → 500000` 경계에서 완료 step learner update 0회 확인
- 완료 다음 step부터 update 허용되는 제어 흐름 확인
- 코드에 `102_500` 고정값이 없음을 확인
- Resume 첫 learner update 전 best checkpoint 갱신 차단 확인

### 2026-07-20 — P0 non-finite fail-fast 안전 조치 완료

대상 파일:

- `PythonCode/cocel_rl/algorithms/token_td7/numerics.py`
- `PythonCode/cocel_rl/algorithms/token_td7/learner.py`
- `PythonCode/cocel_rl/algorithms/token_td7/replay_buffer.py`
- `PythonCode/cocel_rl/algorithms/token_td7/networks.py`
- `PythonCode/ClientAlgorithm_region.py`
- `PythonCode/ClientAlgorithm.py`
- `PythonCode/main.py`

구현 내용:

1. `NumericalIntegrityError(FloatingPointError)`를 추가하고 tensor/array/module/checkpoint tree의 NaN·Inf를 검사한다.
2. Replay `push()`에서 state/action/reward/next state/done/context를 저장 전에 검사한다. 실패 시 buffer size, position, storage, priority를 변경하지 않는다.
3. Replay sample과 priority update도 finite를 검사하며, non-finite priority를 uniform sampling으로 숨기지 않는다.
4. Encoder batch, `next_zs`, `pred_zs`, loss를 검사한 뒤 backward를 수행한다.
5. Encoder grad clip 값 `1.0`을 run config에 명시하고, clip을 `error_if_nonfinite=True`로 변경했다. NaN·Inf gradient 또는 non-finite global norm이면 `encoder_optimizer.step()`을 호출하지 않는다.
6. 동일한 gradient fail-fast를 critic과 actor에도 적용하고, 각 optimizer step 뒤 source module parameter를 검사한다.
7. `training_steps`는 update 예정값으로 계산하고 전체 update가 성공한 뒤에만 commit한다. 실패 batch는 step counter를 소비하지 않는다.
8. 250-update hard sync 전에 `actor.fixed_encoder`, `actor.encoder`, `actor.mlp`, `critic` source 전체를 먼저 검사한다. 하나라도 비정상이면 네 개 copy를 모두 시작하지 않는다.
9. Action의 `np.nan_to_num(..., nan=0)` fallback을 제거했다. action input/output non-finite는 즉시 fatal exception이다.
10. `masked_max()`의 `torch.nan_to_num()` 은폐 경로를 제거했다.
11. Checkpoint는 load 전에 모델·target·optimizer뿐 아니라 normalizer, `parameterDw`, optimizer hyperparameter 같은 NumPy/Python scalar metadata도 검사한다. 오염 checkpoint는 현재 learner를 덮어쓰기 전에 거부한다.
12. TD7의 `value_min/max`, `target_min/max`를 새 checkpoint부터 저장·복원한다. 기존 checkpoint는 해당 필드가 없어 기존 초기값으로 fallback한다.
13. Save payload 전체 검사가 끝난 뒤에만 checkpoint 디렉터리와 번호를 만든다. Numeric 실패 시 빈 디렉터리나 건너뛴 번호를 남기지 않는다.
14. `ClientAlgorithm.Algorithm()`과 `main.py`의 inner/outer exception handler가 numeric exception을 삼키지 않고 최상위까지 재전파한다.

실패 메시지에는 stage, learner step, buffer size, sampled replay index, tensor 이름/shape/dtype, NaN·±Inf 개수와 최초 좌표가 포함된다. 첫 실패 후 learner failure latch가 걸리며 세 optimizer의 gradient를 지운다.

### 다음 run 확인 기준

시작 직후 다음 로그가 순서대로 나와야 한다.

```text
[Resume] loaded ...checkpoint_9...
[Resume] replay refill mode enabled: size=0/500000
[Resume] replay refill in progress: ...
[Resume] replay buffer refill complete: 500000/500000
```

Refill 구간에는 다음 조건을 만족해야 한다.

- `learner.training_steps == 90000` 유지
- `phase/resume_buffer_refill == 1`
- `learner/update_enabled == 0`
- 모델 및 optimizer state 불변

Refill 완료 다음 transition부터 다음 조건을 확인한다.

- `learner.training_steps == 90001`로 증가
- `phase/resume_buffer_refill == 0`
- `learner/update_enabled == 1`
- encoder/actor/critic loss와 gradient가 모두 finite

Non-finite가 발생하면 다음 로그를 남기고 socket reconnect나 zero-action fallback 없이 프로세스가 종료되어야 한다.

```text
[NUMERIC FAIL-FAST] [encoder/gradient] ... training_step=... sample_indices=...
```

## 2026-07-20 — W&B `sqk6vzy8` 재붕괴 분석

### 자료와 실행 맥락

- History: `results/wandb/sqk6vzy8/history.csv`, `history.jsonl`
- Run info: `results/wandb/sqk6vzy8/run_info.json`
- Stdout: `wandb/run-20260719_152217-sqk6vzy8/files/output.log`
- Run name: `run_0719_1522_b_rl_a[-1,1]_b[0,1]_J_obs_fix`
- Resume source: `checkpoints/region_td7/20260716-203024/checkpoint_9`
- 새 checkpoint root: `checkpoints/region_td7/20260719-162714/`
- Replay refill 완료: env step 102,174, 총 2,174 environment steps
- Learner update 재개: env step 102,175

### 확정된 타임라인

| Env step | Encoder loss | Encoder raw grad norm | 다른 상태 | 판정 |
|---:|---:|---:|---|---|
| 224,900 | 0.003665 | 0.0231 | actor/critic/action finite | 정상 |
| 225,000 | 0.068343 | 4,096.56 | actor/critic/action finite | encoder gradient 폭증 시작 |
| 225,100 | 0.120747 | 1.36729×10¹¹ | critic grad 7.02, actor grad 0.0425, action ratio 1 | encoder만 비정상 전조 |
| 225,174 | 100-step learner 지표 없음 | 내부 최초 NaN step은 이 시점까지 발생 | Q1 `-13.5880` finite 직후 action finite ratio 0 | hard sync로 fixed encoder 전파 |
| 225,176 | — | — | Q1 최초 NaN | critic 전파 시작 |
| 225,200 | NaN | NaN | actor/critic/loss/grad 47개 NaN, action ratio 0 | 전체 붕괴 |

Target sync 정렬은 checkpoint metadata로 정확히 확인했다.

```text
checkpoint_12: env_step=220000, training_steps=207826
225174 시점: 207826 + (225174 - 220000) = 213000
213000 % target_update_rate(250) = 0
```

따라서 env step 225,174의 learner update 끝에서 `actor.fixed_encoder <- actor.encoder` hard copy가 실행되었고, 같은 step의 다음 action부터 NaN이 되었다. 이는 `5ytouc3j`의 `encoder-only NaN → 250-step sync → action NaN → critic/actor NaN` 순서와 동일하다.

History가 100-step 간격이므로 online encoder 내부의 정확한 최초 non-finite update는 225,101~225,174 범위로만 한정된다. 외부에서 최초 관측된 non-finite action은 정확히 225,174이다.

### Checkpoint tensor 포렌식

| 체크포인트 | 판정 | 근거 |
|---|---|---|
| `checkpoint_12` | 사용 가능 | env 220,000, training steps 207,826, 모델·target·optimizer 전체 finite |
| `checkpoint_13` | 사용 금지 | env 230,000, online/fixed encoder, actor, critic, optimizer 광범위 NaN |
| `checkpoint_14` | 사용 금지 | 붕괴 후 저장 |
| `best` at 234,998 | 사용 금지 | 붕괴 후 zero-action 상태로 저장 |

`checkpoint_13`의 component signature:

- 실제 encoder loss gradient를 받는 36개 tensor, 648,960개 원소와 해당 Adam `exp_avg`/`exp_avg_sq`가 전부 NaN
- gradient 경로가 없는 `encoder.zs_proj` 6개 tensor, 230,144개 원소는 `checkpoint_12`와 bit-identical finite
- hard sync 뒤 `actor.fixed_encoder`의 동일 648,960개 원소가 NaN
- actor MLP 197,377개와 critic 526,338개 원소가 전부 NaN
- `target_actor.encoder`는 계속 finite

이 signature는 simulator/reward/serialization 문제가 아니라 trainable online encoder의 unchecked non-finite update가 원발점임을 뒷받침한다. 붕괴 구간에는 traceback, socket disconnect, reset, episode boundary가 없었고 reward와 environment 진단도 finite였다.

### Fail-fast 회귀 검증

- 전체 수정 파일 UTF-8 AST parse 통과
- 작은 실제 Token TD7 learner의 정상 update 통과, 모든 model/target tensor finite
- Encoder 사용 parameter에 Inf gradient hook 주입:
  - `NumericalIntegrityError` 발생
  - encoder/critic/actor optimizer state 불변
  - online/fixed/target actor와 critic state bit-identical
  - `training_steps` 불변
  - 모든 encoder gradient cleanup 확인
- Target-update 경계에서 online encoder parameter에 NaN 주입:
  - pre-sync에서 실패
  - 네 destination 모두 bit-identical 불변
- Replay state에 NaN 주입:
  - storage mutation 전에 실패
  - size/position/priority/storage 불변
- 실제 `checkpoint_9` load/save preflight 통과, `training_steps=90000`
- TD7 `value/target min/max`의 신규 checkpoint round-trip 확인
- NaN normalizer metadata가 checkpoint directory/counter mutation 전에 거부되는 것 확인
- 실제 `checkpoint_12` load 통과
- 실제 오염된 `checkpoint_13` load 거부 및 기존 checkpoint_12 model/optimizer state 불변 확인
- NaN action 주입 시 zero-action fallback 없이 fatal exception 확인
- `ClientAlgorithm.Algorithm()` 및 `main.py` inner/outer loop의 numeric exception 재전파 확인

현재 `ClientAlgorithm_region.py`의 resume 경로는 실행 결정에 따라 `20260719-162714/checkpoint_11`로 변경했다. `sqk6vzy8`의 마지막 정상본은 `checkpoint_12`이지만, 한 단계 앞선 `checkpoint_11`에서 안전 여유를 두고 재개한다.

기존 `checkpoint_9`, `checkpoint_11`, `checkpoint_12` 파일에는 과거 포맷상 TD7 `value/target min/max`가 없다. 따라서 legacy load는 기존 초기값으로 fallback하여 학습 재개 후 첫 target sync까지 `q_next.clamp(0, 0)`이 적용된다. `sqk6vzy8`도 같은 조건에서 123k update 이상 정상 진행했으므로 225k encoder 붕괴 원인은 아니지만, 이번 변경 이후 저장되는 checkpoint부터는 네 범위를 정확히 보존한다.

## 2026-07-20 — `checkpoint_11` fail-fast 재시작

| 항목 | 값 |
|---|---|
| Resume path | `checkpoints/region_td7/20260719-162714/checkpoint_11` |
| Env timestep | 210,000 |
| Learner training steps | 197,826 |
| Tensor/metadata finite 검사 | 통과 |
| Replay 실제 내용 | 저장되지 않음 (`buffer_saved=false`) |
| 저장 당시 buffer metadata | 500,000 |
| 예상 refill 완료 | 약 env 212,174, 실제 capacity 도달 여부로 결정 |
| 첫 learner update | refill 완료 다음 transition |
| 다음 hard sync | learner step 198,000 |
| Legacy target-bound fallback | 재개 후 174 learner update |

실행 metadata:

- `EXP_META.note=ff_ckpt11`
- Reward version J 및 obs_dim 14 유지
- Replay refill 완료 전 optimizer update 금지
- Non-finite batch/gradient/action/checkpoint 발견 시 optimizer/hard-sync 전에 fatal stop

## 2026-07-20 — W&B `dpzmmnup` encoder norm overflow

### 실행 결과

| 항목 | 값 |
|---|---|
| W&B run | `dpzmmnup` |
| Resume source | `20260719-162714/checkpoint_11` |
| Replay refill 완료 | env 212,174, 2,174 environment steps |
| Fail-fast env step | 225,439 |
| 실패 예정 learner step | 211,091 |
| 마지막 commit learner step | 211,090 |
| Buffer | 500,000 |
| Failure stage | `encoder/gradient` |
| Hard-sync 경계 여부 | 아님; 이전 211,000, 다음 211,250 |

직전 100-step 진단:

| Env step | Encoder loss | Raw encoder grad norm | Actor/critic/action |
|---:|---:|---:|---|
| 225,000 | 0.003153 | 0.0283 | finite |
| 225,100 | 0.008732 | 0.9086 | finite |
| 225,200 | 0.007021 | 56.947 | finite |
| 225,300 | 0.095364 | 10,189.18 | finite |
| 225,400 | 0.039133 | 3,296.32 | finite |
| 225,439 | scalar 미기록 | float32 total norm non-finite | optimizer step 전 중단 |

이번 failure message가 개별 bad tensor가 아니라 `gradient total norm is non-finite`였다는 점이 중요하다. Guard가 모든 개별 gradient 원소의 finite 검사를 통과한 뒤 PyTorch의 float32 L2 norm 계산만 overflow한 경우다. 따라서 true gradient norm은 최소 약 `sqrt(float32_max) = 1.84e19`를 넘은 것으로 판단한다.

Encoder loss는 최대 0.095 수준인데 gradient만 `0.028 → 0.91 → 56.9 → 1.02e4 → >1.84e19`로 폭증했다. 이는 scalar MSE 폭발이 아니라 `zsa/backbone` parameter Jacobian 폭발이다. Reward와 Q는 encoder MSE의 직접 gradient 입력이 아니며, 225,400까지 actor/critic/action은 finite였다.

Fail-fast 결과:

- `encoder_optimizer.step()` 미실행
- learner step 211,091 미commit
- hard sync 미실행
- zero-action fallback 및 후속 actor/critic 오염 없음
- 프로세스가 의도대로 즉시 종료

현재 run의 마지막 저장본 `checkpoints/region_td7/20260720-111404/checkpoint_1`은 env 220,000, learner step 205,652이며 모델·target·optimizer·metadata 전체 finite다. 새 포맷의 `value_min/max`, `target_min/max`도 정상 저장되어 있다.

정확한 최초 layer와 replay feature는 이번 실행에서도 저장되지 않았다. 다음 재현에서 확정하려면 aggregate norm overflow 시 float64 per-parameter gradient norm/max, offending batch tensor, replay priority와 encoder 중간 activation을 failure artifact로 저장해야 한다.

## 2026-07-20 - FP64 gradient guard and encoder failure artifact

- `EXP_META.note=grad64_artifact`
- Raw gradient norms are calculated per parameter in float64. The total norm is
  combined in Python float64 before clipping, so finite float32 gradients no
  longer fail merely because the float32 sum of squares overflowed.
- Encoder gradients are clipped to `encoder_grad_clip=1.0` using that float64
  norm. A raw encoder norm above `encoder_grad_abort_norm=1000.0` stops before
  `optimizer.step()` and writes a forensic artifact.
- Artifact directory: `results/numeric_failures`
- Each artifact contains the complete sampled batch, all replay sample indices,
  sampled priority/probability, the full active priority vector, float64 raw
  per-parameter gradient norms, the pre-step encoder/optimizer state, per-sample
  encoder loss, and the 16 highest-loss samples' attention Q/K/V and
  encoder/action/zsa intermediate activations. The full batch plus encoder state
  permits exact offline forward/backward analysis even when the highest-gradient
  sample is not among the highest-loss samples.
- Synthetic verification passed:
  - a raw float32 gradient norm of `2.828427e20` was measured in float64 and
    clipped to norm `1.0`;
  - forced threshold abort produced a loadable artifact with every required
    batch, replay, gradient, Q/K, and activation field.

This guard prevents a pathological update from poisoning the online encoder.
It does not by itself remove the upstream Jacobian-growth mechanism; the saved
artifact is intended to identify the first offending sample and layer before an
architectural stabilization change is selected.

## 2026-07-20 - SALE current-state `zs -> zsa` connection fix

- `EXP_META.note=sale_zs_fix`
- Fresh training (`resume=false`); legacy token checkpoints are not loaded.
- Model type: `region_td7_concat_attention_sale_zs`
- Root bug: token TD7 previously predicted `zsa` directly from a second
  backbone/action path, so the current-state `zs_proj` output was never consumed
  by the predictor. Because next-state `zs` is stop-gradient, all six `zs_proj`
  parameters had `grad=None` and remained bit-identical across checkpoints.
- Fix: retain masked mean/max pooling and append current-state `zs` to the
  `zsa_proj` input. Add `zsa_from_encoded(H, C, zs, action, mask)` and reuse the
  same encoded state in encoder, critic, target critic, and actor paths.
- `zsa_proj.0` input changes from `2*hdim + embed_dim` to
  `2*hdim + embed_dim + zs_dim` (640 to 896 with the current config).
- TD7 invariants retained: only next-state `zs` is stop-gradient, `zs` alone uses
  AvgL1Norm, `zsa` remains unnormalized, and fixed encoders roll over every 250
  learner steps.
- FP64 gradient clipping, abort threshold, and failure artifacts remain enabled.
- Verification completed on a synthetic masked region batch:
  - all six `zs_proj` weight/bias tensors received finite, nonzero gradients;
  - changing only `zs` changed `zsa`, confirming a functional dependency;
  - two complete learner updates ran through encoder, critic, and actor paths;
  - `zs_proj` weights changed after training;
  - forced failure produced a valid artifact containing the corrected `zs` and
    896-dimensional `zsa` pooled activation (48 dimensions in the reduced test).

## 2026-07-22 - Single-projection bounded attention hot-fix

- Failed run: `atndxx19`, learner step 243,799, encoder raw FP64 gradient norm
  `63,219,393.7`; artifact `encoder_failure_step_243799_20260722-014214-753950.pt`.
- The encoder loss was only `0.00307838`; the explosion was localized to the
  attention Q/K and upstream rail/context encoders rather than `zs_proj` or
  `zsa_proj`.
- Root cause: custom Q/K/V projections were followed by another set of internal
  Q/K/V projections in `nn.MultiheadAttention`. Unnormalized compounded
  projections produced Q/K up to roughly 685/888 and attention output near
  26,000, while the following LayerNorm hid the forward scale.
- `EXP_META.note=stable_attn`; reward version remains `J` because reward and
  pooling formulas are unchanged.
- Fresh training (`resume=false`); legacy region checkpoints are incompatible.
- Model type: `region_td7_single_projection_bounded_attention_sale_zs`.
- Fix: non-affine rail/context input LayerNorm, one custom Q/K/V projection,
  per-head Q/K RMS capped at one without amplifying small vectors, explicit
  float32 scaled-dot-product softmax, one output projection, and padded-query
  masking. Existing FP64 gradient guards and artifacts remain enabled.
- Verification: the exact 1,024-transition failure batch plus 40 compatible
  learned tensors produced a finite encoder gradient norm of `0.0746` instead
  of `63,219,393.7`; attention output abs-max fell from about 26,000 to 23.4.
  Two complete synthetic learner updates also passed without a numeric failure.
- Full failure analysis and migration notes: `docs/hot_fix.md`.

## 2026-07-22 - Full TD7 actor/critic and region masking hot-fix

- Source run analyzed: `atndxx19`, 2,676 history rows through env step 253,700.
- Actor first-layer weight abs-mean grew by 4.25x (H), 5.89x (C), and 3.24x
  (zs). At step 160k the second Linear abs-mean was 31.19 but only 0.0057
  remained after ReLU, with pre-tanh token std only 0.00023.
- Root bugs:
  - region policy omitted TD7's `AvgL1Norm(l0(state))`, mixed `zs` into the
    first unnormalized layer, and had one fewer policy Linear;
  - pre-tanh penalty averaged padded tokens (typically 55-65% of the batch);
  - only the attention-local context was normalized while raw C was broadcast
    to SALE, actor, and critic;
  - region rewards use a rail mean but critic Q used `sum/sqrt(n_valid)`;
  - region critic also omitted TD7's normalized state-action projection.
- Fixes:
  - `[H,C] -> state_proj -> AvgL1Norm -> concat(zs)` followed by two ReLU
    hidden layers and an output layer;
  - boolean-selected valid-token pre-tanh mean-square penalty, avoiding both
    padding gradients and the `Inf*0` masking failure mode;
  - one non-affine normalized C shared by attention, SALE, actor, and critic;
  - TD7-style critic state-action AvgL1Norm and valid-token mean Q aggregation;
  - stage-correct W&B weight, contribution, activation-scale, and valid-fraction
    diagnostics.
- `EXP_META.note=td7_full_fix`; reward version remains `J` because the reward
  formula and masked mean/max pooling are unchanged.
- Fresh training (`resume=false`); all previous region checkpoints are
  incompatible. Model type: `region_td7_bounded_attention_sale_actor_critic_v2`.
- Verification: 10x state projection scaling changed pre-tanh by at most
  `1.53e-7`; extreme padded values did not change the finite penalty; duplicated
  token critic Q differed by at most `8.20e-8`; the prior 1,024-sample failure
  artifact produced finite encoder gradient norm `0.0613`; two full learner
  updates completed with all metrics finite.
- Full report: `docs/hot_fix.md`.
# Phase 4B contextual reward/transition alignment

- 2026-07-23: Added controlled-center reward contract v1 and one-step
  contextual transition alignment. Implementation milestone only; no
  experiment or W&B run was launched.
# Phase 5 contextual step-snapshot replay

- 2026-07-23: Added raw state-ring plus transition-metadata replay v1,
  generation-safe uniform rail sampling, and CPU/device materialization.
  Implementation milestone only; no experiment or W&B run was launched.
# Phase 6 contextual TD7 offline smoke

- 2026-07-23: `b_rl / 0.1-scaled / reward A / offline500`. Synthetic
  100-step replay and 500 offline learner updates for applied-action critic,
  target, checkpoint, and numerical validation. No simulator or W&B run.
# Phase 7 contextual training runtime integration

- 2026-07-23: `contextual_td7_uniform_no_sale_v1`,
  `baseline_exp_residual`, fixed `action_scale=0.05`, policy exploration
  std/clip `0.10/0.20`, uniform snapshot replay, SALE/LAP disabled.
  Fake 1,500-tick runtime smoke only; no live simulator or W&B run.

# Contextual SALE/LAP offline validation

- 2026-07-23: `b_rl / 0.05-scaled / reward A / sale_lap_offline`.
  Added four immutable algorithm variants, independent structured SALE,
  hierarchical logical-transition LAP, automatic Huber/MSE selection, value
  bounds, strict checkpoint metadata, and W&B/export diagnostics. Synthetic
  offline validation only; no simulator or W&B run was launched.

- 2026-07-24: Added contextual W&B compatibility diagnostics for reward,
  curriculum, global simulator state, job pressure, and OHT state counts.
  Logging schema only; reward formula and running experiment are unchanged.

- 2026-07-24: Contextual resume refill `global_step_v2`. Count resumed
  policy-generated transitions using restored global runtime steps instead of
  reset episode-local steps. Replay is still intentionally refilled before
  learner updates resume; reward formula is unchanged.

- 2026-07-24: Fixed CUDA checkpoint RNG restore by converting RNG state
  tensors loaded through CUDA map-location back to CPU ByteTensors before
  `torch.cuda.set_rng_state_all`. Checkpoint contents and reward are unchanged.

- 2026-07-24: Checkpoint RNG `exploration_rng_v2` and send-cost logging
  `post_send_v2`. New checkpoints persist the dedicated exploration generator;
  older checkpoints derive a non-repeating deterministic fallback stream.
  W&B tick logging now occurs after rail-cost transmission timing is recorded.
  Reward and optimizer updates are unchanged.

# Contextual protocol/reward correctness v2

- 2026-07-24: `baseline_exp_residual / 0.05-scaled /
  contextual_reward_v2 / protocol_fix`. Moved configurable simulator end time
  into the single expected PClient initialization/reset handshake field,
  removing duplicate unframed bytes from `main_contextual`. Added a fail-closed
  SimTime progress watchdog and restored queue/TAT early termination.
- Reward v2 no longer treats an unavailable zero TAT EMA as a real zero-second
  TAT observation; until a command completes, the TAT contribution is neutral.
  Existing contextual checkpoints are intentionally incompatible and must not
  be resumed.
- Corrected contextual diagnostics: use `Job.Priority`, rename the joint
  encoder/critic objective, remove false region aliases, and include checkpoint
  plus socket-send time in `runtime/total_ms`.

# Contextual region b_rl action contract v1 / reward C

- 2026-07-24: `b_rl / b_rl_0.0-1.0 / reward C /
  contextual_region_b_rl_v1`. Changed the main contextual action mode to the
  legacy region mapping `b_rl=0.5+0.5*a_applied` and
  `cost=base+w*c*b_rl`; retained `exp_residual` as an explicit ablation.
  Warm-up is exactly neutral (`a_applied=0`, `b_rl=0.5`).
- Restored the existing region curriculum exactly: global step
  `warmup_steps..40000`, geometric scale `0.05..1.0`. Replay stores the
  deterministic policy action separately from the clipped, curriculum-scaled
  applied action; critic, target critic, actor objective, and SALE consume
  applied-action space.
- Reward C replaces residual-action smoothing in the main mode with
  `0.05*abs(delta b_rl)`. The exp-residual ablation retains its separate
  `0.5*abs(delta residual)` weight. Action/reward/learner versions are bumped;
  all earlier contextual checkpoints are intentionally incompatible and this
  action mode must start as a fresh run.

# Contextual TD7 runtime performance optimization

- 2026-07-24: Added performance-only implementations
  `contextual_runtime_bounded_reporting_v1`,
  `contextual_lap_cached_vectorized_v2`, and
  `contextual_td7_sparse_diagnostics_v1`.
- Replaced unbounded per-tick smoke diagnostic dictionaries with one latest
  row, incremental maxima, compact total-time samples, and a default 100-tick
  report interval. Console/admin tick logging now defaults to every 100 ticks.
- Preserved hierarchical LAP probabilities while caching per-step priority
  sums/statistics, vectorizing stale-key validation, and sampling rail rows by
  grouped cumulative distributions. On the existing 4,996-rail/100-snapshot
  synthetic benchmark, batch-1024 LAP sampling fell from the recorded
  53.16 ms median to 5.46 ms; priority update median was 0.95 ms.
- Reused `clip_grad_norm_` results, reused the frozen SALE state during delayed
  actor updates, and moved full online/target parameter-distance diagnostics
  to every 10 learner updates plus every hard target update.
- Action, observation, reward C, TD7/SALE/LAP equations, W&B metric names,
  checkpoint schema, and checkpoint compatibility are unchanged. The active
  simulator process was not restarted; changes apply on the next process
  start/resume.

# Contextual independent twin critic v1

- 2026-07-24: The live run `6rgz58pk` is retained for integration/smoke use
  only. Its SALE Q2 head was initialized by copying Q1, so the two critics
  remained exactly symmetric. It must not be used for performance comparison
  and its checkpoints must not be resumed.
- The replacement algorithm version is
  `contextual_directional_td7_independent_twin_critic_v1` with
  `critic_initialization=independent` and checkpoint
  `contextual_td7_checkpoint_v3_independent_twin_critic`. A fresh
  initialization is required.
- Added Q-output, Q-exclusive parameter-distance, per-head loss/gradient, and
  stateful actor-last-update diagnostics. `EXP_META.note=twincritic`; reward C,
  action mapping/range, observation, curriculum, SALE, LAP, replay, hidden
  dimensions, and learning rates are unchanged. No simulator or W&B run was
  launched for this implementation change.
2026-07-23 — contextual TD7 runtime safety v2: added fail-closed process/reconnect handling, crash-checkpoint resume refusal, generation-aligned Bellman target-Q bounds, idempotent W&B lifecycle, and real SALE+LAP production-shape runtime tests. Reward formula unchanged.

# 2026-07-25 - Contextual twin-critic live run `gurxblnf`

- W&B: `bjy6614-postech/oht-routing-contextual-td7/gurxblnf`
- Run name:
  `run_0724_2102_b_rl_b_rl_0.0-1.0_curriculum_0.05-1_C_twincritic_sale1_lap1`
- Configuration: independent twin critic, SALE on, LAP on, reward C
  (`global/local=0.7/0.3`), replay horizon 1,000 environment snapshots,
  batch 1,024, warm-up 10,000, geometric action curriculum ending at
  global step 40,000, and simulator end time 45,000.
- Final W&B state: `crashed` at global step 183,020 during episode 19.
  The last numerical diagnostics were finite
  (`numeric/learner_finite_ratio=1`, critic loss 1.059); the W&B state does
  not indicate a numerical learner failure.
- Result: training mechanics remained active, but task performance did not
  converge. After full-scale curriculum, episodes with positive mean policy
  action survived about 23.4k steps on average, while negative-mean episodes
  survived about 2.3k steps; action mean and episode survival had Pearson
  correlation about 0.75. Episodes 10-15 repeatedly terminated after roughly
  1.3k-2.3k steps, including an effective policy collapse near
  `action=-0.985`, `b_rl~0.03`.
- Decision: retain the run and checkpoints for diagnostics only. Do not treat
  it as a converged policy or resume it for the follow-up experiment.

# 2026-07-25 - Planned reward D / replay-10k follow-up

- Fresh run; results pending and will be appended after completion.
- Hypothesis: a 10,000-snapshot replay horizon will reduce the episode-scale
  actor oscillation observed in `gurxblnf`.
- Reward D:
  `contextual_controlled_reward_v4_balanced_global_local`, changing the
  normalized global/local mixture from `0.7/0.3` to `0.5/0.5`. Rail-TAT and
  action-smoothing equations are unchanged.
- Replay horizon: 10,000 environment snapshots.
- Warm-up / curriculum: warm-up 10,000; geometric action curriculum
  `0.05 -> 1.0` ending at global step 20,000.
- SALE and LAP remain enabled; batch size remains 1,024; the run starts from
  fresh model, optimizer, normalizer, and replay state.
- This follow-up changes reward balance, replay horizon, and curriculum
  duration together, so it is a multi-factor follow-up rather than a clean
  replay-only ablation.

# 2026-07-26 - Exploration annealing and policy-jitter diagnostics

- Action behavior version:
  `contextual_exploration_linear_anneal_v1`. Training exploration noise now
  remains at `0.10` through warm-up, then follows a linear schedule from
  `0.10` at global step `warmup_steps` to `0.02` at global step
  `warmup_steps + 100,000`, remaining at `0.02` afterward. Setting the initial
  std to zero keeps exploration disabled for deterministic checkpoint
  evaluation.
- Diagnostic schema: `contextual_action_temporal_diag_v4`. Added separate
  signed per-rail temporal-delta mean/std metrics for deterministic policy
  action and clipped exploratory action, plus the effective exploration std.
  Temporal state resets at episode, reconnect, and training-failure
  boundaries.
- Reward D, region `b_rl` mapping, replay contents, learner update equations,
  SALE, and LAP are unchanged. No simulator or W&B run was launched for this
  implementation change.

# 2026-07-27 - Episode-local deterministic policy burn-in

- Action behavior version: `deterministic_policy_burnin_v1`. Every training
  episode suppresses exploration, replay insertion, and learner updates for
  its first 2,000 steps. A restored/trained actor rolls out deterministically;
  a never-trained actor uses the zero-residual baseline fallback.
- The replay buffer persists across episode resets. At the burn-in boundary,
  only transition-alignment state is clean: reward/TAT history and the last
  applied action remain continuous, and the first stored transition begins
  with the first post-burn-in observation and action.
- Reward E, topology, actor/critic architecture, TD7 targets, optimizers,
  replay sampling, and action curriculum equations are unchanged.

# 2026-07-31 - Short checkpoint directory slug

- Storage-path version: `contextual_checkpoint_dir_slug_v1`. Default
  checkpoint directories now use a compact SALE/LAP/action/sampling slug plus
  a ten-character hash of the full runtime variant. Full algorithm and replay
  version strings remain unchanged in checkpoint and W&B metadata.
- This is a Windows path-length fix only. Model, reward, replay sampling,
  learner updates, and checkpoint payload contents are unchanged.

# 2026-07-28 - Selectable rail/full-snapshot replay sampling

- Replay sampling version:
  `contextual_replay_rail_or_full_snapshot_v1`.
- The default `--replay-buffer-rail` mode preserves the existing behavior:
  `batch_size` independent `(environment step, controlled rail)` samples.
- The new `--replay-buffer-snapshot --no-lap` mode interprets `batch_size` as
  the number of distinct environment steps and includes every controlled rail
  from each selected snapshot exactly once. The resulting rail rows are
  flattened and passed to the unchanged learner.
- Reward E, replay storage, observation structure, network architecture, and
  learner update equations are unchanged. `EXP_META.note=replaymode`. No
  simulator or W&B run was launched for this implementation change.

# 2026-07-28 - Flat random-rail replay sampling

- Replay sampling version: `contextual_replay_sampling_modes_v2`.
- Added `--replay-buffer-random-rail` (`--random-rail-mode`) for uniform,
  without-replacement sampling directly from the full logical
  `(environment step, controlled rail)` pool.
- Raw physical/global states remain shared by step, so the new sampler does
  not duplicate observation storage. Each sampled logical transition is
  materialized into the unchanged centered-rail learner batch contract.
- Random-rail mode requires `--no-lap`. Reward E, observation contents,
  networks, and learner equations are unchanged.
  `EXP_META.note=randomrail`. No simulator or W&B run was launched for this
  implementation change.
## 2026-07-29 — Contextual reward F: TAT confidence ramp and full trace

- `EXP_META.note=tat_ramp`, reward contract
  `contextual_controlled_reward_v8_tat_confidence_diagnostics`.
- Global TAT raw reward is multiplied by episode completion confidence
  `n / (n + 50)` by default; OP and backlog terms are unchanged.
- Added raw subterm, normalize-before-update, clipping, signed contribution,
  and absolute-scale/share W&B diagnostics.
- Reward F refuses resume from earlier reward checkpoints because their reward
  normalizer statistics are not compatible. Use a fresh run.
- Standalone reward-normalizer NPZ files now persist `reward_version`; legacy,
  missing-version, and mismatched-version NPZ files are rejected before state
  mutation.
- SALE, LAP, action curriculum, replay sampling, and learner updates are
  unchanged.

## 2026-07-30 — Command 6 selectable live-cost dispatch

- Dispatch selection version: `command6_live_path_cost_v1`.
- Added `--dispatch-mode {first-match,cost}` to `main_contextual.py`;
  `first-match` remains the default and preserves the existing eligibility,
  Job order, OHT iteration order, and `RouteList[:16]` behavior.
- `cost` selects the eligible OHT with the smallest cumulative command-0
  final rail cost over the current-to-pickup route prefix. Ties use pickup
  hops and then numeric OHT ID.
- The final cost snapshot is captured only after command 0 applies its action,
  is cleared on episode reset, and falls back to first-match before the first
  valid snapshot.
- Added aggregate dispatch diagnostics to W&B/export metadata. Reward F,
  learner, replay, observation, action mapping, and curriculum are unchanged.
- No simulator or W&B run was launched for this implementation change.

## 2026-07-30 — Command 6 indexed dispatch candidates

- Dispatch selection version:
  `command6_live_path_cost_v2_indexed_candidates`.
- Replaced the per-Job full scan over every OHT with a per-command-6
  `pickup rail -> OHT candidates` index built from `RouteList[:16]`.
- Job order, OHT insertion order, first pickup occurrence, eligibility,
  one-OHT-per-batch usage, first-match selection, and cost-mode tie-breaking
  are unchanged.
- A 510-Job/1004-OHT synthetic workload with approximately three candidates
  per Job produced identical assignments and reduced dispatch selection time
  from about 599 ms to 67 ms. Reward, observation, action, and learner
  contracts are unchanged.

## 2026-07-30 - Command 6 reassignment-churn safety fix

- Dispatch selection version:
  `command6_live_path_cost_v4_safe_assignment_payload`.
- Python dispatch now proposes only `QUEUED` Jobs with no existing OHT and
  only unbound `IDLE` OHTs. `RESERVATED`/`WAITING` Jobs and OHTs referenced
  by `JobID`, `DispatchedCommand`, or another command's `OHTId` are preserved.
- The selected OHT-to-pickup prefix is sent as the assignment route instead
  of the stale Job snapshot route.
- Continuation assignment buffers are zero-initialized after every send, and
  three Job snapshot running-area copy/paste errors were corrected.
- `EXP_META.note=dispatchsafe`. Reward F, observation, learner, replay, and
  rail-cost action contracts are unchanged.

## 2026-07-30 - First-match counterfactual dispatch diagnostics

- Diagnostic schema version: `contextual_reward_trace_v8`.
- Added cost-mode comparisons against the first-match candidate from the
  exact same candidate set: absolute/relative saving, strict-improvement
  ratio, candidate-cost spread, multi-candidate ratio, and first-match cost.
- Added cost-snapshot readiness, fallback, age, and eligible/zero-candidate
  coverage diagnostics. Existing dispatch metric names remain as aliases.
- Updated `export_wandb_run.py` with the same dispatch schema.
- `EXP_META.note=dispatchdiag`. Dispatch selection, reward F, observation,
  action, learner, and replay contracts are unchanged.

## 2026-08-03 - Same-snapshot action variance decomposition

- Diagnostic schema version: `contextual_action_decomposition_v9`.
- Added per-environment-step cross-rail deterministic-policy, raw exploration,
  post-clip action, and effective-noise variance diagnostics.
- Added policy-to-noise variance ratio, action-clip direction/frequency,
  saturation, range, and noise-suppression diagnostics. These are computed in
  normalized actor action space before `action_scale`; existing
  `action/applied_controlled_*` metrics remain the critic/applied-action view.
- Updated `export_wandb_run.py` with the same action decomposition schema.
- `EXP_META.note=actiondecomp`. Reward F, observation, action mapping,
  dispatch selection, learner, and replay contracts are unchanged.

## 2026-08-03 - Optional TAT confidence ramp CLI

- Added opt-in `args.use_tat_cofidence`. Omitting the option passes
  `tat_confidence_ramp=False`; only `--use-tat-confidence` (or the requested
  `--use-tat-cofidence` alias) enables it.
- The CLI choice is a launch control and is therefore preserved when loading
  a checkpoint instead of being overwritten by the checkpoint's saved value.
- Enabled preserves the existing
  `completed / (completed + tat_confidence_n0)` TAT reward ramp. Disabled
  applies the unramped TAT term with confidence `1.0` whenever the TAT signal
  is available.
- `EXP_META.note=tatoptin`. Reward version F and diagnostic schema v9 are
  unchanged because the existing reward branches and metrics are unchanged.

## 2026-08-03 - Reward G global level terms

- Reward version: `G`; contract version:
  `contextual_controlled_reward_v9_global_tat_op_backlog_levels`.
- Global raw reward is now
  `9.2 * (174.4236 - marginal_tat_ema) / 174.4236`
  `+ 5.0 * (0.80 - current_operation_rate)`
  `- 0.002 * (waiting + queued)`.
- Replaced the OP-rate delta reward with a current-level term. The previous
  `op_delta` remains diagnostic-only; added `op_rate`, `op_reference`, and
  `op_error` diagnostics.
- TAT confidence defaults to disabled and remains opt-in through
  `--use-tat-confidence`. TAT still uses the existing marginal-TAT EMA.
- Local reward, rail-TAT penalty, smooth penalty, global/local alpha, replay,
  action mapping, learner, and dispatch contracts are unchanged.
- Diagnostic schema version: `contextual_global_reward_v10`; synchronized
  `contextual_wandb.py` and `export_wandb_run.py`.

## 2026-08-04 - Reward H direct TotalTat level

- Reward version: `H`; contract version:
  `contextual_controlled_reward_v10_total_tat_level`.
- Removed marginal-TAT reconstruction from quantized cumulative `TotalTat` and
  completion counts, including `_prev_tat_sum`, `_prev_completed`, and the
  marginal EMA state. Global TAT reward now directly uses
  `9.2 * (174.4236 - TotalTat) / 174.4236` whenever `TotalTat > 0`.
- Completion counts remain diagnostic and are used only by the explicitly
  enabled confidence ramp. OP level, backlog, local, rail-TAT, smooth-control,
  action, replay sampling, learner, and dispatch formulas are unchanged.
- Canonical diagnostics are `reward/total_tat_level` and
  `reward/global/total_tat`; obsolete marginal-TAT metric names were removed.
  Diagnostic schema version is now `contextual_global_reward_v11`.
- `EXP_META.note=tatlevel`. Start a fresh run: do not reuse an old replay
  buffer or reward normalizer, and do not reuse an old critic/target critic.
  If reusing an actor, initialize critic, target critic, replay, and reward
  normalizer from scratch. Reward-version checks reject incompatible
  checkpoints, replay snapshots, and standalone normalizer statistics.

## 2026-08-05 - Reward I fixed scale with rail-TAT diagnostics

- Reward version: `I`; contract version:
  `contextual_controlled_reward_v11_fixed_scale_rail100`.
- Running global/local reward normalization is inactive and is never read,
  updated, or frozen by Reward I. Deprecated persistence remains only for
  compatibility/version rejection. Observation/state normalization is unchanged.
- The default contextual reward is `0.5 * global_raw +
  0.5 * (local_raw / 3) - clip(100 * rail_tat_raw, -0.5, 0.5) -
  0.25 * abs(delta_b_rl)`.
- The weighted global TAT term is clipped to `[-1, 1]`; local idle-OHT reward
  is zero while the idle count remains diagnostic-only. Rail-TAT retains the
  existing OHTTat/fixed-reference/elapsed-time attribution contract, with its
  weight applied exactly once before per-rail clipping.
- Opt-in append-only step/cycle JSONL diagnostics use global-step windows
  `0:1000`, `10000:11000`, and `20000:21000`. Occurrence-aware
  DistancePerVelocity free-flow values are diagnostic-only and never affect
  reward. Diagnostic schema version: `contextual_reward_diagnostic_v12`.
- `EXP_META.note=fixedscale_localdiv3_idleoff_rail100clip05_smooth025`;
  W&B and export schemas contain fixed-scale and bounded cycle summaries, with
  obsolete normalizer metrics removed. Start a fresh Reward I run; older
  checkpoints, replay snapshots, and reward-normalizer statistics are rejected
  by reward-version guards.

## 2026-08-05 - Reward J Phase 1 free-flow neutral ratio 2

- Reward version: `J`; contract version:
  `contextual_controlled_reward_v12_free_flow_neutral2`.
- Rail cycle raw reward is `2.0 - route_time / route_free_flow_time`; ratio 2
  is neutral, lower ratios are positive, and higher ratios are negative.
- Default mode has no offline baseline reference, calibration file, or online
  reference update. The fixed-TAT and baseline modes remain compatibility-only.
- Existing actual-elapsed-time controlled/uncontrolled rail attribution is
  retained. Phase 1 uses `rail_tat_weight=1.0` and `rail_tat_clip=1.0`; scale
  tuning is explicitly deferred to Phase 2.
- Added `cycle_fully_observed`, free-flow ratio/reward, raw attribution,
  and clip-scale cycle diagnostics plus bounded W&B summaries. Diagnostic
  schema: `contextual_reward_diagnostic_v14_free_flow_neutral2`.
- Added neutral/positive/negative, monotonicity, path-length invariance,
  attribution conservation, clipping, total-sign, and invalid-cycle tests.
- `EXP_META.note=ffreward_neutral2_w1_clip1`. Reward J requires a fresh
  critic, target critic, replay, and reward-normalizer state; Reward I
  checkpoints/replay are rejected by version guards.

## 2026-08-06 - Reward K provisional balanced neutral-2 first run

- Reward version: `K`; contract version:
  `contextual_controlled_reward_v13_balanced_freeflow_neutral2`.
- User-selected provisional coefficients: `tat_weight=18.4`,
  `backlog_weight=0.0005`, `local_predicted_oht_weight=0.10`,
  `local_reward_scale=2.0`, `rail_tat_weight=50.0`, and
  `rail_tat_clip=1.0`.
- Preserved invariants: `tat_reference=174.4236`, neutral ratio `2.0`,
  `global_alpha=local_alpha=0.5`, `smooth_b_rl_weight=0.25`, and the Phase 1
  occurrence-aware actual-elapsed rail attribution.
- Calibration status is `provisional`; these values will be re-evaluated from
  the fresh run's steady-state JSONL and W&B diagnostics.
- Added bounded representative-scale, good/bad sign, rail clipping, and
  100/1000-update Q-mean-delta metrics. Diagnostic schema:
  `contextual_reward_diagnostic_v15_balanced_neutral2`.
- `EXP_META.note=neutral2_balanced111_tatup_backlogdown_preddown`. Reward I/J
  checkpoints and replay are rejected; start with a fresh critic and replay.

## 2026-08-07 - Reward L leading-pressure rebalance and diagnostics

- Reward version: `L`; contract version:
  `contextual_controlled_reward_v14_leading_pressure_rebalanced`.
- Fixed provisional coefficients are `tat_weight=11.0`, `op_weight=4.0`,
  `backlog_weight=0.0004`, `local_predicted_oht_weight=0.075`,
  `rail_tat_weight=30.0`, and `rail_tat_clip=1.0`. Preserved settings include
  `tat_reference=165.0`, neutral-2 rail reward, actual elapsed-time rail
  attribution, global/local alpha `0.5/0.5`, local divisor `2.0`, and b_rl
  smooth weight `0.25`.
- Added global backlog-growth pressure
  `-0.16 * clip(max(0, B_t-B_{t-300})/30, 0, 1)` and idle-reserve pressure
  `-0.20 * clip(max(0, 200-idle)/50, 0, 1)`. Local idle reward remains zero.
- TAT early termination remains `TotalTat > 170` after the configured minimum
  episode steps. The final replay transition retains `done=True` and a
  broadcast terminal penalty of `-2`, excluded from continuous reward budgets.
- Added final-alpha/divisor contribution budgets using median absolute nonzero
  active-rail reward for rail calibration, target-band diagnostics, bounded
  100/300/500/1000-step leading indicators, job-ID arrival/completion flow,
  route-ratio tails, Welford composite pressure, and constant-memory delayed
  Pearson correlations at 200/500/1000 steps. Diagnostic schema:
  `contextual_reward_diagnostic_v16_leading_indicators`.
- `EXP_META.note=leadreward_tat30_pred20_flow15_idle10_rail18_op7`. Reward L
  changes the reward contract, so start a fresh critic, target critic, replay,
  and reward-normalizer/version state; older checkpoints and replay snapshots
  are rejected by the reward-version guards.

## 2026-08-07 - Compact W&B metric profile

- Added W&B metric schema `contextual_wandb_compact_v1`: the training logger
  and both exporters now share an exact 86-key allowlist.
- Kept only the requested environment, action, reward-budget, rail, learner,
  critic, replay, SALE, 300-step lead, and 500-step lead-lag metrics.
- Removed all `lap/*`, `dispatch/*`, `burnin/*`, `attention/*`, `boundary/*`,
  `protocol/*`, and `value/*` metrics from W&B, together with redundant reward,
  critic, action, and lead variants.
- This is a logging/export-only change. Rich runtime and JSONL diagnostics stay
  available for debugging; Reward L, replay compatibility, and checkpoint
  compatibility are unchanged.

## 2026-08-09 - Reward M one-sided TAT and patient termination

- Reward version: `M`; contract version:
  `contextual_controlled_reward_v15_tat_one_sided_patience`.
- Replaced only global TAT reward with
  `-11 * clip(TotalTat - 160, 0, 10) / 165`; global alpha remains `0.5` and all
  OP, backlog, backlog-growth, idle-reserve, local, neutral-2 rail, and smooth
  reward terms remain unchanged.
- TAT termination is disabled through episode step 9,999, then requires 300
  consecutive steps at `TotalTat >= 170`. The terminating replay transition
  stores `done=True` and broadcasts terminal penalty `-20` exactly once.
- `EXP_META.note=tatonesided_grace10k_patience300_terminal20`. Reward L
  checkpoints, replay, and normalizer version state are incompatible; start a
  fresh Reward M run.

## 2026-08-09 - Reward N unbounded one-sided TAT penalty

- Reward version: `N`; contract version:
  `contextual_controlled_reward_v16_tat_one_sided_unbounded`.
- Replaced only continuous global TAT with
  `-11 * max(0, TotalTat - 160) / 165`. It no longer saturates at 170 and the
  one-sided mode bypasses `tat_raw_clip` (`None` by default).
- The episode-local 10,000-step grace, `TotalTat >= 170` threshold, 300-step
  patience, terminal replay broadcast `-20`, and every non-TAT reward term are
  unchanged.
- `EXP_META.note=tatonesided_unbounded_grace10k_patience300_terminal20`.
  Reward M checkpoints, replay, and normalizer version state are incompatible;
  start a fresh Reward N run.

## 2026-08-10 - Reward O restores K plus backlog growth

- Reward version: `O`; contract version:
  `contextual_controlled_reward_v17_k_backlog_growth`.
- Restored Reward K's symmetric clipped TotalTat term and coefficients:
  `tat_reference=174.4236`, `tat_weight=18.4`, `tat_raw_clip=1.0`,
  `op_weight=5.0`, `backlog_weight=0.0005`,
  `local_predicted_oht_weight=0.10`, `local_reward_scale=2.0`,
  `rail_tat_weight=50.0`, `rail_tat_clip=1.0`, and smooth weight `0.25`.
- Added only backlog-growth raw reward
  `-0.16 * clip(max(0, B_t-B_{t-300})/30, 0, 1)`, where
  `B_t=Queued_t+Waiting_t`. Global alpha `0.5` bounds its final contribution
  to `[-0.08, 0]`; episode reset clears the backlog history.
- Disabled idle-reserve reward (`idle_reserve_weight=0.0`), all TAT early
  termination, and terminal TAT penalties. The simulator default remains the
  fixed 45,000-step horizon; queue and protocol fail-safe termination remain.
- Preserved the Reward K action curriculum, exploration schedule, neutral-2
  rail reward, SALE/LAP/replay/dispatch defaults, and compact W&B metric schema.
- `EXP_META.note=k_backloggrowth300`. Reward N and all older checkpoints,
  replay, and reward-version state are incompatible; start a fresh Reward O
  run.

## 2026-08-10 - Reward O warmup inference optimization

- Kept the Reward O formula, action schedule, and replay warmup unchanged.
- Fresh training runs now skip unused actor/encoder inference while
  `total_steps < warmup_steps`; replay continues to store the baseline-applied
  transitions, with zero diagnostic `policy_action` during that interval.
- Actor inference still runs at the warmup boundary, in `actor_inference` mode,
  and during episode burn-in when a trained checkpoint policy is available.
- `EXP_META.note=k_backloggrowth300_fastwarmup`.

## 2026-08-11 - Reward P Reward-N pressure ablation

- Reward version: `P`; contract version:
  `contextual_controlled_reward_v18_n_pressure_ablation`.
- Preserved the actual `9qokskqq` Reward N function:
  `-11 * max(0, TotalTat-160) / 165`, no TAT raw clipping, global/local alpha
  `0.5/0.5`, local divisor `2`, neutral-2 rail attribution, and b_rl smooth
  weight `0.25`.
- Preserved the run's actual termination config: 10,000-step episode grace,
  `TotalTat >= 200` for 300 consecutive steps, and terminal replay broadcast
  `-20`. This intentionally follows W&B config rather than the stale 170
  threshold text in that run's description.
- Ablation coefficients: `tat_weight=11`, `op_weight=4`,
  `backlog_weight=0.0008`, `backlog_growth_horizon=300`,
  `backlog_growth_scale=30`, `backlog_growth_weight=0.24`,
  `idle_reserve_weight=0.20`, `local_predicted_oht_weight=0.10`, and
  `rail_tat_weight=40` with clip `1`.
- Preserved the Reward O warmup inference optimization: replay collection
  continues during fresh-run warmup while unused actor/encoder inference is
  skipped.
- `EXP_META.note=n_ablate_backlog08_growth24_pred10_rail40`. Reward O and all
  older checkpoints, replay, and reward-version state are incompatible; start
  a fresh Reward P run.

## 2026-08-11 - Reward Q disables idle-reserve pressure

- Reward version: `Q`; contract version:
  `contextual_controlled_reward_v19_n_pressure_ablation_no_idle`.
- Preserved Reward P's one-sided unbounded TAT function, OP and backlog
  pressure, backlog-growth term, predicted-OHT weight `0.10`, rail weight `40`,
  smoothing, exploration, and actual `9qokskqq` TAT termination contract.
- Changed only `idle_reserve_weight=0.20 -> 0.0`, so idle-reserve observations
  remain available for diagnostics but contribute exactly zero reward.
- `EXP_META.note=n_ablate_backlog08_growth24_pred10_rail40_noidle`. Reward P
  and all older checkpoints, replay, and reward-version state are incompatible;
  start a fresh Reward Q run.

## 2026-08-11 - Episode-local parameterDw reset

- Baseline-state version: `parameter_dw_episode_reset_v1`; reward remains `Q`
  because the reward formula is unchanged.
- `ClientAlgorithm_contextual.Reset()` now clears `parameterDw`,
  `parameterPassTimes`, and `parameterC`. The next active snapshot initializes
  every rail's delay estimator from the common `1.0` prior before incorporating
  only post-reset pass-time observations.
- This removes policy-dependent delay estimates carried from episode N into
  episode N+1. `parameterDw` remains an observation feature and still directly
  scales the baseline congestion cost within an episode; observation shape,
  action mapping, reward, learner, and replay schemas are unchanged.
- The baseline-state version is included in the runtime/checkpoint variant, so
  carry-over-era checkpoints are incompatible. Start a fresh run without old
  replay or frozen observation-normalizer state.
- `EXP_META.note=n_ablate_backlog08_growth24_pred10_rail40_noidle_dwreset`.
  No simulator or W&B run was launched for this implementation change.

## 2026-08-11 - Reward R backlog pressure rebalance

- Reward version: `R`; contract version:
  `contextual_controlled_reward_v20_backlog_pressure_rebalance`.
- Changed only `backlog_weight=0.0008 -> 0.008` and
  `backlog_growth_weight=0.24 -> 0.16`.
- Preserved Reward Q's one-sided unbounded TAT term, OP term, predicted-OHT
  weight `0.10`, rail weight `40`, disabled idle-reserve term, smoothing,
  termination contract, parameterDw episode reset, actor/critic architecture,
  SALE, replay, and every learner hyperparameter.
- `EXP_META.note=backlog8e3_growth16_dwreset`. Reward Q and all older
  checkpoints, replay, and reward-normalizer state are incompatible; start a
  fresh Reward R run. No simulator or W&B run was launched for this change.

## 2026-08-11 - Contextual latent observation/action stacking

- Added configurable `num_stacks` and `stack_interval` runtime/CLI settings;
  both default to `1`, preserving the former one-frame input contract.
- Each structured observation frame is processed by the shared directional
  contextual encoder before latent concatenation. The actor receives
  `[z_t, z_t-I, ...]`; the critic receives frame-interleaved
  `[z_t, applied_a_t, z_t-I, applied_a_t-I, ...]`.
- Replay continues to store one raw state snapshot per environment step and
  materializes current/next histories at sample time. If an episode's first
  transition is retained, its initial history repeats both that transition's
  observation and applied action. If the circular buffer has already discarded
  the episode origin, sampling begins at `oldest_retained + history_horizon`
  (`history_horizon=(num_stacks-1)*stack_interval`) instead of maintaining
  per-transition overwrite-invalid flags. Ring indices never wrap into
  unrelated history or another episode.
- SALE online/fixed/target representations, online actor inference, checkpoint
  metadata, and W&B experiment metadata use the same stack contract. Versions
  were bumped for the algorithm, learner, replay, SALE, and checkpoint schemas;
  Reward remains `R` because its formula is unchanged.
- `EXP_META.note=stack`. No simulator or W&B run was launched for this
  implementation change. Stacked runs require a fresh checkpoint/replay.

## 2026-08-12 - Reward S component rebalance

- Reward version: `S`; contract version:
  `contextual_controlled_reward_v21_component_rebalance`.
- Changed `backlog_weight=0.008 -> 0.0048`,
  `backlog_growth_weight=0.16 -> 0.10`, `tat_weight=11 -> 5.5`,
  `local_reward_scale=2.0 -> 0.5`, and `rail_tat_weight=40 -> 85`; preserved
  `local_predicted_oht_weight=0.10`, `rail_tat_clip=1`, `op_weight=4`,
  `idle_reserve_weight=0`, and `smooth_b_rl_weight=0.25`.
- `EXP_META.note=rewardrebalance`. Reward R and older checkpoints, replay, and
  reward-normalizer state are incompatible; start a fresh Reward S run. No
  simulator or W&B run was launched for this change.

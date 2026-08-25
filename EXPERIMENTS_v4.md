# Contextual TD7 v4 experiments

## v4.0.0 — Reward O recent-completion TAT

### Purpose

Replace episode-cumulative `TotalTat` in the learning signal with the mean
`CmdTat` of commands whose verified unloading cycle completed during the
latest 300 simulation seconds.

### Breaking contract changes

- The global observation changes from 17 to 18 features. It replaces
  cumulative TAT with `recent_completed_tat_300s_mean_s`, adds an availability
  bit, and changes the 60-second trend to the same recent signal.
- Reward O uses `-4.3 * max(0, recent_tat - 160) / 165`.
- A sample is recorded only at a verified
  `UNLOADING -> IDLE/MOVE_TO_LOAD/LOADING` boundary using the final state-5
  `CmdTat`; command IDs are deduplicated for the episode.
- Before 300 seconds have elapsed, or when there is no completion in the
  window, the TAT value and reward are zero and availability is zero.
- v3 checkpoints and state normalizers are incompatible with v4.

### Reward profile

The initial profile was retuned offline against
`capture_20260821_081812_v2.2.0_actor_inference`. Global/dense terms use the
steady portion of the capture, while StopTime uses only rail-step observations
where the rail-local StopTime sum is positive:

| Term | Coefficient / rule | Desired observed share |
| --- | ---: | ---: |
| TAT | `4.3`, one-sided above 160 s | 28.42% |
| Backlog | `0.0007` | 8.42% |
| Backlog growth | `0.17` | 5.26% |
| Idle reserve | `0.09` | 3.16% |
| Predicted OHT | `0.05` | 12.63% |
| StopTime | `0.12 * sum(rail OHT StopTime)`, unclipped | 18.95% |
| Rail outcome | `30.0`, clipped to `[-1, 1]` | 20.00% |
| Smooth | `0.25` | 3.16% |

OP, local current-OHT, local idle, and local capacity weights are zero. These
percentages are calibration targets, not coefficient percentages.

Across steps 300--1999, the capture contains 80,055 active StopTime
rail-step observations. Their StopTime sum has mean `5.10 s`, p50 `3 s`,
p90 `12 s`, p95 `18 s`, and p99 `37 s`. With `local_alpha=0.5` and
`local_reward_scale=2`, the final StopTime contribution is therefore
`-0.03 * StopTime_sum`; it remains unbounded by design so increasing local
congestion remains distinguishable.

### Scale diagnostics

Every step logs each term's absolute final contribution under
`reward/term_scale/*_abs_mean` and normalized observed share under
`reward/term_share/*`. It also logs the absolute sum, share-sum error, reward
reconstruction error, and recent-TAT mean, p90, sample count, availability,
window age, new-event count, duplicate count, missing-final-TAT count, and
ambiguous-entry count.

### Safety and experiment status

The cumulative `TotalTat >= 200` early-termination rule is unchanged. Only the
observation and ordinary TAT reward use recent TAT. No v4 simulator result is
claimed here. The calibration is an offline replay of a fixed v2.2 trajectory;
active-rail term distributions must be checked again on the first v4 capture
before any further retuning.

## v4.1.0 — centered recent-completion TAT

### Purpose

Remove only the one-sided `max(..., 0)` dead zone from Reward O. Observation,
network, topology, action, replay, SALE/LAP, the 300-second completion window,
and every reward coefficient remain unchanged from v4.0.0. Same-major v4.0
checkpoints therefore remain compatible.

### Reward change

```text
v4.0: -4.3 * max(recent_tat_300s - 160, 0) / 165
v4.1:  4.3 * (160 - recent_tat_300s) / 165
```

The signal is still zero while the recent-completion window is unavailable.
Once available, 160 seconds is neutral; lower TAT gives positive credit and
higher TAT gives the same negative penalty as v4.0. The cumulative
`TotalTat >= 200` termination rule is unchanged.

### Scale analysis on the v4 diagnostic capture

The counterfactual was evaluated against
`results/reward_diagnostics/v4_reward_o_activescale_seed0` with recorded
states and actions held fixed.

| Window | Available samples below 160 s | v4.0 TAT abs share | v4.1 counterfactual TAT abs share | Mean total-reward shift |
| --- | ---: | ---: | ---: | ---: |
| 0--1999 | 340 / 1700 (20.00%) | 41.27% | 48.12% | +0.0392 |
| 10000--11999 | 309 / 1700 (18.18%) | 37.26% | 45.14% | +0.0367 |
| 20000--21999 | 0 / 2000 (0.00%) | 49.73% | 49.73% | 0.0000 |

Only available samples below 160 seconds change. The requested edit keeps the
TAT coefficient at `4.3`, so the observed absolute TAT share remains above the
original 28.42% calibration target. Existing W&B field names
`tat_penalty_start` and `tat_excess` are retained for v4 schema compatibility;
in v4.1 `tat_excess` is a signed `(recent_tat - 160)` delta.

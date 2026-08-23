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
- Reward O uses `-11 * max(0, recent_tat - 160) / 165`.
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

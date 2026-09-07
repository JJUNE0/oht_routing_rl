# Contextual TD7 experiment history — V9

## v9.1.0 — Reward N selectable, Stage 1 policy decoupled

### Purpose

Make the reward coefficient set a launch choice instead of a single locked
profile, and make a trained Stage 1 policy reusable across reward versions.

### Reward N

`--reward-version` now accepts `Q` (default) and `N`. Reward N restores the
coefficients of W&B run `1y9sx4a5` verbatim from that run's saved config:
`tat_weight` 11.0, operation-rate term back on at 4.0, `backlog_weight`
0.0004, `backlog_growth_weight` 0.16, `idle_reserve_weight` 0.2, local
oht/predicted/stop/capacity 0.3/0.075/0.3/0.1, no density term, rail-cycle
weight 30 with clip 1.0 at the 2.0 neutral point. Only the coefficients are
restored; the observation, action mapping, Stage 1/2 contract, replay, and
network are the current v9 ones, so this is not a reproduction of that run.

The TAT clamp is part of Reward N. Commit `2ad214e`, the code `1y9sx4a5`
actually ran, computed `max(0.0, cur_tat - TAT_PENALTY_START)` with
`tat_one_sided=True`; the clamp was dropped later, in v6.0.0.

`ContextualRewardConfig.for_version` previously ignored the profile
dictionaries and returned the dataclass defaults, so `REWARD_Q_PROFILE` was
documentation rather than the executed contract. It now builds from the
selected profile, and `REWARD_Q_PROFILE["tat_weight"]` was corrected to the
4.0 the dataclass already used.

### Stage 1 policy as a fixed frozen prefix

`--load-stage1-policy` no longer seeds the Stage 2 learner by default and no
longer requires the artifact's reward version to match the run's. A frozen
prefix is never trained, so it does not have to share the learner's
objective; only a warm-start, which seeds trainable weights, does. Opt back
in with `--stage1-policy-warm-start`, which restores both the seeding and the
reward-version check. This lets one trained Stage 1 artifact drive the prefix
across reward versions. Topology and observation compatibility are still
enforced.

## v9.2.0 — FIFO state reservation and Stage 2 evaluation contract

### Purpose

Two runtime-contract additions. Neither changes the reward, observation,
action mapping, network, or checkpoint tensors, so v9.0/v9.1 artifacts stay
loadable.

### FIFO replay state reservation

`state_capacity` was `2 * capacity` for both eviction modes. Measuring the
distinct state slots that live transitions actually reference shows the two
modes need very different reservations:

| eviction | 1-step episodes | 2 | 10 | 500 | long |
| --- | ---: | ---: | ---: | ---: | ---: |
| fifo | 2.00x | 1.50x | 1.10x | 1.01x | **1.00x** |
| random | 2.00x | 1.81x | 1.62x | 1.59x | **1.58x** |

FIFO retires transitions in insertion order, so its live set is a contiguous
window that needs one extra state per episode boundary. Random eviction
replaces uniformly chosen slots, so its live set scatters across history and
adjacent transitions rarely survive together.

`state_capacity` is now `capacity + min(margin, capacity)` for FIFO with
`replay_state_capacity_margin` defaulting to 10,000, and stays `2 * capacity`
for random eviction. Random is unchanged byte for byte: its allocator fails
hard on exhaustion, and the measured requirement reaches 2.00x.

At capacity 100,000 with LAP the estimate drops from 16.77 GiB to 11.74 GiB
for FIFO (10.81 GiB with `--no-lap`); random stays at 16.77 GiB. Undersizing
the FIFO margin is not a correctness failure: `_valid_transition_slots` drops
transitions whose state was overwritten, so the buffer loses samples instead
of corrupting them. Three diagnostics make that visible —
`replay/state_capacity`, `replay/state_slots_referenced`, and
`replay/unsamplable_env_steps`.

### Stage 2 actor_inference

`--stage 2` previously required training mode, so a Stage 2 policy could only
be evaluated without its frozen Stage 1 prefix, leaving the first 2,000 ticks
of every episode out of distribution. `actor_inference` now accepts
`--stage 2`, reproducing the training contract exactly: frozen Stage 1 for
ticks 1-2,000, the resumed Stage 2 policy from 2,001, and zero exploration
noise. It requires `--resume-checkpoint`, since the prefix alone is not a
policy under evaluation. `baseline_only` is still rejected.

No runtime change was needed: the prefix dispatch is mode-independent,
`use_stage2_actor` already covered `actor_inference`, and non-training runs
get an `_InferenceReplayContext` that satisfies the Stage 1 loader.

### Verification

- Full suite: 381 tests pass.
- Peak state usage measured for both eviction modes across episode lengths of
  1, 2, 10, 500, and none; the new reservation covers every case that the old
  `2 * capacity` covered for random, and every case with episodes of 10 steps
  or more for FIFO at the default margin.
- Undersized FIFO margins were exercised explicitly and degrade to fewer
  samplable transitions without raising.

## v9.3.0 — Explicit resume overrides and staged collection controls

### Purpose

Three runtime fixes and additions, all driven by experiments in this release.
Reward, observation, action mapping, network, and checkpoint tensors are
unchanged, so v9.0-v9.2 artifacts stay loadable.

### Explicit command-line values were ignored on resume

`runtime_config_from_args` passed `asdict(config)` into
`restore_checkpoint_runtime_config`, so by that point an explicitly typed
option was indistinguishable from a default. Every saved field not listed in
`RESUME_LAUNCH_CONTROL_FIELDS` then overwrote it. In practice
`--exploration-noise-std`, `--warmup-steps`, `--curriculum-*`, and
`--learn-every-env-steps` were silently discarded on every resumed run.

The resolver now tracks which options the user actually typed and skips those
during restore. This matches the contract v6.4.0 already documented. Untouched
options still come back from the checkpoint.

This was found while preparing a zero-noise experiment: `--exploration-noise-std 0`
resolved to the checkpoint's 0.05.

### Staged collection before learning

Two options freeze the learner for whole episodes after a resume while replay
keeps filling, so the first updates do not run against a nearly empty buffer:

| option | exploration noise | learner updates | replay |
| --- | --- | --- | --- |
| `--resume-deterministic-episodes N` | forced to 0 | frozen | kept |
| `--resume-stochastic-episodes N` | configured value | frozen | kept |

`--resume-deterministic-episodes` generalises the existing
`--resume-deterministic-first-episode` (N=1). `--resume-stochastic-episodes`
exists because a deterministic collection phase produces replay with no action
variation at all, which the zero-noise experiment below showed to be harmful.
It gates only the learner; the exploration noise path is untouched. Both are
mutually exclusive with each other, with
`--resume-deterministic-first-episode`, `--resume-warmstart-steps`, and (for
the stochastic form) `--resume-inference-until-replay-full`.

### Verification

Full suite: 385 tests pass.

## Experiments — exploration noise

### Noise is critic training data, not an execution tax

Run `zw6vkys4` resumed `stage2_env_steps=220,000` with
`--exploration-noise-std 0` and `--resume-deterministic-episodes 2`.

The two collection episodes reproduced the deterministic evaluation exactly:
TAT 168.3 and 168.0, `learner/updates` frozen at the restored 219,899. The
load path is therefore correct.

Learning then broke the policy in about 1,600 updates:

| updates | act_mean | act_std | b_rl | TAT |
| ---: | ---: | ---: | ---: | ---: |
| 0 | +0.129 | 0.401 | 0.565 | 164.8 |
| 650 | -0.673 | 0.284 | 0.163 | 167.9 |
| 910 | -0.941 | **0.055** | 0.030 | 171.9 |
| 1,560 | -0.903 | 0.080 | 0.048 | 186.4 |

`action/applied_std` collapsed from 0.47 to 0.08: the actor stopped being a
function of state and emitted a near-constant -0.9 on every rail. Meanwhile
`critic/q1_mean` barely moved (-32 to -34) and the twin critics agreed to
within 0.008. The critic saw the whole action direction as flat because
replay contained only `a = pi(s)`, so its action gradient was pure
extrapolation and nothing anchored the actor.

The collapse is not permanent. Within the next episode `act_std` returned to
0.5-0.76 and TAT fell back from 187.2 to 180.3: the collapsed action earned
bad rewards, the critic learned that, and the actor moved off the boundary.
It is a slow self-correcting oscillation, not a dead run — but 9,000 steps
after the collapse it had still not returned to its starting 168.

**`action/applied_std` dropping below roughly 0.4 is the early warning.**

### Noise level A/B

`jki8xzcv` (0.05) and `s31qyeko` (0.10) differed only in exploration noise and
device; same Stage 1 artifact by SHA, same reward, same eviction mode.

| | 0.05 | 0.10 |
| --- | ---: | ---: |
| TAT first -> last | 175.2 -> **172.3** | 174.7 -> **188.0** |
| backlog | 170.4 | 237.1 |
| action clip ratio | 1.3% | **3.9%** |
| policy's own std | 0.453 | **0.353** |

0.10 pushed actions past the bound three times as often and *reduced* the
policy's own differentiation. Combined with the zero-noise collapse, the
usable range is narrow: 0 collapses, 0.05 trains, 0.10 degrades.

Noise contributes only 1.1% of action variance at 0.05, yet removing it
breaks learning. Its value is coverage for the critic's action gradient, not
diversity in the executed policy.

### Measurement caveat

Training TAT is measured under the noisy policy. Deterministic evaluation of
the same checkpoint is better by about 0.9 s in Stage 1 and 3-5 s in Stage 2.
Compare policies with `--mode actor_inference`, never with training TAT
across runs that use different noise.

### Where the remaining headroom is not

Baseline (no RL) is 174 s; the current policy evaluates at 168 s
deterministically, against a 165 s target. On `jki8xzcv` in steady state
(`episode_step >= 20,000`):

| correlation with TAT | |
| --- | ---: |
| `env/operation_rate` | **+0.921** |
| `action/applied_std` | **-0.526** |
| `oht/idle_count` | -0.275 |
| `lead/route_ratio/p95` | **+0.024** |

Route detour length has essentially no relationship with TAT, so the routing
lever the policy controls is close to exhausted; TAT tracks load instead.
Action differentiation still helps, which is consistent with the collapse
result. `jki8xzcv` plateaued at 170.5-173.8 across episodes 5-8 with
`q1` flat at -36 and `td_error` flat at 0.80-0.85.

An earlier hypothesis in this file — that the `density` term drives detours
that raise TAT — is **not supported**: `corr(TAT, route_ratio_p95)` is +0.024,
and the best and worst episodes have indistinguishable route ratios. The
`s31qyeko` degradation was caused by the noise level, not by `density`.

## v9.3.1 — Replay diagnostics no longer scale with buffer size

### Purpose

`replay/state_slots_referenced` and `replay/unsamplable_env_steps`, added in
v9.2.0, were recomputed from scratch on every environment step. The first ran
`np.unique(..., axis=0)` over every live transition's two state references,
which is O(n log n) in the buffer size, for a single logging counter.

Measured cost per call, before:

| live transitions | cost |
| ---: | ---: |
| 1,000 | 4.2 ms |
| 5,000 | 7.7 ms |
| 20,000 | 39.0 ms |
| 60,000 | **145.7 ms** |

Run `zw6vkys4` (capacity 140,000) showed the effect directly: while the
learner was still frozen for its collection episodes, so `learner_update_ms`
and `replay_sample_ms` were both zero, `total_algorithm_ms` still grew from
205 ms at 3% fill to 418 ms at 50% fill. The v9.0.0 run `jki8xzcv` was flat
near 400 ms from 41% to 100% fill.

### Change

Both counters are now O(1) per step. Random eviction already reference-counts
its state slots, so the live count is read from that; FIFO has no such
bookkeeping and reports slots written instead. The unsamplable count is
recorded by `_valid_transition_slots` where the scan already happens and read
from that cache, so it reflects the most recent sample rather than a fresh
scan — it stays zero while a run is only collecting.

`replay/state_slots_referenced` is renamed `replay/state_slots_in_use`
because under FIFO it reports written rather than referenced slots. Verified
against an exact recomputation: under random eviction the reported value
matches the referenced set exactly (463/523/450 across episode lengths 10, 2
and none), and under FIFO it is a correct upper bound (600 vs 330, 304 vs
304, 600 vs 301). `replay/unsamplable_env_steps` matches exactly in both
modes.

The check also showed the original slow computation was itself counting the
wrong thing: it took distinct `(slot, generation)` pairs without testing
whether the generation was still current, so it included references to states
that had already been overwritten.

`diagnostics()` at 60,000 live transitions drops from about 146 ms to 1.7 ms.

PATCH: no contract, tensor, or schema change. 385 tests pass.

## v9.0.0 — Reward Q delay-weighted per-rail credit

### Purpose

Move the per-rail reward budget off the predicted-traffic forecast and onto
measured delay. Reward P's global TAT/backlog/idle coefficients, the
observation schema, networks, replay tensors, SALE/LAP, exploration, the
curriculum, the action mapping, and simulator transport are all unchanged.

### Why

Measured on
`results/environment_capture/capture_20260821_081812_v2.2.0_actor_inference`
(steps 300–1999, 4,996 controlled rails = 8,493,200 rail-steps) and on the
9,008 completed cycles in
`results/reward_diagnostics/v4_reward_o_activescale_seed0`:

- `predicted_oht_count` carried 98% of the local reward budget while its
  within-rail correlation with actual local delay was −0.014 (StopTime) and
  −0.035 (excess dwell), at every lag from 0 to 60 s. Its relationship to
  congestion is real but non-monotonic: the probability of any stop rises
  from 0.28% at `pred=0` to 1.87% at `pred=11–20`, then falls back to 0.68%
  at `pred≥41`. A linear penalty cannot express that shape, so it penalised
  the free-flowing trunk rails hardest.
- The forecast itself is accurate — 28.1% of predicted OHTs appear on the
  rail within 30 s against a 0.33% base rate (86× lift), and 74.1% of the
  OHTs that do appear within 30 s were predicted — so it belongs in the
  observation, where it already is (v5 local feature 5), not in the reward.
- The rail-cycle outcome term is the only term with correct per-rail credit
  for actual delay, and it held 2.9% of the budget. Its neutral point of 2.0
  sat well above the measured route-ratio median of 1.693, so 79.9% of
  completed cycles received positive credit; it paid out more than it
  discriminated.
- `local_oht_weight`, removed earlier, was a defensible removal for the wrong
  reason: raw occupancy correlates +0.604 with rail length, so it taught
  "avoid long rails" (the longest quartile carries 3.2× the OHTs but the
  *lowest* stop time per OHT). Normalising by length removes the confound
  (cross-rail correlation with length −0.044) while keeping the signal.

### Contract changes

| Parameter | Reward P | Reward Q |
| --- | ---: | ---: |
| `tat_weight` | 4.3 | 4.0 |
| `local_predicted_oht_weight` | 0.05 | 0.01 |
| `local_stop_weight` | 0.12 | 0.30 |
| `local_density_weight` | — | 5.5 |
| `rail_free_flow_neutral_ratio` | 2.0 | 1.70 |
| `rail_tat_weight` | 30.0 | 660.0 |
| `rail_tat_clip` | 1.0 | 22.0 |

The target is a Stage 2 budget of roughly 50% delay and 30% TAT. TAT needs
almost no coefficient change to get there — raising the delay terms dilutes it
from 49.7% to 29.9% on its own, and `tat_weight` only trims 4.3 -> 4.0.

The increase is deliberately *not* spread evenly across the three delay terms.
`stop_time` fires on 0.94% of rail-steps with std/mean 17.8 and a maximum of
117 s, so at Reward Q's earlier 0.60 weight it held 2-3% of the budget but
74.5% of the local reward variance. Scaling all three delay terms equally to
reach 50% would push the local std/mean from 3.67 to 5.02 and the maximum to
606x the mean. Routing the increase through density (dense: 17.4% nonzero,
std/mean 2.6, bounded at 3.5) and the rail-cycle outcome instead, and lowering
`stop_time` to 0.30, reaches the same 50% delay share at std/mean **2.40** and
83x maximum — less noisy than Reward Q was before the change, and far less
noisy than the equal-scaling alternative.

`local_density_weight` applies to OHT count per metre of rail
(`len(rail.OhtList) / (rail.Distance / 1000)`), read from the same live rail
record as the existing terms. The clip is raised in proportion to the weight
so it still binds at the same ~11.5% route-time share; holding it at 1.0 with
weight 660 would have truncated credit at a 0.5% share, that is, exactly the
rails an OHT dwelt longest on.

Every other coefficient — `backlog_weight` 0.0007,
`backlog_growth_weight` 0.17, `idle_reserve_weight` 0.09, `global_alpha` and
`local_alpha` 0.5, `local_reward_scale` 2.0, `smooth_b_rl_weight` 0.25,
`op_weight` 0.0 — is inherited from Reward P unchanged.

### Measured term budget

Replayed on the capture window above (steps 300–1999). The rail-outcome term
is not computable from the capture, which holds no completed-cycle ledger, so
it is derived as `cycles_per_step * mean|neutral - route_ratio| * weight /
rail_count` — the route length cancels because per-cycle credit is split
across the route in proportion to time — calibrated by a 0.82 attribution
factor that reproduces the measured Reward P value of 0.0090 and absorbs
clipping and controlled-only assignment.

| Term | Reward P share | Reward Q share |
| --- | ---: | ---: |
| `rail_outcome` | 3.93% | 35.02% |
| `density` | — | 25.34% |
| `backlog_level` | 27.79% | 15.25% |
| `tat` | 15.92% | 8.74% |
| `backlog_growth` | 11.86% | 6.51% |
| `idle_reserve` | 7.95% | 4.36% |
| `predicted_oht` | 30.99% | 3.40% |
| `stop_time` | 0.63% | 0.86% |
| `smooth` | 0.92% | 0.50% |

Delay-carrying credit moves from **4.6% to 61.2%** on this window and the
demand forecast from **31.0% to 3.4%**. The delay share reads higher here than
the 50% Stage 2 target because the capture is the Stage 1 window, where
`TotalTat` sits at p50 161.3 — barely above the 160 s threshold — so the TAT
term is nearly inactive. The 50/30 split is calibrated against Stage 2, where
`TotalTat` runs at p50 172.8; see the cross-check below.

One thing this exposes that is outside the v9 change and worth deciding
separately: `TotalTat` on this capture spans 120.1–169.1, so a TAT term that
only activates above 160 is inactive across most of the Stage 1 range.

### One-sided TAT clamp restored

Reward P documented `-4.3 * max(TotalTat - 160, 0) / 165` in its contract,
its `EXP_META`, its `reward/global/tat_excess` diagnostic, and its tests, but
`_global_raw` shipped the unclamped difference:

```python
tat_excess = cur_tat - TAT_PENALTY_START      # no max(0, ...)
```

The clamp was dropped deliberately, to give Stage 1 a TAT signal: Stage 1
episodes start with `TotalTat` near zero and spend much of their 2,000 ticks
below the 160 s threshold, where a clamped term is identically zero. Reward Q
restores the clamp anyway. The scope of that decision, measured:

| Regime | source | points | below 160 s | min |
| --- | --- | ---: | ---: | ---: |
| Stage 2 (`episode_step >= 2000`) | run `017jxhcc` | 18,826 | **0.0%** | 165.1 |
| Stage 1 prefix inside that run | run `017jxhcc` | 1,000 | 67.8% | 0.0 |
| Standalone Stage 1 | run `vp0nzi3s`, 64 episodes | 12,583 | 56.6% | 0.0 |

So the clamp is a **no-op in Stage 2** — `TotalTat` never once dropped below
160 there — and the Stage 1 prefix inside a Stage 2 run produces no replay
entry, learner sample, or update (v6.1.0), so its rewards never reach
training. The clamp changes behaviour only in standalone Stage 1 runs.

In that regime the unclamped term is mostly episode phase, not policy:

| Stage 1 signal (steps >= 300) | corr with episode step | linear-trend R² |
| --- | ---: | ---: |
| cumulative `TotalTat` | +0.904 | 81.7% |
| `recent_completed_tat_300s_mean` | +0.790 | 62.4% |

On the v2.2.0 capture, whose 2,000 steps are exactly the Stage 1 window,
`corr(step, TotalTat)` is +0.932 after step 300 and the trend explains 86.8%
of variance; residual fluctuation around it is 4.32 s against a raw std of
11.92 s. The unclamped term therefore paid a bonus that decays with episode
step — action-independent credit the critic must model and the actor cannot
earn. Over that window it averaged **+0.0883** (a *penalty* term with a
positive mean) at 0.2451 mean absolute magnitude, against −0.0784/0.0784
clamped: 3.1x the size it was calibrated for.

What the clamp costs is real but small: Stage 1 loses the ~13–18% of TAT
variation that is genuine fluctuation, for roughly the first 1,000 ticks of
each episode. Two things offset it. Reward Q raises the phase-free per-rail
delay term (`rail_outcome`) from 3.9% to 12.2% of the budget, and it is
active from the first completed cycle. And the warm-up confound is intrinsic
to the Stage 1 window rather than to the cumulative statistic — the
recent-300 s tracker trends with episode step almost as strongly. Note that
its Stage 1 mean is 167.2 s, above the 160 s threshold, so a clamped rule on
*that* signal would stay active in Stage 1; that is a separate experiment,
not part of v9.

This is also why `test_total_tat_uses_one_sided_unbounded_reward_p_curve` was
failing on `v8.0.0` for `TotalTat` of 100 and 159, and why
`reward/global/tat_excess` — which always applied the clamp — disagreed with
the reward actually paid.

#### Budget split by `TotalTat` regime

Reward Q coefficients on the same capture window, split at the 160 s
threshold (`reward_budget_by_tat_regime.py`):

| Term | `>= 160` clamped | `< 160` clamped | `< 160` unclamped (v8) |
| --- | ---: | ---: | ---: |
| `tat` | 15.04% | **0.00%** | **30.74%** |
| `rail_outcome` | 32.54% | 38.47% | 26.64% |
| `density` | 23.64% | 27.70% | 19.19% |
| `backlog_level` | 14.75% | 15.96% | 11.05% |
| `backlog_growth` | 3.61% | 10.53% | 7.30% |
| `idle_reserve` | 5.61% | 2.63% | 1.82% |
| `predicted_oht` | 3.25% | 3.61% | 2.50% |
| `stop_time` | 0.78% | 0.98% | 0.68% |
| `smooth` | 0.79% | 0.10% | 0.07% |
| delay credit | 56.95% | **67.16%** | 46.51% |
| backlog | 18.36% | **26.49%** | 18.35% |

Below the threshold the clamped reward is exactly "delay + backlog + local":
67.2% delay, 26.5% backlog, 3.6% demand, and every term is one the policy can
move. Unclamped, the TAT bonus takes 30.7% — its signed mean is **+0.1812**, a
payout, not a penalty — and it displaces the actionable terms proportionally
(delay 67.2% -> 46.5%). That dilution, not the bonus itself, is the reason to
keep the clamp.

#### Cross-check against a real Stage 2 run

The capture's `>= 160` slice is late Stage 1, where `TotalTat` p50 is 166.1 —
close to the threshold, so it understates the TAT term. Measured on the Stage
2 segment of run `017jxhcc` (18,826 points, `TotalTat` p50 172.8) under
Reward P, and rescaled by the Reward Q coefficient factors:

| Term | Reward P (measured) | Reward Q (projected) |
| --- | ---: | ---: |
| `rail_outcome` | 2.89% | 30.52% |
| `tat` | 49.73% | 29.94% |
| `density` | — | 18.91% |
| `backlog_level` | 18.28% | 11.83% |
| `backlog_growth` | 4.30% | 2.78% |
| `predicted_oht` | 20.13% | 2.60% |
| `idle_reserve` | 2.34% | 1.52% |
| `smooth` | 1.90% | 1.23% |
| `stop_time` | 0.42% | 0.67% |
| **delay** (stop + density + rail) | **3.31%** | **50.10%** |
| backlog | 22.58% | 14.61% |

`density` has no Stage 2 measurement, so it is estimated from the capture's
density-to-`predicted_oht` magnitude ratio of 1.32; the other factors are
exact. Stage 2 keeps TAT dominant at about half the budget, which is intended,
while delay credit rises 5x and the demand forecast falls 5x.

### Compatibility

MAJOR. The reward changes meaning, so V8 and older checkpoints must not be
resumed or loaded as Stage 1 policies, and `reward_version` is locked to `Q`:
`O` and `P` are retained as historical contracts and profiles and are rejected
by `canonical_reward_version`. Observation, network, topology mapping, replay
tensors, and normalizer schemas are untouched, so state-normalizer snapshots
remain structurally valid — but a policy trained under a different objective
should not be carried across.

### Verification

- Full suite: 371 tests pass
  (`python -m unittest discover -s tests -p 'test_contextual*.py'`).
- Term budget replayed from the capture; see the measured shares recorded in
  `_REWARD_Q_TARGET_SHARES` in
  `PythonCode/oht_routing/utils/wandb_logging.py`.
- No simulator or W&B run has been launched for Reward Q.

### Experiment-design note

`v8.0.0` (the `region_b_rl` `c+1` offset) has not been run either. Running
v8 and v9 as one experiment would confound an action-mapping change with an
objective change; they should be run as separate lineages. The capture used
for this calibration is a v2.2.0 actor-inference episode, so its state
distribution is not the one a v9 policy will produce — re-check the observed
shares against the first v9 capture before any further retuning.

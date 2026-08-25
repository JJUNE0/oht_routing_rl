# Contextual TD7 v5 experiments

## v5.0.0 — compact asymmetric observation and Reward P

### Purpose

Replace the v4 observation with a compact V5 schema and keep cumulative
`TotalTat` out of every actor-visible path. The critic receives a separately
normalized `total_tat_s` scalar, producing an asymmetric actor-critic
observation. In the same unreleased major, replace executable Reward O with
Reward P, whose global TAT term is a one-sided penalty over cumulative
`pclient.TotalTat`. Action, dispatch, TD7, and SALE/LAP contracts do not
change.

The experiment note is `v5_compact_obs_reward_p_total_tat` and the observation
metadata version is `v5`.

### Actor-visible observation

Each controlled rail uses the following inputs:

```text
center_local            [N, 14]
incoming_local          [N, 15, 14]
outgoing_local          [N, 15, 14]
center_rail_index       [N]
incoming_rail_indices   [N, 15]
outgoing_rail_indices   [N, 15]
incoming_relation       [N, 15, 2]
outgoing_relation       [N, 15, 2]
actor_global_state      [5]
previous_applied_action [N, 1]
```

The local feature order is fixed:

1. `free_flow_time_s`
2. `port_count`
3. `incoming_degree`
4. `outgoing_degree`
5. `predicted_oht_count`
6. `reservation_port_count`
7. `idle_count`
8. `stage_count`
9. `move_to_load_count`
10. `loading_count`
11. `move_to_unload_count`
12. `unloading_count`
13. `stop_time_sum`
14. `stopped_oht_count`

V5 removes `oht_density` and `next_10_route_oht_count` from the RL local
input. They may remain available to diagnostics. Rail identity still uses the
8-dimensional learned embedding. Relation features remain `directed_hop` and
`cumulative_free_flow_time_s` for 15 incoming and 15 outgoing rails.

The actor-global feature order is fixed:

1. `operation_rate`
2. `queued_ratio`
3. `waiting_ratio`
4. `transferring_ratio`
5. `mean_reassign`

The actor no longer observes recent/cumulative TAT, OHT state ratios, stop
rollups, or the 60-second trend features. `previous_applied_action` remains an
actor and critic input outside the observation encoder.

### Critic-only observation and replay path

The critic receives the complete actor-visible observation plus:

```text
critic_total_tat [1]  # normalized cumulative total_tat_s
```

The scalar is conditioned only in the critic branch; it must not enter the
shared actor encoder, actor-side SALE state encoder, online actor, or target
actor. Replay stores both `critic_total_tat_t` and
`critic_total_tat_t_plus_1`. The target actor uses only actor observation at
`t+1`, while the target critic receives the matching next-state critic scalar.
The online critic and actor-loss critic call receive the current-state scalar.

### Normalizer and artifact compatibility

Actor-visible features and the critic-only scalar use separate V5 normalizer
state. V4 normalizers are rejected because the dimensions and feature order
changed. V4 checkpoints and replay are also incompatible with V5 because the
network, replay observation, reward identity, and replay-target semantics
changed. A topology cache may be
reused only when its topology version, rail mapping, neighbor count, and hashes
still satisfy the unchanged 15-in/15-out topology contract.

### Reward P one-sided cumulative TotalTat contract

Reward P is the only executable runtime reward profile. Reward O remains only
as a historical contract/profile. Reward P reads the raw simulator value,
independently of the normalized critic-only observation:

```text
total_tat = pclient.TotalTat  # cumulative seconds
tat_raw   = -4.3 * max(total_tat - 160, 0) / 165
```

The boundary contract is exact:

- `TotalTat == 0`: unavailable/reset sentinel, `tat_raw = 0`
- `0 < TotalTat <= 160`: valid measurement, `tat_raw = 0`
- `TotalTat > 160`: unbounded linear negative penalty
- negative or non-finite `TotalTat`: fail fast

The recent-300-second completed-command TAT tracker remains diagnostic-only
and cannot affect Reward P. All non-TAT reward terms, coefficients, rail/local
reward meanings, action smoothing, and termination settings remain unchanged
from Reward O. Operation rate remains actor-visible state but contributes no
reward:

```text
op_weight = 0.0
use_op = false
```

Cumulative `TotalTat` remains the early-termination signal and the single
critic-only observation feature.

### Schema diagnostics

Experiment metadata records the following exact schema fields:

```text
obs/version = v5
obs/local_dim = 14
obs/incoming_neighbors = 15
obs/outgoing_neighbors = 15
obs/relation_dim = 2
obs/actor_global_dim = 5
obs/critic_extra_dim = 1
obs/actor_has_total_tat = 0
obs/critic_has_total_tat = 1
reward_version = P
tat_signal = one_sided_cumulative_total_tat_penalty
reward_tat_input = pclient.TotalTat
reward_recent_300_tat_used = false
tat_formula = -4.3*max(TotalTat-160,0)/165_for_TotalTat_gt_0
op_weight = 0.0
use_op = false
```

The startup summary prints the same V5 dimensions and the critic-only TAT
split together with the Reward P source, formula, and boundary policies. Shape,
actor TotalTat-leakage, next-state target-critic propagation, and exact Reward P
boundary tests form the compatibility gate for this release.

## v5.1.0 — Stage 1 CLI alias

### Purpose

Add `main.py --stage 1` as a CLI-only alias for:

```text
sim_end_time = 2000
```

`--stage 1` and an explicit `--sim-end-time` are mutually exclusive. On a
resume launch, explicitly passing `--stage 1` keeps the launch-controlled
simulation end time at 2,000. Without either flag, the current runtime default
of 45,000 applies; the checkpoint's saved end time is not restored.

### Compatibility

This is a checkpoint-compatible MINOR release. It does not change the model,
observation, action, reward, normalizer, replay, or checkpoint schema. V5.0
artifacts remain compatible with the v5.1 runtime under the existing same-major
and not-newer artifact rule.

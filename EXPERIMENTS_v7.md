# Contextual TD7 experiment history — V7

## v7.0.0 — Free-flow additive RL rail-cost residual

### Purpose

Add `free_flow_residual` as the default rail-cost action mode. For controlled
rail `i`, the normalized actor action is applied directly as

```text
C_i = t_ff_i + 0.5 * d_w_i * c_i + rl_cost_lambda * t_ff_i * a_i
rl_cost_lambda = 0.5
a_i in [-1, 1]
```

Neither `b_rl` nor `action_scale` participates in this mode. The executed
action stored for the existing previous-action, replay, reward-smoothing, and
critic paths is the same normalized action that drives the rail cost, so the
environment and learner action spaces remain aligned. The legacy
`region_b_rl` and `exp_residual` modes remain explicit ablations.

The runtime config adds `rl_cost_lambda=0.5`. W&B and capture diagnostics add
normalized action statistics plus the free-flow, congestion, residual, and
final-cost decomposition. Observation features, Reward P formulas, actor and
critic networks, replay structure, exploration and target noise, `d_w`, `c`,
Dijkstra routing, simulator communication, and learning hyperparameters are
unchanged.

### Compatibility

This is a MAJOR release because the default action-to-cost meaning changes and
V6 checkpoints were trained against a different environment action contract.
V7 training and inference require V7-compatible policy artifacts; V6 and older
checkpoints must not be resumed or used as Stage 1 policies.

### Verification

- Exact synthetic checks passed: zero action equals the baseline bit-for-bit;
  `t_ff=2`, `d_w*c=0.3` produces `1.15/2.15/3.15` for actions
  `-1/0/1`; and zero congestion produces `0.5/1.0/1.5 * t_ff`.
- `action_scale=0.1` and `action_scale=1.0` produced identical residual-mode
  costs and identical executed normalized actions.
- The first two records of
  `capture_20260821_081812_v2.2.0_actor_inference` passed the production cost
  function checks. At capture index 0 all 4,996 controlled rails had zero
  congestion and nonzero action, yet all costs changed according to the
  free-flow residual; rail 1 changed from `0.73` to `1.0897705835103988`.

## v7.1.0 — Configurable full-buffer replay eviction

### Purpose

Add `replay_eviction_mode` with `fifo` and `random` choices. `fifo` remains the
default and preserves the existing circular overwrite behavior exactly. Once a
replay configured for `random` reaches capacity, each incoming environment-step
transition replaces a uniformly selected live transition slot instead of the
oldest slot. Insertion before capacity remains sequential.

Random eviction uses a dedicated seeded RNG so replacement decisions do not
advance the replay-sampling RNG. Live transition states use reference counts
and a fixed-size free-slot stack, so a randomly retained transition cannot
lose its current or next state to the former FIFO state cursor. Random eviction
is currently restricted to `num_stacks=1`;
stacked observations continue to require the contiguous-history guarantees of
FIFO eviction. The runtime and checkpoint directory identities, startup
summary, checkpoint metadata/config, and W&B config, run note, replay label,
and description all record the resolved eviction mode.

### Compatibility

This is a MINOR release because it adds an opt-in replay policy and runtime
configuration without changing network tensors, topology mapping, action or
observation shapes, reward meaning, replay sample tensors, or the default FIFO
behavior. V7.0 checkpoints remain compatible with V7.1; artifacts without the
new configuration or eviction-RNG state resolve to FIFO and a deterministic
seeded fallback. Replay contents remain excluded from checkpoints and are
refilled after resume.

### Verification

- Focused configuration tests cover the FIFO default, both CLI choices,
  rejection of unknown modes, and the `random` plus `num_stacks=1` contract.
- Replay checks cover unchanged FIFO ordering, sequential fill followed by
  seeded random replacement, sampling/eviction RNG independence, live-key and
  state-reference integrity, and LAP priority reset on a replaced slot.
- Provenance checks confirm distinct runtime, checkpoint, and W&B identities
  for FIFO and random eviction modes.

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

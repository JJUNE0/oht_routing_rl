# Contextual TD7 experiment history — V8

## v8.0.0 — Offset region congestion cost

### Purpose

Make `region_b_rl` the default rail-cost action mode and give the policy a
cost effect even when the predicted near-route congestion count is zero. For
rail `i`, the runtime now uses

```text
b_rl_i = 0.5 + 0.5 * applied_action_i
C_i = t_ff_i + d_w_i * (c_i + 1) * b_rl_i
```

The previous region formula was
`C_i = t_ff_i + d_w_i * c_i * b_rl_i`. At neutral action
`b_rl_i = 0.5`, the new region baseline is therefore

```text
t_ff_i + 0.5 * d_w_i * (c_i + 1)
```

which is exactly `0.5 * d_w_i` above the previous baseline. This shift is
intentional: disabled actions, warm-up actions, and boundary rails use the
strengthened neutral region baseline as well. The `free_flow_residual` and
`exp_residual` modes remain explicit ablations and retain their existing
`t_ff_i + 0.5 * d_w_i * c_i` baseline.

Reward P, observations, networks, topology mapping, replay tensors, SALE/LAP,
exploration, curriculum scaling, and simulator transport are unchanged.

### Compatibility

This is a MAJOR release because both the default action mode and the
`region_b_rl` action-to-cost meaning change. V7 and older checkpoints were
trained against a different environment action contract and must not be
resumed or loaded as Stage 1 policies by V8.

### Verification

- Pending implementation verification.

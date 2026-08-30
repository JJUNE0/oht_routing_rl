# Contextual TD7 V6 experiments

## v6.0.0 — Optional directional attention with flat-token default

### Purpose

Make directional cross-attention opt-in through `--use-attention`. The default
encoder keeps the learned rail-identity embedding and the existing shared
center, neighbor, and global feature encoders, but replaces each directional
cross-attention block with a position-sensitive flat projection:

```text
15 encoded neighbor tokens x d_model
-> flatten in deterministic topology rank order
-> direction-specific Linear + LayerNorm + SiLU
-> d_model directional context
```

Incoming and outgoing projections remain independent. Relation features stay
paired with their corresponding neighbor before the shared token encoder. The
fusion output remains `context_dim=128`, so the actor, critic, SALE, stacking,
action, observation, reward, replay, and normalizer contracts are unchanged.

Passing `--use-attention` selects the prior one-query directional
cross-attention aggregation. The selected encoder mode is stored in both the
runtime and network checkpoint configuration and is shown in the startup and
W&B experiment metadata.

### Compatibility

This is a MAJOR release because the default encoder parameter schema changes
and V5 checkpoints do not contain the flat directional projection parameters.
V5 checkpoints must not be resumed into either V6 encoder mode. Within V6, a
checkpoint can only be resumed with the same saved `use_attention` setting.
Standalone frozen V5 state-normalizer snapshots remain loadable because the
observation schema did not change; all existing exact feature-order, dimension,
topology, mapping, epsilon, clip, population, and frozen-state checks still
apply.

## v6.1.0 — Frozen Stage 1 prefix for Stage 2

### Purpose

Add an explicit two-policy training contract selected by
`--stage 2 --load-stage1-policy <checkpoint>`:

```text
episode ticks 1..2,000       frozen Stage 1 policy
episode ticks 2,001..45,000  trainable Stage 2 policy
```

The switch is in-place: it does not send an episode termination or simulator
reset at tick 2,000. The Stage 1 prefix is deterministic and creates no pending
transition, replay entry, exploration sample, learner sample, or learner
update. At the boundary the stacked-observation history is cleared; the first
Stage 2 action is staged at zero-based `episode_steps=2000`, and its transition
is committed when the next observation arrives.

Stage 1 is loaded into a separate inference-only bundle containing only its
online encoder, online actor, and frozen SALE state encoder. Its modules are
strict-loaded, finite-checked, put in evaluation mode, frozen with
`requires_grad=False`, and checked not to share parameter objects with the
Stage 2 learner. The loader also restores the populated frozen local, global,
and critic observation normalizers and the checkpoint's applied-action scale.
It intentionally leaves the Stage 2 critic, targets, optimizers, counters,
replay, reward state, and all process/replay/exploration RNG states untouched.

Stage 2 curriculum, exploration annealing, learner cadence, and checkpoint
cadence use cumulative `stage2_env_steps`, excluding every Stage 1 prefix.
Periodic checkpoint names therefore represent Stage 2 environment time; the
full simulator time remains available as `runtime_env_step` metadata. A Stage
2 checkpoint stores only the Stage 2 learner plus Stage 1 provenance
(path/SHA-256/applied scale). Resume requires a Stage 2 checkpoint and the same
Stage 1 policy artifact.

### Compatibility

This is a MINOR release. V6.0 Stage 1 artifacts remain valid policy-only input
when their reward, topology hash, mapping hash, full network configuration,
action mode, SALE configuration, and frozen normalizers match exactly. The
new policy-only path must not be confused with full checkpoint resume:
`--resume-checkpoint` restores a Stage 2 learner, while
`--load-stage1-policy` supplies only its immutable per-episode prefix policy.
The removed `--episode-burnin-steps` option is no longer part of the CLI,
runtime configuration, diagnostics, or replay contract. The independent
`--resume-warmstart-steps` mechanism remains available for ordinary checkpoint
resume but is rejected with Stage 2.

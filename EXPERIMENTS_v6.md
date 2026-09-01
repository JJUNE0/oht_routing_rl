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

## v6.2.0 — Restore cumulative TotalTat to the actor

### Purpose

Restore normalized simulator cumulative `pclient.TotalTat` as the first
actor-visible global feature. The compact actor-global order is now:

```text
total_tat_s
operation_rate
queued_ratio
waiting_ratio
transferring_ratio
mean_reassign
```

The actor global dimension therefore changes from 5 to 6. TotalTat continues
to use pre-update running-normalizer statistics and remains available to the
critic through its existing direct one-dimensional input. Replay stores the
raw current and next TotalTat in both the actor-global vector and the existing
direct-critic scalar. Sampling reconstructs the actor and direct-critic paths
with their corresponding frozen global and critic normalizers.

Stage 1 and Stage 2 use the same six-feature actor observation. A new V6.2
Stage 1 policy must therefore be trained before starting a V6.2 Stage 2 run.

### Compatibility

This encoder-input change is checkpoint-incompatible. The V6 major is retained
by explicit experiment-lineage request, so V6.0/V6.1 model checkpoints and
state-normalizer snapshots must not be resumed or reused. The saved
`network_config.global_dim` and `observation_version`/feature-order checks reject
those artifacts before parameter or normalizer restoration. Local inspection
before this change found no semantic V6 checkpoint artifact, so no local V6
policy was migrated or promoted.

## v6.3.0 — Multi-simulator central learner runtime

### Purpose

Add an opt-in distributed Stage 2 launch contract:

```text
one main.py launch
-> N explicitly assigned simulator ports
-> one independent collector runtime per simulator
-> one central replay, TD7 learner, Stage 1 policy, CUDA owner, checkpoint owner,
   and W&B owner
```

`--num-sim N --port P0 ... PN-1` validates a one-to-one, unique TCP port
mapping before launch. Omitting both options preserves the existing single
simulator `wpconfig.json` path. Each collector retains its own PClient,
topology-backed observation builder, reward temporal state, transition aligner,
episode clock, action history, and exploration RNG. The central coordinator
performs barrier-free, deadline-bounded policy microbatching and serializes all
replay sampling, learner updates, checkpoint writes, and W&B writes on the
single GPU-owning thread.

Every learner-active collector tick consumes exactly one aggregate schedule
slot. Its central commit is ordered as learner-cadence check, optional completed
transition insertion, aggregate clock increment, and exact-boundary checkpoint
check. This prevents four queued collectors from evaluating `learn_every` at
the same stale clock and preserves the legacy update check on the first Stage 2
tick, which has no previous transition.

The simulator protocol's legacy class-level mutable dictionaries are shadowed
per `PClient` instance before its first handshake, and per-session data/TCP log
files receive unique identities. On Windows, listeners use
`SO_EXCLUSIVEADDRUSE`; this makes the all-port pre-bind fail immediately if an
old runtime already owns any requested port instead of allowing ambiguous
connection distribution through Windows `SO_REUSEADDR` semantics.

Stage 1 remains the first 2,000 local ticks of every worker episode. Stage 2
curriculum and checkpoint cadence use the aggregate count of learner-active
worker ticks. Replay episode IDs are namespaced by worker before insertion so
equal local `(episode_id, env_step)` pairs cannot merge stack histories.

The first implementation requires distributed launches to use Stage 2 with a
frozen Stage 1 policy and frozen saved observation normalizers. Zarr is not on
the live training path; the existing packed RAM replay remains authoritative.

### Compatibility

This is a MINOR release because the network, action, observation, reward,
packed replay payload, SALE/LAP, and checkpoint tensor schemas are unchanged.
V6.2 Stage 1 policy artifacts remain valid policy-only inputs after the
existing topology, mapping, network, action, SALE, reward, and frozen
normalizer checks pass. The legacy one-simulator path remains the default.
Distributed full-state resume is intentionally rejected in this first release
because one legacy checkpoint does not contain every collector's episode,
reward-history, connection-generation, and exploration-RNG state.
The rejection is performed before learner/optimizer/normalizer restoration,
including when a distributed artifact is accidentally passed to the legacy
single-simulator training path. Stage 2 prefix loading also requires explicit
`stage=1` checkpoint metadata.

## v6.4.0 — One-time Stage 2 policy warm-start and constant-noise experiment

### Purpose

Change a fresh Stage 2 launch from a fully random learner policy to a
policy-pipeline warm-start from the required Stage 1 artifact. The immutable
Stage 1 prefix remains a separate inference-only copy. Exactly once, before
the first Stage 2 transition, the fresh learner receives:

```text
Stage 1 online encoder -> Stage 2 online encoder + target encoder
Stage 1 online actor   -> Stage 2 online actor + target actor
Stage 1 sale_fixed     -> Stage 2 sale_online + sale_fixed + sale_target_fixed
```

The Stage 2 critic and target critic, every optimizer, replay, update counter,
Q bounds, reward state, applied-action scale, and RNG stream remain freshly
initialized. The loader refuses policy initialization unless replay is empty,
all update counters are zero, and optimizer state is fresh. Episode `Reset()`
does not repeat the copy, so later Stage 2 segments continue learning from the
same learner and replay. Full Stage 2 checkpoint resume restores its learned
policy and explicitly bypasses the Stage 1 warm-start.

The corresponding Stage 2 experiment uses constant raw actor-action Gaussian
exploration with both CLI endpoints set to `0.05`:

```text
--exploration-noise-std 0.05
--exploration-noise-final-std 0.05
```

For a fresh `--stage 2` CLI launch, omitting both endpoints now resolves to
this same `0.05 -> 0.05` preset; explicit endpoint arguments still override
it, and checkpoint resume restores its saved schedule. The shared Stage 1 and
general runtime defaults are intentionally unchanged. Equal endpoints make the
existing anneal calculation a no-op, so every learner-active Stage 2 tick uses
sigma `0.05`; the deterministic Stage 1 prefix still uses zero noise.
Because the existing region-B action curriculum remains `0.05 -> 1.0`, this is
constant in raw policy-action space, not in simulator-applied `b_rl` space.
Likewise, copied policy weights do not guarantee an exactly continuous
handoff while the Stage 1 saved scale and Stage 2 curriculum scale differ.

### Compatibility

This is a MINOR release because it changes the fresh Stage 2 initialization
contract without changing network tensors, observation, action, reward,
replay, SALE/LAP, or checkpoint schemas. Compatible V6.2/V6.3 Stage 1
artifacts remain valid warm-start sources. Existing V6.3 Stage 2 checkpoints
remain resumable and are never retroactively overwritten by Stage 1 weights.
Fresh-random Stage 2 and V6.4 warm-start runs are different experiment
lineages and should not be treated as directly equivalent training starts.

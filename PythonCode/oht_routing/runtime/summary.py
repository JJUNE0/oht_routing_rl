"""Human-readable startup diagnostics for the contextual runtime."""

from oht_routing.algorithms.rl.contextual_td7 import ContextualStepReplayBuffer
from oht_routing.mdp.action import FREE_FLOW_RESIDUAL, REGION_B_RL
from oht_routing.mdp.observation import (
    ACTOR_GLOBAL_DIM,
    CRITIC_EXTRA_DIM,
    LOCAL_PHYSICAL_DIM,
    OBSERVATION_VERSION,
    RELATION_DIM,
)
from oht_routing.runtime.console import print_header, print_section
from oht_routing.runtime.config_validation import make_reward_config
from oht_routing.runtime.stages import (
    STAGE_TWO,
    STAGE_TWO_STAGE1_POLICY_STEPS,
)
from oht_routing.version import CONTEXTUAL_VERSION


def print_runtime_summary(client):
    """Print the resolved runtime contract once before accepting a simulator."""
    config = client.config
    network = client.encoder.config
    estimated_replay_gib = ContextualStepReplayBuffer.estimate_capacity_bytes(
        config.replay_capacity_env_steps,
        physical_count=network.num_rails,
        controlled_count=network.num_rails - 3,
        neighbor_count=network.neighbor_count,
        lap_enabled=config.lap_enabled,
        eviction_mode=config.replay_eviction_mode,
    ) / (1024.0 ** 3)
    replay_storage = "packed V5 local u8/u16, critic TAT f32, Q15 actions"
    if config.lap_enabled:
        replay_storage += ", fp16 LAP"
    reward = make_reward_config(config)
    refill_mode = (
        "full capacity"
        if config.resume_inference_until_replay_full
        else "minimum"
    )
    exploration_end_step = (
        config.effective_warmup_steps
        + config.exploration_noise_anneal_steps
    )
    rail_tat_diagnostic = (
        "disabled"
        if client.rail_tat_diagnostic_path is None
        else (
            f"{client.rail_tat_diagnostic_path} "
            f"(global step < {config.rail_tat_diagnostic_max_step})"
        )
    )
    action_scale = (
        "normalized action direct (no action_scale)"
        if config.action_mode == FREE_FLOW_RESIDUAL
        else (
            f"{config.curriculum_scale_start} -> {config.curriculum_scale_end} "
            f"by global step {config.curriculum_end_step}"
            if config.action_mode == REGION_B_RL
            else f"fixed {config.action_scale}"
        )
    )
    checkpoint_status = (
        "deferred until simulator topology"
        if config.resume_checkpoint_path and not client.checkpoint_loaded
        else "loaded"
        if client.checkpoint_loaded
        else "not requested"
    )
    environment_capture = (
        f"enabled: {client.environment_capture.output_dir} "
        f"(first {client.environment_capture.max_steps} actor ticks)"
        if client.environment_capture is not None
        else "disabled"
    )

    print_header("contextual-runtime")
    print_section(
        "execution",
        (
            ("version", CONTEXTUAL_VERSION),
            ("mode", config.mode),
            ("action enabled", config.action_enabled),
            ("device", client.device),
            ("seed", config.seed),
            ("simulators", config.num_sim),
            (
                "explicit ports",
                config.sim_ports if config.sim_ports is not None else "wpconfig.json",
            ),
            ("dispatch mode", config.dispatch_mode),
            ("stage", config.stage),
            ("sim end time", config.sim_end_time),
            ("console log interval", config.console_log_interval),
        ),
    )
    print_section(
        "policy and observation",
        (
            ("action mode", config.action_mode),
            ("action scale schedule", action_scale),
            ("RL cost lambda", config.rl_cost_lambda),
            ("observation version", OBSERVATION_VERSION),
            ("local physical features", LOCAL_PHYSICAL_DIM),
            ("rail embedding dimensions", network.rail_embedding_dim),
            (
                "neighbor aggregation",
                "directional cross-attention"
                if network.use_attention
                else "directional flat projection (default)",
            ),
            ("relation features", RELATION_DIM),
            ("actor global features", ACTOR_GLOBAL_DIM),
            (
                "direct critic features",
                f"{CRITIC_EXTRA_DIM} (normalized total_tat_s copy)",
            ),
            ("actor has TotalTat", True),
            ("critic has TotalTat", True),
            (
                "context rails",
                f"center + {network.neighbor_count} incoming + "
                f"{network.neighbor_count} outgoing",
            ),
            ("observation stacks", config.num_stacks),
            ("stack interval", config.stack_interval),
            (
                "Stage 1 frozen prefix",
                (
                    STAGE_TWO_STAGE1_POLICY_STEPS
                    if config.stage == STAGE_TWO else "disabled"
                ),
            ),
            (
                "warm-up steps",
                f"{config.warmup_steps} -> {config.effective_warmup_steps}",
            ),
            (
                "terminate after warm-up",
                config.terminate_on_warmup_complete,
            ),
            ("normalizer load", config.load_state_normalizer_path),
            ("Stage 1 policy", config.load_stage1_policy_path),
            ("normalizer save", config.save_state_normalizer_path),
        ),
    )
    print_section(
        "checkpoint and replay",
        (
            ("resume checkpoint", config.resume_checkpoint_path),
            ("load status", checkpoint_status),
            ("sampling mode", config.replay_sampling_mode),
            ("eviction mode", config.replay_eviction_mode),
            ("capacity (environment steps)", config.replay_capacity_env_steps),
            (
                "storage format",
                replay_storage,
            ),
            ("estimated full replay RAM", f"{estimated_replay_gib:.2f} GiB"),
            ("batch size", config.batch_size),
            (
                "periodic checkpoint interval",
                config.periodic_checkpoint_interval,
            ),
            (
                "resume refill",
                f"{refill_mode} (target={config.resume_refill_target_env_steps})",
            ),
            (
                "deterministic first episode",
                config.resume_deterministic_first_episode,
            ),
            ("resume warm-start steps", config.resume_warmstart_steps),
        ),
    )
    print_section(
        "exploration",
        (
            (
                "noise std",
                f"{config.exploration_noise_std} -> "
                f"{min(config.exploration_noise_std, config.exploration_noise_final_std)}",
            ),
            (
                "anneal global steps",
                f"[{config.effective_warmup_steps}, {exploration_end_step}]",
            ),
        ),
    )
    print_section(
        f"Reward {config.reward_version}",
        (
            ("version", config.reward_version),
            ("rail reward mode", reward.rail_reward_mode),
            ("free-flow neutral ratio", reward.rail_free_flow_neutral_ratio),
            ("TAT signal", reward.contract.tat_signal_description),
            ("TAT source", "pclient.TotalTat (cumulative seconds)"),
            ("TAT formula", "-4.3 * max(TotalTat - 160, 0) / 165"),
            ("TAT zero sentinel", "TotalTat == 0 -> zero contribution"),
            ("TAT no-penalty range", "0 < TotalTat <= 160 -> zero"),
            ("TAT invalid", "TotalTat < 0 -> fail fast"),
            ("TAT weight", reward.tat_weight),
            ("operation-rate weight", reward.op_weight),
            ("operation-rate enabled", reward.use_op),
            ("backlog weight", reward.backlog_weight),
            ("backlog-growth weight", reward.backlog_growth_weight),
            ("idle-reserve weight", reward.idle_reserve_weight),
            ("predicted-OHT weight", reward.local_predicted_oht_weight),
            ("StopTime aggregation", "sum (unclipped)"),
            ("StopTime weight", reward.local_stop_weight),
            ("density signal", "OHT count per rail metre"),
            ("density weight", reward.local_density_weight),
            ("local reward scale", reward.local_reward_scale),
            ("rail TAT weight", reward.rail_tat_weight),
            ("rail TAT clip", reward.rail_tat_clip),
        ),
    )
    print_section(
        "termination",
        (
            ("policy", config.tat_termination_policy),
            ("TAT threshold", config.early_stop_tat_threshold),
            ("patience", config.tat_above_threshold_patience),
            ("grace steps", config.tat_termination_grace_steps),
            (
                "start episode",
                f"{config.tat_termination_start_episode} -> "
                f"{config.effective_tat_termination_start_episode}",
            ),
            ("terminal TAT penalty", config.terminal_tat_penalty),
        ),
    )
    print_section(
        "diagnostics",
        (
            (
                "W&B",
                (
                    f"enabled ({config.mode} profile, every "
                    f"{config.wandb_log_interval} steps)"
                    if config.wandb_enabled
                    else "disabled"
                ),
            ),
            ("environment capture", environment_capture),
            ("reward directory", config.reward_diagnostic_dir),
            ("rail TAT", rail_tat_diagnostic),
        ),
    )

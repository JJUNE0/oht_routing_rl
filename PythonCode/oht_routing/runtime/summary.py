"""Human-readable startup diagnostics for the contextual runtime."""

from oht_routing.mdp.action import REGION_B_RL
from oht_routing.runtime.console import print_header, print_section
from oht_routing.runtime.config_validation import make_reward_config
from oht_routing.version import CONTEXTUAL_VERSION


def print_runtime_summary(args, client):
    """Print the resolved runtime contract once before accepting a simulator."""
    config = client.config
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
        f"{config.curriculum_scale_start} -> {config.curriculum_scale_end} "
        f"by global step {config.curriculum_end_step}"
        if config.action_mode == REGION_B_RL
        else f"fixed {config.action_scale}"
    )
    checkpoint_status = (
        "deferred until simulator topology"
        if config.resume_checkpoint_path and not client.checkpoint_loaded
        else "loaded"
        if client.checkpoint_loaded
        else "not requested"
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
            ("dispatch mode", config.dispatch_mode),
        ),
    )
    print_section(
        "policy and observation",
        (
            ("action mode", config.action_mode),
            ("action scale schedule", action_scale),
            ("observation stacks", config.num_stacks),
            ("stack interval", config.stack_interval),
            ("episode burn-in steps", config.episode_burnin_steps),
            (
                "warm-up steps",
                f"{config.warmup_steps} -> {config.effective_warmup_steps}",
            ),
            (
                "terminate after warm-up",
                config.terminate_on_warmup_complete,
            ),
            ("normalizer load", config.load_state_normalizer_path),
            ("normalizer save", config.save_state_normalizer_path),
        ),
    )
    print_section(
        "checkpoint and replay",
        (
            ("resume checkpoint", config.resume_checkpoint_path),
            ("load status", checkpoint_status),
            ("sampling mode", config.replay_sampling_mode),
            ("batch size", config.batch_size),
            (
                "resume refill",
                f"{refill_mode} (target={config.resume_refill_target_env_steps})",
            ),
            (
                "deterministic first episode",
                config.resume_deterministic_first_episode,
            ),
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
        "Reward N",
        (
            ("version", config.reward_version),
            ("rail reward mode", reward.rail_reward_mode),
            ("free-flow neutral ratio", reward.rail_free_flow_neutral_ratio),
            ("TAT weight", reward.tat_weight),
            ("operation-rate weight", reward.op_weight),
            ("backlog weight", reward.backlog_weight),
            ("backlog-growth weight", reward.backlog_growth_weight),
            ("idle-reserve weight", reward.idle_reserve_weight),
            ("predicted-OHT weight", reward.local_predicted_oht_weight),
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
            ("reward directory", config.reward_diagnostic_dir),
            ("rail TAT", rail_tat_diagnostic),
        ),
    )

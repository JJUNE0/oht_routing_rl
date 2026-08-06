from __future__ import annotations

from dataclasses import dataclass

from contextual_action import ACTION_MODES, REGION_B_RL


ALGORITHM_VERSION = "contextual_directional_sac_v1"
LEARNER_VERSION = "contextual_sac_soft_target_auto_entropy_v1"


@dataclass(frozen=True)
class ContextualSACLearnerConfig:
    gamma: float = 0.99
    action_mode: str = REGION_B_RL
    action_scale: float = 0.1
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    encoder_lr: float = 3e-4
    temperature_lr: float = 3e-4
    adam_eps: float = 1e-8
    batch_size: int = 1024
    tau: float = 0.005
    initial_alpha: float = 0.2
    target_entropy: float | None = None
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    encoder_grad_clip: float = 1.0
    actor_grad_clip: float = 10.0
    critic_grad_clip: float = 10.0
    minimum_replay_env_steps: int = 100
    minimum_action_enabled_env_steps: int = 100
    require_normalizer_frozen: bool = True
    sale_enabled: bool = False
    lap_enabled: bool = False
    critic_loss_mode: str = "mse"

    def __post_init__(self):
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if self.action_mode not in ACTION_MODES:
            raise ValueError(f"action_mode must be one of {ACTION_MODES}")
        if self.action_scale <= 0.0:
            raise ValueError("action_scale must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")
        if self.initial_alpha <= 0.0 or self.temperature_lr <= 0.0:
            raise ValueError("SAC alpha and temperature_lr must be positive")
        if self.log_std_min >= self.log_std_max:
            raise ValueError("log_std_min must be less than log_std_max")
        if self.sale_enabled or self.lap_enabled:
            raise ValueError("contextual SAC requires uniform replay without SALE/LAP")
        if self.critic_loss_mode != "mse":
            raise ValueError("contextual SAC uses the CO-GYM MSE critic loss")

    @property
    def algorithm_variant(self) -> str:
        return "contextual_sac_uniform_v1"

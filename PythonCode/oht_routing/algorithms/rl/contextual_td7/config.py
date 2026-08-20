from dataclasses import dataclass

from oht_routing.mdp.action import ACTION_MODES, REGION_B_RL

from .stacking import validate_stack_config


ALGORITHM_VERSION = (
    "contextual_directional_td7_previous_applied_action_v3"
)


def contextual_algorithm_variant(sale_enabled: bool, lap_enabled: bool) -> str:
    return {
        (True, True): "contextual_td7_sale_lap_v3_prevact",
        (False, True): "contextual_td7_no_sale_lap_v3_prevact",
        (True, False): "contextual_td7_sale_uniform_v3_prevact",
        (False, False): "contextual_twin_delayed_uniform_v3_prevact",
    }[(bool(sale_enabled), bool(lap_enabled))]


@dataclass(frozen=True)
class ContextualNetworkConfig:
    local_dim: int = 8
    relation_dim: int = 2
    global_dim: int = 6
    neighbor_count: int = 10
    d_model: int = 64
    num_heads: int = 4
    attention_layers: int = 1
    global_emb_dim: int = 32
    context_dim: int = 128
    hidden_dim: int = 256
    action_dim: int = 1
    dropout: float = 0.0
    num_stacks: int = 1
    stack_interval: int = 1

    def __post_init__(self):
        positive = (
            "local_dim",
            "relation_dim",
            "global_dim",
            "neighbor_count",
            "d_model",
            "num_heads",
            "attention_layers",
            "global_emb_dim",
            "context_dim",
            "hidden_dim",
            "action_dim",
        )
        for name in positive:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.attention_layers != 1:
            raise ValueError("contextual network v1 requires attention_layers=1")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        validate_stack_config(self.num_stacks, self.stack_interval)

    @property
    def neighbor_token_dim(self) -> int:
        return self.local_dim + self.relation_dim

    @property
    def fusion_input_dim(self) -> int:
        return self.d_model * 3 + self.global_emb_dim

    @property
    def stacked_context_dim(self) -> int:
        return self.context_dim * self.num_stacks

    @property
    def stacked_action_dim(self) -> int:
        return self.action_dim * self.num_stacks

    @property
    def actor_input_dim(self) -> int:
        return (self.context_dim + self.action_dim) * self.num_stacks

    @property
    def critic_input_dim(self) -> int:
        return (self.context_dim + 2 * self.action_dim) * self.num_stacks


@dataclass(frozen=True)
class ContextualLearnerConfig:
    gamma: float = 0.99
    action_mode: str = REGION_B_RL
    action_scale: float = 0.1
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    encoder_lr: float = 3e-4
    adam_eps: float = 1e-8
    batch_size: int = 1024
    policy_update_delay: int = 2
    target_update_mode: str = "hard"
    target_update_interval: int = 250
    target_noise: float = 0.02
    target_noise_clip: float = 0.05
    pretanh_penalty: float = 1e-3
    encoder_grad_clip: float = 1.0
    actor_grad_clip: float = 10.0
    critic_grad_clip: float = 10.0
    minimum_replay_env_steps: int = 100
    minimum_action_enabled_env_steps: int = 100
    require_normalizer_frozen: bool = True
    sale_enabled: bool = True
    lap_enabled: bool = True
    sale_embedding_dim: int = 256
    sale_feature_dim: int = 256
    sale_lr: float = 3e-4
    lap_alpha: float = 0.4
    lap_min_priority: float = 1.0
    critic_loss_mode: str = "auto"

    def __post_init__(self):
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be in [0,1]")
        if self.action_scale <= 0:
            raise ValueError("action_scale must be positive")
        if self.action_mode not in ACTION_MODES:
            raise ValueError(f"action_mode must be one of {ACTION_MODES}")
        if self.batch_size <= 0 or self.policy_update_delay <= 0:
            raise ValueError("batch size and policy delay must be positive")
        if self.target_update_mode != "hard":
            raise ValueError("contextual learner v1 supports hard targets only")
        if self.target_update_interval <= 0:
            raise ValueError("target_update_interval must be positive")
        if self.target_noise < 0 or self.target_noise_clip < 0:
            raise ValueError("target noise values must be non-negative")
        if self.sale_embedding_dim <= 0 or self.sale_feature_dim <= 0:
            raise ValueError("SALE dimensions must be positive")
        if self.lap_alpha <= 0 or self.lap_min_priority <= 0:
            raise ValueError("LAP alpha/min priority must be positive")
        if self.critic_loss_mode not in {"auto", "huber", "mse"}:
            raise ValueError("critic_loss_mode must be auto, huber, or mse")

    @property
    def algorithm_variant(self):
        return contextual_algorithm_variant(
            self.sale_enabled, self.lap_enabled
        )

    @property
    def resolved_critic_loss_mode(self):
        if self.critic_loss_mode != "auto":
            return self.critic_loss_mode
        return "huber" if self.lap_enabled else "mse"

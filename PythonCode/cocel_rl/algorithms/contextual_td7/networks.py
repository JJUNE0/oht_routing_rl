from dataclasses import dataclass

import torch
from torch import nn

from .attention import DirectionalCrossAttention
from .config import ContextualNetworkConfig
from .stacking import interleave_state_action


class ContextualNetworkError(FloatingPointError):
    """Raised on shape or numerical contract violations."""


def _require_tensor(name: str, value: torch.Tensor, shape_tail: tuple[int, ...]):
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    expected_ndim = 1 + len(shape_tail)
    if value.ndim != expected_ndim or tuple(value.shape[1:]) != shape_tail:
        raise ValueError(
            f"{name} shape must be [B, {', '.join(map(str, shape_tail))}], "
            f"got {tuple(value.shape)}"
        )
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating dtype")
    if not bool(torch.isfinite(value).all().item()):
        raise ContextualNetworkError(f"{name} contains NaN or Inf")


def _require_finite(name: str, value: torch.Tensor):
    if not bool(torch.isfinite(value).all().item()):
        raise ContextualNetworkError(f"{name} contains NaN or Inf")


class FeatureEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


@dataclass(frozen=True)
class ContextualEncoding:
    state: torch.Tensor
    center_embedding: torch.Tensor
    global_embedding: torch.Tensor
    incoming_context: torch.Tensor
    outgoing_context: torch.Tensor
    incoming_attention: torch.Tensor | None
    outgoing_attention: torch.Tensor | None


class DirectionalContextEncoder(nn.Module):
    def __init__(self, config: ContextualNetworkConfig | None = None):
        super().__init__()
        self.config = config or ContextualNetworkConfig()
        cfg = self.config
        self.center_encoder = FeatureEncoder(cfg.local_dim, cfg.d_model)
        self.neighbor_encoder = FeatureEncoder(
            cfg.neighbor_token_dim, cfg.d_model
        )
        self.incoming_attention = DirectionalCrossAttention(
            cfg.d_model, cfg.num_heads, cfg.dropout
        )
        self.outgoing_attention = DirectionalCrossAttention(
            cfg.d_model, cfg.num_heads, cfg.dropout
        )
        self.global_encoder = FeatureEncoder(
            cfg.global_dim, cfg.global_emb_dim
        )
        self.fusion = nn.Sequential(
            nn.Linear(cfg.fusion_input_dim, cfg.context_dim),
            nn.LayerNorm(cfg.context_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        center_local: torch.Tensor,
        incoming_local: torch.Tensor,
        outgoing_local: torch.Tensor,
        incoming_relation: torch.Tensor,
        outgoing_relation: torch.Tensor,
        global_state: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> ContextualEncoding:
        cfg = self.config
        _require_tensor("center_local", center_local, (cfg.local_dim,))
        _require_tensor(
            "incoming_local",
            incoming_local,
            (cfg.neighbor_count, cfg.local_dim),
        )
        _require_tensor(
            "outgoing_local",
            outgoing_local,
            (cfg.neighbor_count, cfg.local_dim),
        )
        _require_tensor(
            "incoming_relation",
            incoming_relation,
            (cfg.neighbor_count, cfg.relation_dim),
        )
        _require_tensor(
            "outgoing_relation",
            outgoing_relation,
            (cfg.neighbor_count, cfg.relation_dim),
        )
        _require_tensor("global_state", global_state, (cfg.global_dim,))
        batch_sizes = {
            int(value.shape[0])
            for value in (
                center_local,
                incoming_local,
                outgoing_local,
                incoming_relation,
                outgoing_relation,
                global_state,
            )
        }
        if len(batch_sizes) != 1:
            raise ValueError(f"contextual input batch sizes differ: {batch_sizes}")

        center_embedding = self.center_encoder(center_local)
        incoming_tokens = torch.cat(
            (incoming_local, incoming_relation), dim=-1
        )
        outgoing_tokens = torch.cat(
            (outgoing_local, outgoing_relation), dim=-1
        )
        incoming_neighbor_embedding = self.neighbor_encoder(incoming_tokens)
        outgoing_neighbor_embedding = self.neighbor_encoder(outgoing_tokens)
        incoming_context, incoming_weights = self.incoming_attention(
            center_embedding,
            incoming_neighbor_embedding,
            return_attention=return_attention,
        )
        outgoing_context, outgoing_weights = self.outgoing_attention(
            center_embedding,
            outgoing_neighbor_embedding,
            return_attention=return_attention,
        )
        global_embedding = self.global_encoder(global_state)
        state = self.fusion(
            torch.cat(
                (
                    center_embedding,
                    incoming_context,
                    outgoing_context,
                    global_embedding,
                ),
                dim=-1,
            )
        )
        outputs = (
            ("state", state),
            ("center_embedding", center_embedding),
            ("global_embedding", global_embedding),
            ("incoming_context", incoming_context),
            ("outgoing_context", outgoing_context),
        )
        for name, value in outputs:
            _require_finite(name, value)
        if return_attention:
            if incoming_weights is None or outgoing_weights is None:
                raise ContextualNetworkError("attention weights were requested but absent")
            _require_finite("incoming_attention", incoming_weights)
            _require_finite("outgoing_attention", outgoing_weights)
        return ContextualEncoding(
            state=state,
            center_embedding=center_embedding,
            global_embedding=global_embedding,
            incoming_context=incoming_context,
            outgoing_context=outgoing_context,
            incoming_attention=incoming_weights,
            outgoing_attention=outgoing_weights,
        )


@dataclass(frozen=True)
class ActorOutput:
    action: torch.Tensor
    pre_tanh: torch.Tensor


class ContextualActor(nn.Module):
    def __init__(
        self, config: ContextualNetworkConfig | None = None,
        *, sale_embedding_dim: int = 0, sale_feature_dim: int = 0,
    ):
        super().__init__()
        self.config = config or ContextualNetworkConfig()
        cfg = self.config
        self.sale_enabled = sale_embedding_dim > 0
        self.task_projection = (
            nn.Linear(cfg.actor_input_dim, sale_feature_dim)
            if self.sale_enabled else None
        )
        input_dim = (
            sale_feature_dim + sale_embedding_dim
            if self.sale_enabled else cfg.actor_input_dim
        )
        self.network = nn.Sequential(
            nn.Linear(input_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.action_dim),
        )
        nn.init.uniform_(self.network[-1].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.network[-1].bias, -3e-3, 3e-3)

    def forward(
        self,
        state: torch.Tensor,
        sale_state: torch.Tensor | None = None,
        *,
        previous_action: torch.Tensor | None = None,
    ) -> ActorOutput:
        _require_tensor("state", state, (self.config.stacked_context_dim,))
        if previous_action is None:
            previous_action = state.new_zeros(
                state.shape[0], self.config.stacked_action_dim
            )
        _require_tensor(
            "previous_action",
            previous_action,
            (self.config.stacked_action_dim,),
        )
        if state.shape[0] != previous_action.shape[0]:
            raise ValueError("state/previous_action batch sizes differ")
        actor_state = interleave_state_action(
            state,
            previous_action,
            num_stacks=self.config.num_stacks,
            context_dim=self.config.context_dim,
            action_dim=self.config.action_dim,
        )
        if self.sale_enabled:
            from .sale import avg_l1_norm
            if sale_state is None:
                raise ValueError("SALE actor requires fixed SALE state")
            actor_input = torch.cat(
                (
                    avg_l1_norm(self.task_projection(actor_state)),
                    sale_state.detach(),
                ),
                dim=-1,
            )
        else:
            actor_input = actor_state
        pre_tanh = self.network(actor_input)
        action = torch.tanh(pre_tanh)
        _require_finite("actor.pre_tanh", pre_tanh)
        _require_finite("actor.action", action)
        return ActorOutput(action=action, pre_tanh=pre_tanh)


class CriticHead(nn.Module):
    def __init__(self, config: ContextualNetworkConfig):
        super().__init__()
        input_dim = config.critic_input_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, state_action: torch.Tensor) -> torch.Tensor:
        return self.network(state_action)


def _sale_critic_head(input_dim: int, hidden_dim: int) -> nn.Sequential:
    """Build one Q-exclusive SALE critic head.

    Q1 and Q2 must call this constructor independently. Copying a head or its
    state dict would preserve exact symmetry under the shared twin-Q loss.
    """
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, 1),
    )


@dataclass(frozen=True)
class TwinCriticOutput:
    q1: torch.Tensor
    q2: torch.Tensor


class ContextualTwinCritic(nn.Module):
    def __init__(
        self, config: ContextualNetworkConfig | None = None,
        *, sale_embedding_dim: int = 0, sale_feature_dim: int = 0,
    ):
        super().__init__()
        self.config = config or ContextualNetworkConfig()
        self.sale_enabled = sale_embedding_dim > 0
        if self.sale_enabled:
            from .sale import avg_l1_norm
            self.task_sa_projection = nn.Linear(
                self.config.critic_input_dim,
                sale_feature_dim,
            )
            input_dim = sale_feature_dim + 2 * sale_embedding_dim
            self.q1 = _sale_critic_head(
                input_dim, self.config.hidden_dim
            )
            self.q2 = _sale_critic_head(
                input_dim, self.config.hidden_dim
            )
        else:
            self.task_sa_projection = None
            self.q1 = CriticHead(self.config)
            self.q2 = CriticHead(self.config)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor,
        sale_state: torch.Tensor | None = None,
        sale_state_action: torch.Tensor | None = None,
        *,
        previous_action: torch.Tensor | None = None,
    ) -> TwinCriticOutput:
        _require_tensor("state", state, (self.config.stacked_context_dim,))
        _require_tensor("action", action, (self.config.stacked_action_dim,))
        if previous_action is None:
            previous_action = state.new_zeros(
                state.shape[0], self.config.stacked_action_dim
            )
        _require_tensor(
            "previous_action",
            previous_action,
            (self.config.stacked_action_dim,),
        )
        if not (
            state.shape[0] == action.shape[0] == previous_action.shape[0]
        ):
            raise ValueError("state/previous-action/current-action batches differ")
        state_previous_action = interleave_state_action(
            state,
            previous_action,
            num_stacks=self.config.num_stacks,
            context_dim=self.config.context_dim,
            action_dim=self.config.action_dim,
        )
        state_action = interleave_state_action(
            state_previous_action,
            action,
            num_stacks=self.config.num_stacks,
            context_dim=self.config.context_dim + self.config.action_dim,
            action_dim=self.config.action_dim,
        )
        if self.sale_enabled:
            from .sale import avg_l1_norm
            if sale_state is None or sale_state_action is None:
                raise ValueError("SALE critic requires fixed z_s and z_sa")
            state_action = torch.cat((
                avg_l1_norm(self.task_sa_projection(state_action)),
                sale_state.detach(), sale_state_action.detach(),
            ), -1)
        q1 = self.q1(state_action)
        q2 = self.q2(state_action)
        _require_finite("critic.q1", q1)
        _require_finite("critic.q2", q2)
        return TwinCriticOutput(q1=q1, q2=q2)

    def q1_value(
        self,
        state,
        action,
        sale_state=None,
        sale_state_action=None,
        *,
        previous_action=None,
    ):
        return self(
            state,
            action,
            sale_state,
            sale_state_action,
            previous_action=previous_action,
        ).q1

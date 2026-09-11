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


def _require_index_tensor(
    name: str,
    value: torch.Tensor,
    shape_tail: tuple[int, ...],
    *,
    upper_bound: int,
):
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    expected_ndim = 1 + len(shape_tail)
    if value.ndim != expected_ndim or tuple(value.shape[1:]) != shape_tail:
        raise ValueError(
            f"{name} shape must be [B"
            f"{''.join(f', {part}' for part in shape_tail)}], "
            f"got {tuple(value.shape)}"
        )
    if value.dtype != torch.long:
        raise TypeError(f"{name} must have dtype torch.long")
    if value.numel() and (
        int(value.min().item()) < 0 or int(value.max().item()) >= int(upper_bound)
    ):
        raise ValueError(
            f"{name} contains an index outside [0, {int(upper_bound)})"
        )


def _flatten_critic_extra(
    value: torch.Tensor,
    config: ContextualNetworkConfig,
    *,
    batch_size: int,
) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError("critic_total_tat must be a torch.Tensor")
    expected = (
        int(batch_size),
        int(config.num_stacks),
        int(config.critic_extra_dim),
    )
    if config.num_stacks == 1 and value.ndim == 2:
        _require_tensor(
            "critic_total_tat", value, (config.critic_extra_dim,)
        )
        stacked = value.unsqueeze(1)
    elif value.ndim == 3 and tuple(value.shape) == expected:
        if not value.is_floating_point():
            raise TypeError("critic_total_tat must have a floating dtype")
        if not bool(torch.isfinite(value).all().item()):
            raise ContextualNetworkError(
                "critic_total_tat contains NaN or Inf"
            )
        stacked = value
    else:
        raise ValueError(
            "critic_total_tat shape must be "
            f"{expected}"
            + (
                f" or ({int(batch_size)}, {config.critic_extra_dim})"
                if config.num_stacks == 1 else ""
            )
            + f", got {tuple(value.shape)}"
        )
    if stacked.shape[0] != int(batch_size):
        raise ValueError("critic TotalTat batch size differs from state")
    return stacked.reshape(batch_size, config.stacked_critic_extra_dim)


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
        self.rail_embedding = nn.Embedding(
            cfg.num_rails, cfg.rail_embedding_dim
        )
        nn.init.normal_(self.rail_embedding.weight, mean=0.0, std=0.02)
        self.center_encoder = FeatureEncoder(
            cfg.center_token_dim, cfg.d_model
        )
        self.neighbor_encoder = FeatureEncoder(
            cfg.neighbor_token_dim, cfg.d_model
        )
        if cfg.use_attention:
            self.incoming_attention = DirectionalCrossAttention(
                cfg.d_model, cfg.num_heads, cfg.dropout
            )
            self.outgoing_attention = DirectionalCrossAttention(
                cfg.d_model, cfg.num_heads, cfg.dropout
            )
            self.incoming_flat_encoder = None
            self.outgoing_flat_encoder = None
        else:
            self.incoming_attention = None
            self.outgoing_attention = None
            self.incoming_flat_encoder = FeatureEncoder(
                cfg.flat_direction_input_dim, cfg.d_model
            )
            self.outgoing_flat_encoder = FeatureEncoder(
                cfg.flat_direction_input_dim, cfg.d_model
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
        center_rail_index: torch.Tensor,
        incoming_rail_indices: torch.Tensor,
        outgoing_rail_indices: torch.Tensor,
        incoming_relation: torch.Tensor,
        outgoing_relation: torch.Tensor,
        global_state: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> ContextualEncoding:
        cfg = self.config
        _require_tensor(
            "center_local", center_local, (cfg.local_physical_dim,)
        )
        _require_tensor(
            "incoming_local",
            incoming_local,
            (cfg.neighbor_count, cfg.local_physical_dim),
        )
        _require_tensor(
            "outgoing_local",
            outgoing_local,
            (cfg.neighbor_count, cfg.local_physical_dim),
        )
        _require_index_tensor(
            "center_rail_index",
            center_rail_index,
            (),
            upper_bound=cfg.num_rails,
        )
        _require_index_tensor(
            "incoming_rail_indices",
            incoming_rail_indices,
            (cfg.neighbor_count,),
            upper_bound=cfg.num_rails,
        )
        _require_index_tensor(
            "outgoing_rail_indices",
            outgoing_rail_indices,
            (cfg.neighbor_count,),
            upper_bound=cfg.num_rails,
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
                center_rail_index,
                incoming_rail_indices,
                outgoing_rail_indices,
                incoming_relation,
                outgoing_relation,
                global_state,
            )
        }
        if len(batch_sizes) != 1:
            raise ValueError(f"contextual input batch sizes differ: {batch_sizes}")

        center_identity = self.rail_embedding(center_rail_index)
        incoming_identity = self.rail_embedding(incoming_rail_indices)
        outgoing_identity = self.rail_embedding(outgoing_rail_indices)
        center_embedding = self.center_encoder(
            torch.cat((center_local, center_identity), dim=-1)
        )
        incoming_tokens = torch.cat(
            (incoming_local, incoming_identity, incoming_relation), dim=-1
        )
        outgoing_tokens = torch.cat(
            (outgoing_local, outgoing_identity, outgoing_relation), dim=-1
        )
        incoming_neighbor_embedding = self.neighbor_encoder(incoming_tokens)
        outgoing_neighbor_embedding = self.neighbor_encoder(outgoing_tokens)
        if cfg.use_attention:
            if (
                self.incoming_attention is None
                or self.outgoing_attention is None
            ):
                raise ContextualNetworkError("attention modules are absent")
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
        else:
            if return_attention:
                raise ValueError(
                    "return_attention=True requires use_attention=True"
                )
            if (
                self.incoming_flat_encoder is None
                or self.outgoing_flat_encoder is None
            ):
                raise ContextualNetworkError(
                    "flat directional encoders are absent"
                )
            incoming_context = self.incoming_flat_encoder(
                incoming_neighbor_embedding.flatten(start_dim=1)
            )
            outgoing_context = self.outgoing_flat_encoder(
                outgoing_neighbor_embedding.flatten(start_dim=1)
            )
            incoming_weights = None
            outgoing_weights = None
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
        if cfg.use_attention and return_attention:
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
    """One non-SALE critic head over the flattened state-action input."""

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

    Every ensemble member must call this constructor independently. Copying a
    head or its state dict would preserve exact symmetry, and the ensemble
    shares one target, so independent initialization is the only thing that
    keeps the members - and therefore the UBOC spread - distinct.
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
class EnsembleCriticOutput:
    """Per-critic Q-values as [B, N], ordered by ensemble index."""

    q: torch.Tensor

    @property
    def num_critics(self) -> int:
        return int(self.q.shape[1])

    @property
    def mean(self) -> torch.Tensor:
        return self.q.mean(dim=1, keepdim=True)

    def head(self, index: int) -> torch.Tensor:
        return self.q[:, int(index):int(index) + 1]


class ContextualEnsembleCritic(nn.Module):
    """N independently initialized critic heads over a shared projection.

    N == 2 reproduces the TD7 twin critic; N > 2 supplies the ensemble that
    UBOC needs to estimate the uncertainty penalty of the Bellman target.
    """

    def __init__(
        self, config: ContextualNetworkConfig | None = None,
        *, sale_embedding_dim: int = 0, sale_feature_dim: int = 0,
    ):
        super().__init__()
        self.config = config or ContextualNetworkConfig()
        self.sale_enabled = sale_embedding_dim > 0
        num_critics = int(self.config.num_critics)
        if self.sale_enabled:
            self.task_sa_projection = nn.Linear(
                self.config.critic_state_action_dim,
                sale_feature_dim,
            )
            input_dim = (
                sale_feature_dim
                + 2 * sale_embedding_dim
                + self.config.stacked_critic_extra_dim
            )
            self.q_nets = nn.ModuleList(
                _sale_critic_head(input_dim, self.config.hidden_dim)
                for _ in range(num_critics)
            )
        else:
            self.task_sa_projection = None
            self.q_nets = nn.ModuleList(
                CriticHead(self.config) for _ in range(num_critics)
            )

    @property
    def num_critics(self) -> int:
        return len(self.q_nets)

    def _critic_input(
        self, state, action, sale_state, sale_state_action,
        *, critic_total_tat, previous_action,
    ) -> torch.Tensor:
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
        critic_extra = _flatten_critic_extra(
            critic_total_tat,
            self.config,
            batch_size=state.shape[0],
        )
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
        if not self.sale_enabled:
            return torch.cat((state_action, critic_extra), dim=-1)
        from .sale import avg_l1_norm
        if sale_state is None or sale_state_action is None:
            raise ValueError("SALE critic requires fixed z_s and z_sa")
        return torch.cat(
            (
                avg_l1_norm(self.task_sa_projection(state_action)),
                sale_state.detach(),
                sale_state_action.detach(),
                critic_extra,
            ),
            -1,
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor,
        sale_state: torch.Tensor | None = None,
        sale_state_action: torch.Tensor | None = None,
        *,
        critic_total_tat: torch.Tensor,
        previous_action: torch.Tensor | None = None,
    ) -> EnsembleCriticOutput:
        state_action = self._critic_input(
            state,
            action,
            sale_state,
            sale_state_action,
            critic_total_tat=critic_total_tat,
            previous_action=previous_action,
        )
        q = torch.cat([q_net(state_action) for q_net in self.q_nets], dim=1)
        _require_finite("critic.q", q)
        return EnsembleCriticOutput(q=q)

    def head_value(
        self,
        state,
        action,
        sale_state=None,
        sale_state_action=None,
        *,
        critic_total_tat,
        previous_action=None,
        index: int = 0,
    ) -> torch.Tensor:
        """Evaluate a single ensemble member without building the others."""
        state_action = self._critic_input(
            state,
            action,
            sale_state,
            sale_state_action,
            critic_total_tat=critic_total_tat,
            previous_action=previous_action,
        )
        q = self.q_nets[int(index)](state_action)
        _require_finite(f"critic.q{int(index) + 1}", q)
        return q

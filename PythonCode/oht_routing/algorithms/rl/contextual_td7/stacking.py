"""Shared temporal-stacking contracts for contextual TD7."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from numbers import Integral

import torch


def validate_stack_config(num_stacks: int, stack_interval: int) -> tuple[int, int]:
    values = {
        "num_stacks": num_stacks,
        "stack_interval": stack_interval,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer")
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    return int(num_stacks), int(stack_interval)


def stack_offsets(num_stacks: int, stack_interval: int) -> tuple[int, ...]:
    count, interval = validate_stack_config(num_stacks, stack_interval)
    return tuple(index * interval for index in range(count))


@dataclass(frozen=True)
class EncodedObservationStack:
    """One encoder call flattened over B*K, restored to [B, K, C]."""

    flat_encoding: object
    state: torch.Tensor


def _canonical_observation_stack(observation: tuple[torch.Tensor, ...]):
    if len(observation) != 6:
        raise ValueError("contextual observation must contain six tensors")
    center = observation[0]
    if center.ndim == 2:
        stacked = tuple(value.unsqueeze(1) for value in observation)
    elif center.ndim == 3:
        stacked = observation
    else:
        raise ValueError(
            "stacked center_local must have shape [B,D] or [B,K,D]"
        )
    batch, stacks = stacked[0].shape[:2]
    if any(value.shape[:2] != (batch, stacks) for value in stacked):
        raise ValueError("stacked contextual observation batch/time axes differ")
    return stacked, int(batch), int(stacks)


def encode_observation_stack(
    encoder,
    observation: tuple[torch.Tensor, ...],
    *,
    return_attention: bool = False,
) -> EncodedObservationStack:
    stacked, batch, stacks = _canonical_observation_stack(observation)
    flattened = tuple(
        value.reshape(batch * stacks, *value.shape[2:]) for value in stacked
    )
    encoding = encoder(*flattened, return_attention=return_attention)
    state = encoding.state.reshape(batch, stacks, -1)
    return EncodedObservationStack(flat_encoding=encoding, state=state)


def flatten_state_stack(state: torch.Tensor, expected_stacks: int) -> torch.Tensor:
    if state.ndim != 3 or state.shape[1] != int(expected_stacks):
        raise ValueError(
            "encoded state stack must have shape "
            f"[B,{int(expected_stacks)},C], got {tuple(state.shape)}"
        )
    return state.reshape(state.shape[0], -1)


def canonical_action_stack(
    action: torch.Tensor,
    *,
    num_stacks: int,
    action_dim: int,
) -> torch.Tensor:
    count = int(num_stacks)
    width = int(action_dim)
    if action.ndim == 2:
        if action.shape[1] == count * width:
            return action.reshape(action.shape[0], count, width)
        if count == 1 and action.shape[1] == width:
            return action.unsqueeze(1)
    elif action.ndim == 3 and tuple(action.shape[1:]) == (count, width):
        return action
    raise ValueError(
        "action stack must have shape "
        f"[B,{count * width}] or [B,{count},{width}], got {tuple(action.shape)}"
    )


def flatten_action_stack(
    action: torch.Tensor,
    *,
    num_stacks: int,
    action_dim: int,
) -> torch.Tensor:
    stacked = canonical_action_stack(
        action, num_stacks=num_stacks, action_dim=action_dim
    )
    return stacked.reshape(stacked.shape[0], -1)


def replace_current_action(
    history_action: torch.Tensor,
    current_action: torch.Tensor,
    *,
    num_stacks: int,
    action_dim: int,
) -> torch.Tensor:
    stacked = canonical_action_stack(
        history_action, num_stacks=num_stacks, action_dim=action_dim
    )
    if current_action.ndim != 2 or current_action.shape != (
        stacked.shape[0], int(action_dim)
    ):
        raise ValueError(
            "current action must have shape "
            f"[B,{int(action_dim)}], got {tuple(current_action.shape)}"
        )
    combined = torch.cat(
        (current_action.unsqueeze(1), stacked[:, 1:].detach()), dim=1
    )
    return combined.reshape(combined.shape[0], -1)


def interleave_state_action(
    state: torch.Tensor,
    action: torch.Tensor,
    *,
    num_stacks: int,
    context_dim: int,
    action_dim: int,
) -> torch.Tensor:
    count = int(num_stacks)
    if state.ndim != 2 or state.shape[1] != count * int(context_dim):
        raise ValueError(
            "flattened state stack must have shape "
            f"[B,{count * int(context_dim)}], got {tuple(state.shape)}"
        )
    stacked_action = canonical_action_stack(
        action, num_stacks=count, action_dim=action_dim
    )
    if state.shape[0] != stacked_action.shape[0]:
        raise ValueError("state/action batch sizes differ")
    stacked_state = state.reshape(state.shape[0], count, int(context_dim))
    return torch.cat((stacked_state, stacked_action), dim=-1).reshape(
        state.shape[0], -1
    )


class ContextualObservationHistory:
    """Episode-local online history with deterministic left padding."""

    def __init__(self, num_stacks: int = 1, stack_interval: int = 1):
        self.num_stacks, self.stack_interval = validate_stack_config(
            num_stacks, stack_interval
        )
        self.horizon = (self.num_stacks - 1) * self.stack_interval
        self._frames = deque(maxlen=self.horizon + 1)
        self._episode_id: int | None = None

    def clear(self) -> None:
        self._frames.clear()
        self._episode_id = None

    def append(self, observation, *, env_step: int, episode_id: int) -> None:
        step = int(env_step)
        episode = int(episode_id)
        if self._episode_id is None:
            self._episode_id = episode
        elif episode != self._episode_id:
            raise ValueError("observation history crossed an episode boundary")
        if self._frames and step != self._frames[-1][0] + 1:
            raise ValueError("observation history requires consecutive env steps")
        self._frames.append((step, observation))

    def frames(self) -> tuple[object, ...]:
        if not self._frames:
            raise ValueError("observation history is empty")
        current_step = self._frames[-1][0]
        first_step, first_observation = self._frames[0]
        by_step = dict(self._frames)
        result = []
        for offset in stack_offsets(self.num_stacks, self.stack_interval):
            requested = current_step - offset
            if requested < first_step:
                result.append(first_observation)
            elif requested in by_step:
                result.append(by_step[requested])
            else:
                raise ValueError(
                    "observation history is missing an in-range env step: "
                    f"requested={requested}, available={first_step}..{current_step}"
                )
        return tuple(result)

    @property
    def size(self) -> int:
        return len(self._frames)

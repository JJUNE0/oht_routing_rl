"""Immutable data contracts for step-snapshot contextual replay."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ContextualStepSnapshot:
    physical_local_state: np.ndarray
    global_state: np.ndarray
    critic_total_tat: np.ndarray
    previous_applied_action: np.ndarray
    policy_action: np.ndarray
    applied_action: np.ndarray
    reward: np.ndarray
    next_physical_local_state: np.ndarray
    next_global_state: np.ndarray
    next_critic_total_tat: np.ndarray
    next_previous_applied_action: np.ndarray
    done: bool
    env_step: int
    next_env_step: int
    episode_id: int
    topology_hash: str
    mapping_hash: str
    version: str
    reward_version: str


@dataclass(frozen=True)
class ReplaySampleKey:
    step_slot: int
    generation: int
    controlled_row: int


@dataclass(frozen=True)
class ContextualReplayBatch:
    center_local: torch.Tensor
    incoming_local: torch.Tensor
    outgoing_local: torch.Tensor
    center_rail_index: torch.Tensor
    incoming_rail_indices: torch.Tensor
    outgoing_rail_indices: torch.Tensor
    incoming_relation: torch.Tensor
    outgoing_relation: torch.Tensor
    global_state: torch.Tensor
    critic_total_tat: torch.Tensor
    previous_applied_action: torch.Tensor
    policy_action: torch.Tensor
    applied_action: torch.Tensor
    reward: torch.Tensor
    next_center_local: torch.Tensor
    next_incoming_local: torch.Tensor
    next_outgoing_local: torch.Tensor
    next_incoming_relation: torch.Tensor
    next_outgoing_relation: torch.Tensor
    next_global_state: torch.Tensor
    next_critic_total_tat: torch.Tensor
    next_previous_applied_action: torch.Tensor
    next_applied_action: torch.Tensor
    done: torch.Tensor
    controlled_rail_id: torch.Tensor
    env_step: torch.Tensor
    episode_id: torch.Tensor
    sample_keys: tuple[ReplaySampleKey, ...]

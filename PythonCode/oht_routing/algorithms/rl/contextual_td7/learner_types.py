from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextualLearnerUpdate:
    diagnostics: dict[str, float]
    actor_updated: bool
    target_updated: bool

import time

import numpy as np
import torch

from .numerics import NumericalIntegrityError, assert_finite_arrays


class RegionReplayBuffer:
    def __init__(self, capacity, device="cpu", prioritized=True, min_priority=1.0, lap_alpha=0.4):
        self.type = "region_rollout"
        self.capacity = int(capacity)
        self.device = device
        self.prioritized = bool(prioritized)
        self.min_priority = float(min_priority)
        self.lap_alpha = float(lap_alpha)
        self.storage = [None] * self.capacity
        self.priority = np.zeros(self.capacity, dtype=np.float32)
        self.max_priority = 1.0
        self.position = 0
        self.size = 0
        self.ind = None
        self.last_sample_info = {}

    def clear(self):
        self.storage = [None] * self.capacity
        self.priority.fill(0.0)
        self.max_priority = 1.0
        self.position = 0
        self.size = 0
        self.ind = None

    def push(self, rail_feat, action, reward, next_rail_feat, done, mask, next_mask, context, next_context):
        rail_feat = np.asarray(rail_feat, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        next_rail_feat = np.asarray(next_rail_feat, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        next_mask = np.asarray(next_mask, dtype=bool)
        context = np.asarray(context, dtype=np.float32)
        next_context = np.asarray(next_context, dtype=np.float32)
        if rail_feat.ndim != 2 or action.ndim != 2 or next_rail_feat.ndim != 2:
            raise ValueError("region transition expects [N, dim] rail/action arrays")
        if len(rail_feat) == 0:
            raise ValueError("empty region transition is not allowed")
        if len(rail_feat) != len(action):
            raise ValueError("rail/action lengths must match")
        if len(rail_feat) != len(mask):
            raise ValueError("rail/mask lengths must match")
        if len(next_rail_feat) != len(next_mask):
            raise ValueError("next_rail/next_mask lengths must match")
        if len(next_rail_feat) == 0:
            raise ValueError("empty next region transition is not allowed")

        assert_finite_arrays(
            "replay/push",
            (
                ("rail_feat", rail_feat),
                ("action", action),
                ("reward", np.asarray(reward, dtype=np.float32)),
                ("next_rail_feat", next_rail_feat),
                ("done", np.asarray(done, dtype=np.float32)),
                ("context", context),
                ("next_context", next_context),
            ),
            context=f"position={self.position}, buffer_size={self.size}",
        )

        self.storage[self.position] = {
            "rail_feat": rail_feat.copy(),
            "action": action.copy(),
            "reward": float(reward),
            "next_rail_feat": next_rail_feat.copy(),
            "done": float(done),
            "mask": mask.copy(),
            "next_mask": next_mask.copy(),
            "context": context.copy(),
            "next_context": next_context.copy(),
        }
        self.priority[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if self.size <= 0:
            raise ValueError("cannot sample from an empty RegionReplayBuffer")
        t0 = time.perf_counter()
        batch_size = int(batch_size)
        if self.prioritized:
            p = self.priority[: self.size].astype(np.float64)
            assert_finite_arrays(
                "replay/sample_priority",
                (("priority", p),),
                context=f"buffer_size={self.size}",
            )
            total = p.sum()
            if total <= 0.0:
                raise NumericalIntegrityError(
                    f"[replay/sample_priority] non-positive priority sum={total} "
                    f"| buffer_size={self.size}"
                )
            probs = p / total
            self.ind = np.random.choice(self.size, size=batch_size, replace=True, p=probs)
        else:
            self.ind = np.random.randint(0, self.size, size=batch_size)

        items = [self.storage[int(i)] for i in self.ind]
        lmax = max(
            max(int(x["rail_feat"].shape[0]), int(x["next_rail_feat"].shape[0]))
            for x in items
        )
        rail_dim = int(items[0]["rail_feat"].shape[1])
        action_dim = int(items[0]["action"].shape[1])
        context_dim = int(items[0]["context"].shape[0])

        rail = np.zeros((batch_size, lmax, rail_dim), dtype=np.float32)
        action = np.zeros((batch_size, lmax, action_dim), dtype=np.float32)
        next_rail = np.zeros((batch_size, lmax, rail_dim), dtype=np.float32)
        mask = np.zeros((batch_size, lmax), dtype=bool)
        next_mask = np.zeros((batch_size, lmax), dtype=bool)
        context = np.zeros((batch_size, context_dim), dtype=np.float32)
        next_context = np.zeros((batch_size, context_dim), dtype=np.float32)
        reward = np.zeros((batch_size, 1), dtype=np.float32)
        done = np.zeros((batch_size, 1), dtype=np.float32)

        for b, item in enumerate(items):
            n = int(item["rail_feat"].shape[0])
            n_next = int(item["next_rail_feat"].shape[0])
            rail[b, :n] = item["rail_feat"]
            action[b, :n] = item["action"]
            mask[b, :n] = item["mask"]
            next_rail[b, :n_next] = item["next_rail_feat"]
            next_mask[b, :n_next] = item["next_mask"]
            context[b] = item["context"]
            next_context[b] = item["next_context"]
            reward[b, 0] = item["reward"]
            done[b, 0] = item["done"]

        assert_finite_arrays(
            "replay/sample_batch",
            (
                ("rail_feat", rail),
                ("action", action),
                ("reward", reward),
                ("next_rail_feat", next_rail),
                ("done", done),
                ("context", context),
                ("next_context", next_context),
            ),
            context=f"sample_indices={self.ind[:8].tolist()}",
        )
        if not bool(mask.any(axis=1).all()) or not bool(next_mask.any(axis=1).all()):
            raise NumericalIntegrityError(
                "[replay/sample_batch] sampled transition has no valid tokens "
                f"| sample_indices={self.ind[:8].tolist()}"
            )

        device = self.device
        out = {
            "rail_feat": torch.as_tensor(rail, dtype=torch.float32, device=device),
            "action": torch.as_tensor(action, dtype=torch.float32, device=device),
            "reward": torch.as_tensor(reward, dtype=torch.float32, device=device),
            "next_rail_feat": torch.as_tensor(next_rail, dtype=torch.float32, device=device),
            "done": torch.as_tensor(done, dtype=torch.float32, device=device),
            "mask": torch.as_tensor(mask, dtype=torch.bool, device=device),
            "next_mask": torch.as_tensor(next_mask, dtype=torch.bool, device=device),
            "context": torch.as_tensor(context, dtype=torch.float32, device=device),
            "next_context": torch.as_tensor(next_context, dtype=torch.float32, device=device),
        }
        self.last_sample_info = {
            "learner/collate_ms": (time.perf_counter() - t0) * 1000.0,
            "learner/batch_size": float(batch_size),
            "learner/lmax": float(lmax),
            "region_mask/valid_tokens_mean": float(mask.mean()),
            "region_mask/next_valid_tokens_mean": float(next_mask.mean()),
        }
        return out

    def update_priority(self, priority):
        if self.ind is None:
            return
        p = np.asarray(priority.detach().cpu().reshape(-1), dtype=np.float32)
        assert_finite_arrays(
            "replay/update_priority",
            (("priority", p),),
            context=f"sample_indices={self.ind[:8].tolist()}",
        )
        p = np.maximum(p, self.min_priority) ** self.lap_alpha
        self.priority[self.ind] = p
        if len(p):
            self.max_priority = max(self.max_priority, float(np.max(p)))

    def diagnostic_priority_snapshot(self):
        """Return replay priorities and the exact probabilities of the last sample."""
        active = self.priority[: self.size].astype(np.float32, copy=True)
        result = {
            "active_priority": active,
            "max_priority": float(self.max_priority),
        }
        if self.ind is None:
            return result

        indices = np.asarray(self.ind, dtype=np.int64).copy()
        sampled = active[indices]
        total = float(active.astype(np.float64).sum())
        probabilities = (
            sampled.astype(np.float64) / total
            if total > 0.0
            else np.full(sampled.shape, np.nan, dtype=np.float64)
        )
        result.update(
            {
                "sample_indices": indices,
                "sample_priority": sampled,
                "sample_probability": probabilities,
                "priority_sum_float64": total,
            }
        )
        return result

    def reset_max_priority(self):
        if self.size > 0:
            active = self.priority[: self.size]
            assert_finite_arrays(
                "replay/reset_max_priority",
                (("priority", active),),
                context=f"buffer_size={self.size}",
            )
            self.max_priority = float(np.max(active))

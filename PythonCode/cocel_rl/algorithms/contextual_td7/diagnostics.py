import torch

from .networks import ActorOutput, ContextualEncoding, TwinCriticOutput


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def _attention_stats(prefix: str, weights: torch.Tensor) -> dict[str, float]:
    probabilities = weights.clamp_min(torch.finfo(weights.dtype).tiny)
    entropy = -(probabilities * probabilities.log()).sum(dim=-1).mean()
    return {
        f"attention/{prefix}_entropy": _scalar(entropy),
        f"attention/{prefix}_max_weight": _scalar(weights.max(dim=-1).values.mean()),
    }


def encoding_diagnostics(encoding: ContextualEncoding) -> dict[str, float]:
    if encoding.incoming_attention is None or encoding.outgoing_attention is None:
        raise ValueError("encoding diagnostics require return_attention=True")
    result = {}
    result.update(_attention_stats("incoming", encoding.incoming_attention))
    result.update(_attention_stats("outgoing", encoding.outgoing_attention))
    result.update(
        {
            "attention/incoming_context_norm": _scalar(
                encoding.incoming_context.norm(dim=-1).mean()
            ),
            "attention/outgoing_context_norm": _scalar(
                encoding.outgoing_context.norm(dim=-1).mean()
            ),
            "embedding/center_norm": _scalar(
                encoding.center_embedding.norm(dim=-1).mean()
            ),
            "embedding/global_norm": _scalar(
                encoding.global_embedding.norm(dim=-1).mean()
            ),
            "embedding/fused_norm": _scalar(
                encoding.state.norm(dim=-1).mean()
            ),
        }
    )
    return result


def actor_diagnostics(output: ActorOutput) -> dict[str, float]:
    action = output.action
    pre_tanh = output.pre_tanh
    return {
        "action/pre_tanh_mean": _scalar(pre_tanh.mean()),
        "action/pre_tanh_std": _scalar(pre_tanh.std(unbiased=False)),
        "action/mean": _scalar(action.mean()),
        "action/std": _scalar(action.std(unbiased=False)),
        "action/saturation_ratio": _scalar((action.abs() >= 0.999).float().mean()),
    }


def critic_diagnostics(output: TwinCriticOutput) -> dict[str, float]:
    combined = torch.cat((output.q1, output.q2), dim=0)
    return {
        "critic/q1_mean": _scalar(output.q1.mean()),
        "critic/q2_mean": _scalar(output.q2.mean()),
        "critic/q_range": _scalar(combined.max() - combined.min()),
    }

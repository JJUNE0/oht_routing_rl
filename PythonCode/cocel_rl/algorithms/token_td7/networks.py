import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def avg_l1_norm(x, eps=1e-8):
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


class AvgL1Norm(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return avg_l1_norm(x, eps=self.eps)


def masked_mean(x, mask):
    m = mask.unsqueeze(-1).to(dtype=x.dtype)
    denom = m.sum(dim=1).clamp(min=1.0)
    return (x * m).sum(dim=1) / denom


def masked_max(x, mask):
    m = mask.unsqueeze(-1).bool()
    fill = torch.finfo(x.dtype).min
    y = x.masked_fill(~m, fill)
    out = y.max(dim=1).values
    return out


class ContextConcatRailAttention(nn.Module):
    def __init__(self, rail_dim, context_dim, embed_dim=64, num_heads=4):
        super().__init__()
        self.rail_enc = nn.Sequential(
            nn.Linear(rail_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.ctx_enc = nn.Sequential(
            nn.Linear(context_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # Non-affine normalization keeps attention inputs bounded without a
        # learned LayerNorm gain being able to undo the protection.
        self.rail_attn_norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        # This normalized context is also returned to SALE, actor, and critic.
        # Previously only the attention-local copy was normalized while raw C
        # could dominate every downstream token through broadcast concatenation.
        self.context_norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.q_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.k_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
            )
        self.num_heads = int(num_heads)
        self.head_dim = embed_dim // num_heads
        self.attention_scale = self.head_dim ** -0.5
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def _split_heads(self, x):
        batch, length, _ = x.shape
        return x.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    @staticmethod
    def _bounded_head_rms(x):
        """Limit each Q/K head RMS to one without amplifying small vectors."""
        rms = x.float().square().mean(dim=-1, keepdim=True).sqrt()
        return x / rms.clamp(min=1.0).to(dtype=x.dtype)

    def _attention(self, rail_encoded, context_encoded, mask, diagnostics=False):
        batch, length, _ = rail_encoded.shape
        rail_for_attention = self.rail_attn_norm(rail_encoded)
        context_tokens = context_encoded.unsqueeze(1).expand(batch, length, -1)
        rail_context = torch.cat([rail_for_attention, context_tokens], dim=-1)

        query_raw = self.q_proj(rail_context)
        key_raw = self.k_proj(rail_context)
        value_raw = self.v_proj(rail_for_attention)
        query = self._bounded_head_rms(self._split_heads(query_raw))
        key = self._bounded_head_rms(self._split_heads(key_raw))
        value = self._split_heads(value_raw)

        # Q/K/V have already been projected above, so compute attention
        # directly instead of projecting them a second time in MHA.
        raw_logits = torch.matmul(query, key.transpose(-2, -1)) * self.attention_scale
        key_mask = mask.bool().unsqueeze(1).unsqueeze(2)
        logits = raw_logits.masked_fill(~key_mask, torch.finfo(raw_logits.dtype).min)
        weights = torch.softmax(logits.float(), dim=-1).to(dtype=value.dtype)
        attended = torch.matmul(weights, value)
        attended = attended.transpose(1, 2).contiguous().reshape(batch, length, -1)
        output = self.out_proj(attended)

        # A key-padding mask does not mask padded queries, so do that explicitly.
        output = output * mask.unsqueeze(-1).to(dtype=output.dtype)
        if not diagnostics:
            return output
        return output, {
            "rail_attention_input": rail_for_attention,
            "context_attention_input": context_encoded,
            "attention_q_raw": query_raw,
            "attention_k_raw": key_raw,
            "attention_q": query.transpose(1, 2).contiguous().reshape(batch, length, -1),
            "attention_k": key.transpose(1, 2).contiguous().reshape(batch, length, -1),
            "attention_v": value_raw,
            # Store pre-mask logits so artifact ranges are not dominated by the
            # finite sentinel used for padded keys.
            "attention_logits": raw_logits,
            "attention_weights": weights,
        }

    def forward(self, rail_feat, context_feat, mask):
        H = self.rail_enc(rail_feat)
        C = self.context_norm(self.ctx_enc(context_feat))
        A = self._attention(H, C, mask)
        H = self.norm1(H + A)
        H = self.norm2(H + self.ffn(H))
        return H * mask.unsqueeze(-1).to(dtype=H.dtype), C

    @torch.no_grad()
    def diagnostic_activations(self, rail_feat, context_feat, mask):
        """Return the attention path tensors for a small forensic sub-batch."""
        rail_encoded = self.rail_enc(rail_feat)
        context_encoded_raw = self.ctx_enc(context_feat)
        context_encoded = self.context_norm(context_encoded_raw)
        attention, attention_diagnostics = self._attention(
            rail_encoded, context_encoded, mask, diagnostics=True
        )
        post_attention = self.norm1(rail_encoded + attention)
        feed_forward = self.ffn(post_attention)
        output = self.norm2(post_attention + feed_forward)
        output = output * mask.unsqueeze(-1).to(dtype=output.dtype)
        values = {
            "rail_encoded": rail_encoded,
            "context_encoded_raw": context_encoded_raw,
            "context_encoded": context_encoded,
            "attention_output": attention,
            "post_attention_norm": post_attention,
            "feed_forward": feed_forward,
            "encoder_output": output,
        }
        values.update(attention_diagnostics)
        return values


class RegionEncoder(nn.Module):
    def __init__(self, rail_dim, action_dim, context_dim, zs_dim=256, hdim=256, embed_dim=64, num_heads=4):
        super().__init__()
        self.backbone = ContextConcatRailAttention(
            rail_dim=rail_dim,
            context_dim=context_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
        self.zs_proj = nn.Sequential(
            nn.Linear(embed_dim * 3, hdim),
            nn.ELU(),
            nn.Linear(hdim, hdim),
            nn.ELU(),
            nn.Linear(hdim, zs_dim),
        )
        self.action_enc = nn.Sequential(
            nn.Linear(embed_dim + action_dim, hdim),
            nn.ELU(),
            nn.Linear(hdim, hdim),
            nn.ELU(),
        )
        self.zsa_proj = nn.Sequential(
            nn.Linear(hdim * 2 + embed_dim + zs_dim, hdim),
            nn.ELU(),
            nn.Linear(hdim, hdim),
            nn.ELU(),
            nn.Linear(hdim, zs_dim),
        )

    def token_state(self, rail_feat, context_feat, mask):
        H, C = self.backbone(rail_feat, context_feat, mask)
        pooled = torch.cat([masked_mean(H, mask), masked_max(H, mask), C], dim=-1)
        zs = avg_l1_norm(self.zs_proj(pooled))
        return H, C, zs

    def zs(self, rail_feat, context_feat, mask):
        _, _, zs = self.token_state(rail_feat, context_feat, mask)
        return zs

    def zsa_from_encoded(self, H, C, zs, action, mask):
        """SALE g(f(s), a): build zsa from one coherent encoded state."""
        HA = self.action_enc(torch.cat([H, action], dim=-1))
        HA = HA * mask.unsqueeze(-1).to(dtype=HA.dtype)
        pooled = torch.cat(
            [masked_mean(HA, mask), masked_max(HA, mask), C, zs],
            dim=-1,
        )
        return self.zsa_proj(pooled)

    def zsa(self, rail_feat, action, context_feat, mask):
        H, C, zs = self.token_state(rail_feat, context_feat, mask)
        return self.zsa_from_encoded(H, C, zs, action, mask)

    @torch.no_grad()
    def diagnostic_activations(self, rail_feat, action, context_feat, mask):
        """Capture encoder intermediates without retaining an autograd graph."""
        values = self.backbone.diagnostic_activations(rail_feat, context_feat, mask)
        hidden = values["encoder_output"]
        context_encoded = values["context_encoded"]

        state_pooled = torch.cat(
            [masked_mean(hidden, mask), masked_max(hidden, mask), context_encoded],
            dim=-1,
        )
        zs_fc0 = self.zs_proj[0](state_pooled)
        zs_elu0 = self.zs_proj[1](zs_fc0)
        zs_fc1 = self.zs_proj[2](zs_elu0)
        zs_elu1 = self.zs_proj[3](zs_fc1)
        zs_raw = self.zs_proj[4](zs_elu1)
        zs = avg_l1_norm(zs_raw)

        action_input = torch.cat([hidden, action], dim=-1)
        action_fc0 = self.action_enc[0](action_input)
        action_elu0 = self.action_enc[1](action_fc0)
        action_fc1 = self.action_enc[2](action_elu0)
        action_hidden = self.action_enc[3](action_fc1)
        action_hidden = action_hidden * mask.unsqueeze(-1).to(dtype=action_hidden.dtype)
        pooled = torch.cat(
            [
                masked_mean(action_hidden, mask),
                masked_max(action_hidden, mask),
                context_encoded,
                zs,
            ],
            dim=-1,
        )
        zsa_fc0 = self.zsa_proj[0](pooled)
        zsa_elu0 = self.zsa_proj[1](zsa_fc0)
        zsa_fc1 = self.zsa_proj[2](zsa_elu0)
        zsa_elu1 = self.zsa_proj[3](zsa_fc1)
        prediction = self.zsa_proj[4](zsa_elu1)
        values.update(
            {
                "state_pooled": state_pooled,
                "zs_fc0": zs_fc0,
                "zs_elu0": zs_elu0,
                "zs_fc1": zs_fc1,
                "zs_elu1": zs_elu1,
                "zs_raw": zs_raw,
                "zs": zs,
                "action_input": action_input,
                "action_fc0": action_fc0,
                "action_elu0": action_elu0,
                "action_fc1": action_fc1,
                "action_hidden": action_hidden,
                "zsa_pooled": pooled,
                "zsa_fc0": zsa_fc0,
                "zsa_elu0": zsa_elu0,
                "zsa_fc1": zsa_fc1,
                "zsa_elu1": zsa_elu1,
                "zsa_prediction": prediction,
            }
        )
        return values


class RegionPolicyHead(nn.Module):
    def __init__(self, embed_dim, zs_dim, action_dim, hdim=256, max_action=1.0):
        super().__init__()
        self.max_action = max_action
        self.embed_dim = embed_dim
        self.zs_dim = zs_dim
        self.hdim = hdim
        # Faithful token analogue of TD7 PolicyMLP:
        #   AvgL1Norm(l0(state)) -> concat(zs) -> relu(l1) -> relu(l2) -> l3
        self.state_proj = nn.Linear(embed_dim * 2, hdim)
        self.joint_proj = nn.Linear(hdim + zs_dim, hdim)
        self.hidden_proj = nn.Linear(hdim, hdim)
        self.out_proj = nn.Linear(hdim, action_dim)

    def _stages(self, H, C, zs):
        batch, length, _ = H.shape
        C_tok = C.unsqueeze(1).expand(batch, length, -1)
        zs_tok = zs.unsqueeze(1).expand(batch, length, -1)
        state_input = torch.cat([H, C_tok], dim=-1)
        state_raw = self.state_proj(state_input)
        state_hidden = avg_l1_norm(state_raw)
        joint_input = torch.cat([state_hidden, zs_tok], dim=-1)
        joint_raw = self.joint_proj(joint_input)
        joint_hidden = torch.relu(joint_raw)
        hidden_raw = self.hidden_proj(joint_hidden)
        hidden = torch.relu(hidden_raw)
        pre_tanh = self.out_proj(hidden)
        return {
            "state_input": state_input,
            "state_raw": state_raw,
            "state_hidden": state_hidden,
            "joint_input": joint_input,
            "joint_raw": joint_raw,
            "joint_hidden": joint_hidden,
            "hidden_raw": hidden_raw,
            "hidden": hidden,
            "pre_tanh": pre_tanh,
        }

    def _pre_tanh(self, H, C, zs):
        return self._stages(H, C, zs)["pre_tanh"]

    def _pre_tanh_contrib(self, H, C, zs):
        """Return comparable contributions at each of the two TD7 input stages."""
        batch, length, _ = H.shape
        d = self.embed_dim
        C_tok = C.unsqueeze(1).expand(batch, length, -1)
        zs_tok = zs.unsqueeze(1).expand(batch, length, -1)
        state_weight = self.state_proj.weight
        state_H = H @ state_weight[:, :d].T
        state_C = C_tok @ state_weight[:, d:].T
        state_hidden = avg_l1_norm(state_H + state_C + self.state_proj.bias)
        joint_weight = self.joint_proj.weight
        joint_state = state_hidden @ joint_weight[:, :self.hdim].T
        joint_zs = zs_tok @ joint_weight[:, self.hdim:].T
        return {
            "state_H": state_H,
            "state_C": state_C,
            "joint_state": joint_state,
            "joint_zs": joint_zs,
        }

    def weight_block_norms(self):
        """Report state projection and post-normalization joint blocks separately."""
        d = self.embed_dim
        state_weight = self.state_proj.weight.detach()
        joint_weight = self.joint_proj.weight.detach()
        blocks = {
            "w0_H": state_weight[:, :d],
            "w0_C": state_weight[:, d:],
            "joint_state": joint_weight[:, :self.hdim],
            "joint_zs": joint_weight[:, self.hdim:],
        }
        out = {}
        for name, weight in blocks.items():
            family, block = name.split("_", maxsplit=1)
            out[f"actor/{family}_norm_{block}"] = float(weight.norm().cpu())
            out[f"actor/{family}_absmean_{block}"] = float(weight.abs().mean().cpu())
        return out

    def _layer_scale_probe(self, H, C, zs):
        """Report every stage, including the restored TD7 AvgL1Norm boundary."""
        stages = self._stages(H, C, zs)
        return {
            f"actor/scale_{name}_abs_mean": float(value.abs().mean().detach().cpu())
            for name, value in stages.items()
        }

    def bias_norms(self):
        """Report every policy Linear independently."""
        linears = {
            "state_proj": self.state_proj,
            "joint_proj": self.joint_proj,
            "hidden_proj": self.hidden_proj,
            "out_proj": self.out_proj,
        }
        out = {}
        for name, lin in linears.items():
            out[f"actor/{name}_bias_absmean"] = float(lin.bias.detach().abs().mean().cpu())
            out[f"actor/{name}_weight_absmean"] = float(lin.weight.detach().abs().mean().cpu())
        return out

    def forward(self, H, C, zs, mask):
        action = self.max_action * torch.tanh(self._pre_tanh(H, C, zs))
        return action * mask.unsqueeze(-1).to(dtype=action.dtype)


class RegionTD7Actor(nn.Module):
    def __init__(
        self,
        rail_dim,
        action_dim,
        action_bound,
        context_dim,
        zs_dim=256,
        hdim=256,
        embed_dim=64,
        num_heads=4,
        exploration_noise=0.1,
    ):
        super().__init__()
        self.action_bound = action_bound
        self.max_action = float(max(abs(action_bound[0]), abs(action_bound[1])))
        self.exploration_noise = exploration_noise
        self.rail_dim = rail_dim
        self.act_dim = action_dim
        self.context_dim = context_dim
        self.zs_dim = zs_dim
        self.hidden_dims = [hdim, hdim]
        self.encoder_hidden_dims = [hdim, hdim]
        self.activation_fc_name = "relu"

        self.encoder = RegionEncoder(
            rail_dim=rail_dim,
            action_dim=action_dim,
            context_dim=context_dim,
            zs_dim=zs_dim,
            hdim=hdim,
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
        self.fixed_encoder = copy.deepcopy(self.encoder)
        for p in self.fixed_encoder.parameters():
            p.requires_grad_(False)
        self.mlp = RegionPolicyHead(
            embed_dim=embed_dim,
            zs_dim=zs_dim,
            action_dim=action_dim,
            hdim=hdim,
            max_action=self.max_action,
        )

    def policy_action(self, rail_feat, context_feat, mask):
        H, C, zs = self.fixed_encoder.token_state(rail_feat, context_feat, mask)
        return self.mlp(H, C, zs, mask)

    def pre_tanh_action(self, rail_feat, context_feat, mask):
        """진단용: 실제 env action과 똑같은 경로(fixed_encoder + mlp)로 tanh 직전 raw 값을 반환.
        dead-ReLU(입력 무관 상수, 작은 절대값)와 tanh saturation(입력별로 다르지만 절대값이 큼)을
        구분하기 위한 영구 probe — actor가 또 36k 정체에 빠지면 그 구간의 분포가 여기 찍힌다."""
        H, C, zs = self.fixed_encoder.token_state(rail_feat, context_feat, mask)
        return self.mlp._pre_tanh(H, C, zs)

    def policy_head_weight_norms(self):
        """진단용: 입력 데이터 없이 가중치만 보는 진단(forward 불필요, 비용 거의 0)."""
        return self.mlp.weight_block_norms()

    def pre_tanh_contrib_action(self, rail_feat, context_feat, mask):
        """Separate H/C at state projection and normalized-state/zs at joint projection."""
        H, C, zs = self.fixed_encoder.token_state(rail_feat, context_feat, mask)
        return self.mlp._pre_tanh_contrib(H, C, zs)

    def layer_scale_probe_action(self, rail_feat, context_feat, mask):
        """진단용: 실제 action 경로(fixed_encoder+mlp)로 net 내부 stage별 abs_mean을 본다."""
        H, C, zs = self.fixed_encoder.token_state(rail_feat, context_feat, mask)
        return self.mlp._layer_scale_probe(H, C, zs)

    def policy_head_bias_norms(self):
        """진단용(가중치만, forward 불필요)."""
        return self.mlp.bias_norms()

    def forward(self, rail_feat, context_feat, mask):
        return self.policy_action(rail_feat, context_feat, mask)

    def get_action(self, rail_feat, context_feat, mask, eval=False):
        device = next(self.parameters()).device
        rail = torch.as_tensor(rail_feat, dtype=torch.float32, device=device)
        ctx = torch.as_tensor(context_feat, dtype=torch.float32, device=device)
        m = torch.as_tensor(mask, dtype=torch.bool, device=device)
        squeeze = False
        if rail.dim() == 2:
            rail = rail.unsqueeze(0)
            ctx = ctx.unsqueeze(0)
            m = m.unsqueeze(0)
            squeeze = True
        with torch.no_grad():
            action = self.forward(rail, ctx, m)
        a = action.cpu().numpy()
        if not eval:
            noise = np.random.normal(0.0, self.max_action * self.exploration_noise, size=a.shape)
            a = np.clip(a + noise, self.action_bound[0], self.action_bound[1])
        if squeeze:
            a = a[0]
        return a.astype(np.float32), None


class RegionQHead(nn.Module):
    """Faithful token analogue of one TD7 critic head."""

    def __init__(self, action_dim, zs_dim, hdim, embed_dim):
        super().__init__()
        self.sa_proj = nn.Linear(embed_dim * 2 + action_dim, hdim)
        self.hidden1 = nn.Linear(hdim + zs_dim * 2, hdim)
        self.hidden2 = nn.Linear(hdim, hdim)
        self.out = nn.Linear(hdim, 1)

    def forward(self, state_action, zs_tok, zsa_tok):
        sa = avg_l1_norm(self.sa_proj(state_action))
        hidden = F.elu(self.hidden1(torch.cat([sa, zs_tok, zsa_tok], dim=-1)))
        hidden = F.elu(self.hidden2(hidden))
        return self.out(hidden)


class RegionTD7Critic(nn.Module):
    def __init__(self, action_dim, zs_dim=256, hdim=256, embed_dim=64):
        super().__init__()
        self.q1_token = RegionQHead(action_dim, zs_dim, hdim, embed_dim)
        self.q2_token = RegionQHead(action_dim, zs_dim, hdim, embed_dim)

    def _token_q(self, H, C, zs, zsa, action, mask, token_net):
        B, L, _ = H.shape
        C_tok = C.unsqueeze(1).expand(B, L, C.shape[-1])
        zs_tok = zs.unsqueeze(1).expand(B, L, zs.shape[-1])
        zsa_tok = zsa.unsqueeze(1).expand(B, L, zsa.shape[-1])
        state_action = torch.cat([H, C_tok, action], dim=-1)
        q_tok = token_net(state_action, zs_tok, zsa_tok)

        token_mask = mask.unsqueeze(-1).to(dtype=q_tok.dtype)
        q_tok = q_tok * token_mask
        n_valid = token_mask.sum(dim=1).clamp(min=1.0)
        # Region rewards are the mean of rail rewards, so Q uses the matching
        # token mean rather than introducing a sqrt(region_size) scale bias.
        q = q_tok.sum(dim=1) / n_valid
        return q, q_tok, n_valid

    def forward(self, H, C, zs, zsa, action, mask):
        q1, _, _ = self._token_q(H, C, zs, zsa, action, mask, self.q1_token)
        q2, _, _ = self._token_q(H, C, zs, zsa, action, mask, self.q2_token)
        return torch.cat([q1, q2], dim=1)

    @torch.no_grad()
    def diagnostics(self, H, C, zs, zsa, action, mask):
        q1, q1_tok, n_valid = self._token_q(H, C, zs, zsa, action, mask, self.q1_token)
        q2, q2_tok, _ = self._token_q(H, C, zs, zsa, action, mask, self.q2_token)

        token_mask = mask.unsqueeze(-1).to(dtype=torch.bool)
        valid_q1_tok = q1_tok[token_mask]
        valid_q2_tok = q2_tok[token_mask]
        if valid_q1_tok.numel() == 0:
            return {}

        q = torch.cat([q1, q2], dim=1)
        return {
            "critic/q_token_q1_mean": float(valid_q1_tok.mean().detach().cpu()),
            "critic/q_token_q1_std": float(valid_q1_tok.std(unbiased=False).detach().cpu()),
            "critic/q_token_q1_abs_mean": float(valid_q1_tok.abs().mean().detach().cpu()),
            "critic/q_token_q2_mean": float(valid_q2_tok.mean().detach().cpu()),
            "critic/q_token_q2_std": float(valid_q2_tok.std(unbiased=False).detach().cpu()),
            "critic/q_token_q2_abs_mean": float(valid_q2_tok.abs().mean().detach().cpu()),
            "critic/q_agg_mean": float(q.mean().detach().cpu()),
            "critic/q_agg_std": float(q.std(unbiased=False).detach().cpu()),
            "critic/q_valid_tokens_mean": float(n_valid.mean().detach().cpu()),
        }

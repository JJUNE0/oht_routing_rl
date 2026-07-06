import copy

import numpy as np
import torch
import torch.nn as nn

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
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


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
        self.q_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.k_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, rail_feat, context_feat, mask):
        H = self.rail_enc(rail_feat)
        C = self.ctx_enc(context_feat)
        B, L, _ = H.shape
        C_tok = C.unsqueeze(1).expand(B, L, C.shape[-1])
        HC = torch.cat([H, C_tok], dim=-1)

        Q = self.q_proj(HC)
        K = self.k_proj(HC)
        V = self.v_proj(H)
        key_padding_mask = ~mask.bool()

        A, _ = self.attn(
            Q,
            K,
            V,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        H = self.norm1(H + A)
        H = self.norm2(H + self.ffn(H))
        return H * mask.unsqueeze(-1).to(dtype=H.dtype), C


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
            nn.Linear(hdim * 2 + embed_dim, hdim),
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

    def zsa(self, rail_feat, action, context_feat, mask):
        H, C = self.backbone(rail_feat, context_feat, mask)
        HA = self.action_enc(torch.cat([H, action], dim=-1))
        HA = HA * mask.unsqueeze(-1).to(dtype=HA.dtype)
        pooled = torch.cat([masked_mean(HA, mask), masked_max(HA, mask), C], dim=-1)
        return self.zsa_proj(pooled)


class RegionPolicyHead(nn.Module):
    def __init__(self, embed_dim, zs_dim, action_dim, hdim=256, max_action=1.0):
        super().__init__()
        self.max_action = max_action
        self.embed_dim = embed_dim
        self.zs_dim = zs_dim
        self.hdim = hdim
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2 + zs_dim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, action_dim),
        )

    def _pre_tanh(self, H, C, zs):
        B, L, _ = H.shape
        C_tok = C.unsqueeze(1).expand(B, L, C.shape[-1])
        zs_tok = zs.unsqueeze(1).expand(B, L, zs.shape[-1])
        X = torch.cat([H, C_tok, zs_tok], dim=-1)
        return self.net(X)  # fc0→relu0→fc1→relu1→LayerNorm→fc2

    def _pre_tanh_contrib(self, H, C, zs):
        """진단용: 첫 Linear 층 출력을 H/C/zs 입력 블록별 부분합으로 분해.
        입력 크기(abs mean)는 그 항이 학습된 가중치를 거쳐 실제로 얼마나 기여하는지를
        보장하지 않으므로, W[:, block] 슬라이스를 직접 곱해 '기여'를 따로 측정한다."""
        B, L, _ = H.shape
        d = self.embed_dim
        C_tok = C.unsqueeze(1).expand(B, L, C.shape[-1])
        zs_tok = zs.unsqueeze(1).expand(B, L, zs.shape[-1])
        W = self.net[0].weight  # [hdim, 2*embed_dim+zs_dim]
        contrib_H = H @ W[:, :d].T
        contrib_C = C_tok @ W[:, d:2 * d].T
        contrib_zs = zs_tok @ W[:, 2 * d:].T
        return contrib_H, contrib_C, contrib_zs

    def weight_block_norms(self):
        """진단용: 첫 Linear 가중치를 H/C/zs 입력 블록별로 잘라 norm을 본다.
        입력(zs)은 avg_l1_norm으로 묶여있는데도 contrib_zs가 압도적으로 크다면,
        '입력이 아니라 이 가중치 블록 자체가 커진 것'이라는 가설을 직접 확인한다.
        block마다 column 수가 달라(H/C=embed_dim, zs=zs_dim) Frobenius norm은
        차원 효과를 포함하므로, 차원-불변인 평균 절댓값(absmean)도 같이 본다."""
        d = self.embed_dim
        W = self.net[0].weight.detach()
        blocks = {"H": W[:, :d], "C": W[:, d:2 * d], "zs": W[:, 2 * d:]}
        out = {}
        for name, w in blocks.items():
            out[f"actor/w0_norm_{name}"] = float(w.norm().cpu())
            out[f"actor/w0_absmean_{name}"] = float(w.abs().mean().cpu())
        return out

    def _layer_scale_probe(self, H, C, zs):
        """진단용: net 각 stage(fc0/relu0/fc1/relu1/fc2)의 abs_mean."""
        B, L, _ = H.shape
        C_tok = C.unsqueeze(1).expand(B, L, C.shape[-1])
        zs_tok = zs.unsqueeze(1).expand(B, L, zs.shape[-1])
        x = torch.cat([H, C_tok, zs_tok], dim=-1)
        out = {"actor/scale_in_abs_mean": float(x.abs().mean().detach().cpu())}
        stage_names = ["fc0", "relu0", "fc1", "relu1"]
        for name, layer in zip(stage_names, self.net[:4]):
            x = layer(x)
            out[f"actor/scale_{name}_abs_mean"] = float(x.abs().mean().detach().cpu())
        x = self.net[4](x)  # fc2
        out["actor/scale_fc2_abs_mean"] = float(x.abs().mean().detach().cpu())
        return out

    def bias_norms(self):
        """진단용(가중치만, forward 불필요): 각 Linear의 bias/weight 절댓값 평균을 따로 본다.
        pretanh_std(~0.01~0.06)가 pretanh_abs_mean(~8~14)에 비해 극도로 작다면, 출력이
        공통 오프셋(주로 마지막 Linear의 bias)에 의해 결정되는지를 직접 가른다."""
        stage_names = ["fc0", "fc1", "fc2"]
        linears = [m for m in self.net if isinstance(m, nn.Linear)]
        out = {}
        for name, lin in zip(stage_names, linears):
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
        """진단용: pre-tanh 1층(Linear) 출력을 H/C/zs 입력 블록별 기여로 분해해서 반환.
        입력 크기가 아니라 '학습된 가중치를 거친 후의 실제 기여'를 보기 위함."""
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


class RegionTD7Critic(nn.Module):
    def __init__(self, action_dim, zs_dim=256, hdim=256, embed_dim=64):
        super().__init__()
        token_dim = embed_dim * 2 + zs_dim * 2 + action_dim
        self.q1_token = nn.Sequential(
            nn.Linear(token_dim, hdim),
            nn.ELU(),
            nn.Linear(hdim, hdim),
            nn.ELU(),
            nn.Linear(hdim, 1),
        )
        self.q2_token = nn.Sequential(
            nn.Linear(token_dim, hdim),
            nn.ELU(),
            nn.Linear(hdim, hdim),
            nn.ELU(),
            nn.Linear(hdim, 1),
        )

    def _token_q(self, H, C, zs, zsa, action, mask, token_net):
        B, L, _ = H.shape
        C_tok = C.unsqueeze(1).expand(B, L, C.shape[-1])
        zs_tok = zs.unsqueeze(1).expand(B, L, zs.shape[-1])
        zsa_tok = zsa.unsqueeze(1).expand(B, L, zsa.shape[-1])
        token_input = torch.cat([H, C_tok, zs_tok, zsa_tok, action], dim=-1)
        q_tok = token_net(token_input)

        token_mask = mask.unsqueeze(-1).to(dtype=q_tok.dtype)
        q_tok = q_tok * token_mask
        n_valid = token_mask.sum(dim=1).clamp(min=1.0)
        q = q_tok.sum(dim=1) / torch.sqrt(n_valid)
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

"""TD7 networks (Fujimoto et al. 2023, "For SALE: State-Action Representation Learning").

이 프레임워크 규약:
  - Actor 는 .encoder (학습용), .fixed_encoder (스냅샷, 추론/critic용), .mlp (정책 head) 를 가진다.
    learner.py 가 actor.mlp / actor.encoder 를 각각 별도 옵티마이저로 최적화한다.
  - Critic 은 (state, action, zsa, zs) 를 받아 [B,2] (두 Q head) 를 반환한다.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cocel_rl.utils.utils import AvgL1Norm


class Encoder(nn.Module):
    """SALE encoder: f(s)->zs, g(zs,a)->zsa."""
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256):
        super().__init__()
        # state encoder f(s)
        self.zs1 = nn.Linear(state_dim, hdim)
        self.zs2 = nn.Linear(hdim, hdim)
        self.zs3 = nn.Linear(hdim, zs_dim)
        # state-action encoder g(zs, a)
        self.zsa1 = nn.Linear(zs_dim + action_dim, hdim)
        self.zsa2 = nn.Linear(hdim, hdim)
        self.zsa3 = nn.Linear(hdim, zs_dim)

    def zs(self, state):
        z = F.elu(self.zs1(state))
        z = F.elu(self.zs2(z))
        z = AvgL1Norm(self.zs3(z))
        return z

    def zsa(self, zs, action):
        z = F.elu(self.zsa1(torch.cat([zs, action], 1)))
        z = F.elu(self.zsa2(z))
        z = self.zsa3(z)
        return z


class PolicyMLP(nn.Module):
    """정책 head: (state, zs) -> action."""
    def __init__(self, state_dim, action_dim, zs_dim, hdim, max_action):
        super().__init__()
        self.l0 = nn.Linear(state_dim, hdim)
        self.l1 = nn.Linear(zs_dim + hdim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, action_dim)
        self.max_action = max_action

    def forward(self, state, zs):
        a = AvgL1Norm(self.l0(state))
        a = torch.cat([a, zs], 1)
        a = F.relu(self.l1(a))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound, zs_dim=256, hdim=256, exploration_noise=0.1):
        super().__init__()
        self.action_bound = action_bound
        self.max_action = float(max(abs(action_bound[0]), abs(action_bound[1])))
        self.exploration_noise = exploration_noise
        self.obs_dim = state_dim
        self.act_dim = action_dim
        self.zs_dim = zs_dim
        self.hidden_dims = [hdim, hdim]
        self.encoder_hidden_dims = [hdim, hdim]
        self.activation_fc_name = 'relu'

        self.encoder = Encoder(state_dim, action_dim, zs_dim, hdim)        # 학습됨 (encoder_optimizer)
        self.fixed_encoder = Encoder(state_dim, action_dim, zs_dim, hdim)  # 스냅샷 (추론/critic용)
        self.mlp = PolicyMLP(state_dim, action_dim, zs_dim, hdim, self.max_action)

        # fixed_encoder := encoder 로 초기화하고 autograd 대상에서 제외(스냅샷)
        self.fixed_encoder.load_state_dict(self.encoder.state_dict())
        for p in self.fixed_encoder.parameters():
            p.requires_grad_(False)

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.float32)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        zs = self.fixed_encoder.zs(state)
        return self.mlp(state, zs)

    def get_action(self, state, eval):
        device = next(self.parameters()).device
        s = torch.as_tensor(state, dtype=torch.float32, device=device)
        if s.dim() == 1:
            s = s.unsqueeze(0)
        with torch.no_grad():
            action = self.forward(s)
        a = action.cpu().numpy()
        if not eval:
            a = a + np.random.normal(0.0, self.max_action * self.exploration_noise, size=a.shape)
            a = np.clip(a, self.action_bound[0], self.action_bound[1])
        return a, None


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256):
        super().__init__()
        self.q01 = nn.Linear(state_dim + action_dim, hdim)
        self.q1 = nn.Linear(2 * zs_dim + hdim, hdim)
        self.q2 = nn.Linear(hdim, hdim)
        self.q3 = nn.Linear(hdim, 1)

        self.q02 = nn.Linear(state_dim + action_dim, hdim)
        self.q4 = nn.Linear(2 * zs_dim + hdim, hdim)
        self.q5 = nn.Linear(hdim, hdim)
        self.q6 = nn.Linear(hdim, 1)

    def forward(self, state, action, zsa, zs):
        sa = torch.cat([state, action], 1)
        emb = torch.cat([zsa, zs], 1)

        q1 = AvgL1Norm(self.q01(sa))
        q1 = torch.cat([q1, emb], 1)
        q1 = F.elu(self.q1(q1))
        q1 = F.elu(self.q2(q1))
        q1 = self.q3(q1)

        q2 = AvgL1Norm(self.q02(sa))
        q2 = torch.cat([q2, emb], 1)
        q2 = F.elu(self.q4(q2))
        q2 = F.elu(self.q5(q2))
        q2 = self.q6(q2)

        return torch.cat([q1, q2], 1)


# ================================================================================= #
# =================================== ONNX ======================================== #
# ================================================================================= #

class TD7ONNXPolicy(nn.Module):
    """추론 전용(인코더 학습 부분 제외). create_onnx_policy 에서 mlp/fixed_encoder 를 복사해 넣는다."""
    def __init__(self, obs_dim, act_dim, max_action, zs_dim, hidden_dims, encoder_hidden_dims, activation_fc_name):
        super().__init__()
        hdim = hidden_dims[0]
        self.fixed_encoder = Encoder(obs_dim, act_dim, zs_dim, hdim)
        self.mlp = PolicyMLP(obs_dim, act_dim, zs_dim, hdim, max_action)

    def forward(self, state):
        zs = self.fixed_encoder.zs(state)
        return self.mlp(state, zs)

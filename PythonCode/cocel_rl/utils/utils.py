import torch
import torch.nn as nn
import copy

from typing import Any 

def _get_default(val: Any, default: Any):
    return val if val is not None else default

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers.
        Reference: https://github.com/MishaLaskin/rad/blob/master/curl_sac.py"""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)
        

def soft_update(network, target_network, tau):
    with torch.no_grad():
        for param, target_param in zip(network.parameters(), target_network.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def move_to_cpu(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if isinstance(value, dict):
            new_state_dict[key] = move_to_cpu(value)
        elif isinstance(value, torch.Tensor):
            new_state_dict[key] = value.cpu()
        else:
            new_state_dict[key] = value
    return new_state_dict

def create_onnx_policy(src, algorithm):
    from algorithms.sac.network import SACONNXPolicy
    from algorithms.td3.network import TD3ONNXPolicy
    from algorithms.td7.network import TD7ONNXPolicy
    from algorithms.tqc.network import TQCONNXPolicy
    from algorithms.ddpg.network import DDPGONNXPolicy

    if algorithm == 'tqc':
        onnx_policy = TQCONNXPolicy(src.obs_dim, src.act_dim, src.hidden_dims, src.action_bound, src.activation_fc_name)
        onnx_policy.input_layer = copy.deepcopy(src.input_layer)
        onnx_policy.hidden_layers = copy.deepcopy(src.hidden_layers)
        onnx_policy.mean_layer = copy.deepcopy(src.mean_layer)
    elif algorithm == 'td7':
        onnx_policy = TD7ONNXPolicy(src.obs_dim, src.act_dim, src.max_action, src.zs_dim, src.hidden_dims, src.encoder_hidden_dims, src.activation_fc_name)
        onnx_policy.mlp = copy.deepcopy(src.mlp)
        onnx_policy.fixed_encoder = copy.deepcopy(src.fixed_encoder)
    elif algorithm == 'sac':
        onnx_policy = SACONNXPolicy(src.obs_dim, src.act_dim, src.hidden_dims, src.action_bound, src.activation_fc_name)
        onnx_policy.input_layer = copy.deepcopy(src.input_layer)
        onnx_policy.hidden_layers = copy.deepcopy(src.hidden_layers)
        onnx_policy.output_layer = copy.deepcopy(src.mean_layer)
    elif algorithm == 'td3':
        onnx_policy = TD3ONNXPolicy(src.obs_dim, src.act_dim, src.action_bound, src.hidden_dims, src.activation_fc_name)
        onnx_policy.input_layer = copy.deepcopy(src.input_layer)
        onnx_policy.hidden_layers = copy.deepcopy(src.hidden_layers)
        onnx_policy.output_layer = copy.deepcopy(src.output_layer)
    elif algorithm == 'ddpg':
        onnx_policy = DDPGONNXPolicy(src.obs_dim, src.act_dim, src.action_bound, src.hidden_dims, src.activation_fc_name)
        onnx_policy.input_layer = copy.deepcopy(src.input_layer)
        onnx_policy.hidden_layers = copy.deepcopy(src.hidden_layers)
        onnx_policy.output_layer = copy.deepcopy(src.output_layer)
    else:
        raise NameError

    return onnx_policy

def AvgL1Norm(x, eps=1e-8):
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)

def quantile_huber_loss_f(quantiles, samples, device):
    pairwise_delta = samples[:, None, None, :] - quantiles[:, :, :, None]  # batch x nets x quantiles x samples
    abs_pairwise_delta = torch.abs(pairwise_delta)
    huber_loss = torch.where(abs_pairwise_delta > 1,
                             abs_pairwise_delta - 0.5,
                             pairwise_delta ** 2 * 0.5)

    n_quantiles = quantiles.shape[2]
    tau = torch.arange(n_quantiles, device=device).float() / n_quantiles + 1 / 2 / n_quantiles
    loss = (torch.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss).mean()
    return loss

def LAP_huber(x, min_priority=1):
    return torch.where(x < min_priority, 0.5 * x.pow(2), min_priority * x).sum(1).mean()

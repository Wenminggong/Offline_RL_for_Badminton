# -*- coding: utf-8 -*-
'''
@File    :   actor.py
@Time    :   2024/06/03 15:30:16
@Author  :   Mingjiang Liu 
@Version :   1.0
@Desc    :   actor implementations for DRL algorithms
'''

import torch
import torch.nn as nn
from torch.distributions import Normal, TanhTransform, TransformedDistribution, MultivariateNormal, Categorical
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union


def init_module_weights(module: torch.nn.Sequential, orthogonal_init: bool = False):
    # Specific orthgonal initialization for inner layers
    # If orthogonal init is off, we do not change default initialization
    if orthogonal_init:
        for submodule in module:
            if isinstance(submodule, nn.Linear):
                nn.init.orthogonal_(submodule.weight, gain=np.sqrt(2))
                nn.init.constant_(submodule.bias, 0.0)

def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)


class ReparameterizedGaussian(nn.Module):
    def __init__(
        self, log_std_min: float = -1.0, log_std_max: float = 1.0
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        # self.f_mu = nn.Sigmoid()
        self.f_tho = nn.Tanh()
    
    def _get_distribution(self, landing_preds: torch.Tensor, move_preds: torch.Tensor, repeat: bool = None) -> Tuple[torch.Tensor, torch.Tensor, torch.distributions.Distribution, torch.distributions.Distribution]:
        if repeat is None:
            landing_mu = landing_preds[:, 0:2]
            landing_log_std = torch.clamp(landing_preds[:, 2:4], self.log_std_min, self.log_std_max)
            landing_std = torch.exp(landing_log_std)
            landing_tho = self.f_tho(landing_preds[:, -1])

            landing_cov = torch.zeros(landing_preds.shape[0], 2, 2).to(device=landing_preds.device)
            landing_cov[:, 0, 0] = landing_std[:, 0] * landing_std[:, 0]
            landing_cov[:, 1, 1] = landing_std[:, 1] * landing_std[:, 1]
            landing_cov[:, 0, 1] = landing_std[:, 0] * landing_std[:, 1] * landing_tho
            landing_cov[:, 1, 0] = landing_std[:, 0] * landing_std[:, 1] * landing_tho

            move_mu = move_preds[:, 0:2]
            move_log_std = torch.clamp(move_preds[:, 2:4], self.log_std_min, self.log_std_max)
            move_std = torch.exp(move_log_std)
            move_tho = self.f_tho(move_preds[:, -1])

            move_cov = torch.zeros(move_preds.shape[0], 2, 2).to(device=move_preds.device)
            move_cov[:, 0, 0] = move_std[:, 0] * move_std[:, 0]
            move_cov[:, 1, 1] = move_std[:, 1] * move_std[:, 1]
            move_cov[:, 0, 1] = move_std[:, 0] * move_std[:, 1] * move_tho
            move_cov[:, 1, 0] = move_std[:, 0] * move_std[:, 1] * move_tho
        else:
            landing_mu = landing_preds[:, :, 0:2]
            landing_log_std = torch.clamp(landing_preds[:, :, 2:4], self.log_std_min, self.log_std_max)
            landing_std = torch.exp(landing_log_std)
            landing_tho = self.f_tho(landing_preds[:, :, -1])

            landing_cov = torch.zeros(landing_preds.shape[0], landing_preds.shape[1], 2, 2).to(device=landing_preds.device)
            landing_cov[:, :, 0, 0] = landing_std[:, :, 0] * landing_std[:, :, 0]
            landing_cov[:, :, 1, 1] = landing_std[:, :, 1] * landing_std[:, :, 1]
            landing_cov[:, :, 0, 1] = landing_std[:, :, 0] * landing_std[:, :, 1] * landing_tho
            landing_cov[:, :, 1, 0] = landing_std[:, :, 0] * landing_std[:, :, 1] * landing_tho

            move_mu = move_preds[:, :, 0:2]
            move_log_std = torch.clamp(move_preds[:, :, 2:4], self.log_std_min, self.log_std_max)
            move_std = torch.exp(move_log_std)
            move_tho = self.f_tho(move_preds[:, :, -1])

            move_cov = torch.zeros(move_preds.shape[0], move_preds.shape[1], 2, 2).to(device=move_preds.device)
            move_cov[:, :, 0, 0] = move_std[:, :, 0] * move_std[:, :, 0]
            move_cov[:, :, 1, 1] = move_std[:, :, 1] * move_std[:, :, 1]
            move_cov[:, :, 0, 1] = move_std[:, :, 0] * move_std[:, :, 1] * move_tho
            move_cov[:, :, 1, 0] = move_std[:, :, 0] * move_std[:, :, 1] * move_tho

        landing_distribution = MultivariateNormal(loc=landing_mu, covariance_matrix=landing_cov)
        move_distribution = MultivariateNormal(loc=move_mu, covariance_matrix=move_cov)
        return landing_mu, move_mu, landing_distribution, move_distribution

    def log_prob(
        self, landing_preds: torch.Tensor, move_preds: torch.Tensor, landing_xy: torch.Tensor, move_xy: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        landing_mu, move_mu, landing_distribution, move_distribution = self._get_distribution(landing_preds, move_preds)
        landing_log_prob = landing_distribution.log_prob(landing_xy).unsqueeze(dim=-1)
        move_log_prob = move_distribution.log_prob(move_xy).unsqueeze(dim=-1)
        return landing_log_prob, move_log_prob

    def forward(
        self, landing_preds: torch.Tensor, move_preds: torch.Tensor, deterministic: bool = False, repeat: bool = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        landing_mu, move_mu, landing_distribution, move_distribution = self._get_distribution(landing_preds, move_preds, repeat)

        if deterministic:
            landing_sample = landing_mu
            move_sample = move_mu
        else:
            landing_sample = landing_distribution.rsample()
            move_sample = move_distribution.rsample()

        landing_log_prob = landing_distribution.log_prob(landing_sample).unsqueeze(dim=-1)
        move_log_prob = move_distribution.log_prob(move_sample).unsqueeze(dim=-1)

        return landing_sample, landing_log_prob, move_sample, move_log_prob
    

class GaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        last_time_shot_type_dim: int,
        hit_xy_dim: int,
        player_location_xy_dim: int,
        opponent_location_xy_dim: int,
        shot_type_dim: int,
        landing_xy_dim: int,
        move_xy_dim: int,
        orthogonal_init: bool = False,
        n_hidden_layers: int = 3,
        hidden_layer_dim: int = 256,
        embedding_dim: int = 64,
        activation_function: nn.Module = nn.ReLU(),
        embedding_coordinate:int=1,
    ):
        super().__init__()
        self.observation_dim = state_dim
        self.action_dim = action_dim
        self.last_time_shot_type_dim = last_time_shot_type_dim
        self.hit_xy_dim = hit_xy_dim
        self.player_location_xy_dim = player_location_xy_dim
        self.opponent_location_xy_dim = opponent_location_xy_dim
        self.shot_type_dim = shot_type_dim
        self.landing_xy_dim = landing_xy_dim
        self.move_xy_dim = move_xy_dim
        self.orthogonal_init = orthogonal_init
        self.embedding_coordinate = embedding_coordinate

        self.embed_shot = nn.Linear(last_time_shot_type_dim, embedding_dim)
        self.embed_xy = nn.Linear(hit_xy_dim, embedding_dim)

        if self.embedding_coordinate:
            layers = [
                nn.Linear(embedding_dim * 4, hidden_layer_dim),
                activation_function,
            ]
        else:
            layers = [
                nn.Linear(embedding_dim+hit_xy_dim+player_location_xy_dim+opponent_location_xy_dim, hidden_layer_dim),
                activation_function,
            ]

        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_layer_dim, hidden_layer_dim))
            layers.append(activation_function)

        self.base_network = nn.Sequential(*layers)

        init_module_weights(self.base_network, orthogonal_init)

        self.predict_shot = nn.Linear(hidden_layer_dim, shot_type_dim)
        self.predict_landing = nn.Linear(hidden_layer_dim, 5)
        self.predict_move = nn.Linear(hidden_layer_dim, 5)

        self.f_shot = nn.Softmax(dim=-1)
        self.gaussian = ReparameterizedGaussian()

    def log_prob(
        self,
        last_time_shot_type: torch.Tensor, 
        hit_xy: torch.Tensor, 
        player_location_xy: torch.Tensor, 
        opponent_location_xy: torch.Tensor,
        shot_type: torch.Tensor,
        landing_xy: torch.Tensor,
        move_xy: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        last_time_shot_type_embed = self.embed_shot(last_time_shot_type)
        if self.embedding_coordinate:
            hit_xy_embed = self.embed_xy(hit_xy)
            player_location_xy_embed = self.embed_xy(player_location_xy)
            opponent_location_xy_embed = self.embed_xy(opponent_location_xy)
        else:
            hit_xy_embed = hit_xy
            player_location_xy_embed = player_location_xy
            opponent_location_xy_embed = opponent_location_xy
        input_tensor = torch.cat([last_time_shot_type_embed, hit_xy_embed, player_location_xy_embed, opponent_location_xy_embed], dim=-1)

        base_network_output = self.base_network(input_tensor)

        shot_preds = self.predict_shot(base_network_output)
        landing_preds = self.predict_landing(base_network_output)
        move_preds = self.predict_move(base_network_output)

        shot_probs = self.f_shot(shot_preds)
        log_shot_probs = torch.log(shot_probs+1e-8)
        landing_log_prob, move_log_prob = self.gaussian.log_prob(landing_preds, move_preds, landing_xy, move_xy)
        return log_shot_probs, landing_log_prob, move_log_prob

    def forward(
        self,
        last_time_shot_type: torch.Tensor, 
        hit_xy: torch.Tensor, 
        player_location_xy: torch.Tensor, 
        opponent_location_xy: torch.Tensor,
        deterministic: bool = False,
        repeat: bool = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        if repeat is not None:
            # repeat sampling
            # [batch_size, repeat, n]
            last_time_shot_type = extend_and_repeat(last_time_shot_type, 1, repeat)
            hit_xy = extend_and_repeat(hit_xy, 1, repeat)
            player_location_xy = extend_and_repeat(player_location_xy, 1, repeat)
            opponent_location_xy = extend_and_repeat(opponent_location_xy, 1, repeat)
        last_time_shot_type_embed = self.embed_shot(last_time_shot_type)
        if self.embedding_coordinate:
            hit_xy_embed = self.embed_xy(hit_xy)
            player_location_xy_embed = self.embed_xy(player_location_xy)
            opponent_location_xy_embed = self.embed_xy(opponent_location_xy)
        else:
            hit_xy_embed = hit_xy
            player_location_xy_embed = player_location_xy
            opponent_location_xy_embed = opponent_location_xy
        input_tensor = torch.cat([last_time_shot_type_embed, hit_xy_embed, player_location_xy_embed, opponent_location_xy_embed], dim=-1)

        base_network_output = self.base_network(input_tensor)

        shot_preds = self.predict_shot(base_network_output)
        landing_preds = self.predict_landing(base_network_output)
        move_preds = self.predict_move(base_network_output)

        shot_probs = self.f_shot(shot_preds) # [batch_size, shot_type_dim] or [batch_size, repeat, shot_type_dim], shot_type_probs
        shot_distribution = Categorical(probs=shot_probs)
        if deterministic:
            shot_sample = torch.argmax(shot_probs, dim=-1, keepdim=True)
        else:
            shot_sample = shot_distribution.sample().unsqueeze(dim=-1)
        log_shot_probs = torch.log(shot_probs+1e-8)
        landing_sample, landing_log_prob, move_sample, move_log_prob = self.gaussian(landing_preds, move_preds, deterministic, repeat)
        # shot_sample: [batch_size, 1] or [batch_size, repeat, 1], others: [batch_size, n] or [batch_size, repeat, n]
        return shot_sample, shot_probs, log_shot_probs, landing_sample, landing_log_prob, move_sample, move_log_prob

    @torch.no_grad()
    def act(
        self, 
        last_time_shot_type: torch.Tensor, 
        hit_xy: torch.Tensor, 
        player_location_xy: torch.Tensor, 
        opponent_location_xy: torch.Tensor,
        deterministic: bool = False,
        device: str = "cuda"):
        if len(hit_xy.shape) < 2:
            last_time_shot_type = last_time_shot_type.reshape(1, -1)
            hit_xy = hit_xy.reshape(1, -1)
            player_location_xy = player_location_xy.reshape(1, -1)
            opponent_location_xy = opponent_location_xy.reshape(1, -1)
        with torch.no_grad():
            last_time_shot_type = last_time_shot_type.to(device=device, dtype=torch.float32) 
            hit_xy = hit_xy.to(device=device, dtype=torch.float32)
            player_location_xy = player_location_xy.to(device=device, dtype=torch.float32)
            opponent_location_xy = opponent_location_xy.to(device=device, dtype=torch.float32)
            shot_sample, shot_probs, log_shot_probs, landing_sample, landing_log_prob, move_sample, move_log_prob = self(last_time_shot_type, hit_xy, player_location_xy,opponent_location_xy, deterministic)
        # shot_sample: [batch_size, 1], others: [batch_size, n] - ndarray; shot_probs: [batch_size, shot_type_dim]
        return shot_sample.cpu().numpy(), shot_probs.cpu().numpy(), landing_sample.cpu().numpy(), move_sample.cpu().numpy()
    
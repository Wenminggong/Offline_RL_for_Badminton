# -*- coding: utf-8 -*-
'''
@File    :   critic.py
@Time    :   2024/06/02 22:30:22
@Author  :   Mingjiang Liu 
@Version :   1.0
@Desc    :   critic implementations for DRL algorithms
'''


import torch
import torch.nn as nn
import numpy as np

def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)


def init_module_weights(module: torch.nn.Sequential, orthogonal_init: bool = False):
    # Specific orthgonal initialization for inner layers
    # If orthogonal init is off, we do not change default initialization
    if orthogonal_init:
        for submodule in module[:-1]:
            if isinstance(submodule, nn.Linear):
                nn.init.orthogonal_(submodule.weight, gain=np.sqrt(2))
                nn.init.constant_(submodule.bias, 0.0)

    # Last layers should be initialzied differently as well
    if orthogonal_init:
        nn.init.orthogonal_(module[-1].weight, gain=1e-2)
    else:
        nn.init.xavier_uniform_(module[-1].weight, gain=1e-2)

    nn.init.constant_(module[-1].bias, 0.0)


class FullyConnectedQFunction(nn.Module):
    def __init__(
        self,
        observation_dim: int,
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
        # self.observation_dim = observation_dim
        # self.action_dim = action_dim
        self.orthogonal_init = orthogonal_init
        self.embedding_coordinate = embedding_coordinate

        self.embed_shot = nn.Linear(last_time_shot_type_dim, embedding_dim)
        self.embed_xy = nn.Linear(hit_xy_dim, embedding_dim)

        if self.embedding_coordinate:
            layers = [
                nn.Linear(embedding_dim * 6, hidden_layer_dim),
                activation_function,
            ]
        else:
            layers = [
                nn.Linear(embedding_dim+hit_xy_dim+player_location_xy_dim+opponent_location_xy_dim+landing_xy_dim+move_xy_dim, hidden_layer_dim),
                activation_function,
            ]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_layer_dim, hidden_layer_dim))
            layers.append(activation_function)
        layers.append(nn.Linear(hidden_layer_dim, shot_type_dim))

        self.network = nn.Sequential(*layers)

        init_module_weights(self.network, orthogonal_init)

    def forward(self, last_time_shot_type: torch.Tensor, hit_xy: torch.Tensor, player_location_xy: torch.Tensor, opponent_location_xy: torch.Tensor, shot_type: torch.Tensor, landing_xy: torch.Tensor, move_xy: torch.Tensor) -> torch.Tensor:
        
        if landing_xy.ndim == 3 and last_time_shot_type.ndim == 2:
            repeat_size = landing_xy.shape[1]
            last_time_shot_type = extend_and_repeat(last_time_shot_type, 1, repeat_size)
            hit_xy = extend_and_repeat(hit_xy, 1, repeat_size)
            player_location_xy = extend_and_repeat(player_location_xy, 1, repeat_size)
            opponent_location_xy = extend_and_repeat(opponent_location_xy, 1, repeat_size)
        last_time_shot_type_embed = self.embed_shot(last_time_shot_type)
        if self.embedding_coordinate:
            hit_xy_embed = self.embed_xy(hit_xy)
            player_location_xy_embed = self.embed_xy(player_location_xy)
            opponent_location_xy_embed = self.embed_xy(opponent_location_xy)
            # shot_type_embed = self.embed_shot(shot_type)
            landing_xy_embed = self.embed_xy(landing_xy)
            move_xy_embed = self.embed_xy(move_xy)
        else:
            hit_xy_embed = hit_xy
            player_location_xy_embed = player_location_xy
            opponent_location_xy_embed = opponent_location_xy
            landing_xy_embed = landing_xy
            move_xy_embed = move_xy
        
        input_tensor = torch.cat([last_time_shot_type_embed,
                                  hit_xy_embed,
                                  player_location_xy_embed, 
                                  opponent_location_xy_embed,  
                                  landing_xy_embed, 
                                  move_xy_embed], 
                                  dim=-1)
        q_values = self.network(input_tensor)
        # q_values: tensor [batch_size, shot_type_dim] or [batch_size, repeat_size, shot_type_dim]
        return q_values
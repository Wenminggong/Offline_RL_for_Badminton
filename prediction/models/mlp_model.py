# -*- coding: utf-8 -*-
'''
@File    :   mlp_model.py
@Time    :   2024/07/20 21:14:17
@Author  :   Mingjiang Liu 
@Version :   1.0
@Desc    :   MLP prediction model for win and no_win probs prediction.
'''


import torch
import torch.nn as nn
import torch.nn.functional as F

from offline_rl.models.critic import init_module_weights


class MLPPrediction(nn.Module):
    def __init__(self,
                shot_type_num:int=10,
                shot_type_dim:int=15,
                player_flag:bool=False,
                player_num:int=43,
                player_dim:int=15,
                location_type:bool=False,
                location_num:int=16,
                location_dim:int=10,
                other_fea_type:bool=False,
                other_fea_dim:int=1,
                n_layer:int=3,
                hidden_dim:int=256,
                orthogonal_init:bool=True,
                activation:nn.Module = nn.ReLU(),
                device:str="cuda",
            ) -> None:
        super().__init__()
        self.player_flag = player_flag # True: use, False: not
        self.location_type = location_type # True: discrete area, False: continuous xy
        self.other_fea_type = other_fea_type # True: use [around_head, back_hand], False: [bad_landing_flag, landing_distance_opponent]
        self.device = device

        if self.player_flag:
            self.player_embedding = nn.Embedding(num_embeddings=player_num, embedding_dim=player_dim)
        else:
            player_dim = 0

        if self.location_type:
            # True: discrete, False: continuous
            self.location_embedding = nn.Embedding(num_embeddings=location_num, embedding_dim=location_dim)
        else:
            location_dim = 2
        
        self.shot_type_embedding = nn.Embedding(num_embeddings=shot_type_num, embedding_dim=shot_type_dim)
        # self.shot_type_embedding = nn.Linear(shot_type_num, shot_type_dim)
        shot_embed_dim = shot_type_dim + player_dim + location_dim * 4 + other_fea_dim
        
        predict_layer = [
            nn.Linear(shot_embed_dim, hidden_dim), #[batch_size, hidden_dim]
            activation,
        ]
        for _ in range(n_layer-1):
            predict_layer.append(nn.Linear(hidden_dim, hidden_dim))
            predict_layer.append(activation)
        predict_layer.append(nn.Linear(hidden_dim, 1))
        self.predict_layer = nn.Sequential(*predict_layer)
        self.output = nn.Sigmoid()

        init_module_weights(self.predict_layer, orthogonal_init)

    def forward(self,
            player_id:torch.Tensor, 
            shot_type:torch.Tensor, 
            hit_area:torch.Tensor,
            hit_xy:torch.Tensor, 
            player_area:torch.Tensor, 
            player_xy:torch.Tensor,
            opponent_area:torch.Tensor,
            opponent_xy:torch.Tensor,
            landing_area:torch.Tensor,
            landing_xy:torch.Tensor,
            timesteps:torch.Tensor,   
            time:torch.Tensor, 
            posture_fea:torch.Tensor,
            landing_fea:torch.Tensor, 
            rally_info:torch.Tensor,
            mask:torch.Tensor,
        ) -> torch.Tensor:
        # player_id, shot_type, hit_area, player_area, opponent_area, landing_area: [batch_size], others: [batch_size, n]

        # shot embedding
        player_embed = torch.tensor([]).to(self.device)
        if self.player_flag:
            player_embed = self.player_embedding(player_id)
        
        # if location is continuous, [batch_size, 2]
        if self.location_type:
            # if location is discrete
            hit_location_embed = self.location_embedding(hit_area) # [batch_size, location_dim]
            player_location_embed = self.location_embedding(player_area)
            oppo_location_embed = self.location_embedding(opponent_area)
            landing_location_embed = self.location_embedding(landing_area)
        else:
            hit_location_embed = hit_xy # [batch_size, 2]
            player_location_embed = player_xy
            oppo_location_embed = opponent_xy
            landing_location_embed = landing_xy

        shot_type_embed = self.shot_type_embedding(shot_type) # [batch_size] -> [batch_size, shot_type_dim]

        if self.other_fea_type:
            # use [around_head, back_hand]
            other_fea = posture_fea
        else:
            # use [bad_landing_flag, landing_distance_opponent]
            other_fea = landing_fea

        # [batch_size, n]
        shot_embed = torch.cat([player_embed, hit_location_embed, player_location_embed, oppo_location_embed, landing_location_embed, shot_type_embed, other_fea], dim=-1)
        
        win_probs = self.output(self.predict_layer(shot_embed)) # [batch_size, 1]
        return win_probs
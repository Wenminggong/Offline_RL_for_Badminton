# -*- coding: utf-8 -*-
'''
@File    :   short_long_term_model.py
@Time    :   2024/07/02 17:25:13
@Author  :   Mingjiang Liu 
@Version :   1.0
@Desc    :   short-long-term dependance model for win rate prediction, refer to "How Is the Stroke? Inferring Shot Influence in Badminton Matches via Long Short-term Dependencies"
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class Short1DCNN(nn.Module):
    # two seperate 1D-CNN for two players' short pattern extracting
    def __init__(self, shot_embed_dim:int, short_pattern_dim:int, kernel_size:int) -> None:
        super().__init__()
        self.short_patten_dim = short_pattern_dim
        self.CNN1 = nn.Conv1d(in_channels=shot_embed_dim, out_channels=short_pattern_dim, kernel_size=kernel_size, padding="same")
        self.CNN2 = nn.Conv1d(in_channels=shot_embed_dim, out_channels=short_pattern_dim, kernel_size=kernel_size, padding="same")

    def forward(self, shot_embed: torch.Tensor):
        #shot_embed: [batch_size, max_seq, shot_embed_dim]
        shot_embed_1 = shot_embed[:, ::2, :] # [batch_size, max_seq / 2, shot_embed_dim]
        shot_embed_2 = shot_embed[:, 1::2, :]

        shot_embed_1 = shot_embed_1.permute(0, 2, 1) # [batch_size, shot_embed_dim, max_seq / 2]
        shot_embed_2 = shot_embed_2.permute(0, 2, 1)

        shot_embed_1 = self.CNN1(shot_embed_1).permute(0, 2, 1) # [batch_size, max_seq/2, short_pattern_dim]
        shot_embed_2 = self.CNN2(shot_embed_2).permute(0, 2, 1)

        batch_size = shot_embed.shape[0]
        max_seq = shot_embed.shape[1]
        if max_seq % 2 == 1:
            last_shot_embed_1 = shot_embed_1[:, -1:, :] # [batch_size, 1, n]
            shot_embed_1 = shot_embed_1[:, :-1, :]
        short_pattern_embed = torch.stack(
            (shot_embed_1, shot_embed_2), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, max_seq // 2 * 2, self.short_patten_dim)

        if max_seq % 2 == 1:
            short_pattern_embed = torch.cat([short_pattern_embed, last_shot_embed_1], dim=1) 
        
        # [batch_size, max_seq, short_patten_dim]
        return short_pattern_embed


class ShortLongPrediction(nn.Module):
    def __init__(self,
                shot_type_num:int=10,
                shot_type_dim:int=15,
                player_flag:bool=True,
                player_num:int=43,
                player_dim:int=15,
                location_type:bool=True,
                location_num:int=16,
                location_dim:int=10,
                other_fea_type:bool=True,
                other_fea_dim:int=2,
                time_weight:bool=True, 
                rally_info_dim:int=2,
                short_pattern_dim:int=32,
                kernel_size:int=3, 
                max_seq_len:int=1028,
                n_layer:int=3,
                n_head:int=1,
                hidden_dim:int=64,
                dropout:float=0.1,
                activation:str='relu',
                device:str="cuda",
            ) -> None:
        super().__init__()
        self.player_flag = player_flag # True: use, False: not
        self.location_type = location_type # True: discrete area, False: continuous xy
        self.other_fea_type = other_fea_type # True: use [around_head, back_hand], False: [bad_landing_flag, landing_distance_opponent]
        self.time_weight = time_weight # True: use time-score enhanced shot_type embedding, Flase: not use time-score
        self.rally_info_dim = rally_info_dim # if rally_info_dim = 2, use [score_diff, cons_score]; if rally_info_dim = 0, not use rally_info
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
        if self.time_weight:
            # add time_weight in shot_type_embedding
            self.shot_type_embedding_2 = nn.Embedding(num_embeddings=shot_type_num, embedding_dim=shot_type_dim)
            self.shot_type_embedding_3 = nn.Embedding(num_embeddings=shot_type_num, embedding_dim=shot_type_dim)
            self.time_score_f = nn.Sigmoid()
        
        shot_embed_dim = shot_type_dim + player_dim + location_dim * 4 + other_fea_dim
        self.short_pattern_extractor = Short1DCNN(shot_embed_dim, short_pattern_dim, kernel_size) # [batch_size, max_seq, short_patten_dim]

        self.pos_embedding = nn.Embedding(num_embeddings=max_seq_len, embedding_dim=short_pattern_dim)
        self.embed_ln = nn.LayerNorm(short_pattern_dim)
        rally_encoder_layer = nn.TransformerEncoderLayer(d_model=short_pattern_dim, nhead=n_head, dim_feedforward=hidden_dim, dropout=dropout, activation=activation, batch_first=True)
        self.rally_encoder = nn.TransformerEncoder(encoder_layer=rally_encoder_layer, num_layers=n_layer) # [batch_size, max_seq, hidden_dim]

        self.predict_layer = nn.Linear(short_pattern_dim+rally_info_dim, 1) #[batch_size, 1]
        self.output_f = nn.Sigmoid()
        

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
        # mask, player_id, shot_type, hit_area, player_area, opponent_area, landing_area, timesteps: [batch_size, max_seq], others: [batch_size, max_seq, n]

        # shot embedding
        player_embed = torch.tensor([]).to(self.device)
        if self.player_flag:
            player_embed = self.player_embedding(player_id)
        
        # if location is continuous, [batch_size, max_seq, 2]
        if self.location_type:
            # if location is discrete
            hit_location_embed = self.location_embedding(hit_area) # [batch_size, max_seq, location_dim]
            player_location_embed = self.location_embedding(player_area)
            oppo_location_embed = self.location_embedding(opponent_area)
            landing_location_embed = self.location_embedding(landing_area)
        else:
            hit_location_embed = hit_xy
            player_location_embed = player_xy
            oppo_location_embed = opponent_xy
            landing_location_embed = landing_xy

        shot_type_embed = self.shot_type_embedding(shot_type) # [batch_size, max_seq, 1] -> [batch_size, max_seq, shot_type_dim]
        time_score = 1
        if self.time_weight:
            shot_type_embed_2 = self.shot_type_embedding_2(shot_type)
            shot_type_embed_3 = self.shot_type_embedding_3(shot_type)
            
            time_clone = time.detach().clone()
            time_max = time_clone.max(dim=1, keepdim=True).values # [batch_size, 1, 1]
            time_clone[time_clone == 0] += 1e9
            time_min = time_clone.min(dim=1, keepdim=True).values # [batch_size, 1, 1]
            time = (time - time_min) / (time_max - time_min) # [batch_size, max_seq, 1]
            time_score = self.time_score_f(shot_type_embed_2 + time * shot_type_embed_3)
        shot_type_embed = time_score * shot_type_embed

        if self.other_fea_type:
            # use [around_head, back_hand]
            other_fea = posture_fea
        else:
            # use [bad_landing_flag, landing_distance_opponent]
            other_fea = landing_fea

        # [batch_size, max_seq, n]
        shot_embed = torch.cat([player_embed, hit_location_embed, player_location_embed, oppo_location_embed, landing_location_embed, shot_type_embed, other_fea], dim=-1)

        # short pattern extracting
        short_pattern_embed = self.short_pattern_extractor(shot_embed) # [batch_size, max_seq, short_pattern_dim]

        pos_embed = self.pos_embedding(timesteps) # [batch_size, max_seq, short_pattern_dim]
        short_pattern_embed = self.embed_ln(short_pattern_embed + pos_embed) # layer norm

        # rally encoding
        rally_embed = self.rally_encoder(short_pattern_embed, src_key_padding_mask=mask)
        rally_embed[mask == 1] = -1e9
        rally_embed = torch.max(rally_embed, dim=1).values # [batch_size, hidden_dim], maxpooling
        
        if self.rally_info_dim > 0:
            rally_info_embed = rally_info
            # output layer
            # get last-step rally_info
            predict_input = torch.cat([rally_embed, rally_info_embed[:, -1, :]], dim=-1)
        else:
             predict_input = rally_embed
        
        
        win_probs = self.output_f(self.predict_layer(predict_input)) # [batch_size, 1]
        return win_probs


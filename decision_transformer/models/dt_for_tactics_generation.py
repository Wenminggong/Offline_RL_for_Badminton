# -*- coding: utf-8 -*-
'''
@File    :   dt_for_tactics_generation.py
@Time    :   2024/05/16 21:52:27
@Author  :   Mingjiang Liu 
@Version :   1.0
@Desc    :   decision transformer model for tactics generation
'''


import numpy as np
import torch
import torch.nn as nn

import transformers

from decision_transformer.models.model import TrajectoryModel
from decision_transformer.models.trajectory_gpt2 import GPT2Model


class DecisionTransformerTactics(TrajectoryModel):

    """
    This model uses Decision Transformer to model tactics generation problem 
    input: (return_1, state_1, action_1, return_2, state_2 ...)
    output: action_i
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            last_time_shot_type_dim,
            hit_xy_dim,
            player_location_xy_dim,
            opponent_location_xy_dim,
            shot_type_dim,
            landing_xy_dim,
            move_xy_dim,
            hidden_size,
            embed_size,
            max_length=None,
            max_ep_len=4096,
            use_player_location=1,
            embed_coordinate=1,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)
        self.last_time_shot_type_dim = last_time_shot_type_dim
        self.hit_xy_dim = hit_xy_dim
        self.player_location_xy_dim = player_location_xy_dim
        self.opponent_location_xy_dim = opponent_location_xy_dim
        self.shot_type_dim = shot_type_dim
        self.landing_xy_dim = landing_xy_dim
        self.move_xy_dim = move_xy_dim

        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.use_player_location = use_player_location
        self.embed_coordinate = embed_coordinate
        # set config for GPT2Model
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_shot = nn.Embedding(last_time_shot_type_dim, embed_size)
        self.embed_xy = nn.Linear(hit_xy_dim, embed_size)
        if self.use_player_location:
            if self.embed_coordinate:
                self.embed_state = nn.Linear(embed_size * 4, hidden_size)
            else:
                self.embed_state = nn.Linear(embed_size+hit_xy_dim*3, hidden_size)
        else:
            if self.embed_coordinate:
                self.embed_state = nn.Linear(embed_size * 2, hidden_size)
            else:
                self.embed_state = nn.Linear(embed_size+hit_xy_dim, hidden_size)
        
        if self.embed_coordinate:
            self.embed_action = nn.Linear(embed_size * 3, hidden_size)
        else:
            self.embed_action = nn.Linear(embed_size+landing_xy_dim+move_xy_dim, hidden_size)

        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states 
        self.predict_shot = nn.Linear(hidden_size, shot_type_dim)
        self.predict_landing = nn.Linear(hidden_size, 5)
        self.predict_move = nn.Linear(hidden_size, 5)
        # self.f_mu = nn.Sigmoid() # no output limitation
        self.f_tho = nn.Tanh()

    def forward(self, last_time_shot_type, hit_xy, player_location_xy, opponent_location_xy, shot_type, landing_xy, move_xy, reward, rtg, timesteps, attention_mask=None):

        batch_size, seq_length = last_time_shot_type.shape[0], last_time_shot_type.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long).to(device=last_time_shot_type.device)

        # embed each modality with a different head
        last_time_shot_type_embeddings = self.embed_shot(last_time_shot_type)
        shot_type_embeddings = self.embed_shot(shot_type)
        if self.embed_coordinate:
            hit_xy_embeddings = self.embed_xy(hit_xy)
            if self.use_player_location:
                player_location_xy_embeddings = self.embed_xy(player_location_xy)
                opponent_location_xy_embeddings = self.embed_xy(opponent_location_xy)
            landing_xy_embeddings = self.embed_xy(landing_xy)
            move_xy_embeddings = self.embed_xy(move_xy)
        else:
            hit_xy_embeddings = hit_xy
            if self.use_player_location:
                player_location_xy_embeddings = player_location_xy
                opponent_location_xy_embeddings = opponent_location_xy
            landing_xy_embeddings = landing_xy
            move_xy_embeddings = move_xy
        
        if self.use_player_location:
            states = torch.cat((last_time_shot_type_embeddings, hit_xy_embeddings, player_location_xy_embeddings, opponent_location_xy_embeddings), dim=-1)
        else:
            states = torch.cat((last_time_shot_type_embeddings, hit_xy_embeddings), dim=-1)
        state_embeddings = self.embed_state(states)
        actions = torch.cat((shot_type_embeddings, landing_xy_embeddings, move_xy_embeddings), dim=-1)
        action_embeddings = self.embed_action(actions)
        time_embeddings = self.embed_timestep(timesteps)
        return_embeddings = self.embed_return(rtg)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        return_embeddings = return_embeddings + time_embeddings
        
        # this makes the sequence look like (r1, s1, a1, r2, s2, a2, ...)
        stacked_inputs = torch.stack(
            (return_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        # stack_inputs: [batch_size, 2*seq_len, hidden_size]
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']
        # x = [batch_size, seq_len, hidden_size]

        # reshape x so that the second dimension corresponds to the original
        # states (0), or actions (1); i.e. x[:,0,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        state_preds = x[:,0]
        action_preds = x[:,1]
        return_preds = x[:, 2]

        shot_preds = self.predict_shot(action_preds) # [batch_size, seq, shot_type_size]
        landing_xy = self.predict_landing(action_preds) # [batch_size, seq, 5]
        move_xy = self.predict_move(action_preds) # [batch_size, seq, 5]
        landing_mu = landing_xy[:, :, 0:2]
        landing_sigma = torch.exp(torch.clamp(landing_xy[:, :, 2:4], min=-1, max=1))
        landing_tho = self.f_tho(landing_xy[:, :, -1])

        landing_cov = torch.zeros(batch_size, seq_length, 2, 2).to(device=last_time_shot_type.device)
        landing_cov[:, :, 0, 0] = landing_sigma[:, :, 0] * landing_sigma[:, :, 0]
        landing_cov[:, :, 1, 1] = landing_sigma[:, :, 1] * landing_sigma[:, :, 1]
        landing_cov[:, :, 0, 1] = landing_sigma[:, :, 0] * landing_sigma[:, :, 1] * landing_tho
        landing_cov[:, :, 1, 0] = landing_sigma[:, :, 0] * landing_sigma[:, :, 1] * landing_tho

        move_mu = move_xy[:, :, 0:2]
        move_sigma = torch.exp(torch.clamp(move_xy[:, :, 2:4], min=-1, max=1))
        move_tho = self.f_tho(move_xy[:, :, -1])

        move_cov = torch.zeros(batch_size, seq_length, 2, 2).to(device=last_time_shot_type.device)
        move_cov[:, :, 0, 0] = move_sigma[:, :, 0] * move_sigma[:, :, 0]
        move_cov[:, :, 1, 1] = move_sigma[:, :, 1] * move_sigma[:, :, 1]
        move_cov[:, :, 0, 1] = move_sigma[:, :, 0] * move_sigma[:, :, 1] * move_tho
        move_cov[:, :, 1, 0] = move_sigma[:, :, 0] * move_sigma[:, :, 1] * move_tho

        # return_preds: shot_preds, landing_distribution
        return shot_preds, torch.distributions.MultivariateNormal(loc=landing_mu, covariance_matrix=landing_cov), torch.distributions.MultivariateNormal(loc=move_mu, covariance_matrix=move_cov)

    def get_action(self, last_time_shot_type, hit_xy, player_location_xy, opponent_location_xy, shot_type, landing_xy, move_xy, reward, rtg, timesteps, attention_mask=None):
        if len(hit_xy.shape) < 3:
            last_time_shot_type = last_time_shot_type.reshape(1, -1)
            hit_xy = hit_xy.reshape(1, -1, self.hit_xy_dim)
            player_location_xy = player_location_xy.reshape(1, -1, self.player_location_xy_dim)
            opponent_location_xy = opponent_location_xy.reshape(1, -1, self.opponent_location_xy_dim)
            shot_type = shot_type.reshape(1, -1)
            landing_xy = landing_xy.reshape(1, -1, self.landing_xy_dim)
            move_xy = move_xy.reshape(1, -1, self.move_xy_dim)
            reward = reward.reshape(1, -1, 1)
            rtg = rtg.reshape(1, -1, 1)
            timesteps = timesteps.reshape(1, -1)
            attention_mask = attention_mask.reshape(1, -1)

        shot_preds, landing_distribution, move_distribution = self.forward(last_time_shot_type, hit_xy, player_location_xy, opponent_location_xy, shot_type, landing_xy, move_xy, reward, rtg, timesteps, attention_mask)
        # logsoftmax = nn.LogSoftmax(dim=-1)
        # shot_type_probs = logsoftmax(shot_preds) # [batch_size, seq, shot_type_dim]
        # pred_shot_type = torch.argmax(shot_type_probs, dim=-1)
        return shot_preds, landing_distribution, move_distribution
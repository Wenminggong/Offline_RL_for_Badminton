# -*- coding: utf-8 -*-
'''
@File    :   transformer_for_stroke_prediction.py
@Time    :   2024/05/12 14:18:39
@Author  :   Mingjiang Liu 
@Version :   1.0
@Desc    :   transformer for stroke prediction
'''


import numpy as np
import torch
import torch.nn as nn

import transformers

from decision_transformer.models.model import TrajectoryModel
from decision_transformer.models.trajectory_gpt2 import GPT2Model


class TransformerStroke(TrajectoryModel):

    """
    This model uses GPT to model sequence prediction problem 
    input: (state_1, action_1, state_2 ...)
    output: state_i
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
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            use_player_location=1,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)
        self.last_time_shot_type_dim = last_time_shot_type_dim
        self.hit_xy_dim = hit_xy_dim
        self.player_location_xy_dim = player_location_xy_dim
        self.opponent_location_xy_dim = opponent_location_xy_dim
        self.shot_type_dim = shot_type_dim
        self.landing_xy_dim = landing_xy_dim

        self.hidden_size = hidden_size
        self.use_player_location = use_player_location
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
        self.embed_shot = nn.Linear(last_time_shot_type_dim, hidden_size // 4)
        self.embed_xy = nn.Linear(hit_xy_dim, hidden_size // 4)
        if self.use_player_location:
            self.embed_state = nn.Linear(hidden_size, hidden_size)
        else:
            self.embed_state = nn.Linear(hidden_size // 2, hidden_size)
        self.embed_action = nn.Linear(hidden_size // 2, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states 
        self.predict_shot = nn.Linear(hidden_size, shot_type_dim)
        self.predict_xy = nn.Linear(hidden_size, 5)
        self.f_mu = nn.Sigmoid()
        self.f_tho = nn.Tanh()

    def forward(self, last_time_shot_type, hit_xy, player_location_xy, opponent_location_xy, shot_type, landing_xy, timesteps, attention_mask=None):

        batch_size, seq_length = last_time_shot_type.shape[0], last_time_shot_type.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long).to(device=last_time_shot_type.device)

        # embed each modality with a different head
        last_time_shot_type_embeddings = self.embed_shot(last_time_shot_type)
        hit_xy_embeddings = self.embed_xy(hit_xy)
        if self.use_player_location:
            player_location_xy_embeddings = self.embed_xy(player_location_xy)
            opponent_location_xy_embeddings = self.embed_xy(opponent_location_xy)
        shot_type_embeddings = self.embed_shot(shot_type)
        landing_xy_embeddings = self.embed_xy(landing_xy)
        
        if self.use_player_location:
            states = torch.cat((last_time_shot_type_embeddings, hit_xy_embeddings, player_location_xy_embeddings, opponent_location_xy_embeddings), dim=-1)
        else:
            states = torch.cat((last_time_shot_type_embeddings, hit_xy_embeddings), dim=-1)
        state_embeddings = self.embed_state(states)
        actions = torch.cat((shot_type_embeddings, landing_xy_embeddings), dim=-1)
        action_embeddings = self.embed_action(actions)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        
        # this makes the sequence look like (s1, a1, s2, a2, ...)
        stacked_inputs = torch.stack(
            (state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 2*seq_length, self.hidden_size)
        # stack_inputs: [batch_size, 2*seq_len, hidden_size]
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 2*seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']
        # x = [batch_size, seq_len, hidden_size]

        # reshape x so that the second dimension corresponds to the original
        # states (0), or actions (1); i.e. x[:,0,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 2, self.hidden_size).permute(0, 2, 1, 3)

        state_preds = x[:,1]
        action_preds = x[:,0]

        shot_preds = self.predict_shot(action_preds) # [batch_size, seq, shot_type_size]
        landing_xy = self.predict_xy(action_preds) # [batch_size, seq, 5]
        landing_mu = self.f_mu(landing_xy[:, :, 0:2])
        landing_sigma = torch.exp(torch.clamp(landing_xy[:, :, 2:4], min=-1, max=1))
        landing_tho = self.f_tho(landing_xy[:, :, -1])

        cov = torch.zeros(batch_size, seq_length, 2, 2).to(device=last_time_shot_type.device)
        cov[:, :, 0, 0] = landing_sigma[:, :, 0] * landing_sigma[:, :, 0]
        cov[:, :, 1, 1] = landing_sigma[:, :, 1] * landing_sigma[:, :, 1]
        cov[:, :, 0, 1] = landing_sigma[:, :, 0] * landing_sigma[:, :, 1] * landing_tho
        cov[:, :, 1, 0] = landing_sigma[:, :, 0] * landing_sigma[:, :, 1] * landing_tho

        # return_preds: shot_preds, landing_distribution
        return shot_preds, torch.distributions.MultivariateNormal(loc=landing_mu, covariance_matrix=cov)

    def get_action(self, last_time_shot_type, hit_xy, player_location_xy, opponent_location_xy, shot_type, landing_xy, timesteps, attention_mask=None):
        if len(last_time_shot_type.shape) < 3:
            last_time_shot_type = last_time_shot_type.reshape(1, -1, self.last_time_shot_type_dim)
            hit_xy = hit_xy.reshape(1, -1, self.hit_xy_dim)
            player_location_xy = player_location_xy.reshape(1, -1, self.player_location_xy_dim)
            opponent_location_xy = opponent_location_xy.reshape(1, -1, self.opponent_location_xy_dim)
            shot_type = shot_type.reshape(1, -1, self.shot_type_dim)
            landing_xy = landing_xy.reshape(1, -1, self.landing_xy_dim)
            timesteps = timesteps.reshape(1, -1)
            attention_mask = attention_mask.reshape(1, -1)

        shot_preds, landing_distribution = self.forward(last_time_shot_type, hit_xy, player_location_xy, opponent_location_xy, shot_type, landing_xy, timesteps, attention_mask)
        # logsoftmax = nn.LogSoftmax(dim=-1)
        # shot_type_probs = logsoftmax(shot_preds) # [batch_size, seq, shot_type_dim]
        # pred_shot_type = torch.argmax(shot_type_probs, dim=-1)
        return shot_preds, landing_distribution
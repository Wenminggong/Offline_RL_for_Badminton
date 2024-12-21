# -*- coding: utf-8 -*-
'''
@File    :   fqe.py
@Time    :   2024/06/09 21:58:03
@Author  :   Mingjiang Liu 
@Version :   1.0
@Desc    :   fitted q evaluation, refer to: 2019 ICML Batch Policy Learning under Constraints.
'''


from copy import deepcopy
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
TensorBatch = List[torch.Tensor]
from offline_rl.utils import soft_update


class FQE():
    def __init__(
            self,
            critic,
            critic_optimizer,
            discount: float = 0.99,
            target_update_period: int = 1,
            soft_target_update_rate: float = 5e-3,
            deterministic: bool = False,
            device: str = "cuda",
        ) -> None:
        self.discount = discount
        self.soft_target_update_rate = soft_target_update_rate
        self.target_update_period = target_update_period
        self._deterministic = deterministic
        self._device = device

        self.critic = critic
        self.target_critic = deepcopy(self.critic).to(device)
        self.critic_optimizer = critic_optimizer

        self.total_it = 0

    def update_target_network(self, soft_target_update_rate: float):
        soft_update(self.target_critic, self.critic, soft_target_update_rate)

    def _q_loss(
            self,
            last_time_shot_type: torch.Tensor, 
            hit_xy: torch.Tensor, 
            player_location_xy: torch.Tensor, 
            opponent_location_xy: torch.Tensor, 
            shot_type: torch.Tensor, 
            landing_xy: torch.Tensor, 
            move_xy: torch.Tensor, 
            next_last_time_shot_type: torch.Tensor, 
            next_hit_xy: torch.Tensor, 
            next_player_location_xy: torch.Tensor, 
            next_opponent_location_xy: torch.Tensor,
            next_shot_type: torch.Tensor, 
            next_landing_xy: torch.Tensor, 
            next_move_xy: torch.Tensor,
            next_shot_probs: torch.Tensor,
            rewards: torch.Tensor, 
            dones: torch.Tensor
    ):
        # current step q(s,a), q(s,a): [batch_size, shot_type_dim]
        q_predicted = self.critic(last_time_shot_type, hit_xy, player_location_xy, opponent_location_xy, shot_type, landing_xy, move_xy)

        with torch.no_grad():
            # get next step q(s,a): [batch_size, shot_type_dim]
            next_q_pred = self.target_critic(next_last_time_shot_type, next_hit_xy, next_player_location_xy, next_opponent_location_xy, next_shot_type, next_landing_xy, next_move_xy)
        if not self._deterministic:
            # [batch_size, 1]
            next_q_pred = (next_shot_probs * next_q_pred).sum(dim=-1, keepdim=True)
            # td_target = rewards + (1.0 - dones) * self.discount * (next_shot_probs * next_q_pred).sum(dim=-1, keepdim=True)
        else:
            # [batch_size, 1]
            next_q_pred = next_q_pred[next_shot_type>0].unsqueeze(dim=-1)
        
        td_target = rewards + (1.0 - dones) * self.discount * next_q_pred
        qf_loss = F.mse_loss(q_predicted[shot_type>0], td_target.squeeze(dim=-1))

        log_dict = {
            "q_loss": qf_loss.item(),
            "mean_q_value": q_predicted[shot_type>0].mean().item(),
            "mean_target_next_q_value": next_q_pred.mean().item()
        }

        return qf_loss, log_dict

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        (
            last_time_shot_type, 
            hit_xy, 
            player_location_xy, 
            opponent_location_xy, 
            shot_type, 
            landing_xy, 
            move_xy, 
            next_last_time_shot_type, 
            next_hit_xy, 
            next_player_location_xy, 
            next_opponent_location_xy,
            next_shot_type, 
            next_landing_xy, 
            next_move_xy,
            next_shot_probs,
            rewards, 
            dones
        ) = batch
        self.total_it += 1

        """ Q function loss """
        qf_loss, log_dict = self._q_loss(
            last_time_shot_type, 
            hit_xy, 
            player_location_xy, 
            opponent_location_xy, 
            shot_type, 
            landing_xy, 
            move_xy, 
            next_last_time_shot_type, 
            next_hit_xy, 
            next_player_location_xy, 
            next_opponent_location_xy,
            next_shot_type, 
            next_landing_xy, 
            next_move_xy,
            next_shot_probs,
            rewards, 
            dones
        )

        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.target_update_period == 0:
            self.update_target_network(self.soft_target_update_rate)

        return log_dict
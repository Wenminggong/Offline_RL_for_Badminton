# -*- coding: utf-8 -*-
'''
@File    :   bc.py
@Time    :   2024/06/14 15:42:59
@Author  :   Mingjiang Liu 
@Version :   1.0
@Desc    :   BC algorithm
'''


import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from typing import Any, Dict, List, Optional, Tuple, Union
TensorBatch = List[torch.Tensor]


class BC():
    def __init__(
        self,
        actor,
        actor_optimizer,
        policy_lr: float = 1e-4,
        eval_freq: int = 100,
        device: str = "cuda",
    ) -> None:

        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.policy_lr = policy_lr
        self.eval_freq = eval_freq
        self.device = device

        self.total_it = 0

    def _policy_loss(
        self, 
        shot_type: torch.Tensor,
        log_shot_probs: torch.Tensor,
        landing_log_prob: torch.Tensor,
        move_log_prob: torch.Tensor,
        log_type: str = "training",
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        shot_type_ce = - log_shot_probs[shot_type > 0]
        shot_type_ce_mean = shot_type_ce.mean()
        shot_type_ce_std = shot_type_ce.std()
        landing_xy_lh = - landing_log_prob
        landing_xy_lh_mean = landing_xy_lh.mean()
        landing_xy_lh_std = landing_xy_lh.std()
        move_xy_lh = - move_log_prob
        move_xy_lh_mean = move_xy_lh.mean()
        move_xy_lh_std = move_xy_lh.std()
        policy_loss = shot_type_ce_mean + landing_xy_lh_mean + move_xy_lh_mean
        log_dict = {
            f"{log_type}/loss_mean": policy_loss.cpu().item(),
            f"{log_type}/shot_type_mean": shot_type_ce_mean.cpu().item(),
            f"{log_type}/shot_type_std": shot_type_ce_std.cpu().item(),
            f"{log_type}/landing_xy_mean": landing_xy_lh_mean.cpu().item(),
            f"{log_type}/landing_xy_std": landing_xy_lh_std.cpu().item(),
            f"{log_type}/move_xy_mean": move_xy_lh_mean.cpu().item(),
            f"{log_type}/move_xy_std": move_xy_lh_std.cpu().item()
        }
        return policy_loss, log_dict
    
    def _evaluation(
            self,
            eval_batch: TensorBatch
    ) -> Dict[str, float]:
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
            rewards, 
            dones
        ) = eval_batch

        with torch.no_grad():
            # [batch_size, shot_type_dim], [batch_size, 1]
            log_shot_probs, landing_log_prob, move_log_prob = self.actor.log_prob(
                last_time_shot_type, 
                hit_xy, 
                player_location_xy,
                opponent_location_xy,
                shot_type,
                landing_xy,
                move_xy
            )

        policy_loss, log_dict = self._policy_loss(shot_type, log_shot_probs, landing_log_prob, move_log_prob, log_type="evaluation")
        pred_shot_type = torch.argmax(log_shot_probs, dim=-1) # [batch_size]
        target_shot_type = torch.where(shot_type > 0)[1]
        same_type = (pred_shot_type == target_shot_type)
        same_type_rate = same_type.sum() / len(same_type)
        log_dict.update({"evaluation/same_type_rate": same_type_rate.cpu().item()})
        return log_dict
        

    def train(self, train_batch: TensorBatch, eval_batch: TensorBatch) -> Dict[str, float]:
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
            rewards, 
            dones
        ) = train_batch
        
        self.total_it += 1

        # [batch_size, shot_type_dim], [batch_size, 1]
        log_shot_probs, landing_log_prob, move_log_prob = self.actor.log_prob(
            last_time_shot_type, 
            hit_xy, 
            player_location_xy,
            opponent_location_xy,
            shot_type,
            landing_xy,
            move_xy
        )

        policy_loss, log_dict = self._policy_loss(shot_type, log_shot_probs, landing_log_prob, move_log_prob)

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        if self.total_it % self.eval_freq == 0:
            # evaluate policy
            eval_log = self._evaluation(eval_batch)
            log_dict.update(eval_log)

        return log_dict
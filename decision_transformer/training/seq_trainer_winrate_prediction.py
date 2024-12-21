# -*- coding: utf-8 -*-
'''
@File    :   seq_trainer_winrate_prediction.py
@Time    :   2024/07/08 17:23:16
@Author  :   Mingjiang Liu 
@Version :   1.0
@Desc    :   trainer for win rate prediction
'''


import numpy as np
import torch
import time

from decision_transformer.training.trainer import Trainer


class SequenceTrainerPrediction(Trainer):
    
    def train_step(self):
        # get batch_size sequence data
        player_id, shot_type, hit_area, hit_xy, player_area, player_xy, opponent_area, opponent_xy, landing_area, landing_xy, timesteps, time, posture_fea, landing_fea, rally_info, mask, reward = self.get_batch(self.batch_size)

        # get model predictions, [batch_size, 1]
        win_probs = self.model.forward(
            player_id, 
            shot_type, 
            hit_area, 
            hit_xy, 
            player_area, 
            player_xy, 
            opponent_area, 
            opponent_xy, 
            landing_area, 
            landing_xy, 
            timesteps, 
            time, 
            posture_fea, 
            landing_fea, 
            rally_info,
            mask
        )
        # reward: [batch_size, max_len, 1]
        loss = self.loss_fn(win_probs, reward, mask)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        return loss.detach().cpu().item()
# -*- coding: utf-8 -*-
'''
@File    :   seq_trainer_tactics_generation.py
@Time    :   2024/05/16 22:30:02
@Author  :   Mingjiang Liu 
@Version :   1.0
@Desc    :   seq_trainer for Decision Transformer tactics generation
'''


import numpy as np
import torch
import time

from decision_transformer.training.trainer import Trainer


class SequenceTrainerTactics(Trainer):

    def train_iteration(self, num_steps, iter_num=0, print_logs=False):

        train_losses = []
        type_losses = []
        landing_losses = []
        move_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        for _ in range(num_steps):
            train_loss, type_loss, landing_loss, move_loss = self.train_step()
            train_losses.append(train_loss)
            type_losses.append(type_loss)
            landing_losses.append(landing_loss)
            move_losses.append(move_loss)
            if self.scheduler is not None:
                self.scheduler.step()

        logs['time/training'] = time.time() - train_start

        eval_start = time.time()

        self.model.eval()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)
        logs['training/type_loss_mean'] = np.mean(type_losses)
        logs['training/type_loss_std'] = np.std(type_losses)
        logs['training/landing_loss_mean'] = np.mean(landing_losses)
        logs['training/landing_loss_std'] = np.std(landing_losses)
        logs['training/move_loss_mean'] = np.mean(move_losses)
        logs['training/move_loss_std'] = np.std(move_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    def train_step(self):
        # get batch_size sequence data
        last_time_shot_type, hit_xy, player_location_xy, opponent_location_xy, shot_type, landing_xy, move_xy, reward, rtg, timesteps, attention_mask, move_mask = self.get_batch(self.batch_size)

        # get model predictions
        shot_preds, landing_distribution, move_distribution = self.model.forward(
            last_time_shot_type, hit_xy, player_location_xy, opponent_location_xy, shot_type, landing_xy, move_xy, reward, rtg, timesteps, attention_mask=attention_mask,
        )

        loss, type_loss, landing_loss, move_loss = self.loss_fn(shot_type, landing_xy, move_xy, shot_preds, landing_distribution, move_distribution, attention_mask, move_mask)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        return loss.detach().cpu().item(), type_loss.detach().cpu().item(), landing_loss.detach().cpu().item(), move_loss.detach().cpu().item()
# -*- coding: utf-8 -*-
'''
@File    :   evaluator.py
@Time    :   2024/04/20 11:11:22
@Author  :   Mingjiang Liu 
@Version :   1.0
@Desc    :   None
'''


import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, recall_score, precision_score, average_precision_score
from data.preprocess_badminton_data import ACTIONS
from decision_transformer.utils import get_batch_data_from_shuttleset
from prediction.buffer.replay_buffer import ReplayBuffer
from prediction.utils import convert_rally_to_whole


def discount_cumsum(x, gamma):
    # return discount return for every timestep
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


class PassiveEvaluator():
    def __init__(self) -> None:
        pass

    def action_generation(self, rally, state_mean, state_std, model, device):
        # rally: {"observations": array, ...}
        states = (rally["observations"] - state_mean) / state_std # n x k
        rewards = rally["rewards"] # n
        actions = rally["actions"] # n x action_dims

        timesteps = np.arange(len(rewards)) # n

        states = torch.from_numpy(states).to(dtype=torch.float32, device=device)
        actions = torch.from_numpy(actions).to(dtype=torch.float32, device=device)
        rewards = torch.from_numpy(rewards).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.long, device=device)

        model.eval()
        with torch.no_grad():
            q_preds = model.get_action(states, actions, rewards, rewards, timesteps).cpu() # n x 1

        return q_preds
    
    def terminal_q_same(self, rewards, q_preds):
        rewards = rewards.reshape(-1)
        q_preds = q_preds.reshape(-1)
        terminal = rewards[-1] * q_preds[-1]
        return 1 if terminal >= 0 else 0
    
    def save_pred_q(self, data_path, save_path, rewards, q_preds):
        rally_data = pd.read_csv(data_path)
        bottom_rally_data = rally_data[rally_data["pos"] == "bottom"].copy()
        bottom_rally_data.loc[:, "reward"] = rewards
        bottom_rally_data.loc[:, "pred_q"] = q_preds

        # save_path = os.path.join(os.path.dirname(__file__), save_path)
        save_path = os.path.join(save_path, data_path.split("/")[-2])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, data_path.split("/")[-1])
        bottom_rally_data.to_csv(save_path, index=False)


class ActiveEvaluator():
    def __init__(self) -> None:
        pass

    def action_generation(self, rally, state_mean, state_std, model, device, use_win_return=0):
        # output pred_actions
        # rally: {"observations": array, ...}
        states = (rally["observations"] - state_mean) / state_std # n x k
        rewards = rally["rewards"] # n
        actions = rally["actions"] # n x action_dims
        if use_win_return:
            returns = np.ones_like(rewards)
        else:
            returns = discount_cumsum(rewards, 1) # n
        win_action_num = rewards.sum()
        win_action_num = max(win_action_num, 0) * rewards.shape[0]
        if win_action_num == 0:
            loss_rally = 1
        else:
            loss_rally = 0
        # terminals = rally["terminals"] # n
        timesteps = np.arange(len(rewards)) # n

        states = torch.from_numpy(states).to(dtype=torch.float32, device=device)
        actions = torch.from_numpy(actions).to(dtype=torch.float32, device=device)
        rewards = torch.from_numpy(rewards).to(dtype=torch.float32, device=device)
        returns = torch.from_numpy(returns).to(dtype=torch.float32, device=device)
        # terminals = torch.from_numpy(terminals).to(dtype=torch.long, device=device)
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.long, device=device)

        model.eval()
        with torch.no_grad():
            action_probs = model.get_action(states, actions, rewards, returns, timesteps) # n x nxaction_dims
        logsoftmax_func = nn.LogSoftmax(dim=1)
        action_probs = logsoftmax_func(action_probs) # n x action_dims probabilities

        action_classes = torch.where(actions>0)[1].cpu() # n dims
        pred_action_classes = torch.argmax(action_probs, dim=1).cpu() # n dims

        # action_classes: n-tensor, pred_action_classes: n-tensor
        return action_classes, pred_action_classes, win_action_num, loss_rally

    def action_diff(self, action_classes, pred_action_classes):
        # output nums of pred_action = label_action
        action_num = len(action_classes)
        same_action_num = torch.sum(action_classes == pred_action_classes).item()
        return action_num, same_action_num
    
    def reasonable_action_diff(self, rally, relation_table, action_classes, pred_action_classes):
        # output 1) the rally nums have different actions; 2) the nums of reasonable pred actions except for service action; 3) the nums of different action in 2).
        diff_rally = 0
        reasonable_action_num = 0
        reasonable_diff_action_num = 0
        I_ACTIONS = {v:k for k,v in ACTIONS.items()}
        for index in range(len(action_classes)):
            if not (ACTIONS["short service"] == action_classes[index] or ACTIONS["long service"] == action_classes[index]):
            # not service action
                if not rally["last_actions"][index] in I_ACTIONS.keys():
                    reasonable_action_num += 1
                    if action_classes[index] != pred_action_classes[index]:
                        diff_rally = 1
                        reasonable_diff_action_num += 1
                    continue
                if relation_table.loc[I_ACTIONS[rally["last_actions"][index]], I_ACTIONS[pred_action_classes[index].item()]] > 10:
                    reasonable_action_num += 1
                    if action_classes[index] != pred_action_classes[index]:
                        diff_rally = 1
                        reasonable_diff_action_num += 1
        return diff_rally, reasonable_action_num, reasonable_diff_action_num


    def service_action_diff(self, action_classes, pred_action_classes):
        service_action_index = []
        pred_service_num = 0
        for index in range(len(action_classes)):
            if ACTIONS["short service"] == action_classes[index] or ACTIONS["long service"] == action_classes[index]:
                service_action_index.append(index)
            if ACTIONS["short service"] == pred_action_classes[index] or ACTIONS["long service"] == pred_action_classes[index]:
                pred_service_num += 1
        
        if len(service_action_index) == 0:
            return 0, 0, 0, pred_service_num

        service_action_num = len(service_action_index)
        same_service_action_num = torch.sum(action_classes[service_action_index] == pred_action_classes[service_action_index]).item()
        use_service_action_num = torch.sum(torch.logical_or(pred_action_classes[service_action_index] == ACTIONS["short service"], pred_action_classes[service_action_index] == ACTIONS["long service"])).item()
        
        return service_action_num, same_service_action_num, use_service_action_num, pred_service_num
    
    def other_unreasonable_action(self, rally, relation_table, action_classes, pred_action_classes):
        # output: unreasonable action nums except for service action and total action nums
        # rally: {"observations": array, ...}
        num = 0
        I_ACTIONS = {v:k for k,v in ACTIONS.items()}
        for index in range(len(action_classes)):
            if not (ACTIONS["short service"] == action_classes[index] or ACTIONS["long service"] == action_classes[index]):
            # if not service action
                if not rally["last_actions"][index] in I_ACTIONS.keys():
                    continue
                if relation_table.loc[I_ACTIONS[rally["last_actions"][index]], I_ACTIONS[pred_action_classes[index].item()]] <= 10:
                    num += 1
        return num, len(action_classes)

    def save_action_diff_loss_rally(self, loss_rally, action_num, same_action_num, pred_action_classes, data_path, save_path):
        if (not loss_rally) or (action_num == same_action_num):
            return
        
        rally_data = pd.read_csv(data_path)
        bottom_rally_data = rally_data[rally_data["pos"] == "bottom"].copy()
        court_data_path = data_path.replace("data4drl", "data4acreg")
        if not os.path.exists(court_data_path):
            return
        court_rally_data = pd.read_csv(court_data_path)

        bottom_rally_data.loc[:, "court"] = court_rally_data.loc[0, "court"]
        bottom_rally_data.loc[:, "net"] = court_rally_data.loc[0, "net"]
        new_action = {v: k for k, v in ACTIONS.items()}
        pred_actions = [new_action[key.item()] for key in pred_action_classes]
        bottom_rally_data.loc[:, "pred_action"] = pred_actions

        # save_path = os.path.join(os.path.dirname(__file__), save_path)
        save_path = os.path.join(save_path, data_path.split("/")[-2])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, data_path.split("/")[-1])
        bottom_rally_data.to_csv(save_path, index=False)


class TransformerStrokeEvaluator():
    # evaluator for TransformerStroke evaluation
    def __init__(self, dataset_path, batch_size, sample_num, device) -> None:
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.sample_num = sample_num
        self.device = device
        with open(self.dataset_path, 'rb') as f:
            self.trajectories = pickle.load(f)

    def action_generation(self, model, last_time_shot_type_dim, hit_xy_dim, player_location_xy_dim, opponent_location_xy_dim, shot_type_dim, landing_xy_dim, move_xy_dim):
        rally_num = len(self.trajectories)
        iter_num = rally_num // self.batch_size
        self.shot_type_list = []
        self.landing_xy_list = []
        self.mask_list = []
        self.shot_pred_list = []
        self.landing_xy_sample_list = []
        self.max_seq_len = 0
        for i in range(iter_num+1):
            cur_trajectories = self.trajectories[i*self.batch_size: min((i+1)*self.batch_size, len(self.trajectories))]
            last_time_shot_type, hit_xy, player_location_xy, opponent_location_xy, shot_type, landing_xy, move_xy, reward, rtg, timesteps, mask, _ = get_batch_data_from_shuttleset(cur_trajectories, self.device, last_time_shot_type_dim, hit_xy_dim, player_location_xy_dim, opponent_location_xy_dim, shot_type_dim, landing_xy_dim, move_xy_dim)
            self.max_seq_len = max(self.max_seq_len, last_time_shot_type.shape[1])
            self.shot_type_list.append(shot_type)
            self.landing_xy_list.append(landing_xy)
            self.mask_list.append(mask)
            model.eval()
            with torch.no_grad():
                shot_preds, landing_distribution = model.get_action(last_time_shot_type, hit_xy, player_location_xy, opponent_location_xy, shot_type, landing_xy, timesteps, mask)
            self.shot_pred_list.append(shot_preds)

            landing_xy_samples = landing_distribution.sample([self.sample_num]) # [sample_num, batch_size, seq, 2]
            self.landing_xy_sample_list.append(landing_xy_samples)


    def get_eval_result(self, shot_type_dim, landing_xy_dim):
        type_loss = nn.CrossEntropyLoss()
        mae_loss = nn.L1Loss(reduce=False)
        mse_loss = nn.MSELoss(reduce=False)
        # pading sequence
        for i in range(len(self.shot_type_list)):
            cur_batch_size = self.shot_type_list[i].shape[0]
            cur_seq_len = self.shot_type_list[i].shape[1]
            self.shot_type_list[i] = torch.cat([torch.zeros(cur_batch_size, self.max_seq_len-cur_seq_len, shot_type_dim).to(self.device), self.shot_type_list[i]], axis=1)
            self.shot_pred_list[i] = torch.cat([torch.zeros(cur_batch_size, self.max_seq_len-cur_seq_len, shot_type_dim).to(self.device), self.shot_pred_list[i]], axis=1)
            self.landing_xy_list[i] = torch.cat([torch.zeros(cur_batch_size, self.max_seq_len-cur_seq_len, landing_xy_dim).to(self.device), self.landing_xy_list[i]], axis=1)
            self.landing_xy_sample_list[i] = torch.cat([torch.zeros(self.sample_num, cur_batch_size, self.max_seq_len-cur_seq_len, landing_xy_dim).to(self.device), self.landing_xy_sample_list[i]], axis=2)
            self.mask_list[i] = torch.cat([torch.zeros(cur_batch_size, self.max_seq_len-cur_seq_len).to(self.device), self.mask_list[i]], axis=1)

        shot_type = torch.cat(self.shot_type_list, dim=0)
        landing_xy = torch.cat(self.landing_xy_list, dim=0)
        mask = torch.cat(self.mask_list, dim=0)
        shot_preds = torch.cat(self.shot_pred_list, dim=0)
        landing_xy_samples = torch.cat(self.landing_xy_sample_list, dim=1)

        shot_type_target = shot_type.reshape(-1, shot_type_dim)[mask.reshape(-1) > 0]
        shot_preds = shot_preds.reshape(-1, shot_type_dim)[mask.reshape(-1) > 0]
        # one-hot to class-index
        shot_type_target = torch.where(shot_type_target>0)[1]
        # cross-entropy loss
        type_loss_value = type_loss(shot_preds, shot_type_target)
        ce_loss = type_loss_value.detach().cpu().item()

        landing_xy = landing_xy.unsqueeze(0)
        landing_xy = landing_xy.repeat(self.sample_num, 1, 1, 1)
        landing_xy_mae = mae_loss(landing_xy_samples, landing_xy)
        landing_xy_mae = torch.sum(landing_xy_mae, dim=-1)
        min_landing_xy_mae = torch.min(landing_xy_mae, dim=0)[0]
        mae_loss = min_landing_xy_mae[mask > 0].mean().cpu().item()

        landing_xy_mse = mse_loss(landing_xy_samples, landing_xy)
        landing_xy_mse = torch.sum(landing_xy_mse, dim=-1)
        min_landing_xy_mse = torch.min(landing_xy_mse, dim=0)[0] 
        mse_loss = min_landing_xy_mse[mask > 0].mean().cpu().item()

        return ce_loss, mae_loss, mse_loss
    

class DecisionTransformerTacticsEvaluator():
    # evaluator for DecisionTransformerTactics evaluation
    def __init__(self, dataset_path, batch_size, sample_num, device) -> None:
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.sample_num = sample_num
        self.device = device
        with open(self.dataset_path, 'rb') as f:
            self.trajectories = pickle.load(f)

    def action_generation(self, model):
        rally_num = len(self.trajectories)
        iter_num = rally_num // self.batch_size
        self.shot_type_list = []
        self.landing_xy_list = []
        self.move_xy_list = []
        self.mask_list = []
        self.move_mask_list = []
        self.shot_pred_list = []
        self.landing_xy_sample_list = []
        self.move_xy_sample_list = []
        self.landing_xy_logprobs_list = []
        self.move_xy_logprobs_list = []
        self.max_seq_len = 0
        for i in range(iter_num+1):
            cur_trajectories = self.trajectories[i*self.batch_size: min((i+1)*self.batch_size, len(self.trajectories))]
            batch_data = get_batch_data_from_shuttleset(cur_trajectories, self.device)
            last_time_shot_type = batch_data["last_time_opponent_type"].squeeze().to(dtype=torch.long) # [batch_size, max_len]
            hit_xy = batch_data["hit_xy"].to(dtype=torch.float32) # [batch_size, max_len, 2]
            player_location_xy = batch_data["player_location_xy"].to(dtype=torch.float32) # [batch_size, max_len, 2]
            opponent_location_xy = batch_data["opponent_location_xy"].to(dtype=torch.float32) # [batch_size, max_len, 2]
            shot_type = batch_data["shot_type"].squeeze().to(dtype=torch.long) # [batch_size, max_len]
            landing_xy = batch_data["landing_xy"].to(dtype=torch.float32) # [batch_size, max_len, 2]
            move_xy = batch_data["move_xy"].to(dtype=torch.float32) # [batch_size, max_len, 2]
            reward = batch_data["reward"].to(dtype=torch.float32) # [batch_size, max_len, 1]
            timesteps = batch_data["timesteps"].squeeze().to(dtype=torch.long) # [batch_size, max_len]
            rtg = batch_data["rtg"].to(dtype=torch.float32) # [batch_size, max_len, 1]
            mask = batch_data["mask"].to(dtype=torch.float32) # [batch_size, max_len]
            move_mask = batch_data["move_mask"].to(dtype=torch.float32) # [batch_size, max_len]

            self.max_seq_len = max(self.max_seq_len, last_time_shot_type.shape[1])
            self.shot_type_list.append(shot_type)
            self.landing_xy_list.append(landing_xy)
            self.move_xy_list.append(move_xy)
            self.mask_list.append(mask)
            self.move_mask_list.append(move_mask)
            model.eval()
            with torch.no_grad():
                shot_preds, landing_distribution, move_distribution = model.get_action(last_time_shot_type, hit_xy, player_location_xy, opponent_location_xy, shot_type, landing_xy, move_xy, reward, rtg, timesteps, mask)
            self.shot_pred_list.append(shot_preds)

            landing_xy_samples = landing_distribution.sample([self.sample_num]) # [sample_num, batch_size, seq, 2]
            self.landing_xy_sample_list.append(landing_xy_samples)
            move_xy_samples = move_distribution.sample([self.sample_num]) # [sample_num, batch_size, seq, 2]
            self.move_xy_sample_list.append(move_xy_samples)
            landing_xy_logprobs = landing_distribution.log_prob(landing_xy)
            self.landing_xy_logprobs_list.append(landing_xy_logprobs)
            move_xy_logprobs = move_distribution.log_prob(move_xy)
            self.move_xy_logprobs_list.append(move_xy_logprobs)


    def get_eval_result(self, shot_type_dim, landing_xy_dim, move_xy_dim):
        type_loss = nn.CrossEntropyLoss()
        mae_loss = nn.L1Loss(reduce=False)
        mse_loss = nn.MSELoss(reduce=False)
        # pading sequence
        for i in range(len(self.shot_type_list)):
            cur_batch_size = self.shot_type_list[i].shape[0]
            cur_seq_len = self.shot_type_list[i].shape[1]
            self.shot_type_list[i] = torch.cat([torch.zeros(cur_batch_size, self.max_seq_len-cur_seq_len).to(self.device), self.shot_type_list[i]], dim=1)
            self.shot_pred_list[i] = torch.cat([torch.zeros(cur_batch_size, self.max_seq_len-cur_seq_len, shot_type_dim).to(self.device), self.shot_pred_list[i]], dim=1)
            self.landing_xy_list[i] = torch.cat([torch.zeros(cur_batch_size, self.max_seq_len-cur_seq_len, landing_xy_dim).to(self.device), self.landing_xy_list[i]], dim=1)
            self.landing_xy_sample_list[i] = torch.cat([torch.zeros(self.sample_num, cur_batch_size, self.max_seq_len-cur_seq_len, landing_xy_dim).to(self.device), self.landing_xy_sample_list[i]], dim=2)
            self.move_xy_list[i] = torch.cat([torch.zeros(cur_batch_size, self.max_seq_len-cur_seq_len, move_xy_dim).to(self.device), self.move_xy_list[i]], dim=1)
            self.move_xy_sample_list[i] = torch.cat([torch.zeros(self.sample_num, cur_batch_size, self.max_seq_len-cur_seq_len, move_xy_dim).to(self.device), self.move_xy_sample_list[i]], dim=2)
            self.mask_list[i] = torch.cat([torch.zeros(cur_batch_size, self.max_seq_len-cur_seq_len).to(self.device), self.mask_list[i]], dim=1)
            self.move_mask_list[i] = torch.cat([torch.zeros(cur_batch_size, self.max_seq_len-cur_seq_len).to(self.device), self.move_mask_list[i]], dim=1)
            self.landing_xy_logprobs_list[i] = torch.cat([torch.zeros(cur_batch_size, self.max_seq_len-cur_seq_len).to(self.device), self.landing_xy_logprobs_list[i]], dim=1)
            self.move_xy_logprobs_list[i] = torch.cat([torch.zeros(cur_batch_size, self.max_seq_len-cur_seq_len).to(self.device), self.move_xy_logprobs_list[i]], dim=1)

        shot_type = torch.cat(self.shot_type_list, dim=0).to(dtype=torch.long) # [batch_size, max_len]
        landing_xy = torch.cat(self.landing_xy_list, dim=0)
        move_xy = torch.cat(self.move_xy_list, dim=0) 
        mask = torch.cat(self.mask_list, dim=0) # [batch_size, max_len]
        move_mask = torch.cat(self.move_mask_list, dim=0) # [batch_size, max_len]
        shot_preds = torch.cat(self.shot_pred_list, dim=0)
        landing_xy_samples = torch.cat(self.landing_xy_sample_list, dim=1)
        move_xy_samples = torch.cat(self.move_xy_sample_list, dim=1)
        landing_xy_logprobs = torch.cat(self.landing_xy_logprobs_list, dim=0)
        move_xy_logprobs = torch.cat(self.move_xy_logprobs_list, dim=0)

        shot_type_target = shot_type.reshape(-1)[mask.reshape(-1) > 0]
        shot_preds = shot_preds.reshape(-1, shot_type_dim)[mask.reshape(-1) > 0]
        
        # cross-entropy loss
        type_loss_value = type_loss(shot_preds, shot_type_target)
        ce_loss = type_loss_value.detach().cpu().item()
        same_type = torch.argmax(shot_preds, dim=-1) == shot_type_target
        same_type_rate = same_type.sum() / len(same_type)
        same_type_rate = same_type_rate.cpu().item()

        landing_xy = landing_xy.unsqueeze(0)
        landing_xy = landing_xy.repeat(self.sample_num, 1, 1, 1)
        landing_xy_mae = mae_loss(landing_xy_samples, landing_xy)
        landing_xy_mae = torch.sum(landing_xy_mae, dim=-1)
        min_landing_xy_mae = torch.min(landing_xy_mae, dim=0)[0]
        landing_mae_loss = min_landing_xy_mae[mask > 0].mean().cpu().item()

        landing_xy_mse = mse_loss(landing_xy_samples, landing_xy)
        landing_xy_mse = torch.sum(landing_xy_mse, dim=-1)
        min_landing_xy_mse = torch.min(landing_xy_mse, dim=0)[0] 
        landing_mse_loss = min_landing_xy_mse[mask > 0].mean().cpu().item()

        move_xy = move_xy.unsqueeze(0)
        move_xy = move_xy.repeat(self.sample_num, 1, 1, 1)
        move_xy_mae = mae_loss(move_xy_samples, move_xy)
        move_xy_mae = torch.sum(move_xy_mae, dim=-1)
        min_move_xy_mae = torch.min(move_xy_mae, dim=0)[0]
        move_mae_loss = min_move_xy_mae[move_mask > 0].mean().cpu().item()

        move_xy_mse = mse_loss(move_xy_samples, move_xy)
        move_xy_mse = torch.sum(move_xy_mse, dim=-1)
        min_move_xy_mse = torch.min(move_xy_mse, dim=0)[0] 
        move_mse_loss = min_move_xy_mse[move_mask > 0].mean().cpu().item()

        landing_lh_loss = -landing_xy_logprobs[mask > 0].mean().cpu().item()
        move_lh_loss = -move_xy_logprobs[move_mask > 0].mean().cpu().item()

        return ce_loss, same_type_rate, landing_lh_loss, move_lh_loss, landing_mae_loss, landing_mse_loss, move_mae_loss, move_mse_loss


class WinRatePredictionEvaluator():
    # evaluator for rally win rate prediction
    def __init__(self, dataset_path, batch_size, device) -> None:
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.device = device
        with open(self.dataset_path, 'rb') as f:
            self.trajectories = pickle.load(f)

    def action_generation(self, model):
        rally_num = len(self.trajectories)
        iter_num = rally_num // self.batch_size
        self.win_probs_list = []
        self.reward_list = []
        self.max_seq_len = 0
        self.lose_reason_list = []
        for i in range(iter_num+1):
            cur_trajectories = self.trajectories[i*self.batch_size: min((i+1)*self.batch_size, len(self.trajectories))]
            batch_data = get_batch_data_from_shuttleset(cur_trajectories, self.device)
            player_id = batch_data["player"].squeeze().to(dtype=torch.long) # [batch_size, max_len]
            shot_type = batch_data["shot_type"].squeeze().to(dtype=torch.long) # [batch_size, max_len]
            hit_area = batch_data["hit_area"].squeeze().to(dtype=torch.long)-1 # [batch_size, max_len]
            hit_xy = batch_data["hit_xy"].to(dtype=torch.float32)
            player_area = batch_data["player_location_area"].squeeze().to(dtype=torch.long)-1 # [batch_size, max_len]
            player_xy = batch_data["player_location_xy"].to(dtype=torch.float32)
            opponent_area = batch_data["opponent_location_area"].squeeze().to(dtype=torch.long)-1 # [batch_size, max_len]
            opponent_xy = batch_data["opponent_location_xy"].to(dtype=torch.float32) 
            landing_area = batch_data["landing_area"].squeeze().to(dtype=torch.long)-1 # [batch_size, max_len]
            landing_xy = batch_data["landing_xy"].to(dtype=torch.float32) 
            timesteps = batch_data["timesteps"].squeeze().to(dtype=torch.long) # [batch_size, max_len]
            time = batch_data["frame_num"].to(dtype=torch.float32)
            posture_fea = torch.cat((batch_data["around_head"], batch_data["back_hand"]), dim=-1).to(dtype=torch.float32)
            landing_fea = torch.cat((batch_data["bad_landing_flag"], batch_data["landing_distance_opponent"]), dim=-1).to(dtype=torch.float32)
            rally_info =  torch.cat((batch_data["score_diff"], batch_data["cons_score"]), dim=-1).to(dtype=torch.float32)
            mask = batch_data["mask"].to(dtype=torch.float32) # [batch_size, max_len]
            mask = 1 - mask # for src_key_padding_mask, 1 or True will ignore; 
            reward = batch_data["reward"].to(dtype=torch.float32)
            self.lose_reason_list += batch_data["lose_reason"]

            hit_area[hit_area < 0] = 0
            player_area[player_area < 0] = 0
            opponent_area[opponent_area < 0] = 0
            landing_area[landing_area < 0] = 0 
            
            self.max_seq_len = max(self.max_seq_len, reward.shape[1])
            self.reward_list.append(reward)
            
            model.eval()
            with torch.no_grad():
                # [batch_size, 1]
                win_probs = model.forward(
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
            self.win_probs_list.append(win_probs)


    def get_eval_result(self):
        # pading sequence
        for i in range(len(self.reward_list)):
            cur_batch_size = self.reward_list[i].shape[0]
            cur_seq_len = self.reward_list[i].shape[1]
            self.reward_list[i] = torch.cat([torch.zeros(cur_batch_size, self.max_seq_len-cur_seq_len, 1).to(self.device), self.reward_list[i]], dim=1)

        reward = torch.cat(self.reward_list, dim=0) # [batch, max_len, 1]
        win_probs = torch.cat(self.win_probs_list, dim=0) # [batch, 1]
        
        # brier score
        bs = torch.square(win_probs - reward[:, -1, :])
        bs = bs.mean().cpu().item()

        # AUC
        fpr, tpr, thresholds = roc_curve(reward[:, -1, 0].cpu().numpy(), win_probs.squeeze().cpu().numpy())
        roc_auc = auc(fpr, tpr)

        out_right_pred_num, out_num = 0, 0
        touch_right_pred_num, touch_num = 0, 0
        nopass_right_pred_num, nopass_num = 0, 0
        land_right_pred_num, land_num = 0, 0
        win_probs = (win_probs > 0.5).to(dtype=torch.float32)
        for i in range(len(self.lose_reason_list)):
            if self.lose_reason_list[i] == "out":
                if win_probs[i, 0] == reward[i,-1,0]:
                    out_right_pred_num += 1
                out_num += 1
            elif self.lose_reason_list[i] == "touched the net":
                if win_probs[i, 0] == reward[i,-1,0]:
                    touch_right_pred_num += 1
                touch_num += 1
            elif self.lose_reason_list[i] == "not pass over the net":
                if win_probs[i, 0] == reward[i,-1,0]:
                    nopass_right_pred_num += 1
                nopass_num += 1
            else:
                if win_probs[i, 0] == reward[i,-1,0]:
                    land_right_pred_num += 1
                land_num += 1
        out_right_rate = out_right_pred_num / (out_num+1e-9)
        touch_right_rate = touch_right_pred_num / (touch_num+1e-9)
        nopass_right_rate = nopass_right_pred_num / (nopass_num+1e-9)
        land_right_rate = land_right_pred_num / (land_num+1e-9)
        right_rate = (out_right_pred_num+touch_right_pred_num+nopass_right_pred_num+land_right_pred_num) / (out_num+touch_num+nopass_num+land_num)
        return bs, roc_auc, right_rate, out_right_rate, touch_right_rate, nopass_right_rate, land_right_rate
    

class SequencePredictionEvaluator():
    # evaluator for rally win rate prediction
    def __init__(self, dataset_path, batch_size, device) -> None:
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.device = device
        with open(self.dataset_path, 'rb') as f:
            self.trajectories = pickle.load(f)

    def action_generation(self, model):
        rally_num = len(self.trajectories)
        iter_num = rally_num // self.batch_size
        self.win_probs_list = []
        self.reward_list = []
        self.mask_list = []
        self.max_seq_len = 0
        self.lose_reason_list = []
        for i in range(iter_num+1):
            cur_trajectories = self.trajectories[i*self.batch_size: min((i+1)*self.batch_size, len(self.trajectories))]
            batch_data = get_batch_data_from_shuttleset(cur_trajectories, self.device)
            player_id = batch_data["player"].squeeze().to(dtype=torch.long) # [batch_size, max_len]
            shot_type = batch_data["shot_type"].squeeze().to(dtype=torch.long) # [batch_size, max_len]
            hit_area = batch_data["hit_area"].squeeze().to(dtype=torch.long)-1 # [batch_size, max_len]
            hit_xy = batch_data["hit_xy"].to(dtype=torch.float32)
            player_area = batch_data["player_location_area"].squeeze().to(dtype=torch.long)-1 # [batch_size, max_len]
            player_xy = batch_data["player_location_xy"].to(dtype=torch.float32)
            opponent_area = batch_data["opponent_location_area"].squeeze().to(dtype=torch.long)-1 # [batch_size, max_len]
            opponent_xy = batch_data["opponent_location_xy"].to(dtype=torch.float32) 
            landing_area = batch_data["landing_area"].squeeze().to(dtype=torch.long)-1 # [batch_size, max_len]
            landing_xy = batch_data["landing_xy"].to(dtype=torch.float32) 
            timesteps = batch_data["timesteps"].squeeze().to(dtype=torch.long) # [batch_size, max_len]
            time = batch_data["frame_num"].to(dtype=torch.float32)
            posture_fea = torch.cat((batch_data["around_head"], batch_data["back_hand"]), dim=-1).to(dtype=torch.float32)
            landing_fea = batch_data["landing_distance_opponent"].to(dtype=torch.float32)
            rally_info =  torch.cat((batch_data["score_diff"], batch_data["cons_score"]), dim=-1).to(dtype=torch.float32)
            mask = batch_data["mask"].to(dtype=torch.float32) # [batch_size, max_len] 
            reward = batch_data["reward"].to(dtype=torch.float32)
            reward[:, :-1, :][reward[:, :-1, :] == 1] = 0.0
            self.lose_reason_list += batch_data["lose_reason"]

            hit_area[hit_area < 0] = 0
            player_area[player_area < 0] = 0
            opponent_area[opponent_area < 0] = 0
            landing_area[landing_area < 0] = 0 
            
            self.max_seq_len = max(self.max_seq_len, reward.shape[1])
            self.reward_list.append(reward)
            self.mask_list.append(mask)
            
            model.eval()
            with torch.no_grad():
                # [batch_size, max_len, 1]
                win_probs = model.forward(
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
            self.win_probs_list.append(win_probs)


    def get_eval_result(self):
        # pading sequence
        for i in range(len(self.reward_list)):
            cur_batch_size = self.reward_list[i].shape[0]
            cur_seq_len = self.reward_list[i].shape[1]
            self.reward_list[i] = torch.cat([torch.zeros(cur_batch_size, self.max_seq_len-cur_seq_len, 1).to(self.device), self.reward_list[i]], dim=1)
            self.mask_list[i] = torch.cat([torch.zeros(cur_batch_size, self.max_seq_len-cur_seq_len).to(self.device), self.mask_list[i]], dim=1)
            self.win_probs_list[i] = torch.cat([torch.zeros(cur_batch_size, self.max_seq_len-cur_seq_len, 1).to(self.device), self.win_probs_list[i]], dim=1)

        reward = torch.cat(self.reward_list, dim=0) # [batch, max_len, 1]
        win_probs = torch.cat(self.win_probs_list, dim=0) # [batch, max_len, 1]
        mask = torch.cat(self.mask_list, dim=0) # [batch, max_len]
        mask = mask.unsqueeze(dim=-1) # [batch, max_len, 1]
        mask[:, -1, :][reward[:, -1, :] == 0] = 0.0
        
        # brier score
        bs = torch.square(win_probs - reward)
        bs = bs[mask > 0].mean().cpu().item()

        # AUC
        fpr, tpr, thresholds = roc_curve(reward[mask > 0].to(dtype=torch.long).cpu().numpy(), win_probs[mask > 0].cpu().numpy())
        precision, recall, thresholds = precision_recall_curve(reward[mask > 0].to(dtype=torch.long).cpu().numpy(), win_probs[mask > 0].cpu().numpy())
        f1score = f1_score(reward[mask > 0].to(dtype=torch.long).cpu().numpy(), (win_probs[mask > 0] > 0.5).cpu().numpy())
        roc_auc = auc(fpr, tpr)
        pr_auc = auc(recall, precision)

        recall = recall_score(reward[mask > 0].to(dtype=torch.long).cpu().numpy(), (win_probs[mask > 0] > 0.5).cpu().numpy())
        precision = precision_score(reward[mask > 0].to(dtype=torch.long).cpu().numpy(), (win_probs[mask > 0] > 0.5).cpu().numpy())
        ap = average_precision_score(reward[mask > 0].to(dtype=torch.long).cpu().numpy(), win_probs[mask > 0].cpu().numpy())

        true_min_proba = win_probs[mask > 0][reward[mask > 0] == 1].min()
        false_max_proba = win_probs[mask > 0][reward[mask > 0] == 0].max()
        # recall=100%, precision and fp
        precall_precision = reward[mask > 0].sum() / (win_probs[mask > 0] > true_min_proba).sum()
        precall_fp = ((win_probs[mask > 0] > true_min_proba).sum() - reward[mask > 0].sum()) / (len(reward[mask > 0]) - reward[mask > 0].sum())
        # precision=100%, recall
        pprecision_recall = (win_probs[mask > 0] > false_max_proba).sum() / reward[mask > 0].sum()
        
        # win_prediction = (win_probs[:, -1, :][reward[:, -1, :] == 1] > 0.5)
        # win_pred_right_rate = win_prediction.sum() / (reward[:, -1, :] == 1).sum()

        no_win_prediction = (win_probs[:, :-1, :][mask[:, :-1, :] > 0] < 0.5)
        no_win_right_rate = no_win_prediction.sum() / (mask[:, :-1, :] > 0).sum()
        
        return bs, roc_auc, pr_auc, f1score, ap, recall, precision, no_win_right_rate.cpu().item(), precall_precision.cpu().item(), precall_fp.cpu().item(), pprecision_recall.cpu().item()
    

class MLPPredictionEvaluator():
    # evaluator for MLP-based win-no_win prediction
    def __init__(self, dataset_path, device) -> None:
        self.dataset_path = dataset_path
        self.device = device
        with open(self.dataset_path, 'rb') as f:
            self.trajectories = pickle.load(f)
        
        self.buffer = ReplayBuffer(device=self.device)
        data = convert_rally_to_whole(self.trajectories)
        self.buffer.load_dataset(data)

    def action_generation(self, model):
        
        batch_data = self.buffer.sample_all()
        player_id = batch_data["player"].to(dtype=torch.long) # [batch_size]
        shot_type = batch_data["shot_type"].to(dtype=torch.long) # [batch_size]
        hit_area = batch_data["hit_area"].to(dtype=torch.long)-1 # [batch_size]
        hit_xy = batch_data["hit_xy"].to(dtype=torch.float32)
        player_area = batch_data["player_location_area"].to(dtype=torch.long)-1 # [batch_size]
        player_xy = batch_data["player_location_xy"].to(dtype=torch.float32)
        opponent_area = batch_data["opponent_location_area"].to(dtype=torch.long)-1 # [batch_size]
        opponent_xy = batch_data["opponent_location_xy"].to(dtype=torch.float32) 
        landing_area = batch_data["landing_area"].to(dtype=torch.long)-1 # [batch_size]
        landing_xy = batch_data["landing_xy"].to(dtype=torch.float32) 
        timesteps = batch_data["ball_round"].to(dtype=torch.long) # [batch_size]
        time = batch_data["frame_num"].to(dtype=torch.float32)
        posture_fea = torch.cat((batch_data["around_head"], batch_data["back_hand"]), dim=-1).to(dtype=torch.float32)
        # landing_fea = torch.cat((batch_data["bad_landing_flag"], batch_data["landing_distance_opponent"]), dim=-1).to(dtype=torch.float32)
        landing_fea = batch_data["landing_distance_opponent"].unsqueeze(dim=-1).to(dtype=torch.float32) # just use landing distance from opponent
        rally_info =  torch.cat((batch_data["score_diff"], batch_data["cons_score"]), dim=-1).to(dtype=torch.float32)
        mask = None
        self.reward = batch_data["reward"].to(dtype=torch.float32) #[batch_size]

        model.eval()
        with torch.no_grad():
            # [batch_size, 1]
            win_probs = model.forward(
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
        self.win_probs = win_probs.squeeze()

    def get_eval_result(self):
        # brier score
        bs = torch.square(self.win_probs - self.reward)
        bs = bs.mean().cpu().item()

        # AUC
        fpr, tpr, thresholds = roc_curve(self.reward.to(dtype=torch.long).cpu().numpy(), self.win_probs.cpu().numpy())
        precision, recall, thresholds = precision_recall_curve(self.reward.to(dtype=torch.long).cpu().numpy(), self.win_probs.cpu().numpy())
        f1score = f1_score(self.reward.to(dtype=torch.long).cpu().numpy(), (self.win_probs > 0.5).cpu().numpy())
        roc_auc = auc(fpr, tpr)
        pr_auc = auc(recall, precision)

        recall = recall_score(self.reward.to(dtype=torch.long).cpu().numpy(), (self.win_probs > 0.5).cpu().numpy())
        precision = precision_score(self.reward.to(dtype=torch.long).cpu().numpy(), (self.win_probs > 0.5).cpu().numpy())
        ap = average_precision_score(self.reward.to(dtype=torch.long).cpu().numpy(), self.win_probs.cpu().numpy())

        win_prediction = (self.win_probs[self.reward == 1] > 0.5)
        win_pred_right_rate = win_prediction.sum() / (self.reward == 1).sum()

        no_win_prediction = (self.win_probs[self.reward == 0] < 0.5)
        no_win_right_rate = no_win_prediction.sum() / (self.reward == 0).sum()
        
        # return bs, roc_auc, pr_auc, f1score, win_pred_right_rate.cpu().item(), no_win_right_rate.cpu().item()
        return bs, roc_auc, pr_auc, f1score, ap, recall, precision
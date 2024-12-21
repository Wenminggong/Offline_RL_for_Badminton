# -*- coding: utf-8 -*-
'''
@File    :   reward_model.py
@Time    :   2024/08/12 16:03:49
@Author  :   Mingjiang Liu 
@Version :   1.0
@Desc    :   preference-based reward learning model, specified for evaluating generated tactics
'''


import os

import torch
import torch.nn as nn
import numpy as np


class RewardNet(nn.Module):
    # (s,a) -> r(s,a)
    def __init__(self, 
                shot_type_num:int=10,
                shot_type_dim:int=15,
                location_type:bool=False,
                location_num:int=16,
                location_dim:int=10,
                other_fea_dim:int=2,
                n_layer:int=3,
                hidden_dim:int=256,
                activation:str="tanh",
            ) -> None:
        super().__init__()

        self.location_type = location_type


        self.shot_type_embedding = nn.Embedding(shot_type_num, shot_type_dim)
        if self.location_type:
            # if discrete location
            self.location_embedding = nn.Embedding(location_num, location_dim)
        else:
            location_dim = 2
        input_dim = shot_type_dim * 2 + location_dim * 5 + other_fea_dim

        net = []
        for i in range(n_layer):
            net.append(nn.Linear(input_dim, hidden_dim))
            net.append(nn.LeakyReLU())
            input_dim = hidden_dim
        net.append(nn.Linear(input_dim, 1))
        if activation == "tanh":
            net.append(nn.Tanh())
        elif activation == "sig":
            net.append(nn.Sigmoid())
        elif activation == "relu":
            net.append(nn.ReLU())
        else:
            raise NotImplementedError
        
        self.reward_net = nn.Sequential(*net)

    def forward(self,
                last_shot_type:torch.Tensor, 
                hit_area:torch.Tensor,
                hit_xy:torch.Tensor, 
                player_area:torch.Tensor, 
                player_xy:torch.Tensor,
                opponent_area:torch.Tensor,
                opponent_xy:torch.Tensor,
                shot_type:torch.Tensor,
                landing_area:torch.Tensor,
                landing_xy:torch.Tensor,
                move_area: torch.Tensor,
                move_xy:torch.Tensor,
                landing_fea:torch.Tensor, 
            ):

        last_shot_type_embed = self.shot_type_embedding(last_shot_type)
        shot_type_embed = self.shot_type_embedding(shot_type)
        if self.location_type:
            # if location is discrete
            hit_location_embed = self.location_embedding(hit_area)
            player_location_embed = self.location_embedding(player_area)
            oppo_location_embed = self.location_embedding(opponent_area)
            landing_location_embed = self.location_embedding(landing_area)
            move_location_embed = self.location_embedding(move_area)
        else:
            # if location is continuous
            hit_location_embed = hit_xy
            player_location_embed = player_xy
            oppo_location_embed = opponent_xy
            landing_location_embed = landing_xy
            move_location_embed = move_xy

        input_embed = torch.cat([last_shot_type_embed, hit_location_embed, player_location_embed, oppo_location_embed, shot_type_embed, landing_location_embed, move_location_embed, landing_fea], dim=-1)
        reward = self.reward_net(input_embed) # [m, 1]
        return reward

class RewardModel():
    def __init__(self,
                pref_model:int=0,
                no_end_rally_pref:int=1,
                no_end_rally_pref_mode:int=0,
                no_end_rally_pref_factor:float=1.0,
                loss_type:int=1,
                ensemble_size:int=1,
                action_pref_type:int=1,
                lr:float=1e-4,
                batch_size:int=128,
                action_pref_factor:float=0.5,
                evaluate_flag:int=1,
                shot_type_num:int=10,
                shot_type_dim:int=15,
                location_type:bool=False,
                location_num:int=16,
                location_dim:int=10,
                other_fea_dim:int=1,
                n_layer:int=3,
                hidden_dim:int=256,
                activation:str="tanh",
                device:str="cuda",
            ) -> None:
        self.pref_model = pref_model
        self.no_end_rally_pref = no_end_rally_pref
        self.no_end_rally_pref_mode = no_end_rally_pref_mode
        self.no_end_rally_pref_factor = no_end_rally_pref_factor
        self.loss_type = loss_type
        self.action_pref_type = action_pref_type
        self.ensemble_size = ensemble_size
        self.lr = lr
        self.batch_size = batch_size
        self.action_pref_factor = action_pref_factor
        self.evaluate_flag = evaluate_flag

        self.shot_type_num = shot_type_num
        self.shot_type_dim = shot_type_dim
        self.location_type = location_type
        self.location_num = location_num
        self.location_dim = location_dim
        self.other_fea_dim = other_fea_dim
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.device = device

        self.ensemble = []
        self.paramlst = []

        self._construct_ensemble()

        self.train_buffer_rally1 = None
        self.train_buffer_rally2 = None
        self.train_buffer_label = None
        self.train_buffer_end_flag = None

        self.val_buffer_rally1 = None
        self.val_buffer_rally2 = None
        self.val_buffer_label = None
        self.val_buffer_end_flag = None

        self.ce_loss = nn.CrossEntropyLoss()

    def _construct_ensemble(self):
        # construct ensemble reward models
        for i in range(self.ensemble_size):
            model = RewardNet(
                self.shot_type_num,
                self.shot_type_dim,
                self.location_type,
                self.location_num,
                self.location_dim,
                self.other_fea_dim,
                self.n_layer,
                self.hidden_dim,
                self.activation,
            ).to(self.device)
            self.ensemble.append(model)
            self.paramlst.extend(model.parameters())
        
        self.opt = torch.optim.Adam(self.paramlst, lr=self.lr)

    def _construct_preference_buffer(self, data_list):
        buffer_rally_1 = {}
        buffer_rally_2 = {}
        buffer_label = []
        buffer_end_flag = []
        max_len_1 = 0
        max_len_2 = 0
        for rally in data_list:
            action_num = len(rally["reward"])
            if action_num < 3:
                continue
            
            for key in rally.keys():
                if isinstance(rally[key], np.ndarray):
                    if key in buffer_rally_1.keys():
                        if len(rally[key].shape) < 2:
                            buffer_rally_1[key].append(rally[key][::2][1:].reshape(1, -1, 1))
                            buffer_rally_2[key].append(rally[key][1::2].reshape(1, -1, 1))
                        else:
                            buffer_rally_1[key].append(rally[key][::2][1:].reshape(1, -1, 2))
                            buffer_rally_2[key].append(rally[key][1::2].reshape(1, -1, 2))
                    else:
                        if len(rally[key].shape) < 2:
                            buffer_rally_1[key] = [rally[key][::2][1:].reshape(1, -1, 1)]
                            buffer_rally_2[key] = [rally[key][1::2].reshape(1, -1, 1)]
                        else:
                            buffer_rally_1[key] = [rally[key][::2][1:].reshape(1, -1, 2)]
                            buffer_rally_2[key] = [rally[key][1::2].reshape(1, -1, 2)]

            if (action_num % 2 == 1 and rally["reward"][::2][-1] == 1) or (action_num % 2 == 0 and rally["reward"][1::2][-1] == 0):
                # buffer_rally_1 win
                buffer_label.append(1)
            else:
                # buffer_rally_2 win
                buffer_label.append(0)

            if max_len_1 < buffer_rally_1["reward"][-1].shape[1]:
                max_len_1 = buffer_rally_1["reward"][-1].shape[1]
            if max_len_2 < buffer_rally_2["reward"][-1].shape[1]:
                max_len_2 = buffer_rally_2["reward"][-1].shape[1]

            # modify last reward as 1 or -1
            if buffer_rally_1["reward"][-1][0, -1, 0] == 0:
                buffer_rally_1["reward"][-1][0, -1, 0] = -1
            if buffer_rally_2["reward"][-1][0, -1, 0] == 0:
                buffer_rally_2["reward"][-1][0, -1, 0] = -1

            # padding last_move_xy = 0.5
            if action_num % 2 == 1:
                buffer_rally_1["move_xy"][-1][:, -1] = 0.5
                buffer_end_flag.append(1)
            else:
                buffer_rally_2["move_xy"][-1][:, -1] = 0.5
                buffer_end_flag.append(2)

        mask_1, mask_2 = [], []
        for i in range(len(buffer_label)):
            # padding
            tlen1 = buffer_rally_1["reward"][i].shape[1]
            tlen2 = buffer_rally_2["reward"][i].shape[1]
            for key in buffer_rally_1.keys():
                buffer_rally_1[key][i] = np.concatenate([np.zeros((1, max_len_1-tlen1, buffer_rally_1[key][i].shape[-1])), buffer_rally_1[key][i]], axis=1)
                buffer_rally_2[key][i] = np.concatenate([np.zeros((1, max_len_2-tlen2, buffer_rally_2[key][i].shape[-1])), buffer_rally_2[key][i]], axis=1)

            cur_mask_1 = np.ones((1, tlen1)) 
            cur_mask_2 = np.ones((1, tlen2)) 
            mask_1.append(np.concatenate([np.zeros((1, max_len_1-tlen1)), cur_mask_1], axis=1))
            mask_2.append(np.concatenate([np.zeros((1, max_len_2-tlen2)), cur_mask_2], axis=1))

        buffer_rally_1["mask"] = mask_1
        buffer_rally_2["mask"] = mask_2

        # buffer_data: Dict={key:List} to Dict={key:tensor}
        for key in buffer_rally_1.keys():
            buffer_rally_1[key] = torch.from_numpy(np.concatenate(buffer_rally_1[key], axis=0)).to(device=self.device) # [batch_size, max_len_1, fea_dim]
            buffer_rally_2[key] = torch.from_numpy(np.concatenate(buffer_rally_2[key], axis=0)).to(device=self.device) # [batch_size, max_len_2, fea_dim]

        buffer_label = torch.tensor(buffer_label, device=self.device, dtype=torch.long)
        
        # tensor-device, buffer_end_flag = List -> ndarray
        return buffer_rally_1, buffer_rally_2, buffer_label, np.array(buffer_end_flag)


    def construct_train_buffer(self, train_data_list):
        # train_data_list = [rally, ..., rally]
        # self.train_buffer_end_flag: 1d-ndarray
        self.train_buffer_rally1, self.train_buffer_rally2, self.train_buffer_label, self.train_buffer_end_flag = self._construct_preference_buffer(train_data_list)

    def construct_val_buffer(self, val_data_list):
        # val_data_list = [rally, ..., rally]
        self.val_buffer_rally1, self.val_buffer_rally2, self.val_buffer_label, self.val_buffer_end_flag = self._construct_preference_buffer(val_data_list)

    def save(self, model_dir, model_name):
        for member in range(self.ensemble_size):
            torch.save(
                self.ensemble[member].state_dict(), os.path.join(model_dir, "{}_reward_model_{}.pth".format(model_name, member))
            )

    def load(self, model_dir, model_name):
        for member in range(self.ensemble_size):
            self.ensemble[member].load_state_dict(
                torch.load(os.path.join(model_dir, "{}_reward_model_{}.pth".format(model_name, member)))
            )

    def _r_hat_member(self,
                    last_shot_type,
                    hit_area,
                    hit_xy,
                    player_area,
                    player_xy,
                    opponent_area,
                    opponent_xy,
                    shot_type,
                    landing_area,
                    landing_xy,
                    move_area,
                    move_xy,
                    landing_fea,
                    member,
                ):
        # compute reward for each reward model
        return self.ensemble[member](
            last_shot_type,
            hit_area,
            hit_xy,
            player_area,
            player_xy,
            opponent_area,
            opponent_xy,
            shot_type,
            landing_area,
            landing_xy,
            move_area,
            move_xy,
            landing_fea
        )

    def r_hat(self,
            last_shot_type,
            hit_area,
            hit_xy,
            player_area,
            player_xy,
            opponent_area,
            opponent_xy,
            shot_type,
            landing_area,
            landing_xy,
            move_area,
            move_xy,
            landing_fea,
            ):
        # compute average reward from all reward models
        r_hats = []
        with torch.no_grad():
            for member in range(self.ensemble_size):
                self.ensemble[member].eval()
                r_hats.append(self.ensemble[member](
                    last_shot_type,
                    hit_area,
                    hit_xy,
                    player_area,
                    player_xy,
                    opponent_area,
                    opponent_xy,
                    shot_type,
                    landing_area,
                    landing_xy,
                    move_area,
                    move_xy,
                    landing_fea,
                    ).unsqueeze(dim=0)
                )
            r_hats = torch.cat(r_hats, dim=0)
        # return tensor
        return r_hats.mean(dim=0)
    
    def _compute_action_loss(self, r_hat_1, r_hat_2, reward_1, reward_2, mask_1, mask_2, end_flag, action_pref_type):
        # construct action preferences and compute action-preference loss
        if action_pref_type == 1:
            # 1: player end action with player other actions
            new_r_hat_1_1 = r_hat_1[:, :-1, :]
            new_r_hat_1_2 = r_hat_1[:, -1:, :].expand(-1, new_r_hat_1_1.shape[1], -1)
            new_labels_1 = torch.zeros(new_r_hat_1_1.shape, dtype=torch.long, device=self.device)
            new_reward_1 = reward_1[:, -1:, :].expand(-1, new_r_hat_1_1.shape[1], -1)
            new_labels_1[new_reward_1 > 0] = 1
            new_mask_1 = mask_1[:, :-1].flatten()

            new_r_hat_2_1 = r_hat_2[:, :-1, :]
            new_r_hat_2_2 = r_hat_2[:, -1:, :].expand(-1, new_r_hat_2_1.shape[1], -1)
            new_labels_2 = torch.zeros(new_r_hat_2_1.shape, dtype=torch.long, device=self.device)
            new_reward_2 = reward_2[:, -1:, :].expand(-1, new_r_hat_2_1.shape[1], -1)
            new_labels_2[new_reward_2 > 0] = 1
            new_mask_2 = mask_2[:, :-1].flatten()

            new_r_hat_1 = torch.cat([new_r_hat_1_1.reshape((-1, 1)), new_r_hat_2_1.reshape((-1, 1))], dim=0)
            new_r_hat_2 = torch.cat([new_r_hat_1_2.reshape((-1, 1)), new_r_hat_2_2.reshape((-1, 1))], dim=0)
            new_r_hat = torch.cat([new_r_hat_1, new_r_hat_2], dim=-1)
            new_labels = torch.cat([new_labels_1.flatten(), new_labels_2.flatten()], dim=0)
            new_masks = torch.cat([new_mask_1, new_mask_2], dim=0)

        elif action_pref_type == 0:
            # 0: rally end action with other actions
            new_r_hat_1, new_r_hat_2 = [], []
            new_labels, new_masks = [], []
            for i in range(r_hat_1.shape[0]):
                if end_flag[i] == 1:
                    # end in player 1
                    cur_r_hat_1 = torch.cat([r_hat_1[i, :-1], r_hat_2[i]], dim=0) # [s1-1 + s2, 1]
                    cur_r_hat_2 = r_hat_1[i, -1:].expand(cur_r_hat_1.shape[0], -1)
                    cur_labels = torch.ones(cur_r_hat_1.shape, dtype=torch.long, device=self.device) if r_hat_1[i, -1, 0] > 0 else torch.zeros(cur_r_hat_1.shape, dtype=torch.long, device=self.device)
                    cur_masks = torch.cat([mask_1[i, :-1], mask_2[i]], dim=0)
                    new_r_hat_1.append(cur_r_hat_1)
                    new_r_hat_2.append(cur_r_hat_2)
                    new_labels.append(cur_labels)
                    new_masks.append(cur_masks)
                elif end_flag[i] == 2:
                    # end in player 2
                    cur_r_hat_1 = torch.cat([r_hat_2[i, :-1], r_hat_1[i]], dim=0) # [s2-1 + s1, 1]
                    cur_r_hat_2 = r_hat_2[i, -1:].expand(cur_r_hat_1.shape[0], -1)
                    cur_labels = torch.ones(cur_r_hat_1.shape, dtype=torch.long, device=self.device) if r_hat_2[i, -1, 0] > 0 else torch.zeros(cur_r_hat_1.shape, dtype=torch.long, device=self.device)
                    cur_masks = torch.cat([mask_2[i, :-1], mask_1[i]], dim=0)
                    new_r_hat_1.append(cur_r_hat_1)
                    new_r_hat_2.append(cur_r_hat_2)
                    new_labels.append(cur_labels)
                    new_masks.append(cur_masks)
                else:
                    raise NotImplementedError
                
            new_r_hat_1 = torch.cat(new_r_hat_1, dim=0)
            new_r_hat_2 = torch.cat(new_r_hat_2, dim=0)
            new_r_hat = torch.cat([new_r_hat_1, new_r_hat_2], dim=-1)
            new_labels = torch.cat(new_labels, dim=0).flatten()
            new_masks = torch.cat(new_masks, dim=0).flatten()

        else:
            raise NotImplementedError

        # compute action preferences losses
        action_loss = self.ce_loss(new_r_hat[new_masks > 0], new_labels[new_masks > 0])
        # compute acc
        _, predicted = torch.max(new_r_hat[new_masks > 0].data, 1)
        correct_num = (predicted == new_labels[new_masks > 0]).sum().item()
        action_pair_num = len(new_labels[new_masks > 0])
        return action_loss, action_pair_num, correct_num

    def _compute_no_end_rally_loss(self, r_hat_1, r_hat_2, labels, mask_1, mask_2, end_flag):
        new_mask_1 = mask_1.clone()
        new_mask_2 = mask_2.clone()
        for i in range(len(end_flag)):
            if end_flag[i] == 1:
                new_mask_1[i, -1] = 0
            elif end_flag[i] == 2:
                new_mask_2[i, -1] = 0
            else:
                raise NotImplementedError
            
        new_r_hat_1 = r_hat_1 * new_mask_1.unsqueeze(dim=-1) # ignore padding items
        new_r_hat_1 = new_r_hat_1.sum(dim=1)
        new_r_hat_2 = r_hat_2 * new_mask_2.unsqueeze(dim=-1)
        new_r_hat_2 = new_r_hat_2.sum(dim=1)
        
        if self.no_end_rally_pref_mode:
            # using average pref model, else using standard pref model
            new_r_hat_1 = new_r_hat_1 / (new_mask_1.sum(dim=-1, keepdim=True) + 1e-9)
            new_r_hat_2 = new_r_hat_2 / (new_mask_2.sum(dim=-1, keepdim=True) + 1e-9)

        r_hat = torch.cat([new_r_hat_2, new_r_hat_1], dim=-1) # [batch_size, 2]

        # compute no-end rally loss
        cur_no_end_rally_loss = self.ce_loss(r_hat, labels)

        # compute no-end rally acc
        _, rally_predicted = torch.max(r_hat.data, 1)
        rally_no_end_correct_num = (rally_predicted == labels).sum().item()
        return cur_no_end_rally_loss, rally_no_end_correct_num

    def train_reward(self, cur_epoch, print_logs):
        ensemble_losses = [[] for _ in range(self.ensemble_size)]
        ensemble_rally_losses = [[] for _ in range(self.ensemble_size)]
        ensemble_rally_acc = np.array([0.0 for _ in range(self.ensemble_size)])
        ensemble_no_end_rally_losses = [[] for _ in range(self.ensemble_size)]
        ensemble_no_end_rally_acc = np.array([0.0 for _ in range(self.ensemble_size)])
        ensemble_action_losses = [[] for _ in range(self.ensemble_size)]
        ensemble_action_acc = np.array([0.0 for _ in range(self.ensemble_size)])
        ensemble_action_pair_num = np.array([0.0 for _ in range(self.ensemble_size)])

        train_rally_num = len(self.train_buffer_label)
        
        train_rally_index = []
        for _ in range(self.ensemble_size):
            train_rally_index.append(np.random.permutation(train_rally_num))

        iter_num = int(np.ceil(train_rally_num / self.batch_size))

        for iter in range(iter_num):
            self.opt.zero_grad()
            rally_pref_loss = 0.0
            no_end_rally_pref_loss = 0.0
            action_pref_loss = 0.0

            last_index = min((iter+1) * self.batch_size, train_rally_num)

            for member in range(self.ensemble_size):
                self.ensemble[member].train()
                # get train-data
                idxs = train_rally_index[member][iter*self.batch_size:last_index]
            
                last_shot_type_1 = self.train_buffer_rally1["last_time_opponent_type"][idxs].squeeze().to(dtype=torch.long) # tensor [batch_size, max_seq]
                hit_area_1 = self.train_buffer_rally1["hit_area"][idxs].squeeze().to(dtype=torch.long) - 1
                hit_xy_1 = self.train_buffer_rally1["hit_xy"][idxs].to(dtype=torch.float32)
                player_area_1 = self.train_buffer_rally1["player_location_area"][idxs].squeeze().to(dtype=torch.long) - 1
                player_xy_1 = self.train_buffer_rally1["player_location_xy"][idxs].to(dtype=torch.float32)
                opponent_area_1 = self.train_buffer_rally1["opponent_location_area"][idxs].squeeze().to(dtype=torch.long) - 1
                opponent_xy_1 = self.train_buffer_rally1["opponent_location_xy"][idxs].to(dtype=torch.float32)
                shot_type_1 = self.train_buffer_rally1["shot_type"][idxs].squeeze().to(dtype=torch.long)
                landing_area_1 = self.train_buffer_rally1["landing_area"][idxs].squeeze().to(dtype=torch.long) - 1
                landing_xy_1 = self.train_buffer_rally1["landing_xy"][idxs].to(dtype=torch.float32)
                move_area_1 = self.train_buffer_rally1["move_area"][idxs].squeeze().to(dtype=torch.long) - 1
                move_xy_1 = self.train_buffer_rally1["move_xy"][idxs].to(dtype=torch.float32)
                landing_fea_1 = torch.cat([self.train_buffer_rally1["bad_landing_flag"][idxs], self.train_buffer_rally1["landing_distance_opponent"][idxs]], dim=-1).to(dtype=torch.float32)
                reward_1 = self.train_buffer_rally1["reward"][idxs]
                mask_1 = self.train_buffer_rally1["mask"][idxs]

                last_shot_type_2 = self.train_buffer_rally2["last_time_opponent_type"][idxs].squeeze().to(dtype=torch.long) # tensor [batch_size, max_seq, m]
                hit_area_2 = self.train_buffer_rally2["hit_area"][idxs].squeeze().to(dtype=torch.long) - 1
                hit_xy_2 = self.train_buffer_rally2["hit_xy"][idxs].to(dtype=torch.float32)
                player_area_2 = self.train_buffer_rally2["player_location_area"][idxs].squeeze().to(dtype=torch.long) - 1
                player_xy_2 = self.train_buffer_rally2["player_location_xy"][idxs].to(dtype=torch.float32)
                opponent_area_2 = self.train_buffer_rally2["opponent_location_area"][idxs].squeeze().to(dtype=torch.long) - 1
                opponent_xy_2 = self.train_buffer_rally2["opponent_location_xy"][idxs].to(dtype=torch.float32)
                shot_type_2 = self.train_buffer_rally2["shot_type"][idxs].squeeze().to(dtype=torch.long)
                landing_area_2 = self.train_buffer_rally2["landing_area"][idxs].squeeze().to(dtype=torch.long) - 1
                landing_xy_2 = self.train_buffer_rally2["landing_xy"][idxs].to(dtype=torch.float32)
                move_area_2 = self.train_buffer_rally2["move_area"][idxs].squeeze().to(dtype=torch.long) - 1
                move_xy_2 = self.train_buffer_rally2["move_xy"][idxs].to(dtype=torch.float32)
                landing_fea_2 = torch.cat([self.train_buffer_rally2["bad_landing_flag"][idxs], self.train_buffer_rally2["landing_distance_opponent"][idxs]], dim=-1).to(dtype=torch.float32)
                reward_2 = self.train_buffer_rally2["reward"][idxs]
                mask_2 = self.train_buffer_rally2["mask"][idxs]

                labels = self.train_buffer_label[idxs]
                train_buffer_end_flag = self.train_buffer_end_flag[idxs] # ndarray

                # get reward, [batch_size, max_len, 1]
                r_hat_1 = self._r_hat_member(
                    last_shot_type_1,
                    hit_area_1,
                    hit_xy_1,
                    player_area_1,
                    player_xy_1,
                    opponent_area_1,
                    opponent_xy_1,
                    shot_type_1,
                    landing_area_1,
                    landing_xy_1,
                    move_area_1,
                    move_xy_1,
                    landing_fea_1,
                    member,
                )

                r_hat_2 = self._r_hat_member(
                    last_shot_type_2,
                    hit_area_2,
                    hit_xy_2,
                    player_area_2,
                    player_xy_2,
                    opponent_area_2,
                    opponent_xy_2,
                    shot_type_2,
                    landing_area_2,
                    landing_xy_2,
                    move_area_2,
                    move_xy_2,
                    landing_fea_2,
                    member,
                )

                # r_hat_1 and r_hat_2: [batch_size, max_len, 1]-tensor

                if self.loss_type:
                    # if using action pref
                    cur_action_loss, cur_action_pair_num, cur_action_correct_num = self._compute_action_loss(r_hat_1, r_hat_2, reward_1, reward_2, mask_1, mask_2, train_buffer_end_flag, self.action_pref_type)
                    action_pref_loss = action_pref_loss + cur_action_loss
                    ensemble_action_losses[member].append(cur_action_loss.item())
                    ensemble_action_acc[member] += (cur_action_correct_num)
                    ensemble_action_pair_num[member] += cur_action_pair_num

                if self.no_end_rally_pref:
                    # using no-end rally pref
                    cur_no_end_rally_loss, cur_rally_no_end_correct_num = self._compute_no_end_rally_loss(r_hat_1, r_hat_2, labels, mask_1, mask_2, train_buffer_end_flag)
                    ensemble_no_end_rally_losses[member].append(cur_no_end_rally_loss.item())
                    ensemble_no_end_rally_acc[member] += cur_rally_no_end_correct_num
                    no_end_rally_pref_loss = no_end_rally_pref_loss + cur_no_end_rally_loss

                r_hat_1 = r_hat_1 * mask_1.unsqueeze(dim=-1) # ignore padding items
                r_hat_1 = r_hat_1.sum(dim=1)
                r_hat_2 = r_hat_2 * mask_2.unsqueeze(dim=-1)
                r_hat_2 = r_hat_2.sum(dim=1)
                
                if self.pref_model:
                    # using average pref model, else using standard pref model
                    r_hat_1 = r_hat_1 / mask_1.sum(dim=-1, keepdim=True)
                    r_hat_2 = r_hat_2 / mask_2.sum(dim=-1, keepdim=True)

                r_hat = torch.cat([r_hat_2, r_hat_1], dim=-1) # [batch_size, 2]

                # compute rally loss
                cur_rally_loss = self.ce_loss(r_hat, labels)
                rally_pref_loss = rally_pref_loss + cur_rally_loss
                ensemble_rally_losses[member].append(cur_rally_loss.item())

                # compute rally acc
                _, rally_predicted = torch.max(r_hat.data, 1)
                rally_correct_num = (rally_predicted == labels).sum().item()
                ensemble_rally_acc[member] += (rally_correct_num)

            if self.loss_type:
                # use both rally_loss and action_loss
                loss = rally_pref_loss + self.action_pref_factor * action_pref_loss
            else:
                # only use rally_loss
                if self.no_end_rally_pref:
                    # using no-end rally pref
                    loss = rally_pref_loss + self.no_end_rally_pref_factor * no_end_rally_pref_loss
                else:
                    loss = rally_pref_loss
            loss.backward()
            self.opt.step()

        # ensemble_rally_losses = np.mean(ensemble_rally_losses)
        ensemble_rally_acc /= train_rally_num
        ensemble_no_end_rally_acc /= train_rally_num
        # ensemble_action_losses = np.mean(ensemble_action_losses)
        ensemble_action_acc /= ensemble_action_pair_num

        log_info = {}
        for member in range(self.ensemble_size):
            # log_info["train/total_loss_m{}".format(member)] = np.mean(ensemble_losses[member])
            log_info["train/rally_loss_m{}".format(member)] = np.mean(ensemble_rally_losses[member])
            log_info["train/rally_acc_m{}".format(member)] = ensemble_rally_acc[member]
            if self.no_end_rally_pref:
                log_info["train/rally_no_end_loss_m{}".format(member)] = np.mean(ensemble_no_end_rally_losses[member])
                log_info["train/rally_no_end_acc_m{}".format(member)] = ensemble_no_end_rally_acc[member]
            if self.loss_type:
                log_info["train/action_loss_m{}".format(member)] = np.mean(ensemble_action_losses[member])
                log_info["train/action_acc_m{}".format(member)] = ensemble_action_acc[member]

        # evalation
        if self.evaluate_flag:
            eval_rally_acc, eval_no_end_rally_acc, eval_action_acc_0, eval_action_acc_1 = self.evaluate()
            eval_log = {
                "eval/rally_acc": eval_rally_acc,
                "eval/no_end_rally_acc": eval_no_end_rally_acc,
                "eval/action_acc_end": eval_action_acc_0,
                "eval/action_acc_player": eval_action_acc_1
            }
            log_info.update(eval_log)
        
        if print_logs:
            print('=' * 80)
            print(f'Iteration: {cur_epoch}')
            for k, v in log_info.items():
                print(f'{k}: {v}')

        return log_info

    def evaluate(self):
        # evaluate model
        val_rally_num = len(self.val_buffer_label)
        
        last_shot_type_1 = self.val_buffer_rally1["last_time_opponent_type"].squeeze().to(dtype=torch.long) # tensor [batch_size, max_seq, m]
        hit_area_1 = self.val_buffer_rally1["hit_area"].squeeze().to(dtype=torch.long)
        hit_xy_1 = self.val_buffer_rally1["hit_xy"].to(dtype=torch.float32)
        player_area_1 = self.val_buffer_rally1["player_location_area"].squeeze().to(dtype=torch.long)
        player_xy_1 = self.val_buffer_rally1["player_location_xy"].to(dtype=torch.float32)
        opponent_area_1 = self.val_buffer_rally1["opponent_location_area"].squeeze().to(dtype=torch.long)
        opponent_xy_1 = self.val_buffer_rally1["opponent_location_xy"].to(dtype=torch.float32)
        shot_type_1 = self.val_buffer_rally1["shot_type"].squeeze().to(dtype=torch.long)
        landing_area_1 = self.val_buffer_rally1["landing_area"].squeeze().to(dtype=torch.long)
        landing_xy_1 = self.val_buffer_rally1["landing_xy"].to(dtype=torch.float32)
        move_area_1 = self.val_buffer_rally1["move_area"].squeeze().to(dtype=torch.long)
        move_xy_1 = self.val_buffer_rally1["move_xy"].to(dtype=torch.float32)
        landing_fea_1 = torch.cat([self.val_buffer_rally1["bad_landing_flag"], self.val_buffer_rally1["landing_distance_opponent"]], dim=-1).to(dtype=torch.float32)
        reward_1 = self.val_buffer_rally1["reward"]
        mask_1 = self.val_buffer_rally1["mask"]

        last_shot_type_2 = self.val_buffer_rally2["last_time_opponent_type"].squeeze().to(dtype=torch.long) # tensor [batch_size, max_seq, m]
        hit_area_2 = self.val_buffer_rally2["hit_area"].squeeze().to(dtype=torch.long)
        hit_xy_2 = self.val_buffer_rally2["hit_xy"].to(dtype=torch.float32)
        player_area_2 = self.val_buffer_rally2["player_location_area"].squeeze().to(dtype=torch.long)
        player_xy_2 = self.val_buffer_rally2["player_location_xy"].to(dtype=torch.float32)
        opponent_area_2 = self.val_buffer_rally2["opponent_location_area"].squeeze().to(dtype=torch.long)
        opponent_xy_2 = self.val_buffer_rally2["opponent_location_xy"].to(dtype=torch.float32)
        shot_type_2 = self.val_buffer_rally2["shot_type"].squeeze().to(dtype=torch.long)
        landing_area_2 = self.val_buffer_rally2["landing_area"].squeeze().to(dtype=torch.long)
        landing_xy_2 = self.val_buffer_rally2["landing_xy"].to(dtype=torch.float32)
        move_area_2 = self.val_buffer_rally2["move_area"].squeeze().to(dtype=torch.long)
        move_xy_2 = self.val_buffer_rally2["move_xy"].to(dtype=torch.float32)
        landing_fea_2 = torch.cat([self.val_buffer_rally2["bad_landing_flag"], self.val_buffer_rally2["landing_distance_opponent"]], dim=-1).to(dtype=torch.float32)
        reward_2 = self.val_buffer_rally2["reward"]
        mask_2 = self.val_buffer_rally2["mask"]

        labels = self.val_buffer_label

        # get reward, [batch_size, max_len, 1]
        r_hat_1 = self.r_hat(
            last_shot_type_1,
            hit_area_1,
            hit_xy_1,
            player_area_1,
            player_xy_1,
            opponent_area_1,
            opponent_xy_1,
            shot_type_1,
            landing_area_1,
            landing_xy_1,
            move_area_1,
            move_xy_1,
            landing_fea_1
        )

        r_hat_2 = self.r_hat(
            last_shot_type_2,
            hit_area_2,
            hit_xy_2,
            player_area_2,
            player_xy_2,
            opponent_area_2,
            opponent_xy_2,
            shot_type_2,
            landing_area_2,
            landing_xy_2,
            move_area_2,
            move_xy_2,
            landing_fea_2
        )

        # get rally-end action preference
        _, cur_action_pair_num_0, cur_action_correct_num_0 = self._compute_action_loss(r_hat_1, r_hat_2, reward_1, reward_2, mask_1, mask_2, self.val_buffer_end_flag, 0)
        # get player-end action preference
        _, cur_action_pair_num_1, cur_action_correct_num_1 = self._compute_action_loss(r_hat_1, r_hat_2, reward_1, reward_2, mask_1, mask_2, self.val_buffer_end_flag, 1)

        # get no-end rally pref
        _, cur_rally_no_end_correct_num = self._compute_no_end_rally_loss(r_hat_1, r_hat_2, labels, mask_1, mask_2, self.val_buffer_end_flag)

        r_hat_1[mask_1 < 1] = 0.0 # ignore padding items
        r_hat_1 = r_hat_1.sum(dim=1)
        r_hat_2[mask_2 < 1] = 0.0
        r_hat_2 = r_hat_2.sum(dim=1)

        if self.pref_model:
            # using average pref model, else using standard pref model
            r_hat_1 = r_hat_1 / mask_1.sum(dim=-1, keepdim=True)
            r_hat_2 = r_hat_2 / mask_2.sum(dim=-1, keepdim=True)

        r_hat = torch.cat([r_hat_2, r_hat_1], dim=-1) # [batch_size, 2]

        # compute rally acc
        _, rally_predicted = torch.max(r_hat.data, 1)
        rally_correct_num = (rally_predicted == labels).sum().item()

        rally_acc = rally_correct_num / val_rally_num
        no_end_rally_acc = cur_rally_no_end_correct_num / val_rally_num
        action_acc_0 = cur_action_correct_num_0 / cur_action_pair_num_0
        action_acc_1 = cur_action_correct_num_1 / cur_action_pair_num_1
        return rally_acc, no_end_rally_acc, action_acc_0, action_acc_1

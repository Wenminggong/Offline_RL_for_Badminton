# -*- coding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2024/05/13 15:01:49
@Author  :   Mingjiang Liu 
@Version :   1.0
@Desc    :   some useful functions for decision transformer
'''


import random
import numpy as np
import torch
import pandas as pd
import os


def set_seed(seed):
    # set random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def discount_cumsum(x, gamma):
    # return discount return for every timestep
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


# def get_batch_data_from_shuttleset(trajectories, device, last_time_shot_type_dim, hit_xy_dim, player_location_xy_dim, opponent_location_xy_dim, shot_type_dim, landing_xy_dim, move_xy_dim):
#     # get batch_size data from shuttle, shuttleset or shuttleset22
#     last_time_shot_type, hit_xy, player_location_xy, opponent_location_xy, shot_type, landing_xy, move_xy = [], [], [], [], [], [], []
#     reward, rtg = [], []
#     timesteps, mask, move_mask = [], [], []
#     max_len = 0

#     for j in range(len(trajectories)):
#         traj = trajectories[j]

#         # get sequences from dataset
#         last_time_shot_type.append(traj['last_time_opponent_type'].reshape(1, -1, last_time_shot_type_dim))
#         hit_xy.append(traj['hit_xy'].reshape(1, -1, hit_xy_dim))
#         player_location_xy.append(traj['player_location_xy'].reshape(1, -1, player_location_xy_dim))
#         opponent_location_xy.append(traj["opponent_location_xy"].reshape(1, -1, opponent_location_xy_dim))
#         shot_type.append(traj["cur_time_shot_type"].reshape(1, -1, shot_type_dim))
#         landing_xy.append(traj["landing_xy"].reshape(1, -1, landing_xy_dim))
#         move_xy.append(traj['move_xy'].reshape(1, -1, move_xy_dim))
#         reward.append(traj['reward'].reshape(1, -1, 1))
#         rtg.append(discount_cumsum(traj['reward'], gamma=1).reshape(1, -1, 1))
#         timesteps.append(np.arange(traj["action_num"]).reshape(1, -1))
        
#         max_len = max(max_len, traj["action_num"])
    
#     for j in range(len(trajectories)):
#         # padding
#         tlen = last_time_shot_type[j].shape[1]
#         last_time_shot_type[j] = np.concatenate([np.zeros((1, max_len - tlen, last_time_shot_type_dim)), last_time_shot_type[j]], axis=1)
#         hit_xy[j] = np.concatenate([np.zeros((1, max_len - tlen, hit_xy_dim)), hit_xy[j]], axis=1)
#         player_location_xy[j] = np.concatenate([np.zeros((1, max_len - tlen, player_location_xy_dim)), player_location_xy[j]], axis=1)
#         opponent_location_xy[j] = np.concatenate([np.zeros((1, max_len - tlen, opponent_location_xy_dim)), opponent_location_xy[j]], axis=1)
#         shot_type[j] = np.concatenate([np.zeros((1, max_len - tlen, shot_type_dim)), shot_type[j]], axis=1)
#         landing_xy[j] = np.concatenate([np.zeros((1, max_len - tlen, landing_xy_dim)), landing_xy[j]], axis=1)
#         move_xy[j] = np.concatenate([np.zeros((1, max_len - tlen, move_xy_dim)), move_xy[j]], axis=1)
#         reward[j] = np.concatenate([np.zeros((1, max_len - tlen, 1)), reward[j]], axis=1)
#         rtg[j] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[j]], axis=1)

#         timesteps[j] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[j]], axis=1)
#         cur_mask = np.ones((1, tlen))
#         cur_move_mask = np.ones((1, tlen))
#         if last_time_shot_type[j][0].sum() == 0:
#             # mask service action
#             cur_mask[0][0] = 0
#             cur_move_mask[0][0] = 0
#         mask.append(np.concatenate([np.zeros((1, max_len - tlen)), cur_mask], axis=1))
#         if move_xy[j][-1].sum() == 0:
#             # mask last move action
#             cur_move_mask[0][-1] = 0
#         move_mask.append(np.concatenate([np.zeros((1, max_len - tlen)), cur_move_mask], axis=1))
    
#     last_time_shot_type = torch.from_numpy(np.concatenate(last_time_shot_type, axis=0)).to(dtype=torch.float32, device=device)
#     hit_xy = torch.from_numpy(np.concatenate(hit_xy, axis=0)).to(dtype=torch.float32, device=device)
#     player_location_xy = torch.from_numpy(np.concatenate(player_location_xy, axis=0)).to(dtype=torch.float32, device=device)
#     opponent_location_xy = torch.from_numpy(np.concatenate(opponent_location_xy, axis=0)).to(dtype=torch.float32, device=device)
#     shot_type = torch.from_numpy(np.concatenate(shot_type, axis=0)).to(dtype=torch.float32, device=device)
#     landing_xy = torch.from_numpy(np.concatenate(landing_xy, axis=0)).to(dtype=torch.float32, device=device)
#     move_xy = torch.from_numpy(np.concatenate(move_xy, axis=0)).to(dtype=torch.float32, device=device)
#     reward = torch.from_numpy(np.concatenate(reward, axis=0)).to(dtype=torch.float32, device=device)
#     rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)

#     timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
#     mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
#     move_mask = torch.from_numpy(np.concatenate(move_mask, axis=0)).to(device=device)
    
#     return last_time_shot_type, hit_xy, player_location_xy, opponent_location_xy, shot_type, landing_xy, move_xy, reward, rtg, timesteps, mask, move_mask


def get_batch_data_from_shuttleset(trajectories, device):
    # get batch_size data from shuttle, shuttleset or shuttleset22
    batch_data = dict()
    mask, move_mask = [], []
    max_len = 0

    # trajectories: List = [{key,value}] to batch_data:Dict = {key:List}
    for j in range(len(trajectories)):
        traj = trajectories[j]
        for key in traj.keys():
            if key in batch_data.keys():
                if isinstance(traj[key], (int, float, bool, str, type(None), np.int64)):
                    # if is scalar
                    batch_data[key].append(traj[key])
                elif isinstance(traj[key], np.ndarray):
                    # if is ndarray
                    if len(traj[key].shape) < 2:
                        batch_data[key].append(traj[key].reshape(1, -1, 1))
                    else:
                        batch_data[key].append(traj[key].reshape(1, -1, traj[key].shape[-1]))
                else:
                    raise NotImplementedError
                
            else:
                if isinstance(traj[key], (int, float, bool, str, type(None), np.int64)):
                    # if is scalar
                    batch_data[key] = [traj[key]]
                elif isinstance(traj[key], np.ndarray):
                    # if is ndarray
                    if len(traj[key].shape) < 2:
                        batch_data[key] = [traj[key].reshape(1, -1, 1)]
                    else:
                        batch_data[key] = [traj[key].reshape(1, -1, traj[key].shape[-1])]
                else:
                    raise NotImplementedError
        
        if "rtg" in batch_data.keys():
            batch_data["rtg"].append(discount_cumsum(traj['reward'], gamma=1).reshape(1, -1, 1))
            batch_data["timesteps"].append(np.arange(len(traj["reward"])).reshape(1, -1, 1))
        else:
            batch_data["rtg"] = [discount_cumsum(traj['reward'], gamma=1).reshape(1, -1, 1)]
            batch_data["timesteps"] = [np.arange(len(traj["reward"])).reshape(1, -1, 1)]
        
        max_len = max(max_len, len(traj["reward"]))
    
    # batch_data: Dict={key:List}, padding sequence
    for j in range(len(trajectories)):
        # padding
        tlen = batch_data["reward"][j].shape[1]
        for key in batch_data.keys():
            if isinstance(batch_data[key][j], np.ndarray):
                batch_data[key][j] = np.concatenate([np.zeros((1, max_len - tlen, batch_data[key][j].shape[-1])), batch_data[key][j]], axis=1)

        cur_mask = np.ones((1, tlen))
        cur_move_mask = np.ones((1, tlen))
        if batch_data["hit_xy"][j][0].sum() == 0:
            # mask service action
            cur_mask[0][0] = 0
            cur_move_mask[0][0] = 0
        mask.append(np.concatenate([np.zeros((1, max_len - tlen)), cur_mask], axis=1))
        if batch_data["move_xy"][j][-1].sum() == 0:
            # mask last move action
            cur_move_mask[0][-1] = 0
        move_mask.append(np.concatenate([np.zeros((1, max_len - tlen)), cur_move_mask], axis=1))
    
    batch_data["mask"] = mask # [batch_size, max_len]
    batch_data["move_mask"] = move_mask
    
    # batch_data: Dict={key:List} to Dict={key:tensor}
    for key in batch_data.keys():
        if isinstance(batch_data[key][0], np.ndarray):
            batch_data[key] = torch.from_numpy(np.concatenate(batch_data[key], axis=0)).to(device=device) # [batch_size, max_len, fea_dim]
        else:
            if not isinstance(batch_data[key][0], str):
                # if is not str
                batch_data[key] = torch.tensor(batch_data[key]).unsqueeze(dim=-1).expand(len(mask), max_len).unsqueeze(dim=-1).to(device=device) # [batch_size, max_len, 1]
    
    # return batch_data:Dict={key:tensor or List(str)}
    return batch_data


def save_values_to_csv(file_path, value_dict):
    # value_dict: {key: [value], ...}
    new_data = pd.DataFrame(value_dict)
    file_path = os.path.join(file_path, "result.csv")
    if not os.path.exists(file_path):
        new_data.to_csv(file_path, index=False)
    else:
        # add value to exist file
        new_data.to_csv(file_path, mode='a', header=False, index=False)
# -*- coding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2024/06/09 20:13:29
@Author  :   Mingjiang Liu 
@Version :   1.0
@Desc    :   some useful functions for offline RL
'''

import numpy as np
import torch.nn as nn


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def convert_data_to_drl(trajectories, next_action: bool=False):
    # convert dataset to drl-form, i.g., {"observations", "actions", "rewards", "terminals", "next_observations"}
    dataset = {}
    last_time_shot_type_list = []
    hit_xy_list = []
    player_location_xy_list = []
    opponent_location_xy_list = []
    shot_type_list = []
    landing_xy_list = []
    move_xy_list = []
    reward_list = []
    terminal_list = []
    if next_action:
        pred_shot_type_list = []
        pred_shot_prob_list = []
        pred_landing_xy_list = []
        pred_move_xy_list = []
    for rally in trajectories:
        if rally["move_xy"][-1].sum() == 0:
            # panding last move_xy with [0.5, 0.5]
            rally["move_xy"][-1] = 0.5
        if rally["hit_xy"][0].sum() == 0:
            # if first action is service
            last_time_shot_type_list.append(rally["last_time_opponent_type"][1:])
            hit_xy_list.append(rally["hit_xy"][1:])
            player_location_xy_list.append(rally["player_location_xy"][1:])
            opponent_location_xy_list.append(rally["opponent_location_xy"][1:])
            shot_type_list.append(rally["shot_type"][1:])
            landing_xy_list.append(rally["landing_xy"][1:])
            move_xy_list.append(rally["move_xy"][1:])
            reward_list.append(rally["reward"][1:])
            terminal_list.append(rally["terminal"][1:])
            if next_action:
                pred_shot_type_list.append(rally["pred_shot_type"][1:])
                pred_shot_prob_list.append(rally["pred_shot_probs"][1:])
                pred_landing_xy_list.append(rally["pred_landing_xy"][1:])
                pred_move_xy_list.append(rally["pred_move_xy"][1:])
        else:
            last_time_shot_type_list.append(rally["last_time_opponent_type"])
            hit_xy_list.append(rally["hit_xy"])
            player_location_xy_list.append(rally["player_location_xy"])
            opponent_location_xy_list.append(rally["opponent_location_xy"])
            shot_type_list.append(rally["shot_type"])
            landing_xy_list.append(rally["landing_xy"])
            move_xy_list.append(rally["move_xy"])
            reward_list.append(rally["reward"])
            terminal_list.append(rally["terminal"])
            if next_action:
                pred_shot_type_list.append(rally["pred_shot_type"])
                pred_shot_prob_list.append(rally["pred_shot_probs"])
                pred_landing_xy_list.append(rally["pred_landing_xy"])
                pred_move_xy_list.append(rally["pred_move_xy"])

    last_time_shot_type = np.concatenate(last_time_shot_type_list, axis=0)
    hit_xy = np.concatenate(hit_xy_list, axis=0)
    player_location_xy = np.concatenate(player_location_xy_list, axis=0)
    opponent_location_xy = np.concatenate(opponent_location_xy_list, axis=0)
    shot_type = np.concatenate(shot_type_list, axis=0)
    landing_xy = np.concatenate(landing_xy_list, axis=0)
    move_xy = np.concatenate(move_xy_list, axis=0)
    reward = np.concatenate(reward_list, axis=0)
    terminal = np.concatenate(terminal_list, axis=0)
    if next_action:
        pred_shot_type = np.concatenate(pred_shot_type_list, axis=0)
        pred_shot_prob = np.concatenate(pred_shot_prob_list, axis=0)
        pred_landing_xy = np.concatenate(pred_landing_xy_list, axis=0)
        pred_move_xy = np.concatenate(pred_move_xy_list, axis=0)

    next_last_time_shot_type = np.empty_like(last_time_shot_type)
    next_last_time_shot_type[:-1] = last_time_shot_type[1:].copy()
    next_last_time_shot_type[-1] = last_time_shot_type[0].copy()
    next_hit_xy = np.empty_like(hit_xy)
    next_hit_xy[:-1] = hit_xy[1:].copy()
    next_hit_xy[-1] = hit_xy[0].copy()
    next_player_location_xy = np.empty_like(player_location_xy)
    next_player_location_xy[:-1] = player_location_xy[1:].copy()
    next_player_location_xy[-1] = player_location_xy[0].copy()
    next_opponent_location_xy = np.empty_like(opponent_location_xy)
    next_opponent_location_xy[:-1] = opponent_location_xy[1:].copy()
    next_opponent_location_xy[-1] = opponent_location_xy[0].copy()

    dataset["last_time_shot_type"] = last_time_shot_type # [batch_size]
    dataset["hit_xy"] = hit_xy # [batch_size, n]
    dataset["player_location_xy"] = player_location_xy
    dataset["opponent_location_xy"] = opponent_location_xy
    dataset["next_last_time_shot_type"] = next_last_time_shot_type # [batch_size]
    dataset["next_hit_xy"] = next_hit_xy
    dataset["next_player_location_xy"] = next_player_location_xy
    dataset["next_opponent_location_xy"] = next_opponent_location_xy
    dataset["shot_type"] = shot_type # [batch_size]
    dataset["landing_xy"] = landing_xy
    dataset["move_xy"] = move_xy
    dataset["rewards"] = reward # [batch_size]
    dataset["terminals"] = terminal # [batch_size]
    
    if next_action:
        dataset["pred_shot_type"] = pred_shot_type
        dataset["pred_landing_xy"] = pred_landing_xy
        dataset["pred_move_xy"] = pred_move_xy
        next_shot_type = np.empty_like(pred_shot_type)
        next_shot_type[:-1] = pred_shot_type[1:].copy()
        next_shot_type[-1] = pred_shot_type[0].copy()
        next_shot_prob = np.empty_like(pred_shot_prob)
        next_shot_prob[:-1] = pred_shot_prob[1:].copy()
        next_shot_prob[-1] = pred_shot_prob[0].copy()
        next_landing_xy = np.empty_like(pred_landing_xy)
        next_landing_xy[:-1] = pred_landing_xy[1:].copy()
        next_landing_xy[-1] = pred_landing_xy[0].copy()
        next_move_xy = np.empty_like(pred_move_xy)
        next_move_xy[:-1] = pred_move_xy[1:].copy()
        next_move_xy[-1] = pred_move_xy[0].copy()
        dataset["next_shot_type"] = next_shot_type 
        dataset["next_shot_probs"] = next_shot_prob
        dataset["next_landing_xy"] = next_landing_xy
        dataset["next_move_xy"] = next_move_xy

    return dataset
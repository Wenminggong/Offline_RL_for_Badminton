# -*- coding: utf-8 -*-
'''
@File    :   rule_based_metrics_tactics_main.py
@Time    :   2024/06/23 14:32:55
@Author  :   Mingjiang Liu 
@Version :   1.0
@Desc    :   rule-based domain metrics for tactical policy evaluation.
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import argparse
import os
import pandas as pd
from tqdm import tqdm

from decision_transformer.models.dt_for_tactics_generation import DecisionTransformerTactics
from decision_transformer.models.dt_based_bc_for_tactics_generation import DecisionTransformerBCTactics
from offline_rl.models.actor import GaussianPolicy
from offline_rl.utils import convert_data_to_drl
from decision_transformer.utils import get_batch_data_from_shuttleset, save_values_to_csv
from data.preprocess_badminton_data import ACTIONS

from typing import Dict


ACTIVE_SHOT_TYPE = {
    "drop": 4,
    'push/rush': 5,
    "smash": 6,
}


def compute_rule_metrics(rally: Dict, relation_table: pd.DataFrame, freq_threshold: int) -> Dict:
    loss_rally_num = 0
    win_rally_num = 0
    loss_action_num = 0
    win_action_num = 0
    loss_diff_rally_num = 0
    win_diff_rally_num = 0
    loss_diff_action_num = 0
    win_diff_action_num = 0
    loss_unrea_action_num = 0
    win_unrea_action_num = 0
    win_active_action_num = 0
    loss_active_action_num = 0
    landing_oppo_distance = []
    landing_out_action_num = 0
    landing_pred_distance = []
    move_player_distance = []
    move_center_distance = []
    move_pred_distance = []

    if len(rally["reward"]) <= 1 and rally["hit_xy"][0].sum() == 0:
        return {
        "loss_rally_num": loss_rally_num,
        "win_rally_num": win_rally_num,
        "loss_action_num": loss_action_num,
        "win_action_num": win_action_num,
        "loss_diff_rally_num": loss_diff_rally_num,
        "win_diff_rally_num": win_diff_rally_num,
        "loss_diff_action_num": loss_diff_action_num,
        "win_diff_action_num": win_diff_action_num,
        "loss_unrea_action_num": loss_unrea_action_num,
        "win_unrea_action_num": win_unrea_action_num,
        "landing_oppo_distance": landing_oppo_distance,
        "landing_out_action_num": landing_out_action_num,
        "move_player_distance": move_player_distance,
        "move_center_distance": move_center_distance,
    }
    
    if rally["hit_xy"][0].sum() == 0:
        rewards = rally["reward"][1:]
        last_oppo_shot_type = rally["last_time_opponent_type"][1:]
        shot_type = rally["shot_type"][1:]
        pred_shot_type = rally["pred_shot_type"][1:]
        player_location_xy = rally["player_location_xy"][1:]
        opponent_location_xy = rally["opponent_location_xy"][1:]
        pred_landing_xy = rally["pred_landing_xy"][1:]
        landing_xy = rally["landing_xy"][1:]
        pred_move_xy = rally["pred_move_xy"][1:]
        move_xy = rally["move_xy"][1:]
    else:
        rewards = rally["reward"]
        last_oppo_shot_type = rally["last_time_opponent_type"]
        shot_type = rally["shot_type"]
        pred_shot_type = rally["pred_shot_type"]
        player_location_xy = rally["player_location_xy"]
        opponent_location_xy = rally["opponent_location_xy"]
        pred_landing_xy = rally["pred_landing_xy"]
        landing_xy = rally["landing_xy"]
        pred_move_xy = rally["pred_move_xy"]
        move_xy = rally["move_xy"]
    
    pred_shot_type = np.argmax(pred_shot_type, axis=-1)

    diff_action_num = 0
    unrea_action_num = 0
    diff_rally_num = 0
    active_action_num = 0
    I_ACTIONS = {v:k for k,v in ACTIONS.items()}
    for i in range(len(rewards)):
        if shot_type[i] != pred_shot_type[i]:
            # if diff shot_type
            diff_action_num += 1 
        
        # if unreasonable shot type
        if relation_table.loc[I_ACTIONS[last_oppo_shot_type[i]], I_ACTIONS[pred_shot_type[i]]] < freq_threshold:
            unrea_action_num += 1

        # if use active shot type
        if pred_shot_type[i] in ACTIVE_SHOT_TYPE.values():
            active_action_num += 1

        # if landing out
        if pred_landing_xy[i][0] <= 0 or pred_landing_xy[i][0] >= 1 or pred_landing_xy[i][1] <= 0 or pred_landing_xy[i][1] >= 1:
            landing_out_action_num += 1
        
        landing_oppo_distance.append(np.linalg.norm(pred_landing_xy[i] - opponent_location_xy[i]))
        landing_pred_distance.append(np.linalg.norm(pred_landing_xy[i] - landing_xy[i]))
        move_player_distance.append(np.linalg.norm(pred_move_xy[i] - player_location_xy[i]))
        move_center_distance.append(np.linalg.norm(pred_move_xy[i] - np.array([0.5, 0.5])))
        if not ((i == len(rewards)-1) and move_xy[i].sum() == 0):
            # no consider last move
            move_pred_distance.append(np.linalg.norm(pred_move_xy[i] - move_xy[i]))

    if diff_action_num > 0:
        diff_rally_num += 1

    if rewards.sum() > 0:
        # win rally
        win_rally_num += 1
        win_action_num += len(rewards)
        win_diff_action_num += diff_action_num
        win_diff_rally_num += diff_rally_num
        win_unrea_action_num += unrea_action_num
        win_active_action_num += active_action_num
    else:
        # loss rally
        loss_rally_num += 1
        loss_action_num += len(rewards)
        loss_diff_action_num += diff_action_num
        loss_diff_rally_num += diff_rally_num
        loss_unrea_action_num += unrea_action_num
        loss_active_action_num += active_action_num

    return {
        "loss_rally_num": loss_rally_num,
        "win_rally_num": win_rally_num,
        "loss_action_num": loss_action_num,
        "win_action_num": win_action_num,
        "loss_diff_rally_num": loss_diff_rally_num,
        "win_diff_rally_num": win_diff_rally_num,
        "loss_diff_action_num": loss_diff_action_num,
        "win_diff_action_num": win_diff_action_num,
        "loss_unrea_action_num": loss_unrea_action_num,
        "win_unrea_action_num": win_unrea_action_num,
        "loss_active_action_num": loss_active_action_num,
        "win_active_action_num": win_active_action_num,
        "landing_oppo_distance": landing_oppo_distance,
        "landing_out_action_num": landing_out_action_num,
        "landing_pred_distance": landing_pred_distance,
        "move_player_distance": move_player_distance,
        "move_center_distance": move_center_distance,
        "move_pred_distance": move_pred_distance,
    }


def rule_evaluation(variant) -> None:
    ori_dataset_path = variant["eval_dataset"]
    policy_type = variant["policy_type"]

    # load validation data
    dataset_path = f'data/{ori_dataset_path}_val.pkl'
    with open(dataset_path, "rb") as f:
        # trajectories = [{}, {}, ...]
        trajectories = pickle.load(f)

    last_time_shot_type_dim = len(ACTIONS)
    hit_xy_dim = trajectories[0]["hit_xy"].shape[1]
    player_location_xy_dim = trajectories[0]["player_location_xy"].shape[1]
    opponent_location_xy_dim = trajectories[0]["opponent_location_xy"].shape[1]
    shot_type_dim = len(ACTIONS)
    landing_xy_dim = trajectories[0]["landing_xy"].shape[1]
    move_xy_dim = trajectories[0]["move_xy"].shape[1]
    state_dim = last_time_shot_type_dim + hit_xy_dim + player_location_xy_dim + opponent_location_xy_dim
    action_dim = shot_type_dim + landing_xy_dim + move_xy_dim

    if variant["policy_activation_function"] == 'relu':
        activation_function = nn.ReLU()
    elif variant["policy_activation_function"] == 'tanh':
        activation_function = nn.Tanh()
    else:
        raise NotImplementedError
    
     # load policy
    if policy_type == "cql" or policy_type == "mlp_bc":
        actor = GaussianPolicy(
            state_dim,
            action_dim,
            last_time_shot_type_dim,
            hit_xy_dim,
            player_location_xy_dim,
            opponent_location_xy_dim,
            shot_type_dim,
            landing_xy_dim,
            move_xy_dim,
            variant["policy_orthogonal_init"],
            variant["policy_n_hidden_layers"],
            variant["policy_hidden_dims"],
            variant["policy_embedding_dim"],
            activation_function,
            embedding_coordinate=variant["policy_embedding_coordinate"]
            )
        if policy_type == "cql":
            policy_path = os.path.join(variant["policy_path"], f"{ori_dataset_path}_{policy_type}", "{}_g{}_a{}_b{}_h{}_n{}_e{}_s{}.pth".format(policy_type, variant["cql_target_action_gap"], variant["cql_tune_init_log_alpha"], variant["policy_batch_size"], variant["policy_hidden_dims"], variant["policy_n_hidden_layers"], variant["policy_embedding_dim"], variant["policy_seed"]))
        else:
            policy_path = os.path.join(variant["policy_path"], f"{ori_dataset_path}_{policy_type}", "{}_b{}_h{}_n{}_e{}_s{}.pth".format(policy_type, variant["policy_batch_size"], variant["policy_hidden_dims"], variant["policy_n_hidden_layers"], variant["policy_embedding_dim"], variant["policy_seed"]))
    elif policy_type == "dt":
        max_ep_len = 1024
        actor = DecisionTransformerTactics(
            state_dim=state_dim,
            act_dim=action_dim,
            last_time_shot_type_dim=last_time_shot_type_dim,
            hit_xy_dim=hit_xy_dim,
            player_location_xy_dim=player_location_xy_dim,
            opponent_location_xy_dim=opponent_location_xy_dim,
            shot_type_dim=shot_type_dim,
            landing_xy_dim=landing_xy_dim,
            move_xy_dim=move_xy_dim,
            max_ep_len=max_ep_len+32,
            hidden_size=variant['policy_hidden_dims'],
            embed_size=variant['policy_embedding_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['policy_hidden_dims'],
            activation_function=variant['policy_activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
            use_player_location=1,
            embed_coordinate=variant["policy_embedding_coordinate"]
        )
        policy_path = os.path.join(variant["policy_path"], f"{ori_dataset_path}_{policy_type}", "{}_b{}_h{}_e{}_nl{}_nh{}_s{}.pth".format(policy_type, variant["policy_batch_size"], variant["policy_hidden_dims"], variant["policy_embedding_dim"], variant["n_layer"], variant["n_head"], variant["policy_seed"]))
    elif policy_type == "bc":
        max_ep_len = 1024
        actor = DecisionTransformerBCTactics(
            state_dim=state_dim,
            act_dim=action_dim,
            last_time_shot_type_dim=last_time_shot_type_dim,
            hit_xy_dim=hit_xy_dim,
            player_location_xy_dim=player_location_xy_dim,
            opponent_location_xy_dim=opponent_location_xy_dim,
            shot_type_dim=shot_type_dim,
            landing_xy_dim=landing_xy_dim,
            move_xy_dim=move_xy_dim,
            max_ep_len=max_ep_len+32,
            hidden_size=variant['policy_hidden_dims'],
            embed_size=variant['policy_embedding_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['policy_hidden_dims'],
            activation_function=variant['policy_activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
            use_player_location=1,
            embed_coordinate=variant["policy_embedding_coordinate"]
        )
        policy_path = os.path.join(variant["policy_path"], f"{ori_dataset_path}_{policy_type}", "{}_b{}_h{}_e{}_nl{}_nh{}_s{}.pth".format(policy_type, variant["policy_batch_size"], variant["policy_hidden_dims"], variant["policy_embedding_dim"], variant["n_layer"], variant["n_head"], variant["policy_seed"]))
    else:
        raise NotImplementedError
    actor.load_state_dict(torch.load(policy_path))
    actor.to(variant["device"])

    # load relation table
    relation_table = pd.read_csv(variant["relation_table_path"])
    relation_table.set_index(relation_table.columns[0], inplace=True)

    # evaluate
    # total_rally_num = 0
    # total_action_num = 0
    loss_rally_num = 0
    win_rally_num = 0
    loss_action_num = 0
    win_action_num = 0
    loss_diff_rally_num = 0
    win_diff_rally_num = 0
    loss_diff_action_num = 0
    win_diff_action_num = 0
    loss_unrea_action_num = 0
    win_unrea_action_num = 0
    loss_active_action_num = 0
    win_active_action_num = 0
    landing_oppo_distance = []
    landing_pred_distance = []
    landing_out_action_num = 0
    move_player_distance = []
    move_center_distance = []
    move_pred_distance = []
    actor.eval()
    for rally in tqdm(trajectories):
        if len(rally["reward"]) <= 1 and rally["hit_xy"][0].sum() == 0:
            continue

        # total_rally_num += 1
        # total_action_num += len(rally["reward"])
    
        # rally = {"action_num", "last_time_opponent_type", "cur_time_shot_type", "hit_xy", "player_location_xy", "opponent_location_xy", "move_xy", "landing_xy", "reward", "terminal"}
        if policy_type == "cql" or policy_type == "mlp_bc":
            # convert data to {observations, actions, rewards, next_observations, terminals}, 
            dataset = convert_data_to_drl([rally])
            last_time_shot_type = torch.from_numpy(dataset["last_time_shot_type"]) # [seq]
            hit_xy = torch.from_numpy(dataset["hit_xy"])
            player_location_xy = torch.from_numpy(dataset["player_location_xy"])
            opponent_location_xy = torch.from_numpy(dataset["opponent_location_xy"])

            # [batch_size, n] - ndarray
            with torch.no_grad():
                pred_shot_type, pred_shot_probs, pred_landing_xy, pred_move_xy = actor.act(
                    F.one_hot(last_time_shot_type, shot_type_dim).to(dtype=torch.float32),
                    hit_xy,
                    player_location_xy,
                    opponent_location_xy,
                    deterministic=variant["deterministic"],
                    device=variant["device"]
                )
            pred_shot_type = F.one_hot(torch.from_numpy(pred_shot_type).squeeze(dim=-1), num_classes=10) # [batch_size, shot_type_dim] - tensor
            # if service shot, seq_num = ori_seq_num - 1
            if pred_shot_type.shape[0] != rally["last_time_opponent_type"].shape[0]:
                pred_shot_type = np.concatenate([np.zeros((1, shot_type_dim)), pred_shot_type.numpy()], axis=0)
                pred_landing_xy = np.concatenate([np.zeros((1, landing_xy_dim)), pred_landing_xy], axis=0)
                pred_move_xy = np.concatenate([np.zeros((1, move_xy_dim)), pred_move_xy], axis=0)
            else:
                pred_shot_type = pred_shot_type.numpy()
            rally["pred_shot_type"] = pred_shot_type # [seq, 10]
            rally["pred_landing_xy"] = pred_landing_xy
            rally["pred_move_xy"] = pred_landing_xy
        elif policy_type == "dt" or policy_type == "bc":
            # [1, seq, m] - tensor
            batch_data = get_batch_data_from_shuttleset(
                [rally], 
                variant["device"], 
            )
            last_time_shot_type = batch_data["last_time_opponent_type"].squeeze(dim=-1).to(dtype=torch.long) # [batch_size, max_len]
            hit_xy = batch_data["hit_xy"].to(dtype=torch.float32) # [batch_size, max_len, 2]
            player_location_xy = batch_data["player_location_xy"].to(dtype=torch.float32) # [batch_size, max_len, 2]
            opponent_location_xy = batch_data["opponent_location_xy"].to(dtype=torch.float32) # [batch_size, max_len, 2]
            shot_type = batch_data["shot_type"].squeeze(dim=-1).to(dtype=torch.long) # [batch_size, max_len]
            landing_xy = batch_data["landing_xy"].to(dtype=torch.float32) # [batch_size, max_len, 2]
            move_xy = batch_data["move_xy"].to(dtype=torch.float32) # [batch_size, max_len, 2]
            reward = batch_data["reward"].to(dtype=torch.float32) # [batch_size, max_len, 1]
            timesteps = batch_data["timesteps"].squeeze().to(dtype=torch.long).unsqueeze(dim=0) # [batch_size, max_len]
            rtg = batch_data["rtg"].to(dtype=torch.float32) # [batch_size, max_len, 1]
            mask = batch_data["mask"].to(dtype=torch.float32) # [batch_size, max_len]
            move_mask = batch_data["move_mask"].to(dtype=torch.float32) # [batch_size, max_len]

            if variant["use_win_return"]:
                rtg = torch.ones_like(rtg)
            # [1, seq, m] - tensor
            with torch.no_grad():
                shot_preds, landing_distribution, move_distribution = actor.get_action(
                    last_time_shot_type, 
                    hit_xy, 
                    player_location_xy, 
                    opponent_location_xy, 
                    shot_type, 
                    landing_xy, 
                    move_xy, 
                    reward, 
                    rtg, 
                    timesteps, 
                    mask,
                )
            shot_probs = F.softmax(shot_preds, dim=-1) # [1, seq, shot_type_dim] tensor
            shot_distribution = torch.distributions.Categorical(probs=shot_probs)
            if variant["deterministic"]:
                shot_sample = torch.argmax(shot_probs, dim=-1, keepdim=True) # [1, seq, 1] tensor
                landxing_xy_sample = landing_distribution.mean # [1, seq, xy_dim] tensor
                move_xy_sample = move_distribution.mean
            else:
                shot_sample = shot_distribution.sample().unsqueeze(dim=-1) # [1, seq, 1] tensor
                landxing_xy_sample = landing_distribution.sample() # [1, seq, xy_dim] tensor
                move_xy_sample = move_distribution.sample()

            shot_sample = F.one_hot(shot_sample.squeeze(dim=-1), num_classes=10) # [1, batch_size, shot_type_dim]
            rally["pred_shot_type"] = shot_sample.squeeze(dim=0).cpu().numpy() # [seq, shot_type_dim] ndarray
            rally["pred_landing_xy"] = landxing_xy_sample.squeeze(dim=0).cpu().numpy()
            rally["pred_move_xy"] = move_xy_sample.squeeze(dim=0).cpu().numpy()
        else:
            raise NotImplementedError
        
        # rally = {..., "pred_shot_type", "pred_landing_xy", "pred_move_xy"}
        result = compute_rule_metrics(rally, relation_table, variant["freq_threshold"])
        loss_rally_num += result["loss_rally_num"]
        win_rally_num += result["win_rally_num"]
        loss_action_num += result["loss_action_num"]
        win_action_num += result["win_action_num"]
        loss_diff_rally_num += result["loss_diff_rally_num"]
        win_diff_rally_num += result["win_diff_rally_num"]
        loss_diff_action_num += result["loss_diff_action_num"]
        win_diff_action_num += result["win_diff_action_num"]
        loss_unrea_action_num += result["loss_unrea_action_num"]
        win_unrea_action_num += result["win_unrea_action_num"]
        loss_active_action_num += result["loss_active_action_num"]
        win_active_action_num += result["win_active_action_num"]
        landing_oppo_distance += result["landing_oppo_distance"]
        landing_out_action_num += result["landing_out_action_num"]
        landing_pred_distance += result["landing_pred_distance"]
        move_player_distance += result["move_player_distance"]
        move_center_distance += result["move_center_distance"]
        move_pred_distance += result["move_pred_distance"]

    model_path = os.path.join(variant["save_path"], f"{ori_dataset_path}_{policy_type}")
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if policy_type == "cql" or policy_type == "mlp_bc":
        save_value_dict = {
            "batch_size": [variant["policy_batch_size"]],
            "cql_target_action_gap": [variant["cql_target_action_gap"]],
            "cql_tune_init_log_alpha": [variant["cql_tune_init_log_alpha"]],
            "hidden_size": [variant["policy_hidden_dims"]],
            "n_layer": [variant["policy_n_hidden_layers"]],
            "embed_size": [variant["policy_embedding_dim"]],
            "deterministic": [variant["deterministic"]],
            "policy_seed": [variant["policy_seed"]],
            "total_rally_num": loss_rally_num+win_rally_num,
            "loss_rally_num": loss_rally_num,
            "win_rally_num": win_rally_num,
            "total_action_num": loss_action_num+win_action_num,
            "loss_action_num": loss_action_num,
            "win_action_num": win_action_num,
            "loss_diff_rally_num": loss_diff_rally_num,
            "win_diff_rally_num": win_diff_rally_num,
            "loss_diff_action_num": loss_diff_action_num,
            "win_diff_action_num": win_diff_action_num,
            "loss_unrea_action_num": loss_unrea_action_num,
            "win_unrea_action_num": win_unrea_action_num,
            "loss_active_action_num": loss_active_action_num,
            "win_active_action_num": win_active_action_num,
            "landing_out_action_num": landing_out_action_num,
            "landing_oppo_distance_mean": np.mean(landing_oppo_distance),
            "landing_oppo_distance_std": np.std(landing_oppo_distance),
            "landing_pred_distance_mean": np.mean(landing_pred_distance),
            "landing_pred_distance_std": np.std(landing_pred_distance),
            "move_player_distance_mean": np.mean(move_player_distance),
            "move_player_distance_std": np.std(move_player_distance),
            "move_center_distance_mean": np.mean(move_center_distance),
            "move_center_distance_std": np.std(move_center_distance),
            "move_pred_distance_mean": np.mean(move_pred_distance),
            "move_pred_distance_std": np.std(move_pred_distance),
            "total_diff_rally_rate": (loss_diff_rally_num + win_diff_rally_num) / (loss_rally_num + win_rally_num),
            "loss_diff_rally_rate": loss_diff_rally_num / loss_rally_num,
            "win_diff_rally_rate": win_diff_rally_num / win_rally_num,
            "total_diff_action_rate": (loss_diff_action_num + win_diff_action_num) / (loss_action_num + win_action_num),
            "loss_diff_action_rate": loss_diff_action_num / loss_action_num,
            "win_diff_action_rate": win_diff_action_num / win_action_num,
            "total_unrea_action_rate": (loss_unrea_action_num + win_unrea_action_num) / (loss_action_num + win_action_num),
            "loss_unrea_action_rate": loss_unrea_action_num / loss_diff_action_num,
            "win_unrea_action_rate": win_unrea_action_num / win_diff_action_num,
            "total_active_action_rate": (loss_active_action_num + win_active_action_num) / (loss_action_num + win_action_num),
            "loss_active_action_rate": loss_active_action_num / loss_diff_action_num,
            "win_active_action_rate": win_active_action_num / win_diff_action_num,
            "landing_out_action_rate": landing_out_action_num / (loss_action_num+win_action_num),
        }
    elif policy_type == "dt" or policy_type == "bc":
        save_value_dict = {
            "batch_size": [variant["policy_batch_size"]],
            "hidden_size": [variant["policy_hidden_dims"]],
            "embed_size": [variant["policy_embedding_dim"]],
            "n_layer": [variant["n_layer"]],
            "n_head": [variant["n_head"]],
            "deterministic": [variant["deterministic"]],
            "use_win_return": [variant["use_win_return"]],
            "policy_seed": [variant["policy_seed"]],
            "total_rally_num": loss_rally_num+win_rally_num,
            "loss_rally_num": loss_rally_num,
            "win_rally_num": win_rally_num,
            "total_action_num": loss_action_num+win_action_num,
            "loss_action_num": loss_action_num,
            "win_action_num": win_action_num,
            "loss_diff_rally_num": loss_diff_rally_num,
            "win_diff_rally_num": win_diff_rally_num,
            "loss_diff_action_num": loss_diff_action_num,
            "win_diff_action_num": win_diff_action_num,
            "loss_unrea_action_num": loss_unrea_action_num,
            "win_unrea_action_num": win_unrea_action_num,
            "loss_active_action_num": loss_active_action_num,
            "win_active_action_num": win_active_action_num,
            "landing_out_action_num": landing_out_action_num,
            "landing_oppo_distance_mean": np.mean(landing_oppo_distance),
            "landing_oppo_distance_std": np.std(landing_oppo_distance),
            "landing_pred_distance_mean": np.mean(landing_pred_distance),
            "landing_pred_distance_std": np.std(landing_pred_distance),
            "move_player_distance_mean": np.mean(move_player_distance),
            "move_player_distance_std": np.std(move_player_distance),
            "move_center_distance_mean": np.mean(move_center_distance),
            "move_center_distance_std": np.std(move_center_distance),
            "move_pred_distance_mean": np.mean(move_pred_distance),
            "move_pred_distance_std": np.std(move_pred_distance),
            "total_diff_rally_rate": (loss_diff_rally_num + win_diff_rally_num) / (loss_rally_num + win_rally_num),
            "loss_diff_rally_rate": loss_diff_rally_num / loss_rally_num,
            "win_diff_rally_rate": win_diff_rally_num / win_rally_num,
            "total_diff_action_rate": (loss_diff_action_num + win_diff_action_num) / (loss_action_num + win_action_num),
            "loss_diff_action_rate": loss_diff_action_num / loss_action_num,
            "win_diff_action_rate": win_diff_action_num / win_action_num,
            "total_unrea_action_rate": (loss_unrea_action_num + win_unrea_action_num) / (loss_action_num + win_action_num),
            "loss_unrea_action_rate": loss_unrea_action_num / loss_diff_action_num,
            "win_unrea_action_rate": win_unrea_action_num / win_diff_action_num,
            "total_active_action_rate": (loss_active_action_num + win_active_action_num) / (loss_action_num + win_action_num),
            "loss_active_action_rate": loss_active_action_num / loss_diff_action_num,
            "win_active_action_rate": win_active_action_num / win_diff_action_num,
            "landing_out_action_rate": landing_out_action_num / (loss_action_num+win_action_num),
        }
    else:
        raise NotImplementedError

    save_values_to_csv(model_path, save_value_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--save_path", type=str, default="domain_metrics_save")
    parser.add_argument("--eval_dataset", type=str, default="shuttle_both_agent")
    parser.add_argument("--freq_threshold", type=int, default=10) # freq_threshold for unreasonable shot_type
    parser.add_argument("--relation_table_path", type=str, default="data/action_relation_table.csv")

    # policy parameters
    parser.add_argument("--policy_type", type=str, default="cql") # "cql", "dt", "bc", or "mlp_bc"
    parser.add_argument("--policy_path", type=str, default="policy_models_save")
    parser.add_argument("--deterministic", type=int, default=1) # deterministic policy or not
    parser.add_argument("--policy_seed", type=int, default=2024)
    parser.add_argument("--policy_activation_function", type=str, default="relu") # activation function for neural network
    parser.add_argument("--policy_batch_size", type=int, default=512)
    parser.add_argument("--policy_embedding_dim", type=int, default=64) # policy embedding dims for shot_type and coordinate
    parser.add_argument("--policy_embedding_coordinate", type=int, default=0) # embedding location coordinate or not
    parser.add_argument("--policy_orthogonal_init", type=int, default=1) # Orthogonal initialization for neural network

    # cql actor
    parser.add_argument("--policy_n_hidden_layers", type=int, default=3) # Number of hidden layers in actor networks
    parser.add_argument("--policy_hidden_dims", type=int, default=512) # hidden layer's dims for actor
    parser.add_argument("--cql_target_action_gap", type=float, default=1.0) # Action gap for CQL regularization, value?
    parser.add_argument("--cql_tune_init_log_alpha", type=float, default=-2.0)
    
    # dt or dt-based bc
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument("--use_win_return", type=int, default=1)

    args = parser.parse_args()
    rule_evaluation(variant=vars(args))
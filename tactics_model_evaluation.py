# -*- coding: utf-8 -*-
'''
@File    :   dt_evaluation.py
@Time    :   2024/08/18 16:34:05
@Author  :   Mingjiang Liu 
@Version :   1.0
@Desc    :   evaluation for tactics generation models, including preference-based reward evaluation and ensemble classification evaluation.
'''


import os
import argparse
import pickle 
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

from decision_transformer.utils import set_seed, get_batch_data_from_shuttleset, save_values_to_csv
from data.preprocess_badminton_data import ACTIONS
from decision_transformer.models.dt_for_tactics_generation import DecisionTransformerTactics
from decision_transformer.models.dt_based_bc_for_tactics_generation import DecisionTransformerBCTactics
from offline_rl.models.actor import GaussianPolicy
from offline_rl.utils import convert_data_to_drl
from prediction.models.reward_model import RewardModel


def filter_rally(trajectories, filter_type):
    # trajectories = [{}, {}]
    if filter_type == "both":
        return trajectories
    elif filter_type == "win":
        pred_reward = 1
    elif filter_type == "loss":
        pred_reward = -1
    else:
        raise NotImplementedError
    
    new_trajectories = []
    for rally in trajectories:
        if rally["reward"][-1] == pred_reward:
            new_trajectories.append(rally)
    return new_trajectories


def evaluation(variant):
    # set random seed
    seed = variant.get("seed", 2024)
    set_seed(seed)

    # load validation data
    original_dataset_path = variant["eval_dataset"]
    dataset_path = "data/{}_val.pkl".format(original_dataset_path)
    with open(dataset_path, "rb") as f:
        # trajectories = [{}, {}, ...]
        trajectories = pickle.load(f)
    
    trajectories = filter_rally(trajectories, variant["filter_type"])

    last_time_shot_type_dim = len(ACTIONS)
    hit_xy_dim = trajectories[0]["hit_xy"].shape[1]
    player_location_xy_dim = trajectories[0]["player_location_xy"].shape[1]
    opponent_location_xy_dim = trajectories[0]["opponent_location_xy"].shape[1]
    shot_type_dim = len(ACTIONS)
    landing_xy_dim = trajectories[0]["landing_xy"].shape[1]
    move_xy_dim = trajectories[0]["move_xy"].shape[1]
    state_dim = last_time_shot_type_dim + hit_xy_dim + player_location_xy_dim + opponent_location_xy_dim
    action_dim = shot_type_dim + landing_xy_dim + move_xy_dim

    if variant["policy_activation"] == 'relu':
        activation_function = nn.ReLU()
    elif variant["policy_activation"] == 'tanh':
        activation_function = nn.Tanh()
    else:
        raise NotImplementedError
    
    # load policy
    policy_type = variant["policy_type"]
    if policy_type == "original":
        # evaluate original data 
        dataset = convert_data_to_drl(trajectories)
        dataset["pred_shot_type"] = dataset["shot_type"] #[batch_size]
        dataset["pred_landing_xy"] = dataset["landing_xy"] # [batch_size, 2]
        dataset["pred_move_xy"] = dataset["move_xy"] # [batch_size, 2]
        # dataset["pred_move_xy"] = np.ones_like(dataset["move_xy"]) * 0.5
    else:
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
                embedding_coordinate=variant["policy_embedding_coordinate"],
                )
            if policy_type == "cql":
                policy_path = os.path.join(variant["policy_path"], f"{original_dataset_path}_{policy_type}", "{}_g{}_a{}_b{}_h{}_n{}_e{}_s{}.pth".format(policy_type, variant["cql_target_action_gap"], variant["cql_tune_init_log_alpha"], variant["policy_batch_size"], variant["policy_hidden_dims"], variant["policy_n_hidden_layers"], variant["policy_embedding_dim"], variant["policy_seed"]))
            else:
                policy_path = os.path.join(variant["policy_path"], f"{original_dataset_path}_{policy_type}", "{}_b{}_h{}_n{}_e{}_s{}.pth".format(policy_type, variant["policy_batch_size"], variant["policy_hidden_dims"], variant["policy_n_hidden_layers"], variant["policy_embedding_dim"], variant["policy_seed"]))
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
                activation_function=variant['policy_activation'],
                n_positions=1024,
                resid_pdrop=variant['dropout'],
                attn_pdrop=variant['dropout'],
                use_player_location=1,
                embed_coordinate=variant["policy_embedding_coordinate"]
            )
            policy_path = os.path.join(variant["policy_path"], f"{original_dataset_path}_{policy_type}", "{}_b{}_h{}_e{}_nl{}_nh{}_s{}.pth".format(policy_type, variant["policy_batch_size"], variant["policy_hidden_dims"], variant["policy_embedding_dim"], variant["n_layer"], variant["n_head"], variant["policy_seed"]))
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
                activation_function=variant['policy_activation'],
                n_positions=1024,
                resid_pdrop=variant['dropout'],
                attn_pdrop=variant['dropout'],
                use_player_location=1,
                embed_coordinate=variant["policy_embedding_coordinate"],
            )
            policy_path = os.path.join(variant["policy_path"], f"{original_dataset_path}_{policy_type}", "{}_b{}_h{}_e{}_nl{}_nh{}_s{}.pth".format(policy_type, variant["policy_batch_size"], variant["policy_hidden_dims"], variant["policy_embedding_dim"], variant["n_layer"], variant["n_head"], variant["policy_seed"]))
        else:
            raise NotImplementedError
        actor.load_state_dict(torch.load(policy_path))
        actor.to(variant["device"])

        actor.eval()
        if policy_type == "cql" or policy_type == "mlp_bc":
            # convert data to {observations, actions, rewards, next_observations, terminals}
            dataset = convert_data_to_drl(trajectories)
            last_time_shot_type = torch.from_numpy(dataset["last_time_shot_type"]) # [batch_sizes]
            last_time_shot_type = F.one_hot(last_time_shot_type, num_classes=last_time_shot_type_dim) # [batch_size, 10]
            hit_xy = torch.from_numpy(dataset["hit_xy"]) # [batch_size, 2]
            player_location_xy = torch.from_numpy(dataset["player_location_xy"]) # [batch_size, 2]
            opponent_location_xy = torch.from_numpy(dataset["opponent_location_xy"]) # [batch_size, 2]

            # [batch_size, n] - ndarray
            pred_shot_type, _, pred_landing_xy, pred_move_xy = actor.act(
                last_time_shot_type,
                hit_xy,
                player_location_xy,
                opponent_location_xy,
                deterministic=variant["deterministic"],
                device=variant["device"]
            )
            dataset["pred_shot_type"] = pred_shot_type.reshape(-1) # [batch_size], ndarray
            dataset["pred_landing_xy"] = pred_landing_xy # [batch_size, 2]
            dataset["pred_move_xy"] = pred_move_xy # [batch_size, 2]

        elif policy_type == "dt" or policy_type == "bc":
            for i in range(len(trajectories)):
                # for each rally, get predicted actions
                # [1, seq, m] - tensor
                batch_data = get_batch_data_from_shuttleset(
                    [trajectories[i]], 
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
                timesteps = batch_data["timesteps"].squeeze(dim=-1).to(dtype=torch.long) # [batch_size, max_len]
                rtg = batch_data["rtg"].to(dtype=torch.float32) # [batch_size, max_len, 1]
                mask = batch_data["mask"].to(dtype=torch.float32) # [batch_size, max_len]

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
                    shot_sample = torch.argmax(shot_probs, dim=-1) # [1, seq] tensor
                    landxing_xy_sample = landing_distribution.mean # [1, seq, xy_dim] tensor
                    move_xy_sample = move_distribution.mean
                else:
                    shot_sample = shot_distribution.sample() # [1, seq] tensor
                    landxing_xy_sample = landing_distribution.sample() # [1, seq, xy_dim] tensor
                    move_xy_sample = move_distribution.sample()

                # shot_sample = F.one_hot(shot_sample.squeeze(dim=-1), num_classes=10) # [1, batch_size, shot_type_dim]
                trajectories[i]["pred_shot_type"] = shot_sample.squeeze(dim=0).cpu().numpy() # [seq] ndarray
                trajectories[i]["pred_shot_probs"] = shot_probs.squeeze(dim=0).cpu().numpy() # [seq, shot_type_dim] ndarray
                trajectories[i]["pred_landing_xy"] = landxing_xy_sample.squeeze(dim=0).cpu().numpy()
                trajectories[i]["pred_move_xy"] = move_xy_sample.squeeze(dim=0).cpu().numpy()

            # convert data to {observations, actions, rewards, next_observations, next_actions, terminals}
            dataset = convert_data_to_drl(trajectories, next_action=True)
        else:
            raise NotImplementedError
    
    # evaluation
    if variant["evaluator_type"] == "reward":
        reward_model = RewardModel(
            pref_model=variant["reward_pref_model"],
            no_end_rally_pref=variant["reward_use_no_end_rally_pref"],
            no_end_rally_pref_mode=variant["reward_no_end_rally_pref_mode"],
            no_end_rally_pref_factor=variant["reward_no_end_rally_pref_factor"],
            loss_type=variant["reward_loss_type"],
            ensemble_size=variant["reward_ensemble_size"],
            action_pref_type=variant["reward_action_pref_type"],
            lr=variant["reward_learning_rate"],
            batch_size=variant["reward_batch_size"],
            action_pref_factor=variant["reward_action_pref_factor"],
            evaluate_flag=1,
            shot_type_num=last_time_shot_type_dim,
            shot_type_dim=variant["reward_shot_type_dim"],
            location_type=variant["reward_location_type"],
            location_num=variant["reward_location_num"],
            location_dim=variant["reward_location_dim"],
            other_fea_dim=variant["reward_other_fea_dim"],
            n_layer=variant["reward_n_layer"],
            hidden_dim=variant["reward_hidden_dim"],
            activation=variant["reward_activation"],
            device=variant["device"],
        )
        reward_model_dir = variant["reward_model_dir"]
        reward_model_name = "b{}_lr{}_t{}_sd{}_lt{}_ld{}_od{}_h{}_nl{}_pm{}_une{}_nem{}_nef{}_lt{}_en{}_at{}_af{}_s{}".format(
            variant["reward_batch_size"],
            variant["reward_learning_rate"], 
            variant["reward_max_epoch"],
            variant["reward_shot_type_dim"],
            variant["reward_location_type"],
            variant["reward_location_dim"],
            variant["reward_other_fea_dim"],
            variant["reward_hidden_dim"], 
            variant["reward_n_layer"],
            variant["reward_pref_model"],
            variant["reward_use_no_end_rally_pref"],
            variant["reward_no_end_rally_pref_mode"],
            variant["reward_no_end_rally_pref_factor"],
            variant["reward_loss_type"],
            variant["reward_ensemble_size"],
            variant["reward_action_pref_type"],
            variant["reward_action_pref_factor"], 
            variant["seed"]
        )
        reward_model.load(reward_model_dir, reward_model_name)

        last_shot_type = torch.from_numpy(dataset["last_time_shot_type"]).to(dtype=torch.long, device=variant["device"])
        hit_area = None
        hit_xy = torch.from_numpy(dataset["hit_xy"]).to(dtype=torch.float32, device=variant["device"])
        player_area = None
        player_xy = torch.from_numpy(dataset["player_location_xy"]).to(dtype=torch.float32, device=variant["device"])
        opponent_area = None
        opponent_xy = torch.from_numpy(dataset["opponent_location_xy"]).to(dtype=torch.float32, device=variant["device"])
        shot_type = torch.from_numpy(dataset["pred_shot_type"]).to(dtype=torch.long, device=variant["device"])
        landing_area = None
        landing_xy = torch.from_numpy(dataset["pred_landing_xy"]).to(dtype=torch.float32, device=variant["device"])
        move_area = None
        move_xy = torch.from_numpy(dataset["pred_move_xy"]).to(dtype=torch.float32, device=variant["device"])
        bad_landing = torch.zeros_like(shot_type)
        bad_landing_flag = torch.cat([
            (landing_xy < 0).any(axis=-1, keepdims=True),
            (landing_xy > 1).any(axis=-1, keepdims=True),
        ], dim=-1).any(dim=-1)
        bad_landing[bad_landing_flag] = 1
        landing_opponent_distance = torch.linalg.norm(landing_xy-opponent_xy, dim=-1)
        landing_fea = torch.cat([bad_landing.unsqueeze(dim=-1), landing_opponent_distance.unsqueeze(dim=-1)], dim=-1).to(dtype=torch.float32, device=variant["device"])

        r_hat = reward_model.r_hat(
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
        ) # [batch_size, 1]
        # # considering all actions
        # result = r_hat.mean().item()

        # # considering non terminal-out actions
        # result = 0
        # count = 0
        # for i in range(len(r_hat)):
        #     if not bad_landing[i] == 1:
        #         result += r_hat[i, 0].item()
        #         count += 1
        # result /= count

        # considering non-terminal actions
        result = 0
        count = 0
        for i in range(len(r_hat)):
            if not (dataset["move_xy"][i, 0] == 0.5 and dataset["move_xy"][i, 1] == 0.5):
                result += r_hat[i, 0].item()
                count += 1
        result /= count
    elif variant["evaluator_type"] == "classify":
        pass
    else:
        raise NotImplementedError
    
    print("mean result: {}".format(result))
    
    # save results
    save_path = variant["save_path"]
    save_path = os.path.join(
        save_path, 
        variant["filter_type"], 
        "{}_BT{}_Rally{}_Action{}_{}".format(variant["evaluator_type"], variant["reward_pref_model"], variant["reward_use_no_end_rally_pref"],variant["reward_loss_type"], variant["reward_action_pref_type"]), 
        "{}_win_return_{}".format(policy_type, variant["use_win_return"])
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if policy_type == "cql" or policy_type == "mlp_bc":
        save_value_dict = {
            "deterministic": [variant["deterministic"]],
            "policy_batch_size": [variant["policy_batch_size"]],
            "cql_target_action_gap": [variant["cql_target_action_gap"]],
            "cql_tune_init_log_alpha": [variant["cql_tune_init_log_alpha"]],
            "policy_hidden_size": [variant["policy_hidden_dims"]],
            "policy_n_layer": [variant["policy_n_hidden_layers"]],
            "policy_embed_size": [variant["policy_embedding_dim"]],
            "policy_seed": [variant["policy_seed"]],
            "mean_result": [result]
        }
    elif policy_type == "dt" or policy_type == "bc":
        save_value_dict = {
            "use_win_return": [variant["use_win_return"]],
            "deterministic": [variant["deterministic"]],
            "policy_batch_size": [variant["policy_batch_size"]],
            "policy_hidden_size": [variant["policy_hidden_dims"]],
            "policy_embed_size": [variant["policy_embedding_dim"]],
            "policy_n_layer": [variant["n_layer"]],
            "policy_n_head": [variant["n_head"]],
            "policy_seed": [variant["policy_seed"]],
            "mean_result": [result]
        }
    elif policy_type == "original":
        save_value_dict = {
            "mean_result": [result]
        }
    else:
        raise NotImplementedError
    
    save_values_to_csv(save_path, save_value_dict)


def print_result(variant):
    result_path = os.path.join(variant["save_path"], variant["filter_type"], "{}_pm{}_{}".format(variant["evaluator_type"], variant["reward_pref_model"], variant["reward_loss_type"]+variant["reward_action_pref_type"]), variant["policy_type"], "result.csv")
    result = pd.read_csv(result_path)
    max_metrics = result["mean_result"].max()

    print("optimal metrics: {}".format(max_metrics))
    print("optimal super parameters: ")
    for name in result.columns:
        if name == "mean_result":
            continue
        print("{:^20} | ".format(name), end="")
    print("\n")
    for name in result.columns:
        if name == "mean_result":
            continue
        print("{:^20} | ".format(result[result["mean_result"] == max_metrics][name].item()), end="")
    print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--print", type=int, default=0)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--save_path", type=str, default="evaluation_results")
    parser.add_argument("--eval_dataset", type=str, default="shuttle_both_agent")
    parser.add_argument("--filter_type", type=str, default="both") # "both", "win", or "loss"

    # evaluator parameters
    parser.add_argument("--evaluator_type", type=str, default="reward") # reward: preference-based reward model; classify: ensemble classify model
    # reward model
    parser.add_argument("--reward_pref_model", type=int, default=0)
    parser.add_argument("--reward_use_no_end_rally_pref", type=int, default=0) # 0: not use, 1: use no-end rally pref
    parser.add_argument("--reward_no_end_rally_pref_mode", type=int, default=0) # 0: Bradley-Tarry pref model, 1: average Bradley-Tarry pref model
    parser.add_argument("--reward_no_end_rally_pref_factor", type=float, default=1.0) 
    parser.add_argument("--reward_loss_type", type=int, default=0)
    parser.add_argument("--reward_ensemble_size", type=int, default=5)
    parser.add_argument("--reward_action_pref_type", type=int, default=1)
    parser.add_argument("--reward_action_pref_factor", type=float, default=1.0)
    parser.add_argument("--reward_learning_rate", type=float, default=1e-4)
    parser.add_argument("--reward_batch_size", type=int, default=256)
    parser.add_argument("--reward_shot_type_dim", type=int, default=15)
    parser.add_argument("--reward_location_type", type=int, default=0)
    parser.add_argument("--reward_location_num", type=int, default=16)
    parser.add_argument("--reward_location_dim", type=int, default=10)
    parser.add_argument("--reward_other_fea_dim", type=int, default=2)
    parser.add_argument("--reward_n_layer", type=int, default=3)
    parser.add_argument("--reward_hidden_dim", type=int, default=256)
    parser.add_argument("--reward_activation", type=str, default="tanh")
    parser.add_argument("--reward_model_dir", type=str, default="reward_models_save")
    parser.add_argument("--reward_max_epoch", type=int, default=500)

    # policy parameters
    parser.add_argument("--policy_type", type=str, default="cql") # "cql", "dt", "bc", or "mlp_bc", "original" for original data evaluation
    parser.add_argument("--policy_path", type=str, default="policy_models_save")
    parser.add_argument("--policy_seed", type=int, default=2024)
    parser.add_argument("--policy_batch_size", type=int, default=512)
    parser.add_argument("--policy_embedding_dim", type=int, default=64) # policy embedding dims for shot_type and coordinate
    parser.add_argument("--policy_activation", type=str, default="relu")
    parser.add_argument("--policy_orthogonal_init", type=int, default=1)
    parser.add_argument("--policy_embedding_coordinate", type=int, default=0)
    parser.add_argument("--deterministic", type=int, default=1) # deterministic policy or not
    # cql actor
    parser.add_argument("--policy_n_hidden_layers", type=int, default=3) # Number of hidden layers in actor networks
    parser.add_argument("--policy_hidden_dims", type=int, default=512) # hidden layer's dims for actor
    parser.add_argument("--cql_target_action_gap", type=float, default=5.0) # Action gap for CQL regularization
    parser.add_argument("--cql_tune_init_log_alpha", type=float, default=-2.0)
    # dt or dt-based bc
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument("--use_win_return", type=int, default=1)

    args = parser.parse_args()

    if args.print:
        print_result(variant=vars(args))
    else:
        evaluation(variant=vars(args))
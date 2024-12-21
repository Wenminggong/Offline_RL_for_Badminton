# -*- coding: utf-8 -*-
'''
@File    :   cql_tactics_main.py
@Time    :   2024/05/31 10:50:15
@Author  :   Mingjiang Liu 
@Version :   1.0
@Desc    :   main function of a popular offline RL algorithm CQL for badminton tactics generation, refer to CORL offline RL library (https://github.com/corl-team/CORL).
'''


import argparse
import pickle
import wandb
import os
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import random
TensorBatch = List[torch.Tensor]

from decision_transformer.utils import set_seed, save_values_to_csv
from offline_rl.models.critic import FullyConnectedQFunction
from offline_rl.models.actor import GaussianPolicy
from offline_rl.algorithms.cql import HybridCQL
from offline_rl.algorithms.bc import BC
from offline_rl.utils import convert_data_to_drl
from offline_rl.buffer.replay_buffer import ReplayBuffer
from data.preprocess_badminton_data import ACTIONS


def experiment(exp_prefix, variant):
    dataset = variant["dataset"]
    model_type = variant["model_type"]

    group_name = f'{exp_prefix}-{dataset}-{model_type}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}' 
    
    seed = variant.get('seed', 2024)
    # set random seed
    set_seed(seed)

    # load training data
    dataset_path = f'data/{dataset}_train.pkl'
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

    # convert data to {observations, actions, rewards, next_observations, terminals}
    dataset_data = convert_data_to_drl(trajectories)

    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        last_time_shot_type_dim,
        hit_xy_dim,
        player_location_xy_dim,
        opponent_location_xy_dim,
        shot_type_dim,
        landing_xy_dim,
        move_xy_dim,
        variant["buffer_size"],
        variant["device"],
    )

    replay_buffer.load_dataset(dataset_data)

    if variant["activation_function"] == 'relu':
        activation_function = nn.ReLU()
    elif variant["activation_function"] == 'tanh':
        activation_function = nn.Tanh()
    else:
        raise NotImplementedError
    
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
        variant["orthogonal_init"],
        variant["policy_n_hidden_layers"],
        variant["policy_hidden_dims"],
        variant["embedding_dim"],
        activation_function,
        embedding_coordinate=variant["embedding_coordinate"],
    ).to(variant["device"])
    #TODO: need weight decay?
    actor_optimizer = torch.optim.Adam(actor.parameters(), variant["policy_lr"])
    
    # cql
    if model_type == "cql":
        critic_1 = FullyConnectedQFunction(
            state_dim,
            action_dim,
            last_time_shot_type_dim,
            hit_xy_dim,
            player_location_xy_dim,
            opponent_location_xy_dim,
            shot_type_dim,
            landing_xy_dim,
            move_xy_dim,
            variant["orthogonal_init"],
            variant["q_n_hidden_layers"],
            variant["q_hidden_dims"],
            variant["embedding_dim"],
            activation_function,
            embedding_coordinate=variant["embedding_coordinate"],
        ).to(variant["device"])
        critic_2 = FullyConnectedQFunction(
            state_dim,
            action_dim,
            last_time_shot_type_dim,
            hit_xy_dim,
            player_location_xy_dim,
            opponent_location_xy_dim,
            shot_type_dim,
            landing_xy_dim,
            move_xy_dim,
            variant["orthogonal_init"],
            variant["q_n_hidden_layers"],
            variant["q_hidden_dims"],
            variant["embedding_dim"],
            activation_function,
            embedding_coordinate=variant["embedding_coordinate"],
        ).to(variant["device"])
        #TODO: need weight decay?
        critic_1_optimizer = torch.optim.Adam(list(critic_1.parameters()), variant["qf_lr"])
        critic_2_optimizer = torch.optim.Adam(list(critic_2.parameters()), variant["qf_lr"])


        kwargs = {
            "critic_1": critic_1,
            "critic_2": critic_2,
            "critic_1_optimizer": critic_1_optimizer,
            "critic_2_optimizer": critic_2_optimizer,
            "actor": actor,
            "actor_optimizer": actor_optimizer,
            "discount": variant["discount"],
            "soft_target_update_rate": variant["soft_target_update_rate"],
            "device": variant["device"],
            # CQL
            "target_entropy_d": 0.1 * np.log(shot_type_dim),
            "target_entropy_c": -move_xy_dim,
            "alpha_multiplier": variant["alpha_multiplier"],
            "use_automatic_entropy_tuning": variant["use_automatic_entropy_tuning"],
            "backup_entropy": variant["backup_entropy"],
            "policy_lr": variant["policy_lr"],
            "qf_lr": variant["qf_lr"],
            "sac_alpha_lr":  variant["sac_alpha_lr"],
            "cql_alpha_lr":  variant["cql_alpha_lr"],
            "bc_steps": variant["bc_steps"],
            "target_update_period": variant["target_update_period"],
            "cql_n_actions": variant["cql_n_actions"],
            "cql_importance_sample": variant["cql_importance_sample"],
            "cql_lagrange": variant["cql_lagrange"],
            "cql_target_action_gap": variant["cql_target_action_gap"],
            "cql_temp": variant["cql_temp"],
            "cql_alpha": variant["cql_alpha"],
            "cql_init_log_alpha": variant["cql_tune_init_log_alpha"],
            "cql_max_target_backup": variant["cql_max_target_backup"],
            "cql_clip_diff_min": variant["cql_clip_diff_min"],
            "cql_clip_diff_max": variant["cql_clip_diff_max"],
        }

        trainer = HybridCQL(**kwargs)

        if variant["load_model"] != "":
            policy_file = Path(variant["load_model"])
            trainer.load_state_dict(torch.load(policy_file))
    elif model_type == "mlp_bc":
        trainer = BC(
            actor=actor,
            actor_optimizer=actor_optimizer,
            policy_lr=variant["policy_lr"],
            eval_freq=variant["eval_freq"],
            device=variant["device"],
        )
        
        # load eval data
        eval_dataset_path = f'data/{dataset}_val.pkl'
        with open(eval_dataset_path, "rb") as f:
            # trajectories = [{}, {}, ...]
            eval_trajectories = pickle.load(f)
        eval_dataset_data = convert_data_to_drl(eval_trajectories)

        eval_replay_buffer = ReplayBuffer(
            state_dim,
            action_dim,
            last_time_shot_type_dim,
            hit_xy_dim,
            player_location_xy_dim,
            opponent_location_xy_dim,
            shot_type_dim,
            landing_xy_dim,
            move_xy_dim,
            variant["buffer_size"] // 2,
            variant["device"],
        )

        eval_replay_buffer.load_dataset(eval_dataset_data)
    else:
        raise NotImplementedError



    print("---------------------------------------")
    print(f"Training {model_type}, Seed: {seed}")
    print("---------------------------------------")
    
    if variant["log_to_wandb"]:
        # initial wandb
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='decision-transformer',
            config=variant
        )

    min_evaluation_loss = 1e6
    patience = 10
    count = 0
    for t in range(int(variant["max_timesteps"])):
        print("timestep: {}".format(t))
        batch = replay_buffer.sample(variant["batch_size"])
        if model_type == "cql":
            log_dict = trainer.train(batch)
        elif model_type == "mlp_bc":
            eval_batch = eval_replay_buffer.sample_all()
            log_dict = trainer.train(batch, eval_batch)
        else:
            raise NotImplementedError
        if variant["log_to_wandb"]:
            wandb.log(log_dict)
        if model_type == "mlp_bc" and (t+1) % variant["eval_freq"] == 0:
            cur_evaluation_loss = log_dict["evaluation/loss_mean"]
            if cur_evaluation_loss < min_evaluation_loss:
                min_evaluation_loss = cur_evaluation_loss
                count = 0
            else:
                count += 1
                if count >= patience:
                    break

    if variant["checkpoints_path"]:
        model_path = os.path.join(variant["checkpoints_path"], f"{dataset}_{model_type}")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        # save super-parameters and final loss
        if model_type == "cql":
            save_value_dict = {
                "batch_size": [variant["batch_size"]],
                "cql_target_action_gap": [variant["cql_target_action_gap"]],
                "cql_tune_init_log_alpha": [variant["cql_tune_init_log_alpha"]],
                "hidden_size": [variant["q_hidden_dims"]],
                "n_layer": [variant["q_n_hidden_layers"]],
                "embed_size": [variant["embedding_dim"]],
                "seed": [seed],
                "final_train_policy_loss": [log_dict["loss/policy_loss"]],
                "final_train_q1_loss": [log_dict["loss/qf1_loss"]],
                "final_train_q2_loss": [log_dict["loss/qf2_loss"]]
            }
            torch.save(
                trainer.actor.to("cpu").state_dict(),
                os.path.join(model_path, "{}_g{}_a{}_b{}_h{}_n{}_e{}_s{}.pth".format(model_type, variant["cql_target_action_gap"], variant["cql_tune_init_log_alpha"], variant["batch_size"], variant["q_hidden_dims"], variant["q_n_hidden_layers"], variant["embedding_dim"], seed))
        )
        elif model_type == "mlp_bc":
            save_value_dict = {
                "batch_size": [variant["batch_size"]],
                "hidden_size": [variant["q_hidden_dims"]],
                "n_layer": [variant["q_n_hidden_layers"]],
                "embed_size": [variant["embedding_dim"]],
                "seed": [seed],
                "final_train_loss": [log_dict["training/loss_mean"]],
                "final_train_type_ce": [log_dict["training/shot_type_mean"]],
                "final_train_landing_lh": [log_dict["training/landing_xy_mean"]],
                "final_train_move_lh": [log_dict["training/move_xy_mean"]],
                "final_eval_loss": [log_dict["evaluation/loss_mean"]],
                "final_eval_type_ce": [log_dict["evaluation/shot_type_mean"]],
                "final_eval_landing_lh": [log_dict["evaluation/landing_xy_mean"]],
                "final_eval_move_lh": [log_dict["evaluation/move_xy_mean"]],
            }
            torch.save(
                trainer.actor.to("cpu").state_dict(),
                os.path.join(model_path, "{}_b{}_h{}_n{}_e{}_s{}.pth".format(model_type, variant["batch_size"], variant["policy_hidden_dims"], variant["policy_n_hidden_layers"], variant["embedding_dim"], seed))
            )
        else:
            raise NotImplementedError

        save_values_to_csv(model_path, save_value_dict)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--eval_freq", type=int, default=500) # How often (time steps) we evaluate 
    parser.add_argument("--load_model", type=str, default="") # Model load file name, "": doesn't load
    parser.add_argument("--buffer_size", type=int, default=100000) # replay buffer size
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--alpha_multiplier", type=float, default=0.3) # Multiplier for auto entropy alpha in SAC loss or constant entropy alpha in SAC loss
    parser.add_argument("--use_automatic_entropy_tuning", type=int, default=1) # Tune entropy in SAC
    parser.add_argument("--backup_entropy", type=int, default=1) # Use backup entropy or not, 0: not entropy backup; 1: entropy backup in Q-update
    parser.add_argument("--policy_lr", type=float, default=5e-7) # actor learning rate
    parser.add_argument("--qf_lr", type=float, default=1e-6) # critic learning rate, generally, qf_lr should > policy_lr
    parser.add_argument("--sac_alpha_lr", type=float, default=1e-5)
    parser.add_argument("--cql_alpha_lr", type=float, default=3e-5)
    parser.add_argument("--soft_target_update_rate", type=float, default=5e-3)
    parser.add_argument("--target_update_period", type=int, default=1) # Frequency of target nets updates
    parser.add_argument("--cql_n_actions", type=int, default=10) # Number of sampled actions for computing cql regularization item
    parser.add_argument("--cql_importance_sample", type=int, default=0) # Use importance sampling or not, 0: not, CQL(H);1: yes, CQL(p)
    parser.add_argument("--cql_lagrange", type=int, default=1) # Use Lagrange version of CQL, 1: auto tune cql-alpha; 0: constant cql-alpha
    parser.add_argument("--cql_target_action_gap", type=float, default=5.0) # Action gap for CQL regularization, value?
    parser.add_argument("--cql_temp", type=float, default=1.0) # CQL temperature
    parser.add_argument("--cql_tune_init_log_alpha", type=float, default=-2.0)
    parser.add_argument("--cql_alpha", type=float, default=1.0) # constant cql alpha 
    parser.add_argument("--cql_max_target_backup", type=int, default=0) # Use max target backup or not, 0: not; 1: max Q
    parser.add_argument("--cql_clip_diff_min", type=float, default=-10.0) # Q-function lower loss clipping
    parser.add_argument("--cql_clip_diff_max", type=float, default=10.0) # Q-function upper loss clipping
    parser.add_argument("--orthogonal_init", type=int, default=1) # Orthogonal initialization for neural network
    parser.add_argument("--q_n_hidden_layers", type=int, default=3) # Number of hidden layers in Q networks
    parser.add_argument("--q_hidden_dims", type=int, default=256) # hidden layer's dims for Q
    parser.add_argument("--policy_n_hidden_layers", type=int, default=3) # Number of hidden layers in actor networks
    parser.add_argument("--policy_hidden_dims", type=int, default=256) # hidden layer's dims for actor
    parser.add_argument("--embedding_dim", type=int, default=32) # embedding dims for shot_type and coordinate
    parser.add_argument("--embedding_coordinate", type=int, default=0) # embedding location coordinate or not
    parser.add_argument("--activation_function", type=str, default="relu") # activation function for neural network
    parser.add_argument("--bc_steps", type=int, default=0) # Number of BC steps at start for actor
    parser.add_argument("--policy_log_std_multiplier", type=float, default=1.0) # Stochastic policy std multiplier TODO: DELETE
    parser.add_argument("--log_to_wandb", type=int, default=1)
    parser.add_argument("--checkpoints_path", type=str, default="trained_models")
    parser.add_argument("--dataset", type=str, default="shuttle_both_agent")
    parser.add_argument("--eval_dataset", type=str, default="shuttle_both_agent")
    parser.add_argument("--max_timesteps", type=int, default=100000)
    parser.add_argument("--model_type", type=str, default="cql") # "cql" or "mlp_bc"

    args = parser.parse_args()
    experiment('BadmintonTacticsGenerationFinal', variant=vars(args))
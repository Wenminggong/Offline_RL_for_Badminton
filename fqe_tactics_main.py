# -*- coding: utf-8 -*-
'''
@File    :   ope_tactics_main.py
@Time    :   2024/06/09 17:11:14
@Author  :   Mingjiang Liu 
@Version :   1.0
@Desc    :   offline policy evaluation for badminton tactics generation
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import argparse
import random
import wandb
import os

from decision_transformer.utils import set_seed
from decision_transformer.models.dt_for_tactics_generation import DecisionTransformerTactics
from decision_transformer.models.dt_based_bc_for_tactics_generation import DecisionTransformerBCTactics
from offline_rl.models.critic import FullyConnectedQFunction
from offline_rl.models.actor import GaussianPolicy
from offline_rl.utils import convert_data_to_drl
from offline_rl.buffer.replay_buffer import ReplayBuffer
from offline_rl.ope.fqe import FQE
from decision_transformer.utils import get_batch_data_from_shuttleset, save_values_to_csv
from data.preprocess_badminton_data import ACTIONS


def ope_main(exp_prefix, variant):
    ori_dataset_path = variant["eval_dataset"]
    policy_type = variant["policy_type"]
    group_name = f'{exp_prefix}-{ori_dataset_path}-{policy_type}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    # set random seed
    seed = variant.get("seed", 2024)
    set_seed(seed)

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

    if variant["activation_function"] == 'relu':
        activation_function = nn.ReLU()
    elif variant["activation_function"] == 'tanh':
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
            variant["orthogonal_init"],
            variant["policy_n_hidden_layers"],
            variant["policy_hidden_dims"],
            variant["policy_embedding_dim"],
            activation_function,
            embedding_coordinate=variant["embedding_coordinate"],
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
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
            use_player_location=1,
            embed_coordinate=variant["embedding_coordinate"]
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
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
            use_player_location=1,
            embed_coordinate=variant["embedding_coordinate"],
        )
        policy_path = os.path.join(variant["policy_path"], f"{ori_dataset_path}_{policy_type}", "{}_b{}_h{}_e{}_nl{}_nh{}_s{}.pth".format(policy_type, variant["policy_batch_size"], variant["policy_hidden_dims"], variant["policy_embedding_dim"], variant["n_layer"], variant["n_head"], variant["policy_seed"]))
    else:
        raise NotImplementedError
    actor.load_state_dict(torch.load(policy_path))
    actor.to(variant["device"])

    
    actor.eval()
    # cql
    if policy_type == "cql" or policy_type == "mlp_bc":
        # convert data to {observations, actions, rewards, next_observations, terminals}
        dataset = convert_data_to_drl(trajectories)
        next_last_time_shot_type = torch.from_numpy(dataset["next_last_time_shot_type"])
        next_last_time_shot_type = F.one_hot(next_last_time_shot_type, num_classes=last_time_shot_type_dim) # [batch_size, 10]
        next_hit_xy = torch.from_numpy(dataset["next_hit_xy"])
        next_player_location_xy = torch.from_numpy(dataset["next_player_location_xy"])
        next_opponent_location_xy = torch.from_numpy(dataset["next_opponent_location_xy"])

        # [batch_size, n] - ndarray
        next_shot_type, next_shot_probs, next_landing_xy, next_move_xy = actor.act(
            next_last_time_shot_type,
            next_hit_xy,
            next_player_location_xy,
            next_opponent_location_xy,
            deterministic=variant["deterministic"],
            device=variant["device"]
        )
        # next_shot_type = F.one_hot(torch.from_numpy(next_shot_type).squeeze(dim=-1), num_classes=10) # [batch_size, shot_type_dim] - tensor
        # next_shot_type = next_shot_type.squeeze().numpy()
        dataset["next_shot_type"] = next_shot_type.reshape(-1) # [batch_size]
        dataset["next_shot_probs"] = next_shot_probs # [batch_size, 10]
        dataset["next_landing_xy"] = next_landing_xy
        dataset["next_move_xy"] = next_move_xy

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
        next_action=True,
    )

    replay_buffer.load_dataset(dataset)

    critic = FullyConnectedQFunction(
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
        embedding_coordinate=variant["embedding_coordinate"]
    )
    if variant["q_model_path"] != "":
        if policy_type == "dt" or policy_type == "bc":
            q_model_path = os.path.join(
                variant["checkpoints_path"], 
                f"fqe_{ori_dataset_path}_{policy_type}",
                "fqe_fn{}_fe{}_b{}_h{}_e{}_nl{}_nh{}_s{}_critic.pth".format(variant["q_n_hidden_layers"], variant["embedding_dim"], variant["policy_batch_size"], variant["policy_hidden_dims"], variant["policy_embedding_dim"], variant["n_layer"], variant["n_head"], variant["policy_seed"])
                )
        elif policy_type == "cql" or policy_type == "mlp_bc":
            q_model_path = os.path.join(
                variant["checkpoints_path"], 
                f"fqe_{ori_dataset_path}_{policy_type}",
                "fqe_fn{}_fe{}_g{}_a{}_b{}_h{}_n{}_e{}_s{}_critic.pth".format(variant["q_n_hidden_layers"], variant["embedding_dim"], variant["cql_target_action_gap"], variant["cql_tune_init_log_alpha"], variant["policy_batch_size"], variant["policy_hidden_dims"], variant["policy_n_hidden_layers"], variant["policy_embedding_dim"], variant["policy_seed"])
                )
        else:
            raise NotImplementedError
        critic.load_state_dict(torch.load(q_model_path))
    critic.to(variant["device"])
    critic_optimizer = torch.optim.Adam(list(critic.parameters()), variant["qf_lr"])

    trainer = FQE(
        critic,
        critic_optimizer,
        variant["discount"],
        variant["target_update_period"],
        variant["soft_target_update_rate"],
        deterministic=variant["deterministic"],
        device=variant["device"],
    )

    if variant["log_to_wandb"]:
        # initial wandb
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='decision-transformer',
            config=variant
        )

    for t in range(int(variant["max_timesteps"])):
        print("timestep: {}".format(t))
        batch = replay_buffer.sample(variant["batch_size"])
        log_dict = trainer.train(batch)
        if variant["log_to_wandb"]:
            wandb.log(log_dict)
        # if variant["checkpoints_path"] and (t+1) % (int(variant["max_timesteps"]) // 5) == 0:
        #     print("----------------- save ------------")
        #     torch.save(
        #         trainer.critic.state_dict(),
        #         os.path.join(variant["checkpoints_path"], f"checkpoint_{t}.pth"),
        #     )
    
    with torch.no_grad():
        # evaluate Q(s, \pi(s)) rather than Q(s,a)
        last_time_shot_type, hit_xy, player_location_xy, opponent_location_xy, shot_type, landing_xy, move_xy, next_last_time_shot_type, next_hit_xy, next_player_location_xy, next_opponent_location_xy, next_shot_type, next_landing_xy, next_move_xy, _, _, _ = replay_buffer.sample_all()
        q_predicted = critic(next_last_time_shot_type, next_hit_xy, next_player_location_xy, next_opponent_location_xy, next_shot_type, next_landing_xy, next_move_xy)
        mean_q = q_predicted[next_shot_type > 0].mean().item()
        print("mean q: {}".format(mean_q))

    model_path = os.path.join(variant["checkpoints_path"], "fqe_{}_{}_w{}".format(ori_dataset_path, policy_type, variant["use_win_return"]))
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if policy_type == "cql" or policy_type == "mlp_bc":
        torch.save(
            trainer.critic.to("cpu").state_dict(),
            os.path.join(model_path, "fqe_fn{}_fe{}_g{}_a{}_b{}_h{}_n{}_e{}_s{}_critic.pth".format(variant["q_n_hidden_layers"], variant["embedding_dim"], variant["cql_target_action_gap"], variant["cql_tune_init_log_alpha"], variant["policy_batch_size"], variant["policy_hidden_dims"], variant["policy_n_hidden_layers"], variant["policy_embedding_dim"], variant["policy_seed"]))
        )
        save_value_dict = {
            "fqe_seed": seed,
            "fqe_n_hidden_layers": variant["q_n_hidden_layers"],
            "fqe_embed_sizes": variant["embedding_dim"],
            "batch_size": [variant["policy_batch_size"]],
            "cql_target_action_gap": [variant["cql_target_action_gap"]],
            "cql_tune_init_log_alpha": [variant["cql_tune_init_log_alpha"]],
            "hidden_size": [variant["policy_hidden_dims"]],
            "n_layer": [variant["policy_n_hidden_layers"]],
            "embed_size": [variant["policy_embedding_dim"]],
            "policy_seed": [variant["policy_seed"]],
            "mean_q": mean_q
        }
    elif policy_type == "dt" or policy_type == "bc":
        torch.save(
            trainer.critic.to("cpu").state_dict(),
            os.path.join(model_path, "fqe_fn{}_fe{}_b{}_h{}_e{}_nl{}_nh{}_s{}_critic.pth".format(variant["q_n_hidden_layers"], variant["embedding_dim"], variant["policy_batch_size"], variant["policy_hidden_dims"], variant["policy_embedding_dim"], variant["n_layer"], variant["n_head"], variant["policy_seed"]))
        )
        save_value_dict = {
            "fqe_seed": seed,
            "fqe_n_hidden_layers": variant["q_n_hidden_layers"],
            "fqe_embed_sizes": variant["embedding_dim"],
            "batch_size": [variant["policy_batch_size"]],
            "hidden_size": [variant["policy_hidden_dims"]],
            "embed_size": [variant["policy_embedding_dim"]],
            "n_layer": [variant["n_layer"]],
            "n_head": [variant["n_head"]],
            "policy_seed": [variant["policy_seed"]],
            "mean_q": mean_q
        }
    else:
        raise NotImplementedError
    
    save_values_to_csv(model_path, save_value_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--q_model_path", type=str, default="")
    parser.add_argument("--buffer_size", type=int, default=100000) # replay buffer size
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--qf_lr", type=float, default=1e-6) # critic learning rate
    parser.add_argument("--soft_target_update_rate", type=float, default=5e-3)
    parser.add_argument("--target_update_period", type=int, default=1) # Frequency of target nets updates
    parser.add_argument("--orthogonal_init", type=int, default=1) # Orthogonal initialization for neural network
    parser.add_argument("--q_n_hidden_layers", type=int, default=3) # Number of hidden layers in Q networks
    parser.add_argument("--q_hidden_dims", type=int, default=256) # hidden layer's dims for Q
    parser.add_argument("--embedding_dim", type=int, default=32) # q embedding dims for shot_type and coordinate
    parser.add_argument("--embedding_coordinate", type=int, default=0) # embedding location coordinate or not
    parser.add_argument("--activation_function", type=str, default="relu") # activation function for neural network
    parser.add_argument("--log_to_wandb", type=int, default=1)
    parser.add_argument("--checkpoints_path", type=str, default="fqe_models_save")
    parser.add_argument("--eval_dataset", type=str, default="shuttle_both_agent")
    parser.add_argument("--max_timesteps", type=int, default=10000)
    parser.add_argument("--policy_type", type=str, default="cql") # "cql", "dt", "bc", or "mlp_bc"
    parser.add_argument("--policy_path", type=str, default="policy_models_save")
    parser.add_argument("--policy_seed", type=int, default=2024)
    parser.add_argument("--policy_batch_size", type=int, default=512)
    parser.add_argument("--policy_embedding_dim", type=int, default=64) # policy embedding dims for shot_type and coordinate

    # policy parameters
    parser.add_argument("--deterministic", type=int, default=1) # deterministic policy or not
    # cql actor
    parser.add_argument("--policy_n_hidden_layers", type=int, default=3) # Number of hidden layers in actor networks
    parser.add_argument("--policy_hidden_dims", type=int, default=512) # hidden layer's dims for actor
    parser.add_argument("--cql_target_action_gap", type=float, default=5.0) # Action gap for CQL regularization, value?
    parser.add_argument("--cql_tune_init_log_alpha", type=float, default=-2.0)
    # dt or dt-based bc
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument("--use_win_return", type=int, default=1)

    args = parser.parse_args()
    ope_main('BadmintonTacticsFQEFinal', variant=vars(args))
# -*- coding: utf-8 -*-
# Copyright (c) 2021 Decision Transformer (Decision Transformer: Reinforcement Learning via Sequence Modeling) Authors (https://arxiv.org/abs/2106.01345)
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
@File    :   dt_tactics_main.py
@Time    :   2024/05/16 17:39:25
@Author  :   Mingjiang Liu 
@Version :   1.0
@Desc    :   Decision Transformer for badminton tactics generation, i.e., utilize DT to generate shot_type, landing_xy , and move_xy
'''


import numpy as np
import torch
import torch.nn as nn
import wandb

import argparse
import pickle
import random
import sys
import os

from decision_transformer.models.dt_for_tactics_generation import DecisionTransformerTactics
from decision_transformer.models.dt_based_bc_for_tactics_generation import DecisionTransformerBCTactics
from decision_transformer.training.seq_trainer_tactics_generation import SequenceTrainerTactics
from decision_transformer.evaluation.evaluator import DecisionTransformerTacticsEvaluator
from decision_transformer.utils import set_seed, get_batch_data_from_shuttleset, save_values_to_csv
from data.preprocess_badminton_data import ACTIONS


def experiment(
        exp_prefix,
        variant,
):

    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)

    dataset = variant['dataset']
    model_type = variant['model_type']
    group_name = f'{exp_prefix}-{dataset}-{model_type}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    seed = variant.get('seed', 2024)
    # set random seed
    set_seed(seed)

    # load dataset
    dataset_path = f'data/{dataset}_train.pkl'
    with open(dataset_path, 'rb') as f:
        # trajectories: [{key:array}, {key:array}, ...]
        trajectories = pickle.load(f)
    
    last_time_shot_type_dim = len(ACTIONS)
    hit_xy_dim = trajectories[0]["hit_xy"].shape[1]
    player_location_xy_dim = trajectories[0]["player_location_xy"].shape[1]
    opponent_location_xy_dim = trajectories[0]["opponent_location_xy"].shape[1]
    shot_type_dim = len(ACTIONS)
    landing_xy_dim = trajectories[0]["landing_xy"].shape[1]
    move_xy_dim = trajectories[0]["move_xy"].shape[1]
    use_player_location = variant.get('use_player_location', 1)
    
    if use_player_location:
        state_dim = last_time_shot_type_dim + hit_xy_dim + player_location_xy_dim + opponent_location_xy_dim
    else:
        state_dim = last_time_shot_type_dim + hit_xy_dim
    act_dim = shot_type_dim + landing_xy_dim + move_xy_dim

    traj_lens, returns = [], []
    max_ep_len = 1024
    mode = variant.get('mode', 'normal')
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['reward'][-1] = path['reward'].sum()
            path['reward'][:-1] = 0.
        max_ep_len = max(max_ep_len, len(path["reward"]))
        traj_lens.append(len(path["reward"]))
        returns.append(path["reward"].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns) # M-dims, M = episode_num

    num_timesteps = sum(traj_lens) # total-timesteps = N

    print('=' * 50)
    print(f'Starting new experiment: {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    batch_size = variant['batch_size']
    pct_traj = variant.get('pct_traj', 1.)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=256):
        # get_batch for Badminton trajectories
        # batch_inds: sample batch_size indexs for trajectory selection
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        batch_trajectories = []
        for i in range(batch_size):
            batch_trajectories.append(trajectories[int(sorted_inds[batch_inds[i]])])
        
        batch_data = get_batch_data_from_shuttleset(batch_trajectories, device) # batch_data: dict={key, value}
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
        
        return last_time_shot_type, hit_xy, player_location_xy, opponent_location_xy, shot_type, landing_xy, move_xy, reward, rtg, timesteps, mask, move_mask
    
    def evaluation():
        eval_dataset = variant.get('eval_dataset', 'shuttle_both_agent')
        eval_dataset_path = f'data/{eval_dataset}_val.pkl'
        eval_sample_num = variant.get('eval_sample_num', 10)
        evaluator = DecisionTransformerTacticsEvaluator(eval_dataset_path, batch_size, eval_sample_num, device)
        def fn(model):
            evaluator.action_generation(model)
            ce_loss, same_type_rate, landing_lh_loss, move_lh_loss, landing_mae_loss, landing_mse_loss, move_mae_loss, move_mse_loss  = evaluator.get_eval_result(shot_type_dim, landing_xy_dim, move_xy_dim)
            return {
                "loss_mean": ce_loss + landing_lh_loss + move_lh_loss,
                "type_ce": ce_loss,
                "same_type_rate": same_type_rate,
                "landing_lh": landing_lh_loss,
                "move_lh": move_lh_loss,
                "landing_xy_mae": landing_mae_loss,
                "landing_xy_mse": landing_mse_loss,
                "move_xy_mae": move_mae_loss,
                "move_xy_mse": move_mse_loss,
            }
        return fn

    if model_type == 'dt':
        model = DecisionTransformerTactics(
            state_dim=state_dim,
            act_dim=act_dim,
            last_time_shot_type_dim=last_time_shot_type_dim,
            hit_xy_dim=hit_xy_dim,
            player_location_xy_dim=player_location_xy_dim,
            opponent_location_xy_dim=opponent_location_xy_dim,
            shot_type_dim=shot_type_dim,
            landing_xy_dim=landing_xy_dim,
            move_xy_dim=move_xy_dim,
            max_ep_len=max_ep_len+32,
            hidden_size=variant['hidden_dim'],
            embed_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['hidden_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
            use_player_location=use_player_location,
            embed_coordinate=variant["embed_coordinate"],
        )
    elif model_type == "bc":
        model = DecisionTransformerBCTactics(
            state_dim=state_dim,
            act_dim=act_dim,
            last_time_shot_type_dim=last_time_shot_type_dim,
            hit_xy_dim=hit_xy_dim,
            player_location_xy_dim=player_location_xy_dim,
            opponent_location_xy_dim=opponent_location_xy_dim,
            shot_type_dim=shot_type_dim,
            landing_xy_dim=landing_xy_dim,
            move_xy_dim=move_xy_dim,
            max_ep_len=max_ep_len+32,
            hidden_size=variant['hidden_dim'],
            embed_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['hidden_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
            use_player_location=use_player_location,
            embed_coordinate=variant["embed_coordinate"],
        )
    else:
        raise NotImplementedError

    model = model.to(device=device)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    if model_type == 'dt' or model_type == 'bc':
        def tactics_loss_fn(shot_type, landing_xy, move_xy, shot_preds, landing_distribution, move_distribution, attention_mask,  move_mask):
            # shot_type: [batch_size, max_len], shot_preds: [batch_size, max_len, 10]
            shot_type_target = shot_type.reshape(-1)[attention_mask.reshape(-1) > 0]
            shot_preds = shot_preds.reshape(-1, shot_type_dim)[attention_mask.reshape(-1) > 0]
            
            # cross-entropy loss
            type_loss = nn.CrossEntropyLoss()
            type_loss_value = type_loss(shot_preds, shot_type_target)

            # -log_likelihood landing loss
            landing_loss = landing_distribution.log_prob(landing_xy)
            landing_loss_value = - landing_loss[attention_mask > 0].mean()

            # -log_likelihood move loss
            move_loss = move_distribution.log_prob(move_xy)
            move_loss_value = - move_loss[move_mask > 0].mean()

            return type_loss_value + landing_loss_value + move_loss_value, type_loss_value, landing_loss_value, move_loss_value


        trainer = SequenceTrainerTactics(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=tactics_loss_fn,
            eval_fns=[evaluation()]
        )
    else:
        raise NotImplementedError

    if log_to_wandb:
        # initial wandb
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='decision-transformer',
            config=variant
        )
        # wandb.watch(model)  # wandb has some bug

    min_evaluation_loss = 1e6
    patience = 10
    count = 0
    # cur_evaluation_loss_list = []
    for iter in range(variant['max_iters']):
        outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True)
        if log_to_wandb:
            wandb.log(outputs)
        if model_type == 'bc' or model_type == 'dt':
            cur_evaluation_loss = outputs["evaluation/loss_mean"]
            if cur_evaluation_loss < min_evaluation_loss:
                min_evaluation_loss = cur_evaluation_loss
                count = 0
            else:
                count += 1
                if count >= patience:
                    break

    if device == "cuda":
        model.to("cpu")
    model_path = os.path.join(variant['model_save_path'], f"{dataset}_{model_type}")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(model.state_dict(), 
            os.path.join(model_path, "{}_b{}_h{}_e{}_nl{}_nh{}_s{}.pth".format(model_type, variant["batch_size"], variant["hidden_dim"], variant["embed_dim"], variant["n_layer"], variant["n_head"], seed)))
    # wandb.save(variant['model_save_path']+".pth")

    # save super-parameters and final loss
    save_value_dict = {
        "batch_size": [variant["batch_size"]],
        "hidden_size": [variant["hidden_dim"]],
        "embed_size": [variant["embed_dim"]],
        "n_layer": [variant["n_layer"]],
        "n_head": [variant["n_head"]],
        "seed": [seed],
        "final_train_loss": [outputs["training/train_loss_mean"]],
        "final_train_type_ce": [outputs["training/type_loss_mean"]],
        "final_train_landing_lh": [outputs["training/landing_loss_mean"]],
        "final_train_move_lh": [outputs["training/move_loss_mean"]],
        "final_eval_loss": [outputs["evaluation/loss_mean"]],
        "final_eval_type_ce": [outputs["evaluation/type_ce"]],
        "final_eval_landing_lh": [outputs["evaluation/landing_lh"]],
        "final_eval_move_lh": [outputs["evaluation/move_lh"]],
        "final_eval_landing_mae": [outputs["evaluation/landing_xy_mae"]],
        "final_eval_landing_mse": [outputs["evaluation/landing_xy_mse"]],
        "final_eval_move_mae": [outputs["evaluation/move_xy_mae"]],
        "final_eval_move_mse": [outputs["evaluation/move_xy_mse"]]
    }

    save_values_to_csv(model_path, save_value_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='shuttle_both_agent') # training dataset
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse reward 
    parser.add_argument('--pct_traj', type=float, default=1.) # use top x% data to train model 
    parser.add_argument('--batch_size', type=int, default=128) # batch_size samples， 1 sample = 1 sequence
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for DT-based behaviour cloning
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument("--embed_coordinate", type=int, default=0) # embedding location coordinate or not
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--n_head', type=int, default=2)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-3)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--max_iters', type=int, default=100)
    parser.add_argument('--num_steps_per_iter', type=int, default=100)
    parser.add_argument('--eval_dataset', type=str, default='shuttle_both_agent') # evaluating dataset
    parser.add_argument('--eval_sample_num', type=int, default=10) # sample num coordinates to evaluate landing_xy and move_xy prediction
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=int, default=1)
    parser.add_argument('--model_save_path', type=str, default='trained_models')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--use_player_location', type=int, default=1) # use player_location_xy, opponent_location_xy as input features or not, 1:use; 0:not
    
    args = parser.parse_args()

    experiment('BadmintonTacticsGenerationFinal', variant=vars(args))

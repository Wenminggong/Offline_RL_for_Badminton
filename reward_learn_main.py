# -*- coding: utf-8 -*-
'''
@File    :   reward_learn_main.py
@Time    :   2024/08/14 20:18:26
@Author  :   Mingjiang Liu 
@Version :   1.0
@Desc    :   main function for preference-based reward learning
'''


import argparse
import pickle
import random
import wandb
import os

from prediction.models.reward_model import RewardModel
from decision_transformer.utils import set_seed, save_values_to_csv


def experiment(exp_prefix, variant):
    log_to_wandb = variant.get('log_to_wandb', False)
    dataset = variant['dataset']
    group_name = f'{exp_prefix}-{dataset}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    seed = variant.get('seed', 2024)
    # set random seed
    set_seed(seed)

    # load dataset
    train_dataset_path = f'data/{dataset}_train.pkl'
    with open(train_dataset_path, 'rb') as f:
        # trajectories: [{key:array}, {key:array}, ...]
        train_trajectories = pickle.load(f)
    
    val_dataset = variant["eval_dataset"]
    val_dataset_path = f'data/{val_dataset}_val.pkl'
    with open(val_dataset_path, 'rb') as f:
        # trajectories: [{key:array}, {key:array}, ...]
        val_trajectories = pickle.load(f)

    reward_model = RewardModel(
        pref_model=variant["pref_model"],
        no_end_rally_pref=variant["use_no_end_rally_pref"],
        no_end_rally_pref_mode=variant["no_end_rally_pref_mode"],
        no_end_rally_pref_factor=variant["no_end_rally_pref_factor"],
        loss_type=variant["loss_type"],
        ensemble_size=variant["ensemble_size"],
        action_pref_type=variant["action_pref_type"],
        lr=variant["learning_rate"],
        batch_size=variant["batch_size"],
        action_pref_factor=variant["action_pref_factor"],
        evaluate_flag=1,
        shot_type_num=variant["shot_type_num"],
        shot_type_dim=variant["shot_type_dim"],
        location_type=variant["location_type"],
        location_num=variant["location_num"],
        location_dim=variant["location_dim"],
        other_fea_dim=variant["other_fea_dim"],
        n_layer=variant["n_layer"],
        hidden_dim=variant["hidden_dim"],
        activation=variant["activation_function"],
        device=variant["device"],
    )

    reward_model.construct_train_buffer(train_trajectories)
    reward_model.construct_val_buffer(val_trajectories)

    if log_to_wandb:
        # initial wandb
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='decision-transformer',
            config=variant
        )

    min_evaluation_acc = -1e6
    patience = 10
    count = 0
    for epoch in range(variant["max_epoch"]):
        outputs = reward_model.train_reward(epoch, print_logs=True)
        if log_to_wandb:
            wandb.log(outputs)

        cur_evaluation_acc = outputs["eval/rally_acc"] + outputs["eval/no_end_rally_acc"] if variant["use_no_end_rally_pref"] else outputs["eval/rally_acc"]
        if cur_evaluation_acc > min_evaluation_acc:
            min_evaluation_acc = cur_evaluation_acc
            count = 0
        else:
            count += 1
            if count >= patience:
                break
    
    model_path = variant["model_save_path"]
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_name = "b{}_lr{}_t{}_sd{}_lt{}_ld{}_od{}_h{}_nl{}_pm{}_une{}_nem{}_nef{}_lt{}_en{}_at{}_af{}_s{}".format(
        variant["batch_size"],
        variant["learning_rate"], 
        variant["max_epoch"],
        variant["shot_type_dim"],
        variant["location_type"],
        variant["location_dim"],
        variant["other_fea_dim"],
        variant["hidden_dim"], 
        variant["n_layer"],
        variant["pref_model"],
        variant["use_no_end_rally_pref"],
        variant["no_end_rally_pref_mode"],
        variant["no_end_rally_pref_factor"],
        variant["loss_type"],
        variant["ensemble_size"],
        variant["action_pref_type"],
        variant["action_pref_factor"], 
        seed
    )
    reward_model.save(model_path, model_name)

    # save super-parameters and final result
    save_value_dict = {
        "batch_size": [variant["batch_size"]],
        "learning_rate": [variant["learning_rate"]], 
        "training_epoch": [variant["max_epoch"]],
        "shot_type_dim": [variant["shot_type_dim"]],
        "location_type": [variant["location_type"]],
        "location_dim": [variant["location_dim"]],
        "other_fea_dim": [variant["other_fea_dim"]],
        "hidden_size": [variant["hidden_dim"]],
        "n_layer": [variant["n_layer"]],
        "pref_model": [variant["pref_model"]],
        "use_no_end_rally_pref": [variant["use_no_end_rally_pref"]],
        "no_end_rally_pref_mode": [variant["no_end_rally_pref_mode"]],
        "no_end_rally_pref_factor": [variant["no_end_rally_pref_factor"]],
        "loss_type": [variant["loss_type"]],
        "ensemble_size": [variant["ensemble_size"]],
        "action_pref_type": [variant["action_pref_type"]],
        "action_pref_factor": [variant["action_pref_factor"]],
        "seed": [seed],
        "rally_acc": [outputs["eval/rally_acc"]],
        "no_end_rally_acc": [outputs["eval/no_end_rally_acc"]],
        "action_acc_end": [outputs["eval/action_acc_end"]],
        "action_acc_player": [outputs["eval/action_acc_player"]],
    }

    save_values_to_csv(model_path, save_value_dict)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='shuttle_sequence') # training dataset
    parser.add_argument('--eval_dataset', type=str, default='shuttle_sequence') # evaluating dataset
    parser.add_argument('--batch_size', type=int, default=512) # batch_size samples， 1 sample = 1 rally
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=int, default=1)
    parser.add_argument('--model_save_path', type=str, default='reward_models_save')
    parser.add_argument('--seed', type=int, default=2024)
    # shot embedding
    parser.add_argument("--shot_type_dim", type=int, default=15) # shot_type embedding dim
    parser.add_argument("--shot_type_num", type=int, default=10) # shot_type_num
    parser.add_argument("--location_type", type=int, default=0) # 1: discrete location, 0: continuous location
    parser.add_argument("--location_num", type=int, default=16) # discrete location num
    parser.add_argument("--location_dim", type=int, default=10) # discrete location embedding dim, if continuous location, location_dim=2
    parser.add_argument("--other_fea_dim", type=int, default=2) # other fea dims, e.g., [aroundhead, backhand] or [bad_landing_flag, landing_distance_opponent]
    # MLP model 
    parser.add_argument('--hidden_dim', type=int, default=512) # hidden_dim in hidden layers
    parser.add_argument('--n_layer', type=int, default=5) # layer num
    parser.add_argument('--activation_function', type=str, default='tanh') # activation function for reward last layer
    # reward
    parser.add_argument("--pref_model", type=int, default=0) # 0: Bradley-Tarry pref model, 1: average Bradley-Tarry pref model
    parser.add_argument("--use_no_end_rally_pref", type=int, default=0) # 0: not use, 1: use no-end rally pref
    parser.add_argument("--no_end_rally_pref_mode", type=int, default=0) # 0: Bradley-Tarry pref model, 1: average Bradley-Tarry pref model
    parser.add_argument("--no_end_rally_pref_factor", type=float, default=1.0) 
    parser.add_argument("--loss_type", type=int, default=1) # 0: not use action_loss, 1: use action_loss
    parser.add_argument("--ensemble_size", type=int, default=5)
    parser.add_argument("--action_pref_type", type=int, default=0) # 0: rally end action vs. others; 1: player end actions vs. player others
    parser.add_argument("--action_pref_factor", type=float, default=1.0)
    
    args = parser.parse_args()

    experiment('BadmintonRewardLearnFinal', variant=vars(args))
# -*- coding: utf-8 -*-
'''
@File    :   reward_model_analysis.py
@Time    :   2024/09/08 16:20:22
@Author  :   Mingjiang Liu 
@Version :   1.0
@Desc    :   preference-based reward model analysis
'''

import argparse
import pickle
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from prediction.models.reward_model import RewardModel
from offline_rl.utils import convert_data_to_drl
from data.preprocess_badminton_data import ACTIONS


def draw_heatmap(matrix, title_name, save_path):
    plt.figure(figsize=(8,8))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="viridis")

    plt.title(title_name)
    plt.xlabel('shot type')
    plt.ylabel('last shot type')

    # 保存热图
    plt.savefig(os.path.join(save_path, title_name+".png"))

    # 显示热图
    # plt.show()

def draw_boxplot(data, distance_type, save_path):
    # data: {key: value}

    # 计算平均值和标准差
    keys = list(data.keys())
    means = [np.mean(data[key]) for key in keys]
    std_devs = [np.std(data[key]) for key in keys]

    # 绘制箱线图
    fig, ax = plt.subplots()

    # 绘制箱线图
    ax.boxplot([data[key] for key in keys], labels=keys, sym="bx")

    # # 添加平均值和标准差
    # for i, key in enumerate(keys):
    #     ax.errorbar(i + 1, means[i], yerr=std_devs[i], fmt='o', color='red', capsize=5)

    # 设置标题和标签
    # ax.set_title('Boxplot with Mean and Standard Deviation for {}'.format(distance_type))
    ax.set_xlabel('Action Category')
    ax.set_ylabel('Reward Value')

    plt.savefig(os.path.join(save_path, distance_type + '_distribution.eps'), format="eps")
    # 显示图形
    # plt.show()

def compute_mean(data_list):
    if not data_list:
        return 0
    return np.mean(data_list)

def draw_scatter(data, title_name, x_label, y_label, save_path):
    # data: List[turple(x,y)]
    plt.figure(figsize=(8,8))
    x, y = zip(*data)
    print("< 0.75 num: {}".format((np.array(y) < -0.75).sum()))
    print("{}, {}".format(len(x), len(y)))
    plt.plot(x, y, ".")
    
    plt.title(title_name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.savefig(os.path.join(save_path, title_name + '.png'))
 

def original_data_distribution_analysis(save_path, trajectories):
    win_action_num = 0
    loss_action_num = 0
    win_shot_type_list = [[0] * len(ACTIONS)]
    loss_shot_type_list = [[0] * len(ACTIONS)]
    win_shot_type_table = [[0] * len(ACTIONS) for _ in range(len(ACTIONS))]
    loss_shot_type_table = [[0] * len(ACTIONS) for _ in range(len(ACTIONS))]

    win_landing_oppo_dis_table = [[[] for _ in range(len(ACTIONS))] for _ in range(len(ACTIONS))]
    loss_landing_oppo_dis_table = [[[] for _ in range(len(ACTIONS))] for _ in range(len(ACTIONS))]

    win_move_player_dis_table = [[[] for _ in range(len(ACTIONS))] for _ in range(len(ACTIONS))]
    loss_move_player_dis_table = [[[] for _ in range(len(ACTIONS))] for _ in range(len(ACTIONS))]

    win_move_center_dis_table = [[[] for _ in range(len(ACTIONS))] for _ in range(len(ACTIONS))]
    loss_move_center_dis_table = [[[] for _ in range(len(ACTIONS))] for _ in range(len(ACTIONS))]

    win_landing_oppo_dis = []
    loss_landing_oppo_dis = []
    win_move_player_dis = []
    loss_move_player_dis = []
    win_move_center_dis = []
    loss_move_center_dis = []

    for rally in trajectories:
        for i in range(len(rally["reward"])):
            if rally["hit_xy"][i].sum() == 0:
                # service action
                continue

            if rally["reward"][-1] > 0:
                # win rally
                win_action_num += 1
                win_shot_type_list[0][rally["shot_type"][i]] += 1
                win_shot_type_table[rally["last_time_opponent_type"][i]][rally["shot_type"][i]] += 1
                win_landing_oppo_dis.append(np.linalg.norm(rally["landing_xy"][i]-rally["opponent_location_xy"][i]))
                win_landing_oppo_dis_table[rally["last_time_opponent_type"][i]][rally["shot_type"][i]].append(np.linalg.norm(rally["landing_xy"][i]-rally["opponent_location_xy"][i]))
                if rally["move_xy"][i].sum() == 0:
                    # last action
                    continue
                win_move_player_dis.append(np.linalg.norm(rally["move_xy"][i]-rally["player_location_xy"][i]))
                win_move_center_dis.append(np.linalg.norm(rally["move_xy"][i]-np.array([0.5, 0.5])))
                win_move_player_dis_table[rally["last_time_opponent_type"][i]][rally["shot_type"][i]].append(np.linalg.norm(rally["move_xy"][i]-rally["player_location_xy"][i]))
                win_move_center_dis_table[rally["last_time_opponent_type"][i]][rally["shot_type"][i]].append(np.linalg.norm(rally["move_xy"][i]-np.array([0.5, 0.5])))
            else:
                # loss rally
                loss_action_num += 1
                loss_shot_type_list[0][rally["shot_type"][i]] += 1
                loss_shot_type_table[rally["last_time_opponent_type"][i]][rally["shot_type"][i]] += 1

                if rally["landing_xy"][i][0] > 0 and rally["landing_xy"][i][0] < 1 and rally["landing_xy"][i][1] > 0 and rally["landing_xy"][i][1] < 1:
                    loss_landing_oppo_dis.append(np.linalg.norm(rally["landing_xy"][i]-rally["opponent_location_xy"][i]))
                    loss_landing_oppo_dis_table[rally["last_time_opponent_type"][i]][rally["shot_type"][i]].append(np.linalg.norm(rally["landing_xy"][i]-rally["opponent_location_xy"][i]))
                if rally["move_xy"][i].sum() == 0:
                    # last action
                    continue
                loss_move_player_dis.append(np.linalg.norm(rally["move_xy"][i]-rally["player_location_xy"][i]))
                loss_move_center_dis.append(np.linalg.norm(rally["move_xy"][i]-np.array([0.5, 0.5])))
                loss_move_player_dis_table[rally["last_time_opponent_type"][i]][rally["shot_type"][i]].append(np.linalg.norm(rally["move_xy"][i]-rally["player_location_xy"][i]))
                loss_move_center_dis_table[rally["last_time_opponent_type"][i]][rally["shot_type"][i]].append(np.linalg.norm(rally["move_xy"][i]-np.array([0.5, 0.5])))

    win_shot_type_table = np.array(win_shot_type_table) / win_action_num * 100
    loss_shot_type_table = np.array(loss_shot_type_table) / loss_action_num * 100
    win_shot_type_list = np.array(win_shot_type_list) / win_action_num * 100
    loss_shot_type_list = np.array(loss_shot_type_list) / loss_action_num * 100
    save_path = os.path.join(save_path, "original_data")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    draw_heatmap(win_shot_type_list, "frequency_win_shot_type_list", save_path)
    draw_heatmap(loss_shot_type_list, "frequency_loss_shot_type_list", save_path)
    draw_heatmap(np.abs(win_shot_type_list-loss_shot_type_list), "frequency_win-loss_shot_type_list", save_path)
    draw_heatmap(win_shot_type_table, "frequency_win_shot_type", save_path)
    draw_heatmap(loss_shot_type_table, "frequency_loss_shot_type", save_path)
    draw_heatmap(np.abs(win_shot_type_table-loss_shot_type_table), "frequency_win-loss_shot_type", save_path)

    landing_oppo_dis_data = {}
    landing_oppo_dis_data["win"] = np.exp(np.array(win_landing_oppo_dis))
    landing_oppo_dis_data["loss"] = np.exp(np.array(loss_landing_oppo_dis))
    draw_boxplot(landing_oppo_dis_data, "landing_opponent_distance", save_path)

    move_player_dis_data = {}
    move_player_dis_data["win"] = np.exp(np.array(win_move_player_dis))
    move_player_dis_data["loss"] = np.exp(np.array(loss_move_player_dis))
    draw_boxplot(move_player_dis_data, "move_player_distance", save_path)

    move_center_dis_data = {}
    move_center_dis_data["win"] = np.exp(np.array(win_move_center_dis))
    move_center_dis_data["loss"] = np.exp(np.array(loss_move_center_dis))
    draw_boxplot(move_center_dis_data, "move_center_distance", save_path)

    for i in range(len(ACTIONS)):
        for j in range(len(ACTIONS)):
            win_landing_oppo_dis_table[i][j] = compute_mean(win_landing_oppo_dis_table[i][j])
            loss_landing_oppo_dis_table[i][j] = compute_mean(loss_landing_oppo_dis_table[i][j])
            win_move_player_dis_table[i][j] = compute_mean(win_move_player_dis_table[i][j])
            loss_move_player_dis_table[i][j] = compute_mean(loss_move_player_dis_table[i][j])
            win_move_center_dis_table[i][j] = compute_mean(win_move_center_dis_table[i][j])
            loss_move_center_dis_table[i][j] = compute_mean(loss_move_center_dis_table[i][j])

    win_landing_oppo_dis_table = np.array(win_landing_oppo_dis_table)
    loss_landing_oppo_dis_table = np.array(loss_landing_oppo_dis_table)
    win_move_player_dis_table = np.array(win_move_player_dis_table)
    loss_move_player_dis_table = np.array(loss_move_player_dis_table)
    win_move_center_dis_table = np.array(win_move_center_dis_table)
    loss_move_center_dis_table = np.array(loss_move_center_dis_table)

    draw_heatmap(win_landing_oppo_dis_table, "landing_oppo_dis_win_shot_type", save_path)
    draw_heatmap(loss_landing_oppo_dis_table, "landing_oppo_dis_loss_shot_type", save_path)
    draw_heatmap(np.abs(win_landing_oppo_dis_table-loss_landing_oppo_dis_table), "landing_oppo_dis_win-loss_shot_type", save_path)
    draw_heatmap(win_move_player_dis_table, "move_player_dis_win_shot_type", save_path)
    draw_heatmap(loss_move_player_dis_table, "move_player_dis_loss_shot_type", save_path)
    draw_heatmap(np.abs(win_move_player_dis_table-loss_move_player_dis_table), "move_player_dis_win-loss_shot_type", save_path)
    draw_heatmap(win_move_center_dis_table, "move_center_dis_win_shot_type", save_path)
    draw_heatmap(loss_move_center_dis_table, "move_center_dis_loss_shot_type", save_path)
    draw_heatmap(np.abs(win_move_center_dis_table-loss_move_center_dis_table), "move_center_dis_win-loss_shot_type", save_path)

    print("win actio num: {}".format(win_action_num))
    print("loss actio num: {}".format(loss_action_num))


def reward_distribution_compute(data, variant):
    # data: {key: ndarray}
    save_path = variant["save_path"]
    save_path = os.path.join(save_path, "reward_pred_pref_model_{}".format(variant["reward_pref_model"]))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    pred_reward_shot_type_table = [[[] for _ in range(len(ACTIONS))] for _ in range(len(ACTIONS))]
    pred_reward_shot_type_list = [[] for _ in range(len(ACTIONS))]
    pred_landing_oppo_list = []
    pred_move_player_list = []
    pred_move_center_list = []

    for i in range(len(data["rewards"])):
        if data["hit_xy"][i].sum() == 0:
            # service action
            continue
        
        pred_reward_shot_type_list[data["shot_type"][i]].append(data["pred_reward"][i])
        pred_reward_shot_type_table[data["last_time_shot_type"][i]][data["shot_type"][i]].append(data["pred_reward"][i])

        if data["landing_xy"][i][0] > 0 and data["landing_xy"][i][0] < 1 and data["landing_xy"][i][1] > 0 and data["landing_xy"][i][1] < 1:
            pred_landing_oppo_list.append((np.linalg.norm(data["landing_xy"][i]-data["opponent_location_xy"][i]), data["pred_reward"][i]))
        
        if data["rewards"][i] != 0 and data["move_xy"][i].sum() == 0:
            # last action
            continue
        pred_move_player_list.append((np.linalg.norm(data["move_xy"][i]-data["player_location_xy"][i]), data["pred_reward"][i]))
        pred_move_center_list.append((np.linalg.norm(data["move_xy"][i]-np.array([0.5, 0.5])), data["pred_reward"][i]))

    for i in range(len(ACTIONS)):
        pred_reward_shot_type_list[i] = compute_mean(pred_reward_shot_type_list[i])
        for j in range(len(ACTIONS)):
            pred_reward_shot_type_table[i][j] = compute_mean(pred_reward_shot_type_table[i][j])

    pred_reward_shot_type_list = np.array(pred_reward_shot_type_list).reshape(1, -1)
    draw_heatmap(pred_reward_shot_type_list, "reward_shot_type_list", save_path)
    pred_reward_shot_type_table = np.array(pred_reward_shot_type_table)
    draw_heatmap(pred_reward_shot_type_table, "reward_shot_type", save_path)
    draw_scatter(pred_landing_oppo_list, "reward_landing_oppo_distance", "landing_oppo_dis", "pred_reward", save_path)
    draw_scatter(pred_move_player_list, "reward_move_player_distance", "move_player_dis", "pred_reward", save_path)
    draw_scatter(pred_move_center_list, "reward_move_center_distance", "move_center_dis", "pred_reward", save_path)


def reward_distribution_compute_2(reward_model, sequence_trajectories, variant):
    save_path = os.path.join(variant["save_path"], "reward_pred_pref_model_{}".format(variant["reward_pref_model"]))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    in_end_reward_list = []
    in_end_reward_num = 0
    out_end_reward_list = []
    out_end_reward_num = 0
    mid_action_reward_list = []
    mid_action_num = 0
    mid_win_action_num = 0
    mid_loss_action_num = 0
    mid_win_action_reward_list = []
    mid_loss_action_reward_list = []

    for sequence in sequence_trajectories:
        last_shot_type = torch.from_numpy(sequence["last_time_opponent_type"]).to(dtype=torch.long, device=variant["device"])
        hit_area = None
        hit_xy = torch.from_numpy(sequence["hit_xy"]).to(dtype=torch.float32, device=variant["device"])
        player_area = None
        player_xy = torch.from_numpy(sequence["player_location_xy"]).to(dtype=torch.float32, device=variant["device"])
        opponent_area = None
        opponent_xy = torch.from_numpy(sequence["opponent_location_xy"]).to(dtype=torch.float32, device=variant["device"])
        shot_type = torch.from_numpy(sequence["shot_type"]).to(dtype=torch.long, device=variant["device"])
        landing_area = None
        landing_xy = torch.from_numpy(sequence["landing_xy"]).to(dtype=torch.float32, device=variant["device"])
        move_area = None
        move_xy = torch.from_numpy(sequence["move_xy"]).to(dtype=torch.float32, device=variant["device"])
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

        sequence["pred_reward"] = r_hat.squeeze().cpu().numpy()

        if sequence["reward"][-1] == 0:
            sequence["reward"][-1] = -1
        sequence["reward"][:-1] = 0

        for i in range(1, len(sequence["reward"])):
            if sequence["reward"][i] == 0:
                mid_action_num += 1
                mid_action_reward_list.append(sequence["pred_reward"][i])
                if len(sequence["reward"]) % 2 == 1 and sequence["reward"][-1] == 1 and i % 2 == 1:
                    mid_win_action_reward_list.append(sequence["pred_reward"][i])
                    mid_win_action_num += 1
                if len(sequence["reward"]) % 2 == 1 and sequence["reward"][-1] == 1 and i % 2 == 0:
                    mid_loss_action_reward_list.append(sequence["pred_reward"][i])
                    mid_loss_action_num += 1
                if len(sequence["reward"]) % 2 == 1 and sequence["reward"][-1] == -1 and i % 2 == 1:
                    mid_loss_action_reward_list.append(sequence["pred_reward"][i])
                    mid_loss_action_num += 1
                if len(sequence["reward"]) % 2 == 1 and sequence["reward"][-1] == -1 and i % 2 == 0:
                    mid_win_action_reward_list.append(sequence["pred_reward"][i])
                    mid_win_action_num += 1
                if len(sequence["reward"]) % 2 == 0 and sequence["reward"][-1] == 1 and i % 2 == 0:
                    mid_win_action_reward_list.append(sequence["pred_reward"][i])
                    mid_win_action_num += 1
                if len(sequence["reward"]) % 2 == 0 and sequence["reward"][-1] == 1 and i % 2 == 1:
                    mid_loss_action_reward_list.append(sequence["pred_reward"][i])
                    mid_loss_action_num += 1
                if len(sequence["reward"]) % 2 == 0 and sequence["reward"][-1] == -1 and i % 2 == 0:
                    mid_loss_action_reward_list.append(sequence["pred_reward"][i])
                    mid_loss_action_num += 1
                if len(sequence["reward"]) % 2 == 0 and sequence["reward"][-1] == -1 and i % 2 == 1:
                    mid_win_action_reward_list.append(sequence["pred_reward"][i])
                    mid_win_action_num += 1
            elif sequence["reward"][i] == 1:
                in_end_reward_num += 1
                in_end_reward_list.append(sequence["pred_reward"][i])
            else:
                out_end_reward_num += 1
                out_end_reward_list.append(sequence["pred_reward"][i])

    # show reward dsitribution for rally preference
    pred_reward_for_rally_pref_dict = {
        "win_action": np.array(mid_win_action_reward_list + in_end_reward_list),
        "loss_action": np.array(mid_loss_action_reward_list + out_end_reward_list)
    }

    draw_boxplot(pred_reward_for_rally_pref_dict, "pred_reward_rally_pref", save_path)

    # show reward dsitribution for non-end rally preference
    pred_reward_for_non_end_rally_pref_dict = {
        "win_action": np.array(mid_win_action_reward_list),
        "loss_action": np.array(mid_loss_action_reward_list)
    }

    draw_boxplot(pred_reward_for_non_end_rally_pref_dict, "pred_reward_non_end_rally_pref", save_path)

    # show reward dsitribution for action preference
    pred_reward_for_action_pref_dict = {
        "non_ternimal_action": np.array(mid_win_action_reward_list+mid_loss_action_reward_list),
        "ternimal_win_action": np.array(in_end_reward_list),
        "ternimal_loss_action": np.array(out_end_reward_list)
    }

    draw_boxplot(pred_reward_for_action_pref_dict, "pred_reward_action_pref", save_path)
    print("in-end action num: {}".format(in_end_reward_num))
    print("out-end action num: {}".format(out_end_reward_num))
    print("mid action num: {}".format(mid_action_num))
    print("mid win action num: {}".format(mid_win_action_num))
    print("mid loss action num: {}".format(mid_loss_action_num))
        


def reward_distribution_analysis(variant, trajectories):
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
            shot_type_num=len(ACTIONS),
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

    # generate reward
    dataset = convert_data_to_drl(trajectories)
    last_shot_type = torch.from_numpy(dataset["last_time_shot_type"]).to(dtype=torch.long, device=variant["device"])
    hit_area = None
    hit_xy = torch.from_numpy(dataset["hit_xy"]).to(dtype=torch.float32, device=variant["device"])
    player_area = None
    player_xy = torch.from_numpy(dataset["player_location_xy"]).to(dtype=torch.float32, device=variant["device"])
    opponent_area = None
    opponent_xy = torch.from_numpy(dataset["opponent_location_xy"]).to(dtype=torch.float32, device=variant["device"])
    shot_type = torch.from_numpy(dataset["shot_type"]).to(dtype=torch.long, device=variant["device"])
    landing_area = None
    landing_xy = torch.from_numpy(dataset["landing_xy"]).to(dtype=torch.float32, device=variant["device"])
    move_area = None
    move_xy = torch.from_numpy(dataset["move_xy"]).to(dtype=torch.float32, device=variant["device"])
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
    dataset["pred_reward"] = r_hat.squeeze().cpu().numpy()

    reward_distribution_compute(data=dataset, variant=variant)

    with open(variant["sequence_data_path"], "rb") as f:
        # trajectories = [{}, {}, ...]
        sequence_trajectories = pickle.load(f)

    reward_distribution_compute_2(reward_model, sequence_trajectories, variant)


def distribution_analysis(variant):

    with open(variant["data_path"], "rb") as f:
        # trajectories = [{}, {}, ...]
        trajectories = pickle.load(f)

    # original_data_distribution_analysis(variant["save_path"], trajectories)

    reward_distribution_analysis(variant, trajectories)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/shuttle_both_agent_val.pkl")
    parser.add_argument("--sequence_data_path", type=str, default="data/shuttle_sequence_val.pkl")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--seed", type=int, default=2024)
    # reward model
    parser.add_argument("--reward_pref_model", type=int, default=0)
    parser.add_argument("--reward_use_no_end_rally_pref", type=int, default=0) # 0: not use, 1: use no-end rally pref
    parser.add_argument("--reward_no_end_rally_pref_mode", type=int, default=0) # 0: Bradley-Tarry pref model, 1: average Bradley-Tarry pref model
    parser.add_argument("--reward_no_end_rally_pref_factor", type=float, default=0.5) 
    parser.add_argument("--reward_loss_type", type=int, default=0)
    parser.add_argument("--reward_ensemble_size", type=int, default=5)
    parser.add_argument("--reward_action_pref_type", type=int, default=0)
    parser.add_argument("--reward_learning_rate", type=float, default=1e-4)
    parser.add_argument("--reward_batch_size", type=int, default=512)
    parser.add_argument("--reward_action_pref_factor", type=float, default=1.0)
    parser.add_argument("--reward_shot_type_dim", type=int, default=15)
    parser.add_argument("--reward_location_type", type=int, default=0)
    parser.add_argument("--reward_location_num", type=int, default=16)
    parser.add_argument("--reward_location_dim", type=int, default=10)
    parser.add_argument("--reward_other_fea_dim", type=int, default=2)
    parser.add_argument("--reward_n_layer", type=int, default=5)
    parser.add_argument("--reward_hidden_dim", type=int, default=512)
    parser.add_argument("--reward_activation", type=str, default="tanh")
    parser.add_argument("--reward_model_dir", type=str, default="reward_models_save_final")
    parser.add_argument("--reward_max_epoch", type=int, default=500)

    # save path
    parser.add_argument("--save_path", type=str, default="reward_model_analysis_final")

    args = parser.parse_args()
    distribution_analysis(variant=vars(args))
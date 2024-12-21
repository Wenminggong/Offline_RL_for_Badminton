# -*- coding: utf-8 -*-
'''
@File    :   reward_learn_analysis.py
@Time    :   2024/08/18 13:24:38
@Author  :   Mingjiang Liu 
@Version :   1.0
@Desc    :   printing results of rewarad models
'''

import argparse
import pandas as pd
import os


def get_optimal_result_index(result, max_rally_acc):
    optimal_result = result[result["rally_acc"] == max_rally_acc]
    if len(optimal_result) == 1:
        return optimal_result.index.item()
    
    max_action_acc_end = optimal_result["action_acc_end"].max()
    optimal_result = optimal_result[optimal_result["action_acc_end"] == max_action_acc_end]
    if len(optimal_result) == 1:
        return optimal_result.index.item()
    
    max_action_acc_player = optimal_result["action_acc_player"].max()
    optimal_result = optimal_result[optimal_result["action_acc_player"] == max_action_acc_player]
    if len(optimal_result) == 1:
        return optimal_result.index.item()
    
    return optimal_result[0].index.item()


def result_print(args):
    result_path = os.path.join(args.model_save_path, "result.csv")
    result = pd.read_csv(result_path)

    # use rally pref only
    max_rally_rally_acc = result[(result["pref_model"] == args.pref_model) & (result["loss_type"] == 0)]["rally_acc"].max()
    max_rally_rally_acc_index = get_optimal_result_index(result[(result["pref_model"] == args.pref_model) & (result["loss_type"] == 0)], max_rally_rally_acc)
    max_rally_action_acc_end = result.loc[max_rally_rally_acc_index, "action_acc_end"]
    max_rally_action_acc_player = result.loc[max_rally_rally_acc_index, "action_acc_player"]
    # max_rally_action_acc_end = result[(result["pref_model"] == args.pref_model) & (result["loss_type"] == 0) & (result["rally_acc"] == max_rally_rally_acc)]["action_acc_end"].item()
    # max_rally_action_acc_player = result[(result["pref_model"] == args.pref_model) & (result["loss_type"] == 0) & (result["rally_acc"] == max_rally_rally_acc)]["action_acc_player"].item()

    # use rally pref + action pref 0 (end action pref)
    max_action_rally_acc_end = result[(result["pref_model"] == args.pref_model) & (result["loss_type"] == 1) & (result["action_pref_type"] == 0)]["rally_acc"].max()
    max_action_rally_acc_end_index = get_optimal_result_index(result[(result["pref_model"] == args.pref_model) & (result["loss_type"] == 1) & (result["action_pref_type"] == 0)], max_action_rally_acc_end)
    max_action_action_acc_end_end = result.loc[max_action_rally_acc_end_index, "action_acc_end"]
    max_action_action_acc_end_player = result.loc[max_action_rally_acc_end_index, "action_acc_player"]
    # max_action_action_acc_end_end = result[(result["pref_model"] == args.pref_model) & (result["loss_type"] == 1) & (result["action_pref_type"] == 0) & (result["rally_acc"] == max_action_rally_acc_end)]["action_acc_end"].item()
    # max_action_action_acc_end_player = result[(result["pref_model"] == args.pref_model) & (result["loss_type"] == 1) & (result["action_pref_type"] == 0) & (result["rally_acc"] == max_action_rally_acc_end)]["action_acc_player"].item()

    # use rally pref + action pref 1 (player action pref)
    max_action_rally_acc_player = result[(result["pref_model"] == args.pref_model) & (result["loss_type"] == 1) & (result["action_pref_type"] == 1)]["rally_acc"].max()
    max_action_rally_acc_player_index = get_optimal_result_index(result[(result["pref_model"] == args.pref_model) & (result["loss_type"] == 1) & (result["action_pref_type"] == 1)], max_action_rally_acc_player)
    max_action_action_acc_player_end = result.loc[max_action_rally_acc_player_index, "action_acc_end"]
    max_action_action_acc_player_player = result.loc[max_action_rally_acc_player_index, "action_acc_player"]
    # max_action_action_acc_player_end = result[(result["pref_model"] == args.pref_model) & (result["loss_type"] == 1) & (result["action_pref_type"] == 1) & (result["rally_acc"] == max_action_rally_acc_player)]["action_acc_end"].item()
    # max_action_action_acc_player_player = result[(result["pref_model"] == args.pref_model) & (result["loss_type"] == 1) & (result["action_pref_type"] == 1) & (result["rally_acc"] == max_action_rally_acc_player)]["action_acc_player"].item()

    print("Pref_model:{} ====================================================================================".format(args.pref_model))
    print("OnlyRallyPref      | rally_acc {:.3f} | action_acc_end {:.3f} | action_acc_player {:.3f}".format(max_rally_rally_acc, max_rally_action_acc_end, max_rally_action_acc_player))
    print("WActionPrefEnd     | rally_acc {:.3f} | action_acc_end {:.3f} | action_acc_player {:.3f}".format(max_action_rally_acc_end, max_action_action_acc_end_end, max_action_action_acc_end_player))
    print("WActionPrefPlayer  | rally_acc {:.3f} | action_acc_end {:.3f} | action_acc_player {:.3f}".format(max_action_rally_acc_player, max_action_action_acc_player_end, max_action_action_acc_player_player))

    print("\n")
    print("Optimal_super_parameter============================================================================")
    acc_list = ["rally_acc", "action_acc_end", "action_acc_player"]
    print("{:^15s} | ".format("type"), end="")
    for name in result.columns:
        if name in acc_list:
            continue
        print("{:^14} | ".format(name), end="")
    print("\n")
    print("{:^15s} | ".format("OnlyRallyPref"), end="")
    for name in result.columns:
        if name in acc_list:
            continue
        print("{:^14} | ".format(result.loc[max_rally_rally_acc_index, name]), end="")
    print("\n")
    print("{:^15s} | ".format("WActionPrefEnd"), end="")
    for name in result.columns:
        if name in acc_list:
            continue
        print("{:^14} | ".format(result.loc[max_action_rally_acc_end_index, name]), end="")
    print("\n")
    print("{:^15s} | ".format("WActionPrefPla"), end="")
    for name in result.columns:
        if name in acc_list:
            continue
        print("{:^14} | ".format(result.loc[max_action_rally_acc_player_index, name]), end="")
    print("\n")


def get_optimal_result_index_for_no_end_rally_pref(result, max_rally_no_end_rally_acc):
    optimal_result = result[result["rally_acc"] + result["no_end_rally_acc"] == max_rally_no_end_rally_acc]
    if len(optimal_result) == 1:
        return optimal_result.index.item()
    
    max_action_acc_end = optimal_result["action_acc_end"].max()
    optimal_result = optimal_result[optimal_result["action_acc_end"] == max_action_acc_end]
    if len(optimal_result) == 1:
        return optimal_result.index.item()

    max_action_acc_player = optimal_result["action_acc_player"].max()
    optimal_result = optimal_result[optimal_result["action_acc_player"] == max_action_acc_player]
    if len(optimal_result) == 1:
        return optimal_result.index.item()

    return optimal_result[0].index.item()

def result_print_for_no_end_rally_pref(args):
    result_path = os.path.join(args.model_save_path, "result.csv")
    result = pd.read_csv(result_path)

    max_rally_acc_sum_1 = (result[result["no_end_rally_pref_mode"] == 0]["rally_acc"] + result[result["no_end_rally_pref_mode"] == 0]["no_end_rally_acc"]).max()
    max_rally_acc_sum_1_index = get_optimal_result_index_for_no_end_rally_pref(result[result["no_end_rally_pref_mode"] == 0], max_rally_acc_sum_1)
    max_rally_acc_sum_1_rally_acc = result.loc[max_rally_acc_sum_1_index, "rally_acc"]
    max_rally_acc_sum_1_no_end_rally_acc = result.loc[max_rally_acc_sum_1_index, "no_end_rally_acc"]
    max_rally_acc_sum_1_rally_acc_end = result.loc[max_rally_acc_sum_1_index, "action_acc_end"]
    max_rally_acc_sum_1_rally_acc_player = result.loc[max_rally_acc_sum_1_index, "action_acc_player"]

    max_rally_acc_sum_2 = (result[result["no_end_rally_pref_mode"] == 1]["rally_acc"] + result[result["no_end_rally_pref_mode"] == 1]["no_end_rally_acc"]).max()
    max_rally_acc_sum_2_index = get_optimal_result_index_for_no_end_rally_pref(result[result["no_end_rally_pref_mode"] == 1], max_rally_acc_sum_2)
    max_rally_acc_sum_2_rally_acc = result.loc[max_rally_acc_sum_2_index, "rally_acc"]
    max_rally_acc_sum_2_no_end_rally_acc = result.loc[max_rally_acc_sum_2_index, "no_end_rally_acc"]
    max_rally_acc_sum_2_rally_acc_end = result.loc[max_rally_acc_sum_2_index, "action_acc_end"]
    max_rally_acc_sum_2_rally_acc_player = result.loc[max_rally_acc_sum_2_index, "action_acc_player"]

    print("Pref_model====================================================================================")
    print("NoEndRallyPref0 | rally_acc {:.3f} | no_end_rally_acc {:.3f} | action_acc_end {:.3f} | action_acc_player {:.3f}".format(max_rally_acc_sum_1_rally_acc, max_rally_acc_sum_1_no_end_rally_acc, max_rally_acc_sum_1_rally_acc_end, max_rally_acc_sum_1_rally_acc_player))
    print("NoEndRallyPref1 | rally_acc {:.3f} | no_end_rally_acc {:.3f} | action_acc_end {:.3f} | action_acc_player {:.3f}".format(max_rally_acc_sum_2_rally_acc, max_rally_acc_sum_2_no_end_rally_acc, max_rally_acc_sum_2_rally_acc_end, max_rally_acc_sum_2_rally_acc_player))

    print("\n")
    print("Optimal_super_parameter============================================================================")
    acc_list = ["rally_acc", "no_end_rally_acc", "action_acc_end", "action_acc_player"]
    print("{:^15s} | ".format("type"), end="")
    for name in result.columns:
        if name in acc_list:
            continue
        print("{:^14} | ".format(name), end="")
    print("\n")
    print("{:^15s} | ".format("NoEndRallyPref0"), end="")
    for name in result.columns:
        if name in acc_list:
            continue
        print("{:^14} | ".format(result.loc[max_rally_acc_sum_1_index, name]), end="")
    print("\n")
    print("{:^15s} | ".format("NoEndRallyPref1"), end="")
    for name in result.columns:
        if name in acc_list:
            continue
        print("{:^14} | ".format(result.loc[max_rally_acc_sum_2_index, name]), end="")
    print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_save_path", type=str, default="reward_models_save3")
    parser.add_argument("--pref_model", type=int, default=0)
    parser.add_argument("--print_type", type=str, default="no_end_rally_pref")

    args = parser.parse_args()

    if args.print_type == "no_end_rally_pref":
        result_print_for_no_end_rally_pref(args)
    else:
        result_print(args)
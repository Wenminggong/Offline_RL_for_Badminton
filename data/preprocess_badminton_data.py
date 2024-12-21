# -*- coding: utf-8 -*-
'''
@File    :   preprocess_badminton_data.py
@Time    :   2024/04/13 12:08:47
@Author  :   Mingjiang Liu 
@Version :   2.0
@Desc    :   collect data from preprocessed dataset and save as rally_list.pkl
'''

import os
import numpy as np
import pandas as pd
import pickle
import argparse
from tqdm import tqdm


ACTIONS = {
    "drive": 0,
    "net shot": 1,
    "lob": 2,
    "clear": 3,
    "drop": 4,
    'push/rush': 5,
    "smash": 6,
    "defensive shot": 7,
    "short service": 8,
    "long service": 9
}


class DataProcesser():
    # collect data from SoloShuttlePose
    # state: cur_ball_location, cur_self_pose, cur_opponent_pose
    # action: cur_shot_type
    def __init__(self, dataset_path, target_file_name, player_mode) -> None:
        self.dataset_path = dataset_path
        self.target_file_name = target_file_name
        self.player_mode = player_mode
        self.data_path_list = []
        self.get_filepath()
    
    def get_filepath(self):
        # get all file names under directory
        if not os.path.exists(self.dataset_path):
            self.dataset_path = os.path.join(os.path.dirname(__file__), self.dataset_path)
            assert os.path.exists(self.dataset_path), "directory path not existed!"

        path_list = os.listdir(self.dataset_path)
        for path in path_list:
            real_path = os.path.join(self.dataset_path, path)
            if os.path.isfile(real_path):
                continue
            all_list = os.listdir(real_path)
            for file_name in all_list:
                file_path = os.path.join(real_path, file_name)
                if os.path.isfile(file_path) and "rally" in file_name:
                    self.data_path_list.append(file_path) 

    def collect_data(self, data_path, player_mode, new_state_dim=False):
        try:
            rally = dict()
            rally["data_path"] = data_path
            rally["player_mode"] = player_mode
            rally_data = pd.read_csv(data_path, converters={'ball':eval, 'top':eval, 'bottom':eval})

            collect_rally_data = rally_data.loc[rally_data['pos'] == player_mode]
                
            # collect observations
            bottom_pos = np.array(collect_rally_data['bottom'].tolist())
            bottom_pos = bottom_pos.reshape(bottom_pos.shape[0], -1)
            # print(bottom_pos.shape)
            top_pos = np.array(collect_rally_data['top'].tolist())
            top_pos = top_pos.reshape(top_pos.shape[0], -1)
            # print(top_pos.shape)
            ball_pos = np.array(collect_rally_data['ball'].tolist())
            # print(ball_pos.shape)
            # "observations": self_posture + opponent_posture + ball_position + player_mode_encoding
            if player_mode == "bottom":
                if new_state_dim:
                    rally["observations"] = np.concatenate((bottom_pos, top_pos, ball_pos, np.zeros((ball_pos.shape[0], 1))), axis=1)
                else:
                    rally["observations"] = np.concatenate((bottom_pos, top_pos, ball_pos), axis=1)
            elif player_mode == "top":
                if new_state_dim:
                    rally["observations"] = np.concatenate((top_pos, bottom_pos, ball_pos, np.ones((ball_pos.shape[0], 1))), axis=1)
                else:
                    rally["observations"] = np.concatenate((top_pos, bottom_pos, ball_pos), axis=1)

            # collect actions, one-hot encoding
            rally["actions"] = np.zeros((rally["observations"].shape[0], len(ACTIONS)))
            i = 0
            for action in collect_rally_data['type']:
                rally["actions"][i][ACTIONS[action]] = 1
                i += 1

            # collect last actions, not one-hot, action classes
            collect_rally_ball_round = np.array(collect_rally_data["ball_round"].to_list())
            last_ball_round = collect_rally_ball_round - 1
            last_ball_round_rally = rally_data.loc[rally_data["ball_round"].isin(last_ball_round)]
            rally["last_actions"] = np.zeros(rally["actions"].shape[0])
            num_error = len(rally["last_actions"]) - len(last_ball_round_rally)
            if num_error > 0:
                rally["last_actions"][:num_error] -= 1
                i = num_error
            else:
                i = 0
            for last_action in last_ball_round_rally["type"]:
                rally["last_actions"][i] = ACTIONS[last_action]
                i += 1

            # collect rewards
            rewards = np.zeros(rally["actions"].shape[0])
            if int(rally_data['getpoint_player'].dropna().values[0]) == collect_rally_data['player'].values[0]:
                rewards[-1] = 1
            else:
                rewards[-1] = -1
            rally["rewards"] = rewards

            # collect terminals
            terminals = np.zeros(rally["actions"].shape[0])
            terminals[-1] = 1
            rally["terminals"] = terminals
            return rally
        except Exception as e:
            print("ignore bad data.")
            print(data_path)
            print(e)
            return None

    def process(self):
        data_list = []
        bad_rally_num = 0
        for data_path in tqdm(self.data_path_list):
            if self.player_mode == "bottom":
                rally = self.collect_data(data_path, "bottom")
                if rally:
                    data_list.append(rally)
                else:
                    bad_rally_num += 1
            elif self.player_mode == "top":
                rally = self.collect_data(data_path, "top")
                if rally:
                    data_list.append(rally)
                else:
                    bad_rally_num += 1
            else:
                bottom_rally = self.collect_data(data_path, "bottom", True)
                if bottom_rally:
                    data_list.append(bottom_rally)
                else:
                    bad_rally_num += 1
                top_rally = self.collect_data(data_path, "top", True)
                if top_rally:
                    data_list.append(top_rally)
                else:
                    bad_rally_num += 1

        print("==== total episodes/rallies: {} ===".format(len(data_list)))
        print("==== bad episodes/rallies: {} ===".format(bad_rally_num))
        target_path = os.path.join(os.path.dirname(__file__), self.target_file_name)
        with open(f'{target_path}_{self.player_mode}.pkl', 'wb') as f:
            pickle.dump(data_list, f)

    def count_action_relation(self):
        relation_table = [[0] * len(ACTIONS) for _ in range(len(ACTIONS))]
        for data_path in tqdm(self.data_path_list):
            rally_data = pd.read_csv(data_path)
            action_list = rally_data["type"].to_list()
            for index in range(len(action_list)-1):
                relation_table[ACTIONS[action_list[index]]][ACTIONS[action_list[index+1]]] += 1
        
        I_ACTION = {v:k for k, v in ACTIONS.items()}
        df_table = pd.DataFrame(relation_table, index=[I_ACTION[i] for i in range(len(ACTIONS))], columns=[I_ACTION[j] for j in range(len(ACTIONS))])
        df_table.to_csv("data/action_relation_table.csv")


class AgentBasedDataCollector():
    # collect data from shuttle, shuttleset or shuttleset22 with Agent-based MDP mode
    # state: last_time_opponent_shot_type, cur_hit_xy, cur_opponent_location, cur_self_location
    # action: cur_shot_type, cur_landing_xy, cur_move_xy
    # rewards and terminals
    def __init__(self, dataset_path, target_file_name, player_mode) -> None:
        self.dataset_path = dataset_path
        self.target_file_name = target_file_name
        self.player_mode = player_mode
        if not os.path.exists(self.dataset_path):
            self.dataset_path = os.path.join(os.path.dirname(__file__), self.dataset_path)
        self.target_file_name = os.path.join(os.path.dirname(__file__), self.target_file_name)
    
    def player_data_collect(self, cur_rally_data, player):
        cur_rally_data.reset_index(drop=True, inplace=True)
        rally = dict()
        # scalar features
        rally["match_id"] = cur_rally_data.loc[0, "match_id"]
        rally["set"] = cur_rally_data.loc[0, "set"]
        rally["rally"] = cur_rally_data.loc[0, "rally"]
        rally["rally_id"] = cur_rally_data.loc[0, "rally_id"]
        rally["player_mode"] = player
        rally["lose_reason"] = cur_rally_data.tail(1)["lose_reason"].item()
        rally["getpoint_player"] = cur_rally_data.tail(1)["getpoint_player"].item()

        none_flag = True
        if len(cur_rally_data) == 1:
            # if only one row
            if player == "bottom" and cur_rally_data.loc[0, "player_location_y"] > cur_rally_data.loc[0, "opponent_location_y"]:
                player_id = cur_rally_data.loc[0, "player"]
                none_flag = False
            if player == "top" and cur_rally_data.loc[0, "player_location_y"] < cur_rally_data.loc[0, "opponent_location_y"]:
                player_id = cur_rally_data.loc[0, "player"]
                none_flag = False
        else:
            none_flag = False
            if player == "bottom":
                if cur_rally_data.loc[0, "player_location_y"] > cur_rally_data.loc[0, "opponent_location_y"]:
                    player_id = cur_rally_data.loc[0, "player"]
                else:
                    player_id = cur_rally_data.loc[1, "player"]
            else:
                if cur_rally_data.loc[0, "player_location_y"] < cur_rally_data.loc[0, "opponent_location_y"]:
                    player_id = cur_rally_data.loc[0, "player"]
                else:
                    player_id = cur_rally_data.loc[1, "player"]

        if none_flag:
            # if rally with current player is None
            rally = dict()
        else:
            cur_player_rally_data = cur_rally_data[cur_rally_data["player"] == player_id]
            cur_player_rally_data.reset_index(drop=True, inplace=True)
            # ndarray features
            rally["ball_round"] = cur_player_rally_data.loc[:, "ball_round"].to_numpy() # [sequence], can be used for position-embedding
            rally["frame_num"] = cur_player_rally_data.loc[:, "frame_num"].to_numpy() # [sequence], can be used for time-score
            rally["player"] = cur_player_rally_data.loc[:, "player"].to_numpy() # [sequence], player id, 0-
            rally["shot_type"] = cur_player_rally_data.loc[:, "type"].replace(ACTIONS).to_numpy() # [sequence], 0-9
            rally["last_time_opponent_type"] = cur_player_rally_data.loc[:, "last_time_opponent_type"].fillna(0).replace(ACTIONS).to_numpy() # [sequence], 0-9
            rally["around_head"] = cur_player_rally_data.loc[:, "aroundhead"].to_numpy() # [sequence]
            rally["back_hand"] = cur_player_rally_data.loc[:, "backhand"].to_numpy() # [sequence]
            rally["reward"] = cur_player_rally_data.loc[:, "reward"].to_numpy() #[sequence] 0 or -1 or 1
            rally["terminal"] = np.zeros_like(rally["reward"])
            rally["terminal"][-1] = 1

            # collect area and location coordinates
            rally["hit_area"] = cur_player_rally_data.loc[:, "hit_area"].to_numpy() # [sequence]
            rally["landing_area"] = cur_player_rally_data.loc[:, "landing_area"].to_numpy() # [sequenze]
            rally["player_location_area"] = cur_player_rally_data.loc[:, "landing_area"].to_numpy() # [sequenze]
            rally["opponent_location_area"] = cur_player_rally_data.loc[:, "opponent_location_area"].to_numpy() # [sequenze]
            
            na_flag = cur_player_rally_data["move_area"].isna()
            rally["move_area"] = cur_player_rally_data.loc[:, "move_area"].to_numpy() # [sequenze]
            rally["move_area"][na_flag] = 8 # panding last move_area = 8

            delta_y = (cur_player_rally_data["downleft_y"][0] - cur_player_rally_data["upleft_y"][0]) / 2
            up_delta_x = (1 - 2 * 50 / 610) * (cur_player_rally_data["upright_x"][0] - cur_player_rally_data["upleft_x"][0])
            down_delta_x = (1 - 2 * 50 / 610) * (cur_player_rally_data["downright_x"][0] - cur_player_rally_data["downleft_x"][0])

            up_o_x = cur_player_rally_data["upright_x"][0] - 50 / 610 * (cur_player_rally_data["upright_x"][0] - cur_player_rally_data["upleft_x"][0])
            up_o_y = cur_player_rally_data["upright_y"][0]
            down_o_x = cur_player_rally_data["downleft_x"][0] + 50 / 610 * (cur_player_rally_data["downright_x"][0] - cur_player_rally_data["downleft_x"][0])
            down_o_y = cur_player_rally_data["downleft_y"][0]
            if player == "bottom":
                # collect hit_xy, n x 2, under the reference frame of the player
                na_flag = cur_player_rally_data["hit_x"].isna()
                rally["hit_xy"] = (cur_player_rally_data.loc[:, "hit_x"].fillna(0.0).to_numpy().reshape(-1, 1) - down_o_x) / down_delta_x
                rally["hit_xy"] = np.concatenate([rally["hit_xy"], (down_o_y - cur_player_rally_data.loc[:, "hit_y"].fillna(0.0).to_numpy().reshape(-1, 1)) / delta_y], axis=1)
                rally["hit_xy"][na_flag] = 0.0

                # collect player_location_xy, n x 2, under the reference frame of the player
                rally["player_location_xy"] = (cur_player_rally_data.loc[:, "player_location_x"].to_numpy().reshape(-1, 1) - down_o_x) / down_delta_x
                rally["player_location_xy"] = np.concatenate([rally["player_location_xy"], (down_o_y - cur_player_rally_data.loc[:, "player_location_y"].to_numpy().reshape(-1, 1)) / delta_y], axis=1)

                # collect move_xy, n x 2, under the reference frame of the player
                na_flag = cur_player_rally_data["move_x"].isna()
                rally["move_xy"] = (cur_player_rally_data.loc[:, "move_x"].fillna(0.0).to_numpy().reshape(-1, 1) - down_o_x) / down_delta_x
                rally["move_xy"] = np.concatenate([rally["move_xy"], (down_o_y - cur_player_rally_data.loc[:, "move_y"].fillna(0.0).to_numpy().reshape(-1, 1)) / delta_y], axis=1)
                rally["move_xy"][na_flag] = 0.0

                # collect opponent_location_xy, n x 2, under the reference frame of the opponent
                rally["opponent_location_xy"] = (up_o_x - cur_player_rally_data.loc[:, "opponent_location_x"].to_numpy().reshape(-1, 1)) / up_delta_x
                rally["opponent_location_xy"] = np.concatenate([rally["opponent_location_xy"], (cur_player_rally_data.loc[:, "opponent_location_y"].to_numpy().reshape(-1, 1) - up_o_y) / delta_y], axis=1)

                # collect landing_xy, n x 2, under the reference frame of the opponent
                rally["landing_xy"] = (up_o_x - cur_player_rally_data.loc[:, "landing_x"].to_numpy().reshape(-1, 1)) / up_delta_x
                rally["landing_xy"] = np.concatenate([rally["landing_xy"], (cur_player_rally_data.loc[:, "landing_y"].to_numpy().reshape(-1, 1) - up_o_y) / delta_y], axis=1)
            else:
                # collect hit_xy, n x 2
                na_flag = cur_player_rally_data["hit_x"].isna()
                rally["hit_xy"] = (up_o_x - cur_player_rally_data.loc[:, "hit_x"].fillna(0.0).to_numpy().reshape(-1, 1)) / up_delta_x
                rally["hit_xy"] = np.concatenate([rally["hit_xy"], (cur_player_rally_data.loc[:, "hit_y"].fillna(0.0).to_numpy().reshape(-1, 1) - up_o_y) / delta_y], axis=1)
                rally["hit_xy"][na_flag] = 0.0

                # collect player_location_xy, n x 2
                rally["player_location_xy"] = (up_o_x - cur_player_rally_data.loc[:, "player_location_x"].to_numpy().reshape(-1, 1)) / up_delta_x
                rally["player_location_xy"] = np.concatenate([rally["player_location_xy"], (cur_player_rally_data.loc[:, "player_location_y"].to_numpy().reshape(-1, 1) - up_o_y) / delta_y], axis=1)

                # collect move_xy, n x 2
                na_flag = cur_player_rally_data["move_x"].isna()
                rally["move_xy"] = (up_o_x - cur_player_rally_data.loc[:, "move_x"].fillna(0.0).to_numpy().reshape(-1, 1)) / up_delta_x
                rally["move_xy"] = np.concatenate([rally["move_xy"], (cur_player_rally_data.loc[:, "move_y"].fillna(0.0).to_numpy().reshape(-1, 1) - up_o_y) / delta_y], axis=1)
                rally["move_xy"][na_flag] = 0.0

                # collect opponent_location_xy, n x 2
                rally["opponent_location_xy"] = (cur_player_rally_data.loc[:, "opponent_location_x"].to_numpy().reshape(-1, 1) - down_o_x) / down_delta_x
                rally["opponent_location_xy"] = np.concatenate([rally["opponent_location_xy"], (down_o_y - cur_player_rally_data.loc[:, "opponent_location_y"].to_numpy().reshape(-1, 1)) / delta_y], axis=1)

                # collect landing_xy, n x 2
                rally["landing_xy"] = (cur_player_rally_data.loc[:, "landing_x"].to_numpy().reshape(-1, 1) - down_o_x) / down_delta_x
                rally["landing_xy"] = np.concatenate([rally["landing_xy"], (down_o_y - cur_player_rally_data.loc[:, "landing_y"].to_numpy().reshape(-1, 1)) / delta_y], axis=1)

            # construct bad landing_xy flag, if landing_xy out or no passing or touch net, flag = 1
            rally["bad_landing_flag"] = np.zeros_like(rally["landing_area"])
            bad_landing_flag = np.concatenate([
                (rally["landing_xy"] < 0).any(axis=-1, keepdims=True),
                (rally["landing_xy"] > 1).any(axis=-1, keepdims=True),
                ], axis=-1).any(axis=-1)
            rally["bad_landing_flag"][bad_landing_flag] = 1 # [sequence]

            # construct distance between landing_xy and opponent_location_xy
            rally["landing_distance_opponent"] = np.linalg.norm(rally["landing_xy"] - rally["opponent_location_xy"], axis=-1) # [sequence]

            # collect rally info
            win_player_id = rally["getpoint_player"]
            if cur_player_rally_data["A_player_id"].iloc[0] == win_player_id:
                # win player is A
                roundscore_a = cur_player_rally_data["roundscore_A"].iloc[0] - 1
                roundscore_b = cur_player_rally_data["roundscore_B"].iloc[0]
            else:
                # win player is B
                roundscore_a = cur_player_rally_data["roundscore_A"].iloc[0]
                roundscore_b = cur_player_rally_data["roundscore_B"].iloc[0] - 1
            
            rally["score_diff"] = np.ones_like(rally["player"])
            rally["cons_score"] = np.zeros_like(rally["score_diff"])
            cur_player_id = cur_player_rally_data["player"].iloc[0]
            if cur_player_id == cur_player_rally_data["A_player_id"].iloc[0]:
                # if cur player is A
                rally["score_diff"][:] = roundscore_a - roundscore_b
                rally["cons_score"][:] = cur_player_rally_data["cons_roundscore_A"].iloc[0]
            else:
                # if cur player is B
                rally["score_diff"][:] = roundscore_b - roundscore_a
                rally["cons_score"][:] = cur_player_rally_data["cons_roundscore_B"].iloc[0]
        return rally


    def single_file_collect(self, file_path):
        data_list = []
        data = pd.read_csv(file_path)
        rally_id_list = data["rally_id"].unique().tolist()
        for rally_id in rally_id_list:
            cur_rally_data = data[data["rally_id"] == rally_id]
            if self.player_mode == "bottom":
                rally = self.player_data_collect(cur_rally_data, "bottom")
                if len(rally) > 0:
                    data_list.append(rally)
            elif self.player_mode == "top":
                rally = self.player_data_collect(cur_rally_data, "top")
                if len(rally) > 0:
                    data_list.append(rally)
            else:
                bottom_rally = self.player_data_collect(cur_rally_data, "bottom")
                if len(bottom_rally) > 0:
                    data_list.append(bottom_rally)
                top_rally = self.player_data_collect(cur_rally_data, "top")
                if len(top_rally) > 0:
                    data_list.append(top_rally)
      
        return data_list
    
    def single_file_save(self, file_name):
        file_path = os.path.join(self.dataset_path, file_name)
        if os.path.exists(file_path):
            data_list = self.single_file_collect(file_path)
            print("==== total {} episodes/rallies: {} ===".format(file_name.split(".")[0], len(data_list)))
            with open(f'{self.target_file_name}_{self.player_mode}_agent_{file_name.split(".")[0]}.pkl', 'wb') as f:
                pickle.dump(data_list, f)
        else:
            print("{} file not exist!".format(file_name.split(".")[0]))

    def collect(self):
        # collect train data
        print("===============> collect train data.")
        self.single_file_save("train.csv")
        
        # collect val data
        print("===============> collect val data.")
        self.single_file_save("val.csv")
        
        # collect test data
        print("===============> collect test data.")
        self.single_file_save("test.csv")


class SequenceBasedDataCollector():
    # collect data from shuttle, shuttleset or shuttleset22 with sequence-based mode
    def __init__(self, dataset_path, target_file_name) -> None:
        self.dataset_path = dataset_path
        self.target_file_name = target_file_name
        if not os.path.exists(self.dataset_path):
            self.dataset_path = os.path.join(os.path.dirname(__file__), self.dataset_path)        
        self.target_file_name = os.path.join(os.path.dirname(__file__), self.target_file_name)

    def sequence_data_collect(self, cur_rally_data):
        cur_rally_data.reset_index(drop=True, inplace=True)
        rally = dict()
        # scalar features
        rally["match_id"] = cur_rally_data.loc[0, "match_id"]
        rally["set"] = cur_rally_data.loc[0, "set"]
        rally["rally"] = cur_rally_data.loc[0, "rally"]
        rally["rally_id"] = cur_rally_data.loc[0, "rally_id"]
        rally["lose_reason"] = cur_rally_data.tail(1)["lose_reason"].item()
        rally["getpoint_player"] = cur_rally_data.tail(1)["getpoint_player"].item()

        # ndarray features
        rally["ball_round"] = cur_rally_data.loc[:, "ball_round"].to_numpy() # [sequence], can be used for position-embedding
        rally["frame_num"] = cur_rally_data.loc[:, "frame_num"].to_numpy() # [sequence], can be used for time-score
        rally["player"] = cur_rally_data.loc[:, "player"].to_numpy() # [sequence], player id, 0-
        rally["shot_type"] = cur_rally_data.loc[:, "type"].replace(ACTIONS).to_numpy() # [sequence], 0-9
        rally["last_time_opponent_type"] = cur_rally_data.loc[:, "last_time_opponent_type"].fillna(0).replace(ACTIONS).to_numpy() # [sequence], 0-9
        rally["around_head"] = cur_rally_data.loc[:, "aroundhead"].to_numpy() # [sequence]
        rally["back_hand"] = cur_rally_data.loc[:, "backhand"].to_numpy() # [sequence]
        rally["reward"] = cur_rally_data.loc[:, "reward"].replace(-1, 0).to_numpy() #[sequence] 0 or 1

        # collect area and location coordinates
        rally["hit_area"] = cur_rally_data.loc[:, "hit_area"].to_numpy() # [sequence]
        rally["landing_area"] = cur_rally_data.loc[:, "landing_area"].to_numpy() # [sequenze]
        rally["player_location_area"] = cur_rally_data.loc[:, "landing_area"].to_numpy() # [sequenze]
        rally["opponent_location_area"] = cur_rally_data.loc[:, "opponent_location_area"].to_numpy() # [sequenze]
        
        na_flag = cur_rally_data["move_area"].isna()
        rally["move_area"] = cur_rally_data.loc[:, "move_area"].to_numpy() # [sequenze]
        rally["move_area"][na_flag] = 8 # panding last move_area = 8

        if cur_rally_data.loc[0, "player_location_y"] < cur_rally_data.loc[0, "opponent_location_y"]:
            start_player_location = "up"
        else:
            start_player_location = "down"
        
        up_delta_x = (1 - 2 * 50 / 610) * (cur_rally_data["upright_x"][0] - cur_rally_data["upleft_x"][0])
        down_delta_x = (1 - 2 * 50 / 610) * (cur_rally_data["downright_x"][0] - cur_rally_data["downleft_x"][0])
        delta_y = (cur_rally_data["downleft_y"][0] - cur_rally_data["upleft_y"][0]) / 2
        up_o_x = cur_rally_data["upright_x"][0] - 50 / 610 * (cur_rally_data["upright_x"][0] - cur_rally_data["upleft_x"][0])
        up_o_y = cur_rally_data["upright_y"][0]
        down_o_x = cur_rally_data["downleft_x"][0] + 50 / 610 * (cur_rally_data["downright_x"][0] - cur_rally_data["downleft_x"][0])
        down_o_y = cur_rally_data["downleft_y"][0]

        hit_na_flag = cur_rally_data["hit_x"].isna()
        rally["hit_xy"] = cur_rally_data["hit_x"].fillna(0.0).to_numpy().reshape(-1, 1) # [sequence, 1]
        rally["hit_xy"] = np.concatenate([rally["hit_xy"], cur_rally_data["hit_y"].fillna(0.0).to_numpy().reshape(-1, 1)], axis=-1) # [sequence, 2]

        rally["landing_xy"] = cur_rally_data["landing_x"].to_numpy().reshape(-1, 1) # [sequence, 1]
        rally["landing_xy"] = np.concatenate([rally["landing_xy"], cur_rally_data["landing_y"].to_numpy().reshape(-1, 1)], axis=-1) # [sequence, 2]

        rally["player_location_xy"] = cur_rally_data["player_location_x"].to_numpy().reshape(-1, 1) # [sequence, 1]
        rally["player_location_xy"] = np.concatenate([rally["player_location_xy"], cur_rally_data["player_location_y"].to_numpy().reshape(-1, 1)], axis=-1) # [sequence, 2]

        rally["opponent_location_xy"] = cur_rally_data["opponent_location_x"].to_numpy().reshape(-1, 1) # [sequence, 1]
        rally["opponent_location_xy"] = np.concatenate([rally["opponent_location_xy"], cur_rally_data["opponent_location_y"].to_numpy().reshape(-1, 1)], axis=-1) # [sequence, 2]

        move_na_flag = cur_rally_data["move_x"].isna()
        rally["move_xy"] = cur_rally_data["move_x"].fillna(0.0).to_numpy().reshape(-1, 1) # [sequence, 1]
        rally["move_xy"] = np.concatenate([rally["move_xy"], cur_rally_data["move_y"].fillna(0.0).to_numpy().reshape(-1, 1)], axis=-1) # [sequence, 2]

        if start_player_location == "up":
            rally["hit_xy"][::2, 0] = (up_o_x - rally["hit_xy"][::2, 0]) / up_delta_x
            rally["hit_xy"][::2, 1] = (rally["hit_xy"][::2, 1] - up_o_y) / delta_y
            rally["hit_xy"][1::2, 0] = (rally["hit_xy"][1::2, 0] - down_o_x) / down_delta_x
            rally["hit_xy"][1::2, 1] = (down_o_y - rally["hit_xy"][1::2, 1]) / delta_y
            rally["hit_xy"][hit_na_flag] = 0.0

            rally["landing_xy"][1::2, 0] = (up_o_x - rally["landing_xy"][1::2, 0]) / up_delta_x
            rally["landing_xy"][1::2, 1] = (rally["landing_xy"][1::2, 1] - up_o_y) / delta_y
            rally["landing_xy"][::2, 0] = (rally["landing_xy"][::2, 0] - down_o_x) / down_delta_x
            rally["landing_xy"][::2, 1] = (down_o_y - rally["landing_xy"][::2, 1]) / delta_y

            rally["player_location_xy"][::2, 0] = (up_o_x - rally["player_location_xy"][::2, 0]) / up_delta_x
            rally["player_location_xy"][::2, 1] = (rally["player_location_xy"][::2, 1] - up_o_y) / delta_y
            rally["player_location_xy"][1::2, 0] = (rally["player_location_xy"][1::2, 0] - down_o_x) / down_delta_x
            rally["player_location_xy"][1::2, 1] = (down_o_y - rally["player_location_xy"][1::2, 1]) / delta_y

            rally["opponent_location_xy"][1::2, 0] = (up_o_x - rally["opponent_location_xy"][1::2, 0]) / up_delta_x
            rally["opponent_location_xy"][1::2, 1] = (rally["opponent_location_xy"][1::2, 1] - up_o_y) / delta_y
            rally["opponent_location_xy"][::2, 0] = (rally["opponent_location_xy"][::2, 0] - down_o_x) / down_delta_x
            rally["opponent_location_xy"][::2, 1] = (down_o_y - rally["opponent_location_xy"][::2, 1]) / delta_y

            rally["move_xy"][::2, 0] = (up_o_x - rally["move_xy"][::2, 0]) / up_delta_x
            rally["move_xy"][::2, 1] = (rally["move_xy"][::2, 1] - up_o_y) / delta_y
            rally["move_xy"][1::2, 0] = (rally["move_xy"][1::2, 0] - down_o_x) / down_delta_x
            rally["move_xy"][1::2, 1] = (down_o_y - rally["move_xy"][1::2, 1]) / delta_y
            rally["move_xy"][move_na_flag] = 0.0

        elif start_player_location == "down":
            rally["hit_xy"][1::2, 0] = (up_o_x - rally["hit_xy"][1::2, 0]) / up_delta_x
            rally["hit_xy"][1::2, 1] = (rally["hit_xy"][1::2, 1] - up_o_y) / delta_y
            rally["hit_xy"][::2, 0] = (rally["hit_xy"][::2, 0] - down_o_x) / down_delta_x
            rally["hit_xy"][::2, 1] = (down_o_y - rally["hit_xy"][::2, 1]) / delta_y
            rally["hit_xy"][hit_na_flag] = 0.0

            rally["landing_xy"][::2, 0] = (up_o_x - rally["landing_xy"][::2, 0]) / up_delta_x
            rally["landing_xy"][::2, 1] = (rally["landing_xy"][::2, 1] - up_o_y) / delta_y
            rally["landing_xy"][1::2, 0] = (rally["landing_xy"][1::2, 0] - down_o_x) / down_delta_x
            rally["landing_xy"][1::2, 1] = (down_o_y - rally["landing_xy"][1::2, 1]) / delta_y

            rally["player_location_xy"][1::2, 0] = (up_o_x - rally["player_location_xy"][1::2, 0]) / up_delta_x
            rally["player_location_xy"][1::2, 1] = (rally["player_location_xy"][1::2, 1] - up_o_y) / delta_y
            rally["player_location_xy"][::2, 0] = (rally["player_location_xy"][::2, 0] - down_o_x) / down_delta_x
            rally["player_location_xy"][::2, 1] = (down_o_y - rally["player_location_xy"][::2, 1]) / delta_y

            rally["opponent_location_xy"][::2, 0] = (up_o_x - rally["opponent_location_xy"][::2, 0]) / up_delta_x
            rally["opponent_location_xy"][::2, 1] = (rally["opponent_location_xy"][::2, 1] - up_o_y) / delta_y
            rally["opponent_location_xy"][1::2, 0] = (rally["opponent_location_xy"][1::2, 0] - down_o_x) / down_delta_x
            rally["opponent_location_xy"][1::2, 1] = (down_o_y - rally["opponent_location_xy"][1::2, 1]) / delta_y

            rally["move_xy"][1::2, 0] = (up_o_x - rally["move_xy"][1::2, 0]) / up_delta_x
            rally["move_xy"][1::2, 1] = (rally["move_xy"][1::2, 1] - up_o_y) / delta_y
            rally["move_xy"][::2, 0] = (rally["move_xy"][::2, 0] - down_o_x) / down_delta_x
            rally["move_xy"][::2, 1] = (down_o_y - rally["move_xy"][::2, 1]) / delta_y
            rally["move_xy"][move_na_flag] = 0.0
        else:
            raise NotImplementedError
        
        # construct bad landing_xy flag, if landing_xy out or no passing or touch net, flag = 1
        rally["bad_landing_flag"] = np.zeros_like(rally["landing_area"])
        bad_landing_flag = np.concatenate([
            (rally["landing_xy"] < 0).any(axis=-1, keepdims=True),
            (rally["landing_xy"] > 1).any(axis=-1, keepdims=True),
            ], axis=-1).any(axis=-1)
        rally["bad_landing_flag"][bad_landing_flag] = 1 # [sequence]

        # construct distance between landing_xy and opponent_location_xy
        rally["landing_distance_opponent"] = np.linalg.norm(rally["landing_xy"] - rally["opponent_location_xy"], axis=-1) # [sequence]

        # collect rally info
        win_player_id = cur_rally_data["getpoint_player"].iloc[-1]
        if cur_rally_data["A_player_id"].iloc[0] == win_player_id:
            # win player is A
            roundscore_a = cur_rally_data["roundscore_A"].iloc[0] - 1
            roundscore_b = cur_rally_data["roundscore_B"].iloc[0]
        else:
            # win player is B
            roundscore_a = cur_rally_data["roundscore_A"].iloc[0]
            roundscore_b = cur_rally_data["roundscore_B"].iloc[0] - 1
        
        rally["score_diff"] = np.ones_like(rally["player"])
        rally["cons_score"] = np.zeros_like(rally["score_diff"])
        start_player_id = cur_rally_data["player"].iloc[0]
        if start_player_id == cur_rally_data["A_player_id"].iloc[0]:
            # if A start
            rally["score_diff"][::2] = roundscore_a - roundscore_b
            rally["score_diff"][1::2] = roundscore_b - roundscore_a
            rally["cons_score"][::2] = cur_rally_data["cons_roundscore_A"].iloc[0]
            rally["cons_score"][1::2] = cur_rally_data["cons_roundscore_B"].iloc[0]
        else:
            # if B start
            rally["score_diff"][::2] = roundscore_b - roundscore_a
            rally["score_diff"][1::2] = roundscore_a - roundscore_b
            rally["cons_score"][::2] = cur_rally_data["cons_roundscore_B"].iloc[0]
            rally["cons_score"][1::2] = cur_rally_data["cons_roundscore_A"].iloc[0]

        return rally

    def single_file_collect(self, file_path):
        data_list = []
        data = pd.read_csv(file_path)
        rally_id_list = data["rally_id"].unique().tolist()
        for rally_id in rally_id_list:
            cur_rally_data = data[data["rally_id"] == rally_id]
            rally = self.sequence_data_collect(cur_rally_data)
            data_list.append(rally) # [rally_1, ..., rally_n]
      
        return data_list
    
    def single_file_save(self, file_name):
        file_path = os.path.join(self.dataset_path, file_name)
        if os.path.exists(file_path):
            data_list = self.single_file_collect(file_path)
            print("==== total {} episodes/rallies: {} ===".format(file_name.split(".")[0], len(data_list)))
            with open(f'{self.target_file_name}_sequence_{file_name.split(".")[0]}.pkl', 'wb') as f:
                pickle.dump(data_list, f)
        else:
            print("{} file not exist!".format(file_name.split(".")[0]))

    def collect(self):
        # collect train data
        print("===============> collect train data.")
        self.single_file_save("train.csv")
        
        # collect val data
        print("===============> collect val data.")
        self.single_file_save("val.csv")
        
        # collect test data
        print("===============> collect test data.")
        self.single_file_save("test.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="dt") # "dt": agent-based MDP for tactics generation; "wr_prediction": rally-based or sequence-based for win rate prediction
    parser.add_argument("--dataset_path", type=str, default="shuttle")
    parser.add_argument("--target_file_name", type=str, default="shuttle")
    parser.add_argument("--player_mode", type=str, default="both") # collect "bottom" or "top" or "both" player data, only for type "dt"

    args = parser.parse_args()
    
    if args.type == "dt":
        collector = AgentBasedDataCollector(args.dataset_path, args.target_file_name, args.player_mode)
        collector.collect()
    elif args.type == "wr_prediction":
        collector = SequenceBasedDataCollector(args.dataset_path, args.target_file_name)
        collector.collect()
    else:
        raise NotImplementedError
    
    # with open(os.path.join(os.path.dirname(__file__), "shuttle_sequence_train.pkl"), "rb") as f:
    #     data = pickle.load(f)
    
    # # for key in d[10].keys():
    # #     print("\"{}\": {}".format(key, type(d[10][key])))
    # in_rally_num = 0
    # out_rally_num = 0
    # total_action_num = 0
    # out_landing_flag = 0
    # print(len(data))
    # for rally in data:
    #     total_action_num += len(rally["reward"])
    #     in_rally_num += rally["reward"][-1]
    #     out_rally_num += (1-rally["reward"][-1])
    #     if not rally["reward"][-1]:
    #         out_landing_flag += rally["bad_landing_flag"][-1]
    #         if not rally["bad_landing_flag"][-1]:
    #             print(rally["lose_reason"])
    #             print(rally["landing_xy"][-1])
    # print(rally["bad_landing_flag"])
    # print("total action num: {}".format(total_action_num))
    # print("in action num: {}".format(in_rally_num))
    # print("out action num: {}".format(out_rally_num))
    # print("out landing flag: {}".format(out_landing_flag))

    
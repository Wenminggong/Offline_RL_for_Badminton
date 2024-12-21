# -*- coding: utf-8 -*-
# MIT License

# Copyright (c) 2022 Wei-Yao Wang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
'''
@File    :   shuttleset_data_preprocess.py
@Time    :   2024/05/07 14:42:01
@Author  :   Mingjiang Liu
@Version :   1.0
@Desc    :   preprocess raw data, modify for DT4Badminton.
'''


import os
import re
import ast
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import argparse
import math


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

def compute_consecutive_change_value(value_list, len_list):
    accumulate_v = 0
    consecutive_value_list = [0] * len_list[0]
    for i in range(1, len(value_list)):
        delta_value = value_list[i] - value_list[i-1]
        if delta_value == 0:
            accumulate_v = 0
        else:
            accumulate_v += delta_value
        consecutive_value_list += [accumulate_v] * len_list[i]
    return consecutive_value_list


class PreDataProcessor:
    def __init__(self, path: str):
        # load match info: [match_id, vedio_name, set, duration, winner, loser]
        self.match = pd.read_csv(f"{path}match.csv")
        # convert players to categorical values (anonymize)
        self.show_unique_players(path) 
        self.compute_matchup_counts(path)
        self.match['winner'] = self.match['winner'].apply(lambda x: self.unique_players.index(x)) # 0-n
        self.match['loser'] = self.match['loser'].apply(lambda x: self.unique_players.index(x))
        
        # load homography info: [match_id, vedio_name, homography_matrix, court_coordinate]
        self.homography = pd.read_csv(f"{path}homography.csv")
        
        if 'db' in self.homography.columns:
            self.homography = self.homography.drop(columns=['db'])
        self.homography['set'] = self.match['set']
        self.homography['duration'] = self.match['duration']
        self.homography['winner'] = self.match['winner']
        self.homography['loser'] = self.match['loser']
        self.homography.to_csv(f"{path}match_metadata.csv", index=False)
        # homography_matrix: ndarray
        self.homography_matrix = pd.read_csv(f"{path}match_metadata.csv", converters={'homography_matrix':lambda x: np.array(ast.literal_eval(x))})

        # load all matches data
        all_matches = self.read_metadata(directory=f"{path}")
        cleaned_matches = self.engineer_match(all_matches, path)
        cleaned_matches.to_csv(f"{path}shot_metadata.csv", index=False)

    def read_metadata(self, directory):
        all_matches = []
        for idx in range(len(self.match)):
            match_idx = self.match['id'][idx]
            match_name = self.match['video'][idx]
            winner = self.match['winner'][idx]
            loser = self.match['loser'][idx]
            current_homography = self.homography_matrix[self.homography_matrix['id'] == match_idx]['homography_matrix'].to_numpy()[0]

            match_path = os.path.join(directory, match_name)
            csv_paths = [os.path.join(match_path, f) for f in os.listdir(match_path) if f.endswith('.csv')]
            try:
                assert len(csv_paths) == self.match['set'][idx]
            except:
                print("Bad match {}: num of set is not consistent!".format(match_name))
                continue

            one_match = []
            # match A or B with player id
            a_win_set = 0
            b_win_set = 0
            for csv_path in csv_paths:
                data = pd.read_csv(csv_path)
                a_win_set += (data['roundscore_A'].tail(1).item() > data['roundscore_B'].tail(1).item())
                b_win_set += (data['roundscore_B'].tail(1).item() > data['roundscore_A'].tail(1).item())
                data['set'] = re.findall(r'\d+', os.path.basename(csv_path))[0]

                # compute consecutive score
                a_score_list = data.groupby("rally").agg(lambda x: x.tail(1))["roundscore_A"].to_numpy().tolist()
                rally_action_num = data.groupby("rally").agg(lambda x: len(x))["roundscore_A"].to_numpy().tolist()
                cons_score_a = compute_consecutive_change_value(a_score_list, rally_action_num)
                b_score_list = data.groupby("rally").agg(lambda x: x.tail(1))["roundscore_B"].to_numpy().tolist()
                cons_score_b = compute_consecutive_change_value(b_score_list, rally_action_num)
                data["cons_roundscore_A"] = cons_score_a
                data["cons_roundscore_B"] = cons_score_b
                one_match.append(data)
            for data in one_match:
                if a_win_set > b_win_set:
                    data['A_player_id'] = winner
                    data['B_player_id'] = loser
                    data['player'] = data['player'].replace(['A', 'B'], [winner, loser])
                    data['getpoint_player'] = data['getpoint_player'].replace(['A', 'B'], [winner, loser])
                else:
                    data['A_player_id'] = loser
                    data['B_player_id'] = winner
                    data['player'] = data['player'].replace(['A', 'B'], [loser, winner])
                    data['getpoint_player'] = data['getpoint_player'].replace(['A', 'B'], [loser, winner])

            match = pd.concat(one_match, ignore_index=True, sort=False).assign(match_id=match_idx)

            # project screen coordinate to real coordinate
            match['reward'] = 0
            for i in range(len(match)):
                # project hit coordinate
                if not pd.isna(match['hit_x'][i]):
                    p_real = self.transfer_coordinate(current_homography, match['hit_x'][i], match['hit_y'][i])
                    match.loc[i, 'hit_x'], match.loc[i, 'hit_y'] = round(p_real[0], 1), round(p_real[1], 1)

                # project ball coordinates
                p_real = self.transfer_coordinate(current_homography, match['landing_x'][i], match['landing_y'][i])
                match.loc[i, 'landing_x'], match.loc[i, 'landing_y'] = round(p_real[0], 1), round(p_real[1], 1)

                # project player coordinates
                p_real = self.transfer_coordinate(current_homography, match['player_location_x'][i], match['player_location_y'][i])
                match.loc[i, 'player_location_x'], match.loc[i, 'player_location_y'] = round(p_real[0], 1), round(p_real[1], 1)

                # project opponent coordinates
                p_real = self.transfer_coordinate(current_homography, match['opponent_location_x'][i], match['opponent_location_y'][i])
                match.loc[i, 'opponent_location_x'], match.loc[i, 'opponent_location_y'] = round(p_real[0], 1), round(p_real[1], 1)

                # add reward
                if not pd.isna(match['getpoint_player'][i]):
                    match.loc[i, 'reward'] = 1 if match['getpoint_player'][i] == match['player'][i] else -1
                else:
                    if  i+1 < len(match)  and not pd.isna(match['getpoint_player'][i+1]):
                        match.loc[i, 'reward'] = 1 if match['getpoint_player'][i+1] == match['player'][i] else -1
      
            # add court coordinate
            p_real = self.transfer_coordinate(current_homography, self.homography_matrix['upleft_x'][idx], self.homography_matrix['upleft_y'][idx])
            match.loc[:, 'upleft_x'], match.loc[:, 'upleft_y'] = round(p_real[0], 1), round(p_real[1], 1)

            p_real = self.transfer_coordinate(current_homography, self.homography_matrix['upright_x'][idx], self.homography_matrix['upright_y'][idx])
            match.loc[:, 'upright_x'], match.loc[:, 'upright_y'] = round(p_real[0], 1), round(p_real[1], 1)

            p_real = self.transfer_coordinate(current_homography, self.homography_matrix['downleft_x'][idx], self.homography_matrix['downleft_y'][idx])
            match.loc[:, 'downleft_x'], match.loc[:, 'downleft_y'] = round(p_real[0], 1), round(p_real[1], 1)

            p_real = self.transfer_coordinate(current_homography, self.homography_matrix['downright_x'][idx], self.homography_matrix['downright_y'][idx])
            match.loc[:, 'downright_x'], match.loc[:, 'downright_y'] = round(p_real[0], 1), round(p_real[1], 1)

            all_matches.append(match)

        all_matches = pd.concat(all_matches, ignore_index=True, sort=False)

        # add last timestep shot type and next timestep move coordinate
        last_time_oppo_type = all_matches['type'].copy()
        last_time_oppo_type.loc[-1] = [np.nan]
        last_time_oppo_type.index += 1
        last_time_oppo_type.sort_index(inplace=True)
        last_time_oppo_type = last_time_oppo_type.drop(last_time_oppo_type.index[-1])
        all_matches['last_time_opponent_type'] = last_time_oppo_type
        all_matches.loc[all_matches['ball_round'] == 1, 'last_time_opponent_type'] = np.nan

        next_time_oppo_x, next_time_oppo_y = all_matches['opponent_location_x'].copy(), all_matches['opponent_location_y'].copy()
        next_time_oppo_area = all_matches['opponent_location_area'].copy()
        next_time_oppo_x.loc[next_time_oppo_x.index[-1]+1] = [np.nan]
        next_time_oppo_y.loc[next_time_oppo_y.index[-1]+1] = [np.nan]
        next_time_oppo_area.loc[next_time_oppo_area.index[-1]+1] = [np.nan]
        next_time_oppo_x = next_time_oppo_x.drop(0).reset_index(drop=True)
        next_time_oppo_y = next_time_oppo_y.drop(0).reset_index(drop=True)
        next_time_oppo_area = next_time_oppo_area.drop(0).reset_index(drop=True)
        all_matches['move_x'], all_matches['move_y'] = next_time_oppo_x, next_time_oppo_y
        all_matches['move_area'] = next_time_oppo_area
        all_matches.loc[all_matches['getpoint_player'].notnull(), 'move_x'], all_matches.loc[all_matches['getpoint_player'].notnull(), 'move_y'] = np.nan, np.nan
        all_matches.loc[all_matches['getpoint_player'].notnull(), 'move_area'] = np.nan
        return all_matches
    
    def transfer_coordinate(self, homography, x, y):
        p = np.array([x, y, 1])
        p_real = homography.dot(p)
        p_real /= p_real[2]
        return p_real

    def engineer_match(self, matches, path):
        matches['rally_id'] = matches.groupby(['match_id', 'set', 'rally']).ngroup()
        print("Original: ")
        ori_rally_num, ori_action_num = self.print_current_size(matches)

        # Drop flaw rally
        if 'flaw' in matches.columns:
            flaw_rally = matches[matches['flaw'].notna()]['rally_id']
            matches = matches[~matches['rally_id'].isin(flaw_rally)]
            matches = matches.reset_index(drop=True)
        print("After Dropping flaw: ")
        flaw_rally_num, flaw_action_num = self.print_current_size(matches)

        # drop not getpoint_player rally
        no_getpoint_player_rally = matches[matches['getpoint_player'].notna()]['rally_id']
        matches = matches[matches['rally_id'].isin(no_getpoint_player_rally)]
        matches = matches.reset_index(drop=True)
        print("After Dropping not getpoint_player rally: ")
        not_point_rally_num, not_point_action_num = self.print_current_size(matches)

        # Drop unknown or nan ball type
        matches = self.drop_na_rally(matches, columns=['type'])
        unknown_rally = matches[matches['type'] == '未知球種']['rally_id']
        matches = matches[~matches['rally_id'].isin(unknown_rally)]
        matches = matches.reset_index(drop=True)
        print("After dropping unknown ball type: ")
        unknow_ball_rally_num, unknow_ball_action_num = self.print_current_size(matches)

        # Drop nan hit_area
        # matches.loc[matches['server'] == 1, 'hit_area'] = 8
        # for area in outside_area:
        #     outside_rallies = matches.loc[matches['hit_area'] == area, 'rally_id']
        #     matches = matches[~matches['rally_id'].isin(outside_rallies)]
        #     matches = matches.reset_index(drop=True)
        # Deal with hit_area convert hit_area to integer
        # matches = self.drop_na_rally(matches, columns=['hit_area'])
        # matches['hit_area'] = matches['hit_area'].astype(float).astype(int)
        # print("After converting hit_area: ")
        # hit_area_rally_num, hit_area_action_num = self.print_current_size(matches)

        # drop nan coordinate
        matches.loc[matches['server'] == 1, 'hit_area'] = 8
        matches = self.drop_na_rally(matches, columns=['hit_area'])
        matches['hit_area'] = matches['hit_area'].astype(float).astype(int)
        matches = self.drop_na_rally(matches, columns=['landing_area'])
        matches['landing_area'] = matches['landing_area'].astype(float).astype(int)
        matches = self.drop_na_rally(matches, columns=['landing_x'])
        matches = self.drop_na_rally(matches, columns=['landing_y'])
        matches = self.drop_na_rally(matches, columns=['opponent_location_area'])
        matches['opponent_location_area'] = matches['opponent_location_area'].astype(float).astype(int)
        matches = self.drop_na_rally(matches, columns=['opponent_location_x'])
        matches = self.drop_na_rally(matches, columns=['opponent_location_y'])
        matches = self.drop_na_rally(matches, columns=['player_location_area'])
        matches['player_location_area'] = matches['player_location_area'].astype(float).astype(int)
        matches = self.drop_na_rally(matches, columns=['player_location_x'])
        matches = self.drop_na_rally(matches, columns=['player_location_y'])
        matches.loc[matches["getpoint_player"].notna(), 'move_area'] = 8
        matches['move_area'] = matches['move_area'].astype(float).astype(int)
        print("After converting hit/landing/opponent_location/player_location: ")
        coor_rally_num, coor_action_num = self.print_current_size(matches)

        # Deal with ball type. Convert ball types to general version (10 types)
        # Convert 小平球 to 平球 because of old version
        matches['type'] = matches['type'].replace('小平球', '平球')
        matches['last_time_opponent_type'] = matches['last_time_opponent_type'].replace('小平球', '平球')
        combined_types = {'切球': '切球', '過度切球': '切球', '點扣': '殺球', '殺球': '殺球', '平球': '平球', '後場抽平球': '平球', '擋小球': '接殺防守',
                    '防守回挑': '接殺防守', '防守回抽': '接殺防守', '放小球': '網前球', '勾球': '網前球', '推球': '推撲球', '撲球': '推撲球'}
        shot_type_transform = {
            '發短球': 'short service',
            '長球': 'clear',
            '推撲球': 'push/rush',
            '殺球': 'smash',
            '接殺防守': 'defensive shot',
            '平球': 'drive',
            '網前球': 'net shot',
            '挑球': 'lob',
            '切球': 'drop',
            '發長球': 'long service',
        }
        matches['type'] = matches['type'].replace(combined_types)
        matches['type'] = matches['type'].replace(shot_type_transform)
        matches['last_time_opponent_type'] = matches['last_time_opponent_type'].replace(combined_types)
        matches['last_time_opponent_type'] = matches['last_time_opponent_type'].replace(shot_type_transform)
        print("After converting ball type: ")
        type_rally_num, type_action_num = self.print_current_size(matches)

        # Fill zero value in backhand
        matches['backhand'] = matches['backhand'].fillna(value=0)
        matches['backhand'] = matches['backhand'].astype(float).astype(int)

        # Fill zero value in aroundhead
        matches['aroundhead'] = matches['aroundhead'].fillna(value=0)
        matches['aroundhead'] = matches['aroundhead'].astype(float).astype(int)

        # Convert ball round type to integer
        matches['ball_round'] = matches['ball_round'].astype(float).astype(int)

        # Translate lose reasons from Chinese to English (foul is treated as not pass over the net)
        reason_transform = {'出界': 'out', '落點判斷失誤': 'misjudged', '掛網': 'touched the net', '未過網': 'not pass over the net', '對手落地致勝': "opponent's ball landed", '犯規': 'not pass over the net'}
        matches['lose_reason'] = matches['lose_reason'].replace(reason_transform)
        known_end_reason_rally = matches[matches['lose_reason'].isin(reason_transform.values())]['rally_id']
        matches = matches[matches['rally_id'].isin(known_end_reason_rally)]
        matches = matches.reset_index(drop=True)
        print("After drop unknown lose reason: ")
        unknown_lose_reason_rally_num, unknown_lose_reaso_action_num = self.print_current_size(matches)

        misjudged_rally = matches[matches['lose_reason'] == 'misjudged']['rally_id']
        matches = matches[~matches['rally_id'].isin(misjudged_rally)]
        matches = matches.reset_index(drop=True)
        print("After drop misjudged lose reason: ")
        misjudged_rally_num, misjudged_action_num = self.print_current_size(matches)
    
        # Remove some unrelated fields
        matches = matches.drop(columns=['win_reason','flaw', 'db']) 

        # check lose reason
        out_end_rally = []
        in_end_rally = []
        no_pass_rally = []
        # check hit_xy
        bad_hit_rally = []
        # record action relation
        relation_table = [[0] * len(ACTIONS) for _ in range(len(ACTIONS))]
        # record delta frame num
        delta_frame_num = {}
        # check delta_frame_num
        bad_frame_num_rally = []
        outside_area = [10, 11, 12, 13, 14, 15, 16]
        for i in range(len(matches)):
            if matches.loc[i, 'lose_reason'] == 'out' and matches.loc[i, 'landing_area'] not in outside_area:
                out_end_rally.append(matches.loc[i, 'rally_id'])
            
            if matches.loc[i, 'lose_reason'] == "opponent's ball landed" and matches.loc[i, 'landing_area'] in outside_area:
                in_end_rally.append(matches.loc[i, 'rally_id'])

            if matches.loc[i, 'lose_reason'] == 'touched the net' or matches.loc[i, 'lose_reason'] == 'not pass over the net':
                half_court_len = (matches.loc[i, 'downleft_y'] - matches.loc[i, 'upleft_y']) / 2
                if matches.loc[i, 'player_location_y'] > matches.loc[i, 'upleft_y'] + half_court_len:
                    player_location_flag = "down"
                else:
                    player_location_flag = "up"
                if matches.loc[i, 'landing_y'] > matches.loc[i, 'upleft_y'] + half_court_len:
                    landing_flag = "down"
                else:
                    landing_flag = "up"
                if landing_flag != player_location_flag:
                    no_pass_rally.append(matches.loc[i, 'rally_id'])

            if abs(matches.loc[i, 'hit_x']-matches.loc[i, 'player_location_x']) >= (matches.loc[i, 'upright_x'] - matches.loc[i, 'upleft_x']) or \
            abs(matches.loc[i, 'hit_y']-matches.loc[i, 'player_location_y']) >= (matches.loc[i, 'downleft_y'] - matches.loc[i, 'upleft_y']) / 2:
                bad_hit_rally.append(matches.loc[i, 'rally_id'])

            if matches.loc[i, 'ball_round'] > 1:
                delta_frame = matches.loc[i, 'frame_num'] - matches.loc[i-1, 'frame_num']
                if delta_frame < 0:
                    bad_frame_num_rally.append(matches.loc[i, 'rally_id'])
        
        matches = matches[~matches['rally_id'].isin(out_end_rally)]
        matches = matches.reset_index(drop=True)
        print("After Dropping bad lose_reason-out rally: ")
        out_rally_num, out_action_num = self.print_current_size(matches)

        matches = matches[~matches['rally_id'].isin(in_end_rally)]
        matches = matches.reset_index(drop=True)
        print("After Dropping bad lose_reason-in rally: ")
        in_rally_num, in_action_num = self.print_current_size(matches)

        matches = matches[~matches['rally_id'].isin(no_pass_rally)]
        matches = matches.reset_index(drop=True)
        print("After Dropping bad lose_reason-no-pass rally: ")
        no_pass_rally_num, no_pass_action_num = self.print_current_size(matches)

        matches = matches[~matches['rally_id'].isin(bad_hit_rally)]
        matches = matches.reset_index(drop=True)
        print("After Dropping bad hit rally: ")
        bad_hit_rally_num, bad_hit_action_num = self.print_current_size(matches)

        matches = matches[~matches['rally_id'].isin(bad_frame_num_rally)]
        matches = matches.reset_index(drop=True)
        print("After Dropping bad frame rally: ")
        bad_frame_rally_num, bad_frame_action_num = self.print_current_size(matches)

        for i in range(len(matches)):
            if matches.loc[i, 'ball_round'] > 1:
                relation_table[ACTIONS[matches.loc[i, 'last_time_opponent_type']]][ACTIONS[matches.loc[i, 'type']]] += 1

                delta_frame = matches.loc[i, 'frame_num'] - matches.loc[i-1, 'frame_num']
                if delta_frame in delta_frame_num:
                    delta_frame_num[delta_frame] += 1
                else:
                    delta_frame_num[delta_frame] = 1

        # save rally num and action num
        num_data = pd.DataFrame(
            {'ori_rally_num': ori_rally_num, 'ori_action_num': ori_action_num,
            'flaw_rally_num': flaw_rally_num, 'flaw_action_num': flaw_action_num,
            'not_point_rally_num':not_point_rally_num, 'not_point_action_num': not_point_action_num,
            'unknow_ball_rally_num': unknow_ball_rally_num, 'unknow_ball_action_num': unknow_ball_action_num,
            'coor_rally_num': coor_rally_num, 'coor_action_num': coor_action_num,
            'type_rally_num': type_rally_num, 'type_action_num': type_action_num,
            'unknown_lose_reason_rally_num': unknown_lose_reason_rally_num, 'unknown_lose_reason_action_num': unknown_lose_reaso_action_num,
            'misjudge_rally_num': misjudged_rally_num, 'misjudge_action_num': misjudged_action_num,
            'bad_out_rally_num': out_rally_num, 'bad_out_action_num': out_action_num,
            'bad_in_rally_num': in_rally_num, 'bad_in_action_num': in_action_num,
            'bad_nopass_rally_num': no_pass_rally_num, 'bad_nopass_action_num': no_pass_action_num,
            'bad_hit_rally_num': bad_hit_rally_num, 'bad_hit_action_num': bad_hit_action_num,
            'bad_frame_rally_num': bad_frame_rally_num, 'bad_frame_action_num': bad_frame_action_num}, index=[0]
        )
        num_data.to_csv(f"{path}rally_action_num.csv", index=False)

        # save action relation table
        I_ACTION = {v:k for k, v in ACTIONS.items()}
        df_table = pd.DataFrame(relation_table, index=[I_ACTION[i] for i in range(len(ACTIONS))], columns=[I_ACTION[j] for j in range(len(ACTIONS))])
        df_table.to_csv(f"{path}action_relation_table.csv")

        # save delta frame num
        delta_fn_data = pd.DataFrame(delta_frame_num, index=[0])
        delta_fn_data = delta_fn_data[sorted(delta_fn_data.columns)]
        delta_fn_data['min'] = min(delta_frame_num.keys())
        delta_fn_data['max'] = max(delta_frame_num.keys())
        delta_fn_data.to_csv(f"{path}delta_frame_num.csv", index=False)
        return matches

    def compute_statistics(self, path):
        self.show_unique_players(path)
        self.compute_matchup_counts(path)

    def show_unique_players(self, path):
        # show players
        column_players = self.match.loc[:, ['winner', 'loser']].values.ravel() # convert DataFrame to ndarray to one-dim ndarray
        self.unique_players = pd.unique(column_players).tolist() # str-list
        print(self.unique_players, len(self.unique_players))
        # save player name and their ids
        players_name_id = pd.DataFrame({"name": self.unique_players, "id": range(len(self.unique_players))})
        players_name_id.to_csv(f"{path}players_name_id.csv", index=False)

    def compute_matchup_counts(self, path):
        # compute matchup counts of each player
        column_values = self.match[['winner', 'loser']].values
        players = []
        for column_value in column_values:
            players.append(column_value)

        player_matrix = [[0] * len(self.unique_players) for _ in range(len(self.unique_players))]
        for player in players:
            player_index_row, player_index_col = self.unique_players.index(player[0]), self.unique_players.index(player[1])
            player_matrix[player_index_row][player_index_col] += 1
            player_matrix[player_index_col][player_index_row] += 1
        player_matrix = pd.DataFrame(player_matrix, index=self.unique_players, columns=self.unique_players)
        
        plot = sns.heatmap(player_matrix, annot=True, linewidths=0.5, cbar=False, annot_kws={"size": 8})
        plt.xticks(rotation=30, ha='right')
        plot.get_figure().savefig(f"{path}player_matrix.png", dpi=300, bbox_inches='tight')
        plot.clear()

    def drop_na_rally(self, df, columns=[]):
        """Drop rallies which contain na value in columns."""
        df = df.copy()
        for column in columns:
            rallies = df[df[column].isna()]['rally_id']
            df = df[~df['rally_id'].isin(rallies)]
        df = df.reset_index(drop=True)
        return df

    def print_current_size(self, all_match):
        print('\tUnique rally: {}\t Total rows: {}'.format(all_match['rally_id'].nunique(), len(all_match)))
        return all_match['rally_id'].nunique(), len(all_match)


class CoachAITrainTestSplit:
    def __init__(self, path, seed):
        self.metadata = pd.read_csv(f"{path}shot_metadata.csv")
        self.matches = pd.read_csv(f"{path}match_metadata.csv")
        # self.given_strokes_num = 4

        match_train, match_val, match_test = [], [], []
        match_id = self.metadata['match_id'].unique()
        total_match_num = len(match_id)
        # train : val : test = 7: 2: 1
        np.random.seed(seed)
        val_and_test_id = np.random.choice(match_id, total_match_num // 10 * 3, replace=False)
        test_id = np.random.choice(val_and_test_id, total_match_num // 10, replace=False)
        for match_id in match_id:
            if match_id in test_id:
                match_test.append(self.metadata[self.metadata['match_id']==match_id])
            elif match_id not in test_id and match_id in val_and_test_id:
                match_val.append(self.metadata[self.metadata['match_id']==match_id])
            else:
                match_train.append(self.metadata[self.metadata['match_id']==match_id])

        match_train = pd.concat(match_train, ignore_index=True, sort=False)
        match_val = pd.concat(match_val, ignore_index=True, sort=False)
        match_test = pd.concat(match_test, ignore_index=True, sort=False)
        print("match_num: total: {} / train: {} / val: {} / test: {}".format(total_match_num, match_train['match_id'].nunique(), match_val['match_id'].nunique(), match_test['match_id'].nunique()))
        print("rally_num: train: {} / val: {} / test: {}".format(match_train['rally_id'].nunique(), match_val['rally_id'].nunique(), match_test['rally_id'].nunique()))

        match_train = self.preprocess_files(match_train)
        match_test = self.preprocess_files(match_test)
        match_val = self.preprocess_files(match_val)

        print("========== Val not in Train=========")
        for player in match_val['player'].unique():
            if player not in match_train['player'].unique():
                print(player, sep=', ')
        print("========== Test not in Train=========")
        for player in match_test['player'].unique():
            if player not in match_train['player'].unique():
                print(player, sep=', ')

        # output to csv
        match_train.to_csv(f"{path}train.csv", index=False)
        match_val.to_csv(f"{path}val.csv", index=False)
        match_test.to_csv(f"{path}test.csv", index=False)
        # match_test[match_test['ball_round'].isin([_ for _ in range(self.given_strokes_num+1)])].to_csv(f"{path}test_given.csv", index=False)
        # match_val[match_val['ball_round'].isin([_ for _ in range(self.given_strokes_num+1)])].to_csv(f"{path}val_given.csv", index=False)
        # match_test[~match_test['ball_round'].isin([_ for _ in range(self.given_strokes_num+1)])][['rally_id', 'ball_round', 'type', 'landing_x', 'landing_y']].to_csv(f"{path}test_gt.csv", index=False)
        # match_val[~match_val['ball_round'].isin([_ for _ in range(self.given_strokes_num+1)])][['rally_id', 'ball_round', 'type', 'landing_x', 'landing_y']].to_csv(f"{path}val_gt.csv", index=False)

    def preprocess_files(self, match):
        def flatten(t):
            return [item for sublist in t for item in sublist]

        # compute rally length
        rally_len = []
        for rally_id in match['rally_id'].unique():
            rally_info = match.loc[match['rally_id'] == rally_id]
            rally_len.append([len(rally_info)]*len(rally_info))
        rally_len = flatten(rally_len)
        match['rally_length'] = rally_len

        # # filter rallies that are less than \tau + 1
        # match = match[match['rally_length'] >= self.given_strokes_num+1].reset_index(drop=True)

        return match


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="shuttleset22/")
    parser.add_argument("--seed", type=int, default=666)
    args = parser.parse_args()
    path = os.path.join(os.path.dirname(__file__), args.path)

    data_processor = PreDataProcessor(path=path)

    data_splitter = CoachAITrainTestSplit(path=path, seed=args.seed)
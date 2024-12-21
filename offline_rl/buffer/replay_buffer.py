# -*- coding: utf-8 -*-
'''
@File    :   replay_buffer.py
@Time    :   2024/06/09 20:16:54
@Author  :   Mingjiang Liu 
@Version :   1.0
@Desc    :   replay buffer for offline rl.
'''

import torch
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
TensorBatch = List[torch.Tensor]

class ReplayBuffer():
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        last_time_shot_type_dim: int,
        hit_xy_dim: int,
        player_location_xy_dim: int,
        opponent_location_xy_dim: int,
        shot_type_dim: int,
        landing_xy_dim: int,
        move_xy_dim: int,
        buffer_size: int,
        device: str = "cuda",
        next_action: bool = False,
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0
        self._next_action = next_action
        self._shot_type_dim = shot_type_dim

        # states
        self._last_time_shot_type = torch.zeros(
            buffer_size, dtype=torch.long, device=device
        )
        self._hit_xy = torch.zeros(
            (buffer_size, hit_xy_dim), dtype=torch.float32, device=device
        )
        self._player_location_xy = torch.zeros(
            (buffer_size, player_location_xy_dim), dtype=torch.float32, device=device
        )
        self._opponent_location_xy = torch.zeros(
            (buffer_size, opponent_location_xy_dim), dtype=torch.float32, device=device
        )
        # actions
        self._shot_type = torch.zeros(
            buffer_size, dtype=torch.long, device=device
        )
        self._landing_xy = torch.zeros(
            (buffer_size, landing_xy_dim), dtype=torch.float32, device=device
        )
        self._move_xy = torch.zeros(
            (buffer_size, move_xy_dim), dtype=torch.float32, device=device
        )
        # next states
        self._next_last_time_shot_type = torch.zeros(
            buffer_size, dtype=torch.long, device=device
        )
        self._next_hit_xy = torch.zeros(
            (buffer_size, hit_xy_dim), dtype=torch.float32, device=device
        )
        self._next_player_location_xy = torch.zeros(
            (buffer_size, player_location_xy_dim), dtype=torch.float32, device=device
        )
        self._next_opponent_location_xy = torch.zeros(
            (buffer_size, opponent_location_xy_dim), dtype=torch.float32, device=device
        )
        if next_action:
            # next actions
            self._next_shot_type = torch.zeros(
                buffer_size, dtype=torch.long, device=device
            )
            self._next_landing_xy = torch.zeros(
                (buffer_size, landing_xy_dim), dtype=torch.float32, device=device
            )
            self._next_move_xy = torch.zeros(
                (buffer_size, move_xy_dim), dtype=torch.float32, device=device
            )
            self._next_shot_probs = torch.zeros(
                (buffer_size, shot_type_dim), dtype=torch.float32, device=device
            )

        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["rewards"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._last_time_shot_type[:n_transitions] = self._to_tensor(data["last_time_shot_type"])
        self._hit_xy[:n_transitions] = self._to_tensor(data["hit_xy"])
        self._player_location_xy[:n_transitions] = self._to_tensor(data["player_location_xy"])
        self._opponent_location_xy[:n_transitions] = self._to_tensor(data["opponent_location_xy"])
        self._shot_type[:n_transitions] = self._to_tensor(data["shot_type"])
        self._landing_xy[:n_transitions] = self._to_tensor(data["landing_xy"])
        self._move_xy[:n_transitions] = self._to_tensor(data["move_xy"])
        self._next_last_time_shot_type[:n_transitions] = self._to_tensor(data["next_last_time_shot_type"])
        self._next_hit_xy[:n_transitions] = self._to_tensor(data["next_hit_xy"])
        self._next_player_location_xy[:n_transitions] = self._to_tensor(data["next_player_location_xy"])
        self._next_opponent_location_xy[:n_transitions] = self._to_tensor(data["next_opponent_location_xy"])

        if self._next_action:
            # index = np.arange(n_transitions)[:, None]
            # index = np.concatenate([index, data["next_shot_type"]], axis=-1)
            # self._next_shot_type[index.T] = 1.0
            self._next_shot_type[:n_transitions] = self._to_tensor(data["next_shot_type"])
            self._landing_xy[:n_transitions] = self._to_tensor(data["landing_xy"])
            self._move_xy[:n_transitions] = self._to_tensor(data["move_xy"])
            self._next_shot_probs[:n_transitions] = self._to_tensor(data["next_shot_probs"])

        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][:, None]) # [batch_size, 1]
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][:, None]) # [batch_size, 1]
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)
        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        last_time_shot_type = F.one_hot(self._last_time_shot_type[indices], self._shot_type_dim).to(dtype=torch.float32) # [batch_size, _shot_type_dim]
        hit_xy = self._hit_xy[indices]
        player_location_xy = self._player_location_xy[indices]
        opponent_location_xy = self._opponent_location_xy[indices]
        shot_type = F.one_hot(self._shot_type[indices], self._shot_type_dim).to(dtype=torch.float32)
        landing_xy = self._landing_xy[indices]
        move_xy = self._move_xy[indices]
        next_last_time_shot_type = F.one_hot(self._next_last_time_shot_type[indices], self._shot_type_dim).to(dtype=torch.float32)
        next_hit_xy = self._next_hit_xy[indices]
        next_player_location_xy = self._next_player_location_xy[indices]
        next_opponent_location_xy = self._next_opponent_location_xy[indices]
        
        rewards = self._rewards[indices]
        dones = self._dones[indices]
        # batch_size x n - tensor
        if self._next_action:
            next_shot_type = F.one_hot(self._next_shot_type[indices], self._shot_type_dim).to(dtype=torch.float32)
            next_landing_xy = self._next_landing_xy[indices]
            next_move_xy = self._next_move_xy[indices]
            next_shot_probs = self._next_shot_probs[indices]
            return [last_time_shot_type, hit_xy, player_location_xy, opponent_location_xy, shot_type, landing_xy, move_xy, next_last_time_shot_type, next_hit_xy, next_player_location_xy, next_opponent_location_xy, next_shot_type, next_landing_xy, next_move_xy, next_shot_probs, rewards, dones]
        else:
            return [last_time_shot_type, hit_xy, player_location_xy, opponent_location_xy, shot_type, landing_xy, move_xy, next_last_time_shot_type, next_hit_xy, next_player_location_xy, next_opponent_location_xy, rewards, dones]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError
    
    def sample_all(self) -> TensorBatch:
        if self._next_action:
            return [F.one_hot(self._last_time_shot_type[:self._size], self._shot_type_dim).to(dtype=torch.float32), self._hit_xy[:self._size], self._player_location_xy[:self._size], self._opponent_location_xy[:self._size], F.one_hot(self._shot_type[:self._size], self._shot_type_dim).to(dtype=torch.float32), self._landing_xy[:self._size], self._move_xy[:self._size], F.one_hot(self._next_last_time_shot_type[:self._size], self._shot_type_dim).to(dtype=torch.float32), self._next_hit_xy[:self._size], self._next_player_location_xy[:self._size], self._next_opponent_location_xy[:self._size], F.one_hot(self._next_shot_type[:self._size], self._shot_type_dim).to(dtype=torch.float32), self._next_landing_xy[:self._size], self._next_move_xy[:self._size], self._next_shot_probs[:self._size], self._rewards[:self._size], self._dones[:self._size]]
        else:
            return [F.one_hot(self._last_time_shot_type[:self._size], self._shot_type_dim).to(dtype=torch.float32), self._hit_xy[:self._size], self._player_location_xy[:self._size], self._opponent_location_xy[:self._size], F.one_hot(self._shot_type[:self._size], self._shot_type_dim).to(dtype=torch.float32), self._landing_xy[:self._size], self._move_xy[:self._size], F.one_hot(self._next_last_time_shot_type[:self._size], self._shot_type_dim).to(dtype=torch.float32), self._next_hit_xy[:self._size], self._next_player_location_xy[:self._size], self._next_opponent_location_xy[:self._size], self._rewards[:self._size], self._dones[:self._size]]
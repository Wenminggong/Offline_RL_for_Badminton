# -*- coding: utf-8 -*-
'''
@File    :   replay_buffer.py
@Time    :   2024/07/21 20:29:22
@Author  :   Mingjiang Liu 
@Version :   1.0
@Desc    :   replay buffer for action-based prediction
'''


import torch
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
TensorBatch = List[torch.Tensor]


class ReplayBuffer():
    def __init__(
        self,
        device: str = "cuda",
    ):
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def load_dataset(self, data: Dict[str, np.ndarray]):
        self._action_num = len(data["reward"])
        self._data = data
        # for key in data.keys():
        #     if isinstance(data[key], np.ndarray):
        #         self._data[key] = self._to_tensor(self._data[key])
        print(f"Dataset size: {self._action_num}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, self._action_num, size=batch_size)
        sample_data = {}
        for key in self._data.keys():
            if isinstance(self._data[key], np.ndarray):
                sample_data[key] = self._to_tensor(self._data[key][indices])
        return sample_data
    
    def sample_all(self) -> TensorBatch:
        sample_data = {}
        for key in self._data.keys():
            if isinstance(self._data[key], np.ndarray):
                sample_data[key] = self._to_tensor(self._data[key])
        return sample_data
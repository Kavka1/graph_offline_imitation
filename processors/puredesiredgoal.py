from typing import Dict, List, Optional
import torch
import gym
import numpy as np

from graph_offline_imitation.processors.base    import Processor


class PureDesiredGoalProcessor(Processor):
    def __init__(
        self,
        observation_space:  gym.Space,
        action_space:       gym.Space,
        goal_key:           str = 'desired_goal',
        goal_dim:           int = 2
    ) -> None:
        super().__init__(observation_space, action_space)
        assert isinstance(observation_space, gym.spaces.Dict)
        assert goal_key in list(observation_space.keys())
        self.goal_key   =   goal_key
        self.goal_dim   =   goal_dim

    def forward(self, batch: Dict) -> Dict:
        batch = {k: v for k, v in batch.items()}  # Perform a shallow copy of the batch
        for k in ("obs", "next_obs", "init_obs"):
            if k in batch:
                assert self.goal_key in list(batch[k].keys()) and batch[k][self.goal_key].shape[-1] >= self.goal_dim
                temp_batch                    = torch.zeros_like(batch[k][self.goal_key])
                temp_batch[:, :self.goal_dim] = batch[k][self.goal_key][:, :self.goal_dim]
                batch[k][self.goal_key]       = temp_batch
        return batch
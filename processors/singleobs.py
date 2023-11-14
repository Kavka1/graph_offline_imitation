from typing import Dict, List, Optional

import gym
import numpy as np
import torch

from graph_offline_imitation.utils.utils        import flatten_dict, nest_dict
from graph_offline_imitation.processors.base    import Processor


class SingleObsComponentProcessor(Processor):
    def __init__(
        self,
        observation_space:  gym.Space,
        action_space:       gym.Space,
        obs_include:        str = None,
        obs_dim:            int = -1,
        action_dim:         int = -1,
    ) -> None:
        super().__init__(observation_space, action_space)
        assert isinstance(observation_space, gym.spaces.Dict)
        self.obs_dim            = obs_dim
        self.action_dim         = action_dim
        self.forward_obs_dim    = obs_dim if obs_dim < 0 else obs_dim + 1
        self.forward_action_dim = action_dim if action_dim < 0 else action_dim + 1
        self.obs_include        = obs_include
        if self.obs_include is not None:
            self._observation_space = self.observation_space[self.obs_include]

    def forward(self, batch: Dict) -> Dict:
        batch = {k: v for k, v in batch.items()}  # Perform a shallow copy of the batch
        for k in ("obs", "next_obs", "init_obs"):
            if k in batch:
                batch[k] = batch[k][self.obs_include] 
        return batch
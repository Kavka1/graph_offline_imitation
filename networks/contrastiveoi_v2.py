from typing import Dict, List, Tuple, Union, Optional, Type
import numpy as np
import torch
import torch.nn as nn
import gym

from graph_offline_imitation import networks
from .mlp                    import EnsembleMLP, MLP, weight_init, partial
from .contrastiveoi          import ContrastiveQ


class ContrastiveOfflineImitationV2Network(nn.Module):
    def __init__(
        self, 
        unlabel_observation_space: gym.Space,
        expert_observation_space:  gym.Space,
        # observation_space: gym.Space,
        action_space:      gym.Space,
        policy_class:      Union[str, Type[nn.Module]],
        qfunc_class:       Union[str, Type[nn.Module]],
        policy_kwargs:     Dict = {},
        qfunc_kwargs:      Dict = {},
        encoder_class:     Union[str, Type[nn.Module]] = None,
        encoder_kwargs:    Dict         = {},
        concat_keys:       List[str]    = [
            "observation",
            "achieved_goal",
            "desired_goal",
        ],  # For hiding "achieved_goal" from the Q, pi networks.
        share_encoder:    bool          = True,
        concat_dim:       Optional[int] = None,  # dimension to concatenate states on
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        assert isinstance(unlabel_observation_space, gym.spaces.Dict) and isinstance(expert_observation_space, gym.spaces.Box)
        # define the obs keys required to be concat
        concat_keys         =   concat_keys.copy()
        if 'observation' not in unlabel_observation_space.spaces:
            concat_keys.remove('observation')
        self.concat_keys    =   concat_keys
        assert all([k in unlabel_observation_space.spaces for k in self.concat_keys])
        assert 'achieved_goal' in unlabel_observation_space.spaces
        assert isinstance(action_space, gym.spaces.Box)

        if concat_dim is None:
            space   =   unlabel_observation_space['achieved_goal']
            if (len(space.shape) == 3 or len(space.shape) == 4) and space.dtype == np.uint8:
                concat_dim = 0
            else:
                concat_dim = -1
        self.forward_concat_dim = concat_dim if concat_dim < 0 else concat_dim + 1

        # obs encoder
        obs_encoder_class   = vars(networks)[encoder_class] if isinstance(encoder_class, str) else encoder_class
        obs_encoder_class   = nn.Identity if encoder_class is None else encoder_class
        _encoder_kwargs     = kwargs.copy()
        _encoder_kwargs.update(encoder_kwargs)
        obs_encoder_space   = unlabel_observation_space['observation']
        self._obs_encoder   = obs_encoder_class(obs_encoder_space, action_space, **encoder_kwargs)
        obs_encoder_output_space = self.obs_encoder.output_space if hasattr(self.obs_encoder, 'output_space') else obs_encoder_space

        # goal encoder      
        goal_encoder_class   = vars(networks)[encoder_class] if isinstance(encoder_class, str) else encoder_class
        goal_encoder_class   = nn.Identity if encoder_class is None else encoder_class
        _encoder_kwargs      = kwargs.copy()
        _encoder_kwargs.update(encoder_kwargs)
        goal_encoder_space   = unlabel_observation_space['desired_goal']
        self._goal_encoder   = goal_encoder_class(goal_encoder_space, action_space, **encoder_kwargs)
        goal_encoder_output_space = self.goal_encoder.output_space if hasattr(self.goal_encoder, 'output_space') else goal_encoder_space

        # policy
        policy_space    = obs_encoder_output_space
        policy_class    = vars(networks)[policy_class] if isinstance(policy_class, str) else policy_class
        _policy_kwargs  = kwargs.copy()
        _policy_kwargs.update(policy_kwargs)
        self._policy    = policy_class(policy_space, action_space, **_policy_kwargs)

        # q function
        qfunc_space     = gym.spaces.Box(
            low     =   np.concatenate([obs_encoder_output_space.low, goal_encoder_output_space.low]),
            high    =   np.concatenate([obs_encoder_output_space.high, goal_encoder_output_space.high]),
            dtype   =   unlabel_observation_space['desired_goal'].dtype
        )
        qfunc_class     = vars(networks)[qfunc_class] if isinstance(qfunc_class, str) else vfunc_class
        _qfunc_kwargs   = kwargs.copy()
        _qfunc_kwargs.update(qfunc_kwargs)
        self._qfunc     = qfunc_class(qfunc_space, action_space, **_qfunc_kwargs)

    def forward(self):
        raise NotImplementedError
    
    @property
    def obs_encoder(self):
        return self._obs_encoder
    
    @property
    def goal_encoder(self):
        return self._goal_encoder
    
    @property
    def policy(self):
        return self._policy
    
    @property
    def qfunc(self):
        return self._qfunc
    
    def format_policy_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            return obs
        elif isinstance(obs, Dict):
            assert 'observation' in obs.keys()
            return obs['observation']
        else:
            raise ValueError

    def format_vfunc_input(self, obs: Union[Dict, torch.Tensor], goal: torch.Tensor = None) -> Tuple[torch.Tensor]:
        if goal is None:
            assert isinstance(obs, Dict) and 'observation' in obs.keys() and 'desired_goal' in obs.keys()
            return obs['observation'], obs['desired_goal']
        else:
            assert isinstance(obs, torch.Tensor)
            return obs, goal
    
    def format_qfunc_input(self, obs: Union[Dict, torch.Tensor], action: torch.Tensor, goal: torch.Tensor = None) -> Tuple[torch.Tensor]:
        if goal is None:
            assert isinstance(obs, Dict) and 'observation' in obs.keys() and 'desired_goal' in obs.keys()
            return obs['observation'], action, obs['desired_goal']
        else:
            assert isinstance(obs, torch.Tensor)
            return obs, action, goal
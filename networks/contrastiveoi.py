from typing import Dict, List, Tuple, Union, Optional, Type
import numpy as np
import torch
import torch.nn as nn
import gym

from graph_offline_imitation import networks
from .mlp                    import EnsembleMLP, MLP, weight_init, partial


class ContrastiveQ(nn.Module):
    def __init__(
        self, 
        observation_space:  gym.Space,
        action_space:       gym.Space,
        repr_dim:           int,
        ensemble_size:      int = 2,
        repr_norm:          bool = False,
        repr_norm_temp:     bool = True,
        ortho_init:         bool = False,
        output_gain:        Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert isinstance(observation_space, gym.spaces.Box) and len(observation_space.shape) == 1
        
        self.repr_dim      = repr_dim
        self.ensemble_size = ensemble_size
        self.repr_norm     = repr_norm
        self.repr_norm_temp= repr_norm_temp
        
        input_dim_for_sa   = observation_space.shape[0] // 2 + action_space.shape[0]
        input_dim_for_g    = observation_space.shape[0] // 2

        if self.ensemble_size > 1:
            self.encoder_sa= EnsembleMLP(input_dim_for_sa, repr_dim, ensemble_size=ensemble_size, **kwargs)
            self.encoder_g = EnsembleMLP(input_dim_for_g, repr_dim, ensemble_size=ensemble_size, **kwargs)
        else:
            self.encoder_sa= MLP(input_dim_for_sa, repr_dim, **kwargs)
            self.encoder_g = MLP(input_dim_for_g, repr_dim, **kwargs)

        self.ortho_init    = ortho_init
        self.output_gain   = output_gain
        self.register_parameter()

    def register_parameter(self) -> None:
        if self.ortho_init:
            self.apply(partial(weight_init, gain=float(self.ortho_init)))
            if self.output_gain is not None:
                self.mlp.last_layer.apply(partial(weight_init, gain=self.output_gain))
    
    def encode(self, obs: torch.Tensor, action: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        sa_repr = self.encoder_sa(torch.concat([obs, action], dim=-1))
        g_repr  = self.encoder_g(goal)
        if self.repr_norm:
            sa_repr     =   sa_repr / torch.linalg.norm(sa_repr, dim=-1, keepdim=True)
            g_repr      =   g_repr / torch.linalg.norm(g_repr, dim=-1, keepdim=True)
            if self.repr_norm_temp:
                raise NotImplementedError("The Running normalization is not implemented")
        return sa_repr, g_repr

    def combine_repr(self, sa_repr: torch.Tensor, g_repr: torch.Tensor) -> torch.Tensor:
        if len(sa_repr.shape)==3 and len(g_repr.shape)==3 and sa_repr.shape[0] == self.ensemble_size:
            return torch.einsum('eiz,ejz->eij', sa_repr, g_repr)
        elif len(sa_repr.shape)==2 and len(g_repr.shape)==2:
            return torch.einsum('iz,jz->ij', sa_repr, g_repr)
        else:
            raise ValueError

    def forward(self, obs: torch.Tensor, action: torch.Tensor, goal: torch.Tensor, return_repr: bool = False) -> torch.Tensor:
        sa_repr, g_repr = self.encode(obs, action, goal)    #   [E, B1, Z], [E, B2, Z]
        if return_repr:
            return self.combine_repr(sa_repr, g_repr), sa_repr, g_repr
        else:
            return self.combine_repr(sa_repr, g_repr)           #   [E, B1, B2]


class ContrastiveV(nn.Module):
    def __init__(
        self, 
        observation_space:  gym.Space,
        action_space:       gym.Space,
        repr_dim:           int,
        ensemble_size:      int = 2,
        repr_norm:          bool = False,
        repr_norm_temp:     bool = True,
        ortho_init:         bool = False,
        output_gain:        Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert isinstance(observation_space, gym.spaces.Box) and len(observation_space.shape) == 1
        
        self.repr_dim      = repr_dim
        self.ensemble_size = ensemble_size
        self.repr_norm     = repr_norm
        self.repr_norm_temp= repr_norm_temp
        
        input_dim_for_s    = observation_space.shape[0] // 2
        input_dim_for_g    = observation_space.shape[0] // 2

        if self.ensemble_size > 1:
            self.encoder_s = EnsembleMLP(input_dim_for_s, repr_dim, ensemble_size=ensemble_size, **kwargs)
            self.encoder_g = EnsembleMLP(input_dim_for_g, repr_dim, ensemble_size=ensemble_size, **kwargs)
        else:
            self.encoder_s = MLP(input_dim_for_s, repr_dim, **kwargs)
            self.encoder_g = MLP(input_dim_for_g, repr_dim, **kwargs)

        self.ortho_init    = ortho_init
        self.output_gain   = output_gain
        self.register_parameter()

    def register_parameter(self) -> None:
        if self.ortho_init:
            self.apply(partial(weight_init, gain=float(self.ortho_init)))
            if self.output_gain is not None:
                self.mlp.last_layer.apply(partial(weight_init, gain=self.output_gain))
    
    def encode(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        s_repr  = self.encoder_s(obs)
        g_repr  = self.encoder_g(goal)
        if self.repr_norm:
            s_repr      =   s_repr / torch.linalg.norm(s_repr, dim=-1, keepdim=True)
            g_repr      =   g_repr / torch.linalg.norm(g_repr, dim=-1, keepdim=True)
            if self.repr_norm_temp:
                raise NotImplementedError("The Running normalization is not implemented")
        return s_repr, g_repr

    def combine_repr(self, s_repr: torch.Tensor, g_repr: torch.Tensor) -> torch.Tensor:
        if len(s_repr.shape)==3 and len(g_repr.shape)==3 and s_repr.shape[0] == self.ensemble_size:
            return torch.einsum('eiz,ejz->eij', s_repr, g_repr)
        elif len(s_repr.shape)==2 and len(g_repr.shape)==2:
            return torch.einsum('iz,jz->ij', s_repr, g_repr)
        else:
            raise ValueError

    def forward(self, obs: torch.Tensor, goal: torch.Tensor, return_repr: bool = False) -> torch.Tensor:
        s_repr, g_repr = self.encode(obs, goal)         #   [E, B1, Z], [E, B2, Z]
        if return_repr:
            return self.combine_repr(s_repr, g_repr), s_repr, g_repr
        else:
            return self.combine_repr(s_repr, g_repr)        #   [E, B1, B2]



class ContrastiveOfflineImitationNetwork(nn.Module):
    def __init__(
        self, 
        unlabel_observation_space: gym.Space,
        expert_observation_space:  gym.Space,
        # observation_space: gym.Space,
        action_space:      gym.Space,
        policy_class:      Union[str, Type[nn.Module]],
        vfunc_class:       Union[str, Type[nn.Module]],
        qfunc_class:       Union[str, Type[nn.Module]],
        policy_kwargs:     Dict = {},
        vfunc_kwargs:      Dict = {},
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

        # v function
        vfunc_space     = gym.spaces.Box(
            low     =   np.concatenate([obs_encoder_output_space.low, goal_encoder_output_space.low]),
            high    =   np.concatenate([obs_encoder_output_space.high, goal_encoder_output_space.high]),
            dtype   =   unlabel_observation_space['desired_goal'].dtype
        )
        vfunc_class     = vars(networks)[vfunc_class] if isinstance(vfunc_class, str) else vfunc_class
        _vfunc_kwargs   = kwargs.copy()
        _vfunc_kwargs.update(vfunc_kwargs)
        self._vfunc     = vfunc_class(vfunc_space, action_space, **_vfunc_kwargs)

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
    def vfunc(self):
        return self._vfunc
    
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
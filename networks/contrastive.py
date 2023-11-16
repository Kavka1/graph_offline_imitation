from typing import Dict, List, Tuple, Union, Optional, Type
import numpy as np
import torch
import torch.nn as nn
import gym

from graph_offline_imitation import networks
from .mlp                    import EnsembleMLP, MLP, weight_init, partial


class ContrastiveGoalCritic(nn.Module):
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

    def forward(self, obs: torch.Tensor, action: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        sa_repr, g_repr = self.encode(obs, action, goal)    #   [E, B1, Z], [E, B2, Z]
        return self.combine_repr(sa_repr, g_repr)           #   [E, B1, B2]


class ContrastiveRLNetwork(nn.Module):
    def __init__(
        self, 
        observation_space: gym.Space,
        action_space:      gym.Space,
        actor_class:       Union[str, Type[nn.Module]],
        value_class:       Union[str, Type[nn.Module]],
        encoder_class:     Union[str, Type[nn.Module]] = None,
        actor_kwargs:      Dict = {},
        value_kwargs:      Dict = {},
        encoder_kwargs:    Dict = {},
        concat_keys:       List[str] = [
            "observation",
            "achieved_goal",
            "desired_goal",
        ],  # For hiding "achieved_goal" from the Q, pi networks.
        share_encoder:    bool = True,
        concat_dim:       Optional[int] = None,  # dimension to concatenate states on
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        assert isinstance(observation_space, gym.spaces.Dict)
        
        # define the obs keys required to be concat
        concat_keys         =   concat_keys.copy()
        if 'observation' not in observation_space.spaces:
            concat_keys.remove('observation')
        self.concat_keys    =   concat_keys

        assert all([k in observation_space.spaces for k in self.concat_keys])
        assert 'achieved_goal' in observation_space.spaces
        assert isinstance(action_space, gym.spaces.Box)

        if concat_dim is None:
            space   =   observation_space['achieved_goal']
            if (len(space.shape) == 3 or len(space.shape) == 4) and space.dtype == np.uint8:
                concat_dim = 0
            else:
                concat_dim = -1
        self.forward_concat_dim = concat_dim if concat_dim < 0 else concat_dim + 1

        # create the encoder
        low             = np.concatenate([observation_space[k].low for k in self.concat_keys], axis=concat_dim)
        high            = np.concatenate([observation_space[k].high for k in self.concat_keys], axis=concat_dim)
        encoder_space   = gym.spaces.Box(low=low, high=high, dtype=observation_space['desired_goal'].dtype)

        encoder_class   = vars(networks)[encoder_class] if isinstance(encoder_class, str) else encoder_class
        encoder_class   = nn.Identity if encoder_class is None else encoder_class
        _encoder_kwargs = kwargs.copy()
        _encoder_kwargs.update(encoder_kwargs)
        self._encoder   = encoder_class(encoder_space, action_space, **_encoder_kwargs)

        policy_space    = self._encoder.output_space if hasattr(self._encoder, "output_space") else encoder_space

        # create the policy
        actor_class     = vars(networks)[actor_class] if isinstance(actor_class, str) else actor_class
        _actor_kwargs   = kwargs.copy()
        _actor_kwargs.update(actor_kwargs)
        self._actor     = actor_class(policy_space, action_space, **_actor_kwargs)

        # create the critic
        value_class     = vars(networks)[value_class] if isinstance(value_class, str) else value_class
        _value_kwargs   = kwargs.copy()
        _value_kwargs.update(value_kwargs)
        self._value     = value_class(policy_space, action_space, **_value_kwargs)

    def forward(self):
        raise NotImplementedError
    
    @property
    def encoder(self):
        return self._encoder
    
    @property
    def actor(self):
        return self._actor
    
    @property
    def value(self):
        return self._value
    
    def format_policy_obs(self, obs: torch.Tensor) -> torch.Tensor:
        assert ('observation' not in self.concat_keys and 'achieved_goal' in self.concat_keys and 'desired_goal' in self.concat_keys)
        return torch.concat([obs[k] for k in self.concat_keys], dim=self.forward_concat_dim)
    
    def format_value_input(self, obs: Dict, action: torch.Tensor) -> Tuple[torch.Tensor]:
        assert ('achieved_goal' in obs.keys() and 'desired_goal' in obs.keys())
        obs_goal      = obs['achieved_goal']
        desired_goal  = obs['desired_goal']
        return obs_goal, action, desired_goal
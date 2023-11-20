from typing import Dict, List, Tuple, Union, Optional, Type
import numpy as np
import torch
import torch.nn as nn
import gym

from graph_offline_imitation import networks


class BCNetwork(nn.Module):
    def __init__(
        self, 
        observation_space: gym.Space,
        action_space:      gym.Space,
        actor_class:       Union[str, Type[nn.Module]],
        encoder_class:     Union[str, Type[nn.Module]] = None,
        actor_kwargs:      Dict = {},
        encoder_kwargs:    Dict = {},
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        assert isinstance(observation_space, gym.spaces.Box)
        assert isinstance(action_space, gym.spaces.Box)
        
        # create the encoder
        encoder_space   = observation_space
        encoder_class   = vars(networks)[encoder_class] if isinstance(encoder_class, str) else encoder_class
        encoder_class   = nn.Identity if encoder_class is None else encoder_class
        _encoder_kwargs = kwargs.copy()
        _encoder_kwargs.update(encoder_kwargs)
        self._encoder   = encoder_class(encoder_space, action_space, **_encoder_kwargs)

        # create the policy
        policy_space    = self._encoder.output_space if hasattr(self._encoder, "output_space") else encoder_space
        actor_class     = vars(networks)[actor_class] if isinstance(actor_class, str) else actor_class
        _actor_kwargs   = kwargs.copy()
        _actor_kwargs.update(actor_kwargs)
        self._actor     = actor_class(policy_space, action_space, **_actor_kwargs)

    def forward(self):
        raise NotImplementedError
    
    @property
    def encoder(self):
        return self._encoder
    
    @property
    def actor(self):
        return self._actor
    
    def format_policy_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return obs
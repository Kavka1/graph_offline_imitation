from typing import Dict, List, Tuple, Union, Optional, Type
import numpy as np
import torch
import torch.nn as nn
import copy
import gym

from graph_offline_imitation import networks
from .smodice                import SMODICE_Discriminator


class ORILDiscriminator(SMODICE_Discriminator):
    def __init__(self, observation_space, action_space, **kwargs):
        super().__init__(observation_space, action_space, **kwargs)


class ORILNetwork(nn.Module):
    def __init__(
        self,
        observation_space:  gym.Space,
        action_space:       gym.Space,
        actor_class:        Union[str, Type[nn.Module]],
        critic_class:       Union[str, Type[nn.Module]],
        discr_class:        Union[str, Type[nn.Module]] = ORILDiscriminator,
        encoder_class:      Union[str, Type[nn.Module]] = None,
        actor_kwargs:       Dict = {},
        critic_kwargs:      Dict = {},
        discr_kwargs:       Dict = {},
        encoder_kwargs:     Dict = {},
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
        self._actor_tar = copy.deepcopy(self._actor)

        # create the value
        critic_space    = self._encoder.output_space if hasattr(self._encoder, "output_space") else encoder_space
        critic_class    = vars(networks)[critic_class] if isinstance(critic_class, str) else critic_class
        _critic_kwargs  = kwargs.copy()
        _critic_kwargs.update(critic_kwargs)
        self._critic    = critic_class(critic_space, action_space, **_critic_kwargs)
        self._critic_tar= copy.deepcopy(self._critic)

        # create the discriminator
        discriminator_class     = vars(networks)[discr_class] if isinstance(discr_class, str) else discr_class
        assert discriminator_class is ORILDiscriminator
        _discr_kwargs           = kwargs.copy()
        _discr_kwargs.update(discr_kwargs)
        self._discriminator     = discriminator_class(observation_space, action_space, **_discr_kwargs)

    def forward(self):
        raise NotImplementedError
    
    @property
    def encoder(self):
        return self._encoder
    
    @property
    def actor(self):
        return self._actor
    
    @property
    def actor_target(self):
        return self._actor_tar

    @property
    def critic(self):
        return self._critic

    @property
    def critic_target(self):
        return self._critic_tar

    @property
    def discriminator(self):
        return self._discriminator
    
    def format_actor_input(self, obs: torch.Tensor) -> torch.Tensor:
        return obs
    
    def format_critic_input(self, obs: Dict, action: torch.Tensor) -> torch.Tensor:
        return obs, action

    def format_discriminator_input(self, obs: Dict, action: torch.Tensor) -> Tuple[torch.Tensor]:
        return obs, action
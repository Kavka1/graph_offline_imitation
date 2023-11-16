from typing import Dict, List, Tuple, Union, Optional, Type
import numpy as np
import torch
import torch.nn as nn
import gym

from graph_offline_imitation import networks
from .mlp                    import EnsembleMLP, MLP, weight_init, partial


class DWBCDiscriminator(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        repr_dim: int,
        output_min: float = 0.1,
        output_max: float = 0.9,
        ortho_init: bool = False,
        output_gain: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()
        assert isinstance(observation_space, gym.spaces.Box) and len(observation_space.shape) == 1
        assert kwargs['output_act'] is None

        self.repr_dim   =   repr_dim
        self.output_min =   output_min
        self.output_max =   output_max
        
        self.fc_sa      =   MLP(observation_space.shape[0] + action_space.shape[0], repr_dim, hidden_layers=[], output_act=kwargs['act'])
        self.fc_logp    =   MLP(action_space.shape[0], repr_dim, hidden_layers=[], output_act=kwargs['act'])
        self.head       =   MLP(repr_dim * 2, 1, **kwargs)

        self.ortho_init = ortho_init
        self.output_gain = output_gain
        self.reset_parameters()

    def reset_parameters(self):
        if self.ortho_init:
            self.apply(partial(weight_init, gain=float(self.ortho_init)))  # use the fact that True converts to 1.0
            if self.output_gain is not None:
                self.mlp.last_layer.apply(partial(weight_init, gain=self.output_gain))

    def forward(self, obs: torch.Tensor, action: torch.Tensor, logprob: torch.Tensor) -> torch.Tensor:
        sa_repr     = self.fc_sa(torch.concat([obs, action], -1))
        logp_repr   = self.fc_logp(logprob)
        logits      = self.head(torch.concat([sa_repr, logp_repr], dim=-1))
        prob        = torch.sigmoid(logits)
        return torch.clamp(prob, min=self.output_min, max=self.output_max)



class DWBCNetwork(nn.Module):
    def __init__(
        self, 
        observation_space: gym.Space,
        action_space:      gym.Space,
        actor_class:       Union[str, Type[nn.Module]],
        discr_class:       Union[str, Type[nn.Module]] = DWBCDiscriminator,
        encoder_class:     Union[str, Type[nn.Module]] = None,
        actor_kwargs:      Dict = {},
        discr_kwargs:      Dict = {},
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

        # create the discriminator
        discriminator_class     = vars(networks)[discr_class] if isinstance(discr_class, str) else discr_class
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
    def discriminator(self):
        return self._discriminator
    
    def format_policy_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return obs
    
    def format_discriminator_input(self, obs: Dict, action: torch.Tensor, logp: torch.Tensor) -> Tuple[torch.Tensor]:
        return obs, action, logp
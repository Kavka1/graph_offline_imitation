from typing import Dict, List, Tuple, Union, Optional, Type
import numpy as np
import torch
import torch.nn as nn
from torch import autograd
import gym

from graph_offline_imitation import networks
from .mlp                    import EnsembleMLP, MLP, weight_init, partial


class SMODICE_Discriminator(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        repr_dim: int,
        ortho_init: bool = False,
        output_gain: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()
        assert isinstance(observation_space, gym.spaces.Box) and len(observation_space.shape) == 1
        assert kwargs['output_act'] is None

        self.repr_dim   =   repr_dim
        self.obs_dim    =   observation_space.shape[0]
        
        self.fc_s       =   MLP(observation_space.shape[0], repr_dim, hidden_layers=[], output_act=kwargs['act'])
        self.fc_a       =   MLP(action_space.shape[0], repr_dim, hidden_layers=[], output_act=kwargs['act'])
        self.head       =   MLP(repr_dim * 2, 1, **kwargs)

        self.ortho_init = ortho_init
        self.output_gain = output_gain
        self.reset_parameters()

    def reset_parameters(self):
        if self.ortho_init:
            self.apply(partial(weight_init, gain=float(self.ortho_init)))  # use the fact that True converts to 1.0
            if self.output_gain is not None:
                self.mlp.last_layer.apply(partial(weight_init, gain=self.output_gain))

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        s_repr      = self.fc_s(obs)
        a_repr      = self.fc_a(action)
        logits      = self.head(torch.concat([s_repr, a_repr], dim=-1))
        return logits
    
    def forward_with_concat(self, obs_act: torch.Tensor) -> torch.Tensor:
        obs, act    = obs_act[:, :self.obs_dim], obs_act[:, self.obs_dim:]
        return self(obs, act)

    def compute_grad_pen(
        self, 
        expert_obs: torch.Tensor, 
        expert_act: torch.Tensor,
        unlabel_obs: torch.Tensor,
        unlabel_act: torch.Tensor, 
        lambda_: int = 10
    ) -> torch.Tensor:
        expert_oa   = torch.concat([expert_obs, expert_act], dim=-1)
        unlable_oa  = torch.concat([unlabel_obs, unlabel_act], dim=-1)

        alpha       = torch.rand(expert_obs.shape[0], 1)
        alpha       = alpha.expand_as(expert_oa).to(expert_obs.device)

        mixup_oa    = alpha * expert_oa + (1 - alpha) * unlable_oa
        mixup_oa.requires_grad = True    

        disc        = self.forward_with_concat(mixup_oa)
        ones        = torch.ones(disc.size()).to(disc.device)
        grad        = autograd.grad(
            outputs         =   disc,
            inputs          =   mixup_oa,
            grad_outputs    =   ones,
            create_graph    =   True,
            retain_graph    =   True,
            only_inputs     =   True
        )[0]

        grad_pen    = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen
    
    def loss(
        self, 
        expert_obs: torch.Tensor, 
        expert_act: torch.Tensor, 
        unlabel_obs: torch.Tensor,
        unlabel_act: torch.Tensor,
    ) -> torch.Tensor:
        disc_expert     =   self(expert_obs, expert_act)
        disc_unlabel    =   self(unlabel_obs, unlabel_act)

        expert_loss     =   nn.functional.binary_cross_entropy_with_logits(
            disc_expert, torch.ones_like(disc_expert).to(disc_expert.device)
        )
        unlabel_loss    =   nn.functional.binary_cross_entropy_with_logits(
            disc_unlabel, torch.zeros_like(disc_unlabel).to(disc_unlabel.device)
        )
        gail_loss       =   expert_loss + unlabel_loss
        grad_pen        =   self.compute_grad_pen(expert_obs, expert_act, unlabel_obs, unlabel_act)
        
        return {
            'loss': gail_loss + grad_pen,
            'ce_loss':  gail_loss,
            'grad_pen': grad_pen
        }
    
    def predict_reward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.eval()
            d = self(obs, act)
            s = torch.sigmoid(d)
            reward = s.log() - (1 - s).log()
            return reward 

    

class SMODICENetwork(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space:      gym.Space,
        actor_class:       Union[str, Type[nn.Module]],
        value_class:       Union[str, Type[nn.Module]],
        discr_class:       Union[str, Type[nn.Module]] = SMODICE_Discriminator,
        encoder_class:     Union[str, Type[nn.Module]] = None,
        actor_kwargs:      Dict = {},
        value_kwargs:      Dict = {},
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

        # create the value
        value_space     = self._encoder.output_space if hasattr(self._encoder, "output_space") else encoder_space
        value_class     = vars(networks)[value_class] if isinstance(value_class, str) else value_class
        _value_kwargs   = kwargs.copy()
        _value_kwargs.update(value_kwargs)
        self._value     = value_class(value_space, action_space, **_value_kwargs)

        # create the discriminator
        discriminator_class     = vars(networks)[discr_class] if isinstance(discr_class, str) else discr_class
        assert discriminator_class is SMODICE_Discriminator
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
    def value(self):
        return self._value

    @property
    def discriminator(self):
        return self._discriminator
    
    def format_policy_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return obs
    
    def format_value_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return obs

    def format_discriminator_input(self, obs: Dict, action: torch.Tensor) -> Tuple[torch.Tensor]:
        return obs, action
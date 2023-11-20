import itertools
from typing import Dict, Type, Optional, Tuple, List, Union, Any

import numpy as np
import torch
import gym

from graph_offline_imitation.processors.base            import Processor, IdentityProcessor
from graph_offline_imitation.networks.bc                import BCNetwork
from graph_offline_imitation.algs.offlineimitation.base import OfflineImitation


class BCOfflineImitation(OfflineImitation):
    def __init__(
        self,
        *args,
        use_data:           str      = 'all',
        encoder_gradients:  str      = "null",
        **kwargs,
    ) -> None:
        # After determining dimension parameters, setup the network
        super().__init__(*args, **kwargs)
        assert encoder_gradients in ("actor", "critic", "both", 'null')
        assert use_data in ('all', 'expert', 'unlabel')
        self.encoder_gradients   = encoder_gradients    
        self.use_data            = use_data
        self.action_range        = [
            float(self.expert_processor.action_space.low.min()),
            float(self.expert_processor.action_space.high.max()),
        ]

    def setup_processor(self, processor_class: Optional[Type[Processor]], processor_kwargs: Dict) -> None:
        if isinstance(self.env.observation_space, gym.spaces.Box):
            expert_observation_space    = self.env.observation_space
            unlabel_observation_space   = self.env.observation_space
        else:
            raise ValueError
        
        if processor_class is None:
            self.expert_processor = IdentityProcessor(expert_observation_space, self.env.action_space)
            self.unlabel_processor= IdentityProcessor(unlabel_observation_space, self.env.action_space)
        else:
            self.expert_processor = processor_class(expert_observation_space, self.env.action_space, **processor_kwargs)
            self.unlabel_processor= processor_class(unlabel_observation_space, self.env.action_space, **processor_kwargs)

        if self.expert_processor.supports_gpu:  # move it to device if it supports GPU computation.
            self.expert_processor = self.expert_processor.to(self.device)
        if self.unlabel_processor.supports_gpu:
            self.unlabel_processor= self.unlabel_processor.to(self.device)

    def setup_network(self, network_class: Type[torch.nn.Module], network_kwargs: Dict) -> None:
        assert network_class is BCNetwork, "Must use BCNetwork with BC."
        self.network = network_class(
            observation_space           =   self.env.observation_space,
            action_space                =   self.env.action_space,
            **network_kwargs,
        ).to(self.device)
        
    def setup_optimizers(self) -> None:
        assert self.encoder_gradients == 'null'
        actor_params                    = self.network.actor.parameters()
        self.optim["actor"]             = self.optim_class(actor_params, **self.optim_kwargs)

    def train_step(self, expert_batch: Dict, unlabel_batch: Dict, step: int, total_steps: int) -> Dict:
        # format batch data (e.g., obs goal concat)
        exp_obs, exp_act            = expert_batch['obs'], expert_batch['action']
        unl_obs, unl_act            = unlabel_batch['obs'], unlabel_batch['action']
        exp_obs, unl_obs            = self.network.encoder(exp_obs), self.network.encoder(unl_obs)
        # get log prob
        dist_exp, dist_unl          = self.network.actor(exp_obs), self.network.actor(unl_obs)
        log_pi_exp, log_pi_unl      = dist_exp.log_prob(exp_act), dist_unl.log_prob(unl_act)

        # update policy
        bc_exp_loss     = -torch.sum(log_pi_exp, -1)
        bc_unl_loss     = -torch.sum(log_pi_unl, -1)
        if self.use_data == 'all':
            p_loss      = bc_exp_loss.mean() + bc_unl_loss.mean()
        elif self.use_data == 'expert':
            p_loss      = bc_exp_loss.mean() 
        else:
            p_loss      = bc_unl_loss.mean()

        self.optim['actor'].zero_grad()
        p_loss.backward()
        self.optim['actor'].step()

        return dict(
            loss_policy             = p_loss.item(),
            loss_policy_expert      = bc_exp_loss.mean().item(),
            loss_policy_unlabel     = bc_unl_loss.mean().item(),
        )

    def _predict(self, batch: Dict, sample: bool = False) -> torch.Tensor:
        with torch.no_grad():
            obs             = self.network.format_policy_obs(batch["obs"])
            z               = self.network.encoder(obs)
            dist            = self.network.actor(z)
            if isinstance(dist, torch.distributions.Distribution):
                action = dist.sample() if sample else dist.loc
            elif torch.is_tensor(dist):
                action = dist
            else:
                raise ValueError("Invalid policy output")
            action = action.clamp(*self.action_range)

        return action
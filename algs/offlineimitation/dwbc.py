import itertools
from typing import Dict, Type, Optional, Tuple, List, Union, Any

import numpy as np
import torch
import gym

from graph_offline_imitation.processors.base            import Processor, IdentityProcessor
from graph_offline_imitation.networks.dwbc              import DWBCNetwork
from graph_offline_imitation.algs.offlineimitation.base import OfflineImitation
from graph_offline_imitation.utils                      import utils


class DWBCOfflineImitation(OfflineImitation):
    def __init__(
        self,
        *args,
        alpha:              float    = 7.5,
        eta:                float    = 1.5,
        d_update_num:       int      = 100,
        log_pi_norm_min:    float    = -20.,
        log_pi_norm_max:    float    = 10.,
        encoder_gradients:  str      = "null",
        **kwargs,
    ) -> None:
        # After determining dimension parameters, setup the network
        super().__init__(*args, **kwargs)
        assert encoder_gradients in ("actor", "critic", "both", 'null')
        self.encoder_gradients   = encoder_gradients        
        self.alpha               = alpha
        self.eta                 = eta
        self.d_update_num        = d_update_num
        self.log_pi_norm_min     = log_pi_norm_min
        self.log_pi_norm_max     = log_pi_norm_max
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
        assert network_class is DWBCNetwork, "Must use DWBCNetwork with DWBC."
        self.network = network_class(
            observation_space           =   self.env.observation_space,
            action_space                =   self.env.action_space,
            **network_kwargs,
        ).to(self.device)
        
    def setup_optimizers(self) -> None:
        assert self.encoder_gradients == 'null'
        actor_params                    = self.network.actor.parameters()
        discriminator_params            = self.network.discriminator.parameters()
        self.optim["actor"]             = self.optim_class(actor_params, **self.optim_kwargs)
        self.optim['discriminator']     = self.optim_class(discriminator_params, **self.optim_kwargs)

    def train_step(self, expert_batch: Dict, unlabel_batch: Dict, step: int, total_steps: int) -> Dict:
        # format batch data (e.g., obs goal concat)
        exp_obs, exp_act            = expert_batch['obs'], expert_batch['action']
        unl_obs, unl_act            = unlabel_batch['obs'], unlabel_batch['action']

        exp_obs, unl_obs            = self.network.encoder(exp_obs), self.network.encoder(unl_obs)

        # get log prob
        dist_exp, dist_unl          = self.network.actor(exp_obs), self.network.actor(unl_obs)
        log_pi_exp, log_pi_unl      = dist_exp.log_prob(exp_act), dist_unl.log_prob(unl_act)

        # update discriminator
        log_pi_exp_clip             = torch.clip(log_pi_exp, self.log_pi_norm_min, self.log_pi_norm_max)
        log_pi_unl_clip             = torch.clip(log_pi_unl, self.log_pi_norm_min, self.log_pi_norm_max)

        log_pi_exp_norm             = (log_pi_exp_clip - self.log_pi_norm_min) / (self.log_pi_norm_max - self.log_pi_norm_min)
        log_pi_unl_norm             = (log_pi_unl_clip - self.log_pi_norm_min) / (self.log_pi_norm_max - self.log_pi_norm_min)

        d_e                         = self.network.discriminator(exp_obs, exp_act, log_pi_exp_norm.detach())
        d_u                         = self.network.discriminator(unl_obs, unl_act, log_pi_unl_norm.detach())

        d_loss_e                    = - torch.log(d_e)
        d_loss_u                    = - torch.log(1 - d_u) / self.eta + torch.log(1 - d_e)
        d_loss                      = torch.mean(d_loss_e + d_loss_u)

        if step % self.d_update_num == 0:
            self.optim['discriminator'].zero_grad()
            d_loss.backward()
            self.optim['discriminator'].step()

        # update policy
        d_e_clip = torch.squeeze(d_e).detach()
        d_u_clip = torch.squeeze(d_u).detach()
        d_u_clip[d_u_clip < 0.5] = 0.0

        bc_loss     = -torch.sum(log_pi_exp, 1)
        corr_loss_e = -torch.sum(log_pi_exp, 1) * (self.eta / (d_e_clip * (1.0 - d_e_clip)) + 1.0)
        corr_loss_u = -torch.sum(log_pi_unl, 1) * (1.0 / (1.0 - d_u_clip) - 1.0)
        p_loss      = self.alpha * torch.mean(bc_loss) - torch.mean(corr_loss_e) + torch.mean(corr_loss_u)
        self.optim['actor'].zero_grad()
        p_loss.backward()
        self.optim['actor'].step()

        return dict(
            loss_policy             = p_loss.item(),
            loss_policy_bc          = bc_loss.mean().item(),
            loss_policy_corr_e      = corr_loss_e.mean().item(),
            loss_policy_corr_u      = corr_loss_u.mean().item(),

            loss_discriminator      = d_loss.item(),
            loss_discriminator_e    = d_loss_e.mean().item(),
            loss_discriminator_u    = d_loss_u.mean().item(),
            discriminator_e         = d_e.mean().item(),
            discriminator_u         = d_u.mean().item()
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
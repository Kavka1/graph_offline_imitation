import itertools
from typing import Dict, Type, Optional, Tuple, List, Union, Any

import numpy as np
import torch
import gym

from graph_offline_imitation.processors.base            import Processor, IdentityProcessor
from graph_offline_imitation.networks.oril              import ORILNetwork
from graph_offline_imitation.algs.offlineimitation.base import OfflineImitation
from graph_offline_imitation.utils                      import utils


class ORILOfflineImitation(OfflineImitation):
    def __init__(
        self,
        *args,
        gamma:                  float   = 0.99,
        tau:                    float   = 0.005,
        policy_noise:           float   = 0.2,
        noise_clip:             float   = 0.5,
        policy_freq:            float   = 2,
        alpha:                  float   = 2.5,
        encoder_gradients:      str     = "null",
        **kwargs,
    ) -> None:
        # After determining dimension parameters, setup the network
        super().__init__(*args, **kwargs)
        assert encoder_gradients in ("actor", "critic", "both", 'null')
        
        self.gamma                  = gamma
        self.tau                    = tau
        self.policy_noise           = policy_noise
        self.noise_clip             = noise_clip
        self.policy_freq            = policy_freq
        self.alpha                  = alpha
        self.encoder_gradients      = encoder_gradients  
        self.last_actor_loss        = 0.      
        self.action_range           = [
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

    def setup_datasets(self, env: gym.Env, total_steps: int):
        if hasattr(self, 'expert_dataset') and hasattr(self, 'unlabel_dataset'):
            print("setup_datasets called twice! We skip it.")
            pass
        else:
            # Setup the expert dataset & unlabel
            self.expert_dataset, self.unlabel_dataset = self.dataset_class(self.env.observation_space, self.env.action_space, **self.dataset_kwargs)

        if hasattr(self, 'validation_dataset'):
            print("setup_datasets called twice! We skip it.")
            pass
        else:
            self.validation_dataset = None
            print("[warning] No validation dataset setting for current algorithm")
        
        # set env_step as offline version
        self.env_step = self._empty_step

    def setup_network(self, network_class: Type[torch.nn.Module], network_kwargs: Dict) -> None:
        assert network_class is ORILNetwork, "Must use ORILNetwork with ORIL."
        self.network = network_class(
            observation_space           =   self.env.observation_space,
            action_space                =   self.env.action_space,
            **network_kwargs,
        ).to(self.device)
        
    def setup_optimizers(self) -> None:
        assert self.encoder_gradients == 'null'
        actor_params                    = self.network.actor.parameters()
        critic_params                   = self.network.critic.parameters()
        discriminator_params            = self.network.discriminator.parameters()
        self.optim["actor"]             = self.optim_class(actor_params, **self.optim_kwargs)
        self.optim['critic']            = self.optim_class(critic_params, **self.optim_kwargs)
        self.optim['discriminator']     = self.optim_class(discriminator_params, **self.optim_kwargs)

    def train_discriminator_step(self, expert_batch: Dict, unlabel_batch: Dict) -> Dict:
        exp_obs, exp_act        =   expert_batch['obs'], expert_batch['action']
        unl_obs, unl_act        =   unlabel_batch['obs'], unlabel_batch['action']
        loss_dict               =   self.network.discriminator.loss(exp_obs, exp_act, unl_obs, unl_act)
        loss_discriminator      =   loss_dict['loss']
        loss_disc_ce            =   loss_dict['ce_loss']
        disc_grad_pen           =   loss_dict['grad_pen']
        self.optim['discriminator'].zero_grad()
        loss_discriminator.backward()
        self.optim['discriminator'].step()
        return {
            'loss_discriminator':   loss_discriminator.item(),
            'loss_disc_ce':         loss_disc_ce.item(),
            'discriminator_grad_pen': disc_grad_pen.item()
        }

    def train_step(self, expert_batch: Dict, unlabel_batch: Dict, step: int, total_steps: int) -> Dict:
        # SMODice only use offline/unlabel dataset for policy/value training
        obs         =   torch.concat([unlabel_batch['obs'], expert_batch['obs']], dim=0)
        act         =   torch.concat([unlabel_batch['action'], expert_batch['action']], dim=0)
        next_obs    =   torch.concat([unlabel_batch['next_obs'], expert_batch['next_obs']], dim=0)
        terminal    =   torch.concat([unlabel_batch['done'], expert_batch['done']], dim=0)

        obs, next_obs = self.network.encoder(obs), self.network.encoder(next_obs)

        # compute reward via the discriminator
        with torch.no_grad():
            rewards     = self.network.discriminator.predict_reward(obs, act).detach()

        # update critic
        with torch.no_grad():
            noise       = (torch.randn_like(act) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_dist   = self.network.actor_target(next_obs)
            next_action = next_dist.sample()
            next_action = (next_action + noise).clamp(self.action_range[0], self.action_range[1])
            target_qs   = self.network.critic_target(next_obs, next_action)
            target_q    = torch.min(target_qs, dim=0).values
            target_td   = rewards.squeeze(-1) + (1 - terminal.type(torch.float32)) * self.gamma * target_q
        qs              = self.network.critic(obs, act)
        loss_critic     = torch.nn.functional.mse_loss(qs, target_td.unsqueeze(0))
        self.optim['critic'].zero_grad()
        loss_critic.backward()
        self.optim['critic'].step()

        # update actor
        if step % self.policy_freq == 0:
            dist        =   self.network.actor(obs)
            new_act     =   dist.rsample()
            qs_new      =   self.network.critic(obs, new_act)[0]    # Q1 [B,]
            lmbda       =   self.alpha / qs_new.abs().mean().detach()
            loss_actor  =   - lmbda * qs_new.mean() + torch.nn.functional.mse_loss(new_act, act)
            self.optim['actor'].zero_grad()
            loss_actor.backward()
            self.optim['actor'].step()
            self.last_actor_loss = loss_actor.item()

            # update target networks
            with torch.no_grad():
                for param, target_param in zip(self.network.actor.parameters(), self.network.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.network.critic.parameters(), self.network.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return dict(
            loss_policy             = self.last_actor_loss,
            loss_critic             = loss_critic.item(),
            target_qs               = target_td.mean().item(),
            cur_qs                  = qs.mean().item()
        )
    
    def _predict(self, batch: Dict, sample: bool = False) -> torch.Tensor:
        with torch.no_grad():
            obs             = self.network.format_actor_input(batch["obs"])
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
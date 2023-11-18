import itertools
from typing import Dict, Type, Optional, Tuple, List, Union, Any

import numpy as np
import torch
import gym

from graph_offline_imitation.processors.base            import Processor, IdentityProcessor
from graph_offline_imitation.networks.smodice           import SMODICENetwork
from graph_offline_imitation.algs.off_policy_algorithm  import OffPolicyAlgorithm
from graph_offline_imitation.utils                      import utils


class SMODiceOfflineImitation(OffPolicyAlgorithm):
    def __init__(
        self,
        *args,
        gamma:                  float   = 0.99,
        use_entropy_constraint: bool    = True,
        target_entropy:         float   = None,
        f_func:                 str     = 'chi',
        v_l2_reg:               float   = 0.0001,
        encoder_gradients:      str     = "null",
        **kwargs,
    ) -> None:
        # After determining dimension parameters, setup the network
        super().__init__(*args, **kwargs)
        assert encoder_gradients in ("actor", "critic", "both", 'null')
        assert f_func in ['chi', 'kl']
        self.gamma                  = gamma
        self.use_entropy_constraint = use_entropy_constraint
        if target_entropy is None:
            self.target_entropy     = -np.prod(self.env.action_space.shape)
        else:
            self.target_entropy     = target_entropy
        self.encoder_gradients      = encoder_gradients        
        self.v_l2_reg               = v_l2_reg
        self.action_range        = [
            float(self.expert_processor.action_space.low.min()),
            float(self.expert_processor.action_space.high.max()),
        ]

        self.f_func             =   f_func
        if self.f_func == 'chi':
            self._f_func        =   lambda x: 0.5 * (x - 1) ** 2
            self._f_star_prime  =   lambda x: torch.relu(x + 1)
            self._f_star        =   lambda x: 0.5 * x ** 2 + x
        elif self.f_func == 'kl':
            self._f_func        =   lambda x: x * torch.log(x + 1e-10)
            self._f_star_prime  =   lambda x: torch.exp(x - 1)
        else:
            raise ValueError
        
        if self.use_entropy_constraint:
            self._log_ent_coef  =   torch.zeros(1, requires_grad=True, device=self.device)


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
        """
        Called after everything else has been setup, right before training starts
        This is _only_ called by the trainer and is not called by default.
        This function is responsible for creating the following attributes:
            self.dataset (required)
            self.validation_dataset
        """
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
        assert network_class is SMODICENetwork, "Must use SMODICENetwork with SMODICE."
        self.network = network_class(
            observation_space           =   self.env.observation_space,
            action_space                =   self.env.action_space,
            **network_kwargs,
        ).to(self.device)
        
    def setup_optimizers(self) -> None:
        assert self.encoder_gradients == 'null'
        actor_params                    = self.network.actor.parameters()
        value_params                    = self.network.value.parameters()
        discriminator_params            = self.network.discriminator.parameters()
        self.optim["actor"]             = self.optim_class(actor_params, **self.optim_kwargs)
        self.optim['value']             = self.optim_class(value_params, weight_decay=self.v_l2_reg, **self.optim_kwargs)
        self.optim['discriminator']     = self.optim_class(discriminator_params, **self.optim_kwargs)
        if self.use_entropy_constraint:
            self.optim['entropy']       = self.optim_class([self._log_ent_coef], **self.optim_kwargs)

    def format_expert_batch(self, batch: Any) -> Any:
        if not utils.all_tensors(batch):
            batch = utils.to_tensor(batch)
        if self.expert_processor.supports_gpu:
            batch = utils.to_device(batch, self.device)
            batch = self.expert_processor(batch)
        else:
            batch = self.expert_processor(batch)
            batch = utils.to_device(batch, self.device)
        return batch

    def format_unlabel_batch(self, batch: Any) -> Any:
        if not utils.all_tensors(batch):
            batch = utils.to_tensor(batch)
        if self.unlabel_processor.supports_gpu:
            batch = utils.to_device(batch, self.device)
            batch = self.unlabel_processor(batch)
        else:
            batch = self.unlabel_processor(batch)
            batch = utils.to_device(batch, self.device)
        return batch

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
        init_obs    =   unlabel_batch['initial_obs']
        obs         =   unlabel_batch['obs']
        act         =   unlabel_batch['action']
        next_obs    =   unlabel_batch['next_obs']
        terminal    =   unlabel_batch['done']

        obs, next_obs = self.network.encoder(obs), self.network.encoder(next_obs)

        # compute reward via the discriminator
        with torch.no_grad():
            rewards     = self.network.discriminator.predict_reward(obs, act).detach()

        # update value function
        initial_v       =   self.network.value(init_obs)
        cur_v           =   self.network.value(obs)
        next_v          =   self.network.value(next_obs)
        
        e_v             =   rewards + (1 - terminal.type(torch.float32)) * self.gamma * next_v - cur_v
        v_loss_0        =   (1 - self.gamma) * initial_v
        if self.f_func == 'kl':
            v_loss_1    =   torch.log(torch.mean(torch.exp(e_v)))
        elif self.f_func == 'chi':
            v_loss_1    =   torch.mean(self._f_star(e_v))
        else:
            raise ValueError
        loss_value      =   (v_loss_0 + v_loss_1).mean()
        self.optim['value'].zero_grad()
        loss_value.backward()
        self.optim['value'].step()

        # update policy
        if self.f_func == 'kl':
            w_e =   torch.exp(e_v)
        else:
            w_e =   self._f_star_prime(e_v)
        dist    =   self.network.actor(obs)

        bc_prob     =   dist.log_prob(act).sum(dim=-1)
        loss_policy =   - torch.mean(w_e.detach() * bc_prob)

        new_act         =   dist.rsample()
        new_act_log_prob=   dist.log_prob(new_act)
        neg_entropy     =   dist.log_prob(new_act).mean(dim=-1)
        if self.use_entropy_constraint:
            ent_coef    =   torch.exp(self._log_ent_coef).squeeze(0)
            loss_policy +=  ent_coef * neg_entropy.mean()
            # update entropy
            ent_loss    =   - self._log_ent_coef[0] * (new_act_log_prob + self.target_entropy).mean().detach()
            self.optim['entropy'].zero_grad()
            ent_loss.backward()
            self.optim['entropy'].step()

        self.optim['actor'].zero_grad()
        loss_policy.backward()
        self.optim['actor'].step()

        return dict(
            loss_policy             = loss_policy.item(),
            bc_weight               = w_e.mean().item(),
            neg_entropy             = neg_entropy.mean().item(),

            ent_coef                = ent_coef.clone().item(),

            loss_value              = loss_value.item(),
            loss_value_0            = v_loss_0.mean().item(),
            loss_value_1            = v_loss_1.mean().item(),
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

    def _get_train_action(self, step: int, total_steps: int) -> np.ndarray:
        batch = dict(obs=self._current_obs)
        with torch.no_grad():
            action = self.predict(batch, is_batched=False, sample=True)
        return action
    
    def predict(self, batch: Any, is_batched: bool = False, **kwargs) -> Any:
        is_np = not utils.contains_tensors(batch)
        if not is_batched:
            batch = utils.unsqueeze(batch, 0)
        batch   = self.format_expert_batch(batch)   # org: format batch
        pred    = self._predict(batch, **kwargs)
        if not is_batched:
            pred = utils.get_from_batch(pred, 0)
        if is_np:
            pred = utils.to_np(pred)
        return pred
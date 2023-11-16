import itertools
from typing import Dict, Type, Optional, Tuple, List, Union, Any

import numpy as np
import torch
import gym

from graph_offline_imitation.processors.base            import Processor, IdentityProcessor
from graph_offline_imitation.networks.contrastiveoi     import ContrastiveOfflineImitationNetwork
from graph_offline_imitation.algs.off_policy_algorithm  import OffPolicyAlgorithm
from graph_offline_imitation.utils                      import utils


class ContrastiveOfflineImitation(OffPolicyAlgorithm):
    def __init__(
        self,
        *args,
        expert_bc_coef:     float    = 0.5,
        repr_penalty_coef:  float    = 1,
        adv_temperature:    float    = 1,
        adv_clip:           float    = 100,
        sparse_exp_adv_lb:  float    = None,
        encoder_gradients:  str      = "both",
        **kwargs,
    ) -> None:
        # After determining dimension parameters, setup the network
        super().__init__(*args, **kwargs)
        assert encoder_gradients in ("actor", "critic", "both")
        self.encoder_gradients   = encoder_gradients        
        self.expert_bc_coef             =   expert_bc_coef
        self.repr_penalty_coef          =   repr_penalty_coef
        self.adv_temperature            =   adv_temperature
        self.adv_clip                   =   adv_clip
        self.sparse_exp_adv_lb          =   sparse_exp_adv_lb
        self.action_range        = [
            float(self.expert_processor.action_space.low.min()),
            float(self.expert_processor.action_space.high.max()),
        ]

    def setup_processor(self, processor_class: Optional[Type[Processor]], processor_kwargs: Dict) -> None:
        if isinstance(self.env.observation_space, gym.spaces.Box):
            obs_low, obs_high           = self.env.observation_space.low, self.env.observation_space.high
            expert_observation_space    = self.env.observation_space
            unlabel_observation_space   = gym.spaces.Dict(
                {
                    "observation": gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32),
                    "achieved_goal": gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32),
                    "desired_goal": gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32),
                }
            )
        elif isinstance(self.env.observation_space, gym.space.Dict):
            assert 'observation' in self.env.observation_space.keys()
            obs_low, obs_high           = self.env.observation_space['observation'].low, self.env.observation_space['observation'].high
            expert_observation_space    = gym.space.Box(low = obs_low, high = obs_high, dtype=np.float32)
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
        assert network_class is ContrastiveOfflineImitationNetwork, "Must use ContrastiveOfflineImitationNetwork with ContrastiveRL."
        self.network = network_class(
            unlabel_observation_space   =   self.unlabel_processor.observation_space,
            expert_observation_space    =   self.expert_processor.observation_space,
            action_space                =   self.env.action_space,
            **network_kwargs,
        ).to(self.device)
        
    def setup_optimizers(self) -> None:
        if self.encoder_gradients == "critic" or self.encoder_gradients == "both":
            vfunc_params    = itertools.chain(self.network.vfunc.parameters(), self.network.obs_encoder.parameters(), self.network.goal_encoder.parameters())
            qfunc_params    = itertools.chain(self.network.qfunc.parameters(), self.network.obs_encoder.parameters(), self.network.goal_encoder.parameters())
            policy_params   = self.network.policy.parameters()
        elif self.encoder_gradients == "actor":
            vfunc_params    = self.network.vfunc.parameters()
            qfunc_params    = self.network.qfunc.parameters()
            policy_params   = itertools.chain(self.network.policy.parameters(), self.network.obs_encoder.parameters(), self.network.goal_encoder.parameters())
        else:
            raise ValueError("Unsupported value of encoder_gradients")
        self.optim["policy"]    = self.optim_class(policy_params, **self.optim_kwargs)
        self.optim['vfunc']     = self.optim_class(vfunc_params, **self.optim_kwargs)
        self.optim["qfunc"]     = self.optim_class(qfunc_params, **self.optim_kwargs)

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

    def train_step(self, expert_batch: Dict, unlabel_batch: Dict, step: int, total_steps: int) -> Dict:
        # format batch data (e.g., obs goal concat)
        exp_obs, exp_act            = expert_batch['obs'], expert_batch['action']
        unl_obs, unl_act            = unlabel_batch['obs']['observation'], unlabel_batch['action']
        unl_goal                    = unlabel_batch['obs']['desired_goal']

        exp_obs, unl_obs            = self.network.obs_encoder(exp_obs), self.network.obs_encoder(unl_obs)
        unl_goal                    = self.network.goal_encoder(unl_goal)

        # update v func
        logits_v, s_repr_v, g_repr_v  = self.network.vfunc(unl_obs, unl_goal, return_repr=True)
        I                   = torch.eye(unl_obs.shape[0]).expand_as(logits_v).to(logits_v.device)
        loss_fn_v           = lambda lg, label: torch.nn.functional.cross_entropy(lg, label)

        s_repr_v            = torch.norm(s_repr_v, p=2, dim=-1, keepdim=False).mean()
        g_repr_v            = torch.norm(g_repr_v, p=2, dim=-1, keepdim=False).mean()

        vfunc_loss          = torch.vmap(loss_fn_v, in_dims=0, out_dims=0)(logits_v, I)
        vfunc_loss          = torch.mean(vfunc_loss) + self.repr_penalty_coef * s_repr_v + self.repr_penalty_coef * g_repr_v
        self.optim['vfunc'].zero_grad(set_to_none=True)
        vfunc_loss.backward()
        self.optim['vfunc'].step()

        # update q func
        logits_q, sa_repr_q, g_repr_q = self.network.qfunc(unl_obs, unl_act, unl_goal, return_repr=True)
        I               = torch.eye(unl_obs.shape[0]).expand_as(logits_q).to(logits_q.device)
        loss_fn_q       = lambda lg, label: torch.nn.functional.cross_entropy(lg, label)

        sa_repr_q       = torch.norm(sa_repr_q, p=2, dim=-1, keepdim=False).mean()
        g_repr_q        = torch.norm(g_repr_q, p=2, dim=-1, keepdim=False).mean()

        qfunc_loss      = torch.vmap(loss_fn_q, in_dims=0, out_dims=0)(logits_q, I)
        qfunc_loss      = torch.mean(qfunc_loss) + self.repr_penalty_coef * sa_repr_q + self.repr_penalty_coef * g_repr_q
        self.optim['qfunc'].zero_grad(set_to_none=True)
        qfunc_loss.backward()
        self.optim['qfunc'].step()

        # update policy
        dist_exp        = self.network.policy(exp_obs)
        bc_loss_exp     = - dist_exp.log_prob(exp_act).mean(dim=-1)

        with torch.no_grad():
            exp_proximity_sa             =   self.network.qfunc(unl_obs, unl_act, exp_obs)              #   [E, B_unl, B_exp]
            exp_proximity_sa_cons        =   torch.min(exp_proximity_sa, dim=0, keepdim=False).values   #   [B_unl, B_exp]
            exp_proximity_sa_cons_avg    =   exp_proximity_sa_cons.mean(dim=-1, keepdim=False)          #   [B_unl]

            exp_proximity_s              =   self.network.vfunc(unl_obs, exp_obs)                
            exp_proximity_s_cons         =   torch.min(exp_proximity_s, dim=0, keepdim=False).values
            exp_proximity_s_cons_avg     =   exp_proximity_s_cons.mean(dim=-1, keepdim=False)

            exp_proximity_adv            =   exp_proximity_sa_cons_avg - exp_proximity_s_cons_avg       #   [B_unl]
            exponential_adv              =   torch.exp(exp_proximity_adv / self.adv_temperature)
            exponential_adv_clip         =   torch.clamp(exponential_adv, max=self.adv_clip)

            if self.sparse_exp_adv_lb is not None:
                gate_adv                 =  torch.tensor(exponential_adv_clip.clone().detach() > self.sparse_exp_adv_lb, dtype=torch.float32).to(exponential_adv.device)
            else:
                gate_adv                 =  torch.ones_like(exponential_adv_clip, dtype=torch.float32).to(exponential_adv.device)

        dist_unl        = self.network.policy(unl_obs)
        bc_loss_unl     = - gate_adv * exponential_adv_clip * dist_unl.log_prob(unl_act).mean(dim=-1)

        policy_loss     = self.expert_bc_coef * bc_loss_exp.mean() + (1 - self.expert_bc_coef) * bc_loss_unl.mean()
        self.optim['policy'].zero_grad(set_to_none=True)
        policy_loss.backward()
        self.optim['policy'].step()

        # v func statistics
        mean_logits_v           =   torch.mean(logits_v, dim=0, keepdim=False)  # [B, B]
        correct_v               =   torch.argmax(mean_logits_v, dim=1) == torch.argmax(I, dim=1)
        binary_accuracy_v       =   torch.mean(((mean_logits_v > 0) == I).type(torch.float32)).item()
        categorical_accuracy_v  =   torch.mean(correct_v.type(torch.float32)).item()
        logits_pos_v            =   (torch.sum(mean_logits_v * I) / torch.sum(I)).item()
        logits_neg_v            =   (torch.sum(mean_logits_v * (1 - I)) / torch.sum(1 - I)).item()
        logsumexp_v             =   (torch.logsumexp(mean_logits_v[:, :], dim=1) ** 2).mean().item()
        # q func statistics
        mean_logits_q           =   torch.mean(logits_q, dim=0, keepdim=False)
        correct_q               =   torch.argmax(mean_logits_q, dim=1) == torch.argmax(I, dim=1)
        binary_accuracy_q       =   torch.mean(((mean_logits_q > 0) == I).type(torch.float32)).item()
        categorical_accuracy_q  =   torch.mean(correct_q.type(torch.float32)).item()
        logits_pos_q            =   (torch.sum(mean_logits_q * I) / torch.sum(I)).item()
        logits_neg_q            =   (torch.sum(mean_logits_q * (1 - I)) / torch.sum(1 - I)).item()
        logsumexp_q             =   (torch.logsumexp(mean_logits_q[:, :], dim=1) ** 2).mean().item()
        # policy statistics


        return dict(
            loss_q                  = qfunc_loss.item(),
            s_repr_v                = s_repr_v.item(),
            g_repr_v                = g_repr_v.item(),
            binary_accuracy_q       = binary_accuracy_q,
            categorical_accuracy_q  = categorical_accuracy_q,
            logits_pos_q            = logits_pos_q,
            logits_neg_q            = logits_neg_q,
            logsumexp_q             = logsumexp_q,

            loss_v                  = vfunc_loss.item(),
            sa_repr_q               = sa_repr_q.item(),
            g_repr_q                = g_repr_q.item(),
            binary_accuracy_v       = binary_accuracy_v,
            categorical_accuracy_v  = categorical_accuracy_v,
            logits_pos_v            = logits_pos_v,
            logits_neg_v            = logits_neg_v,
            logsumexp_v             = logsumexp_v,

            loss_policy             = policy_loss.item(),
            loss_policy_exp_bc      = bc_loss_exp.mean().item(),
            loss_policy_unl_bc      = bc_loss_unl.mean().item(),
            exp_proximity_sa_mean   = exp_proximity_sa_cons_avg.mean().item(),
            exp_proximity_sa_max    = exp_proximity_sa_cons_avg.max().item(),
            exp_proximity_sa_min    = exp_proximity_sa_cons_avg.min().item(),
            exp_proximity_s_mean    = exp_proximity_s_cons_avg.mean().item(),
            exp_proximity_s_max     = exp_proximity_s_cons_avg.max().item(),
            exp_proximity_s_min     = exp_proximity_s_cons_avg.min().item(),
            
            exponential_adv_mean    = exponential_adv.mean().item(),
            exponential_adv_max     = exponential_adv.max().item(),
            exponential_adv_min     = exponential_adv.min().item(),

            exponential_adv_clip_mean    = exponential_adv_clip.mean().item(),
            exponential_adv_clip_max     = exponential_adv_clip.max().item(),
            exponential_adv_clip_min     = exponential_adv_clip.min().item(),

            exponential_adv_gate_mean    = gate_adv.mean().item()
        )

    def _predict(self, batch: Dict, sample: bool = False) -> torch.Tensor:
        with torch.no_grad():
            obs             = self.network.format_policy_obs(batch["obs"])
            z               = self.network.obs_encoder(obs)
            dist            = self.network.policy(z)
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
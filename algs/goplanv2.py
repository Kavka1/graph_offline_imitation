from typing import Any, List, Dict, Tuple, Type, Union, Optional
import gym
import numpy as np
import torch
import os, yaml, importlib

from graph_offline_imitation.networks.base             import ActorCriticPolicy, ModuleContainer
from graph_offline_imitation.algs.off_policy_algorithm import OffPolicyAlgorithm
from graph_offline_imitation.algs.sac                  import SAC
from graph_offline_imitation.algs.goplan               import load_pretrain_alg, _parse_helper
from graph_offline_imitation.datasets.replay_buffer    import ReplayBuffer
# from graph_offline_imitation.utils.config              import Config
from graph_offline_imitation.utils.utils               import to_device, to_tensor



class GoPlanV2(SAC):
    def __init__(
        self, 
        *args, 
        tau: float = 0.005, 
        init_temperature: float = 0.1, 
        critic_freq: int = 1, 
        actor_freq: int = 1, 
        target_freq: int = 2, 
        bc_coeff=0, 

        offline_goal_sample_batch_size: int     = 512,
        goal_policy_constraint_coef: float      = 1.,

        offline_module_path: str                = None,
        offline_goal_path: str                  = None,
        online_demo_path: str                   = None,

        **kwargs
    ) -> None:
        super().__init__(
            *args, 
            tau=tau, 
            init_temperature=init_temperature, 
            critic_freq=critic_freq, 
            actor_freq=actor_freq, 
            target_freq=target_freq, 
            bc_coeff=bc_coeff, 
            **kwargs
        )

        self.offline_goal_sample_batch_size = offline_goal_sample_batch_size
        self.goal_policy_constraint_coef    = goal_policy_constraint_coef
        self.temperature                    = 1.

        self._offline_module_path = offline_module_path
        self._offline_goal_path   = offline_goal_path
        self._online_demo_path    = online_demo_path
        self._load_required_module()

    def _load_required_module(self) -> None:
        assert self._offline_module_path is not None and self._offline_goal_path is not None and self._online_demo_path is not None
        # load offline goal-cond policy / value
        # config               = Config.load(os.path.dirname(self._offline_module_path))
        # config["checkpoint"] = None  # Set checkpoint to None
        # model                = config.get_model()
        # metadata             = model.load(self._offline_module_path)
        model                = load_pretrain_alg(self._offline_module_path)
        self.offline_network = model.network
        # load offline goal distribution
        # model.setup_datasets(model.env, total_steps=0)
        # self.offline_dataset = model.dataset
        del model

        offline_obs_dataset = np.load(os.path.join(self._offline_goal_path, "observations_seq.npy")).astype(np.float32)  # Shape (T, B, D)
        offline_mask        = np.load(os.path.join(self._offline_goal_path, "existence_mask.npy"))  # Shape (T, B)
        offline_obs_dataset = offline_obs_dataset.transpose((1, 0, 2))  # (B, T, D)
        end_points          = offline_mask.sum(axis=0).astype(np.int64)
        self.offline_obs_dataset = []
        for i in range(len(offline_mask)):
            end = end_points[i]
            if end < 3:
                continue  # skip this episode
            self.offline_obs_dataset.append(
                offline_obs_dataset[i, :end, :30].reshape((-1, 30))
            )
        self.offline_obs_dataset = np.concatenate(self.offline_obs_dataset, axis=0)

        # load online demo
        self.online_demo     = np.load(self._online_demo_path)
        online_demo_obs      = self.online_demo['obs_seq']
        if isinstance(online_demo_obs, List):
            self.online_demo_obs = torch.as_tensor(np.stack(online_demo_obs, 0)).float().to(self.device)
        elif isinstance(online_demo_obs, np.ndarray):
            self.online_demo_obs = torch.as_tensor(online_demo_obs).float().to(self.device)
        else:
            raise ValueError
    
    def _measure_distance(self, obs_with_goal: torch.Tensor) -> torch.Tensor:
        logits          = self.offline_network.value(obs_with_goal)[0] # [1, B, D] -> [B, D]
        distribution    = torch.nn.functional.softmax(logits, dim=-1)  # (B, D)
        distances       = torch.arange(start=0, end=logits.shape[-1], device=self.device) / logits.shape[-1]
        distances       = distances.unsqueeze(0)  # (B, D)
        if self.temperature is None:
            # Return the expectation
            predicted_distance  = (distribution * distances).sum(dim=-1)
        else:
            # Return the LSE weighted by the distribution.
            exp_q               = torch.exp(-distances / self.temperature)
            predicted_distance  = -self.temperature * torch.log(torch.sum(distribution * exp_q, dim=-1))
        return predicted_distance

    def _update_actor_and_alpha(self, batch: Dict) -> Dict:
        obs         = batch["obs"].detach()  # Detach the encoder so it isn't updated. [B, O]

        dist        = self.network.actor(obs)
        action      = dist.rsample()
        log_prob    = dist.log_prob(action)
        qs          = self.network.critic(obs, action)
        q           = torch.min(qs, dim=0)[0]

        with torch.no_grad():
            n_obs, n_goal, n_demo= obs.shape[0], self.offline_goal_sample_batch_size, self.online_demo_obs.shape[0]

            offline_goal_idxs    = np.random.randint(0, len(self.offline_obs_dataset), size=self.offline_goal_sample_batch_size)
            offline_goals        = self.offline_obs_dataset[offline_goal_idxs]
            offline_goals        = to_device(to_tensor(offline_goals), self.device)         # [B_g, O]
            # obs -> goal candidates
            cur_obs_offgoal      = torch.concat((
                obs.reshape(1, n_obs, obs.shape[-1]).expand((n_goal, n_obs, obs.shape[-1])),
                offline_goals.reshape(n_goal, 1, offline_goals.shape[-1]).expand((n_goal, n_obs, offline_goals.shape[-1]))
            ), dim=-1)
            curobs2offgoal_dist  = self._measure_distance(cur_obs_offgoal)                  # [B_g, B_o]
            # goal candidates -> demo obs
            offgoal_demoobs      = torch.concat((
                offline_goals.reshape(n_goal, 1, offline_goals.shape[-1]).expand((n_goal, n_demo, offline_goals.shape[-1])),
                self.online_demo_obs.reshape(1, n_demo, obs.shape[-1]).expand((n_goal, n_demo, obs.shape[-1])),
            ), dim=-1)
            offgoal2demoobs_dist = self._measure_distance(offgoal_demoobs)                  # [B_g, B_demo]
            offgoal2demo_dist    = torch.mean(offgoal2demoobs_dist, dim=-1, keepdim=True)   # [B_g, 1]
            # mix distance
            mix_dist             = curobs2offgoal_dist + offgoal2demo_dist                  # [B_g, B_o]
            # choose goal
            chosen_goal_idx      = mix_dist.argmax(dim=0, keepdim=False)                    # [B_o]
            chosen_goals         = offline_goals[chosen_goal_idx]                           # [B_o, O]
            # concat obs with chosen goals and inference
            obs_chosen_goal      = torch.concat((obs, chosen_goals), dim=-1)
            goal_policy_action   = self.offline_network.actor(obs_chosen_goal)              # [B_o, A]
        
        # constraint_loss          = dist.log_prob(goal_policy_action).clamp_(min=-1e4)
        constraint_loss          = torch.nn.functional.mse_loss(action, goal_policy_action)

        actor_loss  = (self.alpha.detach() * log_prob - q).mean() + self.goal_policy_constraint_coef * constraint_loss
        if self.bc_coeff > 0.0:
            bc_loss     = -dist.log_prob(batch["action"]).mean()  # Simple NLL loss.
            actor_loss  = actor_loss + self.bc_coeff * bc_loss

        self.optim["actor"].zero_grad(set_to_none=True)
        actor_loss.backward()
        self.optim["actor"].step()
        entropy = -log_prob.mean()

        # Update the learned temperature
        self.optim["log_alpha"].zero_grad(set_to_none=True)
        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.optim["log_alpha"].step()

        return dict(
            actor_loss=actor_loss.item(),
            constraint_loss=constraint_loss.item(),
            entropy=entropy.item(),
            alpha_loss=alpha_loss.item(),
            alpha=self.alpha.detach().item(),
        )

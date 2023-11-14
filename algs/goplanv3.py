from typing import Any, List, Dict, Tuple, Type, Union, Optional
import gym
import numpy as np
import torch
import os, yaml, importlib, itertools

from graph_offline_imitation.networks.base             import ActorCriticValuePolicy, ModuleContainer
from graph_offline_imitation.algs.off_policy_algorithm import OffPolicyAlgorithm
from graph_offline_imitation.algs.goplan               import _parse_helper, load_pretrain_alg
from graph_offline_imitation.algs.sac                  import SAC
from graph_offline_imitation.datasets.replay_buffer    import ReplayBuffer
# from graph_offline_imitation.utils.config              import Config
from graph_offline_imitation.utils.utils               import to_device, to_tensor




class GoPlanV3(OffPolicyAlgorithm):
    def __init__(
        self, 
        *args, 

        tau: float = 0.005, 
        beta: float = 1,
        asy_tau: float = 0.7,
        exp_adv_max: float = 1,

        critic_freq: int = 1, 
        value_freq: int = 1,
        actor_freq: int = 1, 
        target_freq: int = 2, 

        offline_goal_sample_batch_size: int     = 512,
        offline_goal_feasible_quantile: float   = 0.5,
        utilize_goal_policy_ratio: float        = 0.5,
        offline_module_path: str                = None,
        offline_goal_path: str                  = None,
        online_demo_path: str                   = None,
        policy_change_freq: int                 = None,

        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        assert isinstance(self.network, ActorCriticValuePolicy)

        self.critic_freq    =   critic_freq
        self.value_freq     =   value_freq
        self.actor_freq     =   actor_freq
        self.target_freq    =   target_freq

        self.tau            =   tau
        self.beta           =   beta
        self.asy_tau        =   asy_tau
        self.exp_adv_max    =   exp_adv_max

        self.offline_goal_sample_batch_size = offline_goal_sample_batch_size
        self.offline_goal_feasible_quantile = offline_goal_feasible_quantile
        self.temperature                    = 1.
        self.utilize_goal_policy_ratio      = utilize_goal_policy_ratio
        self.policy_change_freq             = policy_change_freq
        self.cumulative_decision_steps      = 0
        self.goal_policy_in_use             = False
        self.num_chosen_goal_policy         = 0
        self.num_chosen_pi_policy           = 0

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

    def setup_network(self, network_class: Type[torch.nn.Module], network_kwargs: Dict) -> None:
        # Setup network and target network
        self.network = network_class(
            self.processor.observation_space, self.processor.action_space, **network_kwargs
        ).to(self.device)
        self.target_network = network_class(
            self.processor.observation_space, self.processor.action_space, **network_kwargs
        ).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        for param in self.target_network.parameters():
            param.requires_grad = False

    def setup_optimizers(self) -> None:
        # Default optimizer initialization
        self.optim["actor"]     = self.optim_class(self.network.actor.parameters(), **self.optim_kwargs)
        # Update the encoder with the critic.
        critic_params           = itertools.chain(self.network.critic.parameters(), self.network.encoder.parameters())
        self.optim["critic"]    = self.optim_class(critic_params, **self.optim_kwargs)
        self.optim['value']     = self.optim_class(self.network.value.parameters(), **self.optim_kwargs)

    def _update_critic(self, batch: Dict) -> Dict:
        with torch.no_grad():
            next_v      = self.network.value(batch['next_obs'])
            target_q    = batch['reward'] + batch['discount'] * next_v
        qs      =   self.network.critic(batch['obs'], batch['action'])
        q_loss  =   torch.nn.functional.mse_loss(qs, target_q.expand(*qs.shape))

        self.optim["critic"].zero_grad(set_to_none=True)
        q_loss.backward()
        self.optim["critic"].step()

        return dict(q_loss=q_loss.item(), target_q=target_q.mean().item())

    def _update_value(self, batch: Dict) -> Tuple[Dict, torch.Tensor]:
        with torch.no_grad():
            target_qs   =   self.target_network.critic(batch['obs'], batch['action'])
            target_q    =   torch.min(target_qs, dim=0, keepdim=False)[0]
        
        v       =   self.network.value(batch['obs'])
        adv     =   target_q - v
        # asymmetric value update
        weight  =   torch.abs(self.asy_tau - (adv < 0).float())
        v_loss  =   torch.mean(weight * adv**2)
        
        self.optim['value'].zero_grad(set_to_none=True)
        v_loss.backward()
        self.optim['value'].step()

        return dict(v_loss=v_loss.item(), adv=adv.mean().item()), adv.detach()

    def _update_actor(self, batch: Dict) -> Dict:
        adv         = batch['adv']
        exp_adv     = torch.exp(self.beta * adv).clamp(max=self.exp_adv_max)
        
        dist        = self.network.actor(batch['obs'])
        likelihood  = dist.log_prob(batch['action']).clamp(min=-1e6)

        actor_loss  = (- exp_adv * likelihood).mean()

        self.optim["actor"].zero_grad(set_to_none=True)
        actor_loss.backward()
        self.optim["actor"].step()

        return dict(actor_loss=actor_loss.item())

    def train_step(self, batch: Dict, step: int, total_steps: int) -> Dict:
        all_metrics = {}
        if "obs" not in batch or step < self.random_steps:
            return all_metrics

        batch["obs"]            = self.network.encoder(batch["obs"])
        with torch.no_grad():
            batch["next_obs"]   = self.target_network.encoder(batch["next_obs"])

        if step % self.value_freq == 0:
            metrics, adv = self._update_value(batch)
            batch['adv'] = adv
            all_metrics.update(metrics)

        if step % self.critic_freq == 0:
            metrics = self._update_critic(batch)
            all_metrics.update(metrics)

        if step % self.actor_freq == 0:
            metrics = self._update_actor(batch)
            all_metrics.update(metrics)

        if step % self.target_freq == 0:
            # Only update the critic and encoder for speed. Ignore the actor.
            with torch.no_grad():
                for param, target_param in zip(self.network.encoder.parameters(), self.target_network.encoder.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.network.critic.parameters(), self.target_network.critic.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return all_metrics
    
    def _get_train_action(self, obs: Any, step: int, total_steps: int) -> np.ndarray:
        batch = dict(obs=obs)
        with torch.no_grad():
            action = self.predict(batch, is_batched=False, sample=True)
        return action

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

    def _env_step(self, step: int, total_steps: int) -> Dict:
        env     =   self.env
        # Return if env is Empty or we we aren't at every env_freq steps
        if step <= self.offline_steps:
            # Purposefully set to nan so we write CSV log.
            return dict(
                env_steps   =   self._env_steps, 
                reward      =   -np.inf, 
                length      =   np.inf, 
                num_ep      =   self._num_ep
            )
        if step < self.random_steps:
            action = env.action_space.sample()
        else:
            self.eval()
            action = self._get_train_action(self._current_obs, step, total_steps)
            self.train()

        if isinstance(env.action_space, gym.spaces.Box):
            action = np.clip(action, env.action_space.low, env.action_space.high)

        next_obs, reward, done, info = env.step(action)
        self._env_steps             += 1
        self._episode_length        += 1
        self._episode_reward        += reward

        if "discount" in info:
            discount = info["discount"]
        elif hasattr(env, "_max_episode_steps") and self._episode_length == env._max_episode_steps:
            discount = 1.0
        else:
            discount = 1 - float(done)

        # Store the consequences.
        self.dataset.add(obs=next_obs, action=action, reward=reward, done=done, discount=discount)

        if done:
            self._num_ep += 1
            # Compute metrics
            metrics = dict(
                env_steps   =   self._env_steps, 
                reward      =   self._episode_reward, 
                length      =   self._episode_length, 
                num_ep      =   self._num_ep,
                goal_policy_utilize_ratio   =   self.num_chosen_goal_policy / (self.num_chosen_goal_policy + self.num_chosen_pi_policy + 1),
                pi_policy_utilize_ratio     =   self.num_chosen_pi_policy / (self.num_chosen_goal_policy + self.num_chosen_pi_policy + 1),
            )
            # Reset the environment
            self._current_obs    = env.reset()
            self.dataset.add(obs = self._current_obs)  # Add the first timestep
            self._episode_length = 0
            self._episode_reward = 0
            return metrics
        else:
            self._current_obs   = next_obs
            return dict(env_steps = self._env_steps)

    def _predict(
        self, 
        batch: Dict, 
        sample: bool                = False, 
        noise: float                = 0.0, 
        noise_clip: Optional[float] = None,
        temperature                 = 1.0
    ) -> torch.Tensor:
        with torch.no_grad():
            if isinstance(self.network, ModuleContainer) and "encoder" in self.network.CONTAINERS:
                obs = self.network.encoder(batch["obs"])
            else:
                obs = batch["obs"]

            if not sample:                          # Evaluation
                dist = self.network.actor(obs)
            else:                                   # Train
                ## policy transformation
                if self.cumulative_decision_steps % self.policy_change_freq == 0:
                    if np.random.random() < self.utilize_goal_policy_ratio:
                        self.goal_policy_in_use     = True
                        self.num_chosen_goal_policy+= 1
                    else:
                        self.goal_policy_in_use     = False
                        self.num_chosen_pi_policy  += 1
                    self.cumulative_decision_steps  = 0
                self.cumulative_decision_steps += 1
                ## perform high-level decision
                if self.goal_policy_in_use:
                    # sample feasible goals from the offline dataset w.r.t. current obs
                    # offline_batch        = self.offline_dataset.sample(batch_size = self.offline_goal_sample_batch_size)
                    offline_goal_idxs    = np.random.randint(0, len(self.offline_obs_dataset), size=self.offline_goal_sample_batch_size)
                    offline_goals        = self.offline_obs_dataset[offline_goal_idxs]
                    offline_goals        = to_device(to_tensor(offline_goals), self.device)
                    cur_obs_offgoal      = torch.concat((obs.expand(offline_goals.shape[0], obs.shape[-1]), offline_goals), dim=-1)
                    curobs2offgoal_dist  = self._measure_distance(cur_obs_offgoal)   # [B, D]

                    quantile_predicted_distance = torch.quantile(curobs2offgoal_dist, q=self.offline_goal_feasible_quantile)
                    feasible_idxs               = curobs2offgoal_dist > quantile_predicted_distance
                    feasible_goals              = offline_goals[feasible_idxs]
                    curobs2feasiblegoal_dist    = curobs2offgoal_dist[feasible_idxs]
                    # measure the distance between feasible goals to demo obs
                    n_feasible_goals, n_demo_obs    = len(feasible_goals), self.online_demo_obs.shape[0]
                    feasible_goals_reshape          = feasible_goals.reshape(n_feasible_goals, 1, feasible_goals.shape[-1]).expand((n_feasible_goals, n_demo_obs, feasible_goals.shape[-1]))
                    online_demo_obs_reshape         = self.online_demo_obs.reshape(1, n_demo_obs, obs.shape[-1]).expand((n_feasible_goals, n_demo_obs, obs.shape[-1]))
                    feasible_goal_with_online_demos = torch.concat((feasible_goals_reshape, online_demo_obs_reshape), dim=-1)
                    feaisblegoal2onlineobs_dist     = self._measure_distance(feasible_goal_with_online_demos)       # [n_feasible_goal, n_online_demo]
                    feasiblegoal2demo_dist          = torch.mean(feaisblegoal2onlineobs_dist, dim=-1, keepdim=True) # [n_feasible_goal, 1]
                    # choose the most promising goal
                    chosen_goal_idx                 = feasiblegoal2demo_dist.argmax(dim=0)
                    chosen_goal                     = feasible_goals[chosen_goal_idx].reshape(1, feasible_goals.shape[-1])
                    # inference goal-cond policy action
                    dist                            = self.offline_network.actor(torch.concat((obs, chosen_goal), dim=-1))
                else:
                    # Could be: Logits (discrete), Float (continuous), or torch Dist
                    # Update: Logits (discrete) and Float (continuous) are deprecated for SAC
                    dist = self.network.actor(obs)


            if isinstance(self.processor.action_space, gym.spaces.Box):
                if isinstance(dist, torch.distributions.Independent):
                    # Guassian Distribution
                    action = dist.sample() if sample else dist.base_dist.loc
                elif isinstance(dist, torch.distributions.MixtureSameFamily):
                    # Mixture of Gaussians.
                    if sample:
                        action          = dist.sample()
                    else:
                        # Robomimic always samples from the Categorical, but then does the mixture deterministically.
                        loc             = dist.component_distribution.base_dist.loc
                        category        = dist.mixture_distribution.sample()

                        # Expand to add Mixture Dim, Action Dim
                        es              = dist.component_distribution.event_shape
                        mix_sample_r    = category.reshape(category.shape + torch.Size([1] * (len(es) + 1)))
                        mix_sample_r    = mix_sample_r.repeat(torch.Size([1] * len(category.shape)) + torch.Size([1]) + es)
                        action          = torch.gather(loc, len(dist.batch_shape), mix_sample_r)
                        action          = action.squeeze(len(dist.batch_shape))
                elif torch.is_tensor(dist):
                    action = dist
                    # raise ValueError("Action type Float (continuous) is deprecated for SAC")
                else:
                    raise ValueError("Model output incompatible with default _predict.")

                if noise > 0.0:
                    eps     = noise * torch.randn_like(action)
                    if noise_clip is not None:
                        eps = torch.clamp(eps, -noise_clip, noise_clip)
                    action  = action + eps
                action = action.clamp(*self.action_range)
                return action
            elif isinstance(self.processor.action_space, gym.spaces.Discrete):
                raise ValueError("Action type Logits (discrete) is deprecated for SAC")
            else:
                raise ValueError("Complex action_space incompatible with default _predict.")
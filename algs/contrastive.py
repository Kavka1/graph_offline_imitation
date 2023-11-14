import itertools
from typing import Dict, Type, Optional, Tuple, List, Union

import numpy as np
import torch

from graph_offline_imitation.networks.contrastive       import ContrastiveRLNetwork
from graph_offline_imitation.algs.off_policy_algorithm  import OffPolicyAlgorithm


class ContrastiveRL(OffPolicyAlgorithm):
    def __init__(
        self,
        *args,
        entropy_coefficient: Optional[float] = None, # None means adaptive entropy coefficient
        target_entropy:      float           = 0.0,
        tau:                 float           = 0.005,
        bc_coef:             float           = 0.05,
        encoder_gradients:  str              = "both",
        **kwargs,
    ) -> None:
        # After determining dimension parameters, setup the network
        super().__init__(*args, **kwargs)
        assert encoder_gradients in ("actor", "critic", "both")
        self.encoder_gradients   = encoder_gradients
        
        self.tau                 = tau
        self.bc_coef             = bc_coef

        self.target_entropy                 = target_entropy
        self.entropy_coefficient            = entropy_coefficient
        self.adaptive_entropy_coefficient   = entropy_coefficient is None

        if self.adaptive_entropy_coefficient:
            log_alpha       = torch.tensor(0.0, dtype=torch.float).to(self.device)
            self.log_alpha  = torch.nn.Parameter(log_alpha, requires_grad=True)
        else:
            self.log_alpha  = torch.tensor(np.log(self.entropy_coefficient), dtype=torch.float).to(self.device)   

        self.action_range        = [
            float(self.processor.action_space.low.min()),
            float(self.processor.action_space.high.max()),
        ]

    def setup_network(self, network_class: Type[torch.nn.Module], network_kwargs: Dict) -> None:
        assert network_class is ContrastiveRLNetwork, "Must use ContrastiveRLNetwork with ContrastiveRL."
        self.network = network_class(
            self.processor.observation_space, self.processor.action_space, **network_kwargs
        ).to(self.device)
        
    def setup_optimizers(self) -> None:
        if self.encoder_gradients == "critic" or self.encoder_gradients == "both":
            critic_params = itertools.chain(self.network.critic.parameters(), self.network.encoder.parameters())
            actor_params = self.network.actor.parameters()
        elif self.encoder_gradients == "actor":
            critic_params = self.network.critic.parameters()
            actor_params = itertools.chain(self.network.actor.parameters(), self.network.encoder.parameters())
        else:
            raise ValueError("Unsupported value of encoder_gradients")
        self.optim["actor"]  = self.optim_class(actor_params, **self.optim_kwargs)
        self.optim["critic"] = self.optim_class(critic_params, **self.optim_kwargs)
        if self.adaptive_entropy_coefficient:
            self.optim["log_alpha"] = self.optim_class([self.log_alpha], **self.optim_kwargs)

    def train_step(self, batch: Dict, step: int, total_steps: int) -> Dict:
        # format batch data (e.g., obs goal concat)
        obs_with_goal               = self.network.format_policy_obs(batch['obs'])
        obs, action, goal           = self.network.format_value_input(batch['obs'], batch['action'])
        next_obs_with_goal          = self.network.format_policy_obs(batch['next_obs'])
        next_obs, none_action, goal = self.network.format_value_input(batch['next_obs'], None)
        # we temporally do not use img encoder
        assert isinstance(self.network.encoder, torch.nn.Identity)

        # update actor
        dist            = self.network.actor(obs_with_goal.detach() if self.encoder_gradients == "critic" else obs_with_goal)
        new_action      = dist.rsample()
        log_prob        = dist.log_prob(new_action)
        q_action        = self.network.critic(obs, new_action, goal)    # [E, B, B]
        assert q_action.shape[0] == self.network.critic.ensemble_size
        min_q_action    = torch.min(q_action, dim=0, keepdim=False)
        actor_q_loss    = torch.exp(self.log_alpha).detach() * log_prob - torch.diag(min_q_action)
        assert 0 < self.bc_coef <= 1.0
        actor_bc_loss   = - 1.0 * dist.log_prob(action)
        actor_loss      = (self.bc_coef * actor_bc_loss + (1 - self.bc_coef) * actor_q_loss).mean()
        self.optim['actor'].zero_grad(set_to_none=True)
        actor_loss.backward()
        self.optim['actor'].step()

        # update alpha
        if self.adaptive_entropy_coefficient:
            alpha_loss  = (torch.exp(self.log_alpha) * (- log_prob - self.target_entropy).detach()).mean()
            self.optim['log_alpha'].zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.optim['log_alpha'].step()

        # update critic
        I       = torch.eye(obs.shape[0])
        logits  = self.network.critic(obs, action, goal)
        loss_fn = lambda lg, label: torch.nn.functional.binary_cross_entropy_with_logits(lg, label)
        critic_loss = torch.vmap(loss_fn, in_dims=0, out_dims=0)(logits, I)    
        critic_loss = torch.mean(critic_loss) # 
        self.optim['critic'].zero_grad(set_to_none=True)
        critic_loss.backward()
        self.optim['critic'].step()

        mean_logits = torch.mean(logits, dim=0, keepdim=False)  # [B, B]
        correct     = (torch.argmax(mean_logits, dim=1) == torch.argmax(I, dim=1))
        logits_pos  = torch.sum(mean_logits * I) / torch.sum(I)
        logits_neg  = torch.sum(mean_logits * (1 - I)) / torch.sum(1 - I)
        logsumexp   = torch.logsumexp(mean_logits[:, :], dim=1) ** 2

        return dict(
            actor_loss      = actor_loss.item(),
            actor_q_loss    = actor_q_loss.item(),
            actor_bc_loss   = actor_bc_loss.item(),

            alpha_loss      = alpha_loss.item() if self.adaptive_entropy_coefficient else 0.,
            alpha           = self.log_alpha.exp().item(),

            critic_loss          = critic_loss.item(),
            binary_accuracy      = torch.mean((mean_logits > 0) == I).item(),
            categorical_accuracy = torch.mean(correct).item(),
            logits_pos           = logits_pos.item(),
            logits_neg           = logits_neg.item(),
            logsumexp            = logsumexp.mean().item()
        )

    def _predict(self, batch: Dict, sample: bool = False) -> torch.Tensor:
        with torch.no_grad():
            obs  = self.network.format_policy_obs(batch["obs"])
            z    = self.network.encoder(obs)
            dist = self.network.actor(z)
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

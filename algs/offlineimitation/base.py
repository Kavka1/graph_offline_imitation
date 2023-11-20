import itertools
from typing import Dict, Type, Optional, Tuple, List, Union, Any

import numpy as np
import torch
import gym

from graph_offline_imitation.processors.base            import Processor, IdentityProcessor
from graph_offline_imitation.algs.off_policy_algorithm  import OffPolicyAlgorithm
from graph_offline_imitation.utils                      import utils


class OfflineImitation(OffPolicyAlgorithm):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        # After determining dimension parameters, setup the network
        super().__init__(*args, **kwargs)

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
        raise NotImplementedError
        
    def setup_optimizers(self) -> None:
        raise NotImplementedError

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
import os
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import gym

from graph_offline_imitation.utils.logger   import Logger
from .base                                  import OfflineTrainer, log_from_dict, log_wrapper, time_wrapper, MAX_VALID_METRICS, _worker_init_fn


class OfflineImitationTrainer(OfflineTrainer):
    def __init__(
        self, 
        eval_env: Optional[gym.Env] = None,
        expert_dataloader_kwargs: Dict = {},
        unlabel_dataloader_kwargs: Dict = {},
        **kwargs,
    ) -> None:
        self._expert_dataloader       = None
        self.expert_dataloader_kwargs = expert_dataloader_kwargs
        self._unlabel_dataloader      = None
        self.unlabel_dataloader_kwargs= unlabel_dataloader_kwargs
        super().__init__(eval_env, **kwargs)

    @property
    def train_dataloader(self) -> Any:
        raise NotImplementedError(f"the offline imitation setting separates expert dataloader and unlabel dataloader.")

    @property
    def expert_dataloader(self) -> torch.utils.data.DataLoader:
        if not hasattr(self.model, "expert_dataset"):
            self.model.setup_datasets(self.model.env, self.total_steps)

        if self.model.expert_dataset is None:
            return None
        
        if self._expert_dataloader is None:
            shuffle     = not isinstance(self.model.expert_dataset, torch.utils.data.IterableDataset)
            pin_memory  = self.model.device.type == "cuda"
            self._expert_dataloader = torch.utils.data.DataLoader(
                self.model.expert_dataset,
                shuffle         =   shuffle,
                pin_memory      =   pin_memory,
                worker_init_fn  =   _worker_init_fn,
                **self.expert_dataloader_kwargs,
            )
        return self._expert_dataloader

    @property
    def unlabel_dataloader(self) -> torch.utils.data.DataLoader:
        if not hasattr(self.model, "unlabel_dataset"):
            self.model.setup_datasets(self.model.env, self.total_steps)

        if self.model.unlabel_dataset is None:
            return None
        
        if self._unlabel_dataloader is None:
            shuffle     = not isinstance(self.model.unlabel_dataset, torch.utils.data.IterableDataset)
            pin_memory  = self.model.device.type == "cuda"
            self._unlabel_dataloader = torch.utils.data.DataLoader(
                self.model.unlabel_dataset,
                shuffle         =   shuffle,
                pin_memory      =   pin_memory,
                worker_init_fn  =   _worker_init_fn,
                **self.unlabel_dataloader_kwargs,
            )
        return self._unlabel_dataloader

    def train(self, path: str):
        # Prepare the model for training by initializing the optimizers and the schedulers
        self.model.setup_optimizers()
        self.check_compilation()
        self.model.setup_schedulers()
        self.model.setup()  # perform any other arbitrary setup needs.
        print("[research] Training a model with", self.model.num_params, "trainable parameters.")
        print("[research] Estimated size: {:.2f} GB".format(self.model.nbytes / 1024**3))

        # First, we should detect if the path already contains a model and a checkpoint
        if os.path.exists(os.path.join(path, "final_model.pt")):
            # If so, we can finetune from that initial checkpoint. When we do this we should load strictly.
            # If we can't load it, we should immediately throw an error.
            metadata = self.model.load(os.path.join(path, "final_model.pt"), strict=True)
            current_step, steps, epochs = metadata["current_step"], metadata["steps"], metadata["epochs"]
            # Try to load the xaxis value if we need to.
        else:
            current_step, steps, epochs = 0, 0, 0

        # Setup benchmarking.
        if self.benchmark:
            torch.backends.cudnn.benchmark = True

        # Setup the Logger
        writers = ["tb", "csv"]
        try:
            # Detect if wandb has been setup. If so, log it.
            import wandb
            if wandb.run is not None:
                writers.append("wandb")
        except:
            pass

        logger = Logger(path=path, writers=writers)

        # Construct all of the metric lists to be used during training
        # Construct all the metric lists to be used during training
        train_metric_lists      = defaultdict(list)
        envstep_metric_lists    = defaultdict(list)
        profiling_metric_lists  = defaultdict(list)
        # Wrap the functions we use in logging and profile wrappers
        train_step      = log_wrapper(self.model.train_step, train_metric_lists)
        train_step      = time_wrapper(train_step, "train_step", profiling_metric_lists)
        env_step        = log_wrapper(self.model.env_step, envstep_metric_lists)
        env_step        = time_wrapper(env_step, "env_step", profiling_metric_lists)

        # separated format batch operation
        format_expert_batch    = time_wrapper(self.model.format_expert_batch, "expert_processor", profiling_metric_lists)
        format_unlabel_batch   = time_wrapper(self.model.format_unlabel_batch, 'unlabel_processor', profiling_metric_lists)

        # Compute validation trackers
        using_max_valid_metric  = self.loss_metric in MAX_VALID_METRICS
        best_valid_metric       = -1 * float("inf") if using_max_valid_metric else float("inf")

        # Compute logging frequencies
        last_train_log          = -self.log_freq  # Ensure that we log on the first step
        last_validation_log     = (
            0 if self.benchmark else -self.eval_freq
        )  # Ensure that we log the first step, except if we are benchmarking.
        last_checkpoint         = 0  # Start at 1 so we don't log the untrained model.
        profile                 = True if self.profile_freq > 0 else False  # must profile to get all keys for csv log
        self.model.train()

        start_time      = time.time()
        current_time    = start_time

        while current_step <= self.total_steps:
            for expert_batch, unlabel_batch in zip(self.expert_dataloader, self.unlabel_dataloader):
                if profile:
                    profiling_metric_lists["dataset"].append(time.time() - current_time)

                # Run any pre-train steps, like stepping the environment or training auxiliary networks.
                # Realistically this is just going to be used for environment stepping, but hey! Good to have.
                # env_step(current_step, self.total_steps, timeit=profile)

                # Next, format the batch
                expert_batch = format_expert_batch(expert_batch, timeit=profile)
                unlabel_batch= format_unlabel_batch(unlabel_batch, timeit=profile)

                # Run the train step
                train_step(expert_batch, unlabel_batch, current_step, self.total_steps, timeit=profile)

                # Update the schedulers
                for scheduler in self.model.schedulers.values():
                    scheduler.step()

                steps += 1
                if self.x_axis == "steps":
                    new_current_step = steps + 1
                elif self.x_axis == "epoch":
                    new_current_step = epochs
                elif self.x_axis in train_metric_lists:
                    new_current_step = train_metric_lists[self.x_axis][-1]  # Get the most recent value
                elif self.x_axis in envstep_metric_lists:
                    new_current_step = envstep_metric_lists[self.x_axis][-1]  # Get the most recent value
                else:
                    raise ValueError("Could not find train value for x_axis " + str(self.x_axis))

                # Now determine if we should dump the logs
                if (current_step - last_train_log) >= self.log_freq:
                    # Record timing metrics
                    current_time = time.time()
                    logger.record("time/steps", steps)
                    logger.record("time/epochs", epochs)
                    logger.record("time/steps_per_second", (current_step - last_train_log) / (current_time - start_time))
                    log_from_dict(logger, profiling_metric_lists, "time")
                    start_time = current_time
                    # Record learning rates
                    for name, scheduler in self.model.schedulers.items():
                        logger.record("lr/" + name, scheduler.get_last_lr()[0])
                    # Record training metrics
                    log_from_dict(logger, envstep_metric_lists, "env_step")
                    log_from_dict(logger, train_metric_lists, "train")
                    logger.dump(step=current_step)
                    # Update the last time we logged.
                    last_train_log = current_step

                if (current_step - last_validation_log) >= self.eval_freq:
                    self.model.eval()
                    current_valid_metric = None
                    model_metadata       = dict(current_step=current_step, epochs=epochs, steps=steps)

                    # Run and time validation step
                    current_time         = time.time()
                    validation_metrics   = self.validate(path, current_step)
                    logger.record("time/validation", time.time() - current_time)
                    if self.loss_metric in validation_metrics:
                        current_valid_metric = validation_metrics[self.loss_metric]
                    log_from_dict(logger, validation_metrics, "validation")

                    # Run and time eval step
                    current_time        = time.time()
                    eval_metrics        = self.evaluate(path, current_step)
                    logger.record("time/eval", time.time() - current_time)
                    if self.loss_metric in eval_metrics:
                        current_valid_metric = eval_metrics[self.loss_metric]
                    log_from_dict(logger, eval_metrics, "eval")

                    # Determine if we have a new best self.model.
                    if current_valid_metric is None:
                        pass
                    elif (
                        using_max_valid_metric and current_valid_metric > best_valid_metric
                    ) or (
                        not using_max_valid_metric and current_valid_metric < best_valid_metric
                    ):
                        best_valid_metric = current_valid_metric
                        self.model.save(path, "best_model", model_metadata)

                    # Render Logger contents
                    logger.render(step=current_step, )
                    # Eval Logger dump to CSV
                    logger.dump(step=current_step, eval=True)  # Mark True on the eval flag
                    # Save the final model
                    self.model.save(path, "final_model", model_metadata)  # Also save the final model every eval period.
                    # Put the model back in train mode.
                    self.model.train()
                    last_validation_log = current_step

                if self.checkpoint_freq is not None and (current_step - last_checkpoint) >= self.checkpoint_freq:
                    # Save a checkpoint
                    model_metadata  = dict(current_step=current_step, epochs=epochs, steps=steps)
                    self.model.save(path, "model_" + str(current_step), model_metadata)
                    last_checkpoint = current_step

                current_step = new_current_step  # Update the current step
                if current_step > self.total_steps:
                    break  # We need to break in the middle of an epoch.

                profile = self.profile_freq > 0 and steps % self.profile_freq == 0
                if profile:
                    current_time = time.time()  # update current time only, not start time

            epochs += 1
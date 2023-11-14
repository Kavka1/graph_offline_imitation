import collections
import os
from typing import Any, Dict, List

import gym
import imageio
import numpy as np
import torch

from . import utils

MAX_METRICS = {"success", "is_success", "completions"}
LAST_METRICS = {"goal_distance"}
MEAN_METRICS = {"discount"}


class EvalMetricTracker(object):
    """
    A simple class to make keeping track of eval metrics easy.
    Usage:
        Call reset before each episode starts
        Call step after each environment step
        call export to get the final metrics
    """

    def __init__(self):
        self.metrics = collections.defaultdict(list)
        self.ep_length = 0
        self.ep_reward = 0
        self.ep_metrics = collections.defaultdict(list)

    def reset(self) -> None:
        if self.ep_length > 0:
            # Add the episode to overall metrics
            self.metrics["reward"].append(self.ep_reward)
            self.metrics["length"].append(self.ep_length)
            for k, v in self.ep_metrics.items():
                if k in MAX_METRICS:
                    self.metrics[k].append(np.max(v))
                elif k in LAST_METRICS:  # Append the last value
                    self.metrics[k].append(v[-1])
                elif k in MEAN_METRICS:
                    self.metrics[k].append(np.mean(v))
                else:
                    self.metrics[k].append(np.sum(v))

            self.ep_length = 0
            self.ep_reward = 0
            self.ep_metrics = collections.defaultdict(list)

    def step(self, reward: float, info: Dict) -> None:
        self.ep_length += 1
        self.ep_reward += reward
        for k, v in info.items():
            if isinstance(v, float) or np.isscalar(v):
                self.ep_metrics[k].append(v)

    def add(self, k: str, v: Any):
        self.metrics[k].append(v)

    def export(self) -> Dict:
        if self.ep_length > 0:
            # We have one remaining episode to log, make sure to get it.
            self.reset()
        metrics = {k: np.mean(v) for k, v in self.metrics.items()}
        metrics["reward_std"] = np.std(self.metrics["reward"])
        return metrics


def eval_multiple(env, model, path: str, step: int, eval_fns: List[str], eval_kwargs: List[Dict]):
    all_metrics = dict()
    for eval_fn, eval_kwarg in zip(eval_fns, eval_kwargs):
        metrics = locals()[eval_fn](env, model, path, step, **eval_kwarg)
        all_metrics.update(metrics)
    return all_metrics


def eval_policy(
    env: gym.Env,
    model,
    path: str,
    step: int,
    num_ep: int = 10,
    num_gifs: int = 0,
    width=200,
    height=200,
    every_n_frames: int = 2,
    terminate_on_success=False,
    render_human: bool = False
) -> Dict:
    metric_tracker = EvalMetricTracker()
    assert num_gifs <= num_ep, "Cannot save more gifs than eval ep."

    for i in range(num_ep):
        # Reset Metrics
        done = False
        ep_length, ep_reward = 0, 0
        frames = []
        save_gif = i < num_gifs
        render_kwargs = dict(mode="rgb_array", width=width, height=height) if save_gif else dict()
        obs = env.reset()

        if not render_human and save_gif:
            frames.append(env.render(**render_kwargs))
        elif render_human:
            env.render('human')

        metric_tracker.reset()
        while not done:
            batch = dict(obs=obs)
            if hasattr(env, "_max_episode_steps"):
                batch["horizon"] = env._max_episode_steps - ep_length
            with torch.no_grad():
                action = model.predict(batch)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            metric_tracker.step(reward, info)
            ep_length += 1

            if not render_human and save_gif and ep_length % every_n_frames == 0:
                frames.append(env.render(**render_kwargs))
            elif render_human:
                env.render('human')

            if terminate_on_success and (info.get("success", False) or info.get("is_success", False)):
                done = True
        if hasattr(env, "get_normalized_score"):
            metric_tracker.add("score", env.get_normalized_score(ep_reward))

        if not render_human and save_gif:
            gif_name = "vis-{}_ep-{}.gif".format(step, i)
            imageio.mimsave(os.path.join(path, gif_name), frames)

    return metric_tracker.export()


def render_policy(
    env: gym.Env,
    model,
    path: str,
    step: int,
    save_ep: bool = False,
    save_gif: bool = False,
    save_only_success: bool = False,
    num_ep: int = 10,
    width: int = 200,
    height: int = 200,
    every_n_frames: int = 2,
    terminate_on_success: bool = False,
) -> Dict:
    metric_tracker = EvalMetricTracker()

    for i in range(num_ep):
        # Reset Metrics
        done                    = False
        ep_length, ep_reward    = 0, 0
        
        render_kwargs = dict(mode="rgb_array", width=width, height=height) if save_gif else dict(mode='human')
        
        frames                  = []
        obs_seq                = []
        act_seq                = []
        rew_seq                = []
        done_seq               = []
        desired_goal_seq       = []
        task_mask_seq          = []

        obs = env.reset()

        if save_gif:
            try:
                pic = env.render(**render_kwargs)
                can_mod_shape = True
            except:
                pic = env.render(mode='rgb_array')
                print(f'cannot modify the shape of the render pics, current pic shape {pic.shape}')
                can_mod_shape = False
            frames.append(pic)
        else:
            env.render('human')

        metric_tracker.reset()

        all_complete = False

        while not done:
            batch = dict(obs=obs)
            if hasattr(env, "_max_episode_steps"):
                batch["horizon"] = env._max_episode_steps - ep_length
            with torch.no_grad():
                action = model.predict(batch)
            obs_, reward, done, info = env.step(action)
            ep_reward += reward
            metric_tracker.step(reward, info)
            ep_length += 1

            if save_gif and ep_length % every_n_frames == 0:
                if can_mod_shape:
                    frames.append(env.render(**render_kwargs))
                else:
                    frames.append(env.render(mode = 'rgb_array'))
            else:
                env.render('human')

            if terminate_on_success and (info.get("success", False) or info.get("is_success", False)):
                done       = True

            if info.get("success", False) == 1.:
                all_complete = True
        
            if save_ep:
                
                if isinstance(obs, np.ndarray):
                    obs_seq.append(obs)
                elif isinstance(obs, dict):
                    assert 'achieved_goal' in obs
                    obs_seq.append(obs['achieved_goal'])
                else:
                    raise ValueError('obs is neither np.array nor dict')
                
                act_seq.append(action)
                rew_seq.append([reward])
                done_seq.append([done])
                if hasattr(env, "desired_goal"):
                    desired_goal_seq.append(env.desired_goal)
                if hasattr(env, "expected_mask"):
                    task_mask_seq.append(env.expected_mask)

            obs = obs_
        
        if hasattr(env, "get_normalized_score"):
            metric_tracker.add("score", env.get_normalized_score(ep_reward))

        if save_gif or save_ep:
            if task_mask_seq:
                task_name = '_'.join([task_name for i_task, task_name in enumerate(env.ALL_TASKS) if task_mask_seq[0][i_task]])
            else:
                task_name = ''

            if save_only_success and not all_complete:
                continue

            if save_gif:
                gif_name = "{}-step{}_ep{}-{}.gif".format(task_name, step, i, 'complete' if all_complete else 'fail')
                imageio.mimsave(os.path.join(path, gif_name), frames)
            if save_ep:
                ep_name = "{}-step{}_ep{}_{}.npz".format(task_name, step, i, 'complete' if all_complete else 'fail')
                np.savez(
                    os.path.join(path, ep_name),
                    obs_seq = np.stack(obs_seq),
                    act_seq = np.stack(act_seq),
                    rew_seq = np.stack(rew_seq),
                    done_seq = np.stack(done_seq),
                    desired_goal_seq = np.stack(desired_goal_seq),
                    task_mask_seq = np.stack(task_mask_seq)
                )


    return metric_tracker.export()
from typing import List, Dict, Optional, Tuple
import gym
import d4rl
import numpy as np
import torch
import h5py
import os
from pathlib import Path
from tqdm import tqdm

from graph_offline_imitation.datasets.replay_buffer import ReplayBuffer, HindsightReplayBuffer


NAME2D4RL = {
    'expert': {
        'umaze':          'antmaze-umaze-v0',
    },
    'diverse': {
        'umaze':          'antmaze-umaze-diverse-v0',
    }
}


def get_keys(h5file):
    keys = []
    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)
    h5file.visititems(visitor)
    return keys


class AntMazeDataset(ReplayBuffer):
    def __init__(
        self, 
        observation_space: gym.Space, 
        action_space: gym.Space, 
        name: str,
        dataset_path: Optional[str] = None,
        action_eps: float = 0.0,
        max_ep: Optional[int] = None,
        **kwargs,
    ):
        assert name.startswith('antmaze_')
        self.env_name       = name              # in the form [antmaze_maze_expert/diverse/play]
        self.action_eps     = action_eps
        self.max_ep         = max_ep
        self.dataset_path   = dataset_path
        super().__init__(observation_space, action_space, **kwargs)

    def _data_generator(self):
        if self.dataset_path is not None:
            dataset = {}
            with h5py.File(self.dataset_path, 'r') as dataset_file:
                for k in tqdm(get_keys(dataset_file), desc="load datafile"):
                    try:  # first try loading as an array
                        dataset[k] = dataset_file[k][:]
                    except ValueError as e:  # try loading as a scalar
                        dataset[k] = dataset_file[k][()]
            # Run a few quick sanity checks
            for key in ['observations', 'actions', 'rewards', 'terminals', 'timeouts', 'infos/goal']:
                assert key in dataset, 'Dataset is missing key %s' % key
        else:
            d4rl_env_name   = self.env_name.split('antmaze_')[1]
            env_id, ds_type = [*d4rl_env_name.split('_')]
            dataset_name    = NAME2D4RL[ds_type][env_id]
            env_temp        = gym.make(dataset_name)
            dataset         = env_temp.get_dataset()
        
        num_ep_added    = 0

        obs_            = []
        action_         = [self.dummy_action]
        reward_         = [0.0]
        done_           = [False]
        discount_       = [1.0]
        episode_step    = 0

        for i in range(dataset["rewards"].shape[0]):
            obs                     = dataset["observations"][i].astype(np.float32)
            action                  = dataset["actions"][i].astype(np.float32)
            reward                  = dataset["rewards"][i].astype(np.float32)
            terminal                = bool(dataset["terminals"][i])
            done                    = dataset["timeouts"][i]

            obs_.append(obs)
            action_.append(action)
            reward_.append(reward)
            discount_.append(1 - float(terminal))
            done_.append(done)
            episode_step += 1

            if done:
                if "next_observations" in dataset:
                    obs_.append(dataset["next_observations"][i].astype(np.float32))
                else:
                    # We need to do something to pad to the full length.
                    # The default solution is to get rid of this transition
                    # but we need a transition with the terminal flag for our replay buffer
                    # implementation to work.
                    # Since we always end up masking this out anyways, it shouldn't matter and we can just repeat
                    obs_.append(dataset["observations"][i].astype(np.float32))

                obs_    = np.stack(obs_, axis=0).astype(np.float32)
                action_ = np.stack(action_, axis=0).astype(np.float32)
                if self.action_eps > 0.0:
                    action_ = np.clip(action_, -1.0 + self.action_eps, 1.0 - self.action_eps)
                reward_     = np.array(reward_).astype(np.float32)
                discount_   = np.array(discount_).astype(np.float32)
                done_       = np.array(done_, dtype=np.bool_)
                kwargs      = dict()
                # Compute the ground truth horizon metrics
                if len(reward_) > 3:
                    # print("yielded")
                    yield (obs_, action_, reward_, done_, discount_, kwargs)

                # reset the episode trackers
                obs_            = []
                action_         = [self.dummy_action]
                reward_         = [0.0]
                done_           = [False]
                discount_       = [1.0]
                episode_step    = 0

                num_ep_added    += 1
                if self.max_ep is not None and num_ep_added == self.max_ep:
                    break

        # Finally clean up the environment
        del dataset
        if self.dataset_path is None:
            del env_temp


class AntMazeGoalCondDataset(HindsightReplayBuffer):
    def __init__(
        self, 
        observation_space: gym.Space, 
        action_space: gym.Space, 
        name: str,
        dataset_path: Optional[str] = None,
        action_eps: float = 0.0,
        max_ep: Optional[int] = None,
        **kwargs,
    ):
        assert name.startswith('antmaze_')
        self.env_name       = name              # in the form [antmaze_maze_expert/diverse/play]
        self.action_eps     = action_eps
        self.max_ep         = max_ep
        self.dataset_path   = dataset_path
        super().__init__(observation_space, action_space, **kwargs)

    def _data_generator(self):
        if self.dataset_path is not None:
            dataset = {}
            with h5py.File(self.dataset_path, 'r') as dataset_file:
                for k in tqdm(get_keys(dataset_file), desc="load datafile"):
                    try:  # first try loading as an array
                        dataset[k] = dataset_file[k][:]
                    except ValueError as e:  # try loading as a scalar
                        dataset[k] = dataset_file[k][()]
            # Run a few quick sanity checks
            for key in ['observations', 'actions', 'rewards', 'terminals', 'timeouts', 'infos/goal']:
                assert key in dataset, 'Dataset is missing key %s' % key
        else:
            d4rl_env_name       = self.env_name.split('antmaze_')[1]
            maze_name, ds_type  = [*d4rl_env_name.split('_')]
            dataset_name        = NAME2D4RL[ds_type][maze_name]
            env_temp            = gym.make(dataset_name)
            dataset             = env_temp.get_dataset()
        
        num_ep_added    = 0

        obs_            = []
        ag_             = []
        g_              = []
        action_         = [self.dummy_action]
        reward_         = [0.0]
        done_           = [False]
        discount_       = [1.0]
        episode_step    = 0

        for i in range(dataset["rewards"].shape[0]):
            obs                     = dataset["observations"][i].astype(np.float32)
            achieved_goal           = dataset["observations"][i].astype(np.float32)     # use the full obs as achieved goal
            desired_goal            = dataset["infos/goal"][i].astype(np.float32)       # init as all zeros before relabelling

            action                  = dataset["actions"][i].astype(np.float32)
            reward                  = dataset["rewards"][i].astype(np.float32)
            terminal                = bool(dataset["terminals"][i])
            done                    = dataset["timeouts"][i]

            obs_.append(obs)
            ag_.append(achieved_goal)
            g_.append(desired_goal)
            action_.append(action)
            reward_.append(reward)
            discount_.append(1 - float(terminal))
            done_.append(done)

            episode_step += 1

            if done:
                if "next_observations" in dataset:
                    obs_.append(dataset["next_observations"][i].astype(np.float32))
                    ag_.append(dataset["next_observations"][i].astype(np.float32))
                    temp_g  = np.zeros_like(dataset["next_observations"][i, :2].astype(np.float32))
                    g_.append(temp_g)
                else:
                    # We need to do somethign to pad to the full length.
                    # The default solution is to get rid of this transtion
                    # but we need a transition with the terminal flag for our replay buffer
                    # implementation to work.
                    # Since we always end up masking this out anyways, it shouldn't matter and we can just repeat
                    obs_.append(dataset["observations"][i].astype(np.float32))
                    ag_.append(dataset["observations"][i].astype(np.float32))
                    temp_g     = np.zeros_like(dataset["observations"][i, :2].astype(np.float32))
                    g_.append(temp_g)

                dict_obs = {
                    "observation":      np.stack(obs_, axis=0).astype(np.float32),
                    self.achieved_key:  np.stack(ag_, axis=0).astype(np.float32),
                    self.goal_key:      np.stack(g_, axis=0).astype(np.float32),
                }
                action_ = np.stack(action_, axis=0).astype(np.float32)
                if self.action_eps > 0.0:
                    action_ = np.clip(action_, -1.0 + self.action_eps, 1.0 - self.action_eps)
                reward_     = np.array(reward_).astype(np.float32)
                discount_   = np.array(discount_).astype(np.float32)
                done_       = np.array(done_, dtype=np.bool_)
                kwargs      = dict()
                # Compute the ground truth horizon metrics
                horizon     = -100 * np.ones(done_.shape, dtype=np.int)
                (ends,)     = np.where(done_)
                starts      = np.concatenate(([0], ends[:-1] + 1))
                for start, end in zip(starts, ends):
                    horizon[start : end + 1] = np.arange(end - start + 1, 0, -1)
                kwargs["horizon"] = horizon

                if len(reward_) > 3:
                    # print("yielded")
                    yield (dict_obs, action_, reward_, done_, discount_, kwargs)

                # reset the episode trackers
                obs_            = []
                ag_             = []
                g_              = []
                action_         = [self.dummy_action]
                reward_         = [0.0]
                done_           = [False]
                discount_       = [1.0]
                episode_step    = 0

                num_ep_added    += 1
                if self.max_ep is not None and num_ep_added == self.max_ep:
                    break

        # Finally clean up the environment
        del dataset
        if self.dataset_path is None:
            del env_temp


def AntmazeExpDivDatasets(
    observation_space: gym.Space,
    action_space:      gym.Space,
    name:              str,
    exp_use_goal:      bool = False,
    rand_use_goal:     bool = False,
    **kwargs
) -> Tuple[ReplayBuffer, ReplayBuffer]:
    assert name.startswith('antmaze_') and 'expdiv' in name
    temp_tuple = tuple(name.split('_'))
    assert len(temp_tuple) == 4
    maze_name, num_exp_traj = temp_tuple[1], temp_tuple[3]

    num_exp_traj                   = int(num_exp_traj)

    exp_dataset_name = f"antmaze_{maze_name}_expert"
    div_dataset_name = f"antmaze_{maze_name}_diverse"

    exp_dataset_path = f"{Path(__file__).resolve().parent}/assets/antmaze_{maze_name}_expdiv_{num_exp_traj}/expert.hdf5"
    div_dataset_path = f"{Path(__file__).resolve().parent}/assets/antmaze_{maze_name}_expdiv_{num_exp_traj}/diverse.hdf5"

    if exp_use_goal or rand_use_goal:
        obs_low     = observation_space.low
        obs_high    = observation_space.high
        goal_observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32),
                "achieved_goal": gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32),
                "desired_goal": gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32),
            }
        )
        assert 'relabel_fraction' in kwargs
        kwargs_wo_relabel = kwargs.copy()
        kwargs_wo_relabel.pop('relabel_fraction')
    else:
        kwargs_wo_relabel = kwargs.copy()
    
    if exp_use_goal:
        exp_dataset  = AntMazeGoalCondDataset(
            goal_observation_space,
            action_space,
            name         = exp_dataset_name,
            dataset_path = exp_dataset_path,
            **kwargs
        )
    else:
        exp_dataset = AntMazeDataset(
            observation_space,
            action_space,
            name        = exp_dataset_name,
            dataset_path= exp_dataset_path,
            **kwargs_wo_relabel
        )
    
    if rand_use_goal:
        rand_dataset = AntMazeGoalCondDataset(
            goal_observation_space,
            action_space,
            name        = div_dataset_name,
            dataset_path= div_dataset_path,
            **kwargs
        )
    else:
        rand_dataset = AntMazeDataset(
            observation_space,
            action_space,
            name        = div_dataset_name,
            dataset_path= div_dataset_path,
            **kwargs_wo_relabel
        )
    
    return exp_dataset, rand_dataset



def make_antmaze_exp_diverse_dataset(
    maze_name:     str,
    num_exp_traj:   int,
):  
    assert maze_name == 'umaze', ("Only Umaze expert-diverse can be implemented currently.")

    exp_env_name, diverse_env_name = NAME2D4RL['expert'][maze_name], NAME2D4RL['diverse'][maze_name]
    exp_env, diverse_env           = gym.make(exp_env_name), gym.make(diverse_env_name)
    exp_dataset, diverse_dataset   = exp_env.get_dataset(), diverse_env.get_dataset()
    
    # pick out multiple expert trajectories
    expert_obs_all      =   []
    expert_act_all      =   []
    expert_rew_all      =   []
    expert_terminal_all =   []
    expert_done_all     =   []
    expert_goal_all     =   []
    expert_return_all   =   []  # log returns of all trajectories
    expert_episode_step_all = []
    
    temp_exp_obs_traj       =   []
    temp_exp_act_traj       =   []
    temp_exp_rew_traj       =   []
    temp_exp_terminal_traj  =   []
    temp_exp_done_traj      =   []
    temp_exp_goal_traj      =   []
    temp_exp_episode_step   =   0
    temp_exp_episode_return =   0.

    for i in range(exp_dataset['rewards'].shape[0]):
        obs         =   exp_dataset['observations'][i].astype(np.float32)
        act         =   exp_dataset['actions'][i].astype(np.float32)
        rew         =   exp_dataset['rewards'][i].astype(np.float32)
        terminal    =   bool(exp_dataset["terminals"][i])
        done        =   exp_dataset['timeouts'][i]
        goal        =   exp_dataset["infos/goal"][i].astype(np.float32)

        temp_exp_obs_traj.append(obs)
        temp_exp_act_traj.append(act)
        temp_exp_rew_traj.append(rew)
        temp_exp_terminal_traj.append(terminal)
        temp_exp_done_traj.append(done)
        temp_exp_goal_traj.append(goal)
        temp_exp_episode_step   += 1
        temp_exp_episode_return += rew

        epi_done = done

        if epi_done:
            expert_obs_all.append(np.array(temp_exp_obs_traj))
            expert_act_all.append(np.array(temp_exp_act_traj))
            expert_rew_all.append(np.array(temp_exp_rew_traj))
            expert_terminal_all.append(np.array(temp_exp_terminal_traj))
            expert_done_all.append(np.array(temp_exp_done_traj))
            expert_goal_all.append(np.array(temp_exp_goal_traj))

            expert_return_all.append(temp_exp_episode_return)
            expert_episode_step_all.append(temp_exp_episode_step)

            # reset the temp episode
            temp_exp_obs_traj       =   []
            temp_exp_act_traj       =   []
            temp_exp_rew_traj       =   []
            temp_exp_terminal_traj  =   []
            temp_exp_done_traj      =   []
            temp_exp_episode_step   =   0
            temp_exp_episode_return =   0.
        
    # choose the episodes with higher return as expert episodes
    traj_idxs_sort_by_return        =   np.argsort(np.array(expert_return_all)).astype(np.int16).tolist()
    
    chosen_as_exp_traj_idxs         =   traj_idxs_sort_by_return[-num_exp_traj:]
    # left_as_unlabel_traj_idxs       =   traj_idxs_sort_by_return[:-num_exp_traj]

    expert_obs                      =   np.concatenate([expert_obs_all[k] for k in chosen_as_exp_traj_idxs], axis=0) 
    expert_act                      =   np.concatenate([expert_act_all[k] for k in chosen_as_exp_traj_idxs], axis=0)
    expert_rew                      =   np.concatenate([expert_rew_all[k] for k in chosen_as_exp_traj_idxs], axis=0)
    expert_terminal                 =   np.concatenate([expert_terminal_all[k] for k in chosen_as_exp_traj_idxs], axis=0)
    expert_done                     =   np.concatenate([expert_done_all[k] for k in chosen_as_exp_traj_idxs], axis=0)
    expert_goal                     =   np.concatenate([expert_goal_all[k] for k in chosen_as_exp_traj_idxs], axis=0)
    expert_dataset                =   {
        'observations':     expert_obs.astype(np.float32),
        'actions':          expert_act.astype(np.float32),
        'rewards':          expert_rew.astype(np.float32),
        'terminals':        expert_terminal.astype(np.float32),
        'timeouts':         expert_done.astype(np.float32),
        'infos/goal':       expert_goal
    }

    # use the diverse dataset as unlabeled dataset
    unlabel_dataset            =   diverse_dataset

    # save to assets
    expert_path =f"{Path(__file__).resolve().parent}/assets/antmaze_{maze_name}_expdiv_{num_exp_traj}/"
    unlabel_path=f"{Path(__file__).resolve().parent}/assets/antmaze_{maze_name}_expdiv_{num_exp_traj}/"

    if not os.path.exists(expert_path):
        os.makedirs(expert_path)
    if not os.path.exists(unlabel_path):
        os.makedirs(unlabel_path)

    for path, data, name in [(expert_path, expert_dataset, 'expert'), (unlabel_path, unlabel_dataset, 'diverse')]:
        with h5py.File(path + f"{name}.hdf5", 'w') as pfile:  # saves the data
            for key in list(data.keys()):
                pfile[key] = data[key]
        print(f'succeed saving {name} antmaze {maze_name} with exp_traj_num {num_exp_traj}')

    # test loading dataset
    for path, name in [(expert_path, 'expert.hdf5'), (unlabel_path, 'diverse.hdf5')]:
        dataset = {}
        with h5py.File(path + name, 'r') as dataset_file:
            for k in tqdm(get_keys(dataset_file), desc="load datafile"):
                try:  # first try loading as an array
                    dataset[k] = dataset_file[k][:]
                except ValueError as e:  # try loading as a scalar
                    dataset[k] = dataset_file[k][()]
            # Run a few quick sanity checks
            for key in ['observations', 'actions', 'rewards', 'terminals', 'timeouts']:
                assert key in dataset, 'Dataset is missing key %s' % key


if __name__ == "__main__":
    # for maze in [
    #     'umaze',
    # ]:
    #     for num_exp_traj in [
    #         3,
    #         5, 
    #         10,
    #         20,
    #     ]:
    #         make_antmaze_exp_diverse_dataset(maze, num_exp_traj)
    dataset = {}
    dataset_path = f"{Path(__file__).resolve().parent}/assets/antmaze_umaze_expdiv_10/diverse.hdf5"
    with h5py.File(dataset_path, 'r') as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            try:  # first try loading as an array
                dataset[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                dataset[k] = dataset_file[k][()]
    print('over')
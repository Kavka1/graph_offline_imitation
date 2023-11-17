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
from graph_offline_imitation.datasets.mujoco_exp_rand_dataset import MujocoDataset, MujocoGoalCondDataset, get_keys


NAME2D4RL = {
    'expert': {
        'ant':          'ant-expert-v2',
        'hopper':       'hopper-expert-v2',
        'walker':       'walker2d-expert-v2',
        'halfcheetah':  'halfcheetah-expert-v2'
    },
    'diverse': {
        'ant':          'ant-random-v2',
        'hopper':       'hopper-random-v2',
        'walker':       'walker2d-random-v2',
        'halfcheetah':  'halfcheetah-random-v2'
    }
}


def MujocoExpDivDatasets(
    observation_space: gym.Space,
    action_space:      gym.Space,
    name:              str,
    exp_use_goal:      bool = False,
    rand_use_goal:     bool = False,
    **kwargs
) -> Tuple[ReplayBuffer, ReplayBuffer]:
    # format "mujoco_{ant/walker/hopper/halfcheetah}_expdiv_10_100"
    assert name.startswith('mujoco_') and 'expdiv' in name
    temp_tuple = tuple(name.split('_'))
    assert len(temp_tuple) == 5
    agent_name, num_exp_traj, num_add_traj = temp_tuple[1], temp_tuple[3], temp_tuple[4]

    num_exp_traj                           = int(num_exp_traj)
    num_add_traj                           = int(num_add_traj)

    exp_dataset_name = f"mujoco_{agent_name}_expert"
    div_dataset_name = f"mujoco_{agent_name}_diverse"

    exp_dataset_path = f"{Path(__file__).resolve().parent}/assets/mujoco_{agent_name}_expdiv_{num_exp_traj}_{num_add_traj}/expert.hdf5"
    div_dataset_path = f"{Path(__file__).resolve().parent}/assets/mujoco_{agent_name}_expdiv_{num_exp_traj}_{num_add_traj}/diverse.hdf5"

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
        exp_dataset  = MujocoGoalCondDataset(
            goal_observation_space,
            action_space,
            name         = exp_dataset_name,
            dataset_path = exp_dataset_path,
            **kwargs
        )
    else:
        exp_dataset = MujocoDataset(
            observation_space,
            action_space,
            name        = exp_dataset_name,
            dataset_path= exp_dataset_path,
            **kwargs_wo_relabel
        )
    
    if rand_use_goal:
        rand_dataset = MujocoGoalCondDataset(
            goal_observation_space,
            action_space,
            name        = div_dataset_name,
            dataset_path= div_dataset_path,
            **kwargs
        )
    else:
        rand_dataset = MujocoDataset(
            observation_space,
            action_space,
            name        = div_dataset_name,
            dataset_path= div_dataset_path,
            **kwargs_wo_relabel
        )
    
    return exp_dataset, rand_dataset



def make_mujoco_exp_div_dataset(
    agent_name:     str,
    num_exp_traj:   int,
    num_add_traj:   int,
):  
    exp_env_name, rand_env_name = NAME2D4RL['expert'][agent_name], NAME2D4RL['diverse'][agent_name]
    exp_env, rand_env           = gym.make(exp_env_name), gym.make(rand_env_name)
    exp_dataset, rand_dataset   = exp_env.get_dataset(), rand_env.get_dataset()
    
    # pick out multiple expert trajectories
    expert_obs_all      =   []
    expert_act_all      =   []
    expert_rew_all      =   []
    expert_terminal_all =   []
    expert_done_all     =   []
    expert_return_all   =   []  # log returns of all trajectories
    expert_episode_step_all = []
    
    temp_exp_obs_traj       =   []
    temp_exp_act_traj       =   []
    temp_exp_rew_traj       =   []
    temp_exp_terminal_traj  =   []
    temp_exp_done_traj      =   []
    temp_exp_episode_step   =   0
    temp_exp_episode_return =   0.

    for i in range(exp_dataset['rewards'].shape[0]):
        obs         =   exp_dataset['observations'][i].astype(np.float32)
        act         =   exp_dataset['actions'][i].astype(np.float32)
        rew         =   exp_dataset['rewards'][i].astype(np.float32)
        terminal    =   bool(exp_dataset["terminals"][i])
        done        =   exp_dataset['timeouts'][i]

        temp_exp_obs_traj.append(obs)
        temp_exp_act_traj.append(act)
        temp_exp_rew_traj.append(rew)
        temp_exp_terminal_traj.append(terminal)
        temp_exp_done_traj.append(done)
        temp_exp_episode_step   += 1
        temp_exp_episode_return += rew

        if agent_name in ['ant', 'hopper', 'walker']:
            epi_done    =   done or terminal
        elif agent_name == 'halfcheetah':
            epi_done    =   done
        else:
            raise ValueError

        if epi_done:
            expert_obs_all.append(np.array(temp_exp_obs_traj))
            expert_act_all.append(np.array(temp_exp_act_traj))
            expert_rew_all.append(np.array(temp_exp_rew_traj))
            expert_terminal_all.append(np.array(temp_exp_terminal_traj))
            expert_done_all.append(np.array(temp_exp_done_traj))

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
    left_as_diverse_traj_idxs       =   traj_idxs_sort_by_return[:-num_exp_traj]
    left_as_diverse_traj_idxs       =   left_as_diverse_traj_idxs[-num_add_traj:]

    expert_obs                      =   np.concatenate([expert_obs_all[k] for k in chosen_as_exp_traj_idxs], axis=0) 
    expert_act                      =   np.concatenate([expert_act_all[k] for k in chosen_as_exp_traj_idxs], axis=0)
    expert_rew                      =   np.concatenate([expert_rew_all[k] for k in chosen_as_exp_traj_idxs], axis=0)
    expert_terminal                 =   np.concatenate([expert_terminal_all[k] for k in chosen_as_exp_traj_idxs], axis=0)
    expert_done                     =   np.concatenate([expert_done_all[k] for k in chosen_as_exp_traj_idxs], axis=0)
    expert_dataset                =   {
        'observations':     expert_obs.astype(np.float32),
        'actions':          expert_act.astype(np.float32),
        'rewards':          expert_rew.astype(np.float32),
        'terminals':        expert_terminal.astype(np.float32),
        'timeouts':         expert_done.astype(np.float32),
    }

    # combine the left expert dataset and random dataset as the unlabeled dataset
    diverse_obs                     =   np.concatenate([expert_obs_all[k] for k in left_as_diverse_traj_idxs], axis=0)
    diverse_act                     =   np.concatenate([expert_act_all[k] for k in left_as_diverse_traj_idxs], axis=0)
    diverse_rew                     =   np.concatenate([expert_rew_all[k] for k in left_as_diverse_traj_idxs], axis=0)
    diverse_terminal                =   np.concatenate([expert_terminal_all[k] for k in left_as_diverse_traj_idxs], axis=0)
    diverse_done                    =   np.concatenate([expert_done_all[k] for k in left_as_diverse_traj_idxs], axis=0)
    diverse_dataset            =   {
        'observations':     np.concatenate((diverse_obs, rand_dataset['observations']), axis=0).astype(np.float32),
        'actions':          np.concatenate((diverse_act, rand_dataset['actions']), axis=0).astype(np.float32),
        'rewards':          np.concatenate((diverse_rew, rand_dataset['rewards']), axis=0).astype(np.float32),
        'terminals':        np.concatenate((diverse_terminal, rand_dataset['terminals']), axis=0).astype(np.float32),
        'timeouts':         np.concatenate((diverse_done, rand_dataset['timeouts']), axis=0).astype(np.float32),
    }

    # save to assets
    expert_path  =f"{Path(__file__).resolve().parent}/assets/mujoco_{agent_name}_expdiv_{num_exp_traj}_{num_add_traj}/"
    diverse_path =f"{Path(__file__).resolve().parent}/assets/mujoco_{agent_name}_expdiv_{num_exp_traj}_{num_add_traj}/"

    if not os.path.exists(expert_path):
        os.makedirs(expert_path)
    if not os.path.exists(diverse_path):
        os.makedirs(diverse_path)

    for path, data, name in [(expert_path, expert_dataset, 'expert'), (diverse_path, diverse_dataset, 'diverse')]:
        with h5py.File(path + f"{name}.hdf5", 'w') as pfile:  # saves the data
            for key in list(data.keys()):
                pfile[key] = data[key]
        # f        = h5py.File(path + f"{name}.hdf5", 'w')
        # dset     = f.create_dataset(name, data = data)
        # f.close()
        print(f'succeed saving {name} mujoco {agent_name} with exp_traj_num {num_exp_traj} and add_traj_num {num_add_traj}')

    # test loading dataset
    for path, name in [(expert_path, 'expert.hdf5'), (diverse_path, 'diverse.hdf5')]:
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
    for agent in [
        'ant',
        'walker',
        'hopper',
        'halfcheetah'
    ]:
        for num_exp_traj, num_add in [
            (1, 100),
            (1, 200),
            (5, 100),
            (5, 200),
            (10, 100),
            (10, 200)
        ]:
            make_mujoco_exp_div_dataset(agent, num_exp_traj, num_add)
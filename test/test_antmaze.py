from graph_offline_imitation import envs
import gym
import numpy as np


if __name__ == '__main__':
    for env_name in [
        'offline_ant_umaze',
        'offline_goal_ant_umaze',
        'offline_ant_umaze_diverse',
        'offline_goal_ant_umaze_diverse',
        'offline_ant_medium_play',
        'offline_goal_ant_medium_play',
        'offline_ant_medium_diverse',
        'offline_goal_ant_medium_diverse',
        'offline_ant_large_play',
        'offline_goal_ant_large_play',
        'offline_ant_large_diverse',
        'offline_goal_ant_large_diverse'
    ]:
        env = gym.make(env_name)
        obs_space = env.observation_space
        act_space = env.action_space
        # obs = env.reset()
        dataset   = env.get_dataset()

        print(f'Complete testing env {env_name}')
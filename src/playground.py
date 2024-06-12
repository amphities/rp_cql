import argparse
import os

from cql_stuff.src.four_room_stuff.shortest_path import find_all_action_values
from cql_stuff.src.four_room_stuff.utils import obs_to_state
from four_room_stuff.env import FourRoomsEnv
from four_room_stuff.wrappers import gym_wrapper
import numpy as np
from pyvirtualdisplay import Display
import dill
import gymnasium as gym
import d3rlpy
import matplotlib.pyplot as plt

gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)

with open('../configs/fourrooms_train_config.pl', 'rb') as file:
    train_config = dill.load(file)

with open('../configs/fourrooms_test_0_config.pl', 'rb') as file:
    unreachable_config = dill.load(file)

with open('../configs/fourrooms_test_100_config.pl', 'rb') as file:
    reachable_config = dill.load(file)

env_configs = [train_config, unreachable_config, reachable_config]
step_cap = 40
seed = 1
def extract_number(filename):
    # Split the filename into parts using underscore as the separator
    parts = filename.split('_')
    # The number is the last part
    number = parts[-1][:-3]
    # Convert the number to an integer and return it
    return int(number)

def optimal_policy(state):
    state = obs_to_state(state)
    q_values = find_all_action_values(state[:2], state[2], state[3:5], state[5:], 0.99)
    optimal_action = np.argmax(q_values)
    return optimal_action

with Display(visible=False) as disp:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()


    for env_index, env_config in enumerate(env_configs):
        d3rlpy.seed(seed)
        bc_env = gym_wrapper(gym.make('MiniGrid-FourRooms-v1',
                                      agent_pos=env_config['agent positions'],
                                      goal_pos=env_config['goal positions'],
                                      doors_pos=env_config['topologies'],
                                      agent_dir=env_config['agent directions'],
                                      render_mode="rgb_array"))
        cql_env = gym_wrapper(gym.make('MiniGrid-FourRooms-v1',
                                       agent_pos=env_config['agent positions'],
                                       goal_pos=env_config['goal positions'],
                                       doors_pos=env_config['topologies'],
                                       agent_dir=env_config['agent directions'],
                                       render_mode="rgb_array"))

        for i in range(1, 51):
            bc = d3rlpy.load_learnable('../models/bc/mixed-suboptimal/dataset_1/44987/model_' + str(i*1000) + '.d3')
            cql = d3rlpy.load_learnable('../models/cql/mixed-suboptimal/dataset_1/44987/model_' + str(i*1000) + '.d3')
            bc_finished_count = 0
            cql_finished_count = 0
            for j in range(len(env_config['topologies'])):
                bc_obs, _ = bc_env.reset()
                bc_done = False

                cql_obs, _ = cql_env.reset()
                cql_done = False
                steps = 0
                bc_steps = 0
                cql_steps = 0
                optimal_steps = 0
                while (not bc_done or not cql_done) and steps < step_cap:
                    if not bc_done:
                        obs_flattened = bc_obs.flatten()[None, :]
                        action = bc.predict(obs_flattened)
                        bc_obs, bc_reward, bc_done, truncated, info = bc_env.step(action[0])
                        bc_steps += 1

                    if not cql_done:
                        obs_flattened = cql_obs.flatten()[None, :]
                        action = cql.predict(obs_flattened)
                        cql_obs, cql_reward, cql_done, truncated, info = cql_env.step(action[0])
                        cql_steps += 1

                    steps += 1

                if cql_reward == 1:
                    cql_finished_count += 1
                else:
                    cql_steps = -1

                if bc_reward == 1:
                    bc_finished_count += 1
                else:
                    bc_steps = -1
            if bc_finished_count != cql_finished_count:
                print(env_index)
                print('bc' + str(bc_finished_count))
                print('cql' + str(cql_finished_count))
                print('')

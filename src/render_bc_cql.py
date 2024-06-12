import argparse
from four_room_stuff.env import FourRoomsEnv
from four_room_stuff.wrappers import gym_wrapper
import imageio
import numpy as np
from pyvirtualdisplay import Display
import dill
import gymnasium as gym
import d3rlpy
import matplotlib.pyplot as plt

gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)

with open('../configs/fourrooms_test_0_config.pl', 'rb') as file:
    train_config = dill.load(file)

dataset_mode = 'optimal'
step_cap = 40

with Display(visible=False) as disp:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()

    bc_env = gym_wrapper(gym.make('MiniGrid-FourRooms-v1',
                           agent_pos=train_config['agent positions'],
                           goal_pos=train_config['goal positions'],
                           doors_pos=train_config['topologies'],
                           agent_dir=train_config['agent directions'],
                           render_mode="rgb_array"))
    cql_env = gym_wrapper(gym.make('MiniGrid-FourRooms-v1',
                           agent_pos=train_config['agent positions'],
                           goal_pos=train_config['goal positions'],
                           doors_pos=train_config['topologies'],
                           agent_dir=train_config['agent directions'],
                           render_mode="rgb_array"))

    bc_images = []
    bc = d3rlpy.load_learnable('../models/bc/' + dataset_mode + '/bc_' + dataset_mode + '_5.d3')
    bc_steps_per_env = []

    cql_images = []
    cql = d3rlpy.load_learnable('../models/cql/' + dataset_mode + '/cql_' + dataset_mode + '_5.d3')
    cql_steps_per_env = []
    for i in range(len(train_config['topologies'])):
        bc_obs, _ = bc_env.reset()
        bc_img = bc_env.render()
        bc_images.append(bc_img)
        bc_done = False

        cql_obs, _ = cql_env.reset()
        cql_img = cql_env.render()
        cql_images.append(cql_img)
        cql_done = False

        steps = 0
        bc_steps = 0
        cql_steps = 0
        while (not bc_done or not cql_done) and steps < step_cap:
            if not bc_done:
                obs_flattened = bc_obs.flatten()[None, :]
                action = bc.predict(obs_flattened)
                bc_obs, reward, bc_done, truncated, info = bc_env.step(action[0])
                bc_img = bc_env.render()
                bc_steps += 1
            bc_images.append(bc_img)

            if not cql_done:
                obs_flattened = cql_obs.flatten()[None, :]
                action = cql.predict(obs_flattened)
                cql_obs, reward, cql_done, truncated, info = cql_env.step(action[0])
                cql_img = cql_env.render()
                cql_steps += 1
            cql_images.append(cql_img)

            steps += 1

        bc_steps_per_env.append(bc_steps)
        cql_steps_per_env.append(cql_steps)

    # Create an array with the positions of the bars on the x axis
    r = np.arange(len(bc_steps_per_env))

    # Create the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(r, bc_steps_per_env, width=0.4, label='BC')
    plt.bar(r + 0.8, cql_steps_per_env, width=0.4, label='CQL')
    plt.xlabel('Environment')
    plt.ylabel('Steps')
    plt.title('Steps per Environment for BC and CQL')
    plt.legend()
    plt.savefig('steps_per_environment_' + dataset_mode + '.png')

    merged_frames = [np.hstack((bc_frame, cql_frame)) for bc_frame, cql_frame in zip(bc_images, cql_images)]

    # Save the merged frames as a new GIF
    imageio.mimsave('merged_' + dataset_mode + '.gif', merged_frames, duration=100)

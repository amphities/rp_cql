import argparse

from stable_baselines3 import DQN

from src.four_room_stuff.shortest_path import find_all_action_values
from src.four_room_stuff.utils import obs_to_state
from four_room_stuff.env import FourRoomsEnv
from four_room_stuff.wrappers import gym_wrapper
import imageio
import numpy as np
from pyvirtualdisplay import Display
import dill
import gymnasium as gym
import d3rlpy

gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)


config_name = 'fourrooms_test_0_config.pl'
step_count = 40

with open('../configs/' + config_name, 'rb') as file:
    config = dill.load(file)
def optimal_policy(state):
    state = obs_to_state(state)
    q_values = find_all_action_values(state[:2], state[2], state[3:5], state[5:], 0.99)
    optimal_action = np.argmax(q_values)
    return optimal_action

with Display(visible=False) as disp:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()

    d3rlpy.seed(args.seed)

    env = gym_wrapper(gym.make('MiniGrid-FourRooms-v1',
                           agent_pos=config['agent positions'],
                           goal_pos=config['goal positions'],
                           doors_pos=config['topologies'],
                           agent_dir=config['agent directions'],
                           render_mode="rgb_array"))
    images = []

    finished = 0
    for i in range(1):
        obs, _ = env.reset()
        img = env.render()
        images.append(img)
        steps = 0
        done = False
        while not done and steps < 0:
            steps += 1
            obs_flattened = obs.flatten()[None, :]
            action = 0
            obs, reward, done, truncated, info = env.step(action)
            if reward == 1:
                finished += 1
            done = done or truncated
            img = env.render()
            images.append(img)

    print(finished)

    gif_name = '../gifs/unreachable.gif'

    # Use the determined gif name when saving the gif
    imageio.mimsave(gif_name, [np.array(img) for i, img in enumerate(images) if i % 1 == 0], duration=1000)

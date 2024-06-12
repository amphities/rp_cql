import os

import numpy as np
import scipy.stats as stats

from src.four_room_stuff.env import FourRoomsEnv
from src.four_room_stuff.wrappers import gym_wrapper
from pyvirtualdisplay import Display
import dill
import gymnasium as gym
import d3rlpy
import matplotlib.pyplot as plt

gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)

with open('configs/fourrooms_train_config.pl', 'rb') as file:
    train_config = dill.load(file)

with open('configs/fourrooms_test_0_config.pl', 'rb') as file:
    unreachable_config = dill.load(file)

with open('configs/fourrooms_test_100_config.pl', 'rb') as file:
    reachable_config = dill.load(file)


def extract_number(filename):
    # Split the filename into parts using underscore as the separator
    parts = filename.split('_')
    # The number is the last part
    number = parts[-1][:-3]
    # Convert the number to an integer and return it
    return int(number)


def get_steps_for_agent(agent, env, env_count):
    total_steps = 0
    for _ in range(env_count):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < step_cap:
            obs_flattened = obs.flatten()[None, :]
            action = agent.predict(obs_flattened)
            obs, _, done, truncated, _ = env.step(action[0])
            done = done or truncated
            steps += 1

        total_steps += steps

    return total_steps

def get_mean_reward_for_agent(agent, env, env_count):
    rewards = []
    for _ in range(env_count):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < step_cap:
            obs_flattened = obs.flatten()[None, :]
            action = agent.predict(obs_flattened)
            obs, reward, done, truncated, _ = env.step(action[0])
            done = done or truncated
            steps += 1

        rewards.append(reward)

    return np.mean(rewards)


def get_step_counts_per_agent(agents, env, env_count):
    step_counts = []
    for agent_index, agent in enumerate(agents):
        step_count = get_steps_for_agent(agent, env, env_count)
        step_counts.append(step_count)

    return step_counts


def plot_agent_overall(ax, training_seeds, dir_path, env, train_env_count, color, offset):
    step_counts = []
    for seed in training_seeds:
        dir_seed_path = dir_path + '/' + str(seed)
        agent_names = sorted(filter(lambda x: x.endswith('.d3'), os.listdir(dir_seed_path)), key=extract_number)

        agents = []
        for agent_name in agent_names:
            agents.append(d3rlpy.load_learnable(dir_seed_path + '/' + agent_name))

        step_counts.append(get_step_counts_per_agent(agents, env, train_env_count))

    mean_steps = np.mean(step_counts, axis=0)
    std_err = stats.sem(step_counts, axis=0)
    confidence_interval = std_err * stats.t.ppf((1 + 0.95) / 2., len(step_counts) - 1)

    ax.bar(np.arange(len(agents)) + offset, mean_steps, color=color, width=1)
    ax.errorbar(np.arange(len(agents)) + offset, mean_steps, yerr=confidence_interval, fmt='o', color=color, capsize=5)
    ax.set_xticks(range(len(agents)), [str((i+1) * 1000) for i in range(len(agents))])

def plot_agent_best_reward_and_rewards(all_rewards_over_steps_plot, mean_reward_over_steps_plot, training_seeds, dataset_seeds, dir_path, env, train_env_count, color, offset, label):
    agents_per_dataset_per_seed = []
    for _ in range(len(dataset_seeds)):
        agents_per_dataset_per_seed.append([])

    for dataset_idx, dataset_seed in enumerate(dataset_seeds):
        for training_seed in training_seeds:
            agents = []
            dir_seed_path = dir_path + '/dataset_' + str(dataset_seed) + '/' + str(training_seed)
            agent_names = sorted(filter(lambda x: x.endswith('.d3'), os.listdir(dir_seed_path)), key=extract_number)

            for agent_name in agent_names:
                agents.append(d3rlpy.load_learnable(dir_seed_path + '/' + agent_name))
            agents_per_dataset_per_seed[dataset_idx].append(agents)

    best_mean_reward = 0
    best_confidence_interval = 0
    mean_rewards = []
    confidence_intervals = []
    print('agents loaded')
    for training_step_idx in range(len(agents_per_dataset_per_seed[0][0])):
        print('evaluating training step', (training_step_idx+1)*1000)
        rewards = []
        for training_seed_idx in range(len(training_seeds)):
            for dataset_seed_idx in range(len(dataset_seeds)):
                agent = agents_per_dataset_per_seed[dataset_seed_idx][training_seed_idx][training_step_idx]
                reward = get_mean_reward_for_agent(agent, env, train_env_count)
                rewards.append(reward)

        mean_reward = np.mean(rewards)
        mean_rewards.append(mean_reward)

        std_err = stats.sem(rewards)
        confidence_interval = std_err * stats.t.ppf((1 + 0.95) / 2., len(rewards) - 1)
        confidence_intervals.append(confidence_interval)

        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            best_confidence_interval = confidence_interval

    mean_reward_over_steps_plot.bar(offset, best_mean_reward, color=color, width=1, label=label)
    mean_reward_over_steps_plot.errorbar(offset, best_mean_reward, yerr=best_confidence_interval, color='black')
    mean_reward_over_steps_plot.set_xticks([])

    lower_bound = np.array(mean_rewards) - np.array(confidence_intervals)
    upper_bound = np.array(mean_rewards) + np.array(confidence_intervals)
    all_rewards_over_steps_plot.plot(range(len(mean_rewards)), mean_rewards, color=color, label=label)
    all_rewards_over_steps_plot.fill_between(range(len(mean_rewards)), lower_bound, upper_bound, color=color, alpha=0.1)
    all_rewards_over_steps_plot.set_xticks(range(0, len(mean_rewards), 5))
    all_rewards_over_steps_plot.set_xticklabels([str((i+1) * 1000) for i in range(0, len(mean_rewards), 5)])


def plot_agent_steps_per_config(env_configs, titles, training_seeds, cql_dir_path, bc_dir_path, plot_save_path):
    for env_index, env_config in enumerate(env_configs):
        with Display(visible=False):
            _, ax = plt.subplots(figsize=(60, 15))
            env = gym_wrapper(gym.make('MiniGrid-FourRooms-v1',
                                        agent_pos=env_config['agent positions'],
                                        goal_pos=env_config['goal positions'],
                                        doors_pos=env_config['topologies'],
                                        agent_dir=env_config['agent directions']))
            plot_agent_overall(ax, training_seeds, bc_dir_path, env, len(env_config['agent positions']), 'blue', offset=-0.5)
            plot_agent_overall(ax, training_seeds, cql_dir_path, env, len(env_config['agent positions']), 'orange', offset=0.5)
            plt.title(titles[env_index])
            plt.xlabel('Number of steps for training')
            plt.ylabel('Number of steps overall environments')

            plt.savefig(f"{plot_save_path}_{env_index}.png", bbox_inches='tight')
            plt.clf()


def plot_agent_reward_per_config(env_configs, titles, training_seeds, dataset_seeds, cql_dir_path, bc_dir_path, plot_save_path):
    for env_index, env_config in enumerate(env_configs):
        print('plotting env', env_index)
        with Display(visible=False):
            _, (all_rewards_over_steps_plot, mean_reward_over_steps_plot) = plt.subplots(1, 2, width_ratios=[20, 1], figsize=(15, 4))
            env = gym_wrapper(gym.make('MiniGrid-FourRooms-v1',
                                        agent_pos=env_config['agent positions'],
                                        goal_pos=env_config['goal positions'],
                                        doors_pos=env_config['topologies'],
                                        agent_dir=env_config['agent directions']))
            plot_agent_best_reward_and_rewards(all_rewards_over_steps_plot, mean_reward_over_steps_plot, training_seeds, dataset_seeds, bc_dir_path, env, len(env_config['agent positions']), 'blue', offset=-0.5, label='bc')
            plot_agent_best_reward_and_rewards(all_rewards_over_steps_plot, mean_reward_over_steps_plot, training_seeds, dataset_seeds, cql_dir_path, env, len(env_config['agent positions']), 'orange', offset=0.5, label='cql')
            all_rewards_over_steps_plot.set_title(titles[env_index])
            all_rewards_over_steps_plot.set_ylabel('Average reward over all environments')
            plt.legend()

            plt.savefig(f"{plot_save_path}_{env_index}.png", bbox_inches='tight')
            plt.clf()



env_configs = [train_config, reachable_config, unreachable_config]
step_cap = 100
titles = [
    'Average reward over models in 1k training step increments (left) and best model average reward (right) for the train set',
    'Average reward over models in 1k training step increments (left) and best model average reward (right) for the reachable set',
    'Average reward over models in 1k training step increments (left) and best model average reward (right) for the unreachable set'
]
bc_path = 'models/bc/mixed_suboptimal_25000'
cql_path = 'models/cql/mixed_suboptimal_25000'
plot_save_path = 'plots/bc_cql/mixed_suboptimal_25000/rewards'
dataset_seeds = [1, 2, 3]
training_seeds = [0, 4219, 17333, 39779, 44987]

plot_agent_reward_per_config(env_configs, titles, training_seeds, dataset_seeds, cql_path, bc_path, plot_save_path)

import os
import pickle

import numpy as np
import scipy.stats as stats

from src.four_room_stuff.env import FourRoomsEnv
from src.four_room_stuff.wrappers import gym_wrapper
import dill
import gymnasium as gym
import d3rlpy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)

with open('../configs/fourrooms_train_config.pl', 'rb') as file:
    train_config = dill.load(file)

with open('../configs/fourrooms_test_0_config.pl', 'rb') as file:
    unreachable_config = dill.load(file)

with open('../configs/fourrooms_test_100_config.pl', 'rb') as file:
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

def get_rewards_for_agent(agent, env, env_count):
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

    return rewards


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


def get_rewards_over_policy(training_seeds, dataset_seeds, dir_path, save_path, env, train_env_count):
    rewards_over_policy = {}
    for dataset_seed in dataset_seeds:
        print('dataset seed', dataset_seed)
        rewards_over_policy[dataset_seed] = {}
        for training_seed in training_seeds:
            print('training seed', training_seed)
            rewards_over_policy[dataset_seed][training_seed] = []
            dir_seed_path = dir_path + '/dataset_' + str(dataset_seed) + '/' + str(training_seed)
            agent_names = sorted(filter(lambda x: x.endswith('.d3'), os.listdir(dir_seed_path)), key=extract_number)
            for agent_name in agent_names:
                agent = d3rlpy.load_learnable(dir_seed_path + '/' + agent_name)
                rewards_over_policy[dataset_seed][training_seed].append(get_rewards_for_agent(agent, env, train_env_count))
    pickle.dump(rewards_over_policy, open(save_path, 'wb'))

def get_best_reward_and_rewards(training_seeds, dataset_seeds, reward_dir_path):
    with open(reward_dir_path, 'rb') as file:
        rewards_over_policy = pickle.load(file)

    best_mean_reward_over_steps = 0
    best_confidence_interval_over_steps = 0
    mean_rewards_over_steps = []
    confidence_intervals_over_steps = []
    for training_step_idx in range(len(rewards_over_policy[dataset_seeds[0]][training_seeds[0]])):
        mean_rewards = []
        for dataset_seed in dataset_seeds:
            for training_seed in training_seeds:
                rewards = rewards_over_policy[dataset_seed][training_seed][training_step_idx]
                mean_reward = np.mean(rewards)
                mean_rewards.append(mean_reward)

        mean_reward_for_step = np.mean(mean_rewards)
        mean_rewards_over_steps.append(mean_reward_for_step)
        std_err = stats.sem(mean_rewards)
        confidence_interval = std_err * stats.t.ppf((1 + 0.95) / 2., len(mean_rewards) - 1)
        confidence_intervals_over_steps.append(confidence_interval)

        if mean_reward_for_step > best_mean_reward_over_steps:
            best_mean_reward_over_steps = mean_reward_for_step
            best_confidence_interval_over_steps = confidence_interval

    return best_mean_reward_over_steps, best_confidence_interval_over_steps, mean_rewards_over_steps, confidence_intervals_over_steps
def plot_agent_rewards(training_seeds, dataset_seeds, reward_dir_path, color, label):
    _, _, mean_rewards_over_steps, confidence_intervals_over_steps = get_best_reward_and_rewards(training_seeds, dataset_seeds, reward_dir_path)

    lower_bound = np.array(mean_rewards_over_steps) - np.array(confidence_intervals_over_steps)
    upper_bound = np.array(mean_rewards_over_steps) + np.array(confidence_intervals_over_steps)
    plt.plot(range(len(mean_rewards_over_steps)), mean_rewards_over_steps, color=color, label=label)
    plt.fill_between(range(len(mean_rewards_over_steps)), lower_bound, upper_bound, color=color, alpha=0.1)
    plt.xticks(range(0, len(mean_rewards_over_steps), 5), [str((i+1) * 1000) for i in range(0, len(mean_rewards_over_steps), 5)])

def plot_agent_reward_per_config(config_dirs, dataset, titles, training_seeds, dataset_seeds, cql_reward_dir, bc_reward_dir, plot_save_path):
    for config_idx, config_dir in enumerate(config_dirs):
        plt.figure(figsize=(15, 5))
        bc_reward_path = bc_reward_dir + '/' + config_dir + '/' + dataset
        cql_reward_path = cql_reward_dir + '/' + config_dir + '/' + dataset
        plot_agent_rewards(training_seeds, dataset_seeds, bc_reward_path, 'blue', label='bc')
        plot_agent_rewards(training_seeds, dataset_seeds, cql_reward_path, 'orange', label='cql')
        plt.title(titles[config_idx])
        plt.ylabel('Average reward over all environments')
        plt.ylim(0, 1)
        plt.legend()

        plt.savefig(f"{plot_save_path}_{config_dir}.png", bbox_inches='tight')
        plt.clf()


def plot_agent_best_reward(ax, training_seeds, dataset_seeds, reward_dir_path, color, offset, label):
    best_mean_reward_over_steps, best_confidence_interval_over_steps, _, _ = get_best_reward_and_rewards(training_seeds, dataset_seeds, reward_dir_path)

    ax.bar(offset, best_mean_reward_over_steps, color=color, width=0.4, label=label)
    ax.errorbar(offset, best_mean_reward_over_steps, yerr=best_confidence_interval_over_steps, color='black')

def plot_best_agent_reward_per_config(training_seeds, dataset_seeds, cql_reward_dir, bc_reward_dir, config_dirs, datasets, plot_save_path):
    for config_idx, config_dir in enumerate(config_dirs):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_ylabel('Average reward over all environments')
        x_labels = []
        for reward_index in range(len(datasets)):
            bc_reward_path = bc_reward_dir + '/' + config_dir + '/' + datasets[reward_index]
            cql_reward_path = cql_reward_dir + '/' + config_dir + '/' + datasets[reward_index]
            plot_agent_best_reward(ax, training_seeds, dataset_seeds, bc_reward_path, 'blue', offset=reward_index - 0.2, label='bc')
            plot_agent_best_reward(ax, training_seeds, dataset_seeds, cql_reward_path, 'orange', offset=reward_index + 0.2, label='cql')
            x_labels.append(datasets[reward_index])
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(x_labels)
        ax.set_ylim(0, 1)
        blue_patch = mpatches.Patch(color='blue', label='bc')
        orange_patch = mpatches.Patch(color='orange', label='cql')
        ax.legend(handles=[blue_patch, orange_patch])

        plt.savefig(f"{plot_save_path}_{config_dir}.png", bbox_inches='tight')
        plt.clf()


step_cap = 100

dataset_seeds = [1, 2, 3]
training_seeds = [0, 4219, 17333, 39779, 44987]
train_env = gym_wrapper(gym.make('MiniGrid-FourRooms-v1',
                                        agent_pos=train_config['agent positions'],
                                        goal_pos=train_config['goal positions'],
                                        doors_pos=train_config['topologies'],
                                        agent_dir=train_config['agent directions']))
reachable_env = gym_wrapper(gym.make('MiniGrid-FourRooms-v1',
                                        agent_pos=reachable_config['agent positions'],
                                        goal_pos=reachable_config['goal positions'],
                                        doors_pos=reachable_config['topologies'],
                                        agent_dir=reachable_config['agent directions']))
unreachable_env = gym_wrapper(gym.make('MiniGrid-FourRooms-v1',
                                        agent_pos=unreachable_config['agent positions'],
                                        goal_pos=unreachable_config['goal positions'],
                                        doors_pos=unreachable_config['topologies'],
                                        agent_dir=unreachable_config['agent directions']))

config_dirs = ['train', 'reachable', 'unreachable']
policy = 'random_walk'
datasets = [policy + '_5000', policy + '_10000', policy + '_25000']
cql_reward_dir = '../rewards/cql'
bc_reward_dir = '../rewards/bc'
for dataset in datasets:
    bc_model_path = '../models/bc/' + dataset
    cql_model_path = '../models/cql/' + dataset
    all_plot_save_path = '../plots/bc_cql/' + dataset + '_all'
    titles = [
        'Average reward for dataset ' + dataset + ' over 1k training step increments for the train set',
        'Average reward for dataset ' + dataset + ' over 1k training step increments for the reachable set',
        'Average reward for dataset ' + dataset + ' over 1k training step increments for the unreachable set',
    ]
    plot_agent_reward_per_config(config_dirs, dataset, titles, training_seeds, dataset_seeds, cql_reward_dir, bc_reward_dir, all_plot_save_path)

best_plot_save_path = '../plots/bc_cql/' + policy + '_best'
plot_best_agent_reward_per_config(training_seeds, dataset_seeds, cql_reward_dir, bc_reward_dir, config_dirs, datasets, best_plot_save_path)




# get_rewards_over_policy(training_seeds, dataset_seeds, cql_path, cql_reward_paths[0], train_env, len(train_config['agent positions']))
# get_rewards_over_policy(training_seeds, dataset_seeds, cql_path, cql_reward_paths[1], reachable_env, len(train_config['agent positions']))
# get_rewards_over_policy(training_seeds, dataset_seeds, cql_path, cql_reward_paths[2], unreachable_env, len(train_config['agent positions']))

# get_rewards_over_policy(training_seeds, dataset_seeds, bc_path, bc_reward_paths[0], train_env, len(train_config['agent positions']))
# get_rewards_over_policy(training_seeds, dataset_seeds, bc_path, bc_reward_paths[1], reachable_env, len(train_config['agent positions']))
# get_rewards_over_policy(training_seeds, dataset_seeds, bc_path, bc_reward_paths[2], unreachable_env, len(train_config['agent positions']))
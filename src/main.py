import pickle
from enum import Enum

import d3rlpy
import dill
import gymnasium as gym
from stable_baselines3 import DQN

from src.create_dataset import create_dataset_from_env
from src.hyper_parameters import get_cql_hyperparameters, get_bc_hyperparameters
from four_room_stuff.wrappers import gym_wrapper
from d3rlpy.algos import DiscreteCQLConfig, DiscreteBCConfig

#dataset creation params
model = DQN.load("../models/dqn_four_rooms")
chance_to_choose_optimal = 0.5
random_walk_step_count_lower_bound = 10
random_walk_step_count_upper_bound = 50

class Dataset_types(Enum):
    OPTIMAL = 'optimal'
    EXPERT_SUBOPTMAL = 'mixed_suboptimal'
    RANDOM_WALK = 'random_walk'

#seeds for reproducible results
training_seeds = [0, 4219, 17333, 39779, 44987]
hyperparameter_seeds = [5, 6, 7, 8, 9]
dataset_seeds = [1, 2, 3]

#cql hyperparameters
available_cql_batch_sizes = [8, 16, 32]
learning_rate_range = [0.0001, 0.01]
alpha_range = [0.1, 5]
n_trials = 50
beta_range = [0.1, 10]
available_bc_batch_sizes = [25, 50, 100]

#plotting params
step_cap = 100

with open('../datasets/optimal_dataset_flattened_372_1.pkl', 'rb') as readFile:
    # Serialize and save the data to the file
    optimal_dataset = pickle.load(readFile)
#
# with open('datasets/mixed_suboptimal_dataset_flattened_1000_1.pkl', 'rb') as readFile:
#     # Serialize and save the data to the file
#     mixed_dataset_1000_1 = pickle.load(readFile)
#
# with open('datasets/mixed_suboptimal_dataset_flattened_1000_2.pkl', 'rb') as readFile:
#     # Serialize and save the data to the file
#     mixed_dataset_1000_2 = pickle.load(readFile)
#
# with open('datasets/mixed_suboptimal_dataset_flattened_1000_3.pkl', 'rb') as readFile:
#     # Serialize and save the data to the file
#     mixed_dataset_1000_3 = pickle.load(readFile)
#
with open('../datasets/mixed_suboptimal_dataset_flattened_5000_1.pkl', 'rb') as readFile:
    # Serialize and save the data to the file
    mixed_dataset_5000_1 = pickle.load(readFile)

with open('../datasets/mixed_suboptimal_dataset_flattened_5000_2.pkl', 'rb') as readFile:
    # Serialize and save the data to the file
    mixed_dataset_5000_2 = pickle.load(readFile)

with open('../datasets/mixed_suboptimal_dataset_flattened_5000_3.pkl', 'rb') as readFile:
    # Serialize and save the data to the file
    mixed_dataset_5000_3 = pickle.load(readFile)

with open('../datasets/mixed_suboptimal_dataset_flattened_10000_1.pkl', 'rb') as readFile:
    # Serialize and save the data to the file
    mixed_dataset_10000_1 = pickle.load(readFile)

with open('../datasets/mixed_suboptimal_dataset_flattened_10000_2.pkl', 'rb') as readFile:
    # Serialize and save the data to the file
    mixed_dataset_10000_2 = pickle.load(readFile)

with open('../datasets/mixed_suboptimal_dataset_flattened_10000_3.pkl', 'rb') as readFile:
    # Serialize and save the data to the file
    mixed_dataset_10000_3 = pickle.load(readFile)

with open('../datasets/mixed_suboptimal_dataset_flattened_25000_1.pkl', 'rb') as readFile:
    # Serialize and save the data to the file
    mixed_dataset_25000_1 = pickle.load(readFile)

with open('../datasets/mixed_suboptimal_dataset_flattened_25000_2.pkl', 'rb') as readFile:
    # Serialize and save the data to the file
    mixed_dataset_25000_2 = pickle.load(readFile)

with open('../datasets/mixed_suboptimal_dataset_flattened_25000_3.pkl', 'rb') as readFile:
    # Serialize and save the data to the file
    mixed_dataset_25000_3 = pickle.load(readFile)

with open('../configs/fourrooms_train_config.pl', 'rb') as readFile:
    train_env_config = dill.load(readFile)

with open('../datasets/random_walk_dataset_flattened_5000_1.pkl', 'rb') as readFile:
    # Serialize and save the data to the file
    random_walk_dataset_5000_1 = pickle.load(readFile)

with open('../datasets/random_walk_dataset_flattened_5000_2.pkl', 'rb') as readFile:
    # Serialize and save the data to the file
    random_walk_dataset_5000_2 = pickle.load(readFile)

with open('../datasets/random_walk_dataset_flattened_5000_3.pkl', 'rb') as readFile:
    # Serialize and save the data to the file
    random_walk_dataset_5000_3 = pickle.load(readFile)

with open('../datasets/random_walk_dataset_flattened_10000_1.pkl', 'rb') as readFile:
    # Serialize and save the data to the file
    random_walk_dataset_10000_1 = pickle.load(readFile)

with open('../datasets/random_walk_dataset_flattened_10000_2.pkl', 'rb') as readFile:
    # Serialize and save the data to the file
    random_walk_dataset_10000_2 = pickle.load(readFile)

with open('../datasets/random_walk_dataset_flattened_10000_3.pkl', 'rb') as readFile:
    # Serialize and save the data to the file
    random_walk_dataset_10000_3 = pickle.load(readFile)

with open('../datasets/random_walk_dataset_flattened_25000_1.pkl', 'rb') as readFile:
    # Serialize and save the data to the file
    random_walk_dataset_25000_1 = pickle.load(readFile)

with open('../datasets/random_walk_dataset_flattened_25000_2.pkl', 'rb') as readFile:
    # Serialize and save the data to the file
    random_walk_dataset_25000_2 = pickle.load(readFile)

with open('../datasets/random_walk_dataset_flattened_25000_3.pkl', 'rb') as readFile:
    # Serialize and save the data to the file
    random_walk_dataset_25000_3 = pickle.load(readFile)


def create_dataset(dataset_type, size, seed):
    # seed is used for datasets that use actions_space.sample()
    train_env = gym_wrapper(gym.make('MiniGrid-FourRooms-v1',
                                 agent_pos=train_env_config['agent positions'],
                                 goal_pos=train_env_config['goal positions'],
                                 doors_pos=train_env_config['topologies'],
                                 agent_dir=train_env_config['agent directions']))
    return create_dataset_from_env(train_env, dataset_type, size, model, chance_to_choose_optimal, random_walk_step_count_lower_bound, random_walk_step_count_upper_bound, seed)


def run_cql_hyperparmeter_search(policy_datasets, dataset_type, dataset_seeds):
    evaluation_env_count = len(train_env_config['agent positions'])
    evaluation_env = gym_wrapper(gym.make('MiniGrid-FourRooms-v1',
                                          agent_pos=train_env_config['agent positions'],
                                          goal_pos=train_env_config['goal positions'],
                                          doors_pos=train_env_config['topologies'],
                                          agent_dir=train_env_config['agent directions']))

    for dataset_idx, dataset in enumerate(policy_datasets):
        #run hyperparameter search for each dataset/optimization seed by training on 20k steps using Optuna which
        #uses TPE to optimize the hyperparameters
        hyper_params = get_cql_hyperparameters(
            #hyperparameters I'm searching for
            available_batch_sizes=available_cql_batch_sizes,
            learning_rate_range=learning_rate_range,
            alpha_range=alpha_range,
            #details for running the optimizer
            n_trials=n_trials,
            seeds=hyperparameter_seeds,
            evaluation_env=evaluation_env,
            step_cap=step_cap,
            evaluation_env_count=evaluation_env_count,
            dataset=dataset
        ).best_params
        pickle.dump(hyper_params, open('../hyper_params/hyper_params_cql_' + dataset_type + '_' + str(dataset_seeds[dataset_idx]) + '.pkl', 'wb'))


def run_bc_hyperparmeter_search(policy_datasets, dataset_type, dataset_seeds):
    evaluation_env_count = len(train_env_config['agent positions'])
    evaluation_env = gym_wrapper(gym.make('MiniGrid-FourRooms-v1',
                                          agent_pos=train_env_config['agent positions'],
                                          goal_pos=train_env_config['goal positions'],
                                          doors_pos=train_env_config['topologies'],
                                          agent_dir=train_env_config['agent directions']))

    for dataset_idx, dataset in enumerate(policy_datasets):
        #run hyperparameter search for each dataset/optimization seed by training on 20k steps using Optuna which
        #uses TPE to optimize the hyperparameters
        hyper_params = get_bc_hyperparameters(
            #hyperparameters I'm searching for
            available_batch_sizes=available_bc_batch_sizes,
            learning_rate_range=learning_rate_range,
            beta_range=beta_range,
            #details for running the optimizer
            n_trials=n_trials,
            seeds=hyperparameter_seeds,
            evaluation_env=evaluation_env,
            step_cap=step_cap,
            evaluation_env_count=evaluation_env_count,
            dataset=dataset
        ).best_params
        pickle.dump(hyper_params, open('../hyper_params/hyper_params_bc_' + dataset_type + '_' + str(dataset_seeds[dataset_idx]) + '.pkl', 'wb'))

def train_agent(agent, n_steps, dataset):
    agent.fit(
        dataset,
        n_steps=n_steps,
        n_steps_per_epoch=1000,
    )


def train_cql(policy_datasets, dataset_type, dataset_seeds):
    for dataset_idx, dataset in enumerate(policy_datasets):
        hyper_params = pickle.load(open('../hyper_params/hyper_params_cql_' + dataset_type + '_' + str(dataset_seeds[dataset_idx]) + '.pkl', 'rb'))
        print(hyper_params)
        batch_size = hyper_params['batch_size']
        learning_rate = hyper_params['learning_rate']
        alpha = hyper_params['alpha']
        #for each training seed
        for training_seed in training_seeds:
            #for the range of 1k to 50k steps train a model with the hyperparams
            d3rlpy.seed(training_seed)
            agent = DiscreteCQLConfig(
                batch_size=batch_size,
                learning_rate=learning_rate,
                alpha=alpha
            ).create(device=True)
            train_agent(agent, 50000, dataset)


def train_bc(policy_datasets, dataset_type, dataset_seeds):
    for dataset_idx, dataset in enumerate(policy_datasets):
        hyper_params = pickle.load(open('../hyper_params/hyper_params_bc_' + dataset_type + '_' + str(dataset_seeds[dataset_idx]) + '.pkl', 'rb'))
        print(hyper_params)
        batch_size = hyper_params['batch_size']
        learning_rate = hyper_params['learning_rate']
        beta = hyper_params['beta']
        #for each training seed
        for training_seed in training_seeds:
            #for the range of 1k to 50k steps train a model with the hyperparams
            d3rlpy.seed(training_seed)
            agent = DiscreteBCConfig(
                batch_size=batch_size,
                learning_rate=learning_rate,
                beta=beta
            ).create(device=True)
            train_agent(agent, 50000, dataset)


mixed_suboptimal_policy_5000 = [mixed_dataset_5000_1, mixed_dataset_5000_2, mixed_dataset_5000_3]
mixed_suboptimal_policy_10000 = [mixed_dataset_10000_1, mixed_dataset_10000_2, mixed_dataset_10000_3]
mixed_suboptimal_policy_25000 = [mixed_dataset_25000_1, mixed_dataset_25000_2, mixed_dataset_25000_3]

random_walk_policy_5000 = [random_walk_dataset_5000_1, random_walk_dataset_5000_2, random_walk_dataset_5000_3]
random_walk_policy_10000 = [random_walk_dataset_10000_1, random_walk_dataset_10000_2, random_walk_dataset_10000_3]
random_walk_policy_25000 = [random_walk_dataset_25000_1, random_walk_dataset_25000_2, random_walk_dataset_25000_3]

from optuna.samplers import TPESampler
import d3rlpy
import gymnasium as gym
import optuna
from functools import partial

from src.four_room_stuff.env import FourRoomsEnv
gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)

def cql_objective(available_batch_sizes, learning_rate_change, alpha_range, evaluation_env, step_cap, evaluation_env_count, dataset, seeds, trial):
    batch_size = trial.suggest_categorical('batch_size', available_batch_sizes)
    learning_rate = trial.suggest_float('learning_rate', learning_rate_change[0], learning_rate_change[1], log=True)
    alpha = trial.suggest_float('alpha', alpha_range[0], alpha_range[1])

    total_steps = 0
    for seed in seeds:
        d3rlpy.seed(seed)
        agent = d3rlpy.algos.DiscreteCQLConfig(batch_size=batch_size, learning_rate=learning_rate, alpha=alpha).create(device=True)
        agent.fit(
            dataset,
            n_steps=20000,
            n_steps_per_epoch=1000,
        )

        for _ in range(evaluation_env_count):
            obs, _ = evaluation_env.reset()
            done = False
            steps = 0
            while not done and steps < step_cap:
                obs_flattened = obs.flatten()[None, :]
                action = agent.predict(obs_flattened)
                obs, _, done, _, _ = evaluation_env.step(action[0])
                steps += 1

            total_steps += steps

    return total_steps

def bc_objective(available_batch_sizes, learning_rate_change, beta_range, evaluation_env, step_cap, evaluation_env_count, dataset, seeds, trial):

    batch_size = trial.suggest_categorical('batch_size', available_batch_sizes)
    learning_rate = trial.suggest_float('learning_rate', learning_rate_change[0], learning_rate_change[1], log=True)
    beta = trial.suggest_float('beta', beta_range[0], beta_range[1])

    total_steps = 0
    for seed in seeds:
        d3rlpy.seed(seed)
        agent = d3rlpy.algos.DiscreteBCConfig(batch_size=batch_size, learning_rate=learning_rate, beta=beta).create(device=True)
        agent.fit(
            dataset,
            n_steps=20000,
            n_steps_per_epoch=1000,
        )

        for _ in range(evaluation_env_count):
            obs, _ = evaluation_env.reset()
            done = False
            steps = 0
            while not done and steps < step_cap:
                obs_flattened = obs.flatten()[None, :]
                action = agent.predict(obs_flattened)
                obs, _, done, _, _ = evaluation_env.step(action[0])
                steps += 1

            total_steps += steps

    return total_steps

def get_cql_hyperparameters(available_batch_sizes, learning_rate_range, alpha_range, n_trials,
                            seeds, evaluation_env, step_cap, evaluation_env_count, dataset):
    sampler = TPESampler(seed=1)
    study = optuna.create_study(sampler=sampler)

    # Curry the cql_objective function
    curried_cql_objective = partial(cql_objective, available_batch_sizes, learning_rate_range,
                                    alpha_range, evaluation_env, step_cap, evaluation_env_count, dataset, seeds)

    study.optimize(curried_cql_objective, n_trials=n_trials)
    return study

def get_bc_hyperparameters(available_batch_sizes, learning_rate_range, beta_range, n_trials,
                           seeds, evaluation_env, step_cap, evaluation_env_count, dataset):
    sampler = TPESampler(seed=1)
    study = optuna.create_study(sampler=sampler)

    # Curry the bc_objective function
    curried_bc_objective = partial(bc_objective, available_batch_sizes, learning_rate_range,
                                   beta_range, evaluation_env, step_cap, evaluation_env_count, dataset, seeds)

    study.optimize(curried_bc_objective, n_trials=n_trials)
    return study

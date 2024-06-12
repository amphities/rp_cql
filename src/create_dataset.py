import gymnasium as gym
import numpy as np
from d3rlpy.dataset import MDPDataset
import pickle
import random

from four_room_stuff.env import FourRoomsEnv
from four_room_stuff.shortest_path import find_all_action_values
from four_room_stuff.utils import obs_to_state

gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)

# Define your custom policy here
def optimal_policy(state):
    state = obs_to_state(state)
    q_values = find_all_action_values(state[:2], state[2], state[3:5], state[5:], 0.99)
    optimal_action = np.argmax(q_values)
    return optimal_action

def record_agent_actions(env, dataset_mode, dataset_size, model, chance_to_choose_optimal,
                         random_walk_lower_bound, random_walk_upper_bound,
                         states, actions, rewards, terminal_flags):
    steps = 0
    while steps < dataset_size:
        state, _ = env.reset()
        action = optimal_policy(state)
        done = False
        random_walk_steps_remaining = random.randrange(random_walk_lower_bound, random_walk_upper_bound)
        while not done:
            match dataset_mode:
                case 'random':
                    action = env.action_space.sample()
                case 'mixed_suboptimal':
                    if random.random() < chance_to_choose_optimal:
                        action = optimal_policy(state)
                    else:
                        action = model.predict(state)[0]
                case 'optimal':
                    action = optimal_policy(state)
                case 'random_walk':
                    if random_walk_steps_remaining > 0:
                        action = env.action_space.sample()
                        random_walk_steps_remaining -= 1
                    else:
                        action = optimal_policy(state)

            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated
            states.append(state.flatten())
            actions.append(action)
            rewards.append(reward)
            terminal_flags.append(done)
            state = next_state
            steps += 1

def create_dataset_from_env(env, dataset_mode, dataset_size, model, chance_to_choose_optimal,
                            random_walk_lower_bound, random_walk_upper_bound, seed):
    #initialize seeds for dataset creation
    random.seed(seed)
    env.reset(seed=seed)

    states = []
    actions = []
    rewards = []
    terminal_flags = []
    if dataset_mode != 'random_walk':
        record_agent_actions(env, dataset_mode, dataset_size, model, chance_to_choose_optimal,
                             random_walk_lower_bound, random_walk_upper_bound,
                             states, actions, rewards, terminal_flags)
    else:
        # Record an expert trajectory for each environment. This is done within 372 transitions. Then perform random walks.
        record_agent_actions(env, 'optimal', 372, model, chance_to_choose_optimal,
                             random_walk_lower_bound, random_walk_upper_bound,
                             states, actions, rewards, terminal_flags)
        record_agent_actions(env, 'random_walk', (dataset_size - 372), model, chance_to_choose_optimal,
                             random_walk_lower_bound, random_walk_upper_bound,
                             states, actions, rewards, terminal_flags)

    # Convert lists to numpy arrays
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    terminal_flags = np.array(terminal_flags)

    # Create MDPDataset
    dataset = MDPDataset(states, actions, rewards, terminal_flags, action_size=3)

    #Save the dataset
    with open(f'../datasets/{dataset_mode}_dataset_flattened_{dataset_size}_{seed}.pkl', 'wb') as writeFile:
        # Serialize and save the data to the file
        pickle.dump(dataset, writeFile)
    return dataset

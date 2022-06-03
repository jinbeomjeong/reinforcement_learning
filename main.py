import gym
import numpy as np
import time, math, random

from typing import Tuple
from sklearn.preprocessing import KBinsDiscretizer
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

env = gym.make('CartPole-v1')
start_time = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
writer = SummaryWriter(".\\logs\\cart_pole" + '_' + start_time)

'''
policy = lambda obs: 1

for _ in range(1):
    obs = env.reset()

    for _ in range(80):
        actions = policy(obs)
        obs, reward, done, info = env.step(actions)
        env.render()
        time.sleep(0.05)

env.close()
'''

policy = lambda _,__,___, tip_velocity : int(tip_velocity > 0)

n_bins = (6, 12)
lower_bounds = [env.observation_space.low[2], -math.radians(50)]
upper_bounds = [env.observation_space.high[2], math.radians(50)]


def discretizer( _ , __ , angle, pole_velocity ) -> Tuple[int,...]:
    """Convert continues state intro a discrete state"""
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    est.fit([lower_bounds, upper_bounds])

    return tuple(map(int, est.transform([[angle, pole_velocity]])[0]))


Q_table = np.zeros(n_bins + (env.action_space.n,))


def policy(state: tuple):
    return np.argmax(Q_table[state])


def new_Q_value(reward: float,  new_state: tuple, discount_factor=1) -> float:
    future_optimal_value = np.max(Q_table[new_state])
    learned_value = reward + discount_factor * future_optimal_value
    return learned_value


# Adaptive learning of Learning Rate
def learning_rate(n: int, min_rate=0.01) -> float:
    """Decaying learning rate"""
    return max(min_rate, min(1.0, 1.0 - math.log10((n + 1) / 25)))


def exploration_rate(n: int, min_rate=0.1) -> float:
    """Decaying exploration rate"""
    return max(min_rate, min(1, 1.0 - math.log10((n  + 1) / 25)))


n_episodes = 10000
for e in tqdm(range(n_episodes), ncols=100, desc='training...'):

    # Siscretize state into buckets
    current_state, done = discretizer(*env.reset()), False
    #print(current_state)

    while done == False:

        # policy action
        action = policy(current_state)  # exploit

        # insert random action
        if np.random.random() < exploration_rate(e):
            action = env.action_space.sample()  # explore

        # increment enviroment
        obs, reward, done, _ = env.step(action)
        new_state = discretizer(*obs)

        # Update Q-Table
        lr = learning_rate(e)
        learnt_value = new_Q_value(reward, new_state)
        old_value = Q_table[current_state][action]
        Q_table[current_state][action] = (1 - lr) * old_value + lr * learnt_value
        #print(obs)
        #print(new_state)
        #writer.add_scalar('action', action, e)
        #writer.add_scalar('reward', reward, e)
        current_state = new_state

        # Render the cartpole environment
        env.render()
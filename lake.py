import gym
import numpy as np
import time
import pygame
from tensorboardX import SummaryWriter
from tqdm import tqdm


writer = SummaryWriter()
env = gym.make('FrozenLake-v1', is_slippery=False)
print(env.action_space)
print(env.observation_space)

# Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Set learning parameters
lr = 0.8
y = 0.95
num_episodes = 2000

# create lists to contain total rewards and steps per episode
rList = []

for i in tqdm(range(num_episodes), desc='episode progress', ncols=100):
    # Reset environment and get first new observation
    s = env.reset()
    d = False
    j = 0
    r = 0
    # The Q-Table learning algorithm
    while j < 99:
        j += 1

        # Choose an action by greedily (with noise) picking from Q table
        a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))

        # Get new state and reward from environment
        s1, r, d, _ = env.step(a)

        # Update Q-Table with new knowledge
        Q[s, a] = Q[s, a] + lr * (r + y * np.max(Q[s1, :]) - Q[s, a])
        s = s1

        env.render()
        time.sleep(0.05)

        if d == True: break

    writer.add_scalar('reward', r, i)


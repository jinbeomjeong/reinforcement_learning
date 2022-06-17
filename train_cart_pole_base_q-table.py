import numpy as np
import gym
import time
import pickle
import random
from sklearn.preprocessing import KBinsDiscretizer
from tensorboardX import SummaryWriter
from tqdm import tqdm


n_episodes = 500000
lr = 0.6
df = 0.9
logging = True
save_q_table = True

env = gym.make('CartPole-v1')

est_obs_space_size = (100, 100, 100, 100)

lower_bounds = [-5, -5, -500, -4]
upper_bounds = [5, 5, 500, 4]

est_obs_space = KBinsDiscretizer(n_bins=est_obs_space_size, encode='ordinal', strategy='uniform')
est_obs_space.fit([lower_bounds, upper_bounds])

q_table = np.zeros(est_obs_space_size + (env.action_space.n, ))
disc_angular_position_current, disc_angular_speed_current = 0, 0
average_reward = 0

if logging:
    writer = SummaryWriter()

loop = tqdm(range(n_episodes))
j = 0


def policy(q_table, est_obs_space_current):
    return np.argmax(q_table[est_obs_space_current[0], est_obs_space_current[1], est_obs_space_current[2], est_obs_space_current[3]])


for i in loop:
    reward_sum, done = 0, False
    observer_current = env.reset()
    epsilon = np.log10(i + 1) / np.log10(n_episodes)

    est_obs_space_current = est_obs_space.transform([observer_current])[0].astype(int)
    # disc_angular_position_current, disc_angular_speed_current = int(est_obs_space_current[0]), int(est_obs_space_current[1])

    while not done:
        est_obs_space_current = est_obs_space.transform([observer_current])[0].astype(int)
        # disc_angular_position_current, disc_angular_speed_current = int(est_obs_space_current[0]), int(est_obs_space_current[1])

        if random.random() > epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[est_obs_space_current[0], est_obs_space_current[1], est_obs_space_current[2], est_obs_space_current[3]])

        observer_future, reward, done, info = env.step(action)
        reward_user = -1 if done else reward
        reward_sum += reward
        est_obs_space_future = est_obs_space.transform([observer_future])[0].astype(int)
        #disc_angular_position_future, disc_angular_speed_future = int(est_obs_space_future[0]), int(est_obs_space_future[1])
        position_current = est_obs_space_current[0]
        position_future = est_obs_space_future[0]
        velocity_current = est_obs_space_current[1]
        velocity_future = est_obs_space_future[1]
        angular_position_current = est_obs_space_current[2]
        angular_position_future = est_obs_space_future[2]
        angular_speed_current = est_obs_space_current[3]
        angular_speed_future = est_obs_space_future[3]

        current_q_value = q_table[position_current, velocity_current, angular_position_current, angular_speed_current, action]
        max_q_value = np.max(q_table[position_future, velocity_future, angular_position_future, angular_speed_future])
        new_q_value = (1 - lr) * current_q_value + (lr * (reward_user + df * max_q_value))
        q_table[position_current, velocity_current, angular_position_current, angular_speed_current][action] = new_q_value
        observer_current = observer_future
        j += 1

    if i > 1:
        average_reward = ((i-1)/i) * average_reward + (1-((i-1)/i)) * reward_sum

    if logging:
        writer.add_scalar('reward_sum', reward_sum, i)

    loop.set_description(f'epsilon [{epsilon:.2f}/{1}]')
    loop.set_postfix(total_reward=reward_sum, average_reward=average_reward)

env.close()

if save_q_table:
    with open('q_table.pkl', 'wb') as file:
        pickle.dump(q_table, file)

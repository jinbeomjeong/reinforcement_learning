import numpy as np
import gym
import pickle
import time
import random
from sklearn.preprocessing import KBinsDiscretizer
from tqdm import tqdm


n_episodes = 20

env = gym.make('CartPole-v1')

est_obs_space_size = (100, 100, 100, 100)

lower_bounds = [-5, -5, -500, -4]
upper_bounds = [5, 5, 500, 4]

est_obs_space = KBinsDiscretizer(n_bins=est_obs_space_size, encode='ordinal', strategy='uniform')
est_obs_space.fit([lower_bounds, upper_bounds])

with open('.//q_table.pkl', 'rb') as file:
    q_table = pickle.load(file)

for i in range(n_episodes):
    reward_sum, done = 0, False
    observer_current = env.reset()
    est_obs_space_current = est_obs_space.transform([observer_current])[0].astype(int)
    #disc_angular_position_current, disc_angular_speed_current = int(est_obs_space_current[0]), int(est_obs_space_current[1])

    while not done:
        if random.random() < 0.9:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[est_obs_space_current[0], est_obs_space_current[1], est_obs_space_current[2], est_obs_space_current[3]])

        observer_future, reward, done, info = env.step(action)
        reward_sum += reward
        est_obs_space_future = est_obs_space.transform([observer_future])[0].astype(int)
        #disc_angular_position_future, disc_angular_speed_future = int(est_obs_space_future[0]), int(est_obs_space_future[1])
        observer_current = observer_future
        reward_sum += reward
        env.render()
        time.sleep(0.01)

    print(reward_sum)
env.close()

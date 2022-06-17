import gym
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from sub_program.dqn_utils import ReplayMemory, QNetwork, train_q_network

# Hyper parameters
learning_rate = 0.001
df = 0.95
batch_size = 64
n_episodes = 30000
exp = 0.8

writer = SummaryWriter()
env = gym.make('CartPole-v1')
actual_q_network = QNetwork()
target_q_network = QNetwork()
target_q_network.load_state_dict(actual_q_network.state_dict())
memory = ReplayMemory(buffer_limit=2000)

print_interval = 20
score = 0.0
optimizer = optim.Adam(actual_q_network.parameters(), lr=learning_rate)

loop = tqdm(range(n_episodes), ncols=100)
for episode in loop:
    reward_sum = 0
    epsilon = max(0.01, 0.08 - 0.01 * (episode / n_episodes))  # Linear annealing from 8% to 1%
    observer_current = env.reset()
    done = False

    while not done:
        action = actual_q_network.sample_action(torch.from_numpy(observer_current).float(), exp > np.log10(episode+1) / np.log10(n_episodes))
        observer_future, reward, done, info = env.step(action)
        done_mask = 0.0 if done else 1.0
        transition = (observer_current, action, reward, observer_future, done_mask)
        memory.put(transition)
        observer_current = observer_future
        reward_sum += reward

        if done:
            break

    if memory.print_buffer_size() > 2000:
        train_q_network(actual_q_network, target_q_network, memory, optimizer, batch_size, df)

    if episode % print_interval == 0 and episode != 0:
        target_q_network.load_state_dict(actual_q_network.state_dict())

    loop.set_description(f'train [{episode}/{n_episodes}]')
    loop.set_postfix(total_reward=reward_sum)
    writer.add_scalar('reward_sum', reward_sum, episode)

env.close()
torch.save(actual_q_network, 'cart_pole_dqn.pt')

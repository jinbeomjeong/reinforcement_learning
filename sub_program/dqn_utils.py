import collections
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as nn_func


gpu = torch.device('cuda')


class ReplayMemory():
    def __init__(self, buffer_limit):
        self.buffer_limit = buffer_limit
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n_sample):
        mini_batch = random.sample(self.buffer, n_sample)
        observer_current_list, action_list, reward_list, observer_future_list, done_mask_list = [], [], [], [], []

        for transition in mini_batch:
            observer_current, action, reward, observer_future, done_mask = transition
            observer_current_list.append(observer_current)
            action_list.append([action])
            reward_list.append([reward])
            observer_future_list.append(observer_future)
            done_mask_list.append([done_mask])

        return torch.FloatTensor(observer_current_list), \
               torch.tensor(action_list), \
               torch.tensor(reward_list), \
               torch.FloatTensor(observer_future_list), \
               torch.tensor(done_mask_list)

    def print_buffer_size(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(4, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, 2)

    def forward(self, x):
        x = nn_func.relu(self.fc1(x))
        x = nn_func.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, observer_current, exploration: bool):
        action = self.forward(observer_current)

        if exploration:
            return random.randint(0, 1)
        else:
            return action.argmax().item()


def train_q_network(q_actual, q_target, memory, optimizer, batch_size, df):
    for i in range(10):
        observer_current, action, reward, observer_future, done_mask = memory.sample(batch_size)

        q_actual_out = q_actual(observer_current)
        q_actual_action = q_actual_out.gather(1, action)
        max_q_target = q_target(observer_future).max(1)[0].unsqueeze(1)
        target = reward + df * max_q_target * done_mask
        loss = nn_func.smooth_l1_loss(q_actual_action, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

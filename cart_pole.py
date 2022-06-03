import numpy as np
import gym
import time
from sub_program.tcp_lib import tcp_client_com


n_episode = 100

tcp_task = tcp_client_com()
tcp_task.connect_to_server()

env = gym.make('CartPole-v1')
observer = env.reset()
print(observer)

for i in range(1000):
    env.render()
    action = env.action_space.sample()

    if observer[2] > 0:
        action = 1
    else:
        action = 0

    observer, reward, done, info = env.step(action)
    tcp_task.send_msg(f'{i:.0f}, {observer[0]:.3f}, {observer[1]:.3f}, {observer[2]:.3f}, {observer[3]:.3f}, {action:.0f}')
    time.sleep(0.01)


    print(reward)
env.close()

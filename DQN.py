import numpy as np
from homework2 import Hw2Env
import torch
import torch.nn as nn
import random

class DQN(nn.Module):
    def __init__(self, N_actions):
        super().__init__()

        self.cnetwork = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.ReLU(),
            nn.AvgPool2d(4),
            nn.Flatten(0)
        )
        self.lnetwork = nn.Linear(512, N_actions)

    def forward(self, state):
        imageout = self.cnetwork(state)
        q_out = self.lnetwork(imageout)
        return q_out


def get_action(epsilon, main_network : DQN, state):
    rnd = random.random()
    if rnd < epsilon:
        return random.randint(0,7)
    else:
        q_vals = main_network(state)
        print(q_vals)


N_ACTIONS = 8

num_episodes = 10 #10000
update_frequency = 10
target_update_frequency = 200
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05
batch_size=64
gamma=0.99
buffer_size=100000


main_network = DQN(N_ACTIONS)
target_network = DQN(N_ACTIONS)




env = Hw2Env(n_actions=N_ACTIONS, render_mode="gui")
for episode in range(num_episodes):
    env.reset()
    done = False
    cumulative_reward = 0.0
    episode_steps = 0
    while not done:
        action = np.random.randint(N_ACTIONS)
        state, reward, is_terminal, is_truncated = env.step(action)
        get_action(0.01, main_network, state)
        quit(0)
        done = is_terminal or is_truncated
        cumulative_reward += reward
        episode_steps += 1
    print(f"Episode={episode}, reward={cumulative_reward}, RPS={cumulative_reward/episode_steps}")
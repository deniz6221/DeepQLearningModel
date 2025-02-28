import numpy as np
from homework2 import Hw2Env
import torch
import torch.nn as nn
import random
from collections import deque

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

# Epsilon greedy policy
def get_action(epsilon, main_network : DQN, state):
    rnd = random.random()
    if rnd < epsilon:
        return random.randint(0,7)
    else:
        q_vals = main_network(state)
        act = torch.argmax(q_vals)[0]
        return act
        

def update_params(buffer, batch_size, main_network: DQN, target_network: DQN, optimizer):
    if len(buffer) < batch_size:
        return
    
    batch = random.sample(batch_size)

    state_tensor = torch.zeros([batch_size, 3, 128, 128], dtype=torch.double)
    next_state_tensor = torch.zeros([batch_size, 3, 128, 128], dtype=torch.double)
    action_tensor = torch.zeros(batch_size, dtype=torch.int)
    reward_tensor = torch.zeros(batch_size, dtype=torch.double)
    done_tensor = torch.zeros(batch_size, dtype=torch.int)

    ctr = 0
    for i in batch:
        state_tensor[ctr] = i["state"]
        next_state_tensor[ctr] = i["next_state"]
        action_tensor[ctr] = i["action"]
        reward_tensor[ctr] = i["reward"]
        done_tensor[ctr] = i["done"]
        ctr += 1

    q_out = main_network(state_tensor).gather(1, action_tensor).squeeze()   

    with torch.no_grad():
        q_target = torch.max(target_network(next_state_tensor),1)[0]
        expected_reward = reward_tensor + q_target





if __name__ == "__main__":

    normalization_tensor = torch.tensor([[[255 for _ in range(128)] for _ in range(128)] for _ in range(3)], dtype=torch.double)

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
    learning_rate = 0.001

    main_network = DQN(N_ACTIONS)
    target_network = DQN(N_ACTIONS)
    target_network.load_state_dict(main_network.state_dict())
    target_network.eval()
    main_network.train()

    optimizer = torch.optim.Adam(params=main_network.parameters(), lr=learning_rate)

    buffer = deque(maxlen=buffer_size)

    env = Hw2Env(n_actions=N_ACTIONS, render_mode="gui")
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        cumulative_reward = 0.0
        episode_steps = 0
        while not done:
            action = get_action(epsilon, main_network, state)
            
            next_state, reward, is_terminal, is_truncated = env.step(action)
            done = is_terminal or is_truncated
            buffer.append({"state": state, "action": action,"reward": reward, "next_state": next_state, "done": 0 if done else 1})

            state = next_state    
            cumulative_reward += reward
            episode_steps += 1
            if episode_steps % target_update_frequency == 0: 
                target_network.load_state_dict(main_network.state_dict())
            if episode_steps % update_frequency == 0:
                update_params(buffer, batch_size, main_network, target_network, optimizer)

        epsilon = max(epsilon_min, epsilon_decay * epsilon)        

        print(f"Episode={episode}, reward={cumulative_reward}, RPS={cumulative_reward/episode_steps}")
import torch
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
from model import DQN

class DQNAgent:
    def __init__(self, lr=1e-3, gamma=0.99, epsilon=1.0, device='cpu'):
        self.model = DQN().to(device)
        self.target = DQN().to(device)
        self.target.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device
        self.memory = deque(maxlen=10000)

    def act(self, state, legal_actions):
        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
            q_values = self.model(state_tensor)
            q_values = q_values.cpu().numpy()
            q_values = [q if i in legal_actions else -float('inf') for i, q in enumerate(q_values)]
            return int(np.argmax(q_values))

    def remember(self, s, a, s_, r):
        self.memory.append((s, a, s_, r))

    def train_step(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, next_states, rewards = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards).unsqueeze(1).to(self.device)

        q_values = self.model(states).gather(1, actions)
        with torch.no_grad():
            max_next_q = self.target(next_states).max(1)[0].unsqueeze(1)
        targets = rewards + self.gamma * max_next_q

        loss = F.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

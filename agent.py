import torch
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
from model import DQN

# --- Hyperparameters ---
LR = 1e-4                  # Learning rate
GAMMA = 0.95               # Discount factor
EPSILON_START = 1.0        # Starting exploration rate
EPSILON_END = 0.01         # Final exploration rate
EPSILON_DECAY = 15000      # How fast to decay exploration
MEMORY_SIZE = 50000        # Replay buffer size
BATCH_SIZE = 128           # Training batch size

class DQNAgent:
    def __init__(self, device='cpu'):
        self.device = device
        self.gamma = GAMMA
        self.steps_done = 0

        self.model = DQN().to(self.device)
        self.target = DQN().to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        self.target.eval() # Target network is for inference only

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=LR, amsgrad=True)
        self.memory = deque(maxlen=MEMORY_SIZE)

    def act(self, state, legal_actions):
        """Choose an action using epsilon-greedy policy."""
        # Epsilon decay
        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                  np.exp(-1. * self.steps_done / EPSILON_DECAY)
        self.steps_done += 1

        if random.random() < epsilon:
            return random.choice(legal_actions)
        
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.model(state_tensor)
            
            # Mask illegal moves by setting their Q-values to negative infinity
            mask = torch.full_like(q_values, -float('inf'))
            mask[0, legal_actions] = 0
            masked_q_values = q_values + mask
            
            return masked_q_values.argmax().item()

    def remember(self, state, action, next_state, reward, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, next_state, reward, done))

    def train_step(self):
        """Train the model on a batch from the replay memory."""
        if len(self.memory) < BATCH_SIZE:
            return # Don't train until we have enough experiences

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, next_states, rewards, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).unsqueeze(1).to(self.device)

        # Get current Q-values: Q(s, a)
        q_values = self.model(states).gather(1, actions)

        with torch.no_grad():
            # Double DQN: Use model to select the best action, and target to evaluate it.
            next_actions = self.model(next_states).argmax(dim=1, keepdim=True)
            max_next_q = self.target(next_states).gather(1, next_actions)
            
            # If the next state is terminal, its Q-value is 0
            max_next_q[dones] = 0.0

        # Calculate target Q-values: r + gamma * max_next_Q
        targets = rewards + (self.gamma * max_next_q)

        # Use Huber loss for more stability
        loss = F.smooth_l1_loss(q_values, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent them from exploding
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

    def update_target_network(self):
        """Update the target network with the model's weights."""
        self.target.load_state_dict(self.model.state_dict())
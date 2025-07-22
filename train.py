from env import TicTacToe
from agent import DQNAgent
import numpy as np
import torch
import os
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def random_opponent(board):
    legal_moves = [i for i, v in enumerate(board) if v == 0]
    return np.random.choice(legal_moves)

# Create environment and agent
env = TicTacToe()
agent = DQNAgent(device=device)

# Training parameters
episodes = 10000
use_self_play_after = 5000

# Training loop
for ep in range(episodes):
    state = env.reset()
    done = False

    while not done:
        legal = env.legal_actions()
        if env.current_player == 1:
            action = agent.act(state, legal)
        else:
            if ep >= use_self_play_after:
                action = agent.act(-state, legal)  # Self-play
            else:
                action = random_opponent(state)

        next_state, reward, done = env.step(action)

        if env.current_player == -1:  # We just acted
            agent.remember(state, action, next_state, reward)

        state = next_state

    agent.train_step()
    agent.epsilon = max(0.05, agent.epsilon * 0.995)

    if ep % 500 == 0:
        print(f"Episode {ep}, Epsilon: {agent.epsilon:.3f}")

# Save model to models/ folder
os.makedirs("models", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = f"models/dqn_{timestamp}.pt"
torch.save(agent.model.state_dict(), save_path)
print(f"\n✅ Training complete. Model saved to: {save_path}")

from env import TicTacToe
from agent import DQNAgent, BATCH_SIZE, EPSILON_START, EPSILON_END, EPSILON_DECAY
import torch
import os
from datetime import datetime
from collections import deque
import numpy as np

def main():
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = TicTacToe()
    agent = DQNAgent(device=device)

    # --- Training Parameters ---
    episodes = 25000
    target_update_frequency = 10  # In episodes
    print_every = 500
    
    # --- Logging ---
    win_loss_draw = {'win': 0, 'loss': 0, 'draw': 0}
    
    print("🚀 Starting self-play training...")
    for ep in range(1, episodes + 1):
        env.reset()
        done = False
        episode_transitions = []

        while not done:
            # The agent always sees the state from the current player's perspective
            current_player_state = env.get_state()
            
            legal_actions = env.legal_actions()
            action = agent.act(current_player_state, legal_actions)
            
            player_at_move = env.current_player
            # Correctly unpack the four values returned by env.step()
            _, _, done, info = env.step(action)
            
            # Store the raw transition; rewards will be assigned later
            episode_transitions.append((current_player_state, action, player_at_move))
            
            # Train the agent on a batch of experiences
            agent.train_step()

        # --- Post-Episode Reward Assignment ---
        final_winner = info['winner']
        if final_winner == 1: win_loss_draw['win'] += 1
        elif final_winner == -1: win_loss_draw['loss'] += 1
        else: win_loss_draw['draw'] += 1

        # Process the episode's transitions with the final outcome
        next_state = np.zeros(9) # Terminal state
        for s, a, player in reversed(episode_transitions):
            reward = 0
            if final_winner is not None and final_winner != 0:
                reward = 1 if player == final_winner else -1
            
            # state, action, next_state, reward, done
            agent.remember(s, a, next_state, reward, True)
            next_state = s # Current state becomes next_state for the previous move

        # Update the target network periodically
        if ep % target_update_frequency == 0:
            agent.update_target_network()

        # --- Logging ---
        if ep % print_every == 0:
            total = sum(win_loss_draw.values())
            if total > 0:
                win_rate = win_loss_draw['win'] / total * 100
                loss_rate = win_loss_draw['loss'] / total * 100
                draw_rate = win_loss_draw['draw'] / total * 100
                epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1. * agent.steps_done / EPSILON_DECAY)
                print(f"Ep {ep}/{episodes} | Win: {win_rate:.1f}% Loss: {loss_rate:.1f}% Draw: {draw_rate:.1f}% | Epsilon: {epsilon:.3f}")
                win_loss_draw = {'win': 0, 'loss': 0, 'draw': 0}

    # --- Save Model ---
    os.makedirs("models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"models/dqn_improved_{timestamp}.pt"
    torch.save(agent.model.state_dict(), save_path)
    print(f"\n✅ Training complete. Model saved to: {save_path}")

if __name__ == "__main__":
    main()
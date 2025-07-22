import pygame
from env import TicTacToe
from agent import DQNAgent
import torch
import numpy as np
import os
import time
from train import random_opponent  # To train a model if none exists
from datetime import datetime

def draw_board(screen, board, scores):
    screen.fill((255, 255, 255))

    # Draw score tab
    font = pygame.font.Font(None, 36)
    score_text = font.render(f"X (Agent): {scores['X']}   O (You): {scores['O']}   Draws: {scores['draw']}", True, (0, 0, 0))
    screen.blit(score_text, (10, 10))

    # Draw grid
    for i in range(1, 3):
        pygame.draw.line(screen, (0, 0, 0), (0, i * 100 + 40), (300, i * 100 + 40), 2)
        pygame.draw.line(screen, (0, 0, 0), (i * 100, 40), (i * 100, 340), 2)

    # Draw moves
    font = pygame.font.Font(None, 72)
    for i, val in enumerate(board):
        x = (i % 3) * 100 + 25
        y = (i // 3) * 100 + 55
        if val == 1:
            text = font.render("X", True, (0, 0, 255))
            screen.blit(text, (x, y))
        elif val == -1:
            text = font.render("O", True, (255, 0, 0))
            screen.blit(text, (x, y))

    pygame.display.flip()

def train_if_no_model():
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]

    if not model_files:
        print("🔄 No trained models found. Training a quick model (~1000 episodes)...")
        from agent import DQNAgent
        from env import TicTacToe

        agent = DQNAgent(device='cpu')
        env = TicTacToe()

        for ep in range(1000):
            state = env.reset()
            done = False
            while not done:
                legal = env.legal_actions()
                if env.current_player == 1:
                    action = agent.act(state, legal)
                else:
                    action = random_opponent(state)

                next_state, reward, done = env.step(action)
                if env.current_player == -1:
                    agent.remember(state, action, next_state, reward)
                state = next_state
            agent.train_step()
            agent.epsilon = max(0.05, agent.epsilon * 0.995)

        save_path = os.path.join(model_dir, f"quick_dqn_{datetime.now().strftime('%H%M%S')}.pt")
        torch.save(agent.model.state_dict(), save_path)
        print(f"✅ Quick model saved: {save_path}")


def select_model():
    model_dir = "models"
    models = sorted(os.listdir(model_dir))
    print("\n📦 Available models:")
    for i, m in enumerate(models):
        print(f"  [{i}] {m}")
    index = int(input("\n🕹️  Select model index to load: "))
    return os.path.join(model_dir, models[index])

def main():
    pygame.init()
    screen = pygame.display.set_mode((300, 340))
    pygame.display.set_caption("DQN Tic-Tac-Toe")

    # Ensure at least one model exists
    train_if_no_model()

    # Load model
    model_path = select_model()
    agent = DQNAgent(device='cpu')
    agent.model.load_state_dict(torch.load(model_path, map_location='cpu'))
    print(f"✅ Loaded model: {model_path}")

    # Ask user to choose side
    player_symbol = input("Do you want to be X or O? (X goes first): ").strip().upper()
    while player_symbol not in ['X', 'O']:
        player_symbol = input("❓ Please enter 'X' or 'O': ").strip().upper()
    player_is_x = player_symbol == "X"

    scores = {"X": 0, "O": 0, "draw": 0}

    game = TicTacToe()
    state = game.reset()
    draw_board(screen, state, scores)
    running = True

    while running:
        # Agent's move
        if (game.current_player == 1 and not player_is_x) or (game.current_player == -1 and player_is_x):
            pygame.time.wait(500)
            legal = game.legal_actions()
            action = agent.act(state, legal)
            state, reward, done = game.step(action)
            draw_board(screen, state, scores)
            if done:
                winner = game.check_winner()
                if winner == 1:
                    scores["X"] += 1
                elif winner == -1:
                    scores["O"] += 1
                else:
                    scores["draw"] += 1
                pygame.time.wait(1500)
                state = game.reset()
                draw_board(screen, state, scores)

        # Player's move
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif ((game.current_player == 1 and player_is_x) or 
                  (game.current_player == -1 and not player_is_x)) and event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                if y >= 40:
                    cell = ((y - 40) // 100) * 3 + (x // 100)
                    if game.board[cell] == 0:
                        state, reward, done = game.step(cell)
                        draw_board(screen, state, scores)
                        if done:
                            winner = game.check_winner()
                            if winner == 1:
                                scores["X"] += 1
                            elif winner == -1:
                                scores["O"] += 1
                            else:
                                scores["draw"] += 1
                            pygame.time.wait(1500)
                            state = game.reset()
                            draw_board(screen, state, scores)

    pygame.quit()

if __name__ == "__main__":
    main()

import pygame
from env import TicTacToe
from agent import DQNAgent
import torch
import os
# Import the training function from train.py
from train import main as run_training

def find_latest_model(model_dir="models"):
    """Finds the most recently saved model file."""
    os.makedirs(model_dir, exist_ok=True)
    model_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".pt")]
    if not model_files:
        return None
    return max(model_files, key=os.path.getctime)

def draw_board(screen, board, scores, player_symbol):
    """Draws the game board and scores."""
    screen.fill((255, 255, 255))
    font = pygame.font.Font(None, 36)
    agent_symbol = 'O' if player_symbol == 'X' else 'X'
    score_text = font.render(f"You ({player_symbol}): {scores['You']}   Agent ({agent_symbol}): {scores['Agent']}   Draws: {scores['Draw']}", True, (0, 0, 0))
    screen.blit(score_text, (10, 10))

    for i in range(1, 3):
        pygame.draw.line(screen, (0, 0, 0), (0, i * 100 + 40), (300, i * 100 + 40), 3)
        pygame.draw.line(screen, (0, 0, 0), (i * 100, 40), (i * 100, 340), 3)

    font = pygame.font.Font(None, 100)
    for i, val in enumerate(board):
        x = (i % 3) * 100 + 50
        y = (i // 3) * 100 + 90
        if val == 1:
            text = font.render("X", True, (0, 0, 255))
            screen.blit(text, text.get_rect(center=(x, y)))
        elif val == -1:
            text = font.render("O", True, (255, 0, 0))
            screen.blit(text, text.get_rect(center=(x, y)))
    pygame.display.flip()

def main():
    pygame.init()
    screen = pygame.display.set_mode((300, 340))
    pygame.display.set_caption("DQN Tic-Tac-Toe")

    model_path = find_latest_model()
    
    # --- ADDED: Prompt to train if no model is found ---
    if not model_path:
        print("❌ No trained models found.")
        choice = input("Would you like to train a new model now? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            run_training()
            # After training, find the new model
            model_path = find_latest_model()
            if not model_path:
                print("🚨 Training did not produce a model file. Exiting.")
                return
        else:
            print("Ok, exiting. Run train.py to create a model.")
            return
    # --- END ADDED SECTION ---

    agent = DQNAgent(device='cpu')
    agent.model.load_state_dict(torch.load(model_path, map_location='cpu'))
    agent.model.eval()
    
    # In the improved agent.py, epsilon is now internal and decays with steps.
    # We can set steps_done high to force a low epsilon for gameplay.
    agent.steps_done = 99999 # Effectively sets epsilon to its minimum

    print(f"✅ Loaded model: {model_path}")

    player_symbol = input("Do you want to be X or O? (X goes first): ").strip().upper()
    while player_symbol not in ['X', 'O']:
        player_symbol = input("❓ Please enter 'X' or 'O': ").strip().upper()
    
    player_id = 1 if player_symbol == 'X' else -1
    agent_id = -player_id

    scores = {"Agent": 0, "You": 0, "Draw": 0}
    game = TicTacToe()
    game.reset()
    
    running = True
    while running:
        draw_board(screen, game.board, scores, player_symbol)
        
        done = False
        info = {}

        if game.current_player == agent_id:
            pygame.time.wait(500)
            agent_state = game.get_state()
            legal = game.legal_actions()
            action = agent.act(agent_state, legal)
            _, _, done, info = game.step(action)
        else: # Player's turn
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if game.current_player == player_id:
                        x, y = pygame.mouse.get_pos()
                        if y >= 40:
                            cell = ((y - 40) // 100) * 3 + (x // 100)
                            if cell in game.legal_actions():
                                _, _, done, info = game.step(cell)
        
        if done:
            draw_board(screen, game.board, scores, player_symbol)
            winner = info.get('winner')
            if winner == agent_id: scores["Agent"] += 1
            elif winner == player_id: scores["You"] += 1
            else: scores["Draw"] += 1
            pygame.time.wait(2000)
            game.reset()

    pygame.quit()

if __name__ == "__main__":
    main()
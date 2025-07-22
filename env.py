import numpy as np

class TicTacToe:
    def __init__(self):
        self.board = np.zeros(9, dtype=int)
        self.current_player = 1

    def reset(self):
        """Resets the board for a new game."""
        self.board.fill(0)
        self.current_player = 1
        return self.get_state()

    def get_state(self):
        """Returns the board state from the perspective of the current player."""
        # The agent always perceives itself as player '1'.
        return self.board.copy() * self.current_player

    def step(self, action):
        """Executes a move and returns the outcome."""
        if self.board[action] != 0:
            # An illegal move ends the game and results in a loss.
            return self.get_state(), -1, True, {'winner': -self.current_player}

        self.board[action] = self.current_player
        winner = self.check_winner()
        done = (winner is not None) or (0 not in self.board)
        
        # Reward is determined post-episode in the training loop.
        # Here we just pass a placeholder.
        reward = 0 

        self.current_player *= -1
        return self.get_state(), reward, done, {'winner': winner}

    def check_winner(self):
        """Checks if there is a winner."""
        wins = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
        for i, j, k in wins:
            if self.board[i] != 0 and self.board[i] == self.board[j] == self.board[k]:
                return self.board[i]
        return None

    def legal_actions(self):
        """Returns a list of valid moves."""
        return [i for i, v in enumerate(self.board) if v == 0]
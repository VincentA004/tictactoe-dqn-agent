import numpy as np

class TicTacToe:
    def __init__(self):
        self.board = np.zeros(9)
        self.current_player = 1

    def reset(self):
        self.board[:] = 0
        self.current_player = 1
        return self.board.copy()

    def step(self, action):
        if self.board[action] != 0:
            return self.board.copy(), -10, True  # Illegal move
        self.board[action] = self.current_player
        winner = self.check_winner()
        done = winner is not None or 0 not in self.board
        reward = 1 if winner == self.current_player else 0
        self.current_player *= -1
        return self.board.copy(), reward, done

    def check_winner(self):
        wins = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
        for i,j,k in wins:
            total = self.board[i] + self.board[j] + self.board[k]
            if abs(total) == 3:
                return self.board[i]
        return None

    def legal_actions(self):
        return [i for i, v in enumerate(self.board) if v == 0]

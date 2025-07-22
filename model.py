import torch.nn as nn

class DQN(nn.Module):
    """A simple MLP for Q-value approximation."""
    def __init__(self):
        super().__init__()
        # A simpler network is more effective for a simple game like Tic-Tac-Toe.
        self.net = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 9)
        )

    def forward(self, x):
        return self.net(x)
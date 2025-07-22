# Tic-Tac-Toe DQN Agent

This project is a simple reinforcement learning agent that masters the game of Tic-Tac-Toe through self-play. It uses a Deep Q-Network (DQN) built with PyTorch to learn the optimal policy.

## Technical Details

* **Framework**: PyTorch
* **Algorithm**: Deep Q-Learning (DQN) with key improvements:
    * **Target Network**: A separate, periodically updated network is used to generate stable target Q-values.
    * **Experience Replay**: A replay buffer stores past experiences, which are sampled randomly during training to break temporal correlations.
    * **Double DQN**: This technique is used to reduce the overestimation of Q-values, leading to better policy evaluation.
    * **Self-Play**: The agent learns entirely by playing against itself, which is an effective method for developing robust strategies in two-player, zero-sum games.
* **Reward Structure**: Rewards are assigned at the end of each game: `+1` for winning, `-1` for losing, and `0` for a draw.


## How to Run

### 1. Installation

Clone the repository and install the required Python packages.

```bash
# Clone the project directory (if you haven't already)
git clone <your-repository-url>
cd <project-directory>

# Install dependencies
pip install -r requirements.txt

```

### 2. Train the Agent

Run the training script from your terminal. This will start the self-play process and save the trained model weights to a .pt file in the models/ directory.

```bash
python train.py
```

### 3. Play the game 

After training is complete, run the GUI to play against your trained agent.

```bash
python gui.py
```
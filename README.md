# Snake Game AI with Deep Q-Learning

This repository contains an implementation of the classic Snake game combined with a Deep Q-Network (DQN) to train an AI agent to play Snake autonomously.

## Table of Contents
1. [Overview](#overview)  
2. [Features](#features)  
3. [Installation](#installation)  
4. [Usage](#usage)  
5. [Project Structure](#project-structure)  
6. [How It Works](#how-it-works)  
7. [Hyperparameters](#hyperparameters)  
8. [Troubleshooting](#troubleshooting)  
9. [License](#license)

---

## Overview

This project demonstrates how to use a Deep Q-Learning approach for training an AI to play Snake. The code includes:
- A Snake game environment built with **Pygame**.
- A neural network model built with **PyTorch**.
- Helper utilities to handle training, plotting results, and saving/loading model checkpoints.

---

## Features

- **Automated Training**: The AI plays the game continuously, learning from rewards or penalties until it masters Snake.
- **Deep Q-Network (DQN)**: A feedforward neural network predicts the Q-values for each possible action.
- **Replay Memory**: Uses experience replay for more stable training.
- **Performance Visualization**: Scores and mean scores are plotted in real-time to observe training progress.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/snake-ai-dqn.git
   cd snake-ai-dqn
   ```

2. **Create a virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On macOS/Linux
   .\venv\Scripts\activate    # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   (If you don't have a `requirements.txt`, you can manually install the key libraries: `pip install pygame torch matplotlib numpy`.)

---

## Usage

1. **Run the training script**:
   ```bash
   python main.py
   ```
   - The script will open a Pygame window where the Snake game runs.
   - A real-time plot of the scores and mean scores will be displayed in a separate window (thanks to Matplotlib).

2. **Monitor training**:
   - In the console, you'll see the number of games played, the current score, and the highest score (record).
   - The plot will also show how scores evolve as training progresses.

3. **Stopping training**:
   - Close the Pygame window or press Ctrl+C in the console.

4. **Load a saved model (optional)**:
   - The script automatically saves the best model to the `model/` folder. If you want to test or continue training, make sure to load the model from this directory.  
   *(You may need to modify the code to load an existing model if you want to resume from a checkpoint.)*

---

## Project Structure

A brief overview of the important files and folders:

```
snake-ai-dqn/
│
├── model.py
├── snake_game.py
├── helper.py
├── main.py
└── README.md
```

- **`main.py`**: Contains the main training loop (`train()` function) and the DQN Agent class.
- **`snake_game.py`**: Pygame-based Snake environment including the `SnakeGameAI` class.
- **`model.py`**: Defines the neural network (`Linear_QNet`) and the Q-learning trainer (`QTrainer`).
- **`helper.py`**: Utility functions (plotting, etc.).
- **`model/`**: (Created automatically) Stores model checkpoints (`model.pth`).

---

## How It Works

1. **State Representation**  
   - The agent observes the Snake’s current situation (e.g. position of the head, food, collision threats, etc.).
   - This state is represented as an 11-dimensional vector.

2. **Action Selection**  
   - Three possible actions are considered in this setup: go straight, turn right, or turn left (relative to the Snake’s current direction).
   - An epsilon-greedy policy is used. With some probability (epsilon), the agent takes a random action to explore; otherwise, it picks the best known action according to the network’s Q-values.

3. **Reward**  
   - The agent receives +10 points for eating the food and -10 if it dies.
   - It gets a small negative reward if it collides with a boundary or its body.

4. **Deep Q-Network**  
   - A feedforward neural network (two-layer in this case) takes the current state and outputs Q-values for each of the three actions.
   - The agent updates its Q-values based on the Bellman equation using experience replay.

5. **Experience Replay**  
   - We store experiences `(state, action, reward, next_state, done)` in a replay buffer.
   - The agent samples mini-batches from this buffer to train the network, which helps to avoid correlated samples and speeds up training.

6. **Training Loop**  
   - **Short Memory**: The agent quickly updates the network after each step with the most recent transition.
   - **Long Memory**: Periodically trains over a mini-batch of past experiences.

---

## Hyperparameters

You can adjust these parameters to tune training performance:

- **`MAX_MEMORY = 100_000`**  
  Maximum number of past experiences stored in the replay buffer.  
- **`BATCH_SIZE = 1000`**  
  Number of samples per training batch.  
- **`LR = 0.001`**  
  Learning rate for the optimizer.  
- **`GAMMA = 0.9`**  
  Discount factor for the Q-learning updates.  
- **`EPSILON`**  
  Initially `80 - number_of_games`. Decreases as the agent plays more games to reduce random actions over time.

---

## Troubleshooting

- **Game window doesn’t appear**: Make sure Pygame is properly installed and that your Python environment is correct.
- **CUDA / GPU usage**: If you want to use a GPU for faster training, modify the code to push tensors and the model to the GPU (e.g. `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`). Currently, the code runs on CPU.
- **Performance**: If training is too slow, try lowering `SPEED` in `snake_game.py`, reducing `BATCH_SIZE`, or simplifying the model architecture.

---

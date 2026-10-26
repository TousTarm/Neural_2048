# Neural 2048

An experiment in teaching a neural network to play 2048 with reinforcement learning. I built it to apply what I learned in a course on the fundamentals of information and knowledge processing, which covered simple neural networks, basic AI methods and the statistics behind them.

The experiment was only partly successful. I wanted the network to learn the game purely from rewards, without a hand-written heuristic telling it which moves are good, because such a heuristic would defeat the point of a neural network. Without one, the agent did not learn to play well.

Code comments and console output are in Czech.

## What it does

- `engine.py`: a 2048 game engine on a NumPy 4×4 grid. The reward for a move is the sum of merged tiles; a move that changes nothing gets −10.
- `ai.py`: a convolutional Q-network (two 2×2 convolutions, then two linear layers) with four outputs, one per direction. The board is converted to log2 values before it is passed to the network.
- `train.py`: Deep Q-learning with an experience replay buffer (100 000 moves, batch 4096) and an epsilon-greedy policy (1.0 → 0.02). It runs until Ctrl+C, prints progress every 100 games, saves `model_cnn.pth` every 500 games and `model_cnn_final.pth` when interrupted. It uses CUDA when available.
- `discord.py`: sends a Discord webhook message when the agent reaches a new highest tile in the session. Optional; without `DISCORD_WEBHOOK_URL` it only prints a warning.
- `test.py`: runs an untrained network on a sample board and prints its move scores.
- `original/`: the game written before the AI part, a console version (`2048_original.py`) and a pygame version (`2048_pygame.py`, arrow keys, R to restart).

## Requirements

- Python 3
- `torch`, `numpy`, `requests`, `python-dotenv`
- `pygame` for the scripts in `original/`

There is no `requirements.txt`.

## Running

```bash
python3 -m venv venv
venv/bin/pip install torch numpy requests python-dotenv
venv/bin/python test.py
venv/bin/python train.py
```

Discord notifications (optional), in `.env`:

```
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

The pygame version of the game:

```bash
venv/bin/pip install pygame
venv/bin/python original/2048_pygame.py
```

## Tests

There are no automated tests. `test.py` is a manual check of the network's output shape.

## Structure

```
engine.py    2048 game logic and rewards
ai.py        CNN Q-network and board preprocessing
train.py     DQN training loop
discord.py   Discord webhook notifications
test.py      network smoke check
original/    2048 without AI (console and pygame)
```

Built with the help of AI coding tools.

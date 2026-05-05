# FlappyBird DQN

A Deep Q-Network (DQN) agent that learns to play Flappy Bird using PyTorch and Pygame.

## Overview

This project implements a DQN that learns to play Flappy Bird from raw pixel input. The agent processes 84x84 grayscale frames through a convolutional neural network and outputs Q-values for two actions: do nothing or flap.

Two DQN variants are trained side-by-side for comparison:

- **Base DQN** — standard DQN with a single network
- **DQN + Target Network** — DQN with a separate, periodically-synced target network for more stable Q-value estimation

Both models are trained from scratch (2,000,000 iterations each) and their reward curves are plotted for comparison.

## Project Structure

```
FlappyBirdDQN/
├── dqn.py                  # DQN model, training loop, interactive CLI
├── startdqn.bat            # Windows launcher
├── game/
│   └── flappy_bird.py      # Flappy Bird game environment (Pygame)
├── assets/
│   ├── sprites/            # Bird, pipe, background, number sprites
|   ├── demos/              # Demonstration GIFs
│   └── audio/              # Sound effects
├── pretrained_model/
│   ├── base/               # Saved base DQN checkpoints
│   └── target/             # Saved target-network DQN checkpoints
├── reward_base.npy         # Logged episode rewards (base DQN)
├── reward_target.npy        # Logged episode rewards (target DQN)
└── reward_compare.png      # Reward comparison plot
```

## Demonstration

### Base DQN Model at Iteration 1,950,000

![](./assets/demos/demo_base.gif)

Above model can be found in `pretrained_model\base\`

### DQN + Target Network Model at Iteration 1,950,000

![](./assets/demos/demo_target.gif)

Above model can be found in `pretrained_model\target\`

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

Run the interactive CLI:

```bash
python dqn.py
```

Or double-click `startdqn.bat` (Windows users only).

You'll be prompted to choose from four modes:

### Train

Trains a DQN agent. Select `base`, `target`, or `both` (trains base first, then target). Training can be interrupted with Ctrl+C and will save a checkpoint for later resumption.

### Test

Loads a saved model checkpoint and runs it in evaluation mode (no exploration). Displays the score for each episode. Press Ctrl+C to stop and see the average score.

### Plot

Generates reward-over-episode plots from saved training logs(NPY files).

### Play

Lets you play Flappy Bird manually (press Space to flap). Press Ctrl+C to stop.

## Network Architecture

| Layer | Type   | Details                              |
| ----- | ------ | ------------------------------------ |
| conv1 | Conv2d | 4→32 channels, 8×8 kernel, stride 4  |
| conv2 | Conv2d | 32→64 channels, 4×4 kernel, stride 2 |
| conv3 | Conv2d | 64→64 channels, 3×3 kernel, stride 1 |
| fc4   | Linear | 3136→512                             |
| fc5   | Linear | 512→2 (Q-values)                     |

Input: 4 stacked 84×84 grayscale frames. ReLU activations throughout.

## Hyperparameters

| Parameter            | Value          |
| -------------------- | -------------- |
| Iterations           | 2,000,000      |
| Replay memory size   | 10,000         |
| Minibatch size       | 32             |
| Discount factor (γ)  | 0.99           |
| Initial ε            | 0.1            |
| Final ε              | 0.0001         |
| Target sync interval | 1,000 steps    |
| Optimizer            | Adam (lr=1e-6) |
| Loss                 | MSE            |

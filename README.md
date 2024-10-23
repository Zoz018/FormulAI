# FormulAI

FormulAI is a reinforcement learning project that enables a Formula 1 race car to drive autonomously in a 2D simulation environment. The game simulates a top-down view with a fixed camera, providing an ideal setting for testing and refining reinforcement learning techniques. Additionally, the game is playable by human players and is built using Pygame.

## Table of Contents
1. [Description](#description)
2. [Technologies](#technologies)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Project Structure](#project-structure)

## Description

FormulAI uses reinforcement learning to train an agent capable of driving a Formula 1 car in a simulated environment. The goal is to teach the agent to navigate the track efficiently, optimize racing lines, and avoid collisions while maximizing speed. The simulation takes place in a 2D game with a top-down view, where the car is represented as a sprite and follows simplified physical laws.

The original game is also playable by human players, providing an interactive way to compare the agent's performance with that of a human driver. The game environment is built using **Pygame**, a popular library for creating 2D games in Python.

The reinforcement learning algorithm uses the **PyTorch** library for training neural networks and optimizing learning policies.

## Technologies

- **Programming Language:** Python
- **Main Libraries:** PyTorch, Pygame
- **Simulation Environment:** Custom 2D game (top-down view, fixed camera)
- **Learning Method:** Reinforcement learning (specific algorithm to be specified, such as DQN, PPO, SAC, etc.)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your_username/formulAI.git
   cd formulAI
   ```

2. **Install dependencies:**
   - Make sure you have Python 3.x installed.
   - Install the required libraries:
     ```bash
     pip install -r requirements.txt
     ```

> Note: The `requirements.txt` file contains the necessary Python libraries, including PyTorch and Pygame.

## Usage

### Playing the Game
You can play the game manually by running the following command:
```bash
python Formule1.py
```

### Training the Agent
To train the reinforcement learning agent, use the following command:
```bash
python train.py
```

### Old version
You can play an old version of the game with more features manually by running the following command:
```bash
python main.py
```

## Project Structure

- `Formule1.py`: Script for playing the game manually.
- `train.py`: Main script for training the agent.


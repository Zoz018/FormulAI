import torch
import random
import numpy as np
from collections import deque
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 10000
LR = 0.001
MAX_SPEED = 30

class Agent:

    def __init__(self, gamma=0.9):
        self.n_games = 0
        self.epsilon = 0  # Randomness
        self.gamma = gamma  # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # Popleft if exceeding memory
        self.model = Linear_QNet(19, 256, 9)  # Updated to match action space
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        distances = game.get_distances()
        state = distances + [

            game.is_on_track(),
            game.car_x,
            game.car_y,
            game.car_speed,
            game.direction,
            game.next_checkpoint.centerx,
            game.next_checkpoint.centery,
            game.distance_pc,
            game.car_speed / MAX_SPEED,
            game.current_lap_time,
            game.current_cp_time

        ]
        return np.array(state, dtype=float)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0, 0, 0, 0, 0, 0, 0]

        if random.randint(0, 80) < self.epsilon : 
            # Random action mais que de l'accélération 
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move




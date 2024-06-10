import torch
import random
import numpy as np
from collections import deque
from game import FormulAI, Direction, Acceleration
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(12, 256, 2)  # 3 for direction + 3 for acceleration
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_s = game.direction == Direction.STRAIGHT
        dir_a = game.acceleration == Acceleration.ACCEL
        dir_b = game.acceleration == Acceleration.BRAKE
        dir_n = game.acceleration == Acceleration.BASE

        state = [
            dir_l,
            dir_r,
            dir_s,
            dir_a,
            dir_b,
            dir_n,

            game.car_speed/30,
            game.car_angle/360, 

            game.next_checkpoint.centerx > game.car_x,
            game.next_checkpoint.centerx < game.car_x,
            game.next_checkpoint.centery < game.car_y,
            game.next_checkpoint.centery > game.car_y
        ]

        return np.array(state, dtype=float)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, action, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, action, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games

        action = [0,0]

        if random.randint(0, 200) < self.epsilon:
            # Random action
            action = np.random.uniform(size=2).tolist()
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            # pred_dir, pred_acc = self.model(state_tensor)
            pred = self.model(state_tensor)

            # Choose the action with the highest Q-value for both direction and acceleration
            # move_dir = torch.argmax(pred_dir).item()
            # move_acc = torch.argmax(pred_acc).item()

            action = pred.tolist()

            # Set the final moves
            # final_move_dir[move_dir] = 1
            # final_move_acc[move_acc] = 1

        return action


def train():
    plot_reward = []
    plot_max_speed = []
    record = 0
    agent = Agent()
    game = FormulAI()
    rewards = 0
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        rewards += reward

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Reward', reward, 'Speed', score)

            plot_max_speed.append(score)
            plot_reward.append(rewards)
            rewards = 0
            plot(plot_reward, plot_max_speed)

if __name__ == '__main__':
    train()
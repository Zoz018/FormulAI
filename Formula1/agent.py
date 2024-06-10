import torch
import random
import numpy as np
from collections import deque
from game import FormulAI, Direction, Acceleration
from model import Linear_QNet, QTrainer
from helper import plot
import pygame

TRACK = pygame.image.load('circuit_ovale.png')
TRACK = pygame.transform.scale(TRACK, (1920, 1080))
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(11, 256, 9)  # 9 actions possibles 
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):

        point_1 = (game.car_x - 20, game.car_y)
        point_2 = (game.car_x + 20, game.car_y)
        point_3 = (game.car_x, game.car_y - 20)
        point_4 = (game.car_x, game.car_y + 20)

        state = [

            game.acceleration,
            game.deceleration,
            game.relative_turn_speed,

            game.car_angle,
            game.car_x,
            game.car_y,

            game.is_collision(),

            game.next_checkpoint.centery < game.car_y,
            game.next_checkpoint.centery > game.car_y,
            game.next_checkpoint.centerx > game.car_x,
            game.next_checkpoint.centerx < game.car_x,

            TRACK.get_at((int(self.car_x), int(self.car_y)))
        ]
        
        #print(np.array(state, dtype=float))
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
        self.epsilon = 200 - self.n_games
        final_move = [0,0,0,0,0,0,0,0,0]

        if random.randint(0, 200) < self.epsilon:
            # Random action
            move = random.random()
            if move < 0.8 :
                move = random.randint(0,2)
            elif move <0.95 :
                move = random.randint(3,5)
            else :
                move = random.randint(6,8)
            final_move[move] = 1

        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = FormulAI()
    while True:

        state_old = agent.get_state(game)

        final_move = agent.get_action(state_old)

        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()
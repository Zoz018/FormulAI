import pygame
import time
import math
from enum import Enum
import numpy as np

class Direction(Enum):
    LEFT = 1
    STRAIGHT = 0
    RIGHT = -1

class Acceleration(Enum):
    BRAKE = -1
    BASE = 0
    ACCEL = 1

# Parameters
MAX_SPEED = 30
ACCELERATION = 0.5
BASE_DECELERATION = 0.1
SAND_DECELERATION = 0.4  # Sand deceleration
BRAKE_DECELERATION = 1
TURN_SPEED = 7

# Game Settings
SPEED = 60

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 71, 0)  # Track border color for collision detection
YELLOW = (239, 228, 176)  # Sand color
BLACK = (0, 0, 0)
BROWN = (120, 67, 21)
GRAY = (41,41,41) # couleur de la piste 

# Load Track and Car
try:
    TRACK = pygame.image.load('circuit_ovale.png')
    TRACK = pygame.transform.scale(TRACK, (1920, 1080))
    CAR = pygame.image.load('car.png')
    CAR = pygame.transform.scale(CAR, (25/2, 50/2))  # Adjusted car dimensions
except pygame.error as e:
    print(f"Failed to load images: {e}")
    exit()

CHECKPOINTS = [pygame.Rect(0, 570, 420, 40), pygame.Rect(960, 0, 40, 325), pygame.Rect(1580, 570, 340, 40), pygame.Rect(997, 800, 40, 280)]

# Utility Functions
def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

class FormulAI:

    def __init__(self, width=1920, height=1080):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('FormulAI')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self, car_x=1035, car_y=940, car_angle=90):
        self.direction = 0
        self.acceleration = 0
        self.deceleration = 0
        self.car_x = car_x
        self.car_y = car_y
        self.car_speed = 10
        self.mean_speed = 0
        self.car_angle = car_angle
        self.relative_turn_speed = 0
        self.start_time = time.time()
        self.current_lap_time = 0
        self.start_time_cp = time.time()
        self.current_cp_time = 0
        self.next_checkpoint = CHECKPOINTS[0]
        self.next_checkpoint_id = 0
        self.distance_pc = distance(self.next_checkpoint.center, (self.car_x, self.car_y))
        self.score = 0
        self.count = 0
        self.last_time = 0
        self.best_time = float("inf")
        self.distance_bord = []

    def play_step(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        old_distance_pc = self.distance_pc
        self._move(action)
        reward = 0
        game_over = False

        if self.is_on_track() and self.car_speed != 0:
            reward += 0.6
        else : 
            reward -= 0.3

        if self.car_speed >= 150 :
            reward += 0.6  
        else :
            reward -= 0.3

        if self.is_collision() or self.current_cp_time > 10:
            game_over = True
            reward -= 50
            return reward, game_over, self.score

        if self._checkpoint_collision():
            self.next_checkpoint_id += 1
            self.score += 1
            reward += 100
            self.current_cp_time = 0
            self.start_time_cp = time.time()

            if self.next_checkpoint_id >= len(CHECKPOINTS):
                self.next_checkpoint_id = 0
                self.count += 1
                if self.current_lap_time > self.best_time:
                    self.best_time = self.current_lap_time
                self.last_time = self.current_lap_time
                self.current_lap_time = 0
                self.start_time = time.time()

            self.next_checkpoint = CHECKPOINTS[self.next_checkpoint_id]

        self._update_ui()
        self.clock.tick(SPEED)
        self.current_lap_time = time.time() - self.start_time
        self.current_cp_time = time.time() - self.start_time_cp
        self.mean_speed += self.car_speed

        return reward, game_over, self.score

    def is_collision(self):
        if self.car_x < 0 or self.car_x > self.width or self.car_y < 0 or self.car_y > self.height:
            return True
        try:
            pixel_color = TRACK.get_at((int(self.car_x), int(self.car_y)))
            if pixel_color == GREEN or pixel_color == BROWN:
                return True
        except IndexError:
            pass
        return False
    
    def is_on_track(self):
        if self.car_x < 0 or self.car_x > self.width or self.car_y < 0 or self.car_y > self.height:
            return True
        try:
            pixel_color = TRACK.get_at((int(self.car_x), int(self.car_y)))
            if pixel_color == GRAY:
                return True
        except IndexError:
            pass
        return False

    def _update_ui(self):
        self.screen.fill(WHITE)
        self.screen.blit(TRACK, (0, 0))
        pygame.draw.rect(self.screen, WHITE, self.next_checkpoint)
        self._show_lap_info()
        rotated_car = pygame.transform.rotate(CAR, self.car_angle)
        car_rect = rotated_car.get_rect(center=(self.car_x, self.car_y))
        self.screen.blit(rotated_car, car_rect.topleft)
        pygame.display.update()

    def _checkpoint_collision(self):
        return self.next_checkpoint.collidepoint(self.car_x, self.car_y)
    
    def get_distances(self):
        directions = [0, 45, 90, 135, 180, 225, 270, 315]  # Angles en degrés
        distances = []
        for angle in directions:
            distance = self.cast_ray(angle)
            distances.append(distance)
        return distances
    
    def cast_ray(self, angle):
        radian_angle = math.radians(angle)
        distance = 0
        x, y = self.car_x , self.car_y

        while 0 <= int(x) < self.width and 0 <= int(y) < self.height:
            x += math.cos(radian_angle)
            y += math.sin(radian_angle)
            distance += 1

            try :
                if TRACK.get_at((int(x), int(y))) == (0, 71, 0):  # Si pixel vert (obstacle)
                 break
            except IndexError :
                pass

        return distance

    def _move(self, action):
        if action[0] == 1:  # Accelerate + Turn left
            self.acceleration = ACCELERATION
            self.direction = 1
        elif action[1] == 1:  # Accelerate + Straight
            self.acceleration = ACCELERATION
            self.direction = 0
        elif action[2] == 1:  # Accelerate + Turn right
            self.acceleration = ACCELERATION
            self.direction = -1
        elif action[3] == 1:  # Base + Turn left
            self.acceleration = BASE_DECELERATION
            self.direction = 1
        elif action[4] == 1:  # Base + Straight
            self.acceleration = BASE_DECELERATION
            self.direction = 0
        elif action[5] == 1:  # Base + Turn right
            self.acceleration = BASE_DECELERATION
            self.direction = -1
        elif action[6] == 1:  # Brake + Turn left
            self.acceleration = -BRAKE_DECELERATION
            self.direction = 1
        elif action[7] == 1:  # Brake + Straight
            self.acceleration = -BRAKE_DECELERATION
            self.direction = 0
        elif action[8] == 1:  # Brake + Turn right
            self.acceleration = -BRAKE_DECELERATION
            self.direction = -1

        self.car_speed += self.acceleration
        self.deceleration = (1 + self.car_speed / MAX_SPEED) * BASE_DECELERATION
        self.car_speed -= self.deceleration
        self.car_speed = min(self.car_speed, MAX_SPEED)
        if self.car_speed < 0:
            self.car_speed = 0

        turn_speed = TURN_SPEED * (1.5 - self.car_speed / MAX_SPEED)
        if turn_speed < 1:
            turn_speed = 1
        if self.car_speed == 0:
            turn_speed = 0

        self.car_angle += self.direction * turn_speed
        self.car_angle %= 360
        rad = math.radians(-self.car_angle)
        self.car_x += self.car_speed * math.sin(rad)
        self.car_y -= self.car_speed * math.cos(rad)
        self.distance_pc = distance(self.next_checkpoint.center, (self.car_x, self.car_y))
        self.distance_bord = self.get_distances()


    def _show_lap_info(self):
        font = pygame.font.Font(None, 30)
        texts = [
            f'Lap: {self.count}',
            f'Last time: {self.last_time:.2f}s',
            f'Best time: {self.best_time:.2f}s',
            f'Lap time: {self.current_lap_time:.2f}s',
            f'CP time: {self.current_cp_time:.2f}s',
            f'Speed: {self.car_speed:.2f}',
            f'Score: {self.score}',
            f'Position: ({self.car_x}, {self.car_y})',
            f'Distance avec un bord de la piste: ({self.distance_bord})'

        ]
        for i, text in enumerate(texts):
            rendered_text = font.render(text, True, WHITE)
            self.screen.blit(rendered_text, (750, 400 + i * 20))


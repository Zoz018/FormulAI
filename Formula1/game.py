import pygame
import time
import math
from enum import Enum


class Direction(Enum):
    LEFT = 1
    STRAIGHT = 0
    RIGHT = -1

class Acceleration(Enum):
    BRAKE = -1
    BASE = 0
    ACCEL = 1

# Paramètres de la voiture
MAX_SPEED = 30
ACCELERATION = 0.5
BASE_DECELERATION = 0.1
SAND_DECELERATION = 0.4  # Décélération dans le sable
BRAKE_DECELERATION = 1
TURN_SPEED = 7

#
SPEED = 60

# Couleurs
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 71, 0)  # Couleur des bords du circuit pour la détection des collisions
YELLOW = (239, 228, 176)  # Couleur du sable
BLACK = (0, 0, 0)
BROWN = (120,67,21) 

TRACK = pygame.image.load('circuit_ovale.png')
TRACK = pygame.transform.scale(TRACK, (1920, 1080))
CAR = pygame.image.load('car.png')
CAR = pygame.transform.scale(CAR, (12.5, 25))


class FormulAI:

    def __init__(self, width=1920, height=1080) -> None:
        # Initialisation de Pygame
        pygame.init()
        
        # Dimensions de la fenêtre
        self.width = width
        self.height = height
        
        # Mise en place de la fenêtre
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('FormulAI')

        # Dessiner le circuit
        self.screen.fill(WHITE)
        self.screen.blit(TRACK, (0, 0))
            
        # Initialisation de la montre
        self.clock = pygame.time.Clock()

        # Initialiser l'état du jeu
        self.reset()

    def reset(self, car_x=1035, car_y=940, car_angle=90):
        # Initialisation de l'état du jeu
        self.direction = Direction.STRAIGHT
        self.acceleration = Acceleration.BASE


        self.car_x = car_x
        self.car_y = car_y
        self.car_speed = 0
        self.car_angle = car_angle

        self.start_time = time.time()
        self.current_lap_time = 0
        self.checkpoint = None
    
    def play_step(self):
        # Récupérer les entrées de l'utilisateur
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT

                if event.key == pygame.K_UP:
                    self.acceleration = Acceleration.ACCEL
                elif event.key == pygame.K_DOWN:
                    self.acceleration = Acceleration.BRAKE
            else:
                self.acceleration = Acceleration.BASE
                self.direction = Direction.STRAIGHT

        action = (self.direction, self.acceleration)
        
        # Faire bouger la voiture
        self._move(action)

        # Verifier le GameOver
        reward = 0
        game_over = False
        if self._is_collision():
            game_over = True
            reward -= 10
            return reward, game_over, self.current_lap_time
        
        # Checkpoint

        # Update UI and Clock
        self._update_ui()
        self.clock.tick(SPEED)
        self.current_lap_time = time.time() - self.start_time

        # Return game over and score
        return reward, game_over, self.current_lap_time

        

    def _move(self, action):
        # action = (direction, acceleration)

        # Calcul de l'accelération du véhicule 
        if action[1] == Acceleration.ACCEL:
            acceleration = ACCELERATION
        elif action[1] == Acceleration.BRAKE:
            acceleration = - BRAKE_DECELERATION
        else:
            acceleration = 0

        self.car_speed += acceleration

        deceleration = (1 + self.car_speed / MAX_SPEED) * BASE_DECELERATION

        # Actualiser la vitesse de la voiture
        self.car_speed -= deceleration

        self.car_speed = min(self.car_speed, MAX_SPEED)

        if self.car_speed < 0:
            self.car_speed = 0

        # Calcul de la vitesse de rotation en fonction de la vitesse de la voiture
        turn_speed = TURN_SPEED * (1.5 - self.car_speed / MAX_SPEED)
        if turn_speed < 1:  # Pour éviter que la voiture ne tourne pas du tout à haute vitesse
            turn_speed = 1  # Valeur minimale pour une sensibilité de direction constante
        if self.car_speed == 0:
            turn_speed = 0

        if action[0] == Direction.LEFT:
            turn_sign = 1
        elif action[0] == Direction.RIGHT:
            turn_sign = -1
        else:
            turn_sign = 0

        self.car_angle += turn_sign*turn_speed

        # Actualiser la position de la voiture
        self.car_x += self.car_speed * math.sin(math.radians(-self.car_angle))
        self.car_y -= self.car_speed * math.cos(math.radians(-self.car_angle))

    def _is_collision(self):
        # Vérification des collisions avec les bords de l'écran et du circuit
        if self.car_x < 0 or self.car_x > self.width or self.car_y < 0 or self.car_y > self.height:
            return True

        # Vérification des collisions avec les bords du circuit (vert)
        try:
            pixelColor = TRACK.get_at((int(self.car_x), int(self.car_y)))
            if pixelColor == GREEN or pixelColor == BROWN:
                return True
        except IndexError:
            pass
        
        return False

    def _update_ui(self):
        # Dessiner le circuit
        self.screen.fill(WHITE)
        self.screen.blit(TRACK, (0, 0))
        
        # Dessiner la voiture orientée
        rotated_car = pygame.transform.rotate(CAR, self.car_angle)
        car_rect = rotated_car.get_rect(center=(self.car_x, self.car_y))
        self.screen.blit(rotated_car, car_rect.topleft)
        pygame.display.update()
        
if __name__ == '__main__':
    game = FormulAI()
    
    # game loop
    while True:
        reward, game_over, score = game.play_step()
        
        if game_over == True:
            break
        
    print('Final Score', score)
        
        
    pygame.quit()
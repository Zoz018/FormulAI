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

# Paramètre du jeu
SPEED = 60

# Couleurs
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 71, 0)  # Couleur des bords du circuit pour la détection des collisions
YELLOW = (239, 228, 176)  # Couleur du sable
BLACK = (0, 0, 0)
BROWN = (120,67,21) 

# Détails du circuit
TRACK = pygame.image.load('assets/circuits/circuit_ovale.png')
TRACK = pygame.transform.scale(TRACK, (1920, 1080))
CAR = pygame.image.load('assets/car.png')
CAR = pygame.transform.scale(CAR, (12.5, 25))
CHECKPOINTS = [pygame.Rect(0,570,420,40) , pygame.Rect(960,0,40,325), pygame.Rect(1580,570,340,40), pygame.Rect(997,800,40,280)]

#Fonctions utiles 
def distance(point1, point2):

        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

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

    def reset(self, car_x=1035, car_y=940, car_angle=-40):
        # Initialisation de l'état du jeu
        self.direction = Direction.STRAIGHT
        self.acceleration = Acceleration.BASE


        self.car_x = car_x
        self.car_y = car_y
        self.car_speed = 0
        self.car_angle = car_angle

        self.start_time = time.time()
        self.current_lap_time = 0

        self.next_checkpoint = CHECKPOINTS[0]
        self.next_checkpoint_id = 0
        self.distance_pc = distance(self.next_checkpoint.center,(self.car_x , self.car_y))

        self.count = 0
        self.last_time = 0
        self.best_time = float("inf") 
    
    def play_step(self):
        # Récupérer les entrées de l'utilisateur
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.direction = Direction.LEFT
        elif keys[pygame.K_RIGHT]:
            self.direction = Direction.RIGHT
        else:
            self.direction = Direction.STRAIGHT

        if keys[pygame.K_UP]:
            self.acceleration = Acceleration.ACCEL
        elif keys[pygame.K_DOWN]:
            self.acceleration = Acceleration.BRAKE
        else:
            self.acceleration = Acceleration.BASE

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
        if self._checkpoint_collision():
            self.next_checkpoint_id += 1

            # Fin du tour
            if self.next_checkpoint_id >= len(CHECKPOINTS):
                self.next_checkpoint_id = 0
                self.count += 1
                if self.current_lap_time > self.best_time:
                    self.best_time = self.current_lap_time
                self.last_time = self.current_lap_time
                self.current_lap_time = 0
                self.start_time = time.time()

            self.next_checkpoint = CHECKPOINTS[self.next_checkpoint_id]
            


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
        self.car_angle %= 360

        # Actualiser la position de la voiture
        self.car_x += self.car_speed * math.sin(math.radians(-self.car_angle))
        self.car_y -= self.car_speed * math.cos(math.radians(-self.car_angle))

        #actualise la distance au prochain checkpoint
        self.distance_pc = distance(self.next_checkpoint.center,(self.car_x , self.car_y))

        
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
    
    def _checkpoint_collision(self):
        return self.next_checkpoint.collidepoint(self.car_x, self.car_y)
    
    def _update_ui(self):
        # Dessiner le circuit
        self.screen.fill(WHITE)
        self.screen.blit(TRACK, (0, 0))

        # Afficher le prochain checkpoint
        pygame.draw.rect(self.screen, WHITE, self.next_checkpoint)
        
        # Afficher les informations de la partie 
        self._show_lap_info()


        # Dessiner la voiture orientée
        rotated_car = pygame.transform.rotate(CAR, self.car_angle)
        car_rect = rotated_car.get_rect(center=(self.car_x, self.car_y))
        self.screen.blit(rotated_car, car_rect.topleft)
        pygame.display.update()

    # Fonction pour afficher le compteur de tours et le temps du tour précédent
    def _show_lap_info(self): 
        font = pygame.font.Font(None, 74)
        text_laps = font.render(f"Laps: {self.count}", True, WHITE)
        text_last_time = font.render(f"Last Lap Time: {self.last_time:.2f}s", True, WHITE)
        text_current_time = font.render(f"Current Lap Time: {self.current_lap_time:.2f}s", True, WHITE)
        text_car_speed = font.render(f"Car speed: {self.car_speed*10:.2f}km/h", True, WHITE)
        text_distance_prochain_checkpoint = font.render(f"Distance au prochain checkpoint: {self.distance_pc:.2f}m", True, WHITE)
        if self.best_time == float('inf') :
            text_best_time = font.render(f"No Lap Time",True, WHITE)
        else :
            text_best_time = font.render(f"Best Lap Time: {self.best_time:.2f}s", True, WHITE)
        information_text_position = (750,400)
        self.screen.blit(text_laps, information_text_position)
        self.screen.blit(text_last_time, (information_text_position[0], information_text_position[1] + 80))
        self.screen.blit(text_current_time, (information_text_position[0], information_text_position[1] + 160))
        self.screen.blit(text_best_time, (information_text_position[0], information_text_position[1] + 240))
        self.screen.blit(text_car_speed, (information_text_position[0], information_text_position[1] + 300))
        self.screen.blit(text_distance_prochain_checkpoint, (10,10))
        
if __name__ == '__main__':
    game = FormulAI()
    
    # game loop
    while True:
        reward, game_over, score = game.play_step()
        
        if game_over == True:
            break
        
    print('Final Score', score)
        
        
    pygame.quit()

import pygame
import sys
import math
import time
import os

def read_scores(file):
    if not os.path.exists(file):
        return []
    with open(file, 'r') as f:
        scores = f.readlines()
    return [float(score.strip()) for score in scores]

def write_scores(file, scores):
    with open(file, 'w') as f:
        for score in scores:
            f.write(f"{score}\n")

def update_scores(file, new_time):
    scores = read_scores(file)
    scores.append(new_time)
    scores = sorted(scores)[:10]  # Garder seulement les 10 meilleurs temps
    write_scores(file, scores)
    return scores

# Initialisation de Pygame
pygame.init()

# Dimensions de la fenêtre
screen_width = 1920
screen_height = 1080
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Jeu de Formule 1')

# Couleurs
white = (255, 255, 255)
red = (255, 0, 0)
green = (0, 71, 0)  # Couleur des bords du circuit pour la détection des collisions
yellow = (239, 228, 176)  # Couleur du sable
green_launcher = (144, 238, 144)  # Couleur de fond du launcher
black = (0, 0, 0)

# Charger les images de voitures et de circuits
car_image = pygame.image.load('car.png')
car_image = pygame.transform.scale(car_image, (12.5, 25))
track_images = {
    "spa": pygame.image.load('circuit_spa.png'),
    "ovale": pygame.image.load('circuit_ovale.png'),
    "ovale+sable": pygame.image.load('circuit_ovale_sable.png')
}
for key in track_images:
    track_images[key] = pygame.transform.scale(track_images[key], (1920, 1080))

# Données spécifiques à chaque circuit
track_data = {
    "spa": {
        "start_line_rect": pygame.Rect(420, 810, 10, 170),
        "car_x": 455,
        "car_y": 880,
        "information_text_position": (10, 10),
        "score_file": "scores_spa.txt",
        "best_time" : min(read_scores("scores_spa.txt"), default=float("inf"))  # Meilleur temps du circuit
    },
    "ovale": {
        "start_line_rect": pygame.Rect(992, 840, 10, 210),
        "car_x": 1035,
        "car_y": 940,
        "information_text_position": (700, 450),
        "score_file": "scores_ovale.txt",
        "best_time" : min(read_scores("scores_ovale.txt"), default=float("inf")) # Meilleur temps du circuit
    },
    "ovale+sable": {
        "start_line_rect": pygame.Rect(935, 700, 10, 265),
        "car_x": 1000,
        "car_y": 800,
        "information_text_position": (700, 350),
        "score_file": "scores_ovale_sable.txt",
        "best_time" : min(read_scores("scores_ovale_sable.txt"), default=float("inf")) # Meilleur temps du circuit
    }
}

# Initialisation des variables de position de la voiture
car_x = 0
car_y = 0
car_angle = 0
car_speed = 0

# Vitesse de la voiture
max_speed = 30
acceleration = 0.5
base_deceleration = 0.1  # Décélération de base
sand_deceleration = 0.4  # Décélération dans le sable
brake_deceleration = 1
base_turn_speed = 7

# Compteur de tours et chronomètre
lap_count = 0
crossed_line = False
start_time = time.time()
last_lap_time = 0
current_lap_time = 0


# Position précédente de la voiture
previous_car_x = car_x

# Score
scores = []

# Fonction pour dessiner la voiture avec rotation
def draw_car(x, y, angle):
    rotated_car = pygame.transform.rotate(car_image, angle)
    rect = rotated_car.get_rect(center=(x, y))
    screen.blit(rotated_car, rect.topleft)
    return rect

def show_game_over(selected_track, best_time):
    global last_lap_time

    scores = read_scores(track_data[selected_track]["score_file"])
    font = pygame.font.Font(None, 74)
    text = font.render("Game Over", True, red)

    if best_time == float('inf'):
        best_time_text = font.render(f"No Lap Time", True, white)
    else:
        best_time_text = font.render(f"Best Lap Time: {best_time:.2f}s", True, white)

    score_texts = [font.render(f"{i + 1}. {time:.2f}s", True, white) for i, time in enumerate(scores)]

    play_again_button = Button("Rejouer", screen_width // 2 - 100, screen_height // 2 + 100, 200, 50, green, green_launcher, lambda: reset_game(selected_track))
    quit_button = Button("Quitter", screen_width // 2 - 100, screen_height // 2 + 200, 200, 50, green, green_launcher, quit_game)
    change_track_button = Button("Choisir un circuit", screen_width // 2 - 125, screen_height // 2 + 300, 250, 50, green, green_launcher, show_track_selection)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            play_again_button.handle_event(event)
            quit_button.handle_event(event)
            change_track_button.handle_event(event)

        keys = pygame.key.get_pressed()

        if keys[pygame.K_RETURN]:
            reset_game(selected_track)

        if keys[pygame.K_ESCAPE]:
            quit_game()

        screen.fill(black)
        screen.blit(text, (screen_width / 2 - text.get_width() / 2, screen_height / 2 - text.get_height() / 2))
        screen.blit(best_time_text, (screen_width / 2 - best_time_text.get_width() / 2, screen_height // 2 - 100))

        for i, score_text in enumerate(score_texts):
            screen.blit(score_text, (screen_width // 2 - score_text.get_width() / 2 + 300, screen_height // 2 + (i + 1) * 40 - 90))

        play_again_button.draw(screen)
        quit_button.draw(screen)
        change_track_button.draw(screen)
        pygame.display.flip()


        

# Fonction pour afficher le compteur de tours et le temps du tour précédent
def show_lap_info(count, last_time, current_time, best_time, selected_track):
    font = pygame.font.Font(None, 74)
    text_laps = font.render(f"Laps: {count}", True, white)
    text_last_time = font.render(f"Last Lap Time: {last_time:.2f}s", True, white)
    text_current_time = font.render(f"Current Lap Time: {current_time:.2f}s", True, white)
    if best_time == float('inf') :
        text_best_time = font.render(f"No Lap Time",True, white)
    else :
        text_best_time = font.render(f"Best Lap Time: {best_time:.2f}s", True, white)
    information_text_position = track_data[selected_track]["information_text_position"]
    screen.blit(text_laps, information_text_position)
    screen.blit(text_last_time, (information_text_position[0], information_text_position[1] + 80))
    screen.blit(text_current_time, (information_text_position[0], information_text_position[1] + 160))
    screen.blit(text_best_time, (information_text_position[0], information_text_position[1] + 240))

# Définir une classe pour représenter un bouton
class Button:
    def __init__(self, text, x, y, width, height, color, hover_color, action=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.hover_color = hover_color
        self.text = text
        self.action = action

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)
        pygame.draw.rect(screen, black, self.rect, 2)  # Bordure noire
        font = pygame.font.Font(None, 36)
        text = font.render(self.text, True, black)
        text_rect = text.get_rect(center=self.rect.center)
        screen.blit(text, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                if self.action:
                    self.action()

# Définir une fonction pour afficher les options de menu
def show_menu_options():
    font = pygame.font.Font(None, 40)
    quit_button = Button("Quitter", screen_width // 2 - 100, 400, 200, 50, green, green_launcher, quit_game)
    change_track_button = Button("Choisir un circuit", screen_width // 2 - 150, 300, 300, 50, green, green_launcher, show_track_selection)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            quit_button.handle_event(event)
            change_track_button.handle_event(event)

        screen.fill(green_launcher)
        quit_button.draw(screen)
        change_track_button.draw(screen)
        pygame.display.flip()

# Ajoutez cette fonction pour gérer la sortie du jeu
def quit_game():
    pygame.quit()
    sys.exit()

# Fonction pour réinitialiser le jeu
def reset_game(selected_track):
    global car_x, car_y, car_angle, car_speed, lap_count, crossed_line, start_time, last_lap_time, current_lap_time, previous_car_x, best_time

    car_x = track_data[selected_track]["car_x"]
    car_y = track_data[selected_track]["car_y"]
    car_angle = 90
    car_speed = 0
    lap_count = 0
    crossed_line = False
    start_time = time.time()
    last_lap_time = 0
    current_lap_time = 0
    previous_car_x = car_x
    best_time = min(read_scores(track_data[selected_track]["score_file"]), default=float('inf'))

    game_loop(selected_track)


# Fonction principale du jeu
def game_loop(selected_track):
    global car_x, car_y, car_angle, car_speed, lap_count, crossed_line, start_time, last_lap_time, current_lap_time, previous_car_x



    # Extraire les informations spécifiques au circuit
    start_line_rect = track_data[selected_track]["start_line_rect"]

    lap_count = 0
    car_x = track_data[selected_track]["car_x"]
    car_y = track_data[selected_track]["car_y"]
    car_angle = 90
    car_speed = 0
    start_time = time.time()
    last_lap_time = 0
    current_lap_time = 0
    scores = read_scores(track_data[selected_track]["score_file"])
    best_time = min(scores, default=float('inf'))
    
    clock = pygame.time.Clock()
    game_exit = False

    deceleration = base_deceleration
    while not game_exit:
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        keys = pygame.key.get_pressed()

        if keys[pygame.K_UP]:
            if car_speed < max_speed:
                car_speed += acceleration
        elif keys[pygame.K_DOWN]:
            car_speed -= brake_deceleration
            if car_speed < 0:
                car_speed = 0
        
        # Appliquer une décélération dépendant de la vitesse actuelle (frottements)
        car_speed -= deceleration + (car_speed / max_speed) * deceleration
        if car_speed < 0:
            car_speed = 0

        # Calcul de la vitesse de rotation en fonction de la vitesse de la voiture
        turn_speed = base_turn_speed * (1.5 - car_speed / max_speed)
        if turn_speed < 1:  # Pour éviter que la voiture ne tourne pas du tout à haute vitesse
            turn_speed = 1  # Valeur minimale pour une sensibilité de direction constante

        if keys[pygame.K_LEFT] and car_speed > 0:
            car_angle += turn_speed
        if keys[pygame.K_RIGHT] and car_speed > 0:
            car_angle -= turn_speed

        car_x += car_speed * math.sin(math.radians(-car_angle))
        car_y -= car_speed * math.cos(math.radians(-car_angle))

        # Dessiner le circuit
        screen.fill(white)
        screen.blit(track_images[selected_track], (0, 0))

        # Dessiner la voiture et obtenir son rectangle
        car_rect = draw_car(car_x, car_y, car_angle)

        # Vérification des collisions avec les bords de l'écran et du circuit
        if car_rect.left < 0 or car_rect.right > screen_width or car_rect.top < 0 or car_rect.bottom > screen_height:
            show_game_over(selected_track,best_time)
            game_exit = True

        # Vérification des collisions avec les bords du circuit (vert)
        try:
            if track_images[selected_track].get_at((int(car_x), int(car_y))) == yellow:
                deceleration = sand_deceleration
            else:
                deceleration = base_deceleration
            if track_images[selected_track].get_at((int(car_x), int(car_y))) == green:
                show_game_over(selected_track,best_time)
                game_exit = True
        except IndexError:
            pass

        # Mise à jour du chronomètre du tour actuel
        current_lap_time = time.time() - start_time

        # Vérification du passage de la ligne de départ/arrivée
        if car_rect.colliderect(start_line_rect):
            if not crossed_line:
                if lap_count > 0:  # Ne pas afficher le temps du tour avant le premier tour complet
                    last_lap_time = current_lap_time
                    if last_lap_time > 0 and car_x < previous_car_x :
                        # Mettre à jour le meilleur temps
                        update_scores(track_data[selected_track]["score_file"], last_lap_time)
                start_time = time.time()
                lap_count += 1
                crossed_line = True
            # Vérifier si la voiture franchit la ligne de droite à gauche pour game over
            elif car_x > previous_car_x:  # La voiture franchit la ligne de droite à gauche
                last_lap_time = 0
                show_game_over(selected_track, best_time)
                game_exit = True
        else:
            crossed_line = False

        # Mise à jour de la position précédente de la voiture
        previous_car_x = car_x

        # Afficher le compteur de tours et les temps
        show_lap_info(lap_count, last_lap_time, current_lap_time, best_time, selected_track)

        # Si l'utilisateur appuie sur une touche pour afficher le menu
        if keys[pygame.K_ESCAPE]:
            show_menu_options()

        # Dessiner la ligne de départ/arrivée (commentée pour être invisible)
        #pygame.draw.rect(screen, red, start_line_rect)

        pygame.display.update()
        clock.tick(6000)

# Fonction pour afficher l'écran de lancement
def show_launcher():
    font = pygame.font.Font(None, 74)
    play_button_rect = pygame.Rect(screen_width / 2 - 100, screen_height / 2 - 50, 200, 100)
    screen.fill(green_launcher)
    pygame.draw.rect(screen, black, play_button_rect)
    text = font.render("Jouer", True, white)
    screen.blit(text, (screen_width / 2 - text.get_width() / 2, screen_height / 2 - text.get_height() / 2))
    pygame.display.flip()

    selecting_track = False
    selected_track = None

    while not selecting_track:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                if play_button_rect.collidepoint(mouse_x, mouse_y):
                    selecting_track = True

        keys = pygame.key.get_pressed()

        if keys[pygame.K_RETURN]:
            show_track_selection()

    show_track_selection()

# Fonction pour afficher la sélection des circuits
def show_track_selection():
    global track_images, track_data
    font = pygame.font.Font(None, 74)
    screen.fill(green_launcher)

    track_rects = []
    x = 100
    y = 200
    for key in track_images:
        img = pygame.transform.scale(track_images[key], (300, 200))
        rect = img.get_rect(topleft=(x, y))
        track_rects.append((key, rect))
        screen.blit(img, rect.topleft)
        x += 350
        if x > screen_width - 300:
            x = 100
            y += 250

    pygame.display.flip()

    selected_track = None
    while selected_track is None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                for key, rect in track_rects:
                    if rect.collidepoint(mouse_x, mouse_y):
                        selected_track = key

    # Retourner les données spécifiques au circuit sélectionné
    while True:
        game_loop(selected_track)

# Lancer l'écran de lancement
show_launcher()


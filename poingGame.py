import pygame
import cv2
import mediapipe as mp
import random
import time
import os
import json
from typing import List, Dict
import numpy as np

# Initialize Pygame
pygame.init()
pygame.mixer.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hand-Controlled Brick Breaker with Camera Background")

# Initialize OpenCV and Mediapipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Colors
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)

# Load images
brick_img_green_original = pygame.image.load("assets/brick_green.png").convert_alpha()
brick_img_blue_original = pygame.image.load("assets/brick_blue.png").convert_alpha()
brick_img_metal_light_original = pygame.image.load("assets/brick_metal_light.png").convert_alpha()
brick_img_metal_dark_original = pygame.image.load("assets/brick_metal_dark.png").convert_alpha()
ball_img_original = pygame.image.load("assets/ball_tennis.png").convert_alpha()
power_up_img_original = pygame.image.load("assets/power_up.png").convert_alpha()
debuff_img_original = pygame.image.load("assets/debuff.png").convert_alpha()
ball_split_img_original = pygame.image.load("assets/ballSplit.png").convert_alpha()
death_img_original = pygame.image.load("assets/deathimg.png").convert_alpha()

# Load sounds
try:
    pygame.mixer.music.load("assets/music.ogg")
    pygame.mixer.music.set_volume(0.5)
    pygame.mixer.music.play(-1)  # Loop indefinitely
except:
    print("Background music not found")

# Paddle setup
paddle_width, paddle_height = 100, 50
paddle = pygame.Rect(WIDTH // 2 - paddle_width // 2, HEIGHT - 50, paddle_width, paddle_height)

# Load paddle image and scale
paddle_img_original = pygame.image.load("assets/paddle-free.png").convert_alpha()

# Ball
ball_radius = 10

class Ball:
    def __init__(self, x, y, speed_x, speed_y):
        self.rect = pygame.Rect(x, y, ball_radius * 2, ball_radius * 2)
        self.speed = [speed_x, speed_y]
        self.active = True
        self.has_split = False  # Track if this ball has already split

    def update(self):
        self.rect.x += self.speed[0]
        self.rect.y += self.speed[1]

        if self.rect.left <= 0 or self.rect.right >= WIDTH:
            self.speed[0] = -self.speed[0]
        if self.rect.top <= 0:
            self.speed[1] = -self.speed[1]

    def draw(self, surface):
        ball_img = pygame.transform.scale(ball_img_original, (self.rect.width, self.rect.height))
        surface.blit(ball_img, self.rect.topleft)

# Bricks
brick_width, brick_height = 75, 30
bricks = []

# Score and Lives
score = 0
lives = 2
level = 1

# Font for displaying text
font = pygame.font.SysFont("Arial", 24)
title_font = pygame.font.SysFont("Arial", 48)
button_font = pygame.font.SysFont("Arial", 32)

# Power-up & Debuff System
power_ups = []
debuffs = []
ball_split_power_ups = []
death_debuffs = []
POWER_UP_SIZE = 20
DEBUFF_SIZE = 20
BALL_SPLIT_SIZE = 20
DEATH_SIZE = 20

# Paddle size management
PADDLE_SIZE_MULTIPLIER = 2  # Maximum size multiplier
PADDLE_SIZE_DIVIDER = 2    # Maximum size divider
current_paddle_multiplier = 1  # Current size multiplier (1 = normal size)

# Ball management
balls = []
initial_speed = 5

# Camera setup
cap = cv2.VideoCapture(0)

class Button:
    def __init__(self, x: int, y: int, width: int, height: int, text: str, color: tuple, hover_color: tuple):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.is_hovered = False

    def draw(self, surface):
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, WHITE, self.rect, 2)
        
        text_surface = button_font.render(self.text, True, WHITE)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.is_hovered:
                return True
        return False

class Menu:
    def __init__(self):
        button_width = 200
        button_height = 50
        spacing = 20
        start_y = HEIGHT // 2 - (button_height * 4 + spacing * 3) // 2

        self.new_game_btn = Button(WIDTH//2 - button_width//2, start_y, 
                                 button_width, button_height, "New Game", BLUE, GRAY)
        self.high_scores_btn = Button(WIDTH//2 - button_width//2, start_y + button_height + spacing,
                                    button_width, button_height, "High Scores", BLUE, GRAY)
        self.settings_btn = Button(WIDTH//2 - button_width//2, start_y + (button_height + spacing) * 2,
                                 button_width, button_height, "Settings", BLUE, GRAY)
        self.quit_btn = Button(WIDTH//2 - button_width//2, start_y + (button_height + spacing) * 3,
                             button_width, button_height, "Quit", BLUE, GRAY)

    def draw(self, surface):
        title = title_font.render("Brick Breaker", True, WHITE)
        title_rect = title.get_rect(center=(WIDTH//2, HEIGHT//4))
        surface.blit(title, title_rect)

        self.new_game_btn.draw(surface)
        self.high_scores_btn.draw(surface)
        self.settings_btn.draw(surface)
        self.quit_btn.draw(surface)

class HighScoreManager:
    def __init__(self):
        self.scores_file = "high_scores.json"
        self.scores = self.load_scores()

    def load_scores(self) -> List[Dict]:
        if os.path.exists(self.scores_file):
            with open(self.scores_file, 'r') as f:
                return json.load(f)
        return []

    def save_scores(self):
        with open(self.scores_file, 'w') as f:
            json.dump(self.scores, f)

    def add_score(self, name: str, score: int):
        self.scores.append({"name": name, "score": score})
        self.scores.sort(key=lambda x: x["score"], reverse=True)
        self.scores = self.scores[:5]  # Keep only top 5 scores
        self.save_scores()

    def get_top_scores(self) -> List[Dict]:
        return self.scores[:5]

class GameOverScreen:
    def __init__(self, score: int):
        self.score = score
        self.name = ""
        self.active = True
        self.input_rect = pygame.Rect(WIDTH//2 - 100, HEIGHT//2, 200, 32)
        self.high_score_manager = HighScoreManager()

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN and self.name:
                self.high_score_manager.add_score(self.name, self.score)
                return "menu"
            elif event.key == pygame.K_BACKSPACE:
                self.name = self.name[:-1]
            else:
                if len(self.name) < 10:  # Limit name length
                    self.name += event.unicode
        return None

    def draw(self, surface):
        # Draw semi-transparent overlay
        overlay = pygame.Surface((WIDTH, HEIGHT))
        overlay.set_alpha(128)
        overlay.fill(BLACK)
        surface.blit(overlay, (0, 0))

        # Draw game over text
        game_over_text = title_font.render("Game Over!", True, WHITE)
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        name_prompt = font.render("Enter your name:", True, WHITE)
        
        surface.blit(game_over_text, (WIDTH//2 - game_over_text.get_width()//2, HEIGHT//4))
        surface.blit(score_text, (WIDTH//2 - score_text.get_width()//2, HEIGHT//3))
        surface.blit(name_prompt, (WIDTH//2 - name_prompt.get_width()//2, HEIGHT//2 - 40))

        # Draw input box
        pygame.draw.rect(surface, WHITE, self.input_rect, 2)
        name_surface = font.render(self.name, True, WHITE)
        surface.blit(name_surface, (self.input_rect.x + 5, self.input_rect.y + 5))

        # Draw instructions
        instructions = font.render("Press ENTER to save score", True, WHITE)
        surface.blit(instructions, (WIDTH//2 - instructions.get_width()//2, HEIGHT//2 + 50))

class PauseScreen:
    def __init__(self):
        button_width = 200
        button_height = 50
        self.continue_btn = Button(WIDTH//2 - button_width//2, HEIGHT//2, 
                                 button_width, button_height, "Continue", BLUE, GRAY)

    def draw(self, surface):
        # Draw semi-transparent overlay
        overlay = pygame.Surface((WIDTH, HEIGHT))
        overlay.set_alpha(128)
        overlay.fill(BLACK)
        surface.blit(overlay, (0, 0))

        # Draw pause text
        pause_text = title_font.render("Game Paused", True, WHITE)
        surface.blit(pause_text, (WIDTH//2 - pause_text.get_width()//2, HEIGHT//4))

        # Draw continue button
        self.continue_btn.draw(surface)

    def handle_event(self, event):
        if self.continue_btn.handle_event(event):
            return "continue"
        return None

class GameSettings:
    def __init__(self):
        self.fullscreen = False
        self.music_enabled = True
        self.load_settings()

    def load_settings(self):
        try:
            with open("settings.json", "r") as f:
                settings = json.load(f)
                self.fullscreen = settings.get("fullscreen", False)
                self.music_enabled = settings.get("music_enabled", True)
        except:
            self.save_settings()

    def save_settings(self):
        settings = {
            "fullscreen": self.fullscreen,
            "music_enabled": self.music_enabled
        }
        with open("settings.json", "w") as f:
            json.dump(settings, f)

    def apply_settings(self):
        if self.music_enabled:
            pygame.mixer.music.set_volume(0.5)
            pygame.mixer.music.play(-1)
        else:
            pygame.mixer.music.stop()

        if self.fullscreen:
            # Get the screen info
            screen_info = pygame.display.Info()
            screen_width = screen_info.current_w
            screen_height = screen_info.current_h
            
            # Calculate the scaling factor to maintain aspect ratio
            scale_x = screen_width / WIDTH
            scale_y = screen_height / HEIGHT
            scale = min(scale_x, scale_y)
            
            # Calculate new dimensions
            new_width = int(WIDTH * scale)
            new_height = int(HEIGHT * scale)
            
            # Center the window
            x = (screen_width - new_width) // 2
            y = (screen_height - new_height) // 2
            
            # Set the window position and size
            os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x},{y}"
            screen = pygame.display.set_mode((new_width, new_height), pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF)
        else:
            # Reset window position for windowed mode
            os.environ['SDL_VIDEO_WINDOW_POS'] = ""
            screen = pygame.display.set_mode((WIDTH, HEIGHT))

class Checkbox:
    def __init__(self, x, y, size, text, checked=False):
        self.rect = pygame.Rect(x, y, size, size)
        self.text = text
        self.checked = checked
        self.text_surface = font.render(text, True, WHITE)
        self.text_rect = self.text_surface.get_rect(midleft=(x + size + 10, y + size//2))
        self.click_rect = pygame.Rect(x, y, size + self.text_rect.width + 10, size)  # Clickable area includes text

    def draw(self, surface):
        # Draw checkbox
        pygame.draw.rect(surface, WHITE, self.rect, 2)
        if self.checked:
            pygame.draw.line(surface, WHITE, (self.rect.x + 5, self.rect.centery),
                           (self.rect.x + self.rect.width//2, self.rect.bottom - 5), 2)
            pygame.draw.line(surface, WHITE, (self.rect.x + self.rect.width//2, self.rect.bottom - 5),
                           (self.rect.right - 5, self.rect.top + 5), 2)
        
        # Draw text
        surface.blit(self.text_surface, self.text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.click_rect.collidepoint(event.pos):
                self.checked = not self.checked
                return True
        return False

class SettingsMenu:
    def __init__(self, settings):
        self.settings = settings
        button_width = 200
        button_height = 50
        checkbox_size = 20
        spacing = 40
        start_y = HEIGHT // 4

        # Display mode section
        self.display_title = font.render("Display Mode", True, WHITE)
        self.display_title_pos = (WIDTH//2 - self.display_title.get_width()//2, start_y)
        
        # Replace buttons with checkboxes
        self.windowed_checkbox = Checkbox(WIDTH//2 - checkbox_size - 100, start_y + spacing,
                                        checkbox_size, "Windowed", not self.settings.fullscreen)
        self.fullscreen_checkbox = Checkbox(WIDTH//2 - checkbox_size - 100, start_y + spacing * 2,
                                          checkbox_size, "Fullscreen", self.settings.fullscreen)

        # Audio section
        self.music_checkbox = Checkbox(WIDTH//2 - checkbox_size - 100, start_y + spacing * 4,
                                     checkbox_size, "Music", self.settings.music_enabled)

        # Back button at the bottom
        self.back_btn = Button(WIDTH//2 - button_width//2, HEIGHT - button_height - 50,
                             button_width, button_height, "Back", BLUE, GRAY)

    def draw(self, surface):
        # Draw title
        title = title_font.render("Settings", True, WHITE)
        surface.blit(title, (WIDTH//2 - title.get_width()//2, HEIGHT//8))

        # Draw display mode section
        surface.blit(self.display_title, self.display_title_pos)
        self.windowed_checkbox.draw(surface)
        self.fullscreen_checkbox.draw(surface)

        # Draw audio section
        self.music_checkbox.draw(surface)

        # Draw back button
        self.back_btn.draw(surface)

    def handle_event(self, event):
        # Handle checkboxes
        if self.windowed_checkbox.handle_event(event):
            self.windowed_checkbox.checked = True
            self.fullscreen_checkbox.checked = False
            self.settings.fullscreen = False
            self.settings.apply_settings()
        elif self.fullscreen_checkbox.handle_event(event):
            self.fullscreen_checkbox.checked = True
            self.windowed_checkbox.checked = False
            self.settings.fullscreen = True
            self.settings.apply_settings()
        elif self.music_checkbox.handle_event(event):
            self.settings.music_enabled = self.music_checkbox.checked
            self.settings.apply_settings()

        # Handle back button
        if self.back_btn.handle_event(event):
            self.settings.save_settings()
            return "menu"
        
        return None

# Level designs with custom shapes
LEVEL_DESIGNS = {
    1: {
        "shape": [
            "1111111",
            "   1   ",
            "1111111",
            "   1   ",
            "1111111"
        ],
        "colors": ["green", "blue"],
        "points": {"green": 1, "blue": 5}
    },
    2: {
        "shape": [
            "  111  ",
            " 11111 ",
            "1111111"
        ],
        "colors": ["green", "blue", "metal_light"],
        "points": {"green": 1, "blue": 5, "metal_light": 0, "metal_dark": 2}
    },
    3: {
        "shape": [
            "   1   ",
            "  111  ",
            " 11111 ",
            "1111111",
            " 11111 ",
            "  111  ",
            "   1   "
        ],
        "colors": ["green", "blue", "metal_light"],
        "points": {"green": 1, "blue": 5, "metal_light": 0, "metal_dark": 2}
    },
    4: {
        "shape": [
            "1     1",
            " 1   1 ",
            "  1 1  ",
            "   1   ",
            "  1 1  ",
            " 1   1 ",
            "1     1"
        ],
        "colors": ["green", "blue", "metal_light"],
        "points": {"green": 1, "blue": 5, "metal_light": 0, "metal_dark": 2}
    },
    5: {
        "shape": [
            "   1   ",
            "  111  ",
            " 11111 ",
            "1111111",
            " 11111 ",
            "  111  ",
            "   1   ",
            "  111  ",
            " 11111 ",
            "1111111"
        ],
        "colors": ["green", "blue", "metal_light"],
        "points": {"green": 1, "blue": 5, "metal_light": 0, "metal_dark": 2}
    }
}

def generate_bricks():
    bricks.clear()
    level_design = LEVEL_DESIGNS.get(level, LEVEL_DESIGNS[1])
    
    # Get the shape pattern
    shape = level_design["shape"]
    rows = len(shape)
    cols = len(shape[0])
    
    # Calculate total width and starting position
    total_width = cols * (brick_width + 10) - 10
    start_x = (WIDTH - total_width) // 2
    start_y = 50  # Starting y position

    # Place bricks according to the shape pattern
    for y, row in enumerate(shape):
        for x, cell in enumerate(row):
            if cell == "1":  # Only place brick where there's a "1" in the pattern
                brick_color = random.choice(level_design["colors"])
                color = BLUE if brick_color == "blue" else GREEN
                if brick_color == "metal_light":
                    color = "metal_light"
                points = level_design["points"][brick_color]
                bricks.append({
                    "rect": pygame.Rect(start_x + x * (brick_width + 10),
                                     start_y + y * (brick_height + 10),
                                     brick_width, brick_height),
                    "color": color,
                    "points": points,
                    "state": "light" if color == "metal_light" else None
                })

def detect_hand(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * WIDTH)
            paddle.centerx = max(min(x, WIDTH - paddle.width // 2), paddle.width // 2)
    return frame

def resize_frame(frame, target_width, target_height):
    return cv2.resize(frame, (target_width, target_height))

def flip_frame(frame):
    return cv2.flip(frame, 1)

def convert_frame_to_pygame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return pygame.surfarray.make_surface(frame.transpose([1, 0, 2]))

def display_text(text, x, y, color):
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, (x, y))

def show_high_scores(surface):
    high_score_manager = HighScoreManager()
    scores = high_score_manager.get_top_scores()
    
    # Draw semi-transparent overlay
    overlay = pygame.Surface((WIDTH, HEIGHT))
    overlay.set_alpha(128)
    overlay.fill(BLACK)
    surface.blit(overlay, (0, 0))
    
    # Draw title
    title = title_font.render("High Scores", True, WHITE)
    surface.blit(title, (WIDTH//2 - title.get_width()//2, HEIGHT//4))
    
    # Draw scores
    if not scores:
        no_scores = font.render("No scores yet!", True, WHITE)
        surface.blit(no_scores, (WIDTH//2 - no_scores.get_width()//2, HEIGHT//2))
    else:
        for i, score in enumerate(scores):
            score_text = font.render(f"{i+1}. {score['name']}: {score['score']}", True, WHITE)
            surface.blit(score_text, (WIDTH//2 - score_text.get_width()//2, HEIGHT//2 + i * 40))
    
    # Draw back button
    back_text = font.render("Press ESC to return to menu", True, WHITE)
    surface.blit(back_text, (WIDTH//2 - back_text.get_width()//2, HEIGHT - 50))

def reset_game():
    global balls, paddle, bricks, score, lives, level, initial_speed, current_paddle_multiplier
    balls = [Ball(WIDTH // 2 - ball_radius, HEIGHT // 2 - ball_radius, initial_speed, -initial_speed)]
    paddle = pygame.Rect(WIDTH // 2 - paddle_width // 2, HEIGHT - 50, paddle_width, paddle_height)
    score = 0
    lives = 2
    level = 1
    current_paddle_multiplier = 1
    generate_bricks()

def count_fingers(hand_landmarks):
    finger_tips = [8, 12, 16, 20]  # Index, middle, ring, pinky finger tips
    thumb_tip = 4
    finger_base = [6, 10, 14, 18]  # Base of each finger
    thumb_base = 2

    # Check if thumb is up
    thumb_up = hand_landmarks.landmark[thumb_tip].y < hand_landmarks.landmark[thumb_base].y

    # Count other fingers
    fingers_up = 0
    for tip, base in zip(finger_tips, finger_base):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y:
            fingers_up += 1

    # Add thumb if it's up
    if thumb_up:
        fingers_up += 1

    return fingers_up

def show_countdown(surface, number):
    # Draw semi-transparent overlay
    overlay = pygame.Surface((WIDTH, HEIGHT))
    overlay.set_alpha(128)
    overlay.fill(BLACK)
    surface.blit(overlay, (0, 0))

    # Draw level text
    level_text = title_font.render(f"Level {level}", True, WHITE)
    surface.blit(level_text, (WIDTH//2 - level_text.get_width()//2, HEIGHT//3))

    # Draw countdown number
    count_text = title_font.render(str(number), True, WHITE)
    surface.blit(count_text, (WIDTH//2 - count_text.get_width()//2, HEIGHT//2))

def apply_blur(frame, kernel_size=25):
    # Apply multiple passes of blur for stronger effect
    blurred = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
    blurred = cv2.GaussianBlur(blurred, (kernel_size, kernel_size), 0)
    # Add slight darkening effect
    blurred = cv2.addWeighted(blurred, 0.8, np.zeros_like(blurred), 0.2, 0)
    return blurred

def split_ball(ball):
    # Create two new balls with opposite horizontal speeds
    new_balls = []
    for speed_x in [-ball.speed[0], ball.speed[0]]:
        new_ball = Ball(ball.rect.x, ball.rect.y, speed_x, ball.speed[1])
        new_balls.append(new_ball)
    return new_balls

def multiply_balls():
    # Get current number of balls
    current_ball_count = len(balls)
    # Calculate how many new balls we need to create
    new_balls_needed = current_ball_count
    
    # Create new balls
    new_balls = []
    for _ in range(new_balls_needed):
        # Create a new ball with random direction
        speed_x = random.choice([-initial_speed, initial_speed])
        speed_y = -initial_speed
        new_ball = Ball(WIDTH // 2 - ball_radius, HEIGHT // 2 - ball_radius, speed_x, speed_y)
        new_balls.append(new_ball)
    
    return new_balls

def update_paddle_size():
    global paddle, current_paddle_multiplier
    paddle.width = paddle_width * current_paddle_multiplier
    # Keep paddle centered
    paddle.centerx = min(max(paddle.centerx, paddle.width // 2), WIDTH - paddle.width // 2)

# Game states
MENU = "menu"
PLAYING = "playing"
GAME_OVER = "game_over"
HIGH_SCORES = "high_scores"
PAUSED = "paused"
COUNTDOWN = "countdown"
SETTINGS = "settings"

# Initialize game objects
settings = GameSettings()
menu = Menu()
settings_menu = SettingsMenu(settings)
current_state = MENU
game_over_screen = None
pause_screen = PauseScreen()
countdown_number = 3
countdown_start_time = 0

clock = pygame.time.Clock()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
        if current_state == MENU:
            if menu.new_game_btn.handle_event(event):
                current_state = COUNTDOWN
                reset_game()
                countdown_number = 3
                countdown_start_time = time.time()
            elif menu.high_scores_btn.handle_event(event):
                current_state = HIGH_SCORES
            elif menu.settings_btn.handle_event(event):
                current_state = SETTINGS
            elif menu.quit_btn.handle_event(event):
                running = False
                
        elif current_state == GAME_OVER:
            result = game_over_screen.handle_event(event)
            if result == "menu":
                current_state = MENU
                
        elif current_state == HIGH_SCORES:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                current_state = MENU
                
        elif current_state == PAUSED:
            result = pause_screen.handle_event(event)
            if result == "continue":
                current_state = PLAYING

        elif current_state == SETTINGS:
            result = settings_menu.handle_event(event)
            if result == "menu":
                current_state = MENU

    ret, frame = cap.read()
    if not ret:
        break

    frame = flip_frame(frame)
    frame = resize_frame(frame, WIDTH, HEIGHT)
    background = convert_frame_to_pygame(frame)
    screen.blit(background, (0, 0))

    if current_state == MENU:
        # Apply blur effect to the camera feed
        frame = apply_blur(frame)
        background = convert_frame_to_pygame(frame)
        screen.blit(background, (0, 0))
        menu.draw(screen)
        
    elif current_state == HIGH_SCORES:
        # Apply blur effect to the camera feed
        frame = apply_blur(frame)
        background = convert_frame_to_pygame(frame)
        screen.blit(background, (0, 0))
        show_high_scores(screen)
        
    elif current_state == GAME_OVER:
        # Apply blur effect to the camera feed
        frame = apply_blur(frame)
        background = convert_frame_to_pygame(frame)
        screen.blit(background, (0, 0))
        game_over_screen.draw(screen)
        
    elif current_state == PAUSED:
        # Apply blur effect to the camera feed
        frame = apply_blur(frame)
        background = convert_frame_to_pygame(frame)
        screen.blit(background, (0, 0))
        pause_screen.draw(screen)
        
    elif current_state == COUNTDOWN:
        # Apply blur effect to the camera feed
        frame = apply_blur(frame)
        background = convert_frame_to_pygame(frame)
        screen.blit(background, (0, 0))
        
        elapsed_time = time.time() - countdown_start_time
        if elapsed_time >= 1:
            countdown_number -= 1
            countdown_start_time = time.time()
            if countdown_number < 0:
                current_state = PLAYING
        
        show_countdown(screen, countdown_number)
        
    elif current_state == SETTINGS:
        # Apply blur effect to the camera feed
        frame = apply_blur(frame)
        background = convert_frame_to_pygame(frame)
        screen.blit(background, (0, 0))
        settings_menu.draw(screen)
        
    elif current_state == PLAYING:
        frame = detect_hand(frame)
        background = convert_frame_to_pygame(frame)
        screen.blit(background, (0, 0))

        # Check for pause gesture (5 fingers)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                finger_count = count_fingers(hand_landmarks)
                if finger_count == 5:
                    current_state = PAUSED
                else:
                    x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * WIDTH)
                    paddle.centerx = max(min(x, WIDTH - paddle.width // 2), paddle.width // 2)

        # Draw bricks first (so they appear behind other elements)
        for brick in bricks:
            if brick["color"] == GREEN:
                brick_img = pygame.transform.scale(brick_img_green_original, (brick["rect"].width, brick["rect"].height))
                screen.blit(brick_img, brick["rect"].topleft)
            elif brick["color"] == BLUE:
                brick_img = pygame.transform.scale(brick_img_blue_original, (brick["rect"].width, brick["rect"].height))
                screen.blit(brick_img, brick["rect"].topleft)
            elif brick["color"] == "metal_light":
                brick_img = pygame.transform.scale(brick_img_metal_light_original, (brick["rect"].width, brick["rect"].height))
                screen.blit(brick_img, brick["rect"].topleft)
            elif brick["color"] == "metal_dark":
                brick_img = pygame.transform.scale(brick_img_metal_dark_original, (brick["rect"].width, brick["rect"].height))
                screen.blit(brick_img, brick["rect"].topleft)

        # Update and draw balls
        for ball in balls[:]:
            ball.update()
            ball.draw(screen)

            # Check for ball collision with paddle
            if ball.rect.colliderect(paddle):
                ball.speed[1] = -ball.speed[1]
                offset = (ball.rect.centerx - paddle.centerx) / (paddle_width // 2)
                ball.speed[0] += int(offset * 5)

            # Check for ball collision with bricks
            for brick in bricks[:]:
                if ball.rect.colliderect(brick["rect"]):
                    current_level_design = LEVEL_DESIGNS.get(level, LEVEL_DESIGNS[1])
                    if brick["color"] == "metal_light":
                        # Change metal light brick to dark
                        brick["color"] = "metal_dark"
                        brick["points"] = current_level_design["points"]["metal_dark"]
                        brick["state"] = "dark"
                        # Bounce the ball
                        if abs(ball.rect.centerx - brick["rect"].left) < 10 or abs(ball.rect.centerx - brick["rect"].right) < 10:
                            ball.speed[0] = -ball.speed[0]
                        if abs(ball.rect.centery - brick["rect"].top) < 10 or abs(ball.rect.centery - brick["rect"].bottom) < 10:
                            ball.speed[1] = -ball.speed[1]
                    elif brick["color"] == "metal_dark":
                        # Remove dark metal brick and add points
                        bricks.remove(brick)
                        score += brick["points"]
                        if abs(ball.rect.centerx - brick["rect"].left) < 10 or abs(ball.rect.centerx - brick["rect"].right) < 10:
                            ball.speed[0] = -ball.speed[0]
                        if abs(ball.rect.centery - brick["rect"].top) < 10 or abs(ball.rect.centery - brick["rect"].bottom) < 10:
                            ball.speed[1] = -ball.speed[1]
                        if random.random() < 0.2:
                            power_ups.append(pygame.Rect(brick["rect"].centerx, brick["rect"].centery, POWER_UP_SIZE, POWER_UP_SIZE))
                        if random.random() < 0.15:
                            debuffs.append(pygame.Rect(brick["rect"].centerx, brick["rect"].centery, DEBUFF_SIZE, DEBUFF_SIZE))
                        if random.random() < 0.1:
                            ball_split_power_ups.append(pygame.Rect(brick["rect"].centerx, brick["rect"].centery, BALL_SPLIT_SIZE, BALL_SPLIT_SIZE))
                        if random.random() < 0.05:
                            death_debuffs.append(pygame.Rect(brick["rect"].centerx, brick["rect"].centery, DEATH_SIZE, DEATH_SIZE))
                    else:
                        # Normal brick behavior
                        bricks.remove(brick)
                        score += brick["points"]
                        if abs(ball.rect.centerx - brick["rect"].left) < 10 or abs(ball.rect.centerx - brick["rect"].right) < 10:
                            ball.speed[0] = -ball.speed[0]
                        if abs(ball.rect.centery - brick["rect"].top) < 10 or abs(ball.rect.centery - brick["rect"].bottom) < 10:
                            ball.speed[1] = -ball.speed[1]
                        if random.random() < 0.2:
                            power_ups.append(pygame.Rect(brick["rect"].centerx, brick["rect"].centery, POWER_UP_SIZE, POWER_UP_SIZE))
                        if random.random() < 0.15:
                            debuffs.append(pygame.Rect(brick["rect"].centerx, brick["rect"].centery, DEBUFF_SIZE, DEBUFF_SIZE))
                        if random.random() < 0.1:
                            ball_split_power_ups.append(pygame.Rect(brick["rect"].centerx, brick["rect"].centery, BALL_SPLIT_SIZE, BALL_SPLIT_SIZE))
                        if random.random() < 0.05:
                            death_debuffs.append(pygame.Rect(brick["rect"].centerx, brick["rect"].centery, DEATH_SIZE, DEATH_SIZE))

            # Check if ball is out of bounds
            if ball.rect.bottom >= HEIGHT:
                balls.remove(ball)
                if not balls:  # If no balls left
                    lives -= 1
                    if lives < 0:
                        current_state = GAME_OVER
                        game_over_screen = GameOverScreen(score)
                    else:
                        balls = [Ball(WIDTH // 2 - ball_radius, HEIGHT // 2 - ball_radius, initial_speed, -initial_speed)]

        # Update and draw ball split power-ups
        for power_up in ball_split_power_ups[:]:
            power_up.y += 5
            power_up_img = pygame.transform.scale(ball_split_img_original, (power_up.width, power_up.height))
            screen.blit(power_up_img, power_up.topleft)
            if paddle.colliderect(power_up):
                ball_split_power_ups.remove(power_up)
                # Double the current number of balls
                new_balls = multiply_balls()
                balls.extend(new_balls)
            elif power_up.top > HEIGHT:
                ball_split_power_ups.remove(power_up)

        # Power-ups
        for power_up in power_ups[:]:
            power_up.y += 5
            power_up_img = pygame.transform.scale(power_up_img_original, (power_up.width, power_up.height))
            screen.blit(power_up_img, power_up.topleft)
            if paddle.colliderect(power_up):
                power_ups.remove(power_up)
                if current_paddle_multiplier < PADDLE_SIZE_MULTIPLIER:
                    current_paddle_multiplier += 1
                    update_paddle_size()
            elif power_up.top > HEIGHT:
                power_ups.remove(power_up)

        # Debuffs
        for debuff in debuffs[:]:
            debuff.y += 5
            debuff_img = pygame.transform.scale(debuff_img_original, (debuff.width, debuff.height))
            screen.blit(debuff_img, debuff.topleft)
            if paddle.colliderect(debuff):
                debuffs.remove(debuff)
                if current_paddle_multiplier > 1/PADDLE_SIZE_DIVIDER:
                    current_paddle_multiplier -= 0.5
                    update_paddle_size()
            elif debuff.top > HEIGHT:
                debuffs.remove(debuff)

        # Update and draw death debuffs
        for debuff in death_debuffs[:]:
            debuff.y += 5
            debuff_img = pygame.transform.scale(death_img_original, (debuff.width, debuff.height))
            screen.blit(debuff_img, debuff.topleft)
            if paddle.colliderect(debuff):
                death_debuffs.remove(debuff)
                lives -= 1
                if lives < 0:
                    current_state = GAME_OVER
                    game_over_screen = GameOverScreen(score)
                else:
                    # Clear all existing balls and create a new one
                    balls.clear()
                    balls = [Ball(WIDTH // 2 - ball_radius, HEIGHT // 2 - ball_radius, initial_speed, -initial_speed)]
            elif debuff.top > HEIGHT:
                death_debuffs.remove(debuff)

        if not bricks:
            current_state = COUNTDOWN
            countdown_number = 3
            countdown_start_time = time.time()
            level += 1
            generate_bricks()
            initial_speed *= 1.2
            # Reset balls with new speed
            balls = [Ball(WIDTH // 2 - ball_radius, HEIGHT // 2 - ball_radius, initial_speed, -initial_speed)]

        # Draw paddle image
        paddle_img = pygame.transform.scale(paddle_img_original, (paddle.width, paddle.height))
        screen.blit(paddle_img, (paddle.x, paddle.y))

        # Draw text
        display_text(f"Score: {score}", 10, 10, WHITE)
        display_text(f"Lives: {lives}", 10, 40, WHITE)

    pygame.display.flip()
    clock.tick(60)

cap.release()
pygame.quit()

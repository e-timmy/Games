import os
import random
import pygame
from constants import *
from entities import Player, Platform


class Game:
    def __init__(self):
        self.state = TITLE_SCREEN
        self.high_scores = self.load_high_scores()
        self.frame_count = 0
        self.highest_generated_y = 0
        self.camera_y = 0
        self.target_camera_y = 0  # New variable for smooth camera movement
        log_debug("Game initialized")
        self.reset_game()

    def reset_game(self):
        log_debug("Game reset triggered")
        self.player = Player()
        self.platforms = []
        self.camera_y = 0
        self.target_camera_y = 0
        self.score = 0
        self.current_name = []
        self.name_cursor_blink = 0
        self.platforms_created = 0
        self.platforms_removed = 0
        self.highest_generated_y = 0
        self.create_starting_platforms()
        log_debug(f"Game reset complete. Initial platform count: {len(self.platforms)}")

    def load_high_scores(self):
        if not os.path.exists(HIGH_SCORES_FILE):
            log_debug(f"High scores file not found: {HIGH_SCORES_FILE}")
            return []

        try:
            with open(HIGH_SCORES_FILE, 'r') as f:
                lines = f.readlines()
                high_scores = []
                for line in lines:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            name = parts[0][:3]
                            try:
                                score = int(parts[1])
                                high_scores.append((name, score))
                            except ValueError:
                                continue

                high_scores.sort(key=lambda x: x[1], reverse=True)
                log_debug(f"Loaded {len(high_scores)} high scores")
                return high_scores[:10]
        except Exception as e:
            log_debug(f"Error loading high scores: {e}")
            return []

    def save_high_scores(self):
        try:
            with open(HIGH_SCORES_FILE, 'w') as f:
                for name, score in self.high_scores:
                    f.write(f"{name} {score}\n")
            log_debug(f"Saved {len(self.high_scores)} high scores")
        except Exception as e:
            log_debug(f"Error saving high scores: {e}")

    def add_high_score(self, name, score):
        self.high_scores.append((name, score))
        self.high_scores.sort(key=lambda x: x[1], reverse=True)
        self.high_scores = self.high_scores[:10]
        log_debug(f"Added high score: {name}={score}")
        self.save_high_scores()

    def calculate_difficulty_factor(self):
        if self.score < 100:
            return 0
        elif self.score > 1000:
            return 1
        else:
            return (self.score - 100) / 900

    def create_platform(self, x, y, width):
        difficulty = self.calculate_difficulty_factor()

        # Adjust platform type probabilities based on difficulty
        normal_chance = 1 - (0.8 * difficulty)  # 100% at start, 20% at max difficulty
        moving_chance = 0.4 * difficulty  # 0% at start, 40% at max difficulty
        breakable_chance = 0.4 * difficulty  # 0% at start, 40% at max difficulty

        rand = random.random()

        # More explicit range checks
        if rand < normal_chance:
            platform_type = NORMAL_PLATFORM
        elif normal_chance <= rand < (normal_chance + moving_chance):
            platform_type = MOVING_PLATFORM
        else:
            platform_type = BREAKABLE_PLATFORM

        # Adjust width based on difficulty
        max_width = PLATFORM_MAX_WIDTH - int(difficulty * 50)  # Reduce max width by up to 50 pixels
        width = max(PLATFORM_MIN_WIDTH, min(width, max_width))

        log_debug(f"Creating platform: type={platform_type}, width={width}, difficulty={difficulty:.2f}")
        log_debug(
            f"Chances: normal={normal_chance:.2f}, moving={moving_chance:.2f}, breakable={breakable_chance:.2f}, rand={rand:.2f}")

        return Platform(x, y, width, platform_type)

    def generate_platforms_ahead(self):
        if not self.platforms:
            log_debug("No platforms exist! Creating starting platforms")
            self.create_starting_platforms()
            return

        current_highest = min(p.rect.y for p in self.platforms)
        y = current_highest - PLATFORM_GAP
        target_y = current_highest - HEIGHT
        platforms_added = 0

        difficulty = self.calculate_difficulty_factor()

        # Adjust gap based on difficulty
        min_gap = PLATFORM_GAP
        max_gap = PLATFORM_GAP + int(difficulty * 50)  # Increase max gap by up to 50 pixels

        log_debug(f"Generating platforms from y={y} to y={target_y}, difficulty={difficulty:.2f}")

        while y > target_y:
            width = random.randint(PLATFORM_MIN_WIDTH, int(PLATFORM_MAX_WIDTH - (difficulty * 50)))
            x = random.randint(0, WIDTH - width)

            new_platform = self.create_platform(x, y, width)
            self.platforms.append(new_platform)

            # Use the adjusted gap
            y -= random.randint(int(min_gap), int(max_gap))
            platforms_added += 1
            self.platforms_created += 1

        self.highest_generated_y = min(p.rect.y for p in self.platforms)

        log_debug(f"Added {platforms_added} platforms ahead. New highest at y={self.highest_generated_y}")

    def create_starting_platforms(self):
        log_debug("Creating starting platforms")
        y = HEIGHT - GROUND_HEIGHT - PLATFORM_GAP
        platform_count = 0

        while y > -HEIGHT:
            width = random.randint(PLATFORM_MIN_WIDTH, PLATFORM_MAX_WIDTH)
            x = random.randint(0, WIDTH - width)

            self.platforms.append(Platform(x, y, width, NORMAL_PLATFORM))
            y -= random.randint(PLATFORM_GAP - 20, PLATFORM_GAP + 20)
            platform_count += 1
            self.platforms_created += 1

        if self.platforms:
            self.highest_generated_y = min(p.rect.y for p in self.platforms)

        log_debug(f"Created {platform_count} starting platforms, highest at y={self.highest_generated_y}")

    def update(self):
        self.frame_count += 1

        if self.frame_count % 60 == 0 and self.state == PLAYING:
            camera_info = f"Camera: {self.camera_y}"
            player_info = f"Player: y={self.player.rect.y}, vy={self.player.velocity_y}"
            platform_info = f"Platforms: {len(self.platforms)}"
            if self.platforms:
                highest_y = min(p.rect.y for p in self.platforms)
                lowest_y = max(p.rect.y for p in self.platforms)
                platform_range = f"Range: {highest_y} to {lowest_y}"
                log_debug(f"{camera_info} | {player_info} | {platform_info} | {platform_range}")
            else:
                log_debug(f"{camera_info} | {player_info} | {platform_info} | No platforms")

        if self.state == PLAYING:
            for platform in self.platforms:
                platform.update()

            self.player.update()

            ground_y = HEIGHT - GROUND_HEIGHT
            if self.player.rect.bottom >= ground_y - self.camera_y and not self.player.jumping:
                self.player.rect.bottom = ground_y - self.camera_y
                self.player.jump()
                log_debug(f"Player hit ground at y={ground_y}")

            platform_hit = False
            for platform in self.platforms:
                if (not platform.is_broken and
                        self.player.rect.bottom >= platform.rect.top and
                        self.player.rect.bottom <= platform.rect.top + 15 and
                        self.player.rect.right >= platform.rect.left and
                        self.player.rect.left <= platform.rect.right and
                        self.player.velocity_y > 0):

                    self.player.rect.bottom = platform.rect.top

                    if platform.platform_type == BREAKABLE_PLATFORM:
                        platform.break_platform()

                    self.player.jump()

                    # Set target camera position for smooth transition
                    self.target_camera_y = platform.rect.top - HEIGHT

                    print(f"CAMERA: Platform at y={platform.rect.top}")
                    print(f"CAMERA: Setting target_camera_y to {self.target_camera_y}")
                    print(f"CAMERA: Bottom of screen will be at y={self.target_camera_y + HEIGHT}")
                    print(f"CAMERA: Platform will appear at y={platform.rect.top - self.target_camera_y} on screen")

                    platform_hit = True

                    # Update score based on highest platform reached
                    previous_score = self.score
                    new_score = max(self.score, int(abs(platform.rect.top) / 10))
                    if new_score > self.score:
                        self.score = new_score
                        log_debug(f"Score updated: {previous_score} -> {self.score}")
                        log_debug(f"Current difficulty factor: {self.calculate_difficulty_factor():.2f}")

                        # Also log the platform probabilities
                        difficulty = self.calculate_difficulty_factor()
                        normal_chance = 1 - (0.8 * difficulty)
                        moving_chance = 0.4 * difficulty
                        breakable_chance = 0.4 * difficulty
                        log_debug(
                            f"Platform chances: normal={normal_chance:.2f}, moving={moving_chance:.2f}, breakable={breakable_chance:.2f}")

                    log_debug(f"Player hit platform at y={platform.rect.top}, type={platform.platform_type}")

                    if platform.platform_type == MOVING_PLATFORM:
                        self.player.velocity_x += platform.direction * 1

                    highest_platform_y = min(p.rect.y for p in self.platforms) if self.platforms else 0
                    if highest_platform_y > self.camera_y - HEIGHT:
                        self.generate_platforms_ahead()
                        log_debug("Generated new platforms ahead")

                    break

            # Smooth camera movement
            self.camera_y = lerp(self.camera_y, self.target_camera_y, CAMERA_SMOOTHNESS)

            if not platform_hit and self.player.velocity_y > 0:
                if self.frame_count % 30 == 0:
                    log_debug(f"Player falling at y={self.player.rect.y}, bottom={self.player.rect.bottom}")

            platform_count_before = len(self.platforms)
            bottom_threshold = self.camera_y + HEIGHT + 300
            self.platforms = [p for p in self.platforms if p.rect.y < bottom_threshold]
            platforms_removed = platform_count_before - len(self.platforms)

            if platforms_removed > 0:
                self.platforms_removed += platforms_removed
                log_debug(f"Removed {platforms_removed} platforms below y={bottom_threshold}")
                log_debug(f"Total platforms: created={self.platforms_created}, removed={self.platforms_removed}")

            if self.player.rect.top > self.camera_y + HEIGHT + 120:
                log_debug("GAME OVER - player fell out of bounds")
                self.state = ENTER_NAME
                self.player.velocity_y = 0
                self.player.velocity_x = 0

        elif self.state == ENTER_NAME:
            self.name_cursor_blink = (self.name_cursor_blink + 1) % 60

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if self.state == TITLE_SCREEN:
                if event.key == pygame.K_SPACE or event.key == pygame.K_RETURN:
                    log_debug("Game starting from title screen")
                    self.state = PLAYING
                    self.reset_game()

            elif self.state == PLAYING:
                if event.key == pygame.K_ESCAPE:
                    log_debug("Returning to title screen from game")
                    self.state = TITLE_SCREEN
                elif event.key == pygame.K_d:
                    x = self.player.rect.x - 30
                    y = self.player.rect.y + 100
                    self.platforms.append(Platform(x, y, 100))
                    log_debug(f"DEBUG: Added platform directly under player at y={y}")

            elif self.state == GAME_OVER:
                if event.key == pygame.K_r:
                    log_debug("Restarting game after game over")
                    self.reset_game()
                    self.state = PLAYING
                elif event.key == pygame.K_RETURN:
                    log_debug("Returning to title screen after game over")
                    self.state = TITLE_SCREEN

            elif self.state == ENTER_NAME:
                if event.key == pygame.K_RETURN:
                    name = ''.join(self.current_name)
                    if not name:
                        name = "AAA"
                    log_debug(f"Submitting score {self.score} with name {name}")
                    self.add_high_score(name, self.score)
                    self.state = GAME_OVER

                elif event.key == pygame.K_BACKSPACE:
                    if self.current_name:
                        self.current_name.pop()

                elif 97 <= event.key <= 122:  # a-z
                    if len(self.current_name) < 3:
                        self.current_name.append(chr(event.key).upper())

                elif 48 <= event.key <= 57:  # 0-9
                    if len(self.current_name) < 3:
                        self.current_name.append(chr(event.key))

        if self.state == PLAYING:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.player.move_left()
                elif event.key == pygame.K_RIGHT:
                    self.player.move_right()

            elif event.type == pygame.KEYUP:
                if event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                    self.player.stop_horizontal()

    def restart(self):
        log_debug("Manual game restart")
        self.reset_game()
        self.state = PLAYING
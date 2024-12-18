import pygame
from config import *
from player import Player
from shield import Shield
from enemy import Enemy
from animation import Animation
from enemy import EnemyFormation
from level import Level


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.running = True
        self.game_over = False
        self.score = 0
        self.lives = PLAYER_LIVES

        self.font = pygame.font.Font(None, 36)

        self.all_sprites = pygame.sprite.Group()
        self.bullets = pygame.sprite.Group()
        self.enemy_bullets = pygame.sprite.Group()
        self.shield_pieces = pygame.sprite.Group()
        self.enemies = pygame.sprite.Group()
        self.animations = pygame.sprite.Group()

        self.player = Player()
        self.all_sprites.add(self.player)

        self.enemy_formation = EnemyFormation()
        self.enemies = self.enemy_formation.enemies
        self.all_sprites.add(self.enemies)

        self.setup_shields()

        # Create life markers
        self.life_markers = []
        self.create_life_markers()

        self.level = Level()
        self.create_new_level()

        self.is_intro_animation = True

    def create_new_level(self):
        level_params = self.level.next_level()
        self.enemy_formation = EnemyFormation(
            enemies_per_row=level_params['enemies_per_row'],
            enemy_speed=level_params['enemy_speed'],
            shoot_chance_multiplier=level_params['shoot_chance_multiplier']
        )
        self.enemies = self.enemy_formation.enemies
        self.all_sprites.add(self.enemies)

    def update(self):
        if not self.game_over:
            self.animations.update()

            if self.enemy_formation.is_intro_animation:
                self.enemy_formation.update()
                return  # Skip other updates during intro animation

            # Check if there are any enemies left
            if not self.enemies:
                self.create_new_level()
            else:
                self.enemy_formation.update()

            self.player.update()
            self.bullets.update()
            self.enemy_bullets.update()

            # Random enemy shooting - only if enemies exist
            if self.enemies:
                for enemy in self.enemies:
                    if enemy.should_shoot():
                        bullet = enemy.shoot()
                        self.enemy_bullets.add(bullet)
                        self.all_sprites.add(bullet)

            # Check for bullet collisions
            pygame.sprite.groupcollide(self.bullets, self.shield_pieces, True, True)
            pygame.sprite.groupcollide(self.enemy_bullets, self.shield_pieces, True, True)

            # Check for bullet collisions with enemies
            hits = pygame.sprite.groupcollide(self.bullets, self.enemies, True, True)
            for enemy in hits.values():
                self.score += enemy[0].points
                self.create_explosion(enemy[0].rect.center, "enemy_explosion")

            # Check for enemy bullets hitting player
            if not self.player.is_respawning and \
                    pygame.sprite.spritecollide(self.player, self.enemy_bullets, True):
                self.lives -= 1
                self.update_life_markers()
                self.create_explosion(self.player.rect.center, "player_death")
                if self.lives <= 0:
                    self.game_over = True
                    self.player.kill()
                else:
                    self.player.respawn()

            # Check for level completion
            if not self.enemies and not self.game_over:
                self.create_new_level()

    def draw(self):
        self.screen.fill(BLACK)
        self.bullets.draw(self.screen)
        self.enemy_bullets.draw(self.screen)
        self.enemies.draw(self.screen)
        self.shield_pieces.draw(self.screen)
        self.animations.draw(self.screen)
        self.player.draw(self.screen)  # Special draw for player to handle blinking
        self.draw_hud()

        if self.game_over:
            font = pygame.font.Font(None, 74)
            if not self.enemies and self.lives > 0:  # Win condition
                text = font.render('YOU WIN!', True, WHITE)
            else:  # Lose condition
                text = font.render('GAME OVER', True, WHITE)

            score_text = font.render(f'Final Score: {self.score}', True, WHITE)
            text_rect = text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
            score_rect = score_text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + 80))
            self.screen.blit(text, text_rect)
            self.screen.blit(score_text, score_rect)

        pygame.display.flip()

    def create_life_markers(self):
        marker_width = PLAYER_WIDTH // 2  # Make life markers smaller
        spacing = 10  # Space between markers
        total_width = (marker_width + spacing) * self.lives - spacing
        start_x = SCREEN_WIDTH - total_width - 10  # 10 pixel margin from right edge

        for i in range(self.lives):
            marker = Player()
            # Scale down the marker
            scaled_image = pygame.transform.scale(marker.image,
                                                  (marker_width, PLAYER_HEIGHT // 2))
            marker.image = scaled_image
            marker.rect = marker.image.get_rect()
            marker.rect.x = start_x + i * (marker_width + spacing)
            marker.rect.y = 10
            self.life_markers.append(marker)

    def update_life_markers(self):
        # Recreate life markers when lives change
        self.life_markers = []
        self.create_life_markers()

    def draw_hud(self):
        # Draw score
        score_text = self.font.render(f'Score: {self.score}', True, WHITE)
        self.screen.blit(score_text, (10, 10))

        # Draw life markers
        for marker in self.life_markers:
            self.screen.blit(marker.image, marker.rect)

    def setup_shields(self):
        shield_spacing = SCREEN_WIDTH // (NUM_SHIELDS + 1)
        for i in range(NUM_SHIELDS):
            shield_x = shield_spacing * (i + 1) - (SHIELD_WIDTH // 2)
            shield_y = SCREEN_HEIGHT - 150
            shield = Shield(shield_x, shield_y)
            self.shield_pieces.add(shield.pieces)
            self.all_sprites.add(shield.pieces)

    def setup_enemies(self):
        for enemy_type, properties in ENEMY_ROWS.items():
            y = ENEMY_START_Y + (properties['row'] * ENEMY_ROW_HEIGHT)
            enemy = Enemy(ENEMY_START_X, y, enemy_type)
            self.enemies.add(enemy)
            self.all_sprites.add(enemy)

    def create_explosion(self, position, animation_type):
        explosion = Animation(*position, animation_type)
        self.animations.add(explosion)
        self.all_sprites.add(explosion)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if self.enemy_formation.is_intro_animation:
                    continue
                if event.key == pygame.K_SPACE and not self.game_over:
                    bullet = self.player.shoot()
                    self.bullets.add(bullet)
                    self.all_sprites.add(bullet)
                elif event.key == pygame.K_q:  # Test enemy shooting
                    for enemy in self.enemies:
                        bullet = enemy.shoot()
                        self.enemy_bullets.add(bullet)
                        self.all_sprites.add(bullet)

    def draw_lives(self):
        player_mini = Player()
        player_mini_rect = player_mini.image.get_rect()
        for i in range(self.lives):
            x = SCREEN_WIDTH - (i + 1) * (player_mini_rect.width + 10)
            self.screen.blit(player_mini.image, (x, 10))


    def run(self):
        while self.running:
            self.clock.tick(FPS)
            self.handle_events()
            self.update()
            self.draw()
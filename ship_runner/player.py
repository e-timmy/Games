import pygame
import pymunk


class Bullet:
    def __init__(self, x, y, game_offset):
        self.x = x
        self.y = y
        self.speed = 10
        self.radius = 3
        self.game_offset = game_offset  # Store the game's current offset

    def update(self, offset_change):
        self.x += self.speed
        self.game_offset = offset_change  # Update the game offset

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 255, 0), (int(self.x), int(self.y)), self.radius)


class Player:
    def __init__(self, space, x, y):
        self.size = 20
        self.color = (0, 255, 0)
        self.body = pymunk.Body(1, pymunk.moment_for_box(1, (self.size, self.size)))
        self.body.position = (x, y)
        self.shape = pymunk.Poly.create_box(self.body, (self.size, self.size))
        self.shape.elasticity = 0
        self.shape.friction = 0.7
        space.add(self.body, self.shape)
        self.space = space
        self.bullets = []

    def apply_thrust(self):
        thrust_force = -150
        current_velocity = self.body.velocity.y

        if current_velocity > -300:
            self.body.apply_impulse_at_local_point((0, thrust_force), (0, 0))

    def shoot(self, game_offset):
        bullet = Bullet(self.body.position.x + self.size / 2, self.body.position.y, game_offset)
        self.bullets.append(bullet)

    def update(self, game_offset):
        self.body.velocity = (0, self.body.velocity.y)

        if self.body.velocity.y > 400:
            self.body.velocity = (0, 400)
        elif self.body.velocity.y < -400:
            self.body.velocity = (0, -400)

        for bullet in self.bullets[:]:  # Use a slice copy to safely remove items
            bullet.update(game_offset)
            if bullet.x > 800:  # Remove bullets that are off-screen
                self.bullets.remove(bullet)

    def draw(self, screen):
        pos = self.body.position
        pygame.draw.rect(screen, self.color,
                         (pos.x - self.size / 2, pos.y - self.size / 2,
                          self.size, self.size))
        for bullet in self.bullets:
            bullet.draw(screen)

    def reset(self, x, y):
        self.space.remove(self.body, self.shape)
        self.__init__(self.space, x, y)
        self.bullets = []
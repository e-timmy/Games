import pygame
import random


class Log:
    def __init__(self, x, y, width, speed):
        self.x = x
        self.y = y
        self.width = width
        self.height = 30
        self.speed = speed
        self.color = (139, 69, 19)  # Brown color for log

    def update(self):
        self.x += self.speed

    def draw(self, screen):
        # Draw main log body
        pygame.draw.rect(screen, self.color, (self.x, self.y - self.height // 2, self.width, self.height))

        # Add wood grain detail
        darker_brown = (119, 49, 0)
        grain_spacing = 20
        for i in range(0, self.width, grain_spacing):
            pygame.draw.line(screen, darker_brown,
                             (self.x + i, self.y - self.height // 2),
                             (self.x + i, self.y + self.height // 2), 2)


class Lilypad:
    def __init__(self, x, y, speed):
        self.x = x
        self.y = y
        self.width = 30
        self.height = 30
        self.speed = speed
        self.color = (0, 200, 0)  # Green color for lilypad
        self.submerge_timer = 0
        self.max_submerge_time = 180  # 3 seconds at 60 FPS
        self.sinking = False

    def update(self):
        self.x += self.speed
        if self.sinking:
            self.submerge_timer += 1
            if self.submerge_timer >= self.max_submerge_time:
                self.sinking = False
                self.submerge_timer = 0

    def start_sinking(self):
        self.sinking = True
        self.submerge_timer = 0

    def is_submerged(self):
        return self.sinking and self.submerge_timer >= self.max_submerge_time

    def draw(self, screen):
        if not self.is_submerged():
            scale_factor = 1 - (self.submerge_timer / self.max_submerge_time) * 0.5
            scaled_width = int(self.width * scale_factor)
            scaled_height = int(self.height * scale_factor)

            pygame.draw.circle(screen, self.color,
                               (int(self.x), int(self.y)),
                               scaled_width // 2)
            # Add a lighter green circle for detail
            pygame.draw.circle(screen, (100, 255, 100),
                               (int(self.x), int(self.y)),
                               scaled_width // 4)


class River:
    def __init__(self, top, height, num_lanes):
        self.top = top
        self.height = height
        self.num_lanes = num_lanes
        self.color = (64, 164, 223)
        self.logs = []
        self.lilypads = []
        self.lane_height = height // num_lanes
        self.spawn_timer = 0
        self.spawn_rate = 60  # Adjusted for better spacing
        self.directions = [random.choice([-1, 1]) for _ in range(num_lanes)]
        self.target_logs_per_lane = 2
        self.target_lilypads_per_lane = 3
        self.min_log_spacing = 200  # Minimum space between logs
        self.min_lilypad_spacing = 100  # Minimum space between lilypads

    def update(self):
        # Update existing logs
        for log in self.logs[:]:
            log.update()
            if log.x < -log.width - 100 or log.x > 900:
                self.logs.remove(log)

        # Update lilypads
        for lilypad in self.lilypads[:]:
            lilypad.update()
            if lilypad.x < -lilypad.width - 100 or lilypad.x > 900:
                self.lilypads.remove(lilypad)

        # Spawn new objects
        self.spawn_timer += 1
        if self.spawn_timer >= self.spawn_rate:
            self.spawn_timer = 0
            self._try_spawn_logs()
            self._try_spawn_lilypads()

    def _try_spawn_logs(self):
        for lane in range(self.num_lanes):
            lane_y = self.top + (lane + 0.5) * self.lane_height
            current_lane_logs = [log for log in self.logs if abs(log.y - lane_y) < 5]

            if len(current_lane_logs) < self.target_logs_per_lane:
                direction = self.directions[lane]
                speed = direction * random.uniform(1, 2)
                width = random.randint(120, 180)  # Consistent log sizes
                x = -width - 50 if direction > 0 else 850  # Spawn further off screen

                # Check spacing with existing logs in the same lane
                can_spawn = True
                for log in current_lane_logs:
                    # Calculate the space between logs based on direction
                    if direction > 0:
                        space = log.x - (x + width)
                    else:
                        space = x - (log.x + log.width)

                    if space < self.min_log_spacing:
                        can_spawn = False
                        break

                if can_spawn:
                    self.logs.append(Log(x, lane_y, width, speed))

    def _try_spawn_lilypads(self):
        for lane in range(self.num_lanes):
            lane_y = self.top + (lane + 0.5) * self.lane_height
            current_lane_lilypads = [pad for pad in self.lilypads if abs(pad.y - lane_y) < 5]

            if len(current_lane_lilypads) < self.target_lilypads_per_lane:
                direction = self.directions[lane]
                speed = direction * random.uniform(0.5, 1.5)
                x = -30 - 50 if direction > 0 else 850  # Spawn further off screen

                # Check spacing with existing lilypads in the same lane
                can_spawn = True
                for pad in current_lane_lilypads:
                    # Calculate the space between lilypads based on direction
                    if direction > 0:
                        space = pad.x - (x + 30)
                    else:
                        space = x - (pad.x + 30)

                    if space < self.min_lilypad_spacing:
                        can_spawn = False
                        break

                if can_spawn:
                    self.lilypads.append(Lilypad(x, lane_y, speed))

    def draw(self, screen):
        # Draw river background
        pygame.draw.rect(screen, self.color, (0, self.top, 800, self.height))

        # Add water effect (simple lines)
        wave_color = (84, 184, 243)
        wave_spacing = 40
        for y in range(int(self.top), int(self.top + self.height), wave_spacing):
            offset = (pygame.time.get_ticks() / 500) % wave_spacing
            for x in range(-wave_spacing + int(offset), 800, wave_spacing):
                pygame.draw.line(screen, wave_color,
                                 (x, y),
                                 (x + wave_spacing // 2, y + wave_spacing // 4),
                                 2)

        # Draw all logs
        for log in self.logs:
            log.draw(screen)

        # Draw all lilypads
        for lilypad in self.lilypads:
            lilypad.draw(screen)

    def get_log_at(self, x, y):
        # Add a small buffer to the collision detection to make it easier to get on logs
        buffer = 5
        for log in self.logs:
            if (log.x - buffer < x < log.x + log.width + buffer and
                    log.y - log.height // 2 - buffer < y < log.y + log.height // 2 + buffer):
                return log
        return None

    def get_lilypad_at(self, x, y):
        # Add a small buffer to the collision detection to make it easier to get on lilypads
        buffer = 5
        for lilypad in self.lilypads:
            if (lilypad.x - lilypad.width // 2 - buffer < x < lilypad.x + lilypad.width // 2 + buffer and
                    lilypad.y - lilypad.height // 2 - buffer < y < lilypad.y + lilypad.height // 2 + buffer):
                return lilypad
        return None

    def get_lane_center(self, lane_index):
        return self.top + (lane_index + 0.5) * self.lane_height
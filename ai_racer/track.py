import pygame

import pygame


class Track:
    def __init__(self, width, height):
        self.inset = 50
        self.track_width = 120
        self.start_line_width = 10
        self.corner_radius = 40

        # Outer track boundary
        self.outer_rect = pygame.Rect(
            self.inset,
            self.inset,
            width - 2 * self.inset,
            height - 2 * self.inset
        )

        # Inner track boundary
        self.inner_rect = pygame.Rect(
            self.inset + self.track_width,
            self.inset + self.track_width,
            width - 2 * (self.inset + self.track_width),
            height - 2 * (self.inset + self.track_width)
        )

        # Start line
        self.start_line = pygame.Rect(
            width // 2 - 5,
            height - self.inset - self.track_width,
            10,
            self.track_width
        )

        # Grid box dimensions (slightly larger than cars)
        box_width = 40
        box_height = 25
        grid_spacing_y = 40  # Vertical spacing between positions

        # Base positions (left of start line)
        base_x = self.start_line.x - 60  # Move cars left of the start line
        base_y = self.start_line.centery  # Vertical center of the track

        # Grid setup for dynamic number of positions
        self.grid_boxes = []
        self.start_positions = []

        # Grid configuration
        box_width = 40
        box_height = 25
        grid_spacing_x = 70  # Increased from 50 - more space between rows
        grid_spacing_y = 40  # Increased from 30 - more lateral space

        # Base position (behind start line)
        base_x = self.start_line.x - 100
        base_y = self.start_line.centery

        # Calculate positions for up to 10 cars (5 rows of 2)
        max_positions = 10
        positions_per_row = 2

        for i in range(max_positions):
            row = i // positions_per_row
            col = i % positions_per_row

            # Calculate position with stagger
            pos_x = base_x - (row * grid_spacing_x)  # Each row starts further back
            pos_y = base_y + ((col - 0.5) * grid_spacing_y)  # Offset from center

            # Add stagger to second position in each row
            if col == 1:
                pos_x -= grid_spacing_x / 2

            # Create grid box
            box_lines = [
                # Back vertical line
                [(pos_x + box_width, pos_y - box_height / 2),
                 (pos_x + box_width, pos_y + box_height / 2)],
                # Top horizontal line
                [(pos_x, pos_y - box_height / 2),
                 (pos_x + box_width, pos_y - box_height / 2)],
                # Bottom horizontal line
                [(pos_x, pos_y + box_height / 2),
                 (pos_x + box_width, pos_y + box_height / 2)]
            ]

            self.grid_boxes.append({'lines': box_lines})
            self.start_positions.append((pos_x + box_width / 2, pos_y))

        # Add checkpoints crossing the track perpendicularly
        self.checkpoints = [
            # Right side checkpoint (vertical track section, horizontal checkpoint)
            pygame.Rect(
                self.outer_rect.right - self.track_width - 5,
                self.outer_rect.centery - 5,
                self.track_width + 10,
                10
            ),
            # Top checkpoint (vertical checkpoint crossing horizontal track section)
            pygame.Rect(
                self.outer_rect.centerx - 5,  # centered on track
                self.outer_rect.top,
                10,
                self.track_width + 10
            ),
            # Left side checkpoint (vertical track section, horizontal checkpoint)
            pygame.Rect(
                self.outer_rect.left,
                self.outer_rect.centery - 5,
                self.track_width + 10,
                10
            )
        ]

    def draw(self, screen):
        # Draw outer boundary
        pygame.draw.rect(screen, (0, 0, 0),
                         pygame.Rect(self.inset - 2, self.inset - 2,
                                     self.outer_rect.width + 4, self.outer_rect.height + 4),
                         border_radius=self.corner_radius + 2)
        pygame.draw.rect(screen, (100, 100, 100), self.outer_rect, border_radius=self.corner_radius)

        # Draw inner boundary
        pygame.draw.rect(screen, (0, 0, 0),
                         pygame.Rect(self.inner_rect.left - 2, self.inner_rect.top - 2,
                                     self.inner_rect.width + 4, self.inner_rect.height + 4),
                         border_radius=self.corner_radius - 10)
        pygame.draw.rect(screen, (0, 100, 0), self.inner_rect, border_radius=self.corner_radius - 10)

        # Draw start line
        pygame.draw.rect(screen, (255, 255, 255), self.start_line)

        # Draw grid boxes
        for box in self.grid_boxes:
            for line in box['lines']:
                pygame.draw.line(screen, (200, 200, 200), line[0], line[1], 2)

        # Optionally draw checkpoints for debugging
        for checkpoint in self.checkpoints:
            pygame.draw.rect(screen, (255, 0, 0), checkpoint, 1)

    def is_on_track(self, pos, radius):
        # Check if the car is within the outer rounded rectangle
        if not self._is_within_rounded_rect(pos, radius, self.outer_rect, self.corner_radius):
            return False

        # Check if the car is outside the inner rounded rectangle
        if self._is_within_rounded_rect(pos, radius, self.inner_rect, self.corner_radius - 10):
            return False

        return True

    def _is_within_rounded_rect(self, pos, radius, rect, corner_radius):
        # Check if point is inside the rounded rectangle
        x, y = pos

        # Quick check: if outside the outer rectangle, return False
        expanded_rect = rect.inflate(radius * 2, radius * 2)
        if not expanded_rect.collidepoint(x, y):
            return False

        # Check corners
        corners = [
            (rect.left + corner_radius, rect.top + corner_radius),
            (rect.right - corner_radius, rect.top + corner_radius),
            (rect.right - corner_radius, rect.bottom - corner_radius),
            (rect.left + corner_radius, rect.bottom - corner_radius)
        ]

        for cx, cy in corners:
            dx, dy = x - cx, y - cy
            if dx ** 2 + dy ** 2 <= (corner_radius + radius) ** 2:
                if dx ** 2 + dy ** 2 >= (corner_radius - radius) ** 2:
                    return True

        # Check straight edges
        if (rect.left + corner_radius <= x <= rect.right - corner_radius and
                rect.top <= y <= rect.bottom):
            return True
        if (rect.top + corner_radius <= y <= rect.bottom - corner_radius and
                rect.left <= x <= rect.right):
            return True

        return False

    def get_reflection_vector(self, pos, radius):
        if not self.is_on_track(pos, radius):
            # Check outer boundaries
            if pos[0] - radius < self.outer_rect.left:
                return (1, 0)
            if pos[0] + radius > self.outer_rect.right:
                return (-1, 0)
            if pos[1] - radius < self.outer_rect.top:
                return (0, 1)
            if pos[1] + radius > self.outer_rect.bottom:
                return (0, -1)

            # Check inner boundaries
            if pos[0] + radius > self.inner_rect.left and pos[0] - radius < self.inner_rect.left + 5:
                return (-1, 0)
            if pos[0] - radius < self.inner_rect.right and pos[0] + radius > self.inner_rect.right - 5:
                return (1, 0)
            if pos[1] + radius > self.inner_rect.top and pos[1] - radius < self.inner_rect.top + 5:
                return (0, -1)
            if pos[1] - radius < self.inner_rect.bottom and pos[1] + radius > self.inner_rect.bottom - 5:
                return (0, 1)

        return (0, 0)
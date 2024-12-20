class Camera:
    def __init__(self, width, height):
        self.x = 0
        self.y = 0
        self.width = width
        self.height = height
        self.transitioning = False
        self.transition_speed = 10
        self.transition_target = 0

    def apply(self, x, y):
        screen_x = x - self.x
        # Only check for lower bound, remove upper bound check
        if screen_x < -self.width * 2:
            return (-1000, -1000)
        return (screen_x, y)

    def get_visible_bounds(self):
        return (self.x, self.x + self.width)

    def start_transition(self):
        self.transitioning = True
        self.transition_target = self.x + self.width

    def update(self):
        if self.transitioning:
            self.x += self.transition_speed
            if self.x >= self.transition_target:
                self.x = self.transition_target
                self.transitioning = False
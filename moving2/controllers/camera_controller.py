class CameraController:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.x = 0
        self.y = 0
        self.transitioning = False
        self.transition_target = 0
        self.transition_speed = 10

    def apply(self, x, y):
        screen_x = x - self.x
        if screen_x < -self.width * 2:
            return (-1000, -1000)
        return (screen_x, y)

    def start_transition(self):
        self.transitioning = True
        self.transition_target = self.x + self.width

    def update(self):
        if self.transitioning:
            self.x += self.transition_speed
            if self.x >= self.transition_target:
                self.x = self.transition_target
                self.transitioning = False
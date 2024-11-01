class GameObject:
    def __init__(self, space, pos):
        self.space = space
        self.body = None
        self.shape = None
        self.create_physics_body(pos)

    def create_physics_body(self, pos):
        raise NotImplementedError

    def update(self):
        pass

    def draw(self, render_engine, camera):
        pass

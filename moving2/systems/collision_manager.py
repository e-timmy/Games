class CollisionManager:
    def __init__(self, space, player_controller, level_controller):
        self.space = space
        self.player_controller = player_controller
        self.level_controller = level_controller
        self.setup_collisions()

    def setup_collisions(self):
        def collect_item(arbiter, space, data):
            item_shape = arbiter.shapes[1]
            item = self.level_controller.get_item_from_shape(item_shape)
            if item and not item.collected:
                self.player_controller.powerup_system.add_powerup(item.powerup_type)
                item.collected = True
            return True

        def bullet_collect_item(arbiter, space, data):
            item_shape = arbiter.shapes[1]
            item = self.level_controller.get_item_from_shape(item_shape)
            if item and not item.collected:
                self.player_controller.powerup_system.add_powerup(item.powerup_type)
                item.collected = True
            return True

        # Player collecting item
        handler = self.space.add_collision_handler(1, 2)  # Player: 1, Item: 2
        handler.begin = collect_item

        # Bullet collecting item
        bullet_handler = self.space.add_collision_handler(4, 2)  # Bullet: 4, Item: 2
        bullet_handler.begin = bullet_collect_item
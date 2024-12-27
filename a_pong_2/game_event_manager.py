class GameEventManager:
    def __init__(self):
        self.listeners = []

    def add_listener(self, listener):
        self.listeners.append(listener)

    def notify(self, event_type, data=None):
        for listener in self.listeners:
            listener.on_event(event_type, data)


class GameEventListener:
    def on_event(self, event_type, data):
        pass
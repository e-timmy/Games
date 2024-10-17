import threading
import queue
import time
from pathfinder import Pathfinder


class AsyncPathfinder:
    def __init__(self, environment, time_limit=0.5):
        self.environment = environment
        self.pathfinder = Pathfinder(environment)
        self.time_limit = time_limit
        self.request_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.thread = threading.Thread(target=self._process_requests, daemon=True)
        self.thread.start()

    def find_path(self, start, goal):
        self.request_queue.put((start, goal))

    def get_result(self):
        if not self.result_queue.empty():
            return self.result_queue.get()
        return None

    def _process_requests(self):
        while True:
            start, goal = self.request_queue.get()
            path = self._time_constrained_pathfinding(start, goal)
            self.result_queue.put(path)

    def _time_constrained_pathfinding(self, start, goal):
        start_time = time.time()
        path = None
        try:
            path = self.pathfinder.find_path(start, goal)
        except Exception as e:
            print(f"Pathfinding error: {e}")

        elapsed_time = time.time() - start_time
        if elapsed_time > self.time_limit:
            print(f"Pathfinding took too long: {elapsed_time:.2f} seconds")

        return path
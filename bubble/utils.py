import math

def calculate_distance(x1, y1, x2, y2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def is_collision(obj1, obj2):
    """Check if two circular objects are colliding."""
    distance = calculate_distance(obj1.x, obj1.y, obj2.x, obj2.y)
    return distance < (obj1.size + obj2.size)

def scale_to_screen(value, max_value, screen_size):
    """Scale a value to fit the screen size."""
    return (value / max_value) * screen_size
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import AmbientLight, DirectionalLight, Vec4, Vec3, Texture, TextureStage
from panda3d.core import LineSegs, NodePath, CardMaker, GeomNode, Geom, GeomVertexFormat, GeomVertexData
from panda3d.core import GeomTriangles, GeomVertexWriter
from math import sin, cos, radians, pi, atan2, degrees
from direct.showbase.ShowBaseGlobal import globalClock


class SimpleDrivingGame(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.disableMouse()

        self.cam_distance = 15
        self.cam_height = 6
        self.cam_offset_z = 1.5

        # Track parameters
        self.track_width = 8  # Width of the track
        self.track_radius_x = 25  # Horizontal radius
        self.track_radius_y = 15  # Vertical radius

        self.create_environment()
        self.create_track()
        self.create_car()
        self.setup_lighting()
        self.setup_controls()

        # Car movement variables
        self.car_speed = 0
        self.car_heading = 0
        self.max_speed = 20
        self.acceleration = 10
        self.deceleration = 5
        self.turn_rate = 60
        self.ground_size = 50

        # Place car at start line
        self.position_car_at_start()

        self.taskMgr.add(self.update_car, "UpdateCar")
        self.taskMgr.add(self.update_camera, "UpdateCamera")

    def create_environment(self):
        """Create a ground plane with grass texture."""
        from panda3d.core import CardMaker

        # Set a skyblue background
        self.setBackgroundColor(0.5, 0.8, 0.9)  # Light blue sky

        # Create main ground plane
        cm = CardMaker('ground')
        cm.setFrame(-50, 50, -50, 50)  # Large square plane

        self.ground = self.render.attachNewNode(cm.generate())
        self.ground.setPos(0, 0, 0)
        self.ground.setP(-90)  # Flat on the ground

        # Create and apply a grass texture to the ground
        try:
            # Create a green texture for grass
            tex = Texture("ground_tex")
            tex.setup2dTexture(2, 2, Texture.T_unsigned_byte, Texture.F_rgb)
            tex.setRamImage(b"\x00\x80\x00\x20\xA0\x20\x20\xA0\x20\x00\x80\x00")  # Green grass pattern

            ts = TextureStage('ts')
            ts.setMode(TextureStage.MModulate)

            # Apply texture to ground and enable repeat
            self.ground.setTexture(ts, tex)
            self.ground.setTexScale(ts, 50, 50)  # Repeat the texture

            # Set ground color to be grassy
            self.ground.setColor(0.3, 0.7, 0.3)
        except:
            # Fallback if texture creation fails
            self.ground.setColor(0.2, 0.6, 0.2)  # Green base color

    def create_track(self):
        """Create an oval racetrack with barriers."""
        # Track node to hold all track components
        self.track = self.render.attachNewNode("track")

        # Create the actual oval track surface
        self.create_oval_track()

        # Create barriers
        self.create_continuous_barriers()

        # Create start/finish line
        self.create_start_line()

    def create_oval_track(self):
        """Create a proper oval track surface using GeomNode."""
        format = GeomVertexFormat.getV3c4()
        vdata = GeomVertexData('track', format, Geom.UHStatic)

        vertex = GeomVertexWriter(vdata, 'vertex')
        color = GeomVertexWriter(vdata, 'color')

        # Number of segments to create smooth oval
        segments = 100

        # Generate points for the oval track
        inner_points = []
        outer_points = []

        for i in range(segments):
            angle = 2 * pi * i / segments

            # Inner ellipse points
            inner_x = (self.track_radius_x - self.track_width / 2) * cos(angle)
            inner_y = (self.track_radius_y - self.track_width / 2) * sin(angle)
            inner_points.append((inner_x, inner_y))

            # Outer ellipse points
            outer_x = (self.track_radius_x + self.track_width / 2) * cos(angle)
            outer_y = (self.track_radius_y + self.track_width / 2) * sin(angle)
            outer_points.append((outer_x, outer_y))

            # Add vertices for inner and outer points
            vertex.addData3(inner_x, inner_y, 0.01)
            color.addData4f(0.3, 0.3, 0.3, 1)  # Asphalt gray

            vertex.addData3(outer_x, outer_y, 0.01)
            color.addData4f(0.3, 0.3, 0.3, 1)  # Asphalt gray

        # Create triangles for the track
        track_tris = GeomTriangles(Geom.UHStatic)

        for i in range(segments):
            next_i = (i + 1) % segments

            # Create two triangles for each segment
            # Triangle 1: inner_i, outer_i, inner_next
            track_tris.addVertices(i * 2, i * 2 + 1, next_i * 2)
            track_tris.closePrimitive()

            # Triangle 2: outer_i, outer_next, inner_next
            track_tris.addVertices(i * 2 + 1, next_i * 2 + 1, next_i * 2)
            track_tris.closePrimitive()

        # Create the geom and add it to a node
        track_geom = Geom(vdata)
        track_geom.addPrimitive(track_tris)

        track_node = GeomNode('oval_track')
        track_node.addGeom(track_geom)

        track_np = NodePath(track_node)
        track_np.reparentTo(self.track)

    def create_continuous_barriers(self):
        """Create continuous barriers around the track."""
        # Parameters for inner and outer barriers
        barrier_height = 1.0
        segments = 100  # Same number as track for alignment

        # Lists to store points for barriers
        inner_barrier_points = []
        outer_barrier_points = []

        # Generate points for both barriers
        for i in range(segments + 1):  # +1 to close the loop
            angle = 2 * pi * i / segments

            # Inner barrier points
            inner_x = (self.track_radius_x - self.track_width / 2) * cos(angle)
            inner_y = (self.track_radius_y - self.track_width / 2) * sin(angle)
            inner_barrier_points.append((inner_x, inner_y, 0))

            # Outer barrier points
            outer_x = (self.track_radius_x + self.track_width / 2) * cos(angle)
            outer_y = (self.track_radius_y + self.track_width / 2) * sin(angle)
            outer_barrier_points.append((outer_x, outer_y, 0))

        # Create inner barrier as a continuous wall
        inner_barrier = self.render.attachNewNode("inner_barrier")
        self.create_barrier_wall(inner_barrier_points, barrier_height, (1, 0, 0, 1), inner_barrier)
        inner_barrier.reparentTo(self.track)

        # Create outer barrier as a continuous wall
        outer_barrier = self.render.attachNewNode("outer_barrier")
        self.create_barrier_wall(outer_barrier_points, barrier_height, (0, 0, 1, 1), outer_barrier)
        outer_barrier.reparentTo(self.track)

    def create_barrier_wall(self, points, height, color, parent_node):
        """Create a continuous barrier wall from a list of points."""
        if len(points) < 2:
            return

        format = GeomVertexFormat.getV3c4()
        vdata = GeomVertexData('barrier_wall', format, Geom.UHStatic)

        vertex = GeomVertexWriter(vdata, 'vertex')
        color_writer = GeomVertexWriter(vdata, 'color')

        # Add vertices for bottom and top of wall
        for point in points:
            # Bottom vertex
            vertex.addData3(point[0], point[1], 0)
            color_writer.addData4f(*color)

            # Top vertex
            vertex.addData3(point[0], point[1], height)
            color_writer.addData4f(*color)

        # Create triangles for the wall
        wall_tris = GeomTriangles(Geom.UHStatic)

        for i in range(len(points) - 1):
            # Bottom left, bottom right, top left
            wall_tris.addVertices(i * 2, (i + 1) * 2, i * 2 + 1)
            wall_tris.closePrimitive()

            # Top left, bottom right, top right
            wall_tris.addVertices(i * 2 + 1, (i + 1) * 2, (i + 1) * 2 + 1)
            wall_tris.closePrimitive()

        # Create the geom and add it to a node
        wall_geom = Geom(vdata)
        wall_geom.addPrimitive(wall_tris)

        wall_node = GeomNode('barrier_wall')
        wall_node.addGeom(wall_geom)

        NodePath(wall_node).reparentTo(parent_node)

    def create_start_line(self):
        """Create a checkered start/finish line."""
        # Position at bottom of track (270 degrees)
        angle = 3 * pi / 2

        # Calculate start/finish line endpoints
        inner_x = (self.track_radius_x - self.track_width / 2) * cos(angle)
        inner_y = (self.track_radius_y - self.track_width / 2) * sin(angle)

        outer_x = (self.track_radius_x + self.track_width / 2) * cos(angle)
        outer_y = (self.track_radius_y + self.track_width / 2) * sin(angle)

        # Vector along the start line
        dx = outer_x - inner_x
        dy = outer_y - inner_y
        length = (dx ** 2 + dy ** 2) ** 0.5

        # Create white start line
        start_line_cm = CardMaker("start_line")
        start_line_cm.setFrame(0, length, -0.5, 0.5)
        start_line = self.render.attachNewNode(start_line_cm.generate())

        # Position and rotate
        heading = degrees(atan2(dy, dx))
        start_line.setH(heading)
        start_line.setP(-90)
        start_line.setPos(inner_x, inner_y, 0.02)
        start_line.setColor(1, 1, 1)

        # Create checkerboard pattern
        checker_count = 8
        checker_size = length / checker_count

        for i in range(checker_count):
            if i % 2 == 0:
                continue

            checker_cm = CardMaker(f"checker_{i}")
            checker_cm.setFrame(i * checker_size, (i + 1) * checker_size, -0.5, 0.5)

            checker = self.render.attachNewNode(checker_cm.generate())
            checker.setH(heading)
            checker.setP(-90)
            checker.setPos(inner_x, inner_y, 0.03)
            checker.setColor(0, 0, 0)
            checker.reparentTo(self.track)

        start_line.reparentTo(self.track)

    def position_car_at_start(self):
        """Position the car at the start line, facing along the track."""
        # Position car at the bottom center of the track
        angle = 3 * pi / 2  # Bottom of the oval

        # Center of track width at start point
        start_x = self.track_radius_x * cos(angle)
        start_y = self.track_radius_y * sin(angle) + 2  # Slightly in front of start line

        self.car.setPos(start_x, start_y, 0.5)

        # Set car heading to face up along the track (90 degrees in Panda3D)
        self.car.setH(90)
        self.car_heading = 90

    def create_car(self):
        """Create a proper 3D rectangular car with correct alignment."""
        self.car = self.render.attachNewNode("car")

        # Create a container node for the car body to control its center
        body_container = self.car.attachNewNode("body_container")
        body_container.setPos(0, 0, 0.5)  # Position at car's center, half height above ground

        # Main body - adjust position to center it within the container
        body = self.loader.loadModel("models/box")
        body.setScale(1, 2, 0.5)
        # Models/box seems to have its origin at the corner, so offset to center it
        body.setPos(-0.5, -1.0, -0.25)  # Negative half of each dimension
        body.setColor(1, 0, 0)
        body.reparentTo(body_container)

        # Front marker - similarly adjusted
        front = self.loader.loadModel("models/box")
        front.setScale(0.8, 0.4, 0.2)
        front.setPos(-0.4, 1.0, -0.1)  # Position relative to body, adjusted for origin
        front.setColor(1, 1, 0)
        front.reparentTo(body_container)

        # Create each wheel with its own container node
        wheel_dimensions = [
            # Left Front, Right Front, Left Rear, Right Rear
            (-0.5, 0.8, 0),  # Front left
            (0.5, 0.8, 0),  # Front right
            (-0.5, -0.8, 0),  # Back left
            (0.5, -0.8, 0)  # Back right
        ]

        for i, pos in enumerate(wheel_dimensions):
            wheel_container = self.car.attachNewNode(f"wheel_container_{i}")
            wheel_container.setPos(pos[0], pos[1], 0.3)  # Position at correct wheel location

            wheel = self.loader.loadModel("models/box")
            wheel.setScale(0.3, 0.3, 0.3)
            wheel.setPos(-0.15, -0.15, -0.15)  # Center the wheel within its container
            wheel.setColor(0.2, 0.2, 0.2)
            wheel.reparentTo(wheel_container)

    def setup_lighting(self):
        """Set up basic lighting for the scene."""
        # Ambient light
        ambient_light = AmbientLight("ambient")
        ambient_light.setColor(Vec4(0.6, 0.6, 0.6, 0.6))
        ambient_np = self.render.attachNewNode(ambient_light)
        self.render.setLight(ambient_np)

        # Directional light (sun)
        sun_light = DirectionalLight("sun")
        sun_light.setColor(Vec4(0.8, 0.8, 0.7, 1))
        sun_np = self.render.attachNewNode(sun_light)
        sun_np.setHpr(45, -45, 0)  # Angle the light
        self.render.setLight(sun_np)

    def setup_controls(self):
        """Set up controls for both car and camera."""
        # Car controls (arrow keys)
        self.accept("arrow_up", self.set_car_key, ["up", True])
        self.accept("arrow_up-up", self.set_car_key, ["up", False])
        self.accept("arrow_down", self.set_car_key, ["down", True])
        self.accept("arrow_down-up", self.set_car_key, ["down", False])
        self.accept("arrow_left", self.set_car_key, ["left", True])
        self.accept("arrow_left-up", self.set_car_key, ["left", False])
        self.accept("arrow_right", self.set_car_key, ["right", True])
        self.accept("arrow_right-up", self.set_car_key, ["right", False])

        # Reset key (if car gets stuck)
        self.accept("r", self.position_car_at_start)

        # Initialize key dictionaries
        self.keys = {"up": False, "down": False, "left": False, "right": False}

    def set_car_key(self, key, value):
        """Track the state of a car control key."""
        self.keys[key] = value

    def update_car(self, task):
        """Update car position and rotation with simpler physics."""
        dt = globalClock.getDt()

        # Apply acceleration based on keys
        if self.keys["up"]:
            self.car_speed += self.acceleration * dt
        elif self.keys["down"]:
            self.car_speed -= self.acceleration * dt * 0.6  # Slower reverse
        else:
            # Apply friction/drag when no keys are pressed
            if abs(self.car_speed) < self.deceleration * dt:
                self.car_speed = 0
            elif self.car_speed > 0:
                self.car_speed -= self.deceleration * dt
            else:
                self.car_speed += self.deceleration * dt

        # Clamp speed to maximum
        if self.car_speed > self.max_speed:
            self.car_speed = self.max_speed
        elif self.car_speed < -self.max_speed * 0.5:
            self.car_speed = -self.max_speed * 0.5

        # Handle turning with no restrictions
        turn_amount = self.turn_rate * dt
        if self.keys["left"]:
            # When moving forward, left = increase heading
            turn_direction = 1 if self.car_speed >= 0 else -1
            self.car_heading += turn_amount * turn_direction
        elif self.keys["right"]:
            # When moving forward, right = decrease heading
            turn_direction = -1 if self.car_speed >= 0 else 1
            self.car_heading += turn_amount * turn_direction

        # Update car's visual rotation
        self.car.setH(self.car_heading)

        # Calculate movement vector based on car's heading
        heading_radians = radians(self.car_heading)
        dx = -self.car_speed * sin(heading_radians) * dt
        dy = self.car_speed * cos(heading_radians) * dt

        # Calculate new position
        new_x = self.car.getX() + dx
        new_y = self.car.getY() + dy

        # Simple boundary check
        boundary = self.ground_size - 2
        if abs(new_x) > boundary or abs(new_y) > boundary:
            self.car_speed *= -0.5  # Bounce with reduced speed
            new_x = max(-boundary, min(boundary, new_x))
            new_y = max(-boundary, min(boundary, new_y))

        # Update car position
        self.car.setPos(new_x, new_y, 0.5)

        return Task.cont

    def update_camera(self, task):
        """Update camera position to follow car from directly behind."""
        # Calculate position behind the car based on car's heading
        heading_radians = radians(self.car_heading)

        # Position camera behind and slightly offset to the side
        cam_x = self.car.getX() + self.cam_distance * sin(heading_radians)
        cam_y = self.car.getY() - self.cam_distance * cos(heading_radians)
        cam_z = self.car.getZ() + self.cam_height

        # Set camera position
        self.camera.setPos(cam_x, cam_y, cam_z)

        # Look at a point slightly above the car for better view
        look_at_point = self.car.getPos() + Vec3(0, 0, self.cam_offset_z)
        self.camera.lookAt(look_at_point)

        return Task.cont


# Run the game
game = SimpleDrivingGame()
game.run()
"""
The PyBullet World
"""
import pybullet as p
import pybullet_data

class AntWorld:
    def __init__(self, render=True, num_ants=5):
        self.render = render
        self.num_ants = num_ants
        if self.render:
            width = 1920 
            height = 1080
            p.connect(p.GUI, options=f"--width={width} --height={height}")

            # Dont display any debug visualizers
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            
            # Zoom out to view the whole colony of ants
            p.resetDebugVisualizerCamera(
                cameraDistance=15.0, 
                cameraYaw=45, 
                cameraPitch=-35, 
                cameraTargetPosition=[0, 0, 0]
            )
        else:
            p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        # Load Models
        self.plane_id = p.loadURDF("plane.urdf")
        self.init_nest()
        self.ant_ids = []
        for i in range(self.num_ants):
            start_pos = [0, 0, 0.5]
            ant_id = p.loadURDF("r2d2.urdf", start_pos)
            self.ant_ids.append(ant_id)
        self.disable_inter_ant_collision()
        self.food_ids = []
    # Disable inter-ant collision for this simulation
    def disable_inter_ant_collision(self):
        for i in range(len(self.ant_ids)):
            for j in range(i + 1, len(self.ant_ids)):
                p.setCollisionFilterPair(self.ant_ids[i], self.ant_ids[j], -1, -1, enableCollision=0)

    def init_nest(self):
        """Creates a visual disk at the origin to represent the nest."""
        # Create a thin cylinder (disk)
        nest_visual = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=0.5,
            length=0.01,
            rgbaColor=[0, 0.5, 1, 0.6]
        )
        
        # Spawn the nest at 0,0
        self.nest_id = p.createMultiBody(
            baseMass=0, # Static
            baseVisualShapeIndex=nest_visual,
            basePosition=[0, 0, 0.01]
        )

    def init_food(self, num_cubes):
        """Creates the visual cubes for food sources"""
        # Clear old food if any
        for f_id in self.food_ids:
            p.removeBody(f_id)
        self.food_ids = []

        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX, 
            halfExtents=[0.2, 0.2, 0.2], 
            rgbaColor=[1, 0, 0, 1]
        )
        for _ in range(num_cubes):
            f_body = p.createMultiBody(
                baseMass=0, # Static object
                baseVisualShapeIndex=visual_shape_id, 
                basePosition=[0, 0, -10] # Hide initially under the floor
            )
            self.food_ids.append(f_body)

    def update_visuals(self, positions, headings, all_food_pos, food_collected_flags):
        """
        positions: array of shape (num_ants, 2)
        headings: array of shape (num_ants,)
        all_food_pos: list of food coordinates, provided by the Gym environment
        """
        # Update ant model position and orientation kinematically
        for i in range(self.num_ants):
            x, y = positions[i]
            yaw = headings[i]
            quaternion = p.getQuaternionFromEuler([0, 0, yaw])
            p.resetBasePositionAndOrientation(self.ant_ids[i], [x, y, 0.5], quaternion)
        
        for i, pos in enumerate(all_food_pos):
            if i < len(self.food_ids):
                if not food_collected_flags[i]:
                    p.resetBasePositionAndOrientation(
                        self.food_ids[i], [pos[0], pos[1], 0.2], [0,0,0,1]
                    )

    def collect_food(self, food_idx):
        """Changes the color of the food to indicate it has been picked up."""
        if food_idx < len(self.food_ids):
            p.changeVisualShape(
                self.food_ids[food_idx], 
                -1, 
                rgbaColor=[0, 1, 0, 0.5]   # Transparent Green
            )
            pos, orn = p.getBasePositionAndOrientation(self.food_ids[food_idx])
            p.resetBasePositionAndOrientation(self.food_ids[food_idx], [pos[0], pos[1], 0.5], orn)

    def reset(self):
        """Resets all ants to the center."""
        for ant_id in self.ant_ids:
            p.resetBasePositionAndOrientation(ant_id, [0, 0, 0.5], [0, 0, 0, 1])

    def close(self):
        if p.isConnected():
            p.disconnect()
"""
The PyBullet World
"""
import pybullet as p
import pybullet_data

class AntWorld:
    def __init__(self, render=True):
        width = 1280
        height = 720

        self.client = p.connect(p.GUI if render else p.DIRECT, options=f"--width={width} --height={height}")
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        #Ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        #Model representing the agent
        self.ant_id = p.loadURDF("r2d2.urdf", [0, 0, 0.5])

    def init_food(self, num_cubes):
        self.food_ids = []
        for _ in range(num_cubes):
            # Spawn cubes at a neutral height (0.25)
            f_id = p.loadURDF("cube.urdf", [0, 0, 0.25], globalScaling=0.5)
            p.changeVisualShape(f_id, -1, rgbaColor=[1, 0, 0, 1]) # Red
            self.food_ids.append(f_id)

    def collect_food(self, index):
        # Changes the color of the specific cube to signal it was found by the agent
        if 0 <= index < len(self.food_ids):
            # Change to Green [0, 1, 0, 1]
            p.changeVisualShape(self.food_ids[index], -1, rgbaColor=[0, 1, 0, 1])

    def update_visuals(self, x, y, yaw, all_food_pos=None):
        pos = [x, y, 0.5]
        orn = p.getQuaternionFromEuler([0, 0, yaw])
        p.resetBasePositionAndOrientation(self.ant_id, pos, orn)
        if all_food_pos is not None:
            for f_id, f_pos in zip(self.food_ids, all_food_pos):
                _, current_orn = p.getBasePositionAndOrientation(f_id)
                p.resetBasePositionAndOrientation(f_id, [f_pos[0], f_pos[1], 0.25], current_orn)
        p.stepSimulation()

    def reset(self):
        """Resets the visual model to the starting point (0,0)"""
        p.resetBasePositionAndOrientation(self.ant_id, [0, 0, 0.5], [0, 0, 0, 1])

    def close(self):
        p.disconnect()
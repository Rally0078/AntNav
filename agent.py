import numpy as np

class AntAgent:
    def __init__(self):
        self.internal_vector = np.array([0.0, 0.0], dtype=np.float32)
        self.found_food = False

    def update_dead_reckoning(self, step_dist, angle_rad):
        # Update internal belief of position (Path Integration)
        self.internal_vector[0] += step_dist * np.cos(angle_rad)
        self.internal_vector[1] += step_dist * np.sin(angle_rad)

    def get_home_heading(self):
        # Calculate vector pointing back to (0,0) from current internal belief
        return np.arctan2(-self.internal_vector[1], -self.internal_vector[0])

    def reset(self):
        self.internal_vector = np.array([0.0, 0.0], dtype=np.float32)
        self.found_food = False
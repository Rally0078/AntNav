from environment import DesertAntEnv
import numpy as np
import time

class DesertAntNavigator:
    SEARCH_MOVE = "SEARCH_MOVE"
    SEARCH_STOP = "SEARCH_STOP"
    HOMING      = "HOMING"
    def __init__(self):
        self.state = self.SEARCH_MOVE
        self.step_count = 0
        # The very first heading is fully random (360 degrees)
        self.current_heading = np.random.uniform(-np.pi, np.pi)
        self.first_step_taken = False

    def get_action(self, env_logic, obs):
        # Only the first step has a truly random direction
        # Subsequent steps do not deviate from the previous step within +/- 90 degrees from the previous direction
        if env_logic.found_food:
            self.state = self.HOMING

        if self.state == self.SEARCH_MOVE:
            angle = self.current_heading
            speed = 0.8
            self.step_count += 1
            if self.step_count > 20:
                self.state = self.SEARCH_STOP
                self.step_count = 0
                self.first_step_taken = True # First leg is over, now onwards the random walk is correlated to the past step

        elif self.state == self.SEARCH_STOP:
            angle = self.current_heading
            speed = 0.0
            self.step_count += 1
            if self.step_count > 10:
                turn_variation = np.random.uniform(-np.pi/2, np.pi/2)
                
                self.current_heading = self.current_heading + turn_variation
                self.current_heading = (self.current_heading + np.pi) % (2 * np.pi) - np.pi
                
                self.state = self.SEARCH_MOVE
                self.step_count = 0

        elif self.state == self.HOMING:
            angle = env_logic.get_home_heading()
            speed = 0.8
            
        return np.array([angle, speed], dtype=np.float32)
    
def run_ant__navigator():
    env = DesertAntEnv(render_mode="human", num_cubes=10)
    obs, _ = env.reset()
    controller = DesertAntNavigator()
    done = False
    print(f"Initial State: {controller.state}")

    while not done:
        action = controller.get_action(env.agent_logic, obs)
        obs, reward, done, truncated, info = env.step(action)
        
        time.sleep(0.01)

    print("Success! Ant has reached the nest!")
    start_wait = time.time()
    while time.time() - start_wait < 5:
        env.step(np.array([0, 0]))
        time.sleep(0.1)
        
    env.close()

if __name__=="__main__":
    run_ant__navigator()
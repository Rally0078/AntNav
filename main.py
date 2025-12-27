from environment import DesertAntEnv
import numpy as np
import time

class DesertAntNavigator:
    SEARCH_MOVE = "SEARCH_MOVE"
    SEARCH_STOP = "SEARCH_STOP"
    HOMING      = "HOMING"

    def __init__(self, ant_id):
        self.ant_id = ant_id
        self.state = self.SEARCH_MOVE
        self.step_count = 0
        self.current_heading = np.random.uniform(-np.pi, np.pi)
        self.trips_completed = 0

    def reset_for_new_trip(self):
        """Resets the internal state of the ant for a new foraging run."""
        self.state = self.SEARCH_MOVE
        self.step_count = 0
        self.current_heading = np.random.uniform(-np.pi, np.pi)

    def get_action(self, specific_ant_logic, obs):
    # Agent action is based on its current state
    # obs is not used, 
    # could be used in the case of an RL agent that has actual senses to stimuli from the environment
        if specific_ant_logic.found_food:
            self.state = self.HOMING

        if self.state == self.SEARCH_MOVE:
            angle = self.current_heading
            speed = 0.8
            self.step_count += 1
            if self.step_count > 20:
                self.state = self.SEARCH_STOP
                self.step_count = 0

        elif self.state == self.SEARCH_STOP:
            angle = self.current_heading
            speed = 0.0
            self.step_count += 1
            if self.step_count > 10:
                turn_variation = np.random.uniform(-np.pi/2, np.pi/2)
                self.current_heading = (self.current_heading + turn_variation + np.pi) % (2 * np.pi) - np.pi
                self.state = self.SEARCH_MOVE
                self.step_count = 0

        elif self.state == self.HOMING:
            angle = specific_ant_logic.get_home_heading()
            speed = 0.8
            
        return np.array([angle, speed], dtype=np.float32)

def run_multi_ant_simulation(num_epochs=1, num_ants=5, num_cubes=100, trips_per_ant=3):
    env = DesertAntEnv(render_mode="human", num_cubes=100, num_ants=num_ants)
    
    for epoch in range(num_epochs):
        print(f"\n--- Starting Epoch {epoch + 1} ---")
        obs, _ = env.reset()
        controllers = [DesertAntNavigator(ant_id=i) for i in range(num_ants)]
        
        # Keep track of who is still working
        active_controllers = list(controllers)
        
        # When there is any active ant
        while active_controllers:
            actions = []
            for ctrl in controllers:
                ant_logic = env.agent_logics[ctrl.ant_id]
                ant_obs = obs[ctrl.ant_id]
                
                if ctrl.trips_completed >= trips_per_ant:
                    actions.append(np.array([0.0, 0.0], dtype=np.float32))
                else:
                    # Get agent's current action based on the handwritten ant logic
                    actions.append(ctrl.get_action(ant_logic, ant_obs))

            # Get the gym environment step outputs
            obs, rewards, dones, truncated, infos = env.step(np.array(actions))
            # Check status for each ant
            for ctrl in active_controllers:
                if dones[ctrl.ant_id]: 
                    ctrl.trips_completed += 1
                    print(f"Ant {ctrl.ant_id} completed trip {ctrl.trips_completed}/{trips_per_ant}")

                    if ctrl.trips_completed < trips_per_ant:
                        # Reset this specific ant to start again
                        env.reset_agent(ctrl.ant_id)
                        ctrl.reset_for_new_trip()
                    else:
                        print(f"Ant {ctrl.ant_id} has finished all required trips")
                        active_controllers.remove(ctrl)

            env.render()
            time.sleep(0.01)

    print("\nSimulation Complete! All ants finished their cycles.")
    import pybullet as p
    while p.getConnectionInfo()['isConnected']:
        p.stepSimulation()
        time.sleep(0.1)

if __name__=="__main__":
    run_multi_ant_simulation(1,8,100,3)
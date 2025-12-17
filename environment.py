import gymnasium as gym
from gymnasium import spaces
import numpy as np
from world import AntWorld
from agent import AntAgent

class DesertAntEnv(gym.Env):
    def __init__(self, render_mode=None, num_cubes=5):
        super().__init__()
        self.world = AntWorld(render=(render_mode == "human"))
        self.agent_logic = AntAgent()
        # Food sources
        self.num_cubes = num_cubes
        self.world.init_food(self.num_cubes)
        
        # Define action
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]), 
            high=np.array([1.0, 1.0]), 
            dtype=np.float32
        )
        
        # Define observation space: (agent_x, agent_y, relativefood_x, relativefood_y)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(4,), 
            dtype=np.float32
        )
        
        self.true_pos = np.array([0.0, 0.0])
        self.all_food_pos = []
        self.target_food_pos = np.array([0.0, 0.0])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.world.reset()
        self.agent_logic.reset()
        self.true_pos = np.array([0.0, 0.0])
        
        # Randomize positions for all food cubes
        self.all_food_pos = []
        for _ in range(self.num_cubes):
            angle = self.np_random.uniform(0, 2 * np.pi)
            dist = self.np_random.uniform(5, 12) # Range from 5 to 12 units away
            pos = np.array([dist * np.cos(angle), dist * np.sin(angle)])
            self.all_food_pos.append(pos)
        
        # Sync the PyBullet world with the new food locations
        self.world.update_visuals(0, 0, 0, all_food_pos=self.all_food_pos)
        
        return self._get_obs(), {}

    def _get_obs(self):
        # Calculate which cube is currently the closest
        distances = [np.linalg.norm(self.true_pos - fp) for fp in self.all_food_pos]
        nearest_idx = np.argmin(distances)
        self.target_food_pos = self.all_food_pos[nearest_idx]
        
        # Vector from agent to the nearest cube
        rel_food = self.target_food_pos - self.true_pos
        
        # Return the 4-element observation
        return np.array([
            self.agent_logic.internal_vector[0],
            self.agent_logic.internal_vector[1],
            rel_food[0],
            rel_food[1]
        ], dtype=np.float32)

    def step(self, action):
        turn_angle = action[0]
        speed = action[1] * 0.1 
        
        self.true_pos[0] += speed * np.cos(turn_angle)
        self.true_pos[1] += speed * np.sin(turn_angle)
        # Update dead reckoning path
        self.agent_logic.update_dead_reckoning(speed, turn_angle)
        
        # Update the PyBullet visuals
        self.world.update_visuals(
            self.true_pos[0], 
            self.true_pos[1], 
            turn_angle, 
            all_food_pos=self.all_food_pos
        )
        
        # Check distances to food
        dist_to_food = np.linalg.norm(self.true_pos - self.target_food_pos)
        dist_to_nest = np.linalg.norm(self.true_pos)
        distances = [np.linalg.norm(self.true_pos - fp) for fp in self.all_food_pos]
        nearest_idx = np.argmin(distances)
        self.target_food_pos = self.all_food_pos[nearest_idx]
        dist_to_food = distances[nearest_idx]
        dist_to_nest = np.linalg.norm(self.true_pos)
        # Check if food is found
        if not self.agent_logic.found_food and dist_to_food < 1.0:
            self.world.collect_food(nearest_idx)
            self.agent_logic.found_food = True

        # Reward Logic (Negative distance to current target, for RL agent training, not the manually coded agent)
        if self.agent_logic.found_food:
            reward = -dist_to_nest
        else:
            reward = -dist_to_food
        
        # Terminate if food found AND reached home
        terminated = bool(self.agent_logic.found_food and dist_to_nest < 0.5)
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, {}

    def close(self):
        self.world.close()
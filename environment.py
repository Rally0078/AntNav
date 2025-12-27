"""
The Gymnasium API-compliant environment for the agent. 

RL may not be the appropriate choice here, because the food cubes are randomly spawned with no patterns. 
The RL agent is practically useless if it learns a random policy based on the randomly placed food. 
If there is an olfactory sense for the ant which could be interpreted as a "smell intensity", 
an RL agent could possibly learn that moving in a straight line to the food would maximize the smell 
intensity and lead it to the food.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from world import AntWorld
from agent import AntAgent

class DesertAntEnv(gym.Env):
    def __init__(self, render_mode=None, num_cubes=5, num_ants=5):
        super().__init__()
        self.num_ants = num_ants
        self.world = AntWorld(render=(render_mode == "human"), num_ants=num_ants)
        self.food_taken = [False] * num_cubes
        self.agent_logics = [AntAgent() for _ in range(num_ants)]
        
        self.num_cubes = num_cubes
        self.world.init_food(self.num_cubes)
        
        # Action space: Batch of actions [num_ants, 2]
        self.action_space = spaces.Box(
            low=np.tile([-np.pi, 0.0], (num_ants, 1)), 
            high=np.tile([np.pi, 1.0], (num_ants, 1)), 
            dtype=np.float32
        )
        
        # Observation space: Batch of observations [num_ants, 4]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_ants, 4), dtype=np.float32
        )
        
        self.true_positions = np.zeros((num_ants, 2))
        self.all_food_pos = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.world.reset()
        
        self.all_food_pos = []
        # Randomly initialize food positions
        for _ in range(self.num_cubes):
            angle = self.np_random.uniform(0, 2 * np.pi)
            dist = self.np_random.uniform(8, 40)
            self.all_food_pos.append(np.array([dist * np.cos(angle), dist * np.sin(angle)]))
        
        for i in range(self.num_ants):
            self.reset_agent(i)
        self.food_taken = [False] * self.num_cubes
        return self._get_obs(), {}

    def reset_agent(self, ant_id):
        """Resets a single ant back to the nest for a new trip."""
        self.true_positions[ant_id] = np.array([0.0, 0.0])
        self.agent_logics[ant_id].reset()

    def _get_obs(self):
        obs_batch = []
        for i in range(self.num_ants):
            distances = [np.linalg.norm(self.true_positions[i] - fp) for fp in self.all_food_pos]
            nearest_idx = np.argmin(distances)
            rel_food = self.all_food_pos[nearest_idx] - self.true_positions[i]
            
            obs_batch.append([
                self.agent_logics[i].internal_vector[0],
                self.agent_logics[i].internal_vector[1],
                rel_food[0],
                rel_food[1]
            ])
        return np.array(obs_batch, dtype=np.float32)

    def step(self, actions):
        """
        actions shape: (num_ants, 2) -> [angle, speed]
        """
        rewards = []
        dones = []
        
        self.last_headings = actions[:, 0] 
        
        for i in range(self.num_ants):
            turn_angle = actions[i][0]
            speed = actions[i][1] * 0.1 
            
            # Update Physical Position in the environment
            self.true_positions[i][0] += speed * np.cos(turn_angle)
            self.true_positions[i][1] += speed * np.sin(turn_angle)
            
            # Path integration for dead reckoning to return to nest
            self.agent_logics[i].update_dead_reckoning(speed, turn_angle)
            
            if not self.agent_logics[i].found_food: #If ant has not found food yet
                for f_idx, f_pos in enumerate(self.all_food_pos):
                    if not self.food_taken[f_idx]:
                        dist_to_this_cube = np.linalg.norm(self.true_positions[i] - f_pos)
                        # Collection logic, collect the food cube if ant is close to it
                        if dist_to_this_cube < 0.8: # Collection radius
                            self.agent_logics[i].found_food = True
                            self.food_taken[f_idx] = True
                            self.world.collect_food(f_idx)
                            print(f"Ant {i} picked up Food Cube {f_idx}")
                            break
            # Internal environment variable used for reward calculation, only needed for RL
            dist_to_nest = np.linalg.norm(self.true_positions[i])
            
            # Reward section, not needed if RL is not used
            if not self.agent_logics[i].found_food:
                # Find the nearest available cube for the reward signal
                active_food = [fp for idx, fp in enumerate(self.all_food_pos) if not self.food_taken[idx]]
                if active_food:
                    dist_to_target = min([np.linalg.norm(self.true_positions[i] - fp) for fp in active_food])
                else:
                    dist_to_target = 10.0 # No food left in the world
                reward = -dist_to_target
            else:
                reward = -dist_to_nest

            # Check if ant has reached the nest
            reached_home = bool(self.agent_logics[i].found_food and dist_to_nest < 0.5)
            dones.append(reached_home)
            rewards.append(reward)

        # Update visuals
        self.world.update_visuals(self.true_positions, self.last_headings, self.all_food_pos, self.food_taken)
        
        return self._get_obs(), rewards, dones, False, {}

    def close(self):
        self.world.close()
    def render(self):
        pass
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import os
from typing import Tuple, Dict, Any


# Import your custom environment
import bipedal_walker


# Create the environment
env_id: str = 'BipedalWalkerEnvCustom-v0'
env: gym.Env = gym.make(env_id)


episodes: int = 5

for episode in range(episodes):
    observation: np.ndarray 
    observation, _ = env.reset()
    
    done: bool = False
    total_reward: float = 0

    while not done:
        action: np.ndarray[float] = env.action_space.sample()

        observation, reward, terminated, _, _ = env.step(env.action_space.sample())
        
        print(f"Action: {action}")
        print(f"Observation: {observation}")
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")

        total_reward += reward

        if terminated:
            done = True

    print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")
env.close()


'''
 The env returns a state:
         state = [
            self.hull.angle,  # Normal angles up to 0.5 here, but sure more is possible.
            2.0 * self.hull.angularVelocity / FPS,
            0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS,  # Normalized to get -1..1 range
            0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS,
            self.joints[0].angle,
            # This will give 1.1 on high up, but it's still OK (and there should be spikes on hitting the ground, that's normal too)
            self.joints[0].speed / SPEED_HIP,
            self.joints[1].angle + 1.0,
            self.joints[1].speed / SPEED_KNEE,
            1.0 if self.legs[1].ground_contact else 0.0,
            self.joints[2].angle,
            self.joints[2].speed / SPEED_HIP,
            self.joints[3].angle + 1.0,
            self.joints[3].speed / SPEED_KNEE,
            1.0 if self.legs[3].ground_contact else 0.0,
        ]
    
    return np.array(state, dtype=np.float32), reward, terminated, False, {}

'''
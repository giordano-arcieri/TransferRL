import gym
import numpy as np
from bipedal_walker import BipedalWalkerEnv

VERY_SLIPPERY = 0.1
SLIPPERY = 0.2
NORMAL = 0.4
HIGH_FRICTION = 4

FLAT = 0
SLIGHTLY_BUMPY = 1
BUMPY = 2
VERY_BUMPY = 5

SLIPPERY_SETTINGS = [VERY_SLIPPERY, SLIPPERY, NORMAL, HIGH_FRICTION]
BUMPY_SETTINGS = [FLAT, SLIGHTLY_BUMPY, BUMPY, VERY_BUMPY]


for S in SLIPPERY_SETTINGS:
    for B in BUMPY_SETTINGS:
        print(f"Friction: {S}, Bumpiness: {B}")
       
        # Create the BipedalWalker environment with render_mode
        env = gym.make('BipedalWalkerEnvCustom-v0', render_mode='human', hardcore=False, bumpiness=B, friction=S)


        # Reset the environment to start
        observation = env.reset()

        # Run the simulation for a fixed number of steps
        for _ in range(100):
            # Render the environment (optional, since render_mode='human' handles rendering)
            # env.render()  # You can remove this line if not needed

            # Define zero actions for all four joints
            action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

            # Take a step using the zero action
            observation, reward, terminated, truncated, info = env.step(action)

            # Check if the episode is terminated or truncated
            if terminated or truncated:
                print("Episode finished. Resetting environment.")
                observation = env.reset()

        # Close the environment when done
        env.close()



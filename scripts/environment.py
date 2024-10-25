import gym
import numpy as np
from bipedal_walker import BipedalWalker

# Create the BipedalWalker environment with render_mode
env = BipedalWalker(render_mode='human')


# Reset the environment to start
observation = env.reset()

# Run the simulation for a fixed number of steps
for _ in range(1000):
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

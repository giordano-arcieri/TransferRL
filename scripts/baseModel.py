import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import os

# Import your custom environment
import bipedal_walker


# Create log directory
log_dir = "./ppo_bipedalwalker_logs/"
os.makedirs(log_dir, exist_ok=True)

all_envs = gym.envs.registry.keys()
print("Available Environments:")
for env in all_envs:
    if(env == 'BipedalWalkerEnvCustom-v0'): 
        print(env)


# Create the environment
env_id = 'BipedalWalkerEnvCustom-v0'
env = gym.make(env_id)

# Instantiate the agent 
model = PPO(
    'MlpPolicy',
    env,
    learning_rate=3e-4,  # Default value, consider experimenting
    gamma=0.99,          # Discount factor
    n_steps=2048,        # Number of steps to run for each environment per update
    batch_size=64,       # Minibatch size
    n_epochs=10,         # Number of epochs when optimizing the surrogate loss
    clip_range=0.2,      # Clipping parameter
    ent_coef=0.0,        # Entropy coefficient for exploration
    verbose=1,
    tensorboard_log=log_dir,
)

# Create an evaluation callback
eval_env = gym.make(env_id)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=log_dir,
    log_path=log_dir,
    eval_freq=10000,
    deterministic=True,
    render=False,
)

# Train the agent
total_timesteps = 1_000_000
model.learn(total_timesteps=total_timesteps, callback=eval_callback)

# Save the trained model
model_path = os.path.join(log_dir, "ppo_bipedalwalker")
model.save(model_path)

# Evaluate the trained agent
episodes = 5
env = gym.make(env_id)

for episode in range(episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            done = True
    print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")
env.close()

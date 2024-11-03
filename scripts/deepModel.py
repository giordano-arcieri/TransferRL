import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import gymnasium as gym
import bipedal_walker

# Hyperparameters
GAMMA = 0.99
LR = 0.0005
BATCH_SIZE = 64
REPLAY_SIZE = 1_000_000
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.2

class BipedalWalkerAgent:
    def __init__(self, state_size=24, action_size=4):
        self.network = nn.Sequential(
            nn.Linear(state_size, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, action_size),
            nn.Tanh()
        )
        self.target_network = nn.Sequential(
            nn.Linear(state_size, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, action_size),
            nn.Tanh()
        )
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.network.parameters(), lr=LR)
        self.replay_buffer = deque(maxlen=REPLAY_SIZE)
        self.epsilon = 1.0

    def select_action(self, state) -> np.ndarray:
        r = random.random()
        if r < self.epsilon:
            action = np.random.uniform(-1, 1, 4)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = self.network(state_tensor).squeeze().numpy()

        return action

    def store_experience(self, state, action, reward: float, next_state, done):
        # Convert elements to np.ndarray before storing them
        state = np.array(state, dtype=np.float32)
        action = np.array(action, dtype=np.float32)
        reward = np.array(reward, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        done = np.array(done, dtype=np.float32)
        
        # Store experience as a tuple of ndarrays
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return  # Ensure we have enough experiences to sample from

        # Sample a mini-batch from the replay buffer
        mini_batch = random.sample(self.replay_buffer, BATCH_SIZE)
        
        # Use np.stack to create batch arrays directly
        states_tensor = torch.FloatTensor(np.stack([experience[0] for experience in mini_batch]))  # (batch_size, state_dim)
        actions_tensor = torch.FloatTensor(np.stack([experience[1] for experience in mini_batch]))  # (batch_size, action_dim)
        rewards_tensor = torch.FloatTensor(np.stack([experience[2] for experience in mini_batch]))  # (batch_size,)
        next_states_tensor = torch.FloatTensor(np.stack([experience[3] for experience in mini_batch]))  # (batch_size, state_dim)
        dones_tensor = torch.FloatTensor(np.stack([experience[4] for experience in mini_batch]))  # (batch_size,)

        # Get Q-values for the selected actions in the current states
        q_values = self.network(states_tensor)
        q_values = q_values.gather(1, actions_tensor.argmax(dim=1).unsqueeze(1)).squeeze()

        # Calculate target Q-values for next states using the target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states_tensor).max(1)[0]
            # Use `done` to zero-out the future reward where the episode ends
            target_q_values = rewards_tensor + (GAMMA * next_q_values * (1 - dones_tensor))

        # Compute the loss as the difference between current and target Q-values
        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)

def train(agent: BipedalWalkerAgent):
    env_id = 'BipedalWalker-v3'
    env = gym.make(env_id)

    episodes = 1000
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        i = 0
        while not done and i < 5000 and total_reward > -50: 
            i += 1
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            if(reward == -100):
                reward = -1
            agent.store_experience(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        if episode % 3 == 0:
            agent.update_target_network()
        print(f"Episode {episode}: Total Reward = {total_reward}")



def test(agent: BipedalWalkerAgent):
    env_id = 'BipedalWalkerEnvCustom-v0'
    env = gym.make(env_id, render_mode='human')

    while True:
        state, _ = env.reset()
        done = False
        total_reward = 0
        i = 0
        while not done and i < 10000:
            i += 1
            action = agent.select_action(state)
            print(i, action)
            state, reward, done, _, _ = env.step(action)
            total_reward += reward

        print(f"Total reward: {total_reward}")
    env.close()

def main():
    agent = BipedalWalkerAgent()
    train(agent)
    test(agent)
if __name__ == "__main__":
    main()

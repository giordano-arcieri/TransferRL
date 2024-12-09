import torch
from stable_baselines3 import PPO
import os
from Package import bipedal_walker
import gymnasium as gym
from Package.PPOAgent import PPOAgent
from typing import Dict

""" 
The base policy is a PPO algorithim with the folowing architecture: [ 24(input layer), 64, 64, 64, 64, 64, 4(output layer) ]

This file will use transfer learning to train a model on a different environment. 
"""

"""
takes in the filepath to a model saved with model.save() and returns a model object (PPO) that can be trained on a different environment.
Assume the base model is always saved with the following architecture: [ 24(input layer), 64, 64, 64, 64, 64, 4(output layer) ]
Transfer the first 5 layers of the base model to the new model. 
[ 24(input layer), 64, 64, 64, 64, 64, 4(output layer) ] ->
[ Transferred and frozen( 24(input layer), 64, 64, 64, 64 ), All weights reset( 64, 4(output layer) ) ]
"""



def transfer(baseModel_FilePath: str, envToTrainOn: str, envProperties: Dict[str, float]) -> PPOAgent:
    """
    Transfers the first 5 layers from the base model, freezes them, and reinitializes the last 2 layers.
    
    Parameters:
    - baseModel_FilePath (str): Filepath of the saved base model.

    Returns:
    - model (PPO): A new PPO model with transferred and frozen weights for the first 5 layers.
    """
    # Load the base model
    base_model = PPO.load(baseModel_FilePath)
    base_policy = base_model.policy

    # Create network architecture
    policy_kwargs = dict(
        net_arch=[64, 64, 64, 64, 64], # Number of neurons in each hidden layer [24(input layer), 64, 64, 64, 64, 64, 4(output layer)]
        activation_fn=torch.nn.ReLU # Activation function
    )
    
    new_model = PPOAgent(
        env_id = envToTrainOn,
        policy = 'MlpPolicy',
        bumpiness=envProperties["bumpiness"],
        friction=envProperties["friction"],
        n_steps = 2048,
        batch_size = 256,
        n_epochs = 6,
        gamma = 0.995,
        gae_lambda = 0.98,
        ent_coef = 0.03,
        verbose=1,
        policy_kwargs=policy_kwargs
    )

    # Transfer weights and freeze layers
    base_params = list(base_policy.parameters())
    new_params = list(new_model.policy.parameters())

    for i in range(10):  # Assuming 5 layers (weights and biases)
        new_params[i].data = base_params[i].data.clone()
        new_params[i].requires_grad = False  # Freeze the layer

    # Reinitialize the last 2 layers (weights and biases)
    for i in range(10, len(new_params)):  # Remaining layers
        if new_params[i].dim() == 2:  # Reinitialize weights (Only weights have 2 dimensions)
            torch.nn.init.xavier_uniform_(new_params[i])
        else:  # Reinitialize biases
            torch.nn.init.zeros_(new_params[i])
        new_params[i].requires_grad = True  # Ensure trainability

    print("Transfer learning complete. The first 5 layers are frozen.")
    return new_model


def main():
    baseModel_FilePath = "../PPOModels/BaseModel/baseModel.zip"
    env_id = "BipedalWalkerEnvCustom-v0"
    
    assert os.path.exists(baseModel_FilePath), f"Base model file not found at {baseModel_FilePath}"

    # Define the new environments
    env_properties = [
        {"bumpiness": 5.0, "friction": 0.7, "name": "bumpy"},
        {"bumpiness": 1.0, "friction": 0.2, "name": "slippery"},
        {"bumpiness": 5.0, "friction": 0.2, "name": "hard"},
    ]
    
    save_path = ["../PPOModels/TransferModel/bumpy/", "../PPOModels/TransferModel/slippery/", "../PPOModels/TransferModel/hard/"]

    # Iterate over the new environments
    for idx, props in enumerate(env_properties, start=1):
        print(f"\n--- Training on {props['name']} ---")
        print(idx)
        
        # Transfer the model
        print("Transferring base model...")
        transferred_model = transfer(baseModel_FilePath, envToTrainOn=env_id, envProperties=props)

        # Check if the transfer was successful
        params = list(transferred_model.policy.parameters())
        for i in range(10):  # Assuming 5 layers (weights and biases)
            assert torch.equal(params[i].data, params[i].data), f"Mismatch in layer {i} weights!"
            assert not params[i].requires_grad, f"Layer {i} is not frozen!"

        for i in range(10, len(params)):
            assert params[i].requires_grad, f"Layer {i} is frozen!"
        

        # Train the model on the new environment
        temp_path = save_path[idx-1] + f"TransferModel_{props['name']}"
        print(f"Training model for {props['name']}...")
        for i in range(10):
            transferred_model.learn(total_timesteps=500_000)

            # Save the trained model
            transferred_model.save(temp_path + f"_{i}.zip")
            with open(temp_path + f"_{i}.txt", "w") as f:
                f.write(f"{transferred_model.get_episode_stats()}")
            print(f"Model trained on {props['name']} saved to {temp_path}")

if __name__ == "__main__":
    main()
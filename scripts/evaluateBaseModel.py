import sys
print(sys.path)

import gymnasium as gym
from Package import bipedal_walker
import os
from Package.ModelTester import ModelTester
from typing import Dict

def main():
    
    # Define
    testing_env_id = "BipedalWalkerEnvCustom-v0"
    testing_sims = ['slippery', 'bumpy', 'hard']
    model_path = '../PPOModels/BaseModel/baseModel.zip'
    
    
    
    # Check if the model exists
    assert os.path.exists(model_path), f"Model not found at {model_path}"
    
    # Check if the environment name is valid
    assert testing_env_id in gym.envs.registry, f"Invalid environment name: {testing_env_id}"

    for testing_sim in testing_sims:
        # Evaluate the model
        model_tester = ModelTester(model_path, testing_env_id, testing_sim)
        stats: Dict[str, float] = model_tester.run()
        
        # Save the statistics
        print(f"############Stats on \"{testing_sim}\": {stats}")

if __name__ == '__main__':
    main()
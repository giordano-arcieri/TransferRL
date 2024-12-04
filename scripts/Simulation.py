'''

'''

import gymnasium as gym

class Simulation:
    def __init__(self, env_id: str = 'BipedalWalkerEnvCustom-v0', render: bool = False, bumpiness: float = 1.0, friction: float = 0.7):
        render_mode = 'human' if render else None
        self.env = gym.make(env_id, render_mode=render_mode, bumpiness=bumpiness, friction=friction)

    def runSimulation(self, model, eval_times: int = 1, maxTime: int = 1000):
        for i in range(eval_times):
            obs, _ = self.env.reset()

            done = False
            time = 0
            print("Running simulation: ", i + 1)
            while not done and time < maxTime:
                if(time % 100 == 0):
                    print("Time: ", time, "/", maxTime)
                time += 1
                action, _states = model.predict(obs)
                obs, _reward, done, _T, _info = self.env.step(action)

                self.env.render()

    def __del__(self):
        self.env.close()
        self.env = None


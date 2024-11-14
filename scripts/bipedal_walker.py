from gymnasium.envs.box2d.bipedal_walker import *
from gymnasium.envs.registration import register
from typing import Optional

'''
    BipedalWalkerEnv is a custom environment that inherits from the BipedalWalker environm. 
    And allows you to set the bumpiness and friction of the terrain in the environment.
    It also changes the reward system to penalize the use of only one leg. (To incentivize human-like walking)

    This can be done using wrappers and that is the recommended way to do it. 
    However, I cant really find a way to edit the friction as it is a global variable. 
    Creating a custom environment allows me more control and achives the same result.
    
    I wanted to be able to tweak bumpiness and friction so I created a class that allows you to set them 
    in the __init__.

    The __init__ method takes in the following parameters:
        - render_mode: str = None - The mode to render the environment. Default is None.
        - hardcore: bool = False - Whether to use the hardcore version of the environment. Default is False.
        - bumpiness: int = 1 - The bumpiness of the terrain. Default is 1. [Must be between 0 and 10]
        - friction: int = 0.7 - The friction of the terrain. Default is 0.7. [Must be between 0.1 and 5]
    
    The constructor performs the following actions:
        - checks if the bumpiness and friction are within the valid range and raises a ValueError if not.
        - sets the bumpiness and friction 
        - calls the super.__init__(render_mode, hardcore) to initialize the BipedalWalker environment.
    
    
    This class overrides the _generate_terrain method of the BipedalWalker class to set the bumpiness and friction of the terrain.
    Additionally I wanted to be able to edit the rewards since with the current system the robot will learn ro walk only using one leg
    so this class also override the step method to change the reward, adding a penalty for using only one leg.
    
'''

MIN_BUMPINESS = 0
MAX_BUMPINESS = 10

MIN_FRICTION = 0.1
MAX_FRICTION = 5



class BipedalWalkerEnv(BipedalWalker):
    """
    Create a BipedalWalker class of wich the bumpiness and friction of the enviorment can be tweaked.
    
        Parameters: 

        render_mode: str = None:
            '''
                The mode to render the environment. Default is None.
                This is the same as the render_mode parameter in the BipedalWalker class.
            '''

        hardcore: bool = False
            '''
                Whether to use the hardcore version of the environment. Default is False.
                This is the same as the hardcore parameter in the BipedalWalker class.
            '''

        bumpiness: int = 1:
            '''
                The bumpiness of the terrain. Default is 1. 
                [Must be between the MIN_BUMPINESS(0) and MAX_BUMPINESS(10) global variables]
            '''

        friction: int = 0.7
            '''
                The friction of the terrain. Default is 0.7.
                [Must be between the MIN_FRICTION(0.1) and MAX_FRICTION(5) global variables]
            '''

    Postcondition:
        The BipedalWalkerEnv class is initialized with the specified bumpiness and friction. and you can call each method of the BipedalWalker class.
        Only generate_terrian method is overriden to set the bumpiness and friction of the terrain.
    """
    def __init__(self, render_mode: Optional[str] = None, hardcore: bool = False, bumpiness: int = 1, friction: int = 0.7):
        
        print(f"Initiating class with: {{ bumpiness: {bumpiness}, friction: {friction} }} ")

        # Initialize the bumpiness and friction
        self.bumpiness = np.clip(bumpiness, MIN_BUMPINESS, MAX_BUMPINESS)
        self.friction = np.clip(friction, MIN_FRICTION, MAX_FRICTION)

        global FRICTION
        FRICTION = friction

        # Some variables
        self.leftLegContact: int = 0
        self.rightLegContact: int = 0

        # Call the super constructor        
        super().__init__(render_mode=render_mode, hardcore=hardcore)
    

    '''
        The step method is overriden to change the reward system.
    '''
    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)        
        return observation, reward, terminated, truncated, info

    '''
        The _generate_terrain method is overriden to set the bumpiness and friction of the terrain.
    '''
    def _generate_terrain(self, hardcore):
            GRASS, STUMP, STAIRS, PIT, _STATES_ = range(5)
            state = GRASS
            velocity = 0.0
            y = TERRAIN_HEIGHT
            counter = TERRAIN_STARTPAD
            oneshot = False
            self.terrain = []
            self.terrain_x = []
            self.terrain_y = []

            stair_steps, stair_width, stair_height = 0, 0, 0
            original_y = 0
            for i in range(TERRAIN_LENGTH):
                x = i * TERRAIN_STEP
                self.terrain_x.append(x)

                if state == GRASS and not oneshot:
                    velocity = 0.8 * velocity + 0.01 * np.sign(TERRAIN_HEIGHT - y) * self.bumpiness  # Increase bump frequency
                    if i > TERRAIN_STARTPAD:
                        velocity += self.np_random.uniform(-self.bumpiness, self.bumpiness) / SCALE  # Adjust bumpiness for height variation
                    y += velocity

                elif state == PIT and oneshot:
                    counter = self.np_random.integers(3, 5)
                    poly = [
                        (x, y),
                        (x + TERRAIN_STEP, y),
                        (x + TERRAIN_STEP, y - 4 * TERRAIN_STEP),
                        (x, y - 4 * TERRAIN_STEP),
                    ]
                    self.fd_polygon.shape.vertices = poly
                    t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                    t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                    self.terrain.append(t)

                    self.fd_polygon.shape.vertices = [
                        (p[0] + TERRAIN_STEP * counter, p[1]) for p in poly
                    ]
                    t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                    t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                    self.terrain.append(t)
                    counter += 2
                    original_y = y

                elif state == PIT and not oneshot:
                    y = original_y
                    if counter > 1:
                        y -= 4 * TERRAIN_STEP

                elif state == STUMP and oneshot:
                    counter = self.np_random.integers(1, 3)
                    poly = [
                        (x, y),
                        (x + counter * TERRAIN_STEP, y),
                        (x + counter * TERRAIN_STEP, y + counter * TERRAIN_STEP),
                        (x, y + counter * TERRAIN_STEP),
                    ]
                    self.fd_polygon.shape.vertices = poly
                    t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                    t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                    self.terrain.append(t)

                elif state == STAIRS and oneshot:
                    stair_height = +1 if self.np_random.random() > 0.5 else -1
                    stair_width = self.np_random.integers(4, 5)
                    stair_steps = self.np_random.integers(3, 5)
                    original_y = y
                    for s in range(stair_steps):
                        poly = [
                            (
                                x + (s * stair_width) * TERRAIN_STEP,
                                y + (s * stair_height) * TERRAIN_STEP,
                            ),
                            (
                                x + ((1 + s) * stair_width) * TERRAIN_STEP,
                                y + (s * stair_height) * TERRAIN_STEP,
                            ),
                            (
                                x + ((1 + s) * stair_width) * TERRAIN_STEP,
                                y + (-1 + s * stair_height) * TERRAIN_STEP,
                            ),
                            (
                                x + (s * stair_width) * TERRAIN_STEP,
                                y + (-1 + s * stair_height) * TERRAIN_STEP,
                            ),
                        ]
                        self.fd_polygon.shape.vertices = poly
                        t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                        t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                        self.terrain.append(t)
                    counter = stair_steps * stair_width

                elif state == STAIRS and not oneshot:
                    s = stair_steps * stair_width - counter - stair_height
                    n = s / stair_width
                    y = original_y + (n * stair_height) * TERRAIN_STEP

                oneshot = False
                self.terrain_y.append(y)
                counter -= 1
                if counter == 0:
                    counter = self.np_random.integers(TERRAIN_GRASS / 2, TERRAIN_GRASS)
                    if state == GRASS and hardcore:
                        state = self.np_random.integers(1, _STATES_)
                        oneshot = True
                    else:
                        state = GRASS
                        oneshot = True

            self.terrain_poly = []
            for i in range(TERRAIN_LENGTH - 1):
                poly = [
                    (self.terrain_x[i], self.terrain_y[i]),
                    (self.terrain_x[i + 1], self.terrain_y[i + 1]),
                ]
                self.fd_edge.shape.vertices = poly
                t = self.world.CreateStaticBody(fixtures=self.fd_edge)
                color = (76, 255 if i % 2 == 0 else 204, 76)
                t.color1 = color
                t.color2 = color
                self.terrain.append(t)
                color = (102, 153, 76)
                poly += [(poly[1][0], 0), (poly[0][0], 0)]
                self.terrain_poly.append((poly, color))
            self.terrain.reverse()


print("REGISTERING BipedalWalkerEnvCustom-v0")
register(
    # The `id` parameter corresponds to the name of the environment, with the syntax as follows:
    # `(namespace)/(env_name)-v(version)` where `namespace` is optional.
    id='BipedalWalkerEnvCustom-v0',  #  id: The environment id
    entry_point='bipedal_walker:BipedalWalkerEnv' # entry_point: The entry point for creating the environment
)

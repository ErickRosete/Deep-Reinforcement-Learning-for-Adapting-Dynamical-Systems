import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
import numpy as np
from env.sawyer_peg_env import custom_sawyer_peg_env

class PegPD():
    def __init__(self, env):
        self.env = env
        self.set_default_values()
    
    def set_default_values(self):
        self.error = 0.002
        self.k_p = 15
        self.k_d = 0.05
        self.dt = 1./240
        self.state = 0

    def reset(self):
        self.set_default_values()
        self.init_target()

    def init_target(self):
        target_position = self.env.get_target_position()
        self.target = [target_position[0] - 0.125, target_position[1], target_position[2] + 0.09]    

    def get_relative_observation(self):
        observation = self.env.get_end_effector_position()
        dx = self.target[0] - observation[0]
        dy = self.target[1] - observation[1]
        dz = self.target[2] - observation[2]
        return (dx, dy, dz)

    def change_target(self, dx, dy, dz):
        if ( abs(dx) < self.error and abs(dy) < self.error and  abs(dz) < self.error ):
            self.state += 1
            target_position = self.env.get_target_position()
            if self.state == 1:
                self.target = [target_position[0], target_position[1], target_position[2] + 0.070]
            elif self.state == 2:
                self.target = [target_position[0], target_position[1], target_position[2] + 0.019]
    
    def clamp_action(self, action):
        # Assure every action component is scaled between -1, 1
        max_action = np.max(np.abs(action))
        if max_action > 1:
            action /= max_action 
        return action

    def get_action(self):   
        dx, dy, dz = self.get_relative_observation()
        
        pd_x = self.k_p*dx + self.k_d*dx/self.dt
        pd_y = self.k_p*dy + self.k_d*dy/self.dt
        pd_z = self.k_p*dz + self.k_d*dz/self.dt
 
        self.change_target(dx, dy, dz)

        action = np.array([pd_x,pd_y,pd_z])
        return self.clamp_action(action) 
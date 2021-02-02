import sys
import hydra
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
import gym
from gym import spaces, ObservationWrapper, ActionWrapper, RewardWrapper
from pathlib import Path
import pybullet as p
import numpy as np
import logging
import tacto
import pybulletX as px
from env.sawyer_gripper import SawyerGripper
from utils.utils import add_cwd

class SawyerPegEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, cfg):
        """ 
        Input: cfg contains the custom configuration of the environment
        cfg.tacto 
        cfg.settings
            show_gui=False, 
            dt=0.005, 
            action_frequency=30, 
            simulation_frequency=240, 
            max_episode_steps=500,
        """
        # Init logger
        self.logger = logging.getLogger(__name__)

        # Init logic parameters
        self.cfg = cfg
        self.show_gui = cfg.settings.show_gui
        self.dt = cfg.settings.dt
        self.action_frequency = cfg.settings.action_frequency
        self.simulation_frequency = cfg.settings.simulation_frequency
        self.max_episode_steps = cfg.settings.max_episode_steps
        self.elapsed_steps = 0

        # Set interaction parameters
        self.action_space = spaces.Box(np.array([-1]*4), np.array([1]*4))
        self.observation_space = spaces.Box(np.array([-1]*6), np.array([1]*6)) # x, y, z, gripper_width, force_f1, force_f2 
        
        # Init environment
        self.logger.info("Initializing world")
        mode = p.GUI if self.show_gui else p.DIRECT
        px.init(mode=mode) 
        p.resetDebugVisualizerCamera(**cfg.pybullet_camera)
        p.setTimeStep(1/self.simulation_frequency)
        
    def reset(self):
        p.resetSimulation() # Remove all elements in simulation
        if self.show_gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0) # Disable rendering during setup
        # Close pyrenderer of digits if it exists
        if hasattr(self, 'digits'):
            self.digits.renderer.r.delete()

        self.load_objects()
        state = self.get_current_state()
        self.reset_logic_parameters()
        if self.show_gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1) 
        return state

    def step(self, action):
        """action: Velocities in xyz and fingers [vx, vy, vz, vf]"""
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        action = self.clamp_action(action)

        state = self.robot.get_states()
        # Relative step in xyz
        state.end_effector.position[0] += action[0] * self.dt
        state.end_effector.position[1] +=  action[1] * self.dt
        state.end_effector.position[2] +=  action[2] * self.dt
        state.gripper_width += action[3] * self.dt
        
        self.robot.set_actions(state)

        # TODO: Check if still required
        # Perform more steps in simulation than querying the model 
        # (Gives time to reach the joint position)
        for _ in range(self.simulation_frequency//self.action_frequency):
            p.stepSimulation()

        observation = self.get_current_state()
        done, success  = self.get_termination()
        reward = self.get_shaped_reward(action, success)
        info = {"target_position": self.get_target_position(), "success": success} 
        self.elapsed_steps += 1
        return observation, reward, done, info

    def load_board_and_peg(self):
        board_orientation = p.getQuaternionFromEuler((0, 0, -np.pi/2))
        board_cfg = {"urdf_path": add_cwd("env/" + self.cfg.objects.board.urdf_path), 
                    "base_position": [np.random.uniform(0.60, 0.70), np.random.uniform(-0.2, 0.2), 0.0],
                    "base_orientation":board_orientation,
                    "use_fixed_base": True}
        self.board = px.Body(**board_cfg)

        peg_position = self.get_end_effector_position()
        peg_position[2] -= 0.02
        self.target = np.random.randint(low=0, high=3)
        peg_urdf_path = ""
        if self.target == 0:
            peg_urdf_path = add_cwd("env/" + self.cfg.objects.cylinder.urdf_path)
        elif self.target == 1:
            peg_urdf_path = add_cwd("env/" + self.cfg.objects.hexagonal_prism.urdf_path)
        elif self.target == 2:
            peg_urdf_path = add_cwd("env/" + self.cfg.objects.square_prism.urdf_path)
        peg_cfg = {"urdf_path": peg_urdf_path, "base_position": peg_position,
                    "base_orientation": board_orientation}
        self.peg = px.Body(**peg_cfg)
        self.digits.add_body(self.peg)

    def load_objects(self):
        # Initialize digit and robot
        self.digits = tacto.Sensor(**self.cfg.tacto)
        if "env" not in self.cfg.sawyer_gripper.robot_params.urdf_path:
            robot_urdf_path = "env/" + self.cfg.sawyer_gripper.robot_params.urdf_path
            self.cfg.sawyer_gripper.robot_params.urdf_path = add_cwd(robot_urdf_path)
        self.robot = SawyerGripper(**self.cfg.sawyer_gripper)
        self.digits.add_camera(self.robot.id, self.robot.digit_links)

        # Load objects
        self.table = px.Body(**self.cfg.objects.table)
        self.load_board_and_peg()

    def clamp_action(self, action):
        # Assure every action component is scaled between -1, 1
        max_action = np.max(np.abs(action))
        if max_action > 1:
            action /= max_action 
        return action

    def get_shaped_reward(self, action, success):
        peg_position = self.get_peg_position()
        target_postion = self.get_target_position()
        
        dist_to_target = np.linalg.norm(target_postion - peg_position)
        dist_to_target = dist_to_target / self.initial_dist
        dist_to_target = 1 if dist_to_target > 1 else dist_to_target
        dist_reward = 1 - dist_to_target ** 0.4 # Positive reward [0, 1]

        #Penalize very high velocities (Smooth transitions)
        action_reward = -0.05 * np.linalg.norm(action[:3])/np.sqrt(3) # [-0.05, 0]
        
        left_steps = self.max_episode_steps - self.elapsed_steps
        total_reward = left_steps * success + dist_reward + action_reward  
        return total_reward
    
    def get_termination(self):
        done, success = False, False
        peg_position = self.get_peg_position()
        target_pose = self.get_target_position()
        end_effector_position = self.get_end_effector_position()
        if (target_pose[0] - 0.020 < peg_position[0] < target_pose[0] + 0.020 and # coord 'x' and 'y' of object
            target_pose[1] - 0.020 < peg_position[1] < target_pose[1] + 0.020 and 
            peg_position[2] <= 0.201): # Coord 'z' of object
            # Inside box
            done, success = True, True
        elif np.linalg.norm(end_effector_position - peg_position) > 0.07:
            # Peg dropped outside box
            done = True
        
        return done, success

    def get_forces(self):
        forces = []
        for cam in self.digits.cameras:
            force_dict = self.digits.get_force(cam)
            av_force = 0
            if len(force_dict) > 0:
                for k, v in force_dict.items():
                    av_force += v
                av_force /= len(force_dict)
            scaled_force = av_force/100
            forces.append(scaled_force)
        return np.array(forces)

    def get_depth(self):
        color, depth = self.digits.render()
        return depth

    def reset_logic_parameters(self):
        self.elapsed_steps = 0
        self.target_noise = np.random.normal(0, 0.01)
        target_position = self.get_target_position()
        peg_position = self.get_peg_position()
        self.initial_dist = np.linalg.norm(target_position - peg_position)

    def get_current_state(self):
        robot_position = self.get_end_effector_position()
        gripper_width = self.get_gripper_width()
        forces = self.get_forces()
        observation = np.concatenate([robot_position, gripper_width, forces])
        return observation

    def render(self, mode='human'):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7, 0, 0.05],
                                                        distance=.7,
                                                        yaw=90,
                                                        pitch=-70,
                                                        roll=0,
                                                        upAxisIndex=2)

        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                aspect=float(960) /720,
                                                nearVal=0.1,
                                                farVal=100.0)

        (w, h, img, depth, segm) = p.getCameraImage(width=960,
                                            height=720,
                                            viewMatrix=view_matrix,
                                            projectionMatrix=proj_matrix,
                                            renderer=p.ER_BULLET_HARDWARE_OPENGL)
        img = np.asarray(img).reshape(720, 960, 4) # H, W, C
        rgb_array = img[:, :, :3] # Ignore alpha channel
        return rgb_array

    def get_gripper_width(self):
        return np.array([self.robot.get_states().gripper_width])

    def get_peg_position(self):
        return np.array(self.peg.get_base_pose()[0])

    def get_target_position(self):
        return np.array(self.board.get_link_state(self.target)["link_world_position"])

    def get_end_effector_position(self):
        end_effector_position = self.robot.get_states().end_effector.position
        end_effector_position[2] -= 0.125
        return end_effector_position

    def close(self):
        p.disconnect()


# Custom wrappers
class TransformObservation(ObservationWrapper):
    def __init__(self, env=None, with_force = True, with_joint=False, relative = True, with_noise=False):
        super(TransformObservation, self).__init__(env)

        self.with_joint = with_joint
        self.with_noise = with_noise
        self.with_force = with_force
        self.relative = relative
        if self.with_force:
            if self.with_joint:
                self.observation_space = spaces.Box(np.array([-1]*6), np.array([1]*6)) # x, y, z, gw, f1, f2 
            else:
                self.observation_space = spaces.Box(np.array([-1]*5), np.array([1]*5)) # x, y, z, f1, f2
        else:
             if self.with_joint:
                self.observation_space = spaces.Box(np.array([-1]*4), np.array([1]*4)) # x, y, z, gw
             else:
                self.observation_space = spaces.Box(np.array([-1]*3), np.array([1]*3)) # x, y, z 

    def observation(self, obs):
        if self.relative:
            state_target = self.env.get_target_position()
            state_target[2] += 0.0175 # Target is a little bit on top
            if self.with_noise:
                state_target +=  self.env.target_noise
            obs[:3] = obs[:3] - state_target  # Relative pose to target
            obs[3:] -= np.array([0.075, 0.865, 0.865]) # substract final gw, f1, f2
        
        if not self.with_joint:
            obs = obs[[0,1,2,4,5]]
        if not self.with_force:
            obs = obs[:-2]

        return obs

class TransformAction(ActionWrapper):
    def __init__(self, env=None, with_joint=False):
        super(TransformAction, self).__init__(env)
        self.with_joint = with_joint

        if with_joint:
            self.action_space = spaces.Box(np.array([-1]*4), np.array([1]*4)) # vel_x, vel_y, vel_z, vel_joint
        else:
            self.action_space = spaces.Box(np.array([-1]*3), np.array([1]*3)) # vel_x, vel_y, vel_z


    def action(self, action):
        if self.with_joint:
            return action
        else:
            action = np.append(action, -0.002/self.env.dt) 
            return action


# Create environment with custom wrapper
def custom_sawyer_peg_env(cfg):
    env = SawyerPegEnv(cfg)
    env = TransformObservation(env, **cfg.observation)
    env = TransformAction(env, **cfg.action)
    return env


@hydra.main(config_path="config", config_name="env_test")
def env_test(cfg):
    env = custom_sawyer_peg_env(cfg)
    for episode in range(100):
        s = env.reset()
        episode_length, episode_reward = 0,0
        for step in range(500):
            a = env.action_space.sample()
            s, r, done, _ = env.step(a)
            if step % 100 == 0:
                print("Action", a)
                print("State", s)
            if done:
                break

if __name__ == "__main__":
    env_test()

import sys
from gym.spaces.box import Box
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
from pybulletX.utils.space_dict import SpaceDict
from env.sawyer_gripper import SawyerGripper
from utils.path import add_cwd

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
        self.difficulty = cfg.settings.difficulty
        self.show_gui = cfg.settings.show_gui
        self.dt = cfg.settings.dt
        self.action_frequency = cfg.settings.action_frequency
        self.simulation_frequency = cfg.settings.simulation_frequency
        self.max_episode_steps = cfg.settings.max_episode_steps
        self.elapsed_steps = 0

        # Set interaction parameters
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()
        
        # Init environment
        self.logger.info("Initializing world")
        mode = p.GUI if self.show_gui else p.DIRECT
        client_id = px.init(mode=mode) 
        self.physics_client = px.Client(client_id=client_id)
        p.resetDebugVisualizerCamera(**cfg.pybullet_camera)
        p.setTimeStep(1/self.simulation_frequency)
    
        p.resetSimulation()
        if self.show_gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0) # Disable rendering during setup
        self.load_objects()
        if self.show_gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        p.setGravity(0,0,-9.8)

    @staticmethod
    def get_observation_space():
        """Return only position and gripper_width by default"""
        observation_space = {}
        observation_space["position"] = spaces.Box(
                low=np.array([0.3, -0.85, 0]), high=np.array([0.85, 0.85, 0.8]))
        observation_space["gripper_width"] = spaces.Box(low=0.03, high=0.11, shape=(1,))
        return SpaceDict(observation_space)

    @staticmethod
    def get_action_space():
        """End effector position and gripper width relative displacement"""      
        return spaces.Box(np.array([-1]*4), np.array([1]*4))

    @staticmethod
    def clamp_action(action):
        """Assure every action component is scaled between -1, 1"""
        max_action = np.max(np.abs(action))
        if max_action > 1:
            action /= max_action
        return action

    def reset_logic_parameters(self):
        self.elapsed_steps = 0
        target_position = self.get_target_position()
        peg_position = self.get_peg_position()
        self.initial_dist = np.linalg.norm(target_position - peg_position)
        self.target_noise = np.random.normal(0, 0.02, target_position.shape) 

    def reset_initial_positions(self):
        if self.show_gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        self.robot.reset()
        self.peg.reset()
        board_position = [np.random.uniform(0.50, 0.55), np.random.uniform(-0.1, 0.1), 0]
        self.board.set_base_pose(board_position, self.board.init_base_orientation)

        if self.show_gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    def reset(self):
        self.reset_initial_positions()
        self.reset_logic_parameters()
        state = self.get_current_state()
        return state

    def step(self, action):
        """action: Velocities in xyz and gripper_width [vx, vy, vz, vf]"""
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        action = self.clamp_action(action)

        state = self.robot.get_states()
        # Relative step in xyz
        state.end_effector.orientation =  self.cfg.sawyer_gripper.init_state.end_effector.orientation
        state.end_effector.position[0] += action[0] * self.dt
        state.end_effector.position[1] +=  action[1] * self.dt
        state.end_effector.position[2] +=  action[2] * self.dt
        state.gripper_width += action[3] * self.dt
        state["gripper_force"] = 35
        self.robot.set_actions(state)

        # TODO: Check if still required
        # Perform more steps in simulation than querying the model 
        # (Gives time to reach the joint position)
        for _ in range(self.simulation_frequency//self.action_frequency):
            p.stepSimulation()
        
        if self.cfg.tacto.visualize_gui:
            self.update_tacto_gui()

        observation = self.get_current_state()
        done, success  = self.get_termination()
        if self.cfg.settings.reward == "dense":
            reward = self.get_shaped_reward(action, success)
        elif self.cfg.settings.reward == "sparse":
            reward = 100 * int(success)
        else:
            raise Exception('Not implemented error')
        
        info = {"target_position": self.get_target_position(), "success": success} 
        self.elapsed_steps += 1
        return observation, reward, done, info

    def load_board_and_peg(self):
        board_orientation = p.getQuaternionFromEuler((np.pi/2, 0, 0))
        if self.difficulty == "easy":
            board_path = add_cwd("env/data/peg_in_hole_board/board_easy.urdf")   
        elif self.difficulty == "hard":
            board_path = add_cwd("env/data/peg_in_hole_board/board_hard.urdf")   
        else:
            raise Exception("Please enter a valid difficulty")

        board_cfg = {"urdf_path": board_path, 
                    "base_position": [np.random.uniform(0.50, 0.55), np.random.uniform(-0.1, 0.1), 0],
                    "base_orientation": board_orientation,
                    "use_fixed_base": True}
        self.board = px.Body(**board_cfg, physics_client=self.physics_client)

        peg_position = self.get_end_effector_position()
        peg_position[2] -= 0.025
        peg_urdf_path = add_cwd("env/" + self.cfg.objects.cylinder.urdf_path)
        peg_cfg = {"urdf_path": peg_urdf_path, "base_position": peg_position,
                    "base_orientation": p.getQuaternionFromEuler((0, 0, 0))}
        self.peg = px.Body(**peg_cfg, physics_client=self.physics_client)
        self.digits.add_body(self.peg)

    def load_objects(self):
        # Initialize digit and robot
        self.digits = tacto.Sensor(**self.cfg.tacto)
        if "env" not in self.cfg.sawyer_gripper.robot_params.urdf_path:
            robot_urdf_path = "env/" + self.cfg.sawyer_gripper.robot_params.urdf_path
            self.cfg.sawyer_gripper.robot_params.urdf_path = add_cwd(robot_urdf_path)
        self.robot = SawyerGripper(**self.cfg.sawyer_gripper, physics_client=self.physics_client)
        self.digits.add_camera(self.robot.id, self.robot.digit_links)

        # Load objects
        self.table = px.Body(**self.cfg.objects.table, physics_client=self.physics_client)
        self.load_board_and_peg()

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
        total_reward = (100 + left_steps) * success + dist_reward + action_reward  
        return total_reward
    
    def get_termination(self):
        done, success = False, False
        peg_position = self.get_peg_position()
        target_pose = self.get_target_position()
        end_effector_position = self.get_end_effector_position()
        if (target_pose[0] - 0.020 < peg_position[0] < target_pose[0] + 0.020 and # coord 'x' and 'y' of object
            target_pose[1] - 0.020 < peg_position[1] < target_pose[1] + 0.020 and 
            peg_position[2] <= 0.195): # Coord 'z' of object
            # Inside box
            done, success = True, True
        elif np.linalg.norm(end_effector_position - peg_position) > 0.125:
            # Peg dropped outside box
            done = True
        return done, success

    @staticmethod
    def render(mode='human'):
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

        (w, h, img, depth, segm) = p.getCameraImage(width=64, height=64,
                                            viewMatrix=view_matrix,
                                            projectionMatrix=proj_matrix,
                                            renderer=p.ER_BULLET_HARDWARE_OPENGL)
        img = np.asarray(img).reshape(64, 64, 4) # H, W, C
        rgb_array = img[:, :, :3] # Ignore alpha channel
        return rgb_array
    
    def update_tacto_gui(self):
        color, depth = self.digits.render()
        self.digits.updateGUI(color, depth)

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

    def get_digit_depth(self):
        color, depth = self.digits.render()
        return depth

    def get_current_state(self):
        observation = {}
        observation["position"] = self.get_end_effector_position()
        observation["gripper_width"] = self.get_gripper_width()
        return observation

    def get_gripper_width(self):
        return np.array([self.robot.get_states().gripper_width])

    def get_peg_position(self):
        return np.array(self.peg.get_base_pose()[0])

    def get_target_position(self):
        return np.array(self.board.get_link_state(0)["link_world_position"])

    def get_end_effector_position(self):
        end_effector_position = self.robot.get_states().end_effector.position
        end_effector_position[2] -= 0.125
        return end_effector_position

    def close(self):
        p.disconnect(self.physics_client.id)


# Custom wrappers
class TransformObservation(ObservationWrapper):
    def __init__(self, env=None, with_force = False, with_tactile_sensor = False,
                 with_gripper_width=False, relative = True, with_noise=False):
        super(TransformObservation, self).__init__(env)
        self.with_tactile_sensor = with_tactile_sensor
        self.with_gripper_width = with_gripper_width
        self.with_noise = with_noise
        self.with_force = with_force
        self.relative = relative
        self.observation_space = self.get_observation_space()

    def get_observation_space(self):
        observation_space = {}
        observation_space["position"] = spaces.Box(
                low=np.array([0.3, -0.85, 0]), high=np.array([0.85, 0.85, 0.8]))
        if self.with_gripper_width:
            observation_space["gripper_width"] = spaces.Box(low=0.03, high=0.11, shape=(1,))
        if self.with_tactile_sensor:
            t_width, t_height = self.env.cfg.tacto.width, self.env.cfg.tacto.height
            observation_space["tactile_sensor"] = spaces.Tuple(
                tuple([spaces.Box(low=-np.ones((t_height,t_width)), high=np.ones((t_height, t_width)))
                       for _ in range(2)]))
        if self.with_force:
            observation_space["force"] = spaces.Box(
                low=np.array([0, 0]), high=np.array([2.0, 2.0]))
        return SpaceDict(observation_space)

    def observation(self, obs):
        if self.with_force:
            obs["force"] = self.env.get_forces()
        if not self.with_gripper_width:
            del obs["gripper_width"]
        if self.with_tactile_sensor:
            obs["tactile_sensor"] = self.env.get_digit_depth()

        if self.relative:
            state_target = self.env.get_target_position()
            state_target[2] += 0.020 
            if self.with_noise:
                state_target +=  self.env.target_noise
            obs["position"] -= state_target
            if self.with_gripper_width:
                obs["gripper_width"] -= 0.075
            if self.with_force:
                obs["force"] -= np.array([ 0.865, 0.865])
        return obs

class TransformAction(ActionWrapper):
    def __init__(self, env=None, with_gripper_width=False):
        super(TransformAction, self).__init__(env)
        self.with_gripper_width = with_gripper_width

        if with_gripper_width:
            self.action_space = spaces.Box(np.array([-1]*4), np.array([1]*4)) # vel_x, vel_y, vel_z, vel_joint
        else:
            self.action_space = spaces.Box(np.array([-1]*3), np.array([1]*3)) # vel_x, vel_y, vel_z

    def action(self, action):
        if self.with_gripper_width:
            return action
        else:
            action = np.append(action, -0.002/self.env.dt) 
            return action

# Create environment with custom wrapper
def register_sawyer_env(max_episode_steps=1000):
    gym.envs.register(
        id='sawyer-peg-v1',
        entry_point='env.sawyer_peg_env:SawyerPegEnv',
        max_episode_steps=max_episode_steps,
    )

def custom_sawyer_peg_env(cfg):
    env = gym.make('sawyer-peg-v1', cfg=cfg)
    env._max_episode_steps = cfg.settings.max_episode_steps
    env = TransformObservation(env, **cfg.observation)
    env = TransformAction(env, **cfg.action)
    return env

@hydra.main(config_path="../config", config_name="sac_config")
def env_test(cfg):
    env = custom_sawyer_peg_env(cfg.env)
    for episode in range(100):
        s = env.reset()
        episode_length, episode_reward = 0,0
        for step in range(100):
            a = env.action_space.sample()
            s, r, done, _ = env.step(a)
            if step % 100 == 0:
                print("Action", a)
                if isinstance(s, dict):
                    print("Position", s["position"])
                else:
                    print("State", s)
            if done:
                break

if __name__ == "__main__":
    register_sawyer_env()
    env_test()

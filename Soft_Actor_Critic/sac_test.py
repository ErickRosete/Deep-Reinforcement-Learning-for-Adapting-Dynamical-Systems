import sys
from pathlib import Path
from sac_agent import SAC_Agent
sys.path.insert(0, str(Path(__file__).parents[1]))
from env.sawyer_peg_env import custom_sawyer_peg_env

def main():
    eval_cfg = {"num_episodes": 20, "render": False}
    env_cfg = {"show_gui": True, "max_episode_steps": 500, "with_force": False, 
        "with_joint":False, "relative": True, "with_noise":False, "reward_type":"shaped_2",
        "dt":0.005}
    agent_cfg = {  "batch_size": 256, "gamma": 0.99, "tau": 0.005, "actor_lr": 3e-4,
        "critic_lr": 3e-4, "alpha_lr": 3e-4, "hidden_dim": 256}
    model_name = "sac_peg_v2_pose_2200.pth"

    env = custom_sawyer_peg_env(env_cfg)
    agent = SAC_Agent(env, **agent_cfg)
    agent.load(model_name)
    stats = agent.evaluate(**eval_cfg)
    print(stats)
    
if __name__ == "__main__":
    main()
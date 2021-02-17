import sys
import hydra
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from Soft_Actor_Critic.sac_agent import SAC_Agent
from env.sawyer_peg_env import custom_sawyer_peg_env
from utils.path import add_cwd

@hydra.main(config_path="../config", config_name="sac_config")
def main(cfg):
    env = custom_sawyer_peg_env(cfg.env)
    agent = SAC_Agent(env, **cfg.agent)
    agent.load(add_cwd(cfg.test.model_name))
    stats = agent.evaluate(**cfg.test.run)
    print(stats)
    
if __name__ == "__main__":
    main()
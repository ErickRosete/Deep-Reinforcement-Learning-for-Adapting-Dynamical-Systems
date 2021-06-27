import hydra
from drlfads.Soft_Actor_Critic.sac_agent import SAC_Agent
from drlfads.env.sawyer_peg_env import custom_sawyer_peg_env, register_sawyer_env
from drlfads.utils.path import add_cwd
import logging

@hydra.main(config_path="../config", config_name="sac_config")
def main(cfg):
    env = custom_sawyer_peg_env(cfg.env)
    agent = SAC_Agent(env, **cfg.agent)
    agent.load(add_cwd(cfg.test.model_name))
    stats = agent.evaluate(**cfg.test.run)
    logger = logging.getLogger(__name__)
    logger.info(stats)    
    
if __name__ == "__main__":
    register_sawyer_env()
    main()
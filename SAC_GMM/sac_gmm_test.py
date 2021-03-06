import sys
import hydra
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from GMM.gmm import GMM
from utils.path import add_cwd
from env.sawyer_peg_env import custom_sawyer_peg_env, register_sawyer_env
from SAC_GMM.sac_gmm_agent import SAC_GMM_Agent

@hydra.main(config_path="../config", config_name="sac_gmm_config")
def main(cfg):
    # Do not show tacto renderer output
    logger = logging.getLogger('tacto.renderer')
    logger.propagate = False

    env = custom_sawyer_peg_env(cfg.env)
    gmm_model = GMM(add_cwd(cfg.gmm_model))
    agent = SAC_GMM_Agent(env=env, model=gmm_model, **cfg.agent)
    agent.load(add_cwd(cfg.test.model_name))
    stats = agent.evaluate(**cfg.test.run)
    print(stats)
    agent.env.close()

if __name__ == "__main__":
    register_sawyer_env()
    main()

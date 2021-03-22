import sys
import hydra
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from SAC_GMM_Residual.sac_gmm_residual_agent import SAC_GMM_Residual_Agent
from env.sawyer_peg_env import custom_sawyer_peg_env, register_sawyer_env
from GMM.gmm import GMM
from utils.path import add_cwd
from utils.misc import get_save_filename

@hydra.main(config_path="../config", config_name="sac_gmm_residual_config")
def main(cfg):
    # Do not show tacto renderer output
    logger = logging.getLogger('tacto.renderer')
    logger.propagate = False

    for i in range(cfg.train.num_random_seeds):
        # Training 
        env = custom_sawyer_peg_env(cfg.env)
        gmm_model = GMM(add_cwd(cfg.gmm_model))
        agent = SAC_GMM_Residual_Agent(env=env, model=gmm_model, **cfg.agent)
        save_filename = get_save_filename("sac_gmm_res", cfg, i)
        agent.train(**cfg.train.run, save_filename=save_filename)
        agent.env.close()

        # Testing
        agent.env = custom_sawyer_peg_env(cfg.env)
        agent.evaluate(**cfg.test.run)
        agent.env.close()


if __name__ == "__main__":
    register_sawyer_env()
    main()

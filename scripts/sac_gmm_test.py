import hydra
import logging
from drlfads.GMM.gmm               import GMM
from drlfads.utils.path            import add_cwd
from drlfads.env.sawyer_peg_env    import custom_sawyer_peg_env, register_sawyer_env
from drlfads.SAC_GMM.sac_gmm_agent import SAC_GMM_Agent

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
    logger = logging.getLogger(__name__)
    logger.info(stats)
    agent.env.close()

if __name__ == "__main__":
    register_sawyer_env()
    main()
